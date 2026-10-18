"""Node 3: questions per scenario (one LLM call each), for any product type.

The mode is copied from each scenario. Grounded scenarios are linked to their Reference Pack
(reference_pack_id) and get the leak check (identifier-like core_ref names; rule from
generation.question.reject_reference_leaks); ungrounded ones get neither. Calls run
concurrently under ``llm.max_concurrency``; results are assembled in scenario order so ids
are stable. Question text is never edited: problems lead to retry or rejection.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field, ValidationError

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import (
    ArtifactIntegrityError,
    LLMOutputValidationError,
    format_validation_error,
)
from synthetic_user.core.identifiers import IdKind, SequentialIdGenerator
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.base import AmbiguityLevel, StrictModel
from synthetic_user.models.persona import ExpertiseLevel, Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import (
    QuestionCase,
    QuestionDraft,
    QuestionGenerationResult,
    QuestionStyle,
    RejectedQuestion,
)
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario
from synthetic_user.prompts.question import QuestionPromptOptions, build_question_prompt
from synthetic_user.references.leaks import find_leaks, pack_identifiers

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

NEAR_DUPLICATE_JACCARD = 0.8
VAGUE_NOT_ALLOWED = "vague_not_allowed"
SURPLUS = "surplus: more questions returned than requested"


class _RawQuestionBatch(StrictModel):
    """Each question is validated individually so one bad item is rejected on its own."""

    questions: list[dict[str, Any]] = Field(default_factory=list)


@dataclass
class _ScenarioOutcome:
    drafts: list[tuple[QuestionDraft, dict[str, Any]]] = field(default_factory=list)
    rejected: list[RejectedQuestion] = field(default_factory=list)


class QuestionGenerator:
    def __init__(
        self,
        llm: LLMClient,
        settings: AppConfig,
        logger: Logger,
        *,
        id_generator: SequentialIdGenerator,
    ) -> None:
        self._llm = llm
        self._settings = settings
        self._log = logger
        self._ids = id_generator

    async def generate(
        self,
        persona: Persona,
        product: ProductDescription,
        scenarios: Sequence[Scenario],
        reference: Reference | None,
        packs: Sequence[ReferencePack] = (),
        *,
        count: int | None = None,
    ) -> QuestionGenerationResult:
        """``reference`` and ``packs`` are needed only for grounded scenarios."""
        per_scenario = (
            count if count is not None else self._settings.generation.question.question_count
        )
        if per_scenario < 1:
            raise ValueError("question count must be >= 1")
        groundings = [_grounding(s, reference, packs) for s in scenarios]
        semaphore = asyncio.Semaphore(self._settings.llm.max_concurrency)
        outcomes = await asyncio.gather(
            *(
                self._for_scenario(
                    persona,
                    product,
                    s,
                    grounding,
                    per_scenario,
                    pack_identifiers(grounding[0]) if grounding is not None else [],
                    semaphore,
                )
                for s, grounding in zip(scenarios, groundings, strict=True)
            )
        )

        result = QuestionGenerationResult()
        seen = _NearDuplicates()
        for utterance in persona.example_utterances:
            seen.add(utterance, "persona example utterance")
        for scenario, grounding, outcome in zip(scenarios, groundings, outcomes, strict=True):
            identifiers = pack_identifiers(grounding[0]) if grounding is not None else []
            result.rejected.extend(outcome.rejected)
            for draft, raw in outcome.drafts:
                duplicate_of = seen.find(draft.utterance)
                if duplicate_of is not None:
                    result.rejected.append(
                        RejectedQuestion(
                            scenario_id=scenario.id,
                            payload=raw,
                            reasons=[f"near-duplicate of {duplicate_of}"],
                        )
                    )
                    continue
                question = QuestionCase(
                    question_id=self._ids.next(IdKind.QUESTION),
                    scenario_id=scenario.id,
                    persona_id=persona.id,
                    mode=scenario.mode,
                    reference_pack_id=scenario.reference_pack_id,
                    utterance=draft.utterance,
                    intended_goal=scenario.business_goal,
                    intent=draft.intent,
                    question_characteristics=draft.question_characteristics,
                    expectations=draft.expectations,
                    leak_warnings=[
                        f"mentions reference name {ident!r}"
                        for ident in find_leaks(draft.utterance, identifiers)
                    ],
                )
                seen.add(draft.utterance, question.question_id)
                result.questions.append(question)
        self._log.info(
            "Questions: %d accepted, %d rejected across %d scenario(s) (%d grounded)",
            len(result.questions),
            len(result.rejected),
            len(scenarios),
            sum(1 for g in groundings if g is not None),
        )
        return result

    async def _for_scenario(
        self,
        persona: Persona,
        product: ProductDescription,
        scenario: Scenario,
        grounding: tuple[ReferencePack, Reference] | None,
        count: int,
        identifiers: Sequence[str],
        semaphore: asyncio.Semaphore,
    ) -> _ScenarioOutcome:
        cfg = self._settings.generation.question
        technical = persona.expertise.sql is ExpertiseLevel.HIGH
        # Leak check only in grounded mode (there are no identifiers otherwise).
        enforce_leaks = bool(identifiers) and cfg.reject_reference_leaks and not technical
        options = QuestionPromptOptions(
            count=count,
            ambiguity_level=cfg.ambiguity_level,
            language=cfg.language,
            allow_vague_questions=cfg.allow_vague_questions,
            technical_persona=technical,
        )
        system, user = build_question_prompt(persona, product, scenario, grounding, options)
        last: list[_RawQuestionBatch] = []

        def post_validate(batch: _RawQuestionBatch) -> list[str]:
            last[:] = [batch]
            if not enforce_leaks:
                return []
            return [
                f"question {i}: uses internal reference names {leaks}; rephrase in the "
                "persona's own words"
                for i, raw in enumerate(batch.questions, start=1)
                if (leaks := find_leaks(str(raw.get("utterance", "")), identifiers))
            ]

        async with semaphore:
            try:
                batch = await generate_structured(
                    self._llm,
                    system,
                    user,
                    _RawQuestionBatch,
                    max_retries=self._settings.llm.max_validation_retries,
                    post_validate=post_validate,
                )
            except LLMOutputValidationError as exc:
                if not last:
                    return _ScenarioOutcome(
                        rejected=[
                            RejectedQuestion(
                                scenario_id=scenario.id,
                                payload={},
                                reasons=list(exc.errors) or [str(exc)],
                            )
                        ]
                    )
                batch = last[0]  # leaks persisted after retries; reject those questions below

        outcome = _ScenarioOutcome()
        for raw in batch.questions:
            if len(outcome.drafts) >= count:
                outcome.rejected.append(_reject(scenario, raw, [SURPLUS]))
                continue
            checked = self._check(raw, identifiers, enforce_leaks)
            if isinstance(checked, QuestionDraft):
                outcome.drafts.append((checked, raw))
            else:
                outcome.rejected.append(_reject(scenario, raw, checked))
        if len(outcome.drafts) < count:
            self._log.warning(
                "%s: %d of %d questions usable", scenario.id, len(outcome.drafts), count
            )
        return outcome

    def _check(
        self, raw: dict[str, Any], identifiers: Sequence[str], enforce_leaks: bool
    ) -> QuestionDraft | list[str]:
        try:
            draft = QuestionDraft.model_validate(raw)
        except ValidationError as exc:
            return [format_validation_error(exc)]
        reasons: list[str] = []
        if enforce_leaks and (leaks := find_leaks(draft.utterance, identifiers)):
            reasons.append(f"reference leak: {leaks}")
        traits = draft.question_characteristics
        if traits.style is QuestionStyle.VAGUE:
            if not self._settings.generation.question.allow_vague_questions:
                reasons.append(VAGUE_NOT_ALLOWED)
            if traits.ambiguity_level is AmbiguityLevel.LOW:
                reasons.append("vague question must have ambiguity_level medium or high")
        return reasons or draft


def _reject(scenario: Scenario, raw: dict[str, Any], reasons: list[str]) -> RejectedQuestion:
    return RejectedQuestion(scenario_id=scenario.id, payload=raw, reasons=reasons)


class _NearDuplicates:
    """Utterances seen so far; near-duplicate = same normalised text or token Jaccard >= 0.8."""

    def __init__(self) -> None:
        self._seen: list[tuple[str, frozenset[str], str]] = []

    def add(self, utterance: str, owner: str) -> None:
        normalised = _normalise(utterance)
        self._seen.append((normalised, frozenset(normalised.split()), owner))

    def find(self, utterance: str) -> str | None:
        normalised = _normalise(utterance)
        tokens = frozenset(normalised.split())
        for other_text, other_tokens, owner in self._seen:
            if normalised == other_text:
                return owner
            union = tokens | other_tokens
            if union and len(tokens & other_tokens) / len(union) >= NEAR_DUPLICATE_JACCARD:
                return owner
        return None


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _grounding(
    scenario: Scenario, reference: Reference | None, packs: Sequence[ReferencePack]
) -> tuple[ReferencePack, Reference] | None:
    """Grounded scenario -> (its pack, the reference); ungrounded -> None."""
    if scenario.mode == GenerationMode.UNGROUNDED:
        return None
    pack = next((p for p in packs if p.pack_id == scenario.reference_pack_id), None)
    if reference is None or pack is None:
        raise ArtifactIntegrityError(
            f"{scenario.id} is grounded but its reference / reference pack "
            f"{scenario.reference_pack_id} is not available"
        )
    if reference.reference_id != scenario.reference_id:
        raise ArtifactIntegrityError(
            f"{scenario.id} was built from reference {scenario.reference_id}, "
            f"not {reference.reference_id}"
        )
    return pack, reference
