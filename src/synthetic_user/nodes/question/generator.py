"""Node 3: ONE independent question-generation call per accepted scenario.

Calls run concurrently under ``generation.question.max_parallel_scenarios`` (default 5, an
internal setting) and results are assembled in stable scenario order. A failed scenario call is
isolated (recorded as a rejection for that scenario); valid items are accepted individually and
nothing is silently regenerated. Normal-path call budget = one per accepted scenario (+ the
bounded validation retries of that call); no separate semantic-validation call.

Grounded questions link to their scenario's validated Reference Pack. Each question carries
its Evaluation Context (context.py): the success criteria (C1..Cn), the Reference Spec (the
scenario's pack, reused) and the Metric Mapping of every selected metric. When no valid question
fits the scenario, the model returns a ``generation_failure`` reason, recorded as a rejection.

Retry validation and final acceptance run the SAME per-item check (``_check``, D252): a reply
passes only when every item would be usable, so an invalid enum value or a persona style rule
is fed back instead of being found after acceptance; anything discarded later (surplus, a
persisting near-duplicate, a cross-scenario exact duplicate) is logged and traced with its
reason. Validation keeps the deterministic, structural checks only: schema / enums, counts
and limits of the existing hard policy (deliverables, sources, words, criteria, complexity,
follow-ups), canonical product data-source membership, Evaluation Context integrity
(criterion ids, mapping keys and ids, criterion / metric evidence compatibility, expectation
shape, canonical tool / source membership), exact structural duplicates (normalized text
equality) and the reference-name leak check (a security check).
There is no regex / keyword semantic rejection.

Semantic diversity (D239): the prompt asks for different angles of the same goal and lists the
scenario's earlier questions (when the step is re-run) in the SAME call; diversity.py flags
near-duplicates (same structure + near-identical content words) and expectations copied from a
sibling. Those feed the existing bounded retry; what persists is rejected per question.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field, ValidationError

from synthetic_user.core.config import AppConfig, QuestionGenerationSettings
from synthetic_user.core.exceptions import (
    ArtifactIntegrityError,
    LLMOutputValidationError,
    LLMProviderError,
    format_validation_error,
)
from synthetic_user.core.identifiers import IdKind, SequentialIdGenerator
from synthetic_user.core.trace import record_validation, trace_subject
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.base import AmbiguityLevel, StrictModel
from synthetic_user.models.metrics import MetricDefinition
from synthetic_user.models.persona import AmbiguityPreference, Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import (
    DeliverableAction,
    EvaluationContext,
    QuestionCase,
    QuestionCharacteristics,
    QuestionComplexity,
    QuestionDraft,
    QuestionGenerationResult,
    QuestionPlan,
    QuestionStyle,
    RejectedQuestion,
)
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.question.context import (
    ContextRequest,
    build_context,
    canonical_problems,
    context_request,
    mapping_problems,
)
from synthetic_user.nodes.question.diversity import (
    QuestionShape,
    diversity_problem,
    diversity_problems,
    shape_of,
)
from synthetic_user.nodes.question.prompt import QuestionPromptOptions, build_question_prompt
from synthetic_user.references.leaks import find_leaks, pack_identifiers

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

# Persona question_preferences.ambiguity -> (style mix level, vague questions allowed).
PERSONA_AMBIGUITY = {
    AmbiguityPreference.NONE: (AmbiguityLevel.LOW, False),
    AmbiguityPreference.OCCASIONAL: (AmbiguityLevel.MEDIUM, True),
    AmbiguityPreference.FREQUENT: (AmbiguityLevel.HIGH, True),
}
VAGUE_NOT_ALLOWED = "style 'vague' is not allowed for this persona: use precise or investigative"
VAGUE_LOW = "a vague question must have ambiguity_level medium or high"
SURPLUS = "surplus: more questions returned than requested"
_WORD = re.compile(r"\b[\w'-]+\b")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


class _RawQuestionBatch(StrictModel):
    """Each question is validated individually so one bad item is rejected on its own."""

    questions: list[dict[str, Any]] = Field(default_factory=list)
    # The model's reason when no valid question fits the scenario (never fabricated).
    generation_failure: str | None = None


@dataclass(frozen=True)
class PersonaQuestionRules:
    """What the persona allows, from question_preferences (config fills what it leaves out)."""

    ambiguity_level: AmbiguityLevel
    allow_vague: bool
    technical: bool


def persona_question_rules(persona: Persona, settings: AppConfig) -> PersonaQuestionRules:
    cfg = settings.generation.question
    prefs = persona.question_preferences
    level, vague = (
        PERSONA_AMBIGUITY[prefs.ambiguity]
        if prefs.ambiguity is not None
        else (cfg.ambiguity_level, cfg.allow_vague_questions)
    )
    return PersonaQuestionRules(level, vague, persona.is_technical)


@dataclass
class _ScenarioOutcome:
    drafts: list[tuple[QuestionDraft, EvaluationContext, dict[str, Any]]] = field(
        default_factory=list
    )
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
        metrics: Sequence[MetricDefinition] = (),
        previous: Mapping[str, Sequence[QuestionCase]] | None = None,
    ) -> QuestionGenerationResult:
        """``reference`` and ``packs`` are needed only for grounded scenarios; ``metrics`` are
        the run's selected (frozen) metric definitions; ``previous`` = questions already
        generated per scenario id (diversity context when the step is re-run)."""
        cfg = self._settings.generation.question
        per_scenario = count if count is not None else cfg.question_count
        if per_scenario < 1:
            raise ValueError("question count must be >= 1")
        groundings = [_grounding(s, reference, packs) for s in scenarios]
        known: dict[str, str] = {}
        for utterance in persona.example_utterances:
            known.setdefault(_normalise(utterance), "persona example utterance")
        for utterance in product.typical_user_questions:
            known.setdefault(_normalise(utterance), "product typical user question")
        semaphore = asyncio.Semaphore(cfg.max_parallel_scenarios)
        outcomes = await asyncio.gather(
            *(
                self._for_scenario(
                    persona,
                    product,
                    s,
                    grounding,
                    per_scenario,
                    semaphore,
                    reference,
                    metrics,
                    list((previous or {}).get(s.id, ())),
                    known,
                )
                for s, grounding in zip(scenarios, groundings, strict=True)
            )
        )

        result = QuestionGenerationResult()
        seen = dict(known)
        for scenario, grounding, outcome in zip(scenarios, groundings, outcomes, strict=True):
            identifiers = pack_identifiers(grounding[0]) if grounding is not None else []
            result.rejected.extend(outcome.rejected)
            for draft, context, raw in outcome.drafts:
                key = _normalise(draft.utterance)
                if (duplicate_of := seen.get(key)) is not None:
                    # Only a duplicate of another scenario's question can reach this point
                    # (the calls run concurrently); everything else was checked per item.
                    reasons = [f"exact duplicate of {duplicate_of}"]
                    self._log.warning("%s: question discarded: %s", scenario.id, "; ".join(reasons))
                    result.rejected.append(_reject(scenario, raw, reasons))
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
                    question_characteristics=QuestionCharacteristics(
                        style=draft.question_characteristics.style,
                        ambiguity_level=draft.question_characteristics.ambiguity_level,
                        complexity=complexity_for_plan(draft.plan),
                    ),
                    evaluation_context=context,
                    plan=draft.plan,
                    leak_warnings=[
                        f"mentions reference name {ident!r}"
                        for ident in find_leaks(draft.utterance, identifiers)
                    ],
                )
                seen[key] = question.question_id
                result.questions.append(question)
        self._log.info(
            "Questions: %d accepted, %d rejected across %d scenario(s) (%d grounded); "
            "%d generation call(s)",
            len(result.questions),
            len(result.rejected),
            len(scenarios),
            sum(1 for g in groundings if g is not None),
            len(scenarios),
        )
        return result

    async def _for_scenario(
        self,
        persona: Persona,
        product: ProductDescription,
        scenario: Scenario,
        grounding: tuple[ReferencePack, Reference] | None,
        count: int,
        semaphore: asyncio.Semaphore,
        reference: Reference | None,
        metrics: Sequence[MetricDefinition],
        previous: Sequence[QuestionCase] = (),
        known: Mapping[str, str] | None = None,
    ) -> _ScenarioOutcome:
        cfg = self._settings.generation.question
        earlier = [
            shape
            for q in previous
            if (shape := shape_of(q, f"earlier question {q.question_id}")) is not None
        ]
        threshold = cfg.max_question_similarity
        rules = persona_question_rules(persona, self._settings)
        identifiers = pack_identifiers(grounding[0]) if grounding is not None else []
        # Leak check only in grounded mode (there are no identifiers otherwise).
        enforce_leaks = bool(identifiers) and cfg.reject_reference_leaks and not rules.technical
        request = context_request(
            metrics, reference, grounding[0] if grounding is not None else None
        )
        options = QuestionPromptOptions(
            count=count,
            max_deliverables=cfg.max_deliverables,
            max_sources=cfg.max_sources,
            max_success_criteria=cfg.max_success_criteria,
            max_words=cfg.max_words,
            max_listed_values=cfg.max_listed_values,
            allowed_complexities=tuple(
                QuestionComplexity(item) for item in cfg.allowed_complexities
            ),
            allow_follow_ups=cfg.allow_follow_ups,
            allow_cross_source_comparison=cfg.allow_cross_source_comparison,
            ambiguity_level=rules.ambiguity_level,
            language=cfg.language,
            allow_vague_questions=rules.allow_vague,
            technical_persona=rules.technical,
            previous_questions=tuple(q.utterance for q in previous),
        )
        system, user = build_question_prompt(
            persona,
            product,
            scenario,
            grounding,
            options,
            request,
            tool_definitions=reference if request.needs_tool_definitions else None,
        )
        last: list[_RawQuestionBatch] = []

        def post_validate(batch: _RawQuestionBatch) -> list[str]:
            # The SAME checks as the final per-item acceptance (_check), schema and enum values
            # included: a reply is accepted only when every item would be usable, so retry
            # feedback names the actual failing condition. After the retries, items are still
            # judged one by one (valid ones kept, the others rejected with their reasons).
            last[:] = [batch]
            errors: list[str] = []
            shapes: list[QuestionShape | None] = []
            for i, raw in enumerate(batch.questions, start=1):
                checked = self._check(
                    raw, product, identifiers, enforce_leaks, rules, request, known
                )
                if not isinstance(checked, QuestionDraft):
                    shapes.append(None)
                    errors.extend(f"question {i}: {problem}" for problem in checked)
                    continue
                shapes.append(shape_of(checked, f"question {i}"))
                errors.extend(
                    f"question {i}: {problem}"
                    for problems in canonical_problems(
                        checked.evaluation_context, request, reference
                    ).values()
                    for problem in problems
                )
            if not batch.questions and not batch.generation_failure:
                errors.append(
                    "questions is empty: return the requested question(s), or "
                    "generation_failure with the specific reason none fits"
                )
            if cfg.diversity_check:
                errors.extend(
                    f"question {problem.index + 1}: {problem.message}"
                    for problem in diversity_problems(shapes, earlier, threshold=threshold)
                )
            return errors

        async with semaphore:
            try:
                with trace_subject(scenario.id):
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
                        rejected=[_reject(scenario, {}, list(exc.errors) or [str(exc)])]
                    )
                batch = last[0]  # problems persisted after retries: judged item by item below
            except LLMProviderError as exc:
                # One scenario's failed call never sinks the others (isolated, not retried).
                self._log.warning("%s: question generation call failed: %s", scenario.id, exc)
                return _ScenarioOutcome(
                    rejected=[_reject(scenario, {}, [f"question generation call failed: {exc}"])]
                )

        outcome = _ScenarioOutcome()
        if not batch.questions and batch.generation_failure:
            # The model could not write a valid question in scope: recorded, never fabricated.
            self._log.warning("%s: generation failure: %s", scenario.id, batch.generation_failure)
            return _ScenarioOutcome(
                rejected=[
                    _reject(scenario, {}, [f"generation failure: {batch.generation_failure}"])
                ]
            )
        if not batch.questions:
            reason = "the model returned no question and no generation_failure reason"
            self._log.warning("%s: %s", scenario.id, reason)
            return _ScenarioOutcome(rejected=[_reject(scenario, {}, [reason])])
        accepted: list[QuestionShape] = []
        for position, raw in enumerate(batch.questions, start=1):
            if len(outcome.drafts) >= count:
                outcome.rejected.append(self._discard(scenario, position, raw, [SURPLUS]))
                continue
            checked = self._check(raw, product, identifiers, enforce_leaks, rules, request, known)
            if isinstance(checked, QuestionDraft):
                draft = checked
                shape = shape_of(draft, f"question {position}")
                problem = (
                    diversity_problem(shape, [*earlier, *accepted], threshold=threshold)
                    if cfg.diversity_check and shape is not None
                    else None
                )
                if problem is not None:  # persisted after the retries: rejected, kept first
                    outcome.rejected.append(self._discard(scenario, position, raw, [problem]))
                    continue
                if shape is not None:
                    accepted.append(shape)
                context = build_context(
                    draft.evaluation_context,
                    request,
                    reference,
                    grounding[0] if grounding is not None else None,
                )
                outcome.drafts.append((draft, context, raw))
            else:
                outcome.rejected.append(self._discard(scenario, position, raw, checked))
        if len(outcome.drafts) < count:
            self._log.warning(
                "%s: %d of %d questions usable", scenario.id, len(outcome.drafts), count
            )
        return outcome

    def _discard(
        self, scenario: Scenario, position: int, raw: dict[str, Any], reasons: list[str]
    ) -> RejectedQuestion:
        """A returned question that is not used: logged and traced with its reasons (never
        dropped silently, even when the reply as a whole passed validation)."""
        self._log.warning(
            "%s: question %d discarded: %s", scenario.id, position, "; ".join(reasons)
        )
        record_validation(f"question {position} of {scenario.id} (final)", 0, reasons)
        return _reject(scenario, raw, reasons)

    def _check(
        self,
        raw: dict[str, Any],
        product: ProductDescription,
        identifiers: Sequence[str],
        enforce_leaks: bool,
        rules: PersonaQuestionRules,
        request: ContextRequest,
        known: Mapping[str, str] | None = None,
    ) -> QuestionDraft | list[str]:
        """Every reason a returned question cannot be used (the output contract's schema and
        enum values, the hard policy, the persona's style rules, the Evaluation Context, a
        reference leak, an exact duplicate of a persona example / typical user question); the
        validated draft when there is none. Used for retry feedback AND final acceptance."""
        try:
            draft = QuestionDraft.model_validate(raw)
        except ValidationError as exc:
            return [
                f"{format_validation_error(exc)} (use exactly the field names and the enum "
                "values the output contract lists)"
            ]
        reasons: list[str] = []
        if enforce_leaks and (leaks := find_leaks(draft.utterance, identifiers)):
            reasons.append(f"reference leak: {leaks}")
        traits = draft.question_characteristics
        reasons.extend(question_policy_problems(draft, product, self._settings.generation.question))
        reasons.extend(mapping_problems(draft.evaluation_context, request))
        if traits.style is QuestionStyle.VAGUE:
            if not rules.allow_vague:
                reasons.append(VAGUE_NOT_ALLOWED)
            if traits.ambiguity_level is AmbiguityLevel.LOW:
                reasons.append(VAGUE_LOW)
        if known and (duplicate_of := known.get(_normalise(draft.utterance))) is not None:
            reasons.append(f"exact duplicate of {duplicate_of}: write a new question")
        return reasons or draft


def complexity_for_plan(plan: QuestionPlan) -> QuestionComplexity:
    """Compute complexity from structure instead of trusting a model-provided label."""
    sources = {source.casefold() for item in plan.deliverables for source in item.sources}
    if len(plan.deliverables) == 1 and len(sources) <= 1:
        return QuestionComplexity.SIMPLE
    if len(sources) >= 2 or len(plan.deliverables) >= 3:
        return QuestionComplexity.COMPLEX
    return QuestionComplexity.MEDIUM


def question_policy_problems(
    draft: QuestionDraft,
    product: ProductDescription,
    policy: QuestionGenerationSettings,
) -> list[str]:
    """Return every hard-policy violation in plain language for retry feedback and audit logs."""
    problems: list[str] = []
    plan = draft.plan
    deliverables = plan.deliverables
    if len(deliverables) > policy.max_deliverables:
        problems.append(
            f"plan has {len(deliverables)} deliverables; maximum is {policy.max_deliverables}"
        )

    source_names = [source for item in deliverables for source in item.sources]
    distinct_sources = {source.casefold() for source in source_names}
    if len(distinct_sources) > policy.max_sources:
        problems.append(
            f"plan uses {len(distinct_sources)} data sources; maximum is {policy.max_sources}"
        )
    known_sources = {source.name.casefold() for source in product.data_sources}
    unknown = sorted({source for source in source_names if source.casefold() not in known_sources})
    if known_sources and unknown:
        problems.append(f"plan uses unknown product data sources: {unknown}")

    for index, item in enumerate(deliverables, start=1):
        if len({source.casefold() for source in item.sources}) != len(item.sources):
            problems.append(f"deliverable {index} repeats a data source")
        if item.action is not DeliverableAction.COMPARE and len(item.sources) > 1:
            problems.append(
                f"deliverable {index} action {item.action.value} must use at most one source"
            )
        if (
            item.action is DeliverableAction.COMPARE
            and len(item.sources) > 1
            and not policy.allow_cross_source_comparison
        ):
            problems.append("cross-source comparison is disabled")

    if plan.is_follow_up and not policy.allow_follow_ups:
        problems.append("follow-up questions are disabled")
    words = _WORD.findall(draft.utterance)
    if len(words) > policy.max_words:
        problems.append(f"utterance has {len(words)} words; maximum is {policy.max_words}")
    criteria = draft.evaluation_context.success_criteria
    if len(criteria) > policy.max_success_criteria:
        problems.append(
            f"question has {len(criteria)} success criteria; maximum is "
            f"{policy.max_success_criteria}"
        )
    complexity = complexity_for_plan(plan)
    allowed_complexities = {QuestionComplexity(item) for item in policy.allowed_complexities}
    if complexity not in allowed_complexities:
        allowed = ", ".join(item.value for item in sorted(allowed_complexities, key=str))
        sources = {source.casefold() for item in plan.deliverables for source in item.sources}
        problems.append(
            f"computed complexity {complexity.value} is not allowed ({allowed}): the plan has "
            f"{len(plan.deliverables)} deliverable(s) from {len(sources)} distinct source(s); "
            "simple = 1 deliverable from at most 1 source, complex = 2+ distinct sources or "
            "3+ deliverables, otherwise medium"
        )
    return problems


def _reject(scenario: Scenario, raw: dict[str, Any], reasons: list[str]) -> RejectedQuestion:
    return RejectedQuestion(scenario_id=scenario.id, payload=raw, reasons=reasons)


def _normalise(text: str) -> str:
    """Case / punctuation / spacing normalization for EXACT duplicate detection only."""
    return " ".join(_NON_ALNUM.sub(" ", text.casefold()).split())


def _grounding(
    scenario: Scenario,
    reference: Reference | None,
    packs: Sequence[ReferencePack],
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
