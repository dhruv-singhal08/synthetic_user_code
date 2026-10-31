"""Node 3: questions per scenario with profile-specific answer keys.

NL2SQL grounded questions link to a validated schema pack. Agent questions instead own their
answer key through expected_behavior and success_criteria and never link to a schema pack.
Calls run concurrently and results are assembled in stable scenario order.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field, ValidationError

from synthetic_user.core.config import AppConfig, QuestionGenerationSettings
from synthetic_user.core.exceptions import (
    ArtifactIntegrityError,
    InputAssetError,
    LLMOutputValidationError,
    format_validation_error,
)
from synthetic_user.core.identifiers import IdKind, SequentialIdGenerator
from synthetic_user.core.trace import trace_subject
from synthetic_user.golden_set import ground_agent_expectations
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.base import AmbiguityLevel, StrictModel
from synthetic_user.models.evaluation_profile import EvaluationProfile
from synthetic_user.models.golden_set import GoldenSet
from synthetic_user.models.persona import AmbiguityPreference, Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import (
    DeliverableAction,
    Expectations,
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
from synthetic_user.models.scenario import CapabilityClassification, Scenario
from synthetic_user.prompts.question import QuestionPromptOptions, build_question_prompt
from synthetic_user.references.leaks import find_leaks, pack_identifiers

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

# Persona question_preferences.ambiguity -> (style mix level, vague questions allowed).
PERSONA_AMBIGUITY = {
    AmbiguityPreference.NONE: (AmbiguityLevel.LOW, False),
    AmbiguityPreference.OCCASIONAL: (AmbiguityLevel.MEDIUM, True),
    AmbiguityPreference.FREQUENT: (AmbiguityLevel.HIGH, True),
}
VAGUE_NOT_ALLOWED = "vague_not_allowed"
SURPLUS = "surplus: more questions returned than requested"

# A plan item must describe one observable output. These narrow terms catch common cases where
# the model hides several outputs in one subject (for example "themes with representative
# comments" or "complaint volume and categories"). Filters and descriptive qualifiers are
# intentionally absent to avoid treating "themes with low satisfaction" as two outputs.
_OUTPUT_TERMS: dict[str, re.Pattern[str]] = {
    "themes": re.compile(r"\b(?:themes?|topics?)\b", re.IGNORECASE),
    "comments": re.compile(r"\b(?:comments?|quotes?|excerpts?)\b", re.IGNORECASE),
    "examples": re.compile(r"\bexamples?\b", re.IGNORECASE),
    "volume": re.compile(r"\bvolume\b", re.IGNORECASE),
    "count": re.compile(r"\bcounts?|totals?\b", re.IGNORECASE),
    "categories": re.compile(r"\bcategor(?:y|ies)\b", re.IGNORECASE),
    "trend": re.compile(r"\btrends?\b", re.IGNORECASE),
    "ranking": re.compile(r"\brank(?:ing|ings)?\b", re.IGNORECASE),
    "breakdown": re.compile(r"\bbreakdowns?\b", re.IGNORECASE),
    "recommendation": re.compile(r"\brecommendations?\b", re.IGNORECASE),
    "explanation": re.compile(r"\bexplanations?\b", re.IGNORECASE),
    "summary": re.compile(r"\bsummar(?:y|ies)\b", re.IGNORECASE),
}
_OUTPUT_CONNECTOR = re.compile(r"\b(?:and|plus|with|along with|as well as)\b", re.IGNORECASE)
_SECOND_REQUEST = re.compile(
    r"(?:,\s*)?\band\s+(?:what|which|how|why|where|when|who|"
    r"provide|include|show|list|give|quote|summarize|explain|identify|rank|compare|recommend)\b|"
    r"\b(?:along with|as well as)\b",
    re.IGNORECASE,
)
_TOP_NUMBER = re.compile(r"\btop\s*[-\N{EN DASH}\N{EM DASH}]?\s*(\d+)\b", re.IGNORECASE)
_RANKING_LIMIT_EXPECTATION = re.compile(
    r"\btop\s*[-\N{EN DASH}\N{EM DASH}]?\s*n\b|\btop\s+\d+\b|"
    r"\blimit\s+(?:the\s+)?(?:results?|rows?)\s+to\b|"
    r"\b(?:results?|rows?)\s+limit(?:ed)?\s+to\s+\d+\b|\blimit\s+\d+\b",
    re.IGNORECASE,
)
_CURRENCY_LANGUAGE = re.compile(
    r"\bcurrenc(?:y|ies)\b|\b(?:usd|eur|gbp|cad|aud|jpy|inr)\b|"
    r"\b(?:exchange|fx)\s+rates?\b|\b(?:dollars?|euros?|pounds?)\b",
    re.IGNORECASE,
)
_CONDITIONAL_LANGUAGE = re.compile(r"\b(?:if|when|unless|where)\b", re.IGNORECASE)


class _RawQuestionBatch(StrictModel):
    """Each question is validated individually so one bad item is rejected on its own."""

    questions: list[dict[str, Any]] = Field(default_factory=list)


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
        evaluation_profile: EvaluationProfile = EvaluationProfile.NL2SQL,
        golden_set: GoldenSet | None = None,
    ) -> QuestionGenerationResult:
        """``reference`` and ``packs`` are needed only for grounded scenarios."""
        per_scenario = (
            count if count is not None else self._settings.generation.question.question_count
        )
        if per_scenario < 1:
            raise ValueError("question count must be >= 1")
        groundings = [_grounding(s, reference, packs, evaluation_profile) for s in scenarios]
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
                    evaluation_profile,
                    reference,
                    golden_set,
                )
                for s, grounding in zip(scenarios, groundings, strict=True)
            )
        )

        result = QuestionGenerationResult()
        seen = _NearDuplicates(self._settings.generation.question.near_duplicate_threshold)
        for utterance in persona.example_utterances:
            seen.add(utterance, "persona example utterance")
        for utterance in product.typical_user_questions:
            seen.add(utterance, "product typical user question")
        if golden_set is not None:
            for case in golden_set.cases:
                seen.add(case.question, f"golden case {case.case_id}")
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
                    evaluation_profile=evaluation_profile,
                    reference_pack_id=(
                        scenario.reference_pack_id
                        if evaluation_profile is EvaluationProfile.NL2SQL
                        else None
                    ),
                    utterance=draft.utterance,
                    intended_goal=scenario.business_goal,
                    intent=draft.intent,
                    question_characteristics=QuestionCharacteristics(
                        style=draft.question_characteristics.style,
                        ambiguity_level=draft.question_characteristics.ambiguity_level,
                        complexity=complexity_for_plan(draft.plan),
                    ),
                    expectations=draft.expectations,
                    plan=draft.plan,
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
        evaluation_profile: EvaluationProfile,
        reference: Reference | None,
        golden_set: GoldenSet | None,
    ) -> _ScenarioOutcome:
        cfg = self._settings.generation.question
        rules = persona_question_rules(persona, self._settings)
        # Leak check only in grounded mode (there are no identifiers otherwise).
        enforce_leaks = bool(identifiers) and cfg.reject_reference_leaks and not rules.technical
        options = QuestionPromptOptions(
            count=count,
            max_deliverables=cfg.max_deliverables,
            max_sources=cfg.max_sources,
            max_success_criteria=cfg.max_success_criteria,
            max_words=cfg.max_words,
            allowed_complexities=tuple(
                QuestionComplexity(item) for item in cfg.allowed_complexities
            ),
            allow_follow_ups=cfg.allow_follow_ups,
            allow_cross_source_comparison=cfg.allow_cross_source_comparison,
            agent_min_tool_calls=cfg.agent_min_tool_calls,
            agent_min_data_sources=cfg.agent_min_data_sources,
            ambiguity_level=rules.ambiguity_level,
            language=cfg.language,
            allow_vague_questions=rules.allow_vague,
            technical_persona=rules.technical,
            agent_golden_examples=cfg.agent_golden_examples,
        )
        system, user = build_question_prompt(
            persona,
            product,
            scenario,
            grounding,
            options,
            evaluation_profile=evaluation_profile,
            agent_reference=reference if evaluation_profile is EvaluationProfile.AGENT else None,
            golden_set=golden_set,
        )
        last: list[_RawQuestionBatch] = []
        avoid_questions = _NearDuplicates(cfg.near_duplicate_threshold)
        if golden_set is not None:
            for case in golden_set.cases:
                avoid_questions.add(case.question, f"golden case {case.case_id}")

        def post_validate(batch: _RawQuestionBatch) -> list[str]:
            last[:] = [batch]
            errors: list[str] = []
            for i, raw in enumerate(batch.questions, start=1):
                if enforce_leaks and (
                    leaks := find_leaks(str(raw.get("utterance", "")), identifiers)
                ):
                    errors.append(
                        f"question {i}: uses internal reference names {leaks}; rephrase in "
                        "the persona's own words"
                    )
                try:
                    draft = QuestionDraft.model_validate(raw)
                except ValidationError:
                    # Schema failures are rejected item-by-item below. Retrying here would let
                    # one malformed item consume retries for otherwise valid questions.
                    continue
                errors.extend(
                    f"question {i}: {problem}"
                    for problem in question_policy_problems(draft, product, cfg)
                )
                if (
                    golden_set is not None
                    and evaluation_profile is EvaluationProfile.AGENT
                    and reference is not None
                ):
                    try:
                        expectations = ground_agent_expectations(
                            draft.expectations,
                            reference,
                            require_tools=(
                                scenario.capability is CapabilityClassification.SUPPORTED
                            ),
                            require_sources=(
                                scenario.capability is CapabilityClassification.SUPPORTED
                            ),
                        )
                        if scenario.capability is CapabilityClassification.SUPPORTED:
                            errors.extend(
                                f"question {i}: {problem}"
                                for problem in agent_operational_complexity_problems(
                                    draft, expectations, cfg
                                )
                            )
                    except InputAssetError as exc:
                        errors.append(f"question {i}: {exc}")
                if golden_set is not None and (
                    duplicate_of := avoid_questions.find(draft.utterance)
                ):
                    errors.append(
                        f"question {i}: near-duplicate of {duplicate_of}; create a genuinely "
                        "new question inspired by, not copied from, the golden set"
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
            checked = self._check(
                raw,
                product,
                identifiers,
                enforce_leaks,
                rules,
                evaluation_profile=evaluation_profile,
                reference=reference,
                require_agent_grounding=(scenario.capability is CapabilityClassification.SUPPORTED),
                ground_agent_contract=golden_set is not None,
            )
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
        self,
        raw: dict[str, Any],
        product: ProductDescription,
        identifiers: Sequence[str],
        enforce_leaks: bool,
        rules: PersonaQuestionRules,
        *,
        evaluation_profile: EvaluationProfile,
        reference: Reference | None,
        require_agent_grounding: bool,
        ground_agent_contract: bool,
    ) -> QuestionDraft | list[str]:
        try:
            draft = QuestionDraft.model_validate(raw)
        except ValidationError as exc:
            return [format_validation_error(exc)]
        reasons: list[str] = []
        if enforce_leaks and (leaks := find_leaks(draft.utterance, identifiers)):
            reasons.append(f"reference leak: {leaks}")
        traits = draft.question_characteristics
        reasons.extend(question_policy_problems(draft, product, self._settings.generation.question))
        if traits.style is QuestionStyle.VAGUE:
            if not rules.allow_vague:
                reasons.append(VAGUE_NOT_ALLOWED)
            if traits.ambiguity_level is AmbiguityLevel.LOW:
                reasons.append("vague question must have ambiguity_level medium or high")
        if evaluation_profile is EvaluationProfile.AGENT and ground_agent_contract:
            if reference is None:
                reasons.append("Agent question generation requires operational Reference data")
            else:
                try:
                    expectations = ground_agent_expectations(
                        draft.expectations,
                        reference,
                        require_tools=require_agent_grounding,
                        require_sources=require_agent_grounding,
                    )
                    draft = draft.model_copy(update={"expectations": expectations})
                    if require_agent_grounding:
                        reasons.extend(
                            agent_operational_complexity_problems(
                                draft, expectations, self._settings.generation.question
                            )
                        )
                except InputAssetError as exc:
                    reasons.append(str(exc))
        return reasons or draft


def complexity_for_plan(plan: QuestionPlan) -> QuestionComplexity:
    """Compute complexity from structure instead of trusting a model-provided label."""
    sources = {source.casefold() for item in plan.deliverables for source in item.sources}
    if len(plan.deliverables) == 1 and len(sources) <= 1:
        return QuestionComplexity.SIMPLE
    if len(sources) >= 2 or len(plan.deliverables) >= 3:
        return QuestionComplexity.COMPLEX
    return QuestionComplexity.MEDIUM


def agent_operational_complexity_problems(
    draft: QuestionDraft,
    expectations: Expectations,
    policy: QuestionGenerationSettings,
) -> list[str]:
    """Enforce selected Agent complexity using observable plans and canonical expectations."""

    plan_sources = {
        source.casefold() for item in draft.plan.deliverables for source in item.sources
    }
    expected_sources = {source.casefold() for source in expectations.expected_data_sources}
    problems: list[str] = []
    if len(plan_sources) < policy.agent_min_data_sources:
        problems.append(
            f"plan uses {len(plan_sources)} data source(s); selected Agent complexity requires "
            f"at least {policy.agent_min_data_sources}"
        )
    if len(expected_sources) < policy.agent_min_data_sources:
        problems.append(
            f"expected_data_sources names {len(expected_sources)} canonical source(s); selected "
            f"Agent complexity requires at least {policy.agent_min_data_sources}"
        )
    if len(expectations.expected_tools) < policy.agent_min_tool_calls:
        problems.append(
            f"expected_tools records {len(expectations.expected_tools)} tool call(s); selected "
            f"Agent complexity requires at least {policy.agent_min_tool_calls}"
        )
    return problems


def _output_terms(text: str) -> set[str]:
    return {name for name, pattern in _OUTPUT_TERMS.items() if pattern.search(text)}


def _compound_output_problem(subject: str) -> str | None:
    """Explain when one subject joins two independently observable output types."""
    for connector in _OUTPUT_CONNECTOR.finditer(subject):
        before = _output_terms(subject[: connector.start()])
        after = _output_terms(subject[connector.end() :])
        if before and after:
            combined = ", ".join(sorted(before | after))
            return f"combines multiple requested outputs ({combined}); split them into items"
    return None


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
        if problem := _compound_output_problem(item.subject):
            problems.append(f"deliverable {index} {problem}")
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

    if len(deliverables) == 1 and _SECOND_REQUEST.search(draft.utterance):
        problems.append(
            "utterance appears to request a second output but plan has one deliverable; "
            "split the outputs into separate plan items or simplify the question"
        )

    if plan.is_follow_up and not policy.allow_follow_ups:
        problems.append("follow-up questions are disabled")
    words = re.findall(r"\b[\w'-]+\b", draft.utterance)
    if len(words) > policy.max_words:
        problems.append(f"utterance has {len(words)} words; maximum is {policy.max_words}")
    if len(draft.expectations.success_criteria) > policy.max_success_criteria:
        problems.append(
            f"question has {len(draft.expectations.success_criteria)} success criteria; maximum "
            f"is {policy.max_success_criteria}"
        )
    problems.extend(expectation_policy_problems(draft))
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


def expectation_policy_problems(draft: QuestionDraft) -> list[str]:
    """Reject expectations that add unsupported limits or unconditional currency duties.

    The LLM still writes the natural-language expectations. These deterministic checks cover the
    two high-impact additions that are machine-checkable from the question plan: invented result
    counts and data-state-dependent currency safeguards presented as unconditional requirements.
    """
    problems: list[str] = []
    plan_limits = {item.limit for item in draft.plan.deliverables if item.limit is not None}
    utterance_limits = {int(value) for value in _TOP_NUMBER.findall(draft.utterance)}
    missing_from_plan = sorted(utterance_limits - plan_limits)
    if missing_from_plan:
        problems.append(
            f"utterance requests top {missing_from_plan} but plan deliverables do not record "
            "the same numeric limit"
        )

    expectation_texts = [
        ("expected_behavior", draft.expectations.expected_behavior),
        *(
            (f"success criterion {index}", criterion)
            for index, criterion in enumerate(draft.expectations.success_criteria, start=1)
        ),
    ]
    question_requests_currency = bool(_CURRENCY_LANGUAGE.search(draft.utterance))
    for label, text in expectation_texts:
        if _RANKING_LIMIT_EXPECTATION.search(text):
            if not plan_limits:
                problems.append(
                    f"{label} invents a top-N/result limit that is absent from the plan"
                )
            else:
                stated = {int(value) for value in _TOP_NUMBER.findall(text)}
                mismatched = sorted(stated - plan_limits)
                if mismatched:
                    problems.append(
                        f"{label} requires top {mismatched}, which does not match plan limit(s) "
                        f"{sorted(plan_limits)}"
                    )
        if (
            _CURRENCY_LANGUAGE.search(text)
            and not question_requests_currency
            and not _CONDITIONAL_LANGUAGE.search(text)
        ):
            problems.append(
                f"{label} adds an unconditional currency requirement not requested by the "
                'utterance; express a necessary data-dependent safeguard as "If ..."'
            )
    return problems


def _reject(scenario: Scenario, raw: dict[str, Any], reasons: list[str]) -> RejectedQuestion:
    return RejectedQuestion(scenario_id=scenario.id, payload=raw, reasons=reasons)


class _NearDuplicates:
    """Utterances seen so far, compared by exact text and token similarity."""

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold
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
            if union and len(tokens & other_tokens) / len(union) >= self._threshold:
                return owner
        return None


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _grounding(
    scenario: Scenario,
    reference: Reference | None,
    packs: Sequence[ReferencePack],
    evaluation_profile: EvaluationProfile = EvaluationProfile.NL2SQL,
) -> tuple[ReferencePack, Reference] | None:
    """Grounded scenario -> (its pack, the reference); ungrounded -> None."""
    if evaluation_profile is EvaluationProfile.AGENT or scenario.mode == GenerationMode.UNGROUNDED:
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
