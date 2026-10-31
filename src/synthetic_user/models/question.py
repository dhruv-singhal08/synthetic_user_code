"""Question cases: one natural-language question the synthetic user asks the product."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import Field, model_validator

from synthetic_user.models.base import AmbiguityLevel, NonEmptyStr, StrictModel
from synthetic_user.models.evaluation_profile import EvaluationProfile
from synthetic_user.models.run import GenerationMode, mode_mismatch
from synthetic_user.models.scenario import TaskType


class QuestionStyle(StrEnum):
    PRECISE = "precise"
    VAGUE = "vague"
    INVESTIGATIVE = "investigative"


class QuestionComplexity(StrEnum):
    """Computed from the number of deliverables and sources in the question plan."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class DeliverableAction(StrEnum):
    """One observable output requested from the product."""

    LOOKUP = "lookup"
    RANK = "rank"
    TREND = "trend"
    AGGREGATE = "aggregate"
    COMPARE = "compare"
    EXPLAIN = "explain"
    SUMMARIZE = "summarize"
    QUOTE = "quote"
    RECOMMEND = "recommend"


class QuestionIntent(StrictModel):
    task_type: TaskType


class QuestionCharacteristics(StrictModel):
    style: QuestionStyle
    ambiguity_level: AmbiguityLevel
    complexity: QuestionComplexity


MAX_SUCCESS_CRITERIA = 4
MAX_DELIVERABLES = 3
# Canonical reference definitions / uploaded golden-set criteria kept per question.
MAX_REFERENCE_CONTEXT = 40
MAX_GOLDEN_SET_CRITERIA = 40


class QuestionDeliverable(StrictModel):
    """A single action, kept separate so several outputs cannot be hidden in one sentence."""

    action: DeliverableAction
    subject: NonEmptyStr
    sources: list[NonEmptyStr] = Field(default_factory=list, max_length=2)
    limit: int | None = Field(default=None, ge=1, le=100)


class QuestionFilter(StrictModel):
    name: NonEmptyStr
    value: NonEmptyStr


class QuestionPlan(StrictModel):
    """Machine-checkable plan behind an utterance; filters do not count as deliverables."""

    deliverables: list[QuestionDeliverable] = Field(min_length=1, max_length=MAX_DELIVERABLES)
    filters: list[QuestionFilter] = Field(default_factory=list, max_length=8)
    is_follow_up: bool = False


class QuestionDraftCharacteristics(StrictModel):
    """The model describes style and ambiguity; code computes complexity from the plan."""

    style: QuestionStyle
    ambiguity_level: AmbiguityLevel


class Expectations(StrictModel):
    """What the evaluator judges against (design v3). The question owns these."""

    expected_behavior: NonEmptyStr = Field(
        description="1-3 sentences: what the product should do for this exact question"
    )
    success_criteria: list[NonEmptyStr] = Field(
        min_length=1,
        max_length=MAX_SUCCESS_CRITERIA,
        description="must-haves of a goal-complete answer; may be negative "
        '("does not mix currencies")',
    )
    expected_tools: list[NonEmptyStr] = Field(default_factory=list, max_length=12)
    expected_journey: NonEmptyStr | None = None
    expected_time_period: NonEmptyStr | None = None
    expected_data_sources: list[NonEmptyStr] = Field(default_factory=list, max_length=8)
    expected_context: list[NonEmptyStr] = Field(default_factory=list, max_length=12)
    prohibited_behaviors: list[NonEmptyStr] = Field(default_factory=list, max_length=12)
    reference_context: list[NonEmptyStr] = Field(
        default_factory=list,
        max_length=MAX_REFERENCE_CONTEXT,
        description="Canonical tool/source/workflow definitions copied from reference data",
    )
    golden_set_criteria: dict[NonEmptyStr, NonEmptyStr] = Field(
        default_factory=dict,
        max_length=MAX_GOLDEN_SET_CRITERIA,
        description="Every non-empty expected criterion preserved exactly from the uploaded row",
    )


class QuestionCase(StrictModel):
    question_id: NonEmptyStr = Field(pattern=r"^Q_\d{3,}$")
    scenario_id: NonEmptyStr = Field(pattern=r"^SCN_\d{3,}$")
    golden_case_id: NonEmptyStr | None = Field(default=None, pattern=r"^GOLD_\d{3,}$")
    persona_id: NonEmptyStr
    mode: GenerationMode
    evaluation_profile: EvaluationProfile = EvaluationProfile.NL2SQL
    reference_pack_id: NonEmptyStr | None = None
    utterance: NonEmptyStr
    intended_goal: NonEmptyStr
    intent: QuestionIntent
    question_characteristics: QuestionCharacteristics
    expectations: Expectations
    # Optional only for backward compatibility with artifacts created before the plan existed.
    plan: QuestionPlan | None = None
    # Reference names (identifier-like core_ref names) the utterance repeats.
    leak_warnings: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.evaluation_profile is EvaluationProfile.NL2SQL:
            problem = mode_mismatch(
                self.mode, self.reference_pack_id is not None, "a reference_pack_id"
            )
        else:
            problem = (
                "Agent questions must not have a reference_pack_id"
                if self.reference_pack_id is not None
                else None
            )
        if problem:
            raise ValueError(f"question {self.question_id}: {problem}")
        return self


class QuestionDraft(StrictModel):
    """LLM-facing question: no ids, no goal (copied from the scenario), no leak warnings."""

    utterance: NonEmptyStr
    plan: QuestionPlan
    intent: QuestionIntent
    question_characteristics: QuestionDraftCharacteristics
    expectations: Expectations


class QuestionBatchDraft(StrictModel):
    """Output contract for one question-generation call (one scenario)."""

    questions: list[QuestionDraft]


class RejectedQuestion(StrictModel):
    scenario_id: NonEmptyStr
    payload: dict[str, Any]
    reasons: list[NonEmptyStr] = Field(min_length=1)


class QuestionGenerationResult(StrictModel):
    questions: list[QuestionCase] = Field(default_factory=list)
    rejected: list[RejectedQuestion] = Field(default_factory=list)
