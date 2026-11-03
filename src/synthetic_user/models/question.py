"""Question cases: one natural-language question the synthetic user asks the product."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import Field, JsonValue, model_validator

from synthetic_user.models.base import AmbiguityLevel, NonEmptyStr, StrictModel
from synthetic_user.models.run import GenerationMode
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


class MetricExpectationStatus(StrEnum):
    GENERATED = "generated"  # a validated, question-specific value exists
    NOT_APPLICABLE = "not_applicable"  # the metric does not apply to this question
    # The metric may apply, but its expectation cannot be established from the question and
    # the authoritative assets: no value is fabricated and the metric is never scored 0.
    CANNOT_GENERATE = "cannot_generate"


class MetricExpectation(StrictModel):
    status: MetricExpectationStatus
    value: dict[NonEmptyStr, JsonValue] | None = None
    reason: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if self.status is MetricExpectationStatus.GENERATED:
            if not self.value:
                raise ValueError("a generated metric expectation needs a value")
        elif self.value is not None:
            raise ValueError(f"a {self.status.value} metric expectation has value null")
        if self.status is MetricExpectationStatus.CANNOT_GENERATE and self.reason is None:
            raise ValueError("cannot_generate needs a specific reason")
        return self


# Fields of the pre-metric-registry expectation schema (Agent profile / golden set). Old
# artifacts keep them under ``legacy_expectations`` (labelled, never used by the new judges).
LEGACY_EXPECTATION_FIELDS = (
    "expected_tools",
    "expected_journey",
    "expected_time_period",
    "expected_data_sources",
    "expected_context",
    "prohibited_behaviors",
    "reference_context",
    "golden_set_criteria",
)


class Expectations(StrictModel):
    """What the evaluator judges against. Universal fields + selected-metric expectations only."""

    expected_behavior: NonEmptyStr = Field(
        description="1-3 sentences: what the product should do for this exact question"
    )
    success_criteria: list[NonEmptyStr] = Field(
        min_length=1,
        max_length=MAX_SUCCESS_CRITERIA,
        description="observable, fair must-haves of a goal-complete answer",
    )
    # Only metrics that need a question-specific value appear here (never empty placeholders).
    metric_expectations: dict[NonEmptyStr, MetricExpectation] = Field(default_factory=dict)
    # Read-only carrier for artifacts written before the metric registry (see module docs).
    legacy_expectations: dict[NonEmptyStr, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _read_legacy(cls, value: Any) -> Any:
        if not isinstance(value, dict) or not any(k in value for k in LEGACY_EXPECTATION_FIELDS):
            return value
        data = dict(value)
        legacy = dict(data.get("legacy_expectations") or {})
        for key in LEGACY_EXPECTATION_FIELDS:
            item = data.pop(key, None)
            if item not in (None, "", [], {}):
                legacy[key] = item
        data["legacy_expectations"] = legacy
        return data


class QuestionCase(StrictModel):
    question_id: NonEmptyStr = Field(pattern=r"^Q_\d{3,}$")
    scenario_id: NonEmptyStr = Field(pattern=r"^SCN_\d{3,}$")
    persona_id: NonEmptyStr
    mode: GenerationMode
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

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy(cls, value: Any) -> Any:
        """Artifacts from the profile / golden-set era: the profile is not a question property
        any more and golden cases no longer exist (mode and expectations are kept)."""
        if isinstance(value, dict) and ("evaluation_profile" in value or "golden_case_id" in value):
            data = dict(value)
            data.pop("evaluation_profile", None)
            data.pop("golden_case_id", None)
            return data
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        # Agent-profile questions of earlier runs were grounded without a pack: accepted on
        # read only (no new question is created that way).
        if self.mode is GenerationMode.UNGROUNDED and self.reference_pack_id is not None:
            raise ValueError(f"question {self.question_id}: mode ungrounded must not have a pack")
        return self


class MetricExpectationDraft(StrictModel):
    status: MetricExpectationStatus
    value: dict[NonEmptyStr, JsonValue] | None = None
    reason: NonEmptyStr | None = None


class ExpectationsDraft(StrictModel):
    expected_behavior: NonEmptyStr
    success_criteria: list[NonEmptyStr] = Field(min_length=1, max_length=MAX_SUCCESS_CRITERIA)
    metric_expectations: dict[NonEmptyStr, MetricExpectationDraft] = Field(default_factory=dict)


class QuestionDraft(StrictModel):
    """LLM-facing question: no ids, no goal (copied from the scenario), no leak warnings."""

    utterance: NonEmptyStr
    plan: QuestionPlan
    intent: QuestionIntent
    question_characteristics: QuestionDraftCharacteristics
    expectations: ExpectationsDraft


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
