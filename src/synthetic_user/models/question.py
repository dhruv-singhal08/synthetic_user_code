"""Question cases: one natural-language question the synthetic user asks the product."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from synthetic_user.models.base import AmbiguityLevel, NonEmptyStr, StrictModel
from synthetic_user.models.reference import ReferenceSnapshot
from synthetic_user.models.scenario import TaskType


class ClarificationExpectation(StrEnum):
    REQUIRED = "required"
    CONDITIONAL = "conditional"
    NOT_EXPECTED = "not_expected"


class QuestionStyle(StrEnum):
    PRECISE = "precise"
    VAGUE = "vague"
    INVESTIGATIVE = "investigative"


class QuestionIntent(StrictModel):
    task_type: TaskType
    metrics: list[NonEmptyStr] = Field(default_factory=list)
    dimensions: list[NonEmptyStr] = Field(default_factory=list)
    filters: list[NonEmptyStr] = Field(default_factory=list)
    time_range: str | None = None


class QuestionCharacteristics(StrictModel):
    style: QuestionStyle = QuestionStyle.PRECISE
    ambiguity_level: AmbiguityLevel
    uses_business_terms: bool = True
    language: NonEmptyStr = "en"


class Expectations(StrictModel):
    clarification: ClarificationExpectation
    acceptable_behaviors: list[NonEmptyStr] = Field(min_length=1)
    unacceptable_behaviors: list[NonEmptyStr] = Field(default_factory=list)


class QuestionCase(StrictModel):
    question_id: NonEmptyStr = Field(pattern=r"^Q_\d{3,}$")
    scenario_id: NonEmptyStr = Field(pattern=r"^SCN_\d{3,}$")
    persona_id: NonEmptyStr
    reference_snapshot_id: NonEmptyStr
    utterance: NonEmptyStr
    intended_goal: NonEmptyStr
    intent: QuestionIntent
    question_characteristics: QuestionCharacteristics
    expectations: Expectations
    interpretation_notes: list[NonEmptyStr] = Field(default_factory=list)
    schema_leak_warnings: list[NonEmptyStr] = Field(default_factory=list)


class QuestionGenerationConfig(StrictModel):
    question_count: int = Field(ge=1)
    ambiguity_level: AmbiguityLevel
    language: NonEmptyStr
    allow_vague_questions: bool
    reject_schema_leaks: bool


class QuestionDraft(StrictModel):
    """LLM-facing question: no ids, no goal (copied from the scenario), no leak warnings."""

    utterance: NonEmptyStr
    intent: QuestionIntent
    question_characteristics: QuestionCharacteristics
    expectations: Expectations
    interpretation_notes: list[NonEmptyStr] = Field(default_factory=list)


class QuestionBatchDraft(StrictModel):
    """Output contract for one question-generation call (one scenario)."""

    questions: list[QuestionDraft]


class RejectedQuestion(StrictModel):
    scenario_id: NonEmptyStr
    payload: dict[str, Any]
    reasons: list[NonEmptyStr] = Field(min_length=1)


class QuestionGenerationResult(StrictModel):
    snapshots: list[ReferenceSnapshot] = Field(default_factory=list)
    questions: list[QuestionCase] = Field(default_factory=list)
    rejected: list[RejectedQuestion] = Field(default_factory=list)
