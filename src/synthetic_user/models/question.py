"""Question cases: one natural-language question the synthetic user asks the product."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import Field, model_validator

from synthetic_user.models.base import AmbiguityLevel, NonEmptyStr, StrictModel
from synthetic_user.models.reference import ReferenceSnapshot
from synthetic_user.models.run import GenerationMode, mode_mismatch
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
    # Copied or refined from the scenario; a QuestionCase requires at least one.
    success_criteria: list[NonEmptyStr] = Field(default_factory=list)


class QuestionCase(StrictModel):
    question_id: NonEmptyStr = Field(pattern=r"^Q_\d{3,}$")
    scenario_id: NonEmptyStr = Field(pattern=r"^SCN_\d{3,}$")
    persona_id: NonEmptyStr
    mode: GenerationMode
    reference_snapshot_id: NonEmptyStr | None = None
    utterance: NonEmptyStr
    intended_goal: NonEmptyStr
    intent: QuestionIntent
    question_characteristics: QuestionCharacteristics
    expectations: Expectations
    interpretation_notes: list[NonEmptyStr] = Field(default_factory=list)
    schema_leak_warnings: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        problems = [
            mode_mismatch(
                self.mode, self.reference_snapshot_id is not None, "a reference_snapshot_id"
            ),
            None if self.expectations.success_criteria else "expectations.success_criteria empty",
        ]
        found = [p for p in problems if p]
        if found:
            raise ValueError(f"question {self.question_id}: {'; '.join(found)}")
        return self


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
