"""Question cases: one natural-language question the synthetic user asks the product."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import Field, model_validator

from synthetic_user.models.base import AmbiguityLevel, NonEmptyStr, StrictModel
from synthetic_user.models.run import GenerationMode, mode_mismatch
from synthetic_user.models.scenario import TaskType


class QuestionStyle(StrEnum):
    PRECISE = "precise"
    VAGUE = "vague"
    INVESTIGATIVE = "investigative"


class QuestionComplexity(StrEnum):
    """Matches the persona's question_preferences.complexity (mixed = any of these)."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class QuestionIntent(StrictModel):
    task_type: TaskType


class QuestionCharacteristics(StrictModel):
    style: QuestionStyle
    ambiguity_level: AmbiguityLevel
    complexity: QuestionComplexity


MAX_SUCCESS_CRITERIA = 4


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
    # Reference names (identifier-like core_ref names) the utterance repeats.
    leak_warnings: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        problem = mode_mismatch(
            self.mode, self.reference_pack_id is not None, "a reference_pack_id"
        )
        if problem:
            raise ValueError(f"question {self.question_id}: {problem}")
        return self


class QuestionDraft(StrictModel):
    """LLM-facing question: no ids, no goal (copied from the scenario), no leak warnings."""

    utterance: NonEmptyStr
    intent: QuestionIntent
    question_characteristics: QuestionCharacteristics
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
