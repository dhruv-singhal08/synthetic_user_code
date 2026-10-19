"""Scenarios: what the synthetic user wants to achieve; grounded ones have a Reference Pack
(stored in reference_packs.json, linked by reference_pack_id)."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import Field, model_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.reference import (
    CatalogPackDraft,
    DocumentPackDraft,
    ReferencePack,
)
from synthetic_user.models.run import GenerationMode, mode_mismatch


class TaskType(StrEnum):
    RANKING = "ranking"
    AGGREGATION = "aggregation"
    TREND = "trend"
    COMPARISON = "comparison"
    FILTER_LOOKUP = "filter_lookup"
    DISTRIBUTION = "distribution"
    ANOMALY_INVESTIGATION = "anomaly_investigation"
    # Product-agnostic tasks (chatbots, search, summarisers, ...).
    EXPLANATION = "explanation"
    HOW_TO = "how_to"
    SUMMARIZATION = "summarization"
    RECOMMENDATION = "recommendation"
    TROUBLESHOOTING = "troubleshooting"
    OTHER = "other"


class CapabilityClassification(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


SuccessCriteria = Annotated[
    list[NonEmptyStr],
    Field(min_length=1, description="what a goal-complete answer must contain"),
]


class ScenarioDraft(StrictModel):
    """LLM-facing scenario: business goal first; no ids, parent ids, mode or signature.

    reference_pack is present only in grounded mode (the node enforces that).
    """

    business_goal: NonEmptyStr
    title: NonEmptyStr
    typical_task: NonEmptyStr | None = Field(
        default=None,
        description="the persona typical_task this scenario serves, copied exactly; null for "
        "another task this persona would clearly do",
    )
    task_type: TaskType
    capability: CapabilityClassification
    capability_rationale: NonEmptyStr
    success_criteria: SuccessCriteria
    # What the user will ask for, in one sentence (never the wording: that is the persona's).
    question_intent: NonEmptyStr
    reference_pack: CatalogPackDraft | DocumentPackDraft | None = None


class ScenarioBatchDraft(StrictModel):
    """Output contract for one scenario-generation call."""

    scenarios: list[ScenarioDraft]


class Scenario(StrictModel):
    id: NonEmptyStr = Field(pattern=r"^SCN_\d{3,}$")
    persona_id: NonEmptyStr
    product_id: NonEmptyStr
    mode: GenerationMode
    reference_id: NonEmptyStr | None = None
    reference_pack_id: NonEmptyStr | None = None
    business_goal: NonEmptyStr
    title: NonEmptyStr
    typical_task: NonEmptyStr | None = None  # the persona's typical task it serves (exact text)
    task_type: TaskType
    capability: CapabilityClassification
    capability_rationale: NonEmptyStr
    # Guidance for question generation only; each question owns its own criteria (D186).
    success_criteria: SuccessCriteria
    question_intent: NonEmptyStr
    signature: NonEmptyStr

    @model_validator(mode="after")
    def _reference_matches_mode(self) -> Self:
        problems = [
            mode_mismatch(self.mode, self.reference_id is not None, "a reference_id"),
            mode_mismatch(self.mode, self.reference_pack_id is not None, "a reference_pack_id"),
        ]
        found = [p for p in problems if p]
        if found:
            raise ValueError(f"scenario {self.id}: {'; '.join(found)}")
        return self


class RejectedScenario(StrictModel):
    payload: dict[str, Any]
    reasons: list[NonEmptyStr] = Field(min_length=1)


class ScenarioGenerationResult(StrictModel):
    accepted: list[Scenario] = Field(default_factory=list)
    rejected: list[RejectedScenario] = Field(default_factory=list)
    duplicates: list[RejectedScenario] = Field(default_factory=list)
    packs: list[ReferencePack] = Field(default_factory=list)  # grounded only
