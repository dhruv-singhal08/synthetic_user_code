"""Scenarios: what the synthetic user wants to achieve; grounded ones carry a cheat sheet."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import Field, model_validator

from synthetic_user.models.base import (
    AmbiguityLevel,
    NonEmptyStr,
    QualifiedColumn,
    StrictModel,
    table_of,
)
from synthetic_user.models.reference import VerificationStatus
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


class CheatRelationship(StrictModel):
    source: QualifiedColumn
    target: QualifiedColumn
    verification: VerificationStatus = VerificationStatus.UNKNOWN


class CheatMetric(StrictModel):
    name: NonEmptyStr
    definition: NonEmptyStr
    columns: list[QualifiedColumn] = Field(default_factory=list)


class ReferenceCheatSheet(StrictModel):
    tables: list[NonEmptyStr] = Field(min_length=1)
    columns: list[QualifiedColumn] = Field(default_factory=list)
    relationships: list[CheatRelationship] = Field(default_factory=list)
    metrics: list[CheatMetric] = Field(default_factory=list)
    dimensions: list[QualifiedColumn] = Field(default_factory=list)
    operations: list[NonEmptyStr] = Field(default_factory=list)
    optional_filters: list[NonEmptyStr] = Field(default_factory=list)
    semantic_considerations: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _references_listed_tables(self) -> Self:
        listed = set(self.tables)
        refs = [
            *self.columns,
            *self.dimensions,
            *(r.source for r in self.relationships),
            *(r.target for r in self.relationships),
            *(c for m in self.metrics for c in m.columns),
        ]
        unknown = sorted({table_of(ref) for ref in refs} - listed)
        if unknown:
            raise ValueError(f"cheat sheet references tables not in its table list: {unknown}")
        return self


class QuestionGenerationGuidance(StrictModel):
    intent: NonEmptyStr
    expected_ambiguity: AmbiguityLevel = AmbiguityLevel.LOW
    phrasing_hints: list[NonEmptyStr] = Field(default_factory=list)
    terms_to_avoid: list[NonEmptyStr] = Field(default_factory=list)


SuccessCriteria = Annotated[
    list[NonEmptyStr],
    Field(min_length=1, description="what a goal-complete answer must contain"),
]


class ScenarioDraft(StrictModel):
    """LLM-facing scenario: business goal first; no ids, parent ids, mode or signature.

    reference_cheat_sheet is present only in grounded mode (the node enforces that).
    """

    business_goal: NonEmptyStr
    title: NonEmptyStr
    task_type: TaskType
    capability: CapabilityClassification
    capability_rationale: NonEmptyStr
    success_criteria: SuccessCriteria
    reference_cheat_sheet: ReferenceCheatSheet | None = None
    question_guidance: QuestionGenerationGuidance


class ScenarioBatchDraft(StrictModel):
    """Output contract for one scenario-generation call."""

    scenarios: list[ScenarioDraft]


class Scenario(StrictModel):
    id: NonEmptyStr = Field(pattern=r"^SCN_\d{3,}$")
    persona_id: NonEmptyStr
    product_id: NonEmptyStr
    mode: GenerationMode
    reference_index_id: NonEmptyStr | None = None
    business_goal: NonEmptyStr
    title: NonEmptyStr
    task_type: TaskType
    capability: CapabilityClassification
    capability_rationale: NonEmptyStr
    success_criteria: SuccessCriteria
    reference_cheat_sheet: ReferenceCheatSheet | None = None
    question_guidance: QuestionGenerationGuidance
    signature: NonEmptyStr

    @model_validator(mode="after")
    def _reference_matches_mode(self) -> Self:
        problems = [
            mode_mismatch(self.mode, self.reference_cheat_sheet is not None, "a cheat sheet"),
            mode_mismatch(self.mode, self.reference_index_id is not None, "a reference_index_id"),
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


class ScenarioGenerationConfig(StrictModel):
    scenario_count: int = Field(ge=1)
    allow_multi_table: bool
    include_unsupported_scenarios: bool
    diversity_preference: NonEmptyStr
    max_topup_rounds: int = Field(default=1, ge=0)
