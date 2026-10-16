"""Scenarios: what the synthetic user wants to find out, plus a reference cheat sheet."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import Field, model_validator

from synthetic_user.models.base import (
    AmbiguityLevel,
    NonEmptyStr,
    QualifiedColumn,
    StrictModel,
    table_of,
)
from synthetic_user.models.reference import VerificationStatus


class TaskType(StrEnum):
    RANKING = "ranking"
    AGGREGATION = "aggregation"
    TREND = "trend"
    COMPARISON = "comparison"
    FILTER_LOOKUP = "filter_lookup"
    DISTRIBUTION = "distribution"
    ANOMALY_INVESTIGATION = "anomaly_investigation"
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


class ScenarioDraft(StrictModel):
    """LLM-facing scenario: business goal first; no ids, parent ids or signature."""

    business_goal: NonEmptyStr
    title: NonEmptyStr
    task_type: TaskType
    capability: CapabilityClassification
    capability_rationale: NonEmptyStr
    cheat_sheet: ReferenceCheatSheet
    question_guidance: QuestionGenerationGuidance


class ScenarioBatchDraft(StrictModel):
    """Output contract for one scenario-generation call."""

    scenarios: list[ScenarioDraft]


class Scenario(StrictModel):
    id: NonEmptyStr = Field(pattern=r"^SCN_\d{3,}$")
    persona_id: NonEmptyStr
    product_id: NonEmptyStr
    reference_index_id: NonEmptyStr
    business_goal: NonEmptyStr
    title: NonEmptyStr
    task_type: TaskType
    capability: CapabilityClassification
    capability_rationale: NonEmptyStr
    cheat_sheet: ReferenceCheatSheet
    question_guidance: QuestionGenerationGuidance
    signature: NonEmptyStr


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
