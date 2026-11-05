"""Run-level metric selection and the frozen metric configuration (no evaluation profiles).

The user picks any built-in metric ids and may add custom metric descriptions (free text and/or
a workbook). Before scenario generation the selection is resolved ONCE into an immutable
``ResolvedMetricConfig`` (built-in definitions + compiled custom definitions, including the ones
that could not be compiled). Generation, execution, evaluation and reporting all read that one
snapshot. Metric ids are validated strings, not a closed enum, so compiled custom ids work.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel, find_duplicates

METRIC_ID_PATTERN = r"^[a-z][a-z0-9_]{1,63}$"
EXPECTATION_FIELD_PATTERN = r"^[a-z][a-z0-9_]{1,63}$"
# The category of the finding made from the General Judge's goal-completion assessment (D243):
# not a metric, so a metric never takes this id. "general" / "cheat_sheet" were success_criteria
# keys of the D240 contract and stay reserved so stored metric configurations keep their meaning.
GOAL_COMPLETION_CATEGORY = "goal_completion"
RESERVED_METRIC_IDS = frozenset({"general", "cheat_sheet", GOAL_COMPLETION_CATEGORY})
# The 0 / 0.5 / 1 scale shared by every judged metric.
JUDGED_SCALE = (0.0, 0.5, 1.0)
METRIC_CONFIG_SCHEMA_VERSION = 1


class MetricOrigin(StrEnum):
    BUILTIN = "builtin"
    CUSTOM = "custom"


class JudgeGroup(StrEnum):
    """Which evaluation call scores a metric. ``deterministic`` = code only, no LLM."""

    GENERAL = "general"
    RETRIEVAL_GROUNDING = "retrieval_grounding"
    DETERMINISTIC = "deterministic"


class MetricCategory(StrEnum):
    """UI / report grouping only. A category never selects a judge or a product type."""

    ANSWER_QUALITY = "answer_quality"
    RETRIEVAL_QUALITY = "retrieval_quality"
    GROUNDING_CITATIONS = "grounding_citations"
    SQL_QUALITY = "sql_quality"
    AGENT = "agent"
    CUSTOM = "custom"


class EvidenceKind(StrEnum):
    """Execution evidence a metric needs. Checked by code after execution."""

    ANSWER = "answer"  # a terminal product response with text or artifacts
    SQL = "sql"  # an artifact of kind "sql"
    TOOL_TRACE = "tool_trace"  # an exposed, ordered tool-call trace
    RETRIEVED_CONTEXT = "retrieved_context"  # the normalized context artifact is available
    RANKED_CONTEXT = "ranked_context"  # retrieved chunks carry an explicit rank
    CITATIONS = "citations"  # the adapter exposes citation -> chunk mappings
    # Retrieved context, or tool results in an exposed tool trace (faithfulness).
    GROUNDING_EVIDENCE = "grounding_evidence"


class ReferenceAsset(StrEnum):
    """Authoritative reference material a metric needs (always the user's own data)."""

    CATALOG_SCHEMA = "catalog_schema"  # cheat-sheet tables + columns (data_catalog)
    CATALOG_RULES = "catalog_rules"  # cheat-sheet rules, table rules or documented joins
    TOOL_DEFINITIONS = "tool_definitions"  # reference agent_context tools


class EvaluationTarget(StrEnum):
    """What a metric judges, derived from its required execution evidence (D234).

    ``sql``: only the SQL artifact; ``both``: the SQL artifact and other evidence (e.g. the
    answer text); ``non_sql``: no SQL artifact; ``unknown``: the definition names no evidence
    (routing then stays as before: the judge group alone decides). Never from the metric name.
    """

    SQL = "sql"
    NON_SQL = "non_sql"
    BOTH = "both"
    UNKNOWN = "unknown"


class ExpectationFieldType(StrEnum):
    STRING = "string"
    STRING_LIST = "string_list"
    BOOLEAN = "boolean"


class MetricAvailability(StrEnum):
    AVAILABLE = "available"
    # Compiled custom metric that cannot be evaluated (reason kept; shown in results).
    UNAVAILABLE = "unavailable"


class MissingEvidencePolicy(StrEnum):
    """What code records when a required execution evidence is absent from an answer."""

    EVIDENCE_UNAVAILABLE = "evidence_unavailable"
    NOT_APPLICABLE = "not_applicable"
    # The missing evidence IS the failure (e.g. no SQL from an SQL-generating answer): score 0
    # with a deterministic finding - unless the product declined the request.
    SCORE_ZERO = "score_zero"


class EmptyContextPolicy(StrEnum):
    """What code records when context is available but zero chunks were retrieved."""

    JUDGE = "judge"
    NOT_APPLICABLE = "not_applicable"
    SCORE_ZERO = "score_zero"


class ExpectationField(StrictModel):
    """One question-specific value a metric needs (never a copy of reference data)."""

    name: NonEmptyStr = Field(pattern=EXPECTATION_FIELD_PATTERN)
    type: ExpectationFieldType
    description: NonEmptyStr
    required: bool = True


class MetricDefinition(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_id: NonEmptyStr = Field(pattern=METRIC_ID_PATTERN)
    display_name: NonEmptyStr
    category: MetricCategory
    version: NonEmptyStr
    origin: MetricOrigin
    description: NonEmptyStr
    # What the judge compares and how it scores; anchors are keyed "1", "0.5", "0".
    rubric: NonEmptyStr
    score_anchors: dict[NonEmptyStr, NonEmptyStr]
    scale: list[float] = Field(default_factory=lambda: list(JUDGED_SCALE))
    # True for metrics computed by code on a continuous 0..1 range (MRR's reciprocal rank).
    continuous_score: bool = False
    higher_is_better: bool = True
    judge_group: JudgeGroup
    # Only question-specific values; authoritative reference data is never duplicated here.
    question_expectation_schema: list[ExpectationField] = Field(default_factory=list)
    generation_instructions: NonEmptyStr | None = None
    required_reference_assets: list[ReferenceAsset] = Field(default_factory=list)
    required_execution_evidence: list[EvidenceKind] = Field(default_factory=list)
    applicability_policy: NonEmptyStr
    evidence_requirements: NonEmptyStr
    when_evidence_missing: MissingEvidencePolicy = MissingEvidencePolicy.EVIDENCE_UNAVAILABLE
    when_no_chunks: EmptyContextPolicy = EmptyContextPolicy.JUDGE
    # The judge must state whether factual content was verified against trusted evidence.
    asks_factual_verification: bool = False
    # Whether the provisional verdict policy reads this metric.
    counts_toward_verdict: bool = True
    # Evaluation Context (reforming): what the metric compares, how it is verified and whether
    # the question's Success Criteria are mapped to it. None = derived from the evidence the
    # metric needs (custom metrics and metric configurations frozen before these fields).
    comparison_basis: NonEmptyStr | None = None
    verification_method: NonEmptyStr | None = None
    uses_success_criteria: bool | None = None
    # Code maps EVERY success criterion to the metric (no generator choice, D256).
    all_success_criteria: bool = False
    # Reference Spec assets the metric also checks when the scenario's pack has them (its
    # schema / applicable rules become checks); unlike required_reference_assets a missing one
    # never makes the metric not applicable (D256).
    checked_reference_assets: list[ReferenceAsset] = Field(default_factory=list)
    availability: MetricAvailability = MetricAvailability.AVAILABLE
    unavailable_reason: NonEmptyStr | None = None
    # Custom metrics: the user's exact words and where they came from (text / workbook row).
    source_text: NonEmptyStr | None = None
    source_provenance: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        problems: list[str] = []
        if self.metric_id in RESERVED_METRIC_IDS:
            problems.append(
                f"the id is reserved for a success_criteria key ({sorted(RESERVED_METRIC_IDS)})"
            )
        if dupes := find_duplicates([f.name for f in self.question_expectation_schema]):
            problems.append(f"duplicate expectation fields {dupes}")
        if self.question_expectation_schema and self.generation_instructions is None:
            problems.append("a metric with question expectations needs generation_instructions")
        if (self.availability is MetricAvailability.UNAVAILABLE) != (
            self.unavailable_reason is not None
        ):
            problems.append("unavailable_reason is set exactly when availability is unavailable")
        if self.origin is MetricOrigin.CUSTOM:
            if self.source_text is None or self.source_provenance is None:
                problems.append("a custom metric keeps its source_text and source_provenance")
            if self.category is not MetricCategory.CUSTOM:
                problems.append("a custom metric has category custom")
        if missing := sorted({"1", "0.5", "0"} - set(self.score_anchors)):
            problems.append(f"score_anchors miss {missing}")
        if problems:
            raise ValueError(f"metric {self.metric_id}: {'; '.join(problems)}")
        return self

    @property
    def available(self) -> bool:
        return self.availability is MetricAvailability.AVAILABLE

    @property
    def needs_expectations(self) -> bool:
        return bool(self.question_expectation_schema)

    @property
    def reference_assets(self) -> list[ReferenceAsset]:
        """Every Reference Spec asset the metric checks: required + checked when present."""
        return list(
            dict.fromkeys([*self.required_reference_assets, *self.checked_reference_assets])
        )

    @property
    def evaluation_target(self) -> EvaluationTarget:
        evidence = set(self.required_execution_evidence)
        if not evidence:
            return EvaluationTarget.UNKNOWN
        if EvidenceKind.SQL not in evidence:
            return EvaluationTarget.NON_SQL
        return EvaluationTarget.BOTH if evidence - {EvidenceKind.SQL} else EvaluationTarget.SQL

    @property
    def maps_criteria(self) -> bool:
        """The question's Success Criteria are mapped to this metric (Metric Mapping). Default:
        every General Judge metric (custom ones included); retrieval metrics judge chunks."""
        if self.uses_success_criteria is not None:
            return self.uses_success_criteria
        return self.judge_group is JudgeGroup.GENERAL

    @property
    def basis(self) -> str:
        """The comparison basis, from the catalog or derived from the evidence it needs."""
        if self.comparison_basis is not None:
            return self.comparison_basis
        if self.judge_group is JudgeGroup.RETRIEVAL_GROUNDING:
            return "criteria_vs_retrieved_context"
        return _DERIVED_BASIS[self.evaluation_target]

    @property
    def method(self) -> str:
        """How the metric is verified, from the catalog or derived from its evidence."""
        if self.verification_method is not None:
            return self.verification_method
        if self.judge_group is JudgeGroup.RETRIEVAL_GROUNDING:
            return "retrieved_context_review"
        return _DERIVED_METHOD[self.evaluation_target]


_DERIVED_BASIS = {
    EvaluationTarget.SQL: "criteria_vs_sql",
    EvaluationTarget.BOTH: "criteria_vs_response_and_sql",
    EvaluationTarget.NON_SQL: "criteria_vs_response",
    EvaluationTarget.UNKNOWN: "criteria_vs_response",
}
_DERIVED_METHOD = {
    EvaluationTarget.SQL: "static_sql_analysis",
    EvaluationTarget.BOTH: "response_review_and_static_sql_analysis",
    EvaluationTarget.NON_SQL: "response_review",
    EvaluationTarget.UNKNOWN: "response_review",
}


class CustomMetricInput(StrictModel):
    """One user-supplied custom metric, exactly as given, with its provenance."""

    text: NonEmptyStr
    provenance: NonEmptyStr  # e.g. "free text item 2" or "metrics.xlsx: Sheet1 row 4"


class CustomMetricInputError(StrictModel):
    provenance: NonEmptyStr
    reason: NonEmptyStr


class MetricSelection(StrictModel):
    """What the user chose for the whole run (built-in ids + custom inputs)."""

    selected_metrics: list[NonEmptyStr] = Field(default_factory=list)
    custom_inputs: list[CustomMetricInput] = Field(default_factory=list)
    # Blank / malformed spreadsheet rows: kept and reported, never silently dropped.
    input_errors: list[CustomMetricInputError] = Field(default_factory=list)

    @model_validator(mode="after")
    def _valid(self) -> Self:
        if dupes := find_duplicates(list(self.selected_metrics)):
            raise ValueError(f"metric selected more than once: {dupes}")
        if not self.selected_metrics and not self.custom_inputs:
            raise ValueError("select at least one metric or add a custom metric")
        return self

    def fingerprint(self) -> str:
        """Stable hash of the selection (custom text included); a change cascades the run.
        Built-in ids are order-insensitive; custom inputs keep their order (it assigns ids)."""
        data = self.model_dump(mode="json")
        data["selected_metrics"] = sorted(data["selected_metrics"])
        payload = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ResolvedMetricConfig(StrictModel):
    """The frozen, versioned metric configuration of one run (``metric_config.json``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    config_id: NonEmptyStr = Field(pattern=r"^MC_[0-9a-f]{12}$")
    compiler_version: NonEmptyStr
    catalog_version: NonEmptyStr
    selection_fingerprint: NonEmptyStr
    # Ordered: selected built-ins in catalog order, then custom metrics in input order.
    selected_metric_ids: list[NonEmptyStr]
    definitions: list[MetricDefinition]
    custom_inputs: list[CustomMetricInput] = Field(default_factory=list)
    input_errors: list[CustomMetricInputError] = Field(default_factory=list)
    compiler_called: bool = False
    compiler_notes: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        ids = [d.metric_id for d in self.definitions]
        if dupes := find_duplicates(ids):
            raise ValueError(f"duplicate metric ids {dupes}")
        if ids != list(self.selected_metric_ids):
            raise ValueError("selected_metric_ids must list every definition, in order")
        return self

    def definition(self, metric_id: str) -> MetricDefinition | None:
        return next((d for d in self.definitions if d.metric_id == metric_id), None)

    def active(self) -> list[MetricDefinition]:
        """Selected metrics that can be evaluated (compiled custom ones included)."""
        return [d for d in self.definitions if d.available]

    def unavailable(self) -> list[MetricDefinition]:
        return [d for d in self.definitions if not d.available]

    def needs_evidence(self, kind: EvidenceKind) -> bool:
        return any(kind in d.required_execution_evidence for d in self.active())


def config_id_for(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return "MC_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
