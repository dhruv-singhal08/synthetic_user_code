"""Evaluation of one product execution against the run's frozen metric configuration.

No evaluation profiles: every selected metric gets exactly one ``MetricResult`` keyed by its
metric id, with an explicit status when it was not scored (the status taxonomy below). Two
separate result columns describe the response: ``product_outcome`` (the executor's typed
transport result, never overwritten) and ``response_classification`` (its semantic kind).

Error, timeout and unresolved clarification are recorded only (no judge call, verdict None).
A substantive typed denial IS judged (schema_version 2); earlier artifacts (schema_version 1)
recorded denials only and stay readable unchanged.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, JsonValue, model_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.execution import ProductOutcome, ResponseClassification
from synthetic_user.models.metrics import JUDGED_SCALE, JudgeGroup
from synthetic_user.models.run import GenerationMode

EVALUATION_SCHEMA_VERSION = 2
# Outcomes that are recorded but never judged (verdict None exactly for these).
RECORDED_ONLY_OUTCOMES = frozenset(
    {ProductOutcome.CLARIFICATION, ProductOutcome.ERROR, ProductOutcome.TIMEOUT}
)
# Before schema 2 a typed denial was recorded only as well.
LEGACY_RECORDED_ONLY_OUTCOMES = RECORDED_ONLY_OUTCOMES | {ProductOutcome.DENIED}


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class CheckStatus(StrEnum):
    """Status of a check (kept for the disconnected artifact-plugin hook)."""

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    NOT_APPLICABLE = "not_applicable"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvidenceStatus(StrEnum):
    SUPPORTED = "supported"
    WEAK = "weak"
    UNSUPPORTED = "unsupported"


class FindingSource(StrEnum):
    DETERMINISTIC = "deterministic"
    LLM = "llm"


class BugStatus(StrEnum):
    NEEDS_REVIEW = "needs_review"


class EvaluationStatus(StrEnum):
    EVALUATED = "evaluated"
    RECORDED_ONLY = "recorded_only"


class ClassificationSource(StrEnum):
    TYPED_OUTCOME = "typed_outcome"  # denial / clarification outcome of the executor
    ADAPTER = "adapter"  # the adapter's typed response-kind mapping
    GENERAL_JUDGE = "general_judge"  # returned within the normal General Judge call
    UNRESOLVED = "unresolved"  # no typed field and no general judge call


class MetricStatus(StrEnum):
    SCORED = "scored"
    NOT_APPLICABLE = "not_applicable"  # genuinely irrelevant for this question / response
    CANNOT_GENERATE = "cannot_generate"  # the required question expectation is absent
    EVIDENCE_UNAVAILABLE = "evidence_unavailable"  # applicable, but telemetry / context missing
    INCONCLUSIVE = "inconclusive"  # the judge could not decide despite the evidence
    EVALUATION_ERROR = "evaluation_error"  # judge / parse / API failure
    NOT_ENABLED = "not_enabled"  # not selected for the run (reporting only)
    # Selected custom metric whose definition could not be compiled (reason kept).
    UNAVAILABLE = "unavailable"


# Non-scored statuses: never averaged and never turned into 0 or 1.
NON_SCORED_STATUSES = tuple(s for s in MetricStatus if s is not MetricStatus.SCORED)


class MetricResult(StrictModel):
    status: MetricStatus
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: NonEmptyStr  # the judge's reason, or why the metric was not scored
    evidence: str | None = None  # exact quote from the response / SQL / chunk, if any
    what_worked: list[NonEmptyStr] = Field(default_factory=list)
    what_didnt: list[NonEmptyStr] = Field(default_factory=list)
    judge_group: JudgeGroup | None = None
    # Non-scored results: what was missing and which component failed.
    missing_requirement: NonEmptyStr | None = None
    failure_source: NonEmptyStr | None = None
    # Retrieved chunk ids / source ids / tool-call orders the result relied on.
    evidence_refs: list[NonEmptyStr] = Field(default_factory=list)
    # Metric-specific transparency (e.g. MRR first relevant rank, recall checklist coverage).
    details: dict[NonEmptyStr, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _score_iff_scored(self) -> Self:
        if (self.status is MetricStatus.SCORED) != (self.score is not None):
            raise ValueError("score is set exactly when status is scored")
        return self


class Finding(StrictModel):
    finding_id: NonEmptyStr = Field(pattern=r"^F_\d{3,}$")
    category: NonEmptyStr  # the metric id this finding explains
    description: NonEmptyStr
    expected_behavior: NonEmptyStr
    observed_behavior: NonEmptyStr
    evidence: str | None = None  # exact quote from the response, SQL, tool trace or a chunk
    severity: Severity
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="uncalibrated; None when not estimated"
    )
    evidence_status: EvidenceStatus
    source: FindingSource
    matches_known_gap: bool = False
    # Stable links into the tagged success criteria and question-specific cheat sheet.
    related_tags: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _read_legacy_category(cls, value: Any) -> Any:
        if isinstance(value, dict):
            category = value.get("category")
            if category in _LEGACY_FINDING_CATEGORIES:
                return {**value, "category": _LEGACY_FINDING_CATEGORIES[category]}
        return value


# Category names written before the VoC -> Agent and metric renames (kept as written then).
_LEGACY_FINDING_CATEGORIES = {
    "relevancy": "semantic_correctness",
    "accuracy": "semantic_correctness",
    "correctness": "semantic_correctness",
    "completeness": "semantic_correctness",
    "answer_relevance": "relevance",
    "answer_completeness": "completion",
    "retrieval_faithfulness": "tool_selection",
}


class CriterionStatus(StrEnum):
    MET = "met"
    PARTLY = "partly"
    NOT_MET = "not_met"
    # The response does not contain enough information to decide this criterion.
    UNVERIFIABLE = "unverifiable"


class CriterionResult(StrictModel):
    tag: NonEmptyStr | None = None
    criterion: NonEmptyStr
    status: CriterionStatus
    rationale: NonEmptyStr
    evidence: str | None = None
    # The selected metric this criterion feeds (None in older artifacts).
    metric_id: NonEmptyStr | None = None


class BugCandidate(StrictModel):
    bug_id: NonEmptyStr = Field(pattern=r"^BUG_\d{3,}$")
    title: NonEmptyStr
    description: NonEmptyStr
    reproduction_question: NonEmptyStr
    product_id: NonEmptyStr
    question_id: NonEmptyStr
    scenario_id: NonEmptyStr
    execution_id: NonEmptyStr
    finding_ids: list[NonEmptyStr] = Field(min_length=1)
    status: BugStatus = BugStatus.NEEDS_REVIEW


class EvaluationResult(StrictModel):
    schema_version: int = Field(default=EVALUATION_SCHEMA_VERSION, ge=1)
    evaluation_id: NonEmptyStr = Field(pattern=r"^EVAL_\d{3,}$")
    question_id: NonEmptyStr
    scenario_id: NonEmptyStr
    execution_id: NonEmptyStr
    product_outcome: ProductOutcome
    response_classification: ResponseClassification | None = None
    classification_source: ClassificationSource | None = None
    classification_note: NonEmptyStr | None = None
    mode: GenerationMode
    metric_config_id: NonEmptyStr | None = None
    # Profile of a schema-1 evaluation, kept as a label for legacy reports only.
    legacy_evaluation_profile: NonEmptyStr | None = None
    verdict: Verdict | None
    verdict_policy: NonEmptyStr | None = None
    evaluation_status: EvaluationStatus
    # Keyed by metric id: every selected metric for an evaluated response; empty otherwise.
    metrics: dict[NonEmptyStr, MetricResult] = Field(default_factory=dict)
    criteria_results: list[CriterionResult] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    bug_candidates: list[BugCandidate] = Field(default_factory=list)
    llm_verdict: Verdict | None = None  # the general judge's own opinion; never decisive
    notes: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _read_legacy(cls, value: Any) -> Any:
        """Stored schema-1 artifacts carry neither a schema_version nor a metric config id."""
        if not isinstance(value, dict) or "schema_version" in value or "metric_config_id" in value:
            return value
        return migrate_legacy_evaluation(value)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        recorded_outcomes = (
            RECORDED_ONLY_OUTCOMES
            if self.schema_version >= EVALUATION_SCHEMA_VERSION
            else LEGACY_RECORDED_ONLY_OUTCOMES
        )
        recorded_outcome = self.product_outcome in recorded_outcomes
        if recorded_outcome and self.verdict is not None:
            raise ValueError(f"{self.product_outcome} responses are recorded only (no verdict)")
        if not recorded_outcome and self.verdict is None:
            raise ValueError(f"an evaluable {self.product_outcome} response needs a verdict")
        recorded_only = self.evaluation_status is EvaluationStatus.RECORDED_ONLY
        if recorded_only != (self.verdict is None):
            raise ValueError("evaluation_status is recorded_only exactly when verdict is None")
        finding_ids = {f.finding_id for f in self.findings}
        for bug in self.bug_candidates:
            unknown = sorted(set(bug.finding_ids) - finding_ids)
            if unknown:
                raise ValueError(f"{bug.bug_id} references unknown findings {unknown}")
        return self


def migrate_legacy_evaluation(old: dict[str, Any]) -> dict[str, Any]:
    """Schema-1 evaluation (NL2SQL / Agent profile era) -> current shape, read-only.

    Metric results keep their ORIGINAL ids (``completion``, ``tool_selection``, ...): they were
    judged under the old rubrics and are never relabelled as new metrics. The profile is kept
    as a label. Very old metric names are migrated the way the previous readers did.
    """
    data = {key: item for key, item in old.items() if key != "claims"}
    profile = data.pop("evaluation_profile", None)
    profile_name = "nl2sql" if profile in (None, "nl2sql") else "agent"
    data["schema_version"] = 1
    data["legacy_evaluation_profile"] = profile_name
    metrics = data.get("metrics")
    data["metrics"] = _legacy_metrics(metrics, profile_name) if isinstance(metrics, dict) else {}
    if profile_name == "nl2sql":
        # A generic "faithfulness" finding of that era meant NL2SQL schema faithfulness.
        data["findings"] = [
            {**finding, "category": "schema_faithfulness"}
            if isinstance(finding, dict) and finding.get("category") == "faithfulness"
            else finding
            for finding in data.get("findings", [])
        ]
    return data


_VERY_OLD_METRICS = (
    "relevancy",
    "accuracy",
    "correctness",
    "completeness",
    "task_completion",
    "groundedness",
    "hallucination_rate",
)


def _legacy_metrics(metrics: dict[str, Any], profile: str) -> dict[str, Any]:
    data = {name: result for name, result in metrics.items() if result is not None}
    renamed = {
        "answer_relevance": "relevance",
        "answer_completeness": "completion",
        "retrieval_faithfulness": "tool_selection",
    }
    for old, new in renamed.items():
        if old in data:
            data.setdefault(new, data.pop(old))
    if profile == "nl2sql" and not {"relevance", "completion"} & set(data):
        if "semantic_correctness" not in data:
            for old in ("accuracy", "correctness", "completeness", "task_completion"):
                if old in data:
                    data["semantic_correctness"] = data[old]
                    break
        if "schema_faithfulness" not in data and "faithfulness" in data:
            data["schema_faithfulness"] = data.pop("faithfulness")
    for name in _VERY_OLD_METRICS:
        data.pop(name, None)
    return data


# --- LLM-facing drafts (answers only; code assigns ids, statuses and verdicts) ---------------


class FactualVerification(StrEnum):
    VERIFIED = "verified"  # checked against trusted evidence supplied to the judge
    PARTLY_VERIFIED = "partly_verified"
    UNVERIFIED = "unverified"  # no trusted evidence: correctness is not asserted
    NOT_APPLICABLE = "not_applicable"  # nothing factual to verify (e.g. a justified decline)


class JudgedScore(StrictModel):
    score: float | None = Field(
        description="exactly 0, 0.5 or 1: 1 = fully, 0.5 = partly, 0 = not at all; "
        "null = cannot tell"
    )
    rationale: NonEmptyStr
    evidence: str | None = Field(
        default=None, description="exact quote copied from the response / SQL / chunk, if any"
    )
    evidence_ids: list[NonEmptyStr] = Field(
        default_factory=list, description="retrieved chunk ids / tool-call orders relied on"
    )
    what_worked: list[NonEmptyStr] = Field(default_factory=list)
    what_didnt: list[NonEmptyStr] = Field(default_factory=list)
    factual_verification: FactualVerification | None = None

    @model_validator(mode="after")
    def _three_levels(self) -> Self:
        if self.score is not None and self.score not in JUDGED_SCALE:
            raise ValueError(f"score must be 0, 0.5 or 1 (or null), not {self.score}")
        return self


class FindingDraft(StrictModel):
    category: NonEmptyStr = Field(description="the metric id this finding explains")
    description: NonEmptyStr
    expected_behavior: NonEmptyStr
    observed_behavior: NonEmptyStr
    evidence: str | None = None
    severity: Severity
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_status: EvidenceStatus
    matches_known_gap: bool = False
    related_tags: list[NonEmptyStr] = Field(default_factory=list)


class CriterionDraft(StrictModel):
    criterion: NonEmptyStr
    status: CriterionStatus
    rationale: NonEmptyStr
    evidence: str | None = None
    metric_id: NonEmptyStr | None = None


class GeneralJudgeDraft(StrictModel):
    """The General Judge's reply for one evaluable response."""

    # Required when code asks for it (answered payload without a typed response kind).
    response_classification: Literal["answered", "unable_to_answer"] | None = None
    classification_note: NonEmptyStr | None = None
    metrics: dict[NonEmptyStr, JudgedScore]
    criteria: list[CriterionDraft] = Field(default_factory=list)
    findings: list[FindingDraft] = Field(default_factory=list)
    overall_verdict: Verdict


class ChunkRelevanceDraft(StrictModel):
    chunk_id: NonEmptyStr
    relevant: bool
    rationale: NonEmptyStr


class CoverageStatus(StrEnum):
    COVERED = "covered"
    PARTLY = "partly"
    NOT_COVERED = "not_covered"


class RequiredInformationDraft(StrictModel):
    item: NonEmptyStr  # copied exactly from required_information
    status: CoverageStatus
    chunk_ids: list[NonEmptyStr] = Field(default_factory=list)
    rationale: NonEmptyStr


class SupportStatus(StrEnum):
    SUPPORTED = "supported"
    PARTLY = "partly"
    UNSUPPORTED = "unsupported"


class CitationCheckDraft(StrictModel):
    citation: NonEmptyStr  # the citation id (or "#<n>" for the n-th listed citation)
    chunk_id: NonEmptyStr | None = None
    support: SupportStatus
    rationale: NonEmptyStr


class RetrievalJudgeDraft(StrictModel):
    """The Retrieval/Grounding Judge's reply; MRR is computed by code from chunk_relevance."""

    metrics: dict[NonEmptyStr, JudgedScore] = Field(default_factory=dict)
    chunk_relevance: list[ChunkRelevanceDraft] = Field(default_factory=list)
    required_information: list[RequiredInformationDraft] = Field(default_factory=list)
    citation_checks: list[CitationCheckDraft] = Field(default_factory=list)
    findings: list[FindingDraft] = Field(default_factory=list)
