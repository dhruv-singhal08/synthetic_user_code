"""Evaluation of one product execution: 5 quality metrics, claims, criteria, findings,
verdict, bug candidates (KPI design: docs/DESIGN_V3_REFERENCE_AND_EVALUATION.md section 11).

Only answered responses are judged. Clarifications, denials, errors and timeouts are recorded
only (verdict None). Metrics score 0..1:
- relevancy, correctness: judged 0 / 0.5 / 1 (correctness: grounded questions only);
- completeness: from the criteria results (met 1, partly 0.5, not met 0; unverifiable left out);
- groundedness: supported claims / all claims (grounded only);
- hallucination_rate: contradicted (grounded) or fabricated (ungrounded) claims / all claims.
Scores are computed by code from the judge's draft; the judge never gives the verdict.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.execution import ProductOutcome
from synthetic_user.models.run import GenerationMode

# Outcomes that are recorded but not judged: verdict None exactly for these.
RECORDED_ONLY_OUTCOMES = frozenset(
    {
        ProductOutcome.CLARIFICATION,
        ProductOutcome.DENIED,
        ProductOutcome.ERROR,
        ProductOutcome.TIMEOUT,
    }
)


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


class MetricName(StrEnum):
    RELEVANCY = "relevancy"
    CORRECTNESS = "correctness"
    GROUNDEDNESS = "groundedness"
    HALLUCINATION_RATE = "hallucination_rate"
    COMPLETENESS = "completeness"


# Lower is better for these (pass = score 0); higher is better for the others (pass = 1).
LOWER_IS_BETTER = frozenset({MetricName.HALLUCINATION_RATE})

# Metrics shown in results.csv, kpis.json / kpis.csv and the UI (D206). Groundedness is still
# computed and stored in evaluations.json, but not reported for now (it overlaps correctness).
REPORTED_METRICS = (
    MetricName.RELEVANCY,
    MetricName.CORRECTNESS,
    MetricName.HALLUCINATION_RATE,
    MetricName.COMPLETENESS,
)


class MetricStatus(StrEnum):
    SCORED = "scored"
    INCONCLUSIVE = "inconclusive"  # applies, but could not be decided
    NOT_APPLICABLE = "not_applicable"  # e.g. correctness without a reference


class MetricResult(StrictModel):
    status: MetricStatus
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: NonEmptyStr
    evidence: str | None = None  # exact quote from the response, if any

    @model_validator(mode="after")
    def _score_iff_scored(self) -> Self:
        if (self.status is MetricStatus.SCORED) != (self.score is not None):
            raise ValueError("score is set exactly when status is scored")
        return self


class Metrics(StrictModel):
    relevancy: MetricResult
    correctness: MetricResult
    groundedness: MetricResult
    hallucination_rate: MetricResult
    completeness: MetricResult

    def by_name(self) -> dict[MetricName, MetricResult]:
        return {name: getattr(self, name.value) for name in MetricName}


class ClaimStatus(StrEnum):
    # grounded questions (checked against the reference pack / reference)
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_IN_REFERENCE = "not_in_reference"
    # ungrounded questions (no reference to check against)
    UNVERIFIED = "unverified"  # plausible, cannot be checked
    FABRICATED = "fabricated"  # a specific fact/figure stated as certain with no basis


GROUNDED_CLAIM_STATUSES = frozenset(
    {ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED, ClaimStatus.NOT_IN_REFERENCE}
)
UNGROUNDED_CLAIM_STATUSES = frozenset({ClaimStatus.UNVERIFIED, ClaimStatus.FABRICATED})
HALLUCINATED = frozenset({ClaimStatus.CONTRADICTED, ClaimStatus.FABRICATED})


class Claim(StrictModel):
    text: NonEmptyStr = Field(description="the claim, copied exactly from the response")
    status: ClaimStatus
    reason: NonEmptyStr
    reference_quote: str | None = Field(
        default=None, description="supporting / contradicting passage copied from the reference"
    )


class Finding(StrictModel):
    finding_id: NonEmptyStr = Field(pattern=r"^F_\d{3,}$")
    category: NonEmptyStr  # a metric name (or a plugin check, when plugins are connected)
    description: NonEmptyStr
    expected_behavior: NonEmptyStr
    observed_behavior: NonEmptyStr
    # Exact quote copied from the product response (text or an artifact such as SQL).
    evidence: str | None = None
    severity: Severity
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="uncalibrated; None when not estimated"
    )
    evidence_status: EvidenceStatus
    source: FindingSource
    matches_known_gap: bool = False


class CriterionStatus(StrEnum):
    MET = "met"
    PARTLY = "partly"
    NOT_MET = "not_met"
    # Cannot be checked without ground truth (e.g. whether a stated figure is correct).
    UNVERIFIABLE = "unverifiable"


class CriterionResult(StrictModel):
    criterion: NonEmptyStr
    status: CriterionStatus
    rationale: NonEmptyStr
    evidence: str | None = None  # exact quote from the response, if any


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
    evaluation_id: NonEmptyStr = Field(pattern=r"^EVAL_\d{3,}$")
    question_id: NonEmptyStr
    scenario_id: NonEmptyStr
    execution_id: NonEmptyStr
    product_outcome: ProductOutcome
    mode: GenerationMode
    verdict: Verdict | None
    evaluation_status: EvaluationStatus
    metrics: Metrics | None = None
    claims: list[Claim] = Field(default_factory=list)
    criteria_results: list[CriterionResult] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    bug_candidates: list[BugCandidate] = Field(default_factory=list)
    llm_verdict: Verdict | None = None  # the judge's own opinion; never decisive
    notes: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        recorded_outcome = self.product_outcome in RECORDED_ONLY_OUTCOMES
        if recorded_outcome and self.verdict is not None:
            raise ValueError(f"{self.product_outcome} responses are recorded only (no verdict)")
        if not recorded_outcome and self.verdict is None:
            raise ValueError("an answered response needs a verdict")
        recorded_only = self.evaluation_status is EvaluationStatus.RECORDED_ONLY
        if recorded_only != (self.verdict is None):
            raise ValueError("evaluation_status is recorded_only exactly when verdict is None")
        if self.mode is GenerationMode.UNGROUNDED and self.metrics is not None:
            for name in (MetricName.CORRECTNESS, MetricName.GROUNDEDNESS):
                if self.metrics.by_name()[name].status is not MetricStatus.NOT_APPLICABLE:
                    raise ValueError(f"ungrounded evaluations have no {name.value}")
        allowed = (
            GROUNDED_CLAIM_STATUSES
            if self.mode is GenerationMode.GROUNDED
            else UNGROUNDED_CLAIM_STATUSES
        )
        wrong = sorted({c.status.value for c in self.claims if c.status not in allowed})
        if wrong:
            raise ValueError(f"claim statuses {wrong} do not fit mode {self.mode.value}")
        finding_ids = {f.finding_id for f in self.findings}
        for bug in self.bug_candidates:
            unknown = sorted(set(bug.finding_ids) - finding_ids)
            if unknown:
                raise ValueError(f"{bug.bug_id} references unknown findings {unknown}")
        return self


# --- LLM-facing draft (answers only) ---------------------------------------------------------


JUDGED_SCORES = (0.0, 0.5, 1.0)


class JudgedScore(StrictModel):
    score: float | None = Field(
        description="exactly 0, 0.5 or 1: 1 = fully, 0.5 = partly, 0 = not at all; "
        "null = cannot tell"
    )
    rationale: NonEmptyStr
    evidence: str | None = Field(
        default=None, description="exact quote copied from the product response, if any"
    )

    @model_validator(mode="after")
    def _three_levels(self) -> Self:
        if self.score is not None and self.score not in JUDGED_SCORES:
            raise ValueError(f"score must be 0, 0.5 or 1 (or null), not {self.score}")
        return self


class FindingDraft(StrictModel):
    category: NonEmptyStr = Field(
        description="the metric this finding explains: relevancy | correctness | "
        "groundedness | hallucination_rate | completeness"
    )
    description: NonEmptyStr
    expected_behavior: NonEmptyStr
    observed_behavior: NonEmptyStr
    evidence: str | None = Field(
        default=None, description="exact quote copied from the product response"
    )
    severity: Severity
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="uncalibrated; null if not estimated"
    )
    evidence_status: EvidenceStatus
    matches_known_gap: bool = False


class EvaluationDraft(StrictModel):
    """The judge's answer for one answered response; code turns it into metric scores."""

    relevancy: JudgedScore
    correctness: JudgedScore | None = Field(
        default=None, description="grounded questions only; null when there is no reference"
    )
    criteria: list[CriterionResult] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    findings: list[FindingDraft] = Field(default_factory=list)
    overall_verdict: Verdict = Field(
        description="your own overall opinion; recorded as llm_verdict, not decisive"
    )
