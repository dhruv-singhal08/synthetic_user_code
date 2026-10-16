"""Evaluation of one product execution: checks, findings, verdict, bug candidates."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.execution import ProductOutcome

# Outcomes that are recorded but not judged: verdict may be None only for these.
RECORDED_ONLY_OUTCOMES = frozenset(
    {ProductOutcome.DENIED, ProductOutcome.ERROR, ProductOutcome.TIMEOUT}
)


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class CheckStatus(StrEnum):
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


class ClarificationRelevance(StrEnum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    UNCLEAR = "unclear"


class ClarificationNecessity(StrEnum):
    NECESSARY = "necessary"
    REASONABLE = "reasonable"
    UNNECESSARY = "unnecessary"
    UNCLEAR = "unclear"


class BugStatus(StrEnum):
    NEEDS_REVIEW = "needs_review"


class EvaluationStatus(StrEnum):
    EVALUATED = "evaluated"
    RECORDED_ONLY = "recorded_only"


class EvaluationChecks(StrictModel):
    intent_alignment: CheckStatus
    table_selection: CheckStatus
    column_selection: CheckStatus
    join_correctness: CheckStatus
    filter_correctness: CheckStatus
    aggregation_correctness: CheckStatus
    sql_syntax: CheckStatus
    semantic_considerations: CheckStatus


class Finding(StrictModel):
    finding_id: NonEmptyStr = Field(pattern=r"^F_\d{3,}$")
    category: NonEmptyStr
    description: NonEmptyStr
    expected_behavior: NonEmptyStr
    observed_behavior: NonEmptyStr
    sql_evidence: str | None = None
    severity: Severity
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="uncalibrated; None when not estimated"
    )
    evidence_status: EvidenceStatus
    source: FindingSource
    matches_known_gap: bool = False


class ClarificationEvaluation(StrictModel):
    relevance: ClarificationRelevance
    necessity: ClarificationNecessity
    asks_for_given_info: bool = False
    reason: NonEmptyStr


class BugCandidate(StrictModel):
    bug_id: NonEmptyStr = Field(pattern=r"^BUG_\d{3,}$")
    title: NonEmptyStr
    description: NonEmptyStr
    reproduction_question: NonEmptyStr
    question_id: NonEmptyStr
    scenario_id: NonEmptyStr
    execution_id: NonEmptyStr
    finding_ids: list[NonEmptyStr] = Field(min_length=1)
    status: BugStatus = BugStatus.NEEDS_REVIEW


class SqlAnalysis(StrictModel):
    dialect: NonEmptyStr
    parse_ok: bool
    tables: list[NonEmptyStr] = Field(default_factory=list)
    columns: list[NonEmptyStr] = Field(default_factory=list)
    parse_errors: list[NonEmptyStr] = Field(default_factory=list)


class EvaluationResult(StrictModel):
    evaluation_id: NonEmptyStr = Field(pattern=r"^EVAL_\d{3,}$")
    question_id: NonEmptyStr
    scenario_id: NonEmptyStr
    execution_id: NonEmptyStr
    product_outcome: ProductOutcome
    verdict: Verdict | None
    evaluation_status: EvaluationStatus
    checks: EvaluationChecks | None = None
    findings: list[Finding] = Field(default_factory=list)
    clarification_evaluation: ClarificationEvaluation | None = None
    bug_candidates: list[BugCandidate] = Field(default_factory=list)
    sql_analysis: SqlAnalysis | None = None
    llm_verdict: Verdict | None = None
    cheat_sheet_conflicts: list[NonEmptyStr] = Field(default_factory=list)
    notes: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.verdict is None and self.product_outcome not in RECORDED_ONLY_OUTCOMES:
            raise ValueError(
                f"verdict may be None only for denied/error/timeout, not {self.product_outcome}"
            )
        recorded_only = self.evaluation_status is EvaluationStatus.RECORDED_ONLY
        if recorded_only != (self.verdict is None):
            raise ValueError("evaluation_status is recorded_only exactly when verdict is None")
        finding_ids = {f.finding_id for f in self.findings}
        for bug in self.bug_candidates:
            unknown = sorted(set(bug.finding_ids) - finding_ids)
            if unknown:
                raise ValueError(f"{bug.bug_id} references unknown findings {unknown}")
        return self


# --- LLM-facing drafts (no ids, no source; code assigns those) -------------------------------


class FindingDraft(StrictModel):
    category: NonEmptyStr = Field(
        description="the EvaluationChecks field this finding explains, e.g. join_correctness"
    )
    description: NonEmptyStr
    expected_behavior: NonEmptyStr
    observed_behavior: NonEmptyStr
    sql_evidence: str | None = Field(
        default=None, description="exact fragment copied from the SQL under evaluation"
    )
    severity: Severity
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="uncalibrated; null if not estimated"
    )
    evidence_status: EvidenceStatus
    matches_known_gap: bool = False


class SqlEvaluationDraft(StrictModel):
    checks: EvaluationChecks
    findings: list[FindingDraft] = Field(default_factory=list)
    cheat_sheet_conflicts: list[NonEmptyStr] = Field(default_factory=list)
    verdict: Verdict
