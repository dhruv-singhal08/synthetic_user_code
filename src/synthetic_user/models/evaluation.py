"""Evaluation of one product execution: checks, findings, verdict, bug candidates.

Generic checks apply to every product; grounded_checks are filled by artifact plugins (e.g.
"sql", nodes/evaluators/plugins/) and only in grounded mode. Plugin-specific models live with
their plugin.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, JsonValue, model_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.execution import ProductOutcome
from synthetic_user.models.run import GenerationMode

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


class GenericChecks(StrictModel):
    """Product-agnostic criteria, judged in both modes."""

    relevance: CheckStatus
    goal_completeness: CheckStatus
    persona_fit: CheckStatus
    capability_consistency: CheckStatus
    clarification_handling: CheckStatus
    safety_and_honesty: CheckStatus


class Finding(StrictModel):
    finding_id: NonEmptyStr = Field(pattern=r"^F_\d{3,}$")
    category: NonEmptyStr
    description: NonEmptyStr
    expected_behavior: NonEmptyStr
    observed_behavior: NonEmptyStr
    # Exact quote copied from the product response (text, message or an artifact such as SQL).
    evidence: str | None = None
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
    checks: GenericChecks | None = None
    grounded_checks: dict[NonEmptyStr, CheckStatus] = Field(default_factory=dict)
    check_rationale: dict[NonEmptyStr, NonEmptyStr] = Field(
        default_factory=dict, description="check name -> why it got its status"
    )
    plugins_applied: list[NonEmptyStr] = Field(default_factory=list)
    criteria_results: list[CriterionResult] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    clarification_evaluation: ClarificationEvaluation | None = None
    bug_candidates: list[BugCandidate] = Field(default_factory=list)
    # Plugin name -> its analysis (e.g. "sql" -> static SQL analysis), grounded mode only.
    plugin_analysis: dict[NonEmptyStr, dict[str, JsonValue]] = Field(default_factory=dict)
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
        if self.mode is GenerationMode.UNGROUNDED and (
            self.grounded_checks
            or self.cheat_sheet_conflicts
            or self.plugins_applied
            or self.plugin_analysis
        ):
            raise ValueError("ungrounded evaluations have no grounded_checks or cheat sheet")
        if self.grounded_checks and not self.plugins_applied:
            raise ValueError("grounded_checks are filled by plugins; plugins_applied is empty")
        known_checks = set(GenericChecks.model_fields) | set(self.grounded_checks)
        unknown_rationale = sorted(set(self.check_rationale) - known_checks)
        if unknown_rationale:
            raise ValueError(f"check_rationale for unknown checks {unknown_rationale}")
        finding_ids = {f.finding_id for f in self.findings}
        for bug in self.bug_candidates:
            unknown = sorted(set(bug.finding_ids) - finding_ids)
            if unknown:
                raise ValueError(f"{bug.bug_id} references unknown findings {unknown}")
        return self


# --- generic (all products) LLM-facing draft -------------------------------------------------


class CheckJudgement(StrictModel):
    status: CheckStatus
    rationale: NonEmptyStr
    evidence: str | None = Field(
        default=None, description="exact quote copied from the product response, if any"
    )


class GenericFindingDraft(StrictModel):
    category: NonEmptyStr = Field(description="the generic check this finding explains")
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


class GenericEvaluationDraft(StrictModel):
    """One judgement per generic check, per-criterion results, optional clarification detail."""

    relevance: CheckJudgement
    goal_completeness: CheckJudgement
    criteria: list[CriterionResult] = Field(default_factory=list)
    persona_fit: CheckJudgement
    capability_consistency: CheckJudgement
    clarification_handling: CheckJudgement
    clarification: ClarificationEvaluation | None = None
    safety_and_honesty: CheckJudgement
    findings: list[GenericFindingDraft] = Field(default_factory=list)
    overall_verdict: Verdict = Field(
        description="your own overall opinion; recorded as llm_verdict, not decisive"
    )

    def judgements(self) -> dict[str, CheckJudgement]:
        return {name: getattr(self, name) for name in GenericChecks.model_fields}


class ReferenceCoverageDraft(StrictModel):
    """LLM draft of the grounded reference_coverage check (no artifact plugin applied)."""

    reference_coverage: CheckJudgement
    covered: list[NonEmptyStr] = Field(default_factory=list)
    missing: list[NonEmptyStr] = Field(default_factory=list)
    findings: list[GenericFindingDraft] = Field(default_factory=list)
