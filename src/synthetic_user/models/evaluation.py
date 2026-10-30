"""Profile-specific evaluation of one product execution.

NL2SQL judges semantic correctness, rules adherence, and schema faithfulness. Agent always
judges relevance and completion against the question's expected behaviour and success criteria;
tool/context metrics are optional and apply only when the response exposes their telemetry.

Clarifications, denials, errors, and timeouts are recorded without a verdict.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import Field, model_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.evaluation_profile import AgentMetric, EvaluationProfile
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
    SEMANTIC_CORRECTNESS = "semantic_correctness"
    RULES_ADHERENCE = "rules_adherence"
    SCHEMA_FAITHFULNESS = "schema_faithfulness"
    RELEVANCE = "relevance"
    COMPLETION = "completion"
    TOOL_SELECTION = "tool_selection"
    FAITHFULNESS = "faithfulness"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"


# All three diagnostic metrics are higher-is-better and are reported everywhere.
LOWER_IS_BETTER: frozenset[MetricName] = frozenset()
REPORTED_METRICS = tuple(MetricName)
NL2SQL_METRICS = (
    MetricName.SEMANTIC_CORRECTNESS,
    MetricName.RULES_ADHERENCE,
    MetricName.SCHEMA_FAITHFULNESS,
)
AGENT_REQUIRED_METRICS = (
    MetricName.RELEVANCE,
    MetricName.COMPLETION,
)
AGENT_METRICS = tuple(MetricName(metric.value) for metric in AgentMetric)


# Metric names written by earlier versions; migrated on read, never written again.
_LEGACY_METRIC_NAMES = (
    "relevancy",
    "accuracy",
    "correctness",
    "completeness",
    "task_completion",
    "groundedness",
    "hallucination_rate",
)


def metrics_for_profile(profile: EvaluationProfile) -> tuple[MetricName, ...]:
    return AGENT_METRICS if profile is EvaluationProfile.AGENT else NL2SQL_METRICS


class MetricStatus(StrEnum):
    SCORED = "scored"
    INCONCLUSIVE = "inconclusive"  # applies, but could not be decided
    NOT_APPLICABLE = "not_applicable"  # enabled, but required evidence/telemetry is absent


class MetricResult(StrictModel):
    status: MetricStatus
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: NonEmptyStr
    evidence: str | None = None  # exact quote from the response, if any
    # Defaults keep historical run artifacts readable. New judgements populate these so the
    # score drawer can show strengths and gaps separately instead of burying both in rationale.
    what_worked: list[NonEmptyStr] = Field(default_factory=list)
    what_didnt: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _score_iff_scored(self) -> Self:
        if (self.status is MetricStatus.SCORED) != (self.score is not None):
            raise ValueError("score is set exactly when status is scored")
        return self


class Metrics(StrictModel):
    semantic_correctness: MetricResult | None = None
    rules_adherence: MetricResult | None = None
    schema_faithfulness: MetricResult | None = None
    relevance: MetricResult | None = None
    completion: MetricResult | None = None
    tool_selection: MetricResult | None = None
    faithfulness: MetricResult | None = None
    context_precision: MetricResult | None = None
    context_recall: MetricResult | None = None

    @model_validator(mode="before")
    @classmethod
    def _read_legacy_metrics(cls, value: Any) -> Any:
        """Old run artifacts remain readable; new dumps use profile-specific names."""
        if isinstance(value, dict):
            data = dict(value)
            # Former VoC fields map into the general Agent evaluator. The old retrieval metric
            # combined operation/source selection, so tool_selection is the closest migration.
            if "answer_relevance" in data:
                data.setdefault("relevance", data.pop("answer_relevance"))
            if "answer_completeness" in data:
                data.setdefault("completion", data.pop("answer_completeness"))
            if "retrieval_faithfulness" in data:
                data.setdefault("tool_selection", data.pop("retrieval_faithfulness"))
            if any(name.value in data for name in AGENT_METRICS):
                return data
            data.setdefault(
                "semantic_correctness",
                data.get(
                    "accuracy",
                    data.get(
                        "correctness",
                        data.get("completeness", data.get("task_completion")),
                    ),
                ),
            )
            data.setdefault(
                "rules_adherence",
                {
                    "status": "not_applicable",
                    "score": None,
                    "rationale": "rules adherence was not separately evaluated in this run",
                },
            )
            data.setdefault(
                "schema_faithfulness",
                data.get(
                    "faithfulness",
                    {
                        "status": "not_applicable",
                        "score": None,
                        "rationale": "schema faithfulness was not computed by this run",
                    },
                ),
            )
            # The legacy values were migrated above; drop their old names so strict models
            # (extra="forbid") can still read the artifact.
            for legacy_name in _LEGACY_METRIC_NAMES:
                data.pop(legacy_name, None)
            return data
        return value

    @model_validator(mode="after")
    def _has_metrics(self) -> Self:
        if not self.by_name():
            raise ValueError("at least one evaluation metric is required")
        return self

    def by_name(self) -> dict[MetricName, MetricResult]:
        return {
            name: result for name in MetricName if (result := getattr(self, name.value)) is not None
        }


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
    # Stable links into the tagged success criteria and question-specific cheat sheet.
    related_tags: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _read_legacy_category(cls, value: Any) -> Any:
        if isinstance(value, dict):
            legacy = {
                "relevancy": "semantic_correctness",
                "accuracy": "semantic_correctness",
                "correctness": "semantic_correctness",
                "completeness": "semantic_correctness",
                "answer_relevance": "relevance",
                "answer_completeness": "completion",
                "retrieval_faithfulness": "tool_selection",
            }
            category = value.get("category")
            if category in legacy:
                return {**value, "category": legacy[category]}
        return value


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
    evaluation_profile: EvaluationProfile = EvaluationProfile.NL2SQL
    verdict: Verdict | None
    evaluation_status: EvaluationStatus
    metrics: Metrics | None = None
    criteria_results: list[CriterionResult] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    bug_candidates: list[BugCandidate] = Field(default_factory=list)
    llm_verdict: Verdict | None = None  # the judge's own opinion; never decisive
    notes: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_claims(cls, value: Any) -> Any:
        if isinstance(value, dict):
            data = {key: item for key, item in value.items() if key != "claims"}
            # Before Agent existed, a generic "faithfulness" finding meant NL2SQL schema
            # faithfulness. Preserve that migration only for NL2SQL artifacts; Agent
            # faithfulness is now a first-class metric and must keep its name.
            profile = EvaluationProfile(data.get("evaluation_profile", "nl2sql"))
            if profile is EvaluationProfile.NL2SQL:
                data["findings"] = [
                    {
                        **finding,
                        "category": "schema_faithfulness",
                    }
                    if isinstance(finding, dict) and finding.get("category") == "faithfulness"
                    else finding
                    for finding in data.get("findings", [])
                ]
            return data
        return value

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
        finding_ids = {f.finding_id for f in self.findings}
        for bug in self.bug_candidates:
            unknown = sorted(set(bug.finding_ids) - finding_ids)
            if unknown:
                raise ValueError(f"{bug.bug_id} references unknown findings {unknown}")
        if self.metrics is not None:
            actual = set(self.metrics.by_name())
            if self.evaluation_profile is EvaluationProfile.NL2SQL:
                valid = actual == set(NL2SQL_METRICS)
                requirement = f"exactly {sorted(m.value for m in NL2SQL_METRICS)}"
            else:
                valid = set(AGENT_REQUIRED_METRICS) <= actual <= set(AGENT_METRICS)
                requirement = (
                    f"required {sorted(m.value for m in AGENT_REQUIRED_METRICS)} plus any "
                    f"enabled subset of {sorted(m.value for m in AGENT_METRICS)}"
                )
            if not valid:
                raise ValueError(
                    f"{self.evaluation_profile} evaluations require {requirement}, got "
                    f"{sorted(m.value for m in actual)}"
                )
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
    what_worked: list[NonEmptyStr] = Field(
        default_factory=list,
        description="concise observable parts of the response that satisfied this metric",
    )
    what_didnt: list[NonEmptyStr] = Field(
        default_factory=list,
        description="concise observable gaps that prevented a perfect score",
    )

    @model_validator(mode="after")
    def _three_levels(self) -> Self:
        if self.score is not None and self.score not in JUDGED_SCORES:
            raise ValueError(f"score must be 0, 0.5 or 1 (or null), not {self.score}")
        return self


class FindingDraft(StrictModel):
    category: NonEmptyStr = Field(description="the metric this finding explains")
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
    related_tags: list[NonEmptyStr] = Field(
        default_factory=list,
        description="exact tags from the traceability catalog that this finding directly affects",
    )

    @model_validator(mode="before")
    @classmethod
    def _read_legacy_category(cls, value: Any) -> Any:
        if isinstance(value, dict):
            legacy = {
                "relevancy": "semantic_correctness",
                "accuracy": "semantic_correctness",
                "correctness": "semantic_correctness",
                "completeness": "semantic_correctness",
            }
            category = value.get("category")
            if category in legacy:
                return {**value, "category": legacy[category]}
        return value


class CriterionDraft(StrictModel):
    criterion: NonEmptyStr
    status: CriterionStatus
    rationale: NonEmptyStr
    evidence: str | None = Field(default=None, description="exact quote from the response, if any")


class EvaluationDraft(StrictModel):
    """The judge's answer for one answered response; code turns it into metric scores."""

    semantic_correctness: JudgedScore
    rules_adherence: JudgedScore
    schema_faithfulness: JudgedScore
    criteria: list[CriterionDraft] = Field(default_factory=list)
    findings: list[FindingDraft] = Field(default_factory=list)
    overall_verdict: Verdict = Field(
        description="your own overall opinion; recorded as llm_verdict, not decisive"
    )

    @model_validator(mode="before")
    @classmethod
    def _read_legacy_metrics(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "semantic_correctness" not in data:
            data["semantic_correctness"] = data.get(
                "accuracy",
                data.get(
                    "correctness",
                    data.get("completeness", data.get("relevancy")),
                ),
            )
        data.setdefault(
            "rules_adherence",
            {
                "score": None,
                "rationale": "rules adherence was not separately evaluated",
                "what_worked": [],
                "what_didnt": [],
            },
        )
        if "schema_faithfulness" not in data:
            data["schema_faithfulness"] = data.get(
                "faithfulness",
                {
                    "score": None,
                    "rationale": "schema faithfulness was not evaluated",
                    "what_worked": [],
                    "what_didnt": [],
                },
            )
        for legacy_name in (
            "relevancy",
            "accuracy",
            "correctness",
            "completeness",
            "faithfulness",
        ):
            data.pop(legacy_name, None)
        return data


class AgentEvaluationDraft(StrictModel):
    """The judge's answer for one Agent response.

    Optional fields are required only when their metric is both enabled and observable for the
    current response. Code injects not-applicable results without asking the judge to guess.
    """

    relevance: JudgedScore
    completion: JudgedScore
    tool_selection: JudgedScore | None = None
    faithfulness: JudgedScore | None = None
    context_precision: JudgedScore | None = None
    context_recall: JudgedScore | None = None
    criteria: list[CriterionDraft] = Field(default_factory=list)
    findings: list[FindingDraft] = Field(default_factory=list)
    overall_verdict: Verdict = Field(
        description="your own overall opinion; recorded as llm_verdict, not decisive"
    )
