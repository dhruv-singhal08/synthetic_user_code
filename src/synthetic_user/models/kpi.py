"""Run-level KPIs for product managers (kpis.json / kpis.csv), computed by code from the
stored evaluations - never by an LLM."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, model_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.evaluation import Severity
from synthetic_user.models.run import GenerationMode

# Example finding descriptions kept per consolidated issue.
MAX_ISSUE_EXAMPLES = 3


class MetricKpi(StrictModel):
    metric: NonEmptyStr  # metric id (original id for legacy runs: never relabelled)
    display_name: NonEmptyStr | None = None
    category: NonEmptyStr | None = None
    origin: NonEmptyStr | None = None  # builtin | custom | legacy
    mean: float | None  # mean score over scored answers; None when nothing was scored
    pass_rate: float | None  # share scoring best (1, or 0 for lower-is-better metrics)
    n: int = Field(ge=0)  # answers scored (the denominator of mean / pass rate)
    not_applicable: int = Field(ge=0)
    inconclusive: int = Field(ge=0)
    cannot_generate: int = Field(default=0, ge=0)
    evidence_unavailable: int = Field(default=0, ge=0)
    evaluation_error: int = Field(default=0, ge=0)
    unavailable: int = Field(default=0, ge=0)  # custom metric that could not be compiled
    lower_is_better: bool
    # Most frequent reasons behind non-scored statuses ("status: reason" -> count).
    reasons: dict[NonEmptyStr, int] = Field(default_factory=dict)
    notes: list[NonEmptyStr] = Field(default_factory=list)


class ReciprocalRankRow(StrictModel):
    question_id: NonEmptyStr
    reciprocal_rank: float | None = Field(ge=0.0, le=1.0)
    first_relevant_rank: int | None = Field(default=None, ge=1)
    status: NonEmptyStr  # scored, or why the question is excluded from the denominator


class MrrKpi(StrictModel):
    """Run-level MRR = mean reciprocal rank over questions with ranked retrieval."""

    value: float | None
    denominator: int = Field(ge=0)  # questions with ranked chunks + relevance labels
    excluded_missing_rank: int = Field(ge=0)
    excluded_other: int = Field(ge=0)
    per_question: list[ReciprocalRankRow] = Field(default_factory=list)
    definition: NonEmptyStr


class CompilerFailure(StrictModel):
    metric_id: NonEmptyStr
    source_text: NonEmptyStr
    provenance: NonEmptyStr
    reason: NonEmptyStr


class GroupKpi(StrictModel):
    """One breakdown row (e.g. task_type = ranking)."""

    group: NonEmptyStr
    questions: int = Field(ge=0)
    judged: int = Field(ge=0)
    pass_rate: float | None
    metric_means: dict[NonEmptyStr, float | None]


class ConsolidatedIssue(StrictModel):
    """One standardized, actionable issue theme across question-level findings."""

    issue_id: NonEmptyStr = Field(pattern=r"^ISSUE_\d{3,}$")
    theme: NonEmptyStr = Field(pattern=r"^[a-z][a-z0-9_]+$")
    categories: list[NonEmptyStr] = Field(min_length=1)
    title: NonEmptyStr
    recommended_action: NonEmptyStr
    severity: Severity
    occurrences: int = Field(ge=1)
    affected_question_ids: list[NonEmptyStr] = Field(min_length=1)
    # A timeout has no judge finding, but still belongs in the consolidated no-SQL issue.
    finding_ids: list[NonEmptyStr] = Field(default_factory=list)
    examples: list[NonEmptyStr] = Field(min_length=1, max_length=MAX_ISSUE_EXAMPLES)
    known_gap: bool = False


class IssueSummary(StrictModel):
    total_findings: int = Field(ge=0)
    affected_questions: int = Field(ge=0)
    issues: list[ConsolidatedIssue] = Field(default_factory=list)


# Very old KPI names that have no counterpart (they were hidden or removed, D206 / D207).
_DROPPED_METRICS = frozenset({"groundedness", "hallucination_rate", "relevancy"})


class RunKpis(StrictModel):
    run_id: NonEmptyStr
    product_id: NonEmptyStr
    mode: GenerationMode
    computed_at: datetime
    questions_asked: int = Field(ge=0)
    outcomes: dict[NonEmptyStr, int]  # answered / clarification / denied / error / timeout
    # answered / unable_to_answer / clarification / unknown (+ "none" for error / timeout)
    response_classifications: dict[NonEmptyStr, int] = Field(default_factory=dict)
    judged: int = Field(ge=0)  # evaluable responses (answers + substantive denials) with a verdict
    verdicts: dict[NonEmptyStr, int]  # pass / fail / inconclusive
    accuracy: float | None = None  # pass verdicts / judged responses (provisional verdict policy)
    # Kept for older readers: the SAME value as accuracy, not a separate measurement.
    task_success_rate: float | None = None
    metric_config_id: NonEmptyStr | None = None
    verdict_policy: NonEmptyStr | None = None
    selected_metrics: list[NonEmptyStr] = Field(default_factory=list)
    # category -> metric ids (descriptive grouping only; no composite category score).
    category_groups: dict[NonEmptyStr, list[NonEmptyStr]] = Field(default_factory=dict)
    compiler_failures: list[CompilerFailure] = Field(default_factory=list)
    metrics: list[MetricKpi]
    mrr: MrrKpi | None = None
    availability: float | None  # 1 - (error + timeout) / asked
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    new_bugs: int = Field(ge=0)  # bug candidates not explained by a known gap
    new_bugs_by_severity: dict[NonEmptyStr, int]
    known_gap_bugs: int = Field(ge=0)  # every finding of the bug matches a known gap
    inconclusive_rate: float | None  # inconclusive verdicts / judged
    by_task_type: list[GroupKpi] = Field(default_factory=list)
    by_capability: list[GroupKpi] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _read_legacy_metrics(cls, value: Any) -> Any:
        """Translate earlier metric names when opening old run artifacts."""
        if not isinstance(value, dict) or value.get("selected_metrics"):
            return value  # current KPIs: metric ids are already the frozen config's ids
        legacy_names = {
            "accuracy": "semantic_correctness",
            "task_completion": "semantic_correctness",
            "completeness": "semantic_correctness",
            "correctness": "semantic_correctness",
        }
        data = dict(value)
        raw_metric_names = {
            item.get("metric") if isinstance(item, dict) else getattr(item, "metric", None)
            for item in data.get("metrics", [])
        }
        is_agent = bool(
            raw_metric_names
            & {"relevance", "completion", "tool_selection", "context_precision", "context_recall"}
        )
        names = {
            **legacy_names,
            **({} if is_agent else {"faithfulness": "schema_faithfulness"}),
        }
        if data.get("accuracy") is None:
            data["accuracy"] = data.get("task_success_rate")
        if data.get("task_success_rate") is None:
            data["task_success_rate"] = data.get("accuracy")
        metrics = []
        for metric in data.get("metrics", []):
            item = dict(metric)
            raw_name = item.get("metric")
            name = names.get(raw_name, raw_name) if isinstance(raw_name, str) else raw_name
            if name in _DROPPED_METRICS:
                continue
            item["metric"] = name
            metrics.append(item)
        data["metrics"] = metrics
        for group_name in ("by_task_type", "by_capability"):
            groups = []
            for group in data.get(group_name, []):
                item = dict(group)
                means = item.get("metric_means", {})
                item["metric_means"] = {
                    names.get(name, name): score
                    for name, score in means.items()
                    if names.get(name, name) not in _DROPPED_METRICS
                }
                groups.append(item)
            data[group_name] = groups
        return data
