"""Run-level KPIs for product managers (kpis.json / kpis.csv), computed by code from the
stored evaluations - never by an LLM."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, model_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.evaluation import MetricName, Severity
from synthetic_user.models.run import GenerationMode


class MetricKpi(StrictModel):
    metric: MetricName
    mean: float | None  # mean score over scored answers; None when nothing was scored
    pass_rate: float | None  # share scoring best (1, or 0 for lower-is-better metrics)
    n: int = Field(ge=0)  # answers scored
    not_applicable: int = Field(ge=0)
    inconclusive: int = Field(ge=0)
    lower_is_better: bool


class GroupKpi(StrictModel):
    """One breakdown row (e.g. task_type = ranking)."""

    group: NonEmptyStr
    questions: int = Field(ge=0)
    judged: int = Field(ge=0)
    pass_rate: float | None
    metric_means: dict[MetricName, float | None]


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
    examples: list[NonEmptyStr] = Field(min_length=1, max_length=3)
    known_gap: bool = False


class IssueSummary(StrictModel):
    total_findings: int = Field(ge=0)
    affected_questions: int = Field(ge=0)
    issues: list[ConsolidatedIssue] = Field(default_factory=list)


class RunKpis(StrictModel):
    run_id: NonEmptyStr
    product_id: NonEmptyStr
    mode: GenerationMode
    computed_at: datetime
    questions_asked: int = Field(ge=0)
    outcomes: dict[NonEmptyStr, int]  # answered / clarification / denied / error / timeout
    judged: int = Field(ge=0)  # answered responses with a verdict
    verdicts: dict[NonEmptyStr, int]  # pass / fail / inconclusive
    accuracy: float | None = None  # pass verdicts / judged answers; the final derived metric
    task_success_rate: float | None = None  # legacy alias for accuracy
    metrics: list[MetricKpi]
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
        if not isinstance(value, dict):
            return value
        legacy_names = {
            "accuracy": "semantic_correctness",
            "task_completion": "semantic_correctness",
            "completeness": "semantic_correctness",
            "correctness": "semantic_correctness",
        }
        current = {name.value for name in MetricName}
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
            if name in current:
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
                    if names.get(name, name) in current
                }
                groups.append(item)
            data[group_name] = groups
        return data
