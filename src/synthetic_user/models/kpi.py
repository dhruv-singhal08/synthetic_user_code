"""Run-level KPIs for product managers (kpis.json / kpis.csv), computed by code from the
stored evaluations - never by an LLM."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.evaluation import MetricName
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


class RunKpis(StrictModel):
    run_id: NonEmptyStr
    product_id: NonEmptyStr
    mode: GenerationMode
    computed_at: datetime
    questions_asked: int = Field(ge=0)
    outcomes: dict[NonEmptyStr, int]  # answered / clarification / denied / error / timeout
    judged: int = Field(ge=0)  # answered responses with a verdict
    verdicts: dict[NonEmptyStr, int]  # pass / fail / inconclusive
    task_success_rate: float | None  # pass / judged
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
