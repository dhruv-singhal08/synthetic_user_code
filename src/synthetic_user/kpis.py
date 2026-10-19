"""Compute run KPIs from stored artifacts (pure code, no LLM) and render them as CSV.

Definitions (docs/DESIGN_V3_REFERENCE_AND_EVALUATION.md section 11):
- metric mean / pass rate over answers where the metric was scored; not_applicable and
  inconclusive answers are counted separately, never averaged in;
- pass rate = share scoring 1 (hallucination_rate: share scoring 0);
- task success rate = pass verdicts / judged answers;
- availability = 1 - (error + timeout) / questions asked;
- latency p50 / p95 (nearest rank) over executions with a measured latency;
- a bug is a known-gap bug when every one of its findings matches a known gap, else new;
  its severity is its most severe finding;
- inconclusive rate = inconclusive verdicts / judged answers.
"""

from __future__ import annotations

import csv
import io
import math
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from synthetic_user.models.evaluation import (
    LOWER_IS_BETTER,
    REPORTED_METRICS,
    EvaluationResult,
    MetricName,
    MetricStatus,
    Severity,
    Verdict,
)
from synthetic_user.models.execution import ExecutionRecord, ProductOutcome
from synthetic_user.models.kpi import GroupKpi, MetricKpi, RunKpis
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario

_SEVERITY_RANK = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
METRIC_LABELS = {
    MetricName.RELEVANCY: "Answer relevancy",
    MetricName.CORRECTNESS: "Correctness",
    MetricName.GROUNDEDNESS: "Groundedness",
    MetricName.HALLUCINATION_RATE: "Hallucination rate",
    MetricName.COMPLETENESS: "Completeness",
}


def compute_kpis(
    *,
    run_id: str,
    product_id: str,
    mode: GenerationMode,
    scenarios: Sequence[Scenario],
    executions: Sequence[ExecutionRecord],
    evaluations: Sequence[EvaluationResult],
    now: datetime | None = None,
) -> RunKpis:
    outcomes = Counter(e.outcome.value for e in executions)
    judged = [e for e in evaluations if e.verdict is not None]
    verdicts = Counter(e.verdict.value for e in judged if e.verdict is not None)
    asked = len(executions)
    failed_calls = outcomes[ProductOutcome.ERROR.value] + outcomes[ProductOutcome.TIMEOUT.value]
    latencies = sorted(e.latency_ms for e in executions if e.latency_ms is not None)

    new_bugs: Counter[str] = Counter()
    known = 0
    for ev in evaluations:
        by_id = {f.finding_id: f for f in ev.findings}
        for bug in ev.bug_candidates:
            findings = [by_id[i] for i in bug.finding_ids if i in by_id]
            if findings and all(f.matches_known_gap for f in findings):
                known += 1
                continue
            worst = max(findings, key=lambda f: _SEVERITY_RANK[f.severity])
            new_bugs[worst.severity.value] += 1

    by_scenario = {s.id: s for s in scenarios}
    return RunKpis(
        run_id=run_id,
        product_id=product_id,
        mode=mode,
        computed_at=now or datetime.now(UTC),
        questions_asked=asked,
        outcomes={o.value: outcomes.get(o.value, 0) for o in ProductOutcome},
        judged=len(judged),
        verdicts={v.value: verdicts.get(v.value, 0) for v in Verdict},
        task_success_rate=_ratio(verdicts[Verdict.PASS.value], len(judged)),
        metrics=[_metric_kpi(name, judged) for name in REPORTED_METRICS],
        availability=_round(1 - failed_calls / asked) if asked else None,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        new_bugs=sum(new_bugs.values()),
        new_bugs_by_severity={s.value: new_bugs.get(s.value, 0) for s in Severity},
        known_gap_bugs=known,
        inconclusive_rate=_ratio(verdicts[Verdict.INCONCLUSIVE.value], len(judged)),
        by_task_type=_breakdown(
            evaluations, lambda ev: _scenario_attr(by_scenario, ev, "task_type")
        ),
        by_capability=_breakdown(
            evaluations, lambda ev: _scenario_attr(by_scenario, ev, "capability")
        ),
    )


def kpis_csv(kpis: RunKpis) -> str:
    """One row per KPI, for spreadsheets: kpi, value, n, excluded, note."""
    rows: list[list[str]] = [["kpi", "value", "n", "excluded", "note"]]

    def row(
        kpi: str, value: float | None, n: int | str = "", excluded: str = "", note: str = ""
    ) -> None:
        rows.append([kpi, "" if value is None else f"{value:g}", str(n), excluded, note])

    row(
        "Task success rate",
        kpis.task_success_rate,
        kpis.judged,
        note="pass verdicts / judged answers",
    )
    for m in kpis.metrics:
        excluded = f"{m.not_applicable} not applicable, {m.inconclusive} inconclusive"
        direction = "lower is better" if m.lower_is_better else "higher is better"
        row(f"{METRIC_LABELS[m.metric]} (mean)", m.mean, m.n, excluded, direction)
        best = "share scoring 0" if m.lower_is_better else "share scoring 1"
        row(f"{METRIC_LABELS[m.metric]} (pass rate)", m.pass_rate, m.n, excluded, best)
    row(
        "Availability",
        kpis.availability,
        kpis.questions_asked,
        note="1 - (error + timeout) / asked",
    )
    row("Latency p50 (ms)", kpis.latency_p50_ms)
    row("Latency p95 (ms)", kpis.latency_p95_ms)
    severities = ", ".join(f"{k} {v}" for k, v in kpis.new_bugs_by_severity.items() if v)
    row("New bugs", kpis.new_bugs, note=severities or "none")
    row("Known-gap bugs", kpis.known_gap_bugs, note="every finding matches a known gap")
    row("Inconclusive rate", kpis.inconclusive_rate, kpis.judged)
    not_judged = ", ".join(
        f"{k} {v}" for k, v in kpis.outcomes.items() if k != ProductOutcome.ANSWERED.value and v
    )
    row("Questions asked", kpis.questions_asked, note=f"recorded only: {not_judged or 'none'}")
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    return buffer.getvalue()


def _metric_kpi(name: MetricName, judged: Sequence[EvaluationResult]) -> MetricKpi:
    results = [ev.metrics.by_name()[name] for ev in judged if ev.metrics is not None]
    scores = [r.score for r in results if r.status is MetricStatus.SCORED and r.score is not None]
    lower = name in LOWER_IS_BETTER
    best = 0.0 if lower else 1.0
    return MetricKpi(
        metric=name,
        mean=_round(sum(scores) / len(scores)) if scores else None,
        pass_rate=_ratio(sum(1 for s in scores if s == best), len(scores)),
        n=len(scores),
        not_applicable=sum(1 for r in results if r.status is MetricStatus.NOT_APPLICABLE),
        inconclusive=sum(1 for r in results if r.status is MetricStatus.INCONCLUSIVE),
        lower_is_better=lower,
    )


def _breakdown(
    evaluations: Sequence[EvaluationResult], key: Callable[[EvaluationResult], str]
) -> list[GroupKpi]:
    groups: dict[str, list[EvaluationResult]] = {}
    for ev in evaluations:
        groups.setdefault(key(ev), []).append(ev)
    out = []
    for group, items in sorted(groups.items()):
        judged = [e for e in items if e.verdict is not None]
        passed = sum(1 for e in judged if e.verdict is Verdict.PASS)
        out.append(
            GroupKpi(
                group=group,
                questions=len(items),
                judged=len(judged),
                pass_rate=_ratio(passed, len(judged)),
                metric_means={m: _metric_kpi(m, judged).mean for m in REPORTED_METRICS},
            )
        )
    return out


def _scenario_attr(by_scenario: dict[str, Scenario], ev: EvaluationResult, attr: str) -> str:
    scenario = by_scenario.get(ev.scenario_id)
    return str(getattr(scenario, attr).value) if scenario is not None else "unknown"


def _percentile(values: Sequence[float], pct: int) -> float | None:
    if not values:
        return None
    rank = max(1, math.ceil(pct / 100 * len(values)))
    return _round(values[rank - 1])


def _ratio(part: int, whole: int) -> float | None:
    return _round(part / whole) if whole else None


def _round(value: float) -> float:
    return round(value, 3)
