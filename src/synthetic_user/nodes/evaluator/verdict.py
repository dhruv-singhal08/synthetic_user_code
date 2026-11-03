"""Final verdict: a small, replaceable, deterministic policy decided from the judges' findings
and their severity (D241), always with a reason.

The judges score each metric and report findings (category = metric, severity, evidence
status, quote). ``SeverityVerdictPolicy`` then decides, over the run's selected metrics that
count toward the verdict (MRR, a run-level ranking measure, does not):

- FAIL: a finding at or above ``evaluation.fail_severity`` (default high) with SUPPORTED
  evidence, in a metric scored below its best value;
- INCONCLUSIVE: nothing proven fails, but something cannot be settled - an ``unresolved``
  finding (D232), a metric that is inconclusive or hit an evaluation error, a fail-level
  finding without supported evidence, a metric scored at its worst value with no fail-level
  finding (the judgement is inconsistent), or no metric could be scored at all;
- PASS otherwise: lower-severity issues (a 0.5 with a medium finding, a cosmetic problem) do
  not fail the response; they are listed in the reason.

not_applicable, cannot_generate, evidence_unavailable and unavailable results are never counted
as passing evidence. The judge's own overall verdict is stored, never decisive. Every score,
status and finding is stored, so a later policy can recompute verdicts without regenerating
questions or re-running judges.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from synthetic_user.models.evaluation import (
    EvidenceStatus,
    Finding,
    MetricResult,
    MetricStatus,
    Severity,
    Verdict,
)
from synthetic_user.models.metrics import MetricDefinition

_UNDECIDED = frozenset({MetricStatus.INCONCLUSIVE, MetricStatus.EVALUATION_ERROR})
SEVERITY_RANK = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
# Reason texts (shown in the CSV / UI next to the verdict).
FAILED = "Failed"
PASSED = "Passed: every scored metric met its expectations"
PASSED_WITH_ISSUES = "Passed with minor issues"
INCONCLUSIVE = "Inconclusive"
NOTHING_SCORED = "no selected metric could be scored"


@dataclass(frozen=True)
class VerdictDecision:
    verdict: Verdict
    reason: str


class VerdictPolicy(Protocol):
    name: str

    def decide(
        self,
        metrics: Mapping[str, MetricResult],
        findings: Sequence[Finding],
        definitions: Mapping[str, MetricDefinition],
    ) -> VerdictDecision: ...


class SeverityVerdictPolicy:
    def __init__(self, fail_severity: Severity = Severity.HIGH) -> None:
        self._fail_rank = SEVERITY_RANK[fail_severity]
        self.name = f"severity-v1 (fail at {fail_severity.value}+)"

    def decide(
        self,
        metrics: Mapping[str, MetricResult],
        findings: Sequence[Finding],
        definitions: Mapping[str, MetricDefinition],
    ) -> VerdictDecision:
        considered = {
            metric_id: result
            for metric_id, result in metrics.items()
            if (definition := definitions.get(metric_id)) is None
            or definition.counts_toward_verdict
        }
        below = {
            metric_id
            for metric_id, result in considered.items()
            if result.status is MetricStatus.SCORED
            and result.score is not None
            and result.score != _best(definitions.get(metric_id))
        }
        relevant = [f for f in findings if f.category in considered]
        settled = [f for f in relevant if not f.unresolved]
        serious = [
            f
            for f in settled
            if SEVERITY_RANK[f.severity] >= self._fail_rank and f.category in below
        ]

        def label(metric_id: str) -> str:
            definition = definitions.get(metric_id)
            return definition.display_name if definition is not None else metric_id

        def item(f: Finding) -> str:
            return f"[{f.severity.value}] {label(f.category)}: {f.description.rstrip('.')}"

        proven = [f for f in serious if f.evidence_status is EvidenceStatus.SUPPORTED]
        if proven:
            return VerdictDecision(Verdict.FAIL, _reason(FAILED, [item(f) for f in proven]))

        open_points = [f"unresolved - {item(f)}" for f in relevant if f.unresolved]
        open_points += [
            f"{label(m)} {result.status.value}: {result.rationale}"
            for m, result in considered.items()
            if result.status in _UNDECIDED
        ]
        open_points += [
            f"not proven ({f.evidence_status.value} evidence) - {item(f)}" for f in serious
        ]
        open_points += [
            f"{label(m)} scored {considered[m].score:g} but no finding explains a serious problem"
            for m in sorted(below)
            if considered[m].score == _worst(definitions.get(m))
            and not any(f.category == m for f in serious)
        ]
        if not any(r.status is MetricStatus.SCORED for r in considered.values()):
            open_points.append(NOTHING_SCORED)
        if open_points:
            return VerdictDecision(Verdict.INCONCLUSIVE, _reason(INCONCLUSIVE, open_points))
        issues = [item(f) for f in settled] + [
            f"{label(m)} scored {considered[m].score:g}"
            for m in sorted(below)
            if not any(f.category == m for f in settled)
        ]
        if issues:
            return VerdictDecision(Verdict.PASS, _reason(PASSED_WITH_ISSUES, issues))
        return VerdictDecision(Verdict.PASS, PASSED)


def _reason(headline: str, items: Sequence[str]) -> str:
    return f"{headline}: " + "; ".join(dict.fromkeys(items))


def _best(definition: MetricDefinition | None) -> float:
    if definition is None or definition.higher_is_better:
        return 1.0
    return 0.0


def _worst(definition: MetricDefinition | None) -> float:
    return 1.0 - _best(definition)
