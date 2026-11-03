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

The General Judge's independent goal-completion assessment (D243) enters as a finding with
category ``goal_completion`` (no metric score is changed): an unjustified decline or a goal not
completed is critical, a partially completed goal high, so a supported request the product
refused without justification never passes, while a justified decline of an unsupported request
can. A finding exempted by an approved exception of the evaluation guidance (``exempted_by``)
never counts: it is listed in the reason as an applied exception.

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
    GoalCompletion,
    GoalStatus,
    MetricResult,
    MetricStatus,
    Severity,
    Verdict,
)
from synthetic_user.models.metrics import GOAL_COMPLETION_CATEGORY, MetricDefinition

_UNDECIDED = frozenset({MetricStatus.INCONCLUSIVE, MetricStatus.EVALUATION_ERROR})
SEVERITY_RANK = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
# Reason texts (shown in the CSV / UI next to the verdict).
FAILED = "Failed"
PASSED = "Passed: every scored metric met its expectations"
PASSED_WITH_ISSUES = "Passed with minor issues"
INCONCLUSIVE = "Inconclusive"
NOTHING_SCORED = "no selected metric could be scored"
GOAL_LABEL = "Goal completion"
EXCEPTIONS_APPLIED = "Exceptions applied (evaluation guidance)"
JUSTIFIED_DECLINE = "Justified decline"


@dataclass(frozen=True)
class VerdictDecision:
    verdict: Verdict
    reason: str
    # FAIL only: the ids of the proven findings that decided it (the only bug sources, D244).
    failing: tuple[str, ...] = ()
    # Every confirmed problem: supported, settled, not exempted, in a metric scored below its
    # best (or the goal check) - the only findings that may support a bug (D244).
    confirmed: tuple[str, ...] = ()


class VerdictPolicy(Protocol):
    name: str

    def decide(
        self,
        metrics: Mapping[str, MetricResult],
        findings: Sequence[Finding],
        definitions: Mapping[str, MetricDefinition],
        goal: GoalCompletion | None = None,
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
        goal: GoalCompletion | None = None,
    ) -> VerdictDecision:
        decision = self._decide(metrics, findings, definitions)
        notes = []
        if goal is not None and goal.status is GoalStatus.JUSTIFIED_DECLINE:
            notes.append(f"{JUSTIFIED_DECLINE}: {goal.reason.rstrip('.')}")
        if exempted := [f for f in findings if f.exempted_by is not None]:
            notes.append(
                _reason(
                    EXCEPTIONS_APPLIED,
                    [
                        f"{(f.exempted_by or '').rstrip('.')} - {f.description.rstrip('.')}"
                        for f in exempted
                    ],
                )
            )
        if not notes:
            return decision
        return VerdictDecision(
            decision.verdict,
            ". ".join([decision.reason, *notes]),
            decision.failing,
            decision.confirmed,
        )

    def _decide(
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
        # Metric findings and the goal finding; an exempted difference is never a problem.
        relevant = [
            f
            for f in findings
            if (f.category in considered or f.category == GOAL_COMPLETION_CATEGORY)
            and f.exempted_by is None
        ]
        settled = [f for f in relevant if not f.unresolved]
        serious = [
            f
            for f in settled
            if SEVERITY_RANK[f.severity] >= self._fail_rank
            and (f.category in below or f.category == GOAL_COMPLETION_CATEGORY)
        ]

        def label(metric_id: str) -> str:
            if metric_id == GOAL_COMPLETION_CATEGORY:
                return GOAL_LABEL
            definition = definitions.get(metric_id)
            return definition.display_name if definition is not None else metric_id

        def item(f: Finding) -> str:
            return f"[{f.severity.value}] {label(f.category)}: {f.description.rstrip('.')}"

        proven = [f for f in serious if f.evidence_status is EvidenceStatus.SUPPORTED]
        if proven:
            confirmed = tuple(
                f.finding_id
                for f in settled
                if f.evidence_status is EvidenceStatus.SUPPORTED
                and (f.category in below or f.category == GOAL_COMPLETION_CATEGORY)
            )
            return VerdictDecision(
                Verdict.FAIL,
                _reason(FAILED, [item(f) for f in proven]),
                tuple(f.finding_id for f in proven),
                confirmed,
            )

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
