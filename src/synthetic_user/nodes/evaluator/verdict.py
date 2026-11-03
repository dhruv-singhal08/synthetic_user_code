"""Final verdict: a small, replaceable, deterministic policy. PROVISIONAL (instruction section 10).

The final aggregation policy (weights, advisory metrics, thresholds) is deliberately deferred.
``CompatibilityVerdictPolicy`` keeps the previous ``metrics_verdict`` semantics as a temporary
default over the run's selected metrics:

- a scored metric below its best value with a SUPPORTED finding in that metric -> FAIL;
- otherwise any sub-perfect, inconclusive or evaluation-error metric -> INCONCLUSIVE;
- no usable score at all (everything not applicable / missing) -> INCONCLUSIVE, never PASS;
- otherwise PASS.

not_applicable, cannot_generate, evidence_unavailable and unavailable results are never
counted as passing evidence. Metrics with ``counts_toward_verdict = false`` (MRR, a run-level
ranking measure) are reported but not read. The LLM's overall verdict is never decisive.
Granular scores, statuses and evidence are stored, so a later policy can recompute verdicts
without regenerating questions or re-running judges.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from synthetic_user.models.evaluation import (
    EvidenceStatus,
    Finding,
    MetricResult,
    MetricStatus,
    Verdict,
)
from synthetic_user.models.metrics import MetricDefinition

_UNDECIDED = frozenset({MetricStatus.INCONCLUSIVE, MetricStatus.EVALUATION_ERROR})


class VerdictPolicy(Protocol):
    name: str

    def decide(
        self,
        metrics: Mapping[str, MetricResult],
        findings: Sequence[Finding],
        definitions: Mapping[str, MetricDefinition],
    ) -> Verdict: ...


class CompatibilityVerdictPolicy:
    name = "compatibility-v1 (provisional)"

    def decide(
        self,
        metrics: Mapping[str, MetricResult],
        findings: Sequence[Finding],
        definitions: Mapping[str, MetricDefinition],
    ) -> Verdict:
        considered = {
            metric_id: result
            for metric_id, result in metrics.items()
            if (definition := definitions.get(metric_id)) is None
            or definition.counts_toward_verdict
        }
        supported = {f.category for f in findings if f.evidence_status is EvidenceStatus.SUPPORTED}
        failing = [
            metric_id
            for metric_id, result in considered.items()
            if result.status is MetricStatus.SCORED
            and result.score is not None
            and result.score != _best(definitions.get(metric_id))
        ]
        if any(metric_id in supported for metric_id in failing):
            return Verdict.FAIL
        undecided = any(result.status in _UNDECIDED for result in considered.values())
        scored = any(result.status is MetricStatus.SCORED for result in considered.values())
        if failing or undecided or not scored:
            return Verdict.INCONCLUSIVE
        return Verdict.PASS


def _best(definition: MetricDefinition | None) -> float:
    if definition is None or definition.higher_is_better:
        return 1.0
    return 0.0


DEFAULT_VERDICT_POLICY: VerdictPolicy = CompatibilityVerdictPolicy()
