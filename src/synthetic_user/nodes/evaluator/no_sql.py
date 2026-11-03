"""Shared detection of executions that should have produced SQL but did not.

"SQL is expected" means the run selected a metric that needs SQL evidence (frozen metric
config) - never a keyword found in the product description or id. Runs from before the metric
registry fall back to their recorded profile label (``nl2sql``).
"""

from __future__ import annotations

from collections.abc import Sequence

from synthetic_user.models.evaluation import EvaluationResult
from synthetic_user.models.execution import AnsweredResponse, ExecutionRecord, ProductOutcome
from synthetic_user.models.metrics import EvidenceKind, ResolvedMetricConfig

NO_SQL_REASON = "No SQL generated."


def run_expects_sql(
    config: ResolvedMetricConfig | None, evaluations: Sequence[EvaluationResult] = ()
) -> bool:
    if config is not None:
        return config.needs_evidence(EvidenceKind.SQL)
    return any(e.legacy_evaluation_profile == "nl2sql" for e in evaluations)


def no_sql_execution(execution: ExecutionRecord, *, sql_expected: bool) -> bool:
    """A timeout, or an answered response with no SQL artifact, when SQL is expected."""
    if not sql_expected:
        return False
    if execution.outcome is ProductOutcome.TIMEOUT:
        return True
    return isinstance(execution.response, AnsweredResponse) and not _has_sql_artifact(
        execution.response
    )


def _has_sql_artifact(response: AnsweredResponse) -> bool:
    return any(
        artifact.kind.casefold() == "sql"
        and (not isinstance(artifact.content, str) or bool(artifact.content.strip()))
        for artifact in response.artifacts
    )
