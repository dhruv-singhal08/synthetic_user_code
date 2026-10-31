"""Shared detection for NL2SQL responses that did not produce SQL."""

from __future__ import annotations

from collections.abc import Sequence

from synthetic_user.models.evaluation import EvaluationResult
from synthetic_user.models.evaluation_profile import EvaluationProfile
from synthetic_user.models.execution import (
    AnsweredResponse,
    ExecutionRecord,
    ProductOutcome,
)
from synthetic_user.models.product import ProductDescription

NO_SQL_REASON = "No SQL generated."


def product_expects_sql(product: ProductDescription) -> bool:
    """Return whether SQL is an expected output of this product."""

    values = [
        product.product_id,
        product.name,
        product.description,
        *product.expected_outputs,
    ]
    return any("sql" in value.casefold() for value in values)


def run_expects_sql(
    executions: Sequence[ExecutionRecord],
    evaluations: Sequence[EvaluationResult] = (),
) -> bool:
    """Infer SQL expectation when rendering a stored run without its product model.

    The evaluation profile is authoritative. A product id can still contain ``nl2sql`` when
    that product is deliberately tested as an Agent; missing SQL is not a defect in that run.
    """

    profiles = {evaluation.evaluation_profile for evaluation in evaluations}
    if profiles == {EvaluationProfile.AGENT}:
        return False
    if EvaluationProfile.NL2SQL in profiles:
        return True

    return any("nl2sql" in execution.product_id.casefold() for execution in executions) or any(
        _has_sql_artifact(execution.response)
        for execution in executions
        if isinstance(execution.response, AnsweredResponse)
    )


def no_sql_execution(execution: ExecutionRecord, *, sql_expected: bool) -> bool:
    """Identify an NL2SQL timeout or answered response with no SQL artifact."""

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
