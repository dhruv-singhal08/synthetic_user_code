from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from synthetic_user.models.execution import (
    EXECUTION_RESULT_ADAPTER,
    ClarificationResult,
    DeniedResult,
    ErrorResult,
    ProductOutcome,
    SqlGeneratedResult,
    TimeoutResult,
)


def _common() -> dict[str, Any]:
    return {
        "execution_id": "EXE_001",
        "question_id": "Q_001",
        "product_id": "sample_nl2sql",
        "submitted_question": "Top merchants by spend?",
        "started_at": datetime(2026, 10, 13, 9, 0, tzinfo=UTC).isoformat(),
        "metadata": {"product_version": "0.1.0", "latency_ms": 812.5},
    }


CASES: list[tuple[dict[str, Any], type[object]]] = [
    ({"outcome": "sql_generated", "sql": "SELECT 1"}, SqlGeneratedResult),
    ({"outcome": "clarification", "message": "Which currency?"}, ClarificationResult),
    ({"outcome": "denied", "message": "Forecasting is not supported."}, DeniedResult),
    ({"outcome": "error", "error_type": "InternalError", "message": "boom"}, ErrorResult),
    ({"outcome": "timeout", "timeout_seconds": 30}, TimeoutResult),
]


@pytest.mark.parametrize(("extra", "expected_type"), CASES)
def test_union_json_round_trip(extra: dict[str, Any], expected_type: type[object]) -> None:
    parsed = EXECUTION_RESULT_ADAPTER.validate_python({**_common(), **extra})
    assert isinstance(parsed, expected_type)
    dumped = EXECUTION_RESULT_ADAPTER.dump_json(parsed)
    again = EXECUTION_RESULT_ADAPTER.validate_json(dumped)
    assert again == parsed
    assert type(again) is expected_type


def test_all_five_outcomes_covered() -> None:
    assert {c[0]["outcome"] for c in CASES} == {o.value for o in ProductOutcome}


@pytest.mark.parametrize(
    "payload",
    [
        {"outcome": "sql_generated"},
        {"outcome": "clarification"},
        {"outcome": "denied"},
        {"outcome": "error", "message": "no type"},
        {"outcome": "sql_generated", "sql": "   "},
    ],
)
def test_required_outcome_fields(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EXECUTION_RESULT_ADAPTER.validate_python({**_common(), **payload})


def test_unknown_outcome_rejected() -> None:
    with pytest.raises(ValidationError):
        EXECUTION_RESULT_ADAPTER.validate_python({**_common(), "outcome": "partial"})


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        EXECUTION_RESULT_ADAPTER.validate_python(
            {**_common(), "outcome": "sql_generated", "sql": "SELECT 1", "rows": []}
        )


def test_wrong_field_for_outcome_rejected() -> None:
    with pytest.raises(ValidationError):
        EXECUTION_RESULT_ADAPTER.validate_python(
            {**_common(), "outcome": "denied", "message": "no", "sql": "SELECT 1"}
        )
