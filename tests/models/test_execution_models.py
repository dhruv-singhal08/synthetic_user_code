from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from synthetic_user.models.execution import (
    LEGACY_ADAPTER,
    AnsweredResponse,
    ClarificationResponse,
    DeniedResponse,
    ErrorResponse,
    ExecutionRecord,
    ProductOutcome,
    TimeoutResponse,
)


def _common() -> dict[str, Any]:
    return {
        "execution_id": "EXE_001",
        "question_id": "Q_001",
        "product_id": "sample_nl2sql",
        "adapter": "fixture",
        "submitted_question": "Top merchants by spend?",
        "started_at": datetime(2026, 10, 13, 9, 0, tzinfo=UTC).isoformat(),
        "latency_ms": 812.5,
    }


def _record(response: dict[str, Any], **extra: Any) -> ExecutionRecord:
    return ExecutionRecord.model_validate({**_common(), "response": response, **extra})


CASES: list[tuple[dict[str, Any], type[object]]] = [
    (
        {"outcome": "answered", "artifacts": [{"kind": "sql", "content": "SELECT 1"}]},
        AnsweredResponse,
    ),
    ({"outcome": "answered", "text": "Hotels are capped at 200 EUR."}, AnsweredResponse),
    ({"outcome": "clarification", "message": "Which currency?"}, ClarificationResponse),
    ({"outcome": "denied", "message": "Forecasting is not supported."}, DeniedResponse),
    ({"outcome": "error", "error_type": "InternalError", "message": "boom"}, ErrorResponse),
    ({"outcome": "timeout", "timeout_seconds": 30}, TimeoutResponse),
]


@pytest.mark.parametrize(("response", "expected_type"), CASES)
def test_record_json_round_trip(response: dict[str, Any], expected_type: type[object]) -> None:
    parsed = _record(response)
    assert isinstance(parsed.response, expected_type)
    assert parsed.outcome == response["outcome"]
    again = ExecutionRecord.model_validate_json(parsed.model_dump_json())
    assert again == parsed
    assert type(again.response) is expected_type


def test_all_five_outcomes_covered() -> None:
    assert {c[0]["outcome"] for c in CASES} == {o.value for o in ProductOutcome}


def test_artifact_content_may_be_structured() -> None:
    record = _record(
        {
            "outcome": "answered",
            "text": "Two rows.",
            "artifacts": [
                {"kind": "table", "content": [{"merchant": "A", "spend": 10}]},
                {"kind": "json", "content": {"rows": 2}, "metadata": {"source": "api"}},
            ],
        }
    )
    assert isinstance(record.response, AnsweredResponse)
    table = record.response.first_artifact("table")
    assert table is not None and table.content == [{"merchant": "A", "spend": 10}]
    assert record.response.first_artifact("sql") is None


@pytest.mark.parametrize(
    "response",
    [
        {"outcome": "answered"},
        {"outcome": "answered", "text": "   "},
        {"outcome": "answered", "artifacts": [{"kind": " ", "content": "x"}]},
        {"outcome": "clarification"},
        {"outcome": "denied"},
        {"outcome": "error", "message": "no type"},
    ],
)
def test_required_outcome_fields(response: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _record(response)


def test_unknown_outcome_rejected() -> None:
    with pytest.raises(ValidationError):
        _record({"outcome": "partial"})


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        _record({"outcome": "answered", "text": "ok", "rows": []})
    with pytest.raises(ValidationError, match="extra"):
        _record({"outcome": "answered", "text": "ok"}, rows=[])


def test_wrong_field_for_outcome_rejected() -> None:
    with pytest.raises(ValidationError):
        _record({"outcome": "denied", "message": "no", "artifacts": []})


def test_adapter_required() -> None:
    data = {**_common(), "response": {"outcome": "answered", "text": "ok"}}
    del data["adapter"]
    with pytest.raises(ValidationError, match="adapter"):
        ExecutionRecord.model_validate(data)


def _legacy(**payload: Any) -> dict[str, Any]:
    common = _common()
    del common["adapter"], common["latency_ms"]
    return {**common, "metadata": {"product_version": "0.1.0", "latency_ms": 812.5}, **payload}


def test_legacy_sql_generated_migrates_to_answered_sql_artifact() -> None:
    record = ExecutionRecord.model_validate(_legacy(outcome="sql_generated", sql="SELECT 1"))
    assert record.outcome is ProductOutcome.ANSWERED
    assert isinstance(record.response, AnsweredResponse)
    assert [(a.kind, a.content) for a in record.response.artifacts] == [("sql", "SELECT 1")]
    assert record.response.product_version == "0.1.0"
    assert (record.adapter, record.latency_ms) == (LEGACY_ADAPTER, 812.5)


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        ({"outcome": "clarification", "message": "Which currency?"}, ClarificationResponse),
        ({"outcome": "error", "error_type": "Boom"}, ErrorResponse),
        ({"outcome": "timeout", "timeout_seconds": 30}, TimeoutResponse),
    ],
)
def test_legacy_other_outcomes_migrate(
    payload: dict[str, Any], expected_type: type[object]
) -> None:
    record = ExecutionRecord.model_validate(_legacy(**payload))
    assert type(record.response) is expected_type


def test_legacy_unknown_keys_still_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        ExecutionRecord.model_validate(_legacy(outcome="sql_generated", sql="SELECT 1", rows=[]))
    with pytest.raises(ValidationError):
        ExecutionRecord.model_validate(_legacy(outcome="sql_generated"))
