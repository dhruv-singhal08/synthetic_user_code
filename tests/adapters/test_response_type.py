"""The product's own response type decides the outcome and is kept (D242).

Live finding: a run with 4 SQL, 4 ``unsupported`` and 2 ``blocked`` responses was recorded as
10 answered with no response classification, because every reply was an HTTP 200 with text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from synthetic_user.adapters.http_common import ResponseMapper, ResponseMapping
from synthetic_user.models.execution import (
    AnsweredResponse,
    ClarificationResponse,
    DeniedResponse,
    ErrorResponse,
    ResponseClassification,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _contract_mapper() -> ResponseMapper:
    """The mapping of the real NL2SQL contract (config/products/generate_sql.adapter.yaml)."""
    contract = yaml.safe_load(
        (REPO_ROOT / "config/products/generate_sql.adapter.yaml").read_text(encoding="utf-8")
    )
    return ResponseMapper(ResponseMapping.model_validate(contract["options"]["response_mapping"]))


def _reply(type_: str, **extra: Any) -> dict[str, Any]:
    return {"type": type_, "result": None, "message": f"{type_} message", **extra}


def test_sql_answer_is_typed_answered_and_keeps_its_type() -> None:
    response = _contract_mapper().map(_reply("sql", result="SELECT 1"), 200)
    assert isinstance(response, AnsweredResponse)
    assert response.response_classification is ResponseClassification.ANSWERED
    assert response.product_response_type == "sql"
    assert [a.kind for a in response.artifacts] == ["sql"]


@pytest.mark.parametrize("type_", ["unsupported", "blocked"])
def test_unsupported_and_blocked_are_denials_not_answers(type_: str) -> None:
    response = _contract_mapper().map(_reply(type_), 200)  # HTTP success, task not done
    assert isinstance(response, DeniedResponse)
    assert (response.message, response.product_response_type) == (f"{type_} message", type_)


def test_clarification_keeps_its_options_and_type() -> None:
    reply = _reply("clarification", options=["Top 10 routes", "Top 25 routes"])
    response = _contract_mapper().map(reply, 200)
    assert isinstance(response, ClarificationResponse)
    assert response.options == ["Top 10 routes", "Top 25 routes"]
    assert response.product_response_type == "clarification"


def test_unlisted_type_is_kept_and_never_assumed_answered(caplog: pytest.LogCaptureFixture) -> None:
    response = _contract_mapper().map(_reply("chart", result="SELECT 2"), 200)
    assert isinstance(response, AnsweredResponse)
    assert response.response_classification is None  # unknown kind: the judge classifies it
    assert response.product_response_type == "chart"
    assert "'chart' is not listed" in caplog.text


def test_error_and_unable_to_answer_types() -> None:
    mapping = ResponseMapping.model_validate(
        {
            "text": "message",
            "error_message": "message",
            "response_type": {
                "path": "status",
                "outcomes": {"failed": "error", "no_data": "unable_to_answer"},
            },
        }
    )
    mapper = ResponseMapper(mapping)
    error = mapper.map({"status": "failed", "message": "backend down"}, 200)
    assert isinstance(error, ErrorResponse) and error.message == "backend down"
    assert error.product_response_type == "failed"
    unable = mapper.map({"status": "no_data", "message": "No data for that period."}, 200)
    assert isinstance(unable, AnsweredResponse)
    assert unable.response_classification is ResponseClassification.UNABLE_TO_ANSWER


def test_contracts_without_response_type_are_unchanged() -> None:
    mapping = ResponseMapping.model_validate(
        {"text": "message", "denied_when": "type == unsupported", "denied_message": "message"}
    )
    mapper = ResponseMapper(mapping)
    denied = mapper.map(_reply("unsupported"), 200)
    assert isinstance(denied, DeniedResponse) and denied.product_response_type is None
    answered = mapper.map(_reply("blocked"), 200)  # not mapped: as before
    assert isinstance(answered, AnsweredResponse) and answered.response_classification is None


def test_response_type_needs_outcomes() -> None:
    with pytest.raises(ValueError, match="outcomes"):
        ResponseMapping.model_validate(
            {"text": "message", "response_type": {"path": "type", "outcomes": {}}}
        )
    with pytest.raises(ValueError, match="outcomes"):
        ResponseMapping.model_validate(
            {"text": "m", "response_type": {"path": "type", "outcomes": {"sql": "ok"}}}
        )
