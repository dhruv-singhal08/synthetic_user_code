from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from synthetic_user.adapters.base import ProductRequest
from synthetic_user.adapters.fixture import FixtureAdapter, ProductFixture, fixture_template
from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import InputAssetError, ProductAdapterError
from synthetic_user.models.execution import (
    AnsweredResponse,
    ClarificationResponse,
    DeniedResponse,
    ErrorResponse,
    ProductResponse,
)
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.product_executor import INVALID_ADAPTER_RESPONSE, ProductExecutor

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "data" / "fixtures" / "product_responses"


def _questions(scenario: Scenario, *texts: str) -> list[QuestionCase]:
    return [
        QuestionCase.model_validate(
            {
                "question_id": f"Q_{i:03d}",
                "scenario_id": scenario.id,
                "persona_id": scenario.persona_id,
                "mode": "grounded",
                "reference_pack_id": scenario.reference_pack_id,
                "utterance": text,
                "intended_goal": scenario.business_goal,
                "intent": {"task_type": "ranking"},
                "question_characteristics": {
                    "style": "precise",
                    "ambiguity_level": "low",
                    "complexity": "simple",
                },
                "expectations": {
                    "expected_behavior": "answers the question",
                    "success_criteria": ["answers the question"],
                },
            }
        )
        for i, text in enumerate(texts, start=1)
    ]


def _fast(settings: AppConfig, timeout: float = 0.05) -> AppConfig:
    return settings.model_copy(
        update={"product": settings.product.model_copy(update={"timeout_seconds": timeout})}
    )


class ScriptedAdapter:
    """Test adapter: per utterance, return a value, raise an exception, or sleep."""

    name = "scripted"

    def __init__(self, script: dict[str, Any], delays: dict[str, float] | None = None) -> None:
        self.script = script
        self.delays = delays or {}
        self.closed = 0
        self.sent: list[str] = []

    async def send(self, req: ProductRequest) -> ProductResponse:
        self.sent.append(req.utterance)
        await asyncio.sleep(self.delays.get(req.utterance, 0))
        value = self.script[req.utterance]
        if isinstance(value, BaseException):
            raise value
        return value  # type: ignore[no-any-return]

    async def aclose(self) -> None:
        self.closed += 1


async def _run(
    adapter: Any,
    settings: AppConfig,
    scenario: Scenario,
    *texts: str,
    raw: dict[str, JsonValue] | None = None,
) -> list[Any]:
    def writer(execution_id: str, body: JsonValue) -> str:
        assert raw is not None
        raw[execution_id] = body
        return f"raw/{execution_id}.json"

    executor = ProductExecutor(adapter, settings, logging.getLogger("t"))
    return await executor.execute_many(
        _questions(scenario, *texts),
        product_id="p",
        raw_writer=writer if raw is not None else None,
    )


async def test_each_outcome_recorded(
    app_config: AppConfig, sample_scenarios: list[Scenario]
) -> None:
    adapter = ScriptedAdapter(
        {
            "a": AnsweredResponse(text="42"),
            "c": ClarificationResponse(message="which year?"),
            "d": DeniedResponse(message="not supported"),
            "e": ErrorResponse(error_type="Boom"),
            "t": AnsweredResponse(text="late"),
        },
        delays={"t": 1.0},
    )
    records = await _run(adapter, _fast(app_config), sample_scenarios[0], "a", "c", "d", "e", "t")
    assert [r.outcome for r in records] == [
        "answered",
        "clarification",
        "denied",
        "error",
        "timeout",
    ]
    assert [r.execution_id for r in records] == [f"EXE_00{i}" for i in range(1, 6)]
    assert all(r.adapter == "scripted" and r.product_id == "p" for r in records)
    assert records[4].response.timeout_seconds == 0.05 and records[4].latency_ms is None


async def test_order_kept_under_concurrency(
    app_config: AppConfig, sample_scenarios: list[Scenario]
) -> None:
    texts = [f"q{i}" for i in range(6)]
    adapter = ScriptedAdapter(
        {t: AnsweredResponse(text=t) for t in texts},
        delays={t: 0.03 * (6 - i) for i, t in enumerate(texts)},  # later questions finish first
    )
    records = await _run(adapter, app_config, sample_scenarios[0], *texts)
    assert [r.response.text for r in records] == texts
    assert [r.question_id for r in records] == [f"Q_00{i}" for i in range(1, 7)]


async def test_product_max_concurrency_limits_requests_in_flight(
    app_config: AppConfig, sample_scenarios: list[Scenario]
) -> None:
    texts = [f"q{i}" for i in range(5)]
    in_flight = peak = 0

    class CountingAdapter(ScriptedAdapter):
        async def send(self, req: ProductRequest) -> ProductResponse:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                return await super().send(req)
            finally:
                in_flight -= 1

    adapter = CountingAdapter(
        {t: AnsweredResponse(text=t) for t in texts}, delays=dict.fromkeys(texts, 0.01)
    )
    for limit in (1, 3):
        peak = 0
        settings = app_config.model_copy(
            update={"product": app_config.product.model_copy(update={"max_concurrency": limit})}
        )
        records = await _run(adapter, settings, sample_scenarios[0], *texts)
        assert [r.response.outcome for r in records] == ["answered"] * 5
        assert peak == limit


async def test_adapter_error_and_unexpected_exception(
    app_config: AppConfig, sample_scenarios: list[Scenario], caplog: pytest.LogCaptureFixture
) -> None:
    adapter = ScriptedAdapter(
        {"known": ProductAdapterError("connection refused"), "bug": KeyError("data")}
    )
    caplog.set_level(logging.ERROR)
    records = await _run(adapter, app_config, sample_scenarios[0], "known", "bug")
    assert [(r.response.error_type, r.outcome) for r in records] == [
        ("ProductAdapterError", "error"),
        ("KeyError", "error"),
    ]
    assert "connection refused" in records[0].response.message
    # the unexpected one is logged with its traceback
    assert any(r.exc_info for r in caplog.records) and "KeyError" in caplog.text


@pytest.mark.parametrize(
    "bad",
    [
        {"outcome": "answered"},  # no text, no artifacts
        AnsweredResponse.model_construct(outcome="answered", text=None, artifacts=[]),
        "just a string",
        None,
    ],
)
async def test_invalid_adapter_output_is_error_never_an_answer(
    app_config: AppConfig, sample_scenarios: list[Scenario], bad: Any
) -> None:
    records = await _run(ScriptedAdapter({"q": bad}), app_config, sample_scenarios[0], "q")
    response = records[0].response
    assert isinstance(response, ErrorResponse)
    assert response.error_type == INVALID_ADAPTER_RESPONSE


async def test_aclose_always_called(
    app_config: AppConfig, sample_scenarios: list[Scenario]
) -> None:
    adapter = ScriptedAdapter({"ok": AnsweredResponse(text="x"), "bug": RuntimeError("x")})
    await _run(adapter, app_config, sample_scenarios[0], "ok", "bug")
    assert adapter.closed == 1

    class Cancelling(ScriptedAdapter):
        async def send(self, req: ProductRequest) -> ProductResponse:
            raise asyncio.CancelledError

    cancelling = Cancelling({})
    with pytest.raises(asyncio.CancelledError):
        await _run(cancelling, app_config, sample_scenarios[0], "q")
    assert cancelling.closed == 1


async def test_latency_from_adapter_or_measured_and_raw_saved(
    app_config: AppConfig, sample_scenarios: list[Scenario]
) -> None:
    adapter = ScriptedAdapter(
        {
            "own": AnsweredResponse(text="x", latency_ms=12.5, raw_response={"answer": "x"}),
            "measured": AnsweredResponse(text="y"),
        }
    )
    raw: dict[str, JsonValue] = {}
    records = await _run(adapter, app_config, sample_scenarios[0], "own", "measured", raw=raw)
    assert records[0].latency_ms == 12.5 and records[1].latency_ms is not None
    assert raw == {"EXE_001": {"answer": "x"}}
    assert records[0].raw_response_path == "raw/EXE_001.json"
    assert records[0].response.raw_response is None  # moved out of the stored record
    assert records[1].raw_response_path is None


# --- sample fixtures ------------------------------------------------------------------------------


CASES = {
    # fixture -> (question texts in Q_001.. order, expected outcomes)
    "sample_responses.json": (
        ["Who were our top 10 card spenders?", "monthly spend?", "forecast it"],
        ["answered", "clarification", "denied"],
    ),
    # answers are keyed by question id first: Q_001 answered, Q_002 denied
    "sample_chatbot_responses.json": (
        [
            "How much can I spend per night on a hotel in Paris?",
            "Can you approve my expense report for me?",
            "Do I need a receipt?",
        ],
        ["answered", "denied", "clarification"],
    ),
}


@pytest.mark.parametrize("name", list(CASES))
async def test_sample_fixtures_cover_all_five_outcomes(
    app_config: AppConfig, sample_scenarios: list[Scenario], name: str
) -> None:
    texts, expected = CASES[name]
    adapter = FixtureAdapter.from_file(FIXTURES / name)
    records = await _run(
        adapter,
        _fast(app_config),
        sample_scenarios[0],
        *texts,
        "simulate outage",
        "simulate timeout please",
    )
    assert [r.outcome for r in records] == [*expected, "error", "timeout"]


async def test_chatbot_fixture_is_text_only(
    app_config: AppConfig, sample_scenarios: list[Scenario]
) -> None:
    adapter = FixtureAdapter.from_file(FIXTURES / "sample_chatbot_responses.json")
    assert adapter.product_id == "sample_policy_chatbot"
    records = await _run(
        adapter,
        app_config,
        sample_scenarios[0],
        "how much can I spend per night on a hotel in paris",
    )
    response = records[0].response
    assert isinstance(response, AnsweredResponse) and response.artifacts == []
    assert "220 EUR" in (response.text or "")


# --- fixture template -----------------------------------------------------------------------------


def test_fixture_template_round_trip(tmp_path: Path) -> None:
    template = fixture_template("sample_policy_chatbot", [("Q_001", "Hi?"), ("Q_002", "Bye?")])
    assert [a["question_id"] for a in template["answers"]] == ["Q_001", "Q_002"]
    assert set(template["answers"][0]) == {
        "question_id",
        "utterance",
        "outcome",
        "text",
        "artifacts",
    }
    path = tmp_path / "t.json"
    path.write_text(json.dumps(template), encoding="utf-8")
    with pytest.raises(InputAssetError, match="answer for Q_001 is not filled in"):
        FixtureAdapter.from_file(path)  # unfilled answers never pass as product answers

    template["answers"][0].update(text="Hello!")
    template["answers"][1].update(
        outcome="answered", artifacts=[{"kind": "sql", "content": "SELECT 1"}]
    )
    path.write_text(json.dumps(template), encoding="utf-8")
    fixture = FixtureAdapter.from_file(path)
    assert fixture.product_id == "sample_policy_chatbot"


async def test_template_answers_match_by_utterance_when_ids_change() -> None:
    fixture = ProductFixture.model_validate(
        {
            "product_id": "p",
            "answers": [
                {
                    "question_id": "Q_001",
                    "utterance": "Hi there?",
                    "outcome": "answered",
                    "text": "Hello",
                }
            ],
            "default": {"outcome": "error", "error_type": "NoFixtureMatch"},
        }
    )
    adapter = FixtureAdapter(fixture)
    by_text = await adapter.send(ProductRequest(question_id="Q_009", utterance="hi there"))
    assert isinstance(by_text, AnsweredResponse) and by_text.text == "Hello"
