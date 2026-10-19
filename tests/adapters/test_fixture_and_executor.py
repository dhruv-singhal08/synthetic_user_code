from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from synthetic_user.adapters.base import ProductAdapter, ProductRequest
from synthetic_user.adapters.fixture import FixtureAdapter, ProductFixture, normalize_utterance
from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import InputAssetError, LLMProviderError, ProductAdapterError
from synthetic_user.core.identifiers import SequentialIdGenerator
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.scripted import LLMScript, ScriptedLLMClient
from synthetic_user.models.execution import (
    AnsweredResponse,
    ClarificationResponse,
    DeniedResponse,
    ErrorResponse,
    ProductResponse,
)
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.product_executor import ProductExecutor

REPO_ROOT = Path(__file__).resolve().parents[2]

SQL_ANSWER = {"outcome": "answered", "artifacts": [{"kind": "sql", "content": "SELECT 1"}]}


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


def _fixture(**overrides: Any) -> ProductFixture:
    return ProductFixture.model_validate(
        {
            "product_id": "p",
            "product_version": "v1",
            "rules": [{"contains": ["Top", "spenders"], "response": SQL_ANSWER}],
            "default": {"outcome": "denied", "message": "no"},
            **overrides,
        }
    )


def _req(utterance: str, question_id: str = "Q_001") -> ProductRequest:
    return ProductRequest(question_id=question_id, utterance=utterance)


# --- product response / fixture adapter ---------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"outcome": "answered"},
        {"outcome": "clarification"},
        {"outcome": "denied"},
        {"outcome": "error"},
        {"outcome": "answered", "text": "ok", "rows": []},
        {"outcome": "sql_generated", "sql": "SELECT 1"},
    ],
)
def test_product_response_requires_outcome_payload(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ProductResponse).validate_python(payload)


def test_product_request_shape() -> None:
    req = ProductRequest(question_id="Q_001", utterance="hi")
    assert (req.session_id, req.context) == (None, {})
    with pytest.raises(ValidationError):
        ProductRequest(question_id="Q_001", utterance="  ")


async def test_fixture_keyword_rules_are_case_insensitive_and_ordered() -> None:
    adapter = FixtureAdapter(_fixture())
    assert isinstance(adapter, ProductAdapter) and adapter.name == "fixture"
    hit = await adapter.send(_req("who are our TOP card SPENDERS?"))
    assert isinstance(hit, AnsweredResponse) and hit.product_version == "v1"
    assert [(a.kind, a.content) for a in hit.artifacts] == [("sql", "SELECT 1")]
    miss = await adapter.send(_req("forecast next quarter", "Q_002"))
    assert miss.outcome == "denied"
    assert [r.utterance for r in adapter.asked] == [
        "who are our TOP card SPENDERS?",
        "forecast next quarter",
    ]


async def test_fixture_lookup_order_question_id_then_utterance_then_default() -> None:
    adapter = FixtureAdapter(
        _fixture(
            rules=[],
            by_question_id={"Q_007": {"outcome": "answered", "text": "by id"}},
            by_utterance={"Who spent the MOST?": {"outcome": "clarification", "message": "When?"}},
        )
    )
    by_id = await adapter.send(_req("Who spent the most?", "Q_007"))
    assert isinstance(by_id, AnsweredResponse) and by_id.text == "by id"
    by_text = await adapter.send(_req("  who   spent the most  ", "Q_001"))
    assert isinstance(by_text, ClarificationResponse) and by_text.message == "When?"
    fallback = await adapter.send(_req("something else", "Q_002"))
    assert isinstance(fallback, DeniedResponse)


def test_normalized_utterance_clash_rejected() -> None:
    assert normalize_utterance("Who, spent  THE most?") == "who spent the most"
    with pytest.raises(ValidationError, match="identical after normalization"):
        _fixture(by_utterance={"Hi there": SQL_ANSWER, "hi, there!": SQL_ANSWER})


async def test_simulated_timeout_hits_executor_timeout_and_aclose_releases(
    app_config: AppConfig, sample_scenarios: list[Scenario]
) -> None:
    adapter = FixtureAdapter(_fixture(by_question_id={"Q_001": {"simulate_timeout": True}}))
    fast = app_config.model_copy(
        update={"product": app_config.product.model_copy(update={"timeout_seconds": 0.01})}
    )
    questions = _questions(sample_scenarios[0], "anything", "Top spenders?")
    records = await _executor(adapter, fast).execute_many(questions, product_id="p")
    assert [r.outcome for r in records] == ["timeout", "answered"]
    assert records[0].latency_ms is None

    fresh = FixtureAdapter(_fixture(by_question_id={"Q_001": {"simulate_timeout": True}}))
    pending = asyncio.ensure_future(fresh.send(_req("anything")))
    await asyncio.sleep(0)
    assert not pending.done()
    await fresh.aclose()
    released = await pending
    assert isinstance(released, ErrorResponse) and released.error_type == "AdapterClosed"


def test_repo_fixture_loads() -> None:
    adapter = FixtureAdapter.from_file(
        REPO_ROOT / "data" / "fixtures" / "product_responses" / "sample_responses.json"
    )
    assert adapter.product_id == "sample_nl2sql"


def test_bad_fixture_file(tmp_path: Path) -> None:
    with pytest.raises(InputAssetError, match="not found"):
        FixtureAdapter.from_file(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"product_id": "p", "default": {"outcome": "error"}}), encoding="utf-8"
    )
    with pytest.raises(InputAssetError, match="error_type"):
        FixtureAdapter.from_file(bad)


# --- executor ---------------------------------------------------------------------------------


class _SlowProduct:
    name = "slow-test"

    async def send(self, req: ProductRequest) -> ProductResponse:
        await asyncio.sleep(5)
        return DeniedResponse(message="late")

    async def aclose(self) -> None:
        return None


class _BrokenProduct:
    name = "broken-test"

    async def send(self, req: ProductRequest) -> ProductResponse:
        raise ProductAdapterError("connection refused")

    async def aclose(self) -> None:
        return None


def _executor(product: ProductAdapter, settings: AppConfig) -> ProductExecutor:
    return ProductExecutor(
        product, settings, logging.getLogger("t"), id_generator=SequentialIdGenerator()
    )


async def test_executor_records_outcomes_in_order(
    app_config: AppConfig, sample_scenarios: list[Scenario]
) -> None:
    adapter = FixtureAdapter(_fixture())
    questions = _questions(sample_scenarios[0], "Top spenders?", "Forecast please")
    results = await _executor(adapter, app_config).execute_many(
        questions, product_id="sample_nl2sql"
    )
    assert [(r.execution_id, r.question_id, r.outcome) for r in results] == [
        ("EXE_001", "Q_001", "answered"),
        ("EXE_002", "Q_002", "denied"),
    ]
    assert results[0].response.product_version == "v1"
    assert results[0].latency_ms is not None
    assert (results[0].product_id, results[0].adapter) == ("sample_nl2sql", "fixture")
    sent = adapter.asked[0]
    assert (sent.question_id, sent.session_id) == ("Q_001", None)
    assert sent.context == {"scenario_id": "SCN_001", "persona_id": "finance_controller"}


async def test_executor_timeout_and_error(
    app_config: AppConfig, sample_scenarios: list[Scenario]
) -> None:
    fast = app_config.model_copy(
        update={"product": app_config.product.model_copy(update={"timeout_seconds": 0.01})}
    )
    questions = _questions(sample_scenarios[0], "anything")
    timed_out = await _executor(_SlowProduct(), fast).execute_many(questions, product_id="p")
    assert timed_out[0].outcome == "timeout"
    assert (timed_out[0].adapter, timed_out[0].latency_ms) == ("slow-test", None)
    errored = await _executor(_BrokenProduct(), app_config).execute_many(questions, product_id="p")
    assert isinstance(errored[0].response, ErrorResponse)
    assert errored[0].response.error_type == "ProductAdapterError"


# --- scripted LLM -----------------------------------------------------------------------------


async def test_scripted_llm_rules() -> None:
    llm = ScriptedLLMClient(
        LLMScript.model_validate(
            {
                "rules": [
                    {"contains": ["alpha", "beta"], "response": {"x": 1}},
                    {"contains": ["alpha"], "response": "plain"},
                ]
            }
        )
    )
    assert isinstance(llm, LLMClient)
    assert await llm.generate_json("alpha", "beta") == {"x": 1}
    assert await llm.generate_text("alpha", "gamma") == "plain"
    assert await llm.generate_json("alpha", "beta") == {"x": 1}  # rules are reusable
    with pytest.raises(LLMProviderError, match="no rule for call #4"):
        await llm.generate_text("zeta", "eta")


def test_repo_fake_script_loads() -> None:
    llm = ScriptedLLMClient.from_file(REPO_ROOT / "tests" / "fixtures" / "fake_llm_script.json")
    assert llm.provider_name == "fake"
