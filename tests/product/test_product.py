from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import InputAssetError, LLMProviderError, ProductClientError
from synthetic_user.core.identifiers import SequentialIdGenerator
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.scripted import LLMScript, ScriptedLLMClient
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.product_executor import ProductExecutor
from synthetic_user.product.base import ProductClient, ProductResponse
from synthetic_user.product.fixture_client import FixtureProductClient, ProductFixture
from synthetic_user.references.snapshot import snapshot_id_for

REPO_ROOT = Path(__file__).resolve().parents[2]


def _questions(scenario: Scenario, *texts: str) -> list[QuestionCase]:
    return [
        QuestionCase.model_validate(
            {
                "question_id": f"Q_{i:03d}",
                "scenario_id": scenario.id,
                "persona_id": scenario.persona_id,
                "reference_snapshot_id": snapshot_id_for(scenario.id),
                "utterance": text,
                "intended_goal": scenario.business_goal,
                "intent": {"task_type": "ranking"},
                "question_characteristics": {"ambiguity_level": "low"},
                "expectations": {"clarification": "conditional", "acceptable_behaviors": ["x"]},
            }
        )
        for i, text in enumerate(texts, start=1)
    ]


def _fixture(**overrides: Any) -> ProductFixture:
    return ProductFixture.model_validate(
        {
            "product_id": "p",
            "product_version": "v1",
            "rules": [
                {
                    "contains": ["Top", "spenders"],
                    "response": {"outcome": "sql_generated", "sql": "SELECT 1"},
                }
            ],
            "default": {"outcome": "denied", "message": "no"},
            **overrides,
        }
    )


# --- product response / fixture client -------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"outcome": "sql_generated"},
        {"outcome": "clarification"},
        {"outcome": "denied"},
        {"outcome": "error"},
        {"outcome": "sql_generated", "sql": "SELECT 1", "rows": []},
    ],
)
def test_product_response_requires_outcome_payload(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ProductResponse.model_validate(payload)


async def test_fixture_matching_is_case_insensitive_and_ordered() -> None:
    client = FixtureProductClient(_fixture())
    assert isinstance(client, ProductClient)
    hit = await client.ask("who are our TOP card SPENDERS?")
    assert (hit.outcome, hit.sql, hit.product_version) == ("sql_generated", "SELECT 1", "v1")
    miss = await client.ask("forecast next quarter")
    assert miss.outcome == "denied"
    assert client.asked == ["who are our TOP card SPENDERS?", "forecast next quarter"]


def test_repo_fixture_loads() -> None:
    client = FixtureProductClient.from_file(
        REPO_ROOT / "data" / "fixtures" / "product_responses" / "sample_responses.json"
    )
    assert client.product_id == "sample_nl2sql"


def test_bad_fixture_file(tmp_path: Path) -> None:
    with pytest.raises(InputAssetError, match="not found"):
        FixtureProductClient.from_file(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"product_id": "p", "default": {"outcome": "error"}}), encoding="utf-8"
    )
    with pytest.raises(InputAssetError, match="error_type"):
        FixtureProductClient.from_file(bad)


# --- executor -------------------------------------------------------------------------------


class _SlowProduct:
    product_id = "slow"

    async def ask(self, utterance: str) -> ProductResponse:
        await asyncio.sleep(5)
        return ProductResponse(outcome="denied", message="late")  # type: ignore[arg-type]


class _BrokenProduct:
    product_id = "broken"

    async def ask(self, utterance: str) -> ProductResponse:
        raise ProductClientError("connection refused")


def _executor(product: Any, settings: AppConfig) -> ProductExecutor:
    return ProductExecutor(
        product, settings, logging.getLogger("t"), id_generator=SequentialIdGenerator()
    )


async def test_executor_records_outcomes_in_order(
    app_config: AppConfig, sample_scenarios: list[Scenario]
) -> None:
    questions = _questions(sample_scenarios[0], "Top spenders?", "Forecast please")
    results = await _executor(FixtureProductClient(_fixture()), app_config).execute(questions)
    assert [(r.execution_id, r.question_id, r.outcome) for r in results] == [
        ("EXE_001", "Q_001", "sql_generated"),
        ("EXE_002", "Q_002", "denied"),
    ]
    assert results[0].metadata.product_version == "v1"
    assert results[0].metadata.latency_ms is not None
    assert results[0].product_id == "p"


async def test_executor_timeout_and_error(
    app_config: AppConfig, sample_scenarios: list[Scenario]
) -> None:
    fast = app_config.model_copy(
        update={"product": app_config.product.model_copy(update={"timeout_seconds": 0.01})}
    )
    questions = _questions(sample_scenarios[0], "anything")
    timed_out = await _executor(_SlowProduct(), fast).execute(questions)
    assert timed_out[0].outcome == "timeout"
    errored = await _executor(_BrokenProduct(), app_config).execute(questions)
    assert errored[0].outcome == "error"
    assert errored[0].error_type == "ProductClientError"  # type: ignore[union-attr]


# --- scripted LLM ---------------------------------------------------------------------------


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
