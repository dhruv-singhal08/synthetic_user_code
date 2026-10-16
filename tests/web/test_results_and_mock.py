from __future__ import annotations

import csv
import io
from datetime import UTC, datetime

import pytest

from synthetic_user.core.exceptions import LLMProviderError, ProductClientError
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.models.execution import EXECUTION_RESULT_ADAPTER
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import ReferenceCatalog
from synthetic_user.models.scenario import Scenario
from synthetic_user.product.contract import request_contract
from synthetic_user.product.mock_client import MockProductClient, parse_mock_reply
from synthetic_user.references.snapshot import snapshot_id_for
from synthetic_user.web.results_csv import COLUMNS, build_rows, to_csv


def _question(scenario: Scenario) -> QuestionCase:
    return QuestionCase.model_validate(
        {
            "question_id": "Q_001",
            "scenario_id": scenario.id,
            "persona_id": scenario.persona_id,
            "reference_snapshot_id": snapshot_id_for(scenario.id),
            "utterance": "Top spenders?",
            "intended_goal": scenario.business_goal,
            "intent": {"task_type": "ranking"},
            "question_characteristics": {"ambiguity_level": "low"},
            "expectations": {"clarification": "conditional", "acceptable_behaviors": ["a", "b"]},
        }
    )


def test_rows_one_per_question_and_blank_for_unasked_scenarios(
    sample_scenarios: list[Scenario],
) -> None:
    question = _question(sample_scenarios[0])
    execution = EXECUTION_RESULT_ADAPTER.validate_python(
        {
            "execution_id": "EXE_001",
            "question_id": "Q_001",
            "product_id": "p",
            "submitted_question": "Top spenders?",
            "started_at": datetime(2026, 10, 14, tzinfo=UTC),
            "outcome": "sql_generated",
            "sql": "SELECT 1",
            "metadata": {"latency_ms": 12.4},
        }
    )
    rows = build_rows("RUN_1", sample_scenarios, [question], [execution])
    assert len(rows) == 3
    assert all(list(r) == COLUMNS for r in rows)
    assert (rows[0]["question_id"], rows[0]["generated_sql"], rows[0]["latency_ms"]) == (
        "Q_001",
        "SELECT 1",
        "12",
    )
    assert rows[0]["acceptable_behaviors"] == "a | b"
    assert rows[1]["question_id"] == "" and rows[1]["scenario_id"] == "SCN_002"
    assert rows[0]["final_verdict"] == ""


def test_csv_round_trip(sample_scenarios: list[Scenario]) -> None:
    rows = build_rows("RUN_1", sample_scenarios)
    parsed = list(csv.DictReader(io.StringIO(to_csv(rows))))
    assert parsed == rows


@pytest.mark.parametrize(
    ("text", "outcome", "field", "value"),
    [
        ("```sql\nSELECT 1\n```", "sql_generated", "sql", "SELECT 1"),
        ("Here you go:\n```SQL\nSELECT 2\n```\nThanks", "sql_generated", "sql", "SELECT 2"),
        ("SELECT 3", "sql_generated", "sql", "SELECT 3"),
        ("CLARIFY: which currency?", "clarification", "message", "which currency?"),
        ("denied: not supported", "denied", "message", "not supported"),
        ("   ", "error", "error_type", "EmptyReply"),
    ],
)
def test_parse_mock_reply(text: str, outcome: str, field: str, value: str) -> None:
    response = parse_mock_reply(text)
    assert response.outcome == outcome
    assert getattr(response, field) == value


async def test_mock_product_client_uses_schema_and_wraps_errors(
    sample_catalog: ReferenceCatalog, sample_product: ProductDescription
) -> None:
    contract = request_contract("https://x.example", "sample_nl2sql", "bigquery")
    llm = FakeLLMClient(["```sql\nSELECT 1\n```", LLMProviderError("down")])
    client = MockProductClient(llm, contract, sample_catalog, sample_product)
    response = await client.ask("Top spenders?")
    assert response.sql == "SELECT 1" and response.product_version == "mock-1"
    assert (
        "card_transactions" in llm.calls[0].user and "Question: Top spenders?" in llm.calls[0].user
    )
    with pytest.raises(ProductClientError, match="mock product failed"):
        await client.ask("again")


@pytest.mark.parametrize("url", ["", "nl2sql.example.com", "ftp://x.example", "https://"])
def test_contract_rejects_bad_urls(url: str) -> None:
    with pytest.raises(ProductClientError):
        request_contract(url, "p", "bigquery")
