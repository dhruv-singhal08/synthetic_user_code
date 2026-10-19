from __future__ import annotations

import csv
import io
from datetime import UTC, datetime

import pytest

from synthetic_user.adapters.base import ProductRequest
from synthetic_user.adapters.contract import request_contract
from synthetic_user.adapters.mock_llm import MockLLMAdapter, parse_mock_reply
from synthetic_user.core.exceptions import LLMProviderError, ProductAdapterError
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.models.evaluation import EvaluationResult
from synthetic_user.models.execution import AnsweredResponse, ExecutionRecord
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.scenario import Scenario
from synthetic_user.web.results_csv import COLUMNS, build_rows, bullets, to_csv


def _question(scenario: Scenario) -> QuestionCase:
    return QuestionCase.model_validate(
        {
            "question_id": "Q_001",
            "scenario_id": scenario.id,
            "persona_id": scenario.persona_id,
            "mode": "grounded",
            "reference_pack_id": scenario.reference_pack_id,
            "utterance": "Top spenders?",
            "intended_goal": scenario.business_goal,
            "intent": {"task_type": "ranking"},
            "question_characteristics": {
                "style": "precise",
                "ambiguity_level": "low",
                "complexity": "simple",
            },
            "expectations": {
                "expected_behavior": "Rank the top spenders.",
                "success_criteria": ["names the top spenders"],
            },
        }
    )


def test_rows_one_per_question_and_blank_for_unasked_scenarios(
    sample_scenarios: list[Scenario], sample_packs: list[ReferencePack]
) -> None:
    question = _question(sample_scenarios[0])
    execution = ExecutionRecord.model_validate(
        {
            "execution_id": "EXE_001",
            "question_id": "Q_001",
            "product_id": "p",
            "adapter": "fixture",
            "submitted_question": "Top spenders?",
            "started_at": datetime(2026, 10, 14, tzinfo=UTC),
            "latency_ms": 12.4,
            "response": {
                "outcome": "answered",
                "text": "Here are the top spenders.",
                "artifacts": [{"kind": "sql", "content": "SELECT 1"}],
            },
        }
    )
    rows = build_rows("RUN_1", sample_scenarios, [question], [execution], packs=sample_packs)
    assert len(rows) == 3
    assert all(list(r) == COLUMNS for r in rows)
    assert (rows[0]["question_id"], rows[0]["artifacts"], rows[0]["latency_ms"]) == (
        "Q_001",
        "sql:\nSELECT 1",
        "12",
    )
    assert rows[0]["expected_behavior"] == "Rank the top spenders."
    assert rows[0]["success_criteria"] == "names the top spenders"  # one item -> as is
    assert rows[0]["cheat_sheet"] == (  # several items -> bullets
        "• card_transactions (amount, currency_code)\n• employees (full_name)"
    )
    assert rows[0]["cheat_sheet_rules"].split("\n") == [
        "• join card_transactions.employee_id -> employees.employee_id (verified)",
        "• card_transactions: Amounts are in the transaction currency; amounts in different "
        "currencies must not be summed directly.",
        "• R1: Amounts in different currency_code values must not be summed directly.",
    ]
    assert rows[0]["answer_text"] == "Here are the top spenders."
    assert rows[1]["question_id"] == "" and rows[1]["scenario_id"] == "SCN_002"
    assert rows[0]["final_verdict"] == ""


def test_columns_are_the_evaluation_view() -> None:
    assert len(COLUMNS) == 26
    assert COLUMNS[-10:-5] == [
        "relevancy",
        "correctness",
        "groundedness",
        "hallucination_rate",
        "completeness",
    ]
    for removed in (
        "failed_checks",
        "inconclusive_checks",
        "task_type",
        "capability_rationale",
        "cheat_tables",
        "clarification_expected",
        "unacceptable_behaviors",
        "interpretation_notes",
        "generated_sql",
        "plugins_applied",
        "llm_verdict",
    ):
        assert removed not in COLUMNS


def test_csv_round_trip(sample_scenarios: list[Scenario]) -> None:
    rows = build_rows("RUN_1", sample_scenarios)
    parsed = list(csv.DictReader(io.StringIO(to_csv(rows))))
    assert parsed == rows


@pytest.mark.parametrize(
    ("text", "answer", "artifacts"),
    [
        ("```sql\nSELECT 1\n```", None, [("sql", "SELECT 1")]),
        (
            "Here you go:\n```SQL\nSELECT 2\n```\nThanks",
            "Here you go:\n\nThanks",
            [("sql", "SELECT 2")],
        ),
        ("The limit is USD 250.", "The limit is USD 250.", []),
        ('Chart:\n```json\n{"a": 1}\n```', "Chart:", [("json", '{"a": 1}')]),
    ],
)
def test_parse_mock_reply_answers(
    text: str, answer: str | None, artifacts: list[tuple[str, str]]
) -> None:
    response = parse_mock_reply(text)
    assert isinstance(response, AnsweredResponse)
    assert response.text == answer
    assert [(a.kind, a.content) for a in response.artifacts] == artifacts


@pytest.mark.parametrize(
    ("text", "outcome", "field", "value"),
    [
        ("CLARIFY: which currency?", "clarification", "message", "which currency?"),
        ("denied: not supported", "denied", "message", "not supported"),
        ("   ", "error", "error_type", "EmptyReply"),
    ],
)
def test_parse_mock_reply_other_outcomes(text: str, outcome: str, field: str, value: str) -> None:
    response = parse_mock_reply(text)
    assert response.outcome == outcome
    assert getattr(response, field) == value


async def test_mock_product_client_uses_reference_and_wraps_errors(
    sample_reference: Reference, sample_product: ProductDescription
) -> None:
    contract = request_contract("https://x.example", "sample_nl2sql")
    llm = FakeLLMClient(["```sql\nSELECT 1\n```", LLMProviderError("down")])
    client = MockLLMAdapter(llm, contract, sample_product, sample_reference)
    response = await client.send(ProductRequest(question_id="Q_001", utterance="Top spenders?"))
    assert isinstance(response, AnsweredResponse) and response.product_version == "mock-1"
    assert [(a.kind, a.content) for a in response.artifacts] == [("sql", "SELECT 1")]
    assert client.name == "mock"
    user = llm.calls[0].user
    assert "card_transactions" in user and "User message: Top spenders?" in user
    with pytest.raises(ProductAdapterError, match="mock product failed"):
        await client.send(ProductRequest(question_id="Q_002", utterance="again"))


async def test_mock_product_works_without_reference(sample_chatbot: ProductDescription) -> None:
    llm = FakeLLMClient(["Hotels in Paris are capped at USD 250 per night."])
    client = MockLLMAdapter(
        llm, request_contract("https://x.example", "sample_policy_chatbot"), sample_chatbot, None
    )
    response = await client.send(ProductRequest(question_id="Q_001", utterance="Paris hotel?"))
    assert isinstance(response, AnsweredResponse) and response.artifacts == []
    assert "No reference data" in llm.calls[0].user


@pytest.mark.parametrize("url", ["", "nl2sql.example.com", "ftp://x.example", "https://"])
def test_contract_rejects_bad_urls(url: str) -> None:
    with pytest.raises(ProductAdapterError):
        request_contract(url, "p")


def test_bullets_and_metric_cells(sample_scenarios: list[Scenario]) -> None:
    assert bullets([]) == "" and bullets(["one"]) == "one"
    assert bullets(["a", "", "b"]) == "• a\n• b"
    question = _question(sample_scenarios[0])
    execution = ExecutionRecord.model_validate(
        {
            "execution_id": "EXE_001",
            "question_id": "Q_001",
            "product_id": "p",
            "adapter": "fixture",
            "submitted_question": "Top spenders?",
            "started_at": datetime(2026, 10, 14, tzinfo=UTC),
            "response": {
                "outcome": "answered",
                "artifacts": [
                    {"kind": "sql", "content": "SELECT 1"},
                    {"kind": "tool_calls", "content": [{"name": "aggregate"}]},
                ],
            },
        }
    )
    metric = {"status": "scored", "score": 1.0, "rationale": "ok"}
    evaluation = EvaluationResult.model_validate(
        {
            "evaluation_id": "EVAL_001",
            "question_id": "Q_001",
            "scenario_id": "SCN_001",
            "execution_id": "EXE_001",
            "product_outcome": "answered",
            "mode": "grounded",
            "verdict": "inconclusive",
            "evaluation_status": "evaluated",
            "metrics": {
                "relevancy": metric,
                "correctness": {"status": "inconclusive", "rationale": "cannot tell"},
                "groundedness": {"status": "not_applicable", "rationale": "no claims"},
                "hallucination_rate": {**metric, "score": 0.25},
                "completeness": {**metric, "score": 0.5},
            },
            "criteria_results": [
                {"criterion": "names the top spenders", "status": "met", "rationale": "ok"},
                {"criterion": "per currency", "status": "not_met", "rationale": "no"},
            ],
        }
    )
    row = build_rows("RUN_1", sample_scenarios[:1], [question], [execution], [evaluation])[0]
    assert row["artifacts"].startswith("• sql:\nSELECT 1\n• tool_calls:\n[")
    assert [row[m] for m in ("relevancy", "correctness", "groundedness")] == [
        "1",
        "inconclusive",
        "",
    ]
    assert (row["hallucination_rate"], row["completeness"]) == ("0.25", "0.5")
    assert row["criteria_results"] == "• met: names the top spenders\n• not_met: per currency"
    parsed = list(csv.DictReader(io.StringIO(to_csv([row]))))
    assert parsed == [row]  # multi-line cells survive the CSV round trip
