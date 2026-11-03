from __future__ import annotations

import csv
import io
import json
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
from synthetic_user.web.results_csv import build_rows, bullets, columns_for, to_csv


def _question(scenario: Scenario, pack: ReferencePack | None = None) -> QuestionCase:
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
                "success_criteria": {
                    "general": ["names the top spenders"],
                    "cheat_sheet": pack.model_dump(mode="json") if pack else None,
                },
            },
        }
    )


def test_rows_one_per_question_and_blank_for_unasked_scenarios(
    sample_scenarios: list[Scenario], sample_packs: list[ReferencePack]
) -> None:
    question = _question(sample_scenarios[0], sample_packs[0])
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
    rows = build_rows(
        "RUN_1",
        sample_scenarios,
        [question],
        [execution],
        packs=sample_packs,
        metric_ids=["relevance"],
    )
    assert len(rows) == 3
    assert all(list(r) == columns_for(["relevance"]) for r in rows)
    assert (rows[0]["question_id"], rows[0]["artifacts"], rows[0]["latency_ms"]) == (
        "Q_001",
        "sql:\nSELECT 1",
        "12",
    )
    assert rows[0]["expected_behavior"] == "Rank the top spenders."
    # One structured JSON cell: general criteria + the cheat sheet copied into the question.
    assert json.loads(rows[0]["success_criteria"]) == {
        "general": ["names the top spenders"],
        "cheat_sheet": {
            "items": ["card_transactions (amount, currency_code)", "employees (full_name)"],
            "rules": [
                "join card_transactions.employee_id -> employees.employee_id (verified)",
                "card_transactions: Amounts are in the transaction currency; amounts in "
                "different currencies must not be summed directly.",
                "R1: Amounts in different currency_code values must not be summed directly.",
            ],
        },
    }
    # A scenario without questions still shows its cheat sheet.
    assert set(json.loads(rows[1]["success_criteria"])) == {"cheat_sheet"}
    assert rows[0]["answer_text"] == "Here are the top spenders."
    assert rows[1]["question_id"] == "" and rows[1]["scenario_id"] == "SCN_002"
    assert rows[0]["final_verdict"] == ""


def test_columns_follow_the_selected_metrics() -> None:
    columns = columns_for(["relevance", "custom_01", "mrr"])
    assert columns[-13:] == [
        "relevance",
        "relevance_status",
        "relevance_reason",
        "custom_01",
        "custom_01_status",
        "custom_01_reason",
        "mrr",
        "mrr_status",
        "mrr_reason",
        "final_verdict",
        "verdict_reason",
        "bug_ids",
        "evaluation_notes",
    ]
    assert {"product_outcome", "response_classification", "success_criteria"} <= set(columns)
    assert {"tool_calls", "retrieved_context"} <= set(columns)
    for removed in (
        "metric_expectations",
        "legacy_expectations",
        "cheat_sheet",
        "cheat_sheet_rules",
        "golden_case_id",
        "expected_behavior_contract",
        "expected_tools",
        "golden_set_criteria",
        "relevancy",
        "groundedness",
        "llm_verdict",
    ):
        assert removed not in columns


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


def _evaluation(**fields: object) -> EvaluationResult:
    return EvaluationResult.model_validate(
        {
            "evaluation_id": "EVAL_001",
            "question_id": "Q_001",
            "scenario_id": "SCN_001",
            "execution_id": "EXE_001",
            "product_outcome": "answered",
            "mode": "grounded",
            "evaluation_status": "evaluated",
            **fields,
        }
    )


def _execution(response: dict[str, object]) -> ExecutionRecord:
    return ExecutionRecord.model_validate(
        {
            "execution_id": "EXE_001",
            "question_id": "Q_001",
            "product_id": "p",
            "adapter": "fixture",
            "submitted_question": "Top spenders?",
            "started_at": datetime(2026, 10, 14, tzinfo=UTC),
            "response": response,
        }
    )


def test_bullets_and_metric_cells(sample_scenarios: list[Scenario]) -> None:
    assert bullets([]) == "" and bullets(["one"]) == "one"
    assert bullets(["a", "", "b"]) == "• a\n• b"
    question = _question(sample_scenarios[0])
    execution = _execution(
        {
            "outcome": "answered",
            "artifacts": [
                {"kind": "sql", "content": "SELECT 1"},
                {"kind": "tool_calls", "content": [{"name": "aggregate"}]},
            ],
        }
    )
    metric = {"status": "scored", "score": 1.0, "rationale": "ok"}
    evaluation = _evaluation(
        verdict="inconclusive",
        metrics={
            "semantic_correctness": metric,
            "rules_adherence": {"status": "inconclusive", "rationale": "cannot tell"},
            "schema_faithfulness": {**metric, "score": 0.5},
        },
    )
    row = build_rows("RUN_1", sample_scenarios[:1], [question], [execution], [evaluation])[0]
    assert row["artifacts"].startswith("• sql:\nSELECT 1\n• tool_calls:\n[")
    assert [row[m] for m in ("semantic_correctness", "rules_adherence")] == ["1", ""]
    assert [row[f"{m}_status"] for m in ("semantic_correctness", "rules_adherence")] == [
        "scored",
        "inconclusive",
    ]
    assert (row["semantic_correctness_reason"], row["rules_adherence_reason"]) == (
        "ok",
        "cannot tell",
    )
    assert row["schema_faithfulness"] == "0.5"
    assert "relevance" not in row  # only the metrics the run recorded have columns
    assert row["verdict_reason"] == ""  # inconclusive without findings
    parsed = list(csv.DictReader(io.StringIO(to_csv([row]))))
    assert parsed == [row]  # multi-line cells survive the CSV round trip


def test_reason_columns_explain_each_score(sample_scenarios: list[Scenario]) -> None:
    question = _question(sample_scenarios[0])
    execution = _execution({"outcome": "answered", "text": "Ann spent most."})
    scored = {"status": "scored", "score": 1.0, "rationale": "on topic", "evidence": "Ann"}
    metrics = {
        "semantic_correctness": scored,
        "rules_adherence": {**scored, "score": 0.0, "rationale": "wrong join", "evidence": None},
        "schema_faithfulness": {"status": "not_applicable", "rationale": "no pack"},
    }
    finding = {
        "finding_id": "F_001",
        "category": "rules_adherence",
        "description": "joins on the wrong key",
        "expected_behavior": "e",
        "observed_behavior": "o",
        "severity": "high",
        "evidence_status": "supported",
        "source": "llm",
    }
    failed = _evaluation(verdict="fail", metrics=metrics, findings=[finding])
    row = build_rows("RUN_1", sample_scenarios[:1], [question], [execution], [failed])[0]
    assert row["semantic_correctness_reason"] == "on topic - 'Ann'"  # reason + its quote
    assert row["rules_adherence_reason"] == "wrong join"
    assert row["schema_faithfulness_reason"] == "no pack"
    assert row["schema_faithfulness_status"] == "not_applicable"
    assert row["verdict_reason"] == "F_001 [high] rules_adherence: joins on the wrong key"
    clean = _evaluation(verdict="pass", metrics={**metrics, "rules_adherence": scored})
    row = build_rows("RUN_1", sample_scenarios[:1], [question], [execution], [clean])[0]
    assert row["verdict_reason"] == "All scored metrics passed"


def test_no_sql_note_explains_every_metric(sample_scenarios: list[Scenario]) -> None:
    question = _question(sample_scenarios[0])
    timeout = _execution({"outcome": "timeout"})
    recorded = _evaluation(
        product_outcome="timeout",
        verdict=None,
        evaluation_status="recorded_only",
        notes=["No SQL generated."],
    )
    row = build_rows(
        "RUN_1",
        sample_scenarios[:1],
        [question],
        [timeout],
        [recorded],
        metric_ids=["semantic_correctness"],
    )[0]
    assert row["semantic_correctness"] == row["semantic_correctness_status"] == ""  # not judged
    assert row["verdict_reason"] == "No SQL generated."
    assert row["final_verdict"] == "recorded_only"


def test_metric_criteria_and_telemetry_columns(sample_scenarios: list[Scenario]) -> None:
    data = _question(sample_scenarios[0]).model_dump(mode="json")
    data["expectations"]["success_criteria"].update(
        {
            "context_recall": {
                "status": "generated",
                "value": {"required_information": ["Q1 spend"]},
            },
            "tool_correctness": {"status": "cannot_generate", "reason": "no tools"},
        }
    )
    question = QuestionCase.model_validate(data)
    execution = _execution(
        {
            "outcome": "answered",
            "text": "x",
            "tool_trace": [{"order": 1, "name": "aggregate", "arguments": {"source": "c"}}],
        }
    ).model_copy(
        update={
            "context": ExecutionRecord.model_validate(
                {
                    **_execution({"outcome": "answered", "text": "x"}).model_dump(mode="json"),
                    "context": {
                        "availability": "available",
                        "source_mode": "context_api",
                        "chunks": [{"chunk_id": "c1", "text": "t"}],
                    },
                }
            ).context
        }
    )
    row = build_rows("RUN_1", sample_scenarios[:1], [question], [execution])[0]
    assert json.loads(row["success_criteria"]) == {
        "general": ["names the top spenders"],
        "context_recall": {"required_information": ["Q1 spend"]},
        "tool_correctness": {"status": "cannot_generate", "reason": "no tools"},
    }
    assert row["tool_calls"] == '1. aggregate({"source": "c"})'
    assert row["retrieved_context"] == "1 chunk(s) via context_api"
