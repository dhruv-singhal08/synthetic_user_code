from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import ArtifactIntegrityError
from synthetic_user.core.identifiers import SequentialIdGenerator
from synthetic_user.core.loaders import load_persona
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.models.evaluation import (
    CheckStatus,
    EvidenceStatus,
    Finding,
    FindingSource,
    Severity,
    Verdict,
)
from synthetic_user.models.execution import ExecutionRecord
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import ReferenceIndex, ReferenceSnapshot
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator import EvaluationCase, Evaluator
from synthetic_user.nodes.evaluators.plugins.base import matches_known_gap
from synthetic_user.nodes.evaluators.plugins.sql.analysis import analyze_sql
from synthetic_user.nodes.evaluators.plugins.sql.models import SqlChecks
from synthetic_user.nodes.evaluators.plugins.sql.plugin import final_verdict
from synthetic_user.references.snapshot import build_snapshot, snapshot_id_for

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "evaluation"
CASES = sorted(FIXTURES.glob("*.json"))


@pytest.fixture
def question(sample_scenarios: list[Scenario]) -> QuestionCase:
    scenario = sample_scenarios[0]
    return QuestionCase.model_validate(
        {
            "question_id": "Q_001",
            "scenario_id": scenario.id,
            "persona_id": scenario.persona_id,
            "mode": "grounded",
            "reference_snapshot_id": snapshot_id_for(scenario.id),
            "utterance": "Who were our top 10 card spenders over the last three months?",
            "intended_goal": scenario.business_goal,
            "intent": {"task_type": "ranking", "metrics": ["spend"], "time_range": "last 3 months"},
            "question_characteristics": {"style": "precise", "ambiguity_level": "medium"},
            "expectations": {
                "clarification": "conditional",
                "acceptable_behaviors": ["ranks employees by spend per currency"],
                "success_criteria": ["ranks employees by spend per currency"],
            },
        }
    )


@pytest.fixture
def snapshot(sample_index: ReferenceIndex, sample_scenarios: list[Scenario]) -> ReferenceSnapshot:
    scenario = sample_scenarios[0]
    return build_snapshot(
        sample_index, scenario.reference_cheat_sheet, snapshot_id_for(scenario.id)
    )


def _execution(response: dict[str, Any], execution_id: str = "EXE_001") -> ExecutionRecord:
    return ExecutionRecord.model_validate(
        {
            "execution_id": execution_id,
            "question_id": "Q_001",
            "product_id": "sample_nl2sql",
            "adapter": "fixture",
            "response": response,
            "submitted_question": "Who were our top 10 card spenders over the last three months?",
            "started_at": datetime(2026, 10, 13, tzinfo=UTC),
        }
    )


PERSONA = load_persona(
    Path(__file__).resolve().parents[2] / "data/personas/finance_controller.yaml"
)
# The generic judgement every answered case gets first (all checks pass, criterion met).
GENERIC_PASS: dict[str, Any] = json.loads(
    (FIXTURES / "01_correct.json").read_text(encoding="utf-8")
)["llm_responses"][0]


class _Shim:
    """Evaluator with the old (question, scenario, snapshot, product, execution) call shape."""

    def __init__(self, evaluator: Evaluator) -> None:
        self.inner = evaluator

    async def evaluate(
        self,
        question: QuestionCase,
        scenario: Scenario,
        snapshot: ReferenceSnapshot | None,
        product: ProductDescription,
        execution: ExecutionRecord,
    ) -> Any:
        case = EvaluationCase(question, scenario, snapshot, execution)
        return await self.inner.evaluate(case, product, PERSONA)

    async def evaluate_many(self, cases: list[EvaluationCase], product: ProductDescription) -> Any:
        return await self.inner.evaluate_many(cases, product, PERSONA)


def _evaluator(llm: FakeLLMClient, settings: AppConfig, catalog: ReferenceIndex) -> _Shim:
    return _Shim(
        Evaluator(
            llm,
            settings,
            logging.getLogger("test"),
            index=catalog,
            id_generator=SequentialIdGenerator(),
        )
    )


def test_fixture_set_is_complete() -> None:
    names = {json.loads(p.read_text(encoding="utf-8"))["name"] for p in CASES}
    required = {
        "correct",
        "wrong_aggregation",
        "missing_date_filter",
        "wrong_join",
        "wrong_metric_column",
        "wrong_grouping",
        "equivalent_sql",
        "cte",
        "ambiguous_but_reasonable",
        "missing_metadata_inconclusive",
        "relevant_clarification",
        "unnecessary_clarification",
        "syntax_error",
        "denied_recorded_only",
        "error_recorded_only",
        "timeout_recorded_only",
    }
    assert required <= names


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
async def test_evaluation_fixture(
    path: Path,
    app_config: AppConfig,
    sample_index: ReferenceIndex,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
    question: QuestionCase,
    snapshot: ReferenceSnapshot,
) -> None:
    case = json.loads(path.read_text(encoding="utf-8"))
    expected = case["expected"]
    llm = FakeLLMClient(case["llm_responses"])
    result = await _evaluator(llm, app_config, sample_index).evaluate(
        question, sample_scenarios[0], snapshot, sample_product, _execution(case["response"])
    )

    assert result.verdict == expected["verdict"], result.model_dump_json(indent=2)
    assert result.evaluation_status == expected["evaluation_status"]
    assert len(llm.calls) == expected["llm_calls"]
    assert sorted({f.category for f in result.findings}) == sorted(expected["finding_categories"])
    assert len(result.bug_candidates) == expected["bug_candidates"]
    if "llm_verdict" in expected:
        assert result.llm_verdict == expected["llm_verdict"]
    if "matches_known_gap" in expected:
        assert all(f.matches_known_gap == expected["matches_known_gap"] for f in result.findings)
    if "notes_contain" in expected:
        assert any(expected["notes_contain"] in n for n in result.notes)

    # Invariants for every case.
    assert result.evaluation_id == "EVAL_001"
    outcome = case["response"]["outcome"]
    if outcome in ("error", "timeout"):
        assert not result.findings and result.checks is None
    else:  # generic checks are filled for answered / clarification / denied
        assert result.checks is not None and result.check_rationale
    assert result.mode == "grounded"
    if outcome == "answered":
        assert result.plugins_applied == ["sql"]
        assert set(result.grounded_checks) == set(SqlChecks.model_fields)
    else:
        assert not result.plugins_applied and not result.grounded_checks
    for bug in result.bug_candidates:
        assert bug.bug_id == "BUG_001"
        assert bug.status == "needs_review"
        assert (bug.question_id, bug.scenario_id, bug.execution_id) == (
            "Q_001",
            "SCN_001",
            "EXE_001",
        )
        assert bug.reproduction_question == question.utterance
        assert set(bug.finding_ids) <= {f.finding_id for f in result.findings}
    for finding in result.findings:
        if finding.category in {"syntax_error", "non_select_statement", "join_correctness"} and (
            expected["llm_calls"] == 0 or case["name"] == "wrong_join"
        ):
            assert finding.source is FindingSource.DETERMINISTIC


# --- deterministic rules and helpers --------------------------------------------------------


def test_non_select_finding_is_critical(sample_index: ReferenceIndex) -> None:
    assert (
        analyze_sql("DELETE FROM card_transactions", "bigquery", sample_index).non_select
        == "Delete"
    )


def test_unverifiable_tables_listed(sample_index: ReferenceIndex) -> None:
    rules = analyze_sql(
        "SELECT * FROM budgets JOIN card_transactions USING (x)", "bigquery", sample_index
    )
    assert rules.unverifiable_tables == ["budgets"]
    assert not rules.findings


def test_unqualified_unknown_column_on_single_catalog_table(
    sample_index: ReferenceIndex,
) -> None:
    rules = analyze_sql(
        "SELECT SUM(amount_usd) AS total FROM card_transactions ORDER BY total",
        "bigquery",
        sample_index,
    )
    assert [f.sql_evidence for f in rules.findings] == ["amount_usd"]


def test_correct_verified_join_has_no_findings(sample_index: ReferenceIndex) -> None:
    rules = analyze_sql(
        "SELECT e.full_name FROM employees e "
        "JOIN card_transactions c ON e.employee_id = c.employee_id",
        "bigquery",
        sample_index,
    )
    assert rules.findings == []


def test_unverified_relationship_join_not_flagged(sample_index: ReferenceIndex) -> None:
    rules = analyze_sql(
        "SELECT d.department_name FROM employees e "
        "JOIN departments d ON e.full_name = d.department_name",
        "bigquery",
        sample_index,
    )
    assert rules.findings == []


def _f(category: str, status: EvidenceStatus) -> Finding:
    return Finding(
        finding_id="F_001",
        category=category,
        description="d",
        expected_behavior="e",
        observed_behavior="o",
        severity=Severity.HIGH,
        evidence_status=status,
        source=FindingSource.LLM,
    )


def _checks(**over: CheckStatus) -> SqlChecks:
    values = dict.fromkeys(SqlChecks.model_fields, CheckStatus.PASS)
    values.update(over)
    return SqlChecks.model_validate(values)


def test_final_verdict_rules() -> None:
    fail = _checks(join_correctness=CheckStatus.FAIL)
    assert final_verdict(fail, [_f("join_correctness", EvidenceStatus.SUPPORTED)]) is Verdict.FAIL
    assert (
        final_verdict(fail, [_f("join_correctness", EvidenceStatus.WEAK)]) is Verdict.INCONCLUSIVE
    )
    assert final_verdict(_checks(sql_syntax=CheckStatus.INCONCLUSIVE), []) is Verdict.INCONCLUSIVE
    assert final_verdict(_checks(join_correctness=CheckStatus.NOT_APPLICABLE), []) is Verdict.PASS


def test_matches_known_gap(sample_product: ProductDescription) -> None:
    gaps = sample_product.known_gaps
    assert matches_known_gap("SUM(amount) mixes currencies; currency_code ignored", gaps)
    assert not matches_known_gap("counts rows instead of summing", gaps)


async def test_llm_failure_gives_inconclusive_with_note(
    app_config: AppConfig,
    sample_index: ReferenceIndex,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
    question: QuestionCase,
    snapshot: ReferenceSnapshot,
) -> None:
    bad = {"checks": {}, "verdict": "pass"}
    llm = FakeLLMClient([GENERIC_PASS, bad, bad, bad])
    result = await _evaluator(llm, app_config, sample_index).evaluate(
        question,
        sample_scenarios[0],
        snapshot,
        sample_product,
        _execution(
            {
                "outcome": "answered",
                "artifacts": [{"kind": "sql", "content": "SELECT e.full_name FROM employees e"}],
            }
        ),
    )
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.llm_verdict is Verdict.PASS  # only the generic judge stated an opinion
    assert "SQL evaluation failed after 3 attempt(s)" in result.notes[0]


async def test_llm_fail_without_evidence_is_retried(
    app_config: AppConfig,
    sample_index: ReferenceIndex,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
    question: QuestionCase,
    snapshot: ReferenceSnapshot,
) -> None:
    all_pass = dict.fromkeys(SqlChecks.model_fields, "pass")
    no_evidence = {
        "checks": {**all_pass, "filter_correctness": "fail"},
        "findings": [],
        "verdict": "fail",
    }
    good = {"checks": all_pass, "findings": [], "verdict": "pass"}
    llm = FakeLLMClient([GENERIC_PASS, no_evidence, good])
    result = await _evaluator(llm, app_config, sample_index).evaluate(
        question,
        sample_scenarios[0],
        snapshot,
        sample_product,
        _execution(
            {
                "outcome": "answered",
                "artifacts": [{"kind": "sql", "content": "SELECT e.full_name FROM employees e"}],
            }
        ),
    )
    assert len(llm.calls) == 3
    assert "check filter_correctness is fail" in llm.calls[2].user
    assert result.verdict is Verdict.PASS


async def test_ids_stable_across_many(
    app_config: AppConfig,
    sample_index: ReferenceIndex,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
    question: QuestionCase,
    snapshot: ReferenceSnapshot,
) -> None:
    cases = [
        EvaluationCase(
            question,
            sample_scenarios[0],
            snapshot,
            _execution(
                {"outcome": "answered", "artifacts": [{"kind": "sql", "content": "SELEC"}]},
                "EXE_001",
            ),
        ),
        EvaluationCase(
            question, sample_scenarios[0], snapshot, _execution({"outcome": "timeout"}, "EXE_002")
        ),
        EvaluationCase(
            question,
            sample_scenarios[0],
            snapshot,
            _execution(
                {
                    "outcome": "answered",
                    "artifacts": [{"kind": "sql", "content": "DELETE FROM card_transactions"}],
                },
                "EXE_003",
            ),
        ),
    ]
    llm = FakeLLMClient([GENERIC_PASS, GENERIC_PASS])
    results = await _evaluator(llm, app_config, sample_index).evaluate_many(cases, sample_product)
    assert [r.evaluation_id for r in results] == ["EVAL_001", "EVAL_002", "EVAL_003"]
    assert [f.finding_id for r in results for f in r.findings] == ["F_001", "F_002"]
    assert [b.bug_id for r in results for b in r.bug_candidates] == ["BUG_001", "BUG_002"]


async def test_mismatched_inputs_rejected(
    app_config: AppConfig,
    sample_index: ReferenceIndex,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
    question: QuestionCase,
    snapshot: ReferenceSnapshot,
) -> None:
    with pytest.raises(ArtifactIntegrityError, match="belongs to SCN_001"):
        await _evaluator(FakeLLMClient(), app_config, sample_index).evaluate(
            question,
            sample_scenarios[1],
            snapshot,
            sample_product,
            _execution({"outcome": "timeout"}),
        )


async def test_grounded_answer_without_plugin_gets_reference_coverage(
    app_config: AppConfig,
    sample_index: ReferenceIndex,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
    question: QuestionCase,
    snapshot: ReferenceSnapshot,
) -> None:
    """Grounded, but no artifact plugin applies (text answer): reference_coverage runs."""
    coverage = {
        "reference_coverage": {"status": "fail", "rationale": "No currency split."},
        "covered": ["total_spend", "employees.full_name", "card_transactions.amount"],
        "missing": ["card_transactions.currency_code"],
        "findings": [
            {
                "category": "reference_coverage",
                "description": "Spend is not split by currency.",
                "expected_behavior": "spend per currency",
                "observed_behavior": "one total across currencies",
                "evidence": "Alice spent the most",
                "severity": "medium",
                "evidence_status": "supported",
            }
        ],
    }
    llm = FakeLLMClient([GENERIC_PASS, coverage])
    result = await _evaluator(llm, app_config, sample_index).evaluate(
        question,
        sample_scenarios[0],
        snapshot,
        sample_product,
        _execution({"outcome": "answered", "text": "Alice spent the most, per currency."}),
    )
    assert len(llm.calls) == 2
    assert result.plugins_applied == ["reference_coverage"]
    assert result.grounded_checks == {"reference_coverage": "fail"}
    assert result.plugin_analysis["reference_coverage"]["missing"] == [
        "card_transactions.currency_code"
    ]
    assert result.check_rationale["reference_coverage"] == "No currency split."
    assert result.verdict is Verdict.FAIL and len(result.bug_candidates) == 1
