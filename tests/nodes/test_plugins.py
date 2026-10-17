"""C09: artifact plugins (grounded only), the SQL plugin and Stage A analysis."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from synthetic_user.core.config import AppConfig, EvaluationSettings
from synthetic_user.core.exceptions import ConfigError
from synthetic_user.core.loaders import load_persona
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.models.execution import ExecutionRecord
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import ReferenceIndex, ReferenceSnapshot
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator import EvaluationCase, Evaluator
from synthetic_user.nodes.evaluators.plugins.base import ArtifactPlugin
from synthetic_user.nodes.evaluators.plugins.registry import create_plugins
from synthetic_user.nodes.evaluators.plugins.sql.analysis import analyze_sql
from synthetic_user.nodes.evaluators.plugins.sql.plugin import SqlPlugin
from synthetic_user.prompts.sql_evaluation import build_sql_evaluation_prompt
from synthetic_user.references.snapshot import build_snapshot, snapshot_id_for

REPO = Path(__file__).resolve().parents[2]
PERSONA = load_persona(REPO / "data/personas/finance_controller.yaml")
FIXTURE = json.loads((REPO / "tests/fixtures/evaluation/01_correct.json").read_text("utf-8"))
GENERIC_PASS: dict[str, Any] = FIXTURE["llm_responses"][0]
SQL_PASS: dict[str, Any] = FIXTURE["llm_responses"][1]
SQL_RESPONSE: dict[str, Any] = FIXTURE["response"]
LOG = logging.getLogger("t")


def _settings(app_config: AppConfig, plugins: list[str]) -> AppConfig:
    return app_config.model_copy(update={"evaluation": EvaluationSettings(plugins=plugins)})


def _question(scenario: Scenario, mode: str) -> QuestionCase:
    return QuestionCase.model_validate(
        {
            "question_id": "Q_001",
            "scenario_id": scenario.id,
            "persona_id": scenario.persona_id,
            "mode": mode,
            "reference_snapshot_id": snapshot_id_for(scenario.id) if mode == "grounded" else None,
            "utterance": "Who were our top 10 card spenders over the last three months?",
            "intended_goal": scenario.business_goal,
            "intent": {"task_type": "ranking"},
            "question_characteristics": {"ambiguity_level": "medium"},
            "expectations": {
                "clarification": "conditional",
                "acceptable_behaviors": ["ranks employees by spend per currency"],
                "success_criteria": ["ranks employees by spend per currency"],
            },
        }
    )


def _execution(response: dict[str, Any]) -> ExecutionRecord:
    return ExecutionRecord.model_validate(
        {
            "execution_id": "EXE_001",
            "question_id": "Q_001",
            "product_id": "sample_nl2sql",
            "adapter": "fixture",
            "submitted_question": "Who were our top 10 card spenders?",
            "started_at": datetime(2026, 10, 18, tzinfo=UTC),
            "response": response,
        }
    )


def _ungrounded(scenario: Scenario) -> Scenario:
    return scenario.model_copy(
        update={
            "mode": GenerationMode.UNGROUNDED,
            "reference_index_id": None,
            "reference_cheat_sheet": None,
        }
    )


# --- registry -------------------------------------------------------------------------------------


def test_registry_builds_configured_plugins(app_config: AppConfig) -> None:
    plugins = create_plugins(FakeLLMClient(), app_config, LOG)
    assert [p.name for p in plugins] == ["sql"]
    assert all(isinstance(p, ArtifactPlugin) for p in plugins)
    assert create_plugins(FakeLLMClient(), _settings(app_config, []), LOG) == []


@pytest.mark.parametrize(("names", "match"), [(["xml"], "Unknown"), (["sql", "sql"], "once")])
def test_registry_rejects_bad_names(app_config: AppConfig, names: list[str], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        create_plugins(FakeLLMClient(), _settings(app_config, names), LOG)


def test_sql_plugin_applies_only_to_grounded_sql_answers(
    app_config: AppConfig, sample_scenarios: list[Scenario]
) -> None:
    plugin = SqlPlugin(FakeLLMClient(), app_config, LOG)
    grounded = sample_scenarios[0]
    sql = _execution(SQL_RESPONSE).response
    text = _execution({"outcome": "answered", "text": "Alice"}).response
    clarify = _execution({"outcome": "clarification", "message": "Which currency?"}).response
    assert plugin.applies(sql, _question(grounded, "grounded"), grounded)
    assert not plugin.applies(text, _question(grounded, "grounded"), grounded)
    assert not plugin.applies(clarify, _question(grounded, "grounded"), grounded)
    assert not plugin.applies(
        sql, _question(_ungrounded(grounded), "ungrounded"), _ungrounded(grounded)
    )


# --- through the evaluator ------------------------------------------------------------------------


async def test_grounded_sql_answer_runs_sql_plugin_with_analysis(
    app_config: AppConfig,
    sample_index: ReferenceIndex,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
) -> None:
    scenario = sample_scenarios[0]
    assert scenario.reference_cheat_sheet is not None
    snapshot = build_snapshot(
        sample_index, scenario.reference_cheat_sheet, snapshot_id_for(scenario.id)
    )
    llm = FakeLLMClient([GENERIC_PASS, SQL_PASS])
    evaluator = Evaluator(llm, app_config, LOG, index=sample_index)
    case = EvaluationCase(
        _question(scenario, "grounded"), scenario, snapshot, _execution(SQL_RESPONSE)
    )
    result = await evaluator.evaluate(case, sample_product, PERSONA)
    assert result.verdict == "pass" and result.plugins_applied == ["sql"]
    analysis = result.plugin_analysis["sql"]
    assert analysis["parse_ok"] is True and "SUM(c.amount)" in analysis["aggregations"]
    assert analysis["limit"] == "10"
    assert '"date_expressions"' in llm.calls[1].user  # Stage A is handed to Stage B


async def test_ungrounded_sql_product_never_runs_plugins(
    app_config: AppConfig,
    sample_index: ReferenceIndex,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
) -> None:
    scenario = _ungrounded(sample_scenarios[0])
    llm = FakeLLMClient([GENERIC_PASS])
    evaluator = Evaluator(llm, app_config, LOG, index=sample_index)
    case = EvaluationCase(
        _question(scenario, "ungrounded"), scenario, None, _execution(SQL_RESPONSE)
    )
    result = await evaluator.evaluate(case, sample_product, PERSONA)
    assert len(llm.calls) == 1  # generic only
    assert result.plugins_applied == [] and result.grounded_checks == {}
    assert result.plugin_analysis == {} and result.verdict == "pass"


async def test_no_configured_plugin_falls_back_to_reference_coverage(
    app_config: AppConfig,
    sample_index: ReferenceIndex,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
) -> None:
    scenario = sample_scenarios[0]
    assert scenario.reference_cheat_sheet is not None
    snapshot = build_snapshot(
        sample_index, scenario.reference_cheat_sheet, snapshot_id_for(scenario.id)
    )
    coverage = {
        "reference_coverage": {"status": "pass", "rationale": "all covered"},
        "covered": [
            "total_spend",
            "employees.full_name",
            "card_transactions.currency_code",
            "card_transactions.amount",
        ],
        "missing": [],
    }
    llm = FakeLLMClient([GENERIC_PASS, coverage])
    evaluator = Evaluator(llm, _settings(app_config, []), LOG, index=sample_index)
    case = EvaluationCase(
        _question(scenario, "grounded"), scenario, snapshot, _execution(SQL_RESPONSE)
    )
    result = await evaluator.evaluate(case, sample_product, PERSONA)
    assert result.plugins_applied == ["reference_coverage"]
    assert result.grounded_checks == {"reference_coverage": "pass"} and result.verdict == "pass"
    assert "You check whether a product's answer covers" in llm.calls[1].system


# --- Stage A --------------------------------------------------------------------------------------


def test_stage_a_inventory(sample_index: ReferenceIndex) -> None:
    sql = """
        WITH spend AS (
          SELECT c.employee_id, c.currency_code, SUM(c.amount) AS total
          FROM card_transactions AS c
          WHERE c.transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH)
          GROUP BY c.employee_id, c.currency_code
        )
        SELECT e.full_name, s.currency_code, s.total
        FROM spend AS s JOIN employees AS e ON s.employee_id = e.employee_id
        ORDER BY s.total DESC LIMIT 10
    """
    result = analyze_sql(sql, "bigquery", sample_index)
    a = result.analysis
    assert not result.findings and not result.blocks_llm
    assert a.ctes == ["spend"] and a.aliases == {"c": "card_transactions", "e": "employees"}
    assert a.joins[0].equality_pairs == [("s.employee_id", "e.employee_id")]
    assert any("DATE_SUB" in d for d in a.date_expressions)
    assert a.aggregations == ["SUM(c.amount)"]
    assert a.group_by == ["c.employee_id", "c.currency_code"]
    assert a.order_by == ["s.total DESC"] and a.limit == "10"
    assert a.unknown_columns == [] and a.unverifiable_tables == []


def test_stage_a_unknown_vs_unverifiable_and_join_issues(sample_index: ReferenceIndex) -> None:
    unknown = analyze_sql("SELECT c.amount_usd FROM card_transactions c", "bigquery", sample_index)
    assert unknown.analysis.unknown_columns == ["c.amount_usd"]
    assert [f.category for f in unknown.findings] == ["column_selection"]
    unverifiable = analyze_sql("SELECT * FROM budgets", "bigquery", sample_index)
    assert unverifiable.analysis.unverifiable_tables == ["budgets"] and not unverifiable.findings
    join = analyze_sql(
        "SELECT 1 FROM card_transactions c JOIN merchants m ON c.merchant_id = m.merchant_id",
        "bigquery",
        sample_index,
    )
    assert join.analysis.join_issues == [
        "join card_transactions.merchant_id = merchants.merchant_id "
        "has no relationship in the index"
    ]
    wrong = analyze_sql(
        "SELECT 1 FROM card_transactions c JOIN employees e ON c.merchant_id = e.employee_id",
        "bigquery",
        sample_index,
    )
    assert [f.category for f in wrong.findings] == ["join_correctness"]
    assert any("contradicts the verified relationship" in i for i in wrong.analysis.join_issues)


def test_stage_a_parse_failure_and_non_select(sample_index: ReferenceIndex) -> None:
    broken = analyze_sql("SELEC 1", "bigquery", sample_index)
    assert broken.blocks_llm and broken.analysis.parse_ok is False and broken.analysis.parse_errors
    delete = analyze_sql("DELETE FROM card_transactions", "bigquery", sample_index)
    assert delete.blocks_llm and delete.non_select == "Delete"


def test_sql_prompt_allows_equivalent_sql_and_includes_analysis(
    sample_index: ReferenceIndex,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
) -> None:
    scenario = sample_scenarios[0]
    assert scenario.reference_cheat_sheet is not None
    snapshot: ReferenceSnapshot = build_snapshot(
        sample_index, scenario.reference_cheat_sheet, snapshot_id_for(scenario.id)
    )
    analysis = analyze_sql(
        "SELECT amount FROM card_transactions", "bigquery", sample_index
    ).analysis
    _system, user = build_sql_evaluation_prompt(
        _question(scenario, "grounded"), scenario, snapshot, sample_product, "SELECT 1", analysis
    )
    assert "NOT required" in user and "equivalent SQL" in user
    assert 'Do not report a "missing join"' in user
    assert "cheat_sheet_conflicts" in user and '"parse_ok": true' in user
