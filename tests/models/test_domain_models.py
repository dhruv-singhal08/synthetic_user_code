from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from synthetic_user.core.config import load_config
from synthetic_user.core.loaders import load_product
from synthetic_user.models.evaluation import (
    EvaluationResult,
    EvaluationStatus,
    Finding,
    Verdict,
)
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import IndexRelationship, IndexTable, ReferenceIndex
from synthetic_user.models.run import GenerationMode, RunManifest
from synthetic_user.models.scenario import ReferenceCheatSheet, Scenario

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "app.yaml"


def _finding(**overrides: Any) -> dict[str, Any]:
    return {
        "finding_id": "F_001",
        "category": "currency_handling",
        "description": "Summed amounts across currencies.",
        "expected_behavior": "Group or filter by currency_code.",
        "observed_behavior": "SUM(amount) with no currency handling.",
        "severity": "high",
        "evidence_status": "supported",
        "source": "deterministic",
        **overrides,
    }


def _evaluation(**overrides: Any) -> dict[str, Any]:
    return {
        "evaluation_id": "EVAL_001",
        "question_id": "Q_001",
        "scenario_id": "SCN_001",
        "execution_id": "EXE_001",
        "product_outcome": "answered",
        "mode": "grounded",
        "verdict": "fail",
        "evaluation_status": "evaluated",
        **overrides,
    }


def _cheat_sheet(**overrides: Any) -> dict[str, Any]:
    return {
        "tables": ["card_transactions", "employees"],
        "columns": ["card_transactions.amount", "card_transactions.currency_code"],
        "relationships": [
            {
                "source": "card_transactions.employee_id",
                "target": "employees.employee_id",
                "verification": "verified",
            }
        ],
        **overrides,
    }


def _scenario(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "SCN_001",
        "persona_id": "finance_controller",
        "product_id": "sample_nl2sql",
        "mode": "grounded",
        "reference_index_id": "IDX_20261013_120000_0001",
        "title": "Top spenders",
        "business_goal": "Find employees with the highest card spend.",
        "task_type": "ranking",
        "capability": "supported",
        "capability_rationale": "Ranking over one join is supported.",
        "success_criteria": ["lists employees ranked by card spend"],
        "reference_cheat_sheet": _cheat_sheet(),
        "question_guidance": {"intent": "rank employees by spend"},
        "signature": "ranking|card_transactions,employees",
        **overrides,
    }


def _question(**overrides: Any) -> dict[str, Any]:
    return {
        "question_id": "Q_001",
        "scenario_id": "SCN_001",
        "persona_id": "finance_controller",
        "mode": "grounded",
        "reference_snapshot_id": "SNAP_001",
        "utterance": "Who spent the most on cards last month?",
        "intended_goal": "Rank employees by card spend",
        "intent": {"task_type": "ranking", "metrics": ["spend"]},
        "question_characteristics": {"ambiguity_level": "medium"},
        "expectations": {
            "clarification": "conditional",
            "acceptable_behaviors": ["asks which currency", "groups by currency"],
            "success_criteria": ["ranks employees by spend"],
        },
        **overrides,
    }


# --- extra fields rejected ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (Finding, _finding(extra_field=1)),
        (EvaluationResult, _evaluation(comment="x")),
        (Scenario, _scenario(priority=1)),
        (QuestionCase, _question(tone="casual")),
        (ReferenceCheatSheet, _cheat_sheet(joins=[])),
    ],
)
def test_extra_fields_rejected(model: type[BaseModel], payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model.model_validate(payload)


# --- bad enums rejected ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (Finding, _finding(severity="severe")),
        (Finding, _finding(evidence_status="strong")),
        (Finding, _finding(source="human")),
        (EvaluationResult, _evaluation(verdict="passed")),
        (EvaluationResult, _evaluation(product_outcome="partial")),
        (Scenario, _scenario(task_type="forecast")),
        (Scenario, _scenario(capability="maybe")),
        (QuestionCase, _question(expectations={"clarification": "optional"})),
        (Scenario, _scenario(mode="semi_grounded")),
        (
            IndexRelationship,
            {
                "source_table": "a",
                "source_column": "x",
                "target_table": "b",
                "target_column": "y",
                "cardinality": "one_to_few",
            },
        ),
        (
            IndexRelationship,
            {
                "source_table": "a",
                "source_column": "x",
                "target_table": "b",
                "target_column": "y",
                "verification": "confirmed",
            },
        ),
    ],
)
def test_bad_enums_rejected(model: type[BaseModel], payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


# --- happy paths -------------------------------------------------------------------------


def test_valid_payloads_parse() -> None:
    sheet = Scenario.model_validate(_scenario()).reference_cheat_sheet
    assert sheet is not None and sheet.relationships[0].target
    assert QuestionCase.model_validate(_question()).intent.metrics == ["spend"]
    result = EvaluationResult.model_validate(
        _evaluation(
            findings=[_finding(confidence=0.7)],
            bug_candidates=[
                {
                    "bug_id": "BUG_001",
                    "title": "Mixed-currency sum",
                    "description": "Spend summed across currencies.",
                    "reproduction_question": "Who spent the most on cards last month?",
                    "product_id": "sample_nl2sql",
                    "question_id": "Q_001",
                    "scenario_id": "SCN_001",
                    "execution_id": "EXE_001",
                    "finding_ids": ["F_001"],
                }
            ],
        )
    )
    assert result.bug_candidates[0].status == "needs_review"


# --- invariants --------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["denied", "error", "timeout"])
def test_verdict_none_allowed_for_recorded_only_outcomes(outcome: str) -> None:
    result = EvaluationResult.model_validate(
        _evaluation(product_outcome=outcome, verdict=None, evaluation_status="recorded_only")
    )
    assert result.evaluation_status is EvaluationStatus.RECORDED_ONLY


@pytest.mark.parametrize("outcome", ["answered", "clarification"])
def test_verdict_none_rejected_for_judged_outcomes(outcome: str) -> None:
    with pytest.raises(ValidationError, match="verdict may be None only"):
        EvaluationResult.model_validate(
            _evaluation(product_outcome=outcome, verdict=None, evaluation_status="recorded_only")
        )


def test_denied_may_still_carry_a_verdict() -> None:
    result = EvaluationResult.model_validate(_evaluation(product_outcome="denied", verdict="pass"))
    assert result.verdict is Verdict.PASS


def test_status_must_match_verdict_presence() -> None:
    with pytest.raises(ValidationError, match="recorded_only exactly when"):
        EvaluationResult.model_validate(_evaluation(evaluation_status="recorded_only"))


def test_bug_must_reference_known_findings() -> None:
    bug = {
        "bug_id": "BUG_001",
        "title": "t",
        "description": "d",
        "reproduction_question": "q",
        "product_id": "p",
        "question_id": "Q_001",
        "scenario_id": "SCN_001",
        "execution_id": "EXE_001",
        "finding_ids": ["F_009"],
    }
    with pytest.raises(ValidationError, match="unknown findings"):
        EvaluationResult.model_validate(_evaluation(findings=[_finding()], bug_candidates=[bug]))


def test_confidence_bounded() -> None:
    with pytest.raises(ValidationError):
        Finding.model_validate(_finding(confidence=1.5))
    assert Finding.model_validate(_finding()).confidence is None


def test_cheat_sheet_cannot_reference_unlisted_table() -> None:
    with pytest.raises(ValidationError, match="not in its table list"):
        ReferenceCheatSheet.model_validate(_cheat_sheet(columns=["merchants.merchant_name"]))


def test_cheat_sheet_column_must_be_qualified() -> None:
    with pytest.raises(ValidationError):
        ReferenceCheatSheet.model_validate(_cheat_sheet(columns=["amount"]))


def test_index_table_requires_columns_and_purpose() -> None:
    with pytest.raises(ValidationError, match="columns"):
        IndexTable.model_validate({"name": "t", "business_purpose": "p", "columns": []})
    with pytest.raises(ValidationError, match="business_purpose"):
        IndexTable.model_validate({"name": "t", "columns": [{"name": "a"}]})


def test_product_supported_overlap_rejected() -> None:
    with pytest.raises(ValidationError, match="both supported and not_supported_yet"):
        ProductDescription.model_validate(
            {
                "product_id": "p",
                "product_version": "1",
                "name": "n",
                "product_type": "chatbot",
                "description": "d",
                "supported": ["forecasting"],
                "not_supported_yet": ["forecasting"],
            }
        )


def test_persona_requires_typical_tasks() -> None:
    with pytest.raises(ValidationError, match="typical_tasks"):
        Persona.model_validate(
            {
                "id": "p",
                "version": "1",
                "identity": {"name": "n", "role": "r"},
                "expertise": {"domain": "high", "sql": "low"},
                "behavior": {"style": "s"},
                "typical_tasks": [],
                "primary_motivation": "m",
            }
        )


# --- run manifest ------------------------------------------------------------------------


def _manifest(effective_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": "RUN_20261013_090507_a1b2",
        "created_at": datetime(2026, 10, 13, tzinfo=UTC),
        "persona_id": "finance_controller",
        "product_id": "sample_nl2sql",
        "mode": "grounded",
        "catalog_id": "sample_card_spend",
        "catalog_version": "1.0",
        "reference_index_id": "IDX_001",
        "llm_provider": "fake",
        "llm_model": "fake",
        "pipeline_version": "0.1.0",
        "effective_config": effective_config,
        "artifacts": {"manifest": "run_manifest.json"},
    }


def test_manifest_accepts_app_effective_config() -> None:
    cfg = load_config(REPO_CONFIG, env={"GOOGLE_API_KEY": "sk-should-not-appear"})
    effective = cfg.effective_config()
    assert "secrets" not in effective
    manifest = RunManifest.model_validate(_manifest(effective))
    assert "sk-should-not-appear" not in manifest.model_dump_json()


def test_manifest_rejects_secret_keys() -> None:
    with pytest.raises(ValidationError, match=r"llm\.api_key"):
        RunManifest.model_validate(_manifest({"llm": {"api_key": "x"}}))


def test_manifest_run_id_format() -> None:
    payload = _manifest({})
    payload["run_id"] = "run-1"
    with pytest.raises(ValidationError):
        RunManifest.model_validate(payload)


def test_manifest_mode_matches_reference() -> None:
    ungrounded = {
        **_manifest({}),
        "mode": "ungrounded",
        "catalog_id": None,
        "catalog_version": None,
        "reference_index_id": None,
    }
    assert RunManifest.model_validate(ungrounded).mode is GenerationMode.UNGROUNDED
    with pytest.raises(ValidationError, match="mode ungrounded must not have"):
        RunManifest.model_validate({**_manifest({}), "mode": "ungrounded"})
    with pytest.raises(ValidationError, match="mode grounded requires"):
        RunManifest.model_validate({**ungrounded, "mode": "grounded"})
    with pytest.raises(ValidationError, match="go together"):
        RunManifest.model_validate({**_manifest({}), "catalog_id": None})


# --- grounded / ungrounded (v2.1) --------------------------------------------------------


def _ungrounded_scenario(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "mode": "ungrounded",
        "reference_index_id": None,
        "reference_cheat_sheet": None,
    }
    return _scenario(**{**fields, **overrides})


def test_ungrounded_scenario_has_no_cheat_sheet() -> None:
    scenario = Scenario.model_validate(_ungrounded_scenario())
    assert scenario.mode is GenerationMode.UNGROUNDED
    assert scenario.reference_cheat_sheet is None


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (_scenario(reference_cheat_sheet=None), "mode grounded requires a cheat sheet"),
        (_scenario(reference_index_id=None), "mode grounded requires a reference_index_id"),
        (_scenario(mode="ungrounded", reference_index_id=None), "must not have a cheat sheet"),
        (
            _ungrounded_scenario(reference_index_id="IDX_1"),
            "must not have a reference_index_id",
        ),
    ],
)
def test_scenario_cheat_sheet_iff_grounded(payload: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        Scenario.model_validate(payload)


@pytest.mark.parametrize("mode", ["grounded", "ungrounded"])
def test_success_criteria_required_in_both_modes(mode: str) -> None:
    base = _scenario() if mode == "grounded" else _ungrounded_scenario()
    for criteria in ([], None):
        with pytest.raises(ValidationError, match="success_criteria"):
            Scenario.model_validate({**base, "success_criteria": criteria})
    missing = dict(base)
    del missing["success_criteria"]
    with pytest.raises(ValidationError, match="success_criteria"):
        Scenario.model_validate(missing)


def test_question_snapshot_iff_grounded() -> None:
    ungrounded = QuestionCase.model_validate(
        _question(mode="ungrounded", reference_snapshot_id=None)
    )
    assert ungrounded.reference_snapshot_id is None
    with pytest.raises(ValidationError, match="mode grounded requires a reference_snapshot_id"):
        QuestionCase.model_validate(_question(reference_snapshot_id=None))
    with pytest.raises(ValidationError, match="must not have a reference_snapshot_id"):
        QuestionCase.model_validate(_question(mode="ungrounded"))


def test_question_requires_success_criteria() -> None:
    expectations = {"clarification": "conditional", "acceptable_behaviors": ["answers"]}
    with pytest.raises(ValidationError, match="success_criteria empty"):
        QuestionCase.model_validate(_question(expectations=expectations))


_GENERIC = {
    "relevance": "pass",
    "goal_completeness": "pass",
    "persona_fit": "pass",
    "capability_consistency": "not_applicable",
    "clarification_handling": "not_applicable",
    "safety_and_honesty": "pass",
}


def test_evaluation_generic_and_grounded_checks() -> None:
    result = EvaluationResult.model_validate(
        _evaluation(
            verdict="pass",
            checks=_GENERIC,
            grounded_checks={"join_correctness": "pass"},
            plugins_applied=["sql"],
            check_rationale={"relevance": "answers the question", "join_correctness": "ok"},
        )
    )
    assert result.checks is not None and result.checks.relevance == "pass"


def test_ungrounded_evaluation_has_generic_checks_only() -> None:
    ungrounded = _evaluation(mode="ungrounded", verdict="pass", checks=_GENERIC)
    assert EvaluationResult.model_validate(ungrounded).grounded_checks == {}
    with pytest.raises(ValidationError, match="ungrounded evaluations have no grounded_checks"):
        EvaluationResult.model_validate(
            {**ungrounded, "grounded_checks": {"sql_syntax": "pass"}, "plugins_applied": ["sql"]}
        )
    with pytest.raises(ValidationError, match="ungrounded evaluations"):
        EvaluationResult.model_validate({**ungrounded, "cheat_sheet_conflicts": ["x"]})


def test_grounded_checks_need_a_plugin_and_rationale_a_known_check() -> None:
    with pytest.raises(ValidationError, match="plugins_applied is empty"):
        EvaluationResult.model_validate(_evaluation(grounded_checks={"sql_syntax": "fail"}))
    with pytest.raises(ValidationError, match="check_rationale for unknown checks"):
        EvaluationResult.model_validate(_evaluation(check_rationale={"vibes": "good"}))


def test_generic_checks_reject_missing_or_extra_criteria() -> None:
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(
            _evaluation(checks={k: v for k, v in _GENERIC.items() if k != "relevance"})
        )
    with pytest.raises(ValidationError, match="Extra inputs"):
        EvaluationResult.model_validate(_evaluation(checks={**_GENERIC, "sql_syntax": "pass"}))


SAMPLE_CHATBOT = Path(__file__).resolve().parents[2] / "data" / "products" / "sample_chatbot.yaml"


def test_sample_chatbot_is_a_non_sql_product() -> None:
    product = load_product(SAMPLE_CHATBOT)
    assert product.product_type == "chatbot"
    assert product.response_artifacts_expected == ["text"]


def test_product_requires_product_type() -> None:
    with pytest.raises(ValidationError, match="product_type"):
        ProductDescription.model_validate(
            {"product_id": "p", "product_version": "1", "name": "n", "description": "d"}
        )


def test_data_catalog_reference_kind_defaults(sample_index: ReferenceIndex) -> None:
    assert sample_index.reference_kind == "data_catalog"
    with pytest.raises(ValidationError):
        ReferenceIndex.model_validate({**sample_index.model_dump(), "reference_kind": "web_pages"})
