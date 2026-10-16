from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from synthetic_user.core.config import load_config
from synthetic_user.models.evaluation import (
    EvaluationResult,
    EvaluationStatus,
    Finding,
    Verdict,
)
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import Relationship, TableMetadata
from synthetic_user.models.run import RunManifest
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
        "product_outcome": "sql_generated",
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
        "reference_index_id": "IDX_20261013_120000_0001",
        "title": "Top spenders",
        "business_goal": "Find employees with the highest card spend.",
        "task_type": "ranking",
        "capability": "supported",
        "capability_rationale": "Ranking over one join is supported.",
        "cheat_sheet": _cheat_sheet(),
        "question_guidance": {"intent": "rank employees by spend"},
        "signature": "ranking|card_transactions,employees",
        **overrides,
    }


def _question(**overrides: Any) -> dict[str, Any]:
    return {
        "question_id": "Q_001",
        "scenario_id": "SCN_001",
        "persona_id": "finance_controller",
        "reference_snapshot_id": "SNAP_001",
        "utterance": "Who spent the most on cards last month?",
        "intended_goal": "Rank employees by card spend",
        "intent": {"task_type": "ranking", "metrics": ["spend"]},
        "question_characteristics": {"ambiguity_level": "medium"},
        "expectations": {
            "clarification": "conditional",
            "acceptable_behaviors": ["asks which currency", "groups by currency"],
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
        (
            Relationship,
            {
                "source_table": "a",
                "source_column": "x",
                "target_table": "b",
                "target_column": "y",
                "cardinality": "one_to_few",
            },
        ),
        (
            Relationship,
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
    assert Scenario.model_validate(_scenario()).cheat_sheet.relationships[0].target
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


@pytest.mark.parametrize("outcome", ["sql_generated", "clarification"])
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


def test_table_duplicate_columns_and_bad_primary_key() -> None:
    with pytest.raises(ValidationError, match="duplicate column names"):
        TableMetadata.model_validate({"name": "t", "columns": [{"name": "a"}, {"name": "a"}]})
    with pytest.raises(ValidationError, match="primary_key"):
        TableMetadata.model_validate(
            {"name": "t", "primary_key": ["id"], "columns": [{"name": "a"}]}
        )


def test_product_supported_overlap_rejected() -> None:
    with pytest.raises(ValidationError, match="both supported and not_supported_yet"):
        ProductDescription.model_validate(
            {
                "product_id": "p",
                "product_version": "1",
                "spec_version": "1",
                "name": "n",
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
