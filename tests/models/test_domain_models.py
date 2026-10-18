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
from synthetic_user.models.reference import (
    Reference,
    ReferencePack,
    ReferencePackDraft,
    normalise_quote,
)
from synthetic_user.models.run import GenerationMode, RunManifest
from synthetic_user.models.scenario import Scenario

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


def _pack(**overrides: Any) -> dict[str, Any]:
    return {
        "pack_id": "RP_SCN_001",
        "scenario_id": "SCN_001",
        "reference_id": "catalog-1234abcd",
        "core_refs": [
            {"name": "card_transactions.amount", "quote": "Transaction amount", "why": "spend"}
        ],
        "rules": [{"rule": "no cross-currency sums", "quote": "must not be summed"}],
        **overrides,
    }


def _scenario(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "SCN_001",
        "persona_id": "finance_controller",
        "product_id": "sample_nl2sql",
        "mode": "grounded",
        "reference_id": "catalog-1234abcd",
        "reference_pack_id": "RP_SCN_001",
        "title": "Top spenders",
        "business_goal": "Find employees with the highest card spend.",
        "task_type": "ranking",
        "capability": "supported",
        "capability_rationale": "Ranking over one join is supported.",
        "success_criteria": ["lists employees ranked by card spend"],
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
        "reference_pack_id": "RP_SCN_001",
        "utterance": "Who spent the most on cards last month?",
        "intended_goal": "Rank employees by card spend",
        "intent": {"task_type": "ranking", "metrics": ["spend"]},
        "question_characteristics": {"ambiguity_level": "medium"},
        "expectations": {
            "expected_behavior": "Rank employees by card spend, per currency.",
            "success_criteria": ["ranks employees by spend"],
            "acceptable_variations": ["asks which currency"],
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
        (ReferencePack, _pack(tables=[])),
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
        (Scenario, _scenario(mode="semi_grounded")),
        (
            Reference,
            {"reference_id": "r", "source_name": "r.pdf", "format": "pdf", "text": "x"},
        ),
    ],
)
def test_bad_enums_rejected(model: type[BaseModel], payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


# --- happy paths -------------------------------------------------------------------------


def test_valid_payloads_parse() -> None:
    assert Scenario.model_validate(_scenario()).reference_pack_id == "RP_SCN_001"
    assert ReferencePack.model_validate(_pack()).rules[0].quote == "must not be summed"
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


def test_pack_needs_a_core_ref_and_a_well_formed_id() -> None:
    with pytest.raises(ValidationError, match="core_refs"):
        ReferencePack.model_validate(_pack(core_refs=[]))
    with pytest.raises(ValidationError, match="core_refs"):
        ReferencePackDraft.model_validate({"core_refs": []})
    with pytest.raises(ValidationError, match="pack_id"):
        ReferencePack.model_validate(_pack(pack_id="PACK_1"))


def test_reference_contains_is_whitespace_case_and_quote_mark_insensitive() -> None:
    reference = Reference(
        reference_id="r-1",
        source_name="policy.md",
        format="markdown",
        text="## Hotels\nThe nightly hotel limit is USD 250 in Tier 1 cities\n(New York).",
    )
    assert reference.contains("the NIGHTLY hotel limit is usd 250 in tier 1 cities (new york)")
    assert reference.contains('"The nightly hotel limit is USD 250."')
    assert not reference.contains("The nightly hotel limit is USD 300")
    assert not reference.contains("   ")
    assert normalise_quote(" \u201cA  b.\u201d ") == "a b"


def test_reference_guide_falls_back_to_default() -> None:
    reference = Reference(reference_id="r", source_name="r.txt", format="text", text="x")
    assert "not exhaustive" in reference.guide_text()
    assert (
        Reference.model_validate(
            {**reference.model_dump(), "guide": "Read it as a policy."}
        ).guide_text()
        == "Read it as a policy."
    )


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
        "reference_id": "sample_card_spend_index-1234abcd",
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
    ungrounded = {**_manifest({}), "mode": "ungrounded", "reference_id": None}
    assert RunManifest.model_validate(ungrounded).mode is GenerationMode.UNGROUNDED
    with pytest.raises(ValidationError, match="mode ungrounded must not have"):
        RunManifest.model_validate({**_manifest({}), "mode": "ungrounded"})
    with pytest.raises(ValidationError, match="mode grounded requires"):
        RunManifest.model_validate({**ungrounded, "mode": "grounded"})


# --- grounded / ungrounded (v2.1) --------------------------------------------------------


def _ungrounded_scenario(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "mode": "ungrounded",
        "reference_id": None,
        "reference_pack_id": None,
    }
    return _scenario(**{**fields, **overrides})


def test_ungrounded_scenario_has_no_reference() -> None:
    scenario = Scenario.model_validate(_ungrounded_scenario())
    assert scenario.mode is GenerationMode.UNGROUNDED
    assert scenario.reference_pack_id is None


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (_scenario(reference_pack_id=None), "mode grounded requires a reference_pack_id"),
        (_scenario(reference_id=None), "mode grounded requires a reference_id"),
        (_scenario(mode="ungrounded", reference_id=None), "must not have a reference_pack_id"),
        (_ungrounded_scenario(reference_id="r-1"), "must not have a reference_id"),
    ],
)
def test_scenario_reference_iff_grounded(payload: dict[str, Any], match: str) -> None:
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


def test_question_pack_iff_grounded() -> None:
    ungrounded = QuestionCase.model_validate(_question(mode="ungrounded", reference_pack_id=None))
    assert ungrounded.reference_pack_id is None
    with pytest.raises(ValidationError, match="mode grounded requires a reference_pack_id"):
        QuestionCase.model_validate(_question(reference_pack_id=None))
    with pytest.raises(ValidationError, match="must not have a reference_pack_id"):
        QuestionCase.model_validate(_question(mode="ungrounded"))


@pytest.mark.parametrize(
    "expectations",
    [
        {"expected_behavior": "answers"},  # criteria required
        {"expected_behavior": "answers", "success_criteria": []},
        {"expected_behavior": "answers", "success_criteria": ["a", "b", "c", "d", "e"]},  # max 4
        {"success_criteria": ["a"]},  # expected_behavior required
        {"expected_behavior": "answers", "success_criteria": ["a"], "clarification": "x"},
        {"expected_behavior": "a", "success_criteria": ["a"], "unacceptable_behaviors": ["b"]},
    ],
)
def test_question_expectations_shape(expectations: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        QuestionCase.model_validate(_question(expectations=expectations))


_GENERIC = {
    "relevance": "pass",
    "goal_completeness": "pass",
    "persona_fit": "pass",
    "capability_consistency": "not_applicable",
    "clarification_handling": "not_applicable",
    "safety_and_honesty": "pass",
}


def test_evaluation_checks_include_reference_alignment() -> None:
    result = EvaluationResult.model_validate(
        _evaluation(
            verdict="pass",
            checks={**_GENERIC, "reference_alignment": "pass"},
            check_rationale={"relevance": "answers the question", "reference_alignment": "ok"},
        )
    )
    assert result.checks is not None and result.checks.reference_alignment == "pass"
    # absent -> not_applicable (e.g. evaluations stored without it)
    default = EvaluationResult.model_validate(_evaluation(verdict="pass", checks=_GENERIC))
    assert default.checks is not None and default.checks.reference_alignment == "not_applicable"


def test_ungrounded_evaluation_has_no_reference_alignment() -> None:
    ungrounded = _evaluation(mode="ungrounded", verdict="pass", checks=_GENERIC)
    assert EvaluationResult.model_validate(ungrounded).mode is GenerationMode.UNGROUNDED
    with pytest.raises(ValidationError, match="ungrounded evaluations have no reference_alignment"):
        EvaluationResult.model_validate(
            {**ungrounded, "checks": {**_GENERIC, "reference_alignment": "pass"}}
        )


def test_rationale_must_name_a_known_check_and_v2_fields_are_gone() -> None:
    with pytest.raises(ValidationError, match="check_rationale for unknown checks"):
        EvaluationResult.model_validate(_evaluation(check_rationale={"vibes": "good"}))
    for field in ("grounded_checks", "plugins_applied", "plugin_analysis"):
        with pytest.raises(ValidationError, match="Extra inputs"):
            EvaluationResult.model_validate(_evaluation(**{field: {}}))


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
