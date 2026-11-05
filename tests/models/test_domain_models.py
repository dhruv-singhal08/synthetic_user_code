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
    MetricStatus,
    Verdict,
)
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import (
    DEFAULT_INSTRUCTIONS,
    CatalogPackDraft,
    Reference,
    ReferenceKind,
    ReferencePack,
    normalise_quote,
    quote_in,
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
        "schema_version": 2,
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
        "kind": "data_catalog",
        "tables": [
            {
                "name": "card_transactions",
                "columns": [{"name": "amount", "description": "Transaction amount"}],
                "why": "spend",
            }
        ],
        "rules": [{"id": "R1", "rule": "Amounts must not be summed across currencies."}],
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
        "question_intent": "rank employees by spend",
        "signature": "ranking|card_transactions,employees",
        **overrides,
    }


PACK: dict[str, Any] = {
    "pack_id": "RP_SCN_001",
    "scenario_id": "SCN_001",
    "reference_id": "catalog-0123abcd",
    "kind": "data_catalog",
    "tables": [
        {
            "name": "card_transactions",
            "why": "spend",
            "rules": ["Keep currencies separate."],
            "columns": [{"name": "amount"}],
        }
    ],
    "rules": [{"id": "R1", "rule": "Never sum across currencies."}],
}
SPEC = {"pack_id": "RP_SCN_001", "reference_id": "catalog-0123abcd", "pack": PACK}


def _context(**overrides: Any) -> dict[str, Any]:
    return {
        "success_criteria": [{"id": "C1", "requirement": "ranks employees by spend"}],
        "reference_spec": SPEC,
        "metric_mapping": {},
        **overrides,
    }


def _question(**overrides: Any) -> dict[str, Any]:
    grounded = overrides.get("reference_pack_id", "RP_SCN_001") is not None
    return {
        "question_id": "Q_001",
        "scenario_id": "SCN_001",
        "persona_id": "finance_controller",
        "mode": "grounded",
        "reference_pack_id": "RP_SCN_001",
        "utterance": "Who spent the most on cards last month?",
        "intended_goal": "Rank employees by card spend",
        "intent": {"task_type": "ranking"},
        "question_characteristics": {
            "style": "precise",
            "ambiguity_level": "medium",
            "complexity": "simple",
        },
        "evaluation_context": _context(reference_spec=SPEC if grounded else None),
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
        (ReferencePack, _pack(note="x")),
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
    assert ReferencePack.model_validate(_pack()).rules[0].id == "R1"
    assert QuestionCase.model_validate(_question()).intent.task_type == "ranking"
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


@pytest.mark.parametrize("outcome", ["clarification", "error", "timeout"])
def test_non_answers_are_recorded_only(outcome: str) -> None:
    result = EvaluationResult.model_validate(
        _evaluation(product_outcome=outcome, verdict=None, evaluation_status="recorded_only")
    )
    assert result.evaluation_status is EvaluationStatus.RECORDED_ONLY
    with pytest.raises(ValidationError, match="recorded only"):
        EvaluationResult.model_validate(_evaluation(product_outcome=outcome, verdict="pass"))


def test_substantive_denials_are_judged_but_legacy_denials_stay_readable() -> None:
    denied = EvaluationResult.model_validate(
        _evaluation(
            product_outcome="denied",
            verdict="pass",
            response_classification="unable_to_answer",
            classification_source="typed_outcome",
        )
    )
    assert denied.verdict is Verdict.PASS
    with pytest.raises(ValidationError, match="evaluable denied response needs a verdict"):
        EvaluationResult.model_validate(
            _evaluation(product_outcome="denied", verdict=None, evaluation_status="recorded_only")
        )
    legacy = {
        k: v
        for k, v in _evaluation(
            product_outcome="denied", verdict=None, evaluation_status="recorded_only"
        ).items()
        if k != "schema_version"
    }
    old = EvaluationResult.model_validate(legacy)  # schema 1: denials were recorded only
    assert old.schema_version == 1 and old.verdict is None


def test_answers_need_a_verdict() -> None:
    with pytest.raises(ValidationError, match="an evaluable answered response needs a verdict"):
        EvaluationResult.model_validate(
            _evaluation(verdict=None, evaluation_status="recorded_only")
        )
    assert EvaluationResult.model_validate(_evaluation(verdict="pass")).verdict is Verdict.PASS


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


def test_pack_fits_its_kind_and_has_a_well_formed_id() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        ReferencePack.model_validate(_pack(tables=[]))
    with pytest.raises(ValidationError, match="documents only"):
        ReferencePack.model_validate(
            _pack(kind="document_collection", documents=[{"name": "d", "summary": "s", "why": "w"}])
        )
    with pytest.raises(ValidationError, match="tables"):
        CatalogPackDraft.model_validate({"tables": []})
    with pytest.raises(ValidationError, match="pack_id"):
        ReferencePack.model_validate(_pack(pack_id="PACK_1"))


def test_quote_in_is_whitespace_case_and_quote_mark_insensitive() -> None:
    text = "## Hotels\nThe nightly hotel limit is USD 250 in Tier 1 cities\n(New York)."
    assert quote_in("the NIGHTLY hotel limit is usd 250 in tier 1 cities (new york)", text)
    assert quote_in('"The nightly hotel limit is USD 250."', text)
    assert not quote_in("The nightly hotel limit is USD 300", text)
    assert not quote_in("   ", text)
    assert normalise_quote(" \u201cA  b.\u201d ") == "a b"


def test_reference_instructions_fall_back_to_the_default_of_its_kind() -> None:
    reference = Reference(
        reference_id="r",
        source_name="r.yaml",
        content={
            "kind": "document_collection",
            "name": "KB",
            "documents": [{"name": "d", "summary": "s"}],
        },
    )
    assert reference.instructions_text() == DEFAULT_INSTRUCTIONS[ReferenceKind.DOCUMENT_COLLECTION]
    guided = reference.model_copy(update={"instructions": "Read it as a policy."})
    assert guided.instructions_text() == "Read it as a policy."


def test_product_supported_overlap_rejected() -> None:
    with pytest.raises(ValidationError, match="both supported and not_supported"):
        ProductDescription.model_validate(
            {
                "product_id": "p",
                "product_version": "1",
                "name": "n",
                "description": "d",
                "supported": ["forecasting"],
                "not_supported": ["forecasting"],
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
    with pytest.raises(ValidationError, match="must not have a pack"):
        QuestionCase.model_validate(_question(mode="ungrounded"))


def test_profile_era_artifacts_stay_readable() -> None:
    """Agent-profile scenarios / questions were grounded without a pack and carried the profile
    and golden ids: readable, with the profile dropped (generation enforces packs now)."""
    scenario = Scenario.model_validate(
        _scenario(reference_pack_id=None, evaluation_profile="agent", golden_case_id=None)
    )
    assert scenario.reference_pack_id is None
    question = QuestionCase.model_validate(
        _question(
            reference_pack_id=None,
            evaluation_profile="agent",
            golden_case_id="GOLD_001",
        )
    )
    assert question.reference_pack_id is None


def test_evaluation_context_holds_criteria_reference_spec_and_mapping() -> None:
    """reforming Part 4: success criteria (C ids), the scenario's pack as Reference Spec and
    one Metric Mapping entry per metric (criteria, reference requirements, basis, expectation)."""
    question = QuestionCase.model_validate(
        _question(
            evaluation_context=_context(
                success_criteria=[
                    {"id": "C1", "requirement": "ranks employees by spend"},
                    {"id": "C2", "requirement": "covers last month"},
                ],
                metric_mapping={
                    "relevance": {"criteria": ["C1", "C2"], "comparison_basis": "q_vs_r"},
                    "rules_adherence": {
                        "reference_requirements": ["TABLE-RULE-01.01", "RULE-R1"],
                        "comparison_basis": "rules_vs_sql",
                        "evidence_types": ["sql"],
                    },
                    "context_recall": {
                        "comparison_basis": "required_information_vs_chunks",
                        "expectation": {
                            "status": "generated",
                            "value": {"required_information": ["Q1"]},
                        },
                    },
                },
            )
        )
    )
    context = question.context
    assert context.pack is not None and context.pack.pack_id == "RP_SCN_001"
    spec = context.reference_spec
    assert spec is not None
    assert [r.id for r in spec.requirements()] == [
        "reference_spec.schema",
        "TABLE-01",
        "COLUMN-01.01",
        "TABLE-RULE-01.01",
        "RULE-R1",
    ]
    assert context.requirement_text("C2") == "covers last month"
    assert context.requirement_text("RULE-R1") == "Never sum across currencies."
    assert context.generated("context_recall") == {"required_information": ["Q1"]}
    assert context.generated("relevance") is None
    assert question.criteria_texts == ["ranks employees by spend", "covers last month"]
    assert "expected_behavior" not in question.model_dump_json()


def test_a_legacy_question_is_kept_for_display_only() -> None:
    """Questions stored before the Evaluation Context load with their expectations kept
    verbatim; they are never executed or evaluated (user decision)."""
    legacy = _question(
        expectations={
            "expected_behavior": "Rank employees by card spend, per currency.",
            "success_criteria": {"general": ["ranks employees by spend"]},
        }
    )
    del legacy["evaluation_context"]
    question = QuestionCase.model_validate(legacy)
    assert question.is_legacy and question.criteria_texts == []
    assert question.legacy_expectations is not None
    assert question.legacy_expectations["expected_behavior"].startswith("Rank employees")
    with pytest.raises(ValueError, match="kept for display only"):
        _ = question.context
    with pytest.raises(ValidationError, match="exactly one of"):
        QuestionCase.model_validate({**legacy, "evaluation_context": _context()})


@pytest.mark.parametrize(
    "evaluation_context",
    [
        _context(success_criteria=[]),  # criteria required
        _context(success_criteria=[{"id": "C2", "requirement": "a"}]),  # ids C1..Cn in order
        _context(
            success_criteria=[{"id": "C1", "requirement": "a"}, {"id": "C1", "requirement": "b"}]
        ),
        _context(expected_behavior="answers"),  # no separate expected_behavior any more
        _context(metric_mapping={"Bad Key": {"comparison_basis": "x"}}),
        _context(metric_mapping={"relevance": {"criteria": ["C9"], "comparison_basis": "x"}}),
        _context(
            metric_mapping={
                "rules_adherence": {"reference_requirements": ["RULE-R9"], "comparison_basis": "x"}
            }
        ),
        _context(metric_mapping={"relevance": {"criteria": ["C1", "C1"], "comparison_basis": "x"}}),
        _context(
            metric_mapping={
                "context_recall": {
                    "comparison_basis": "x",
                    "expectation": {"status": "generated", "value": None},
                }
            }
        ),
        _context(
            metric_mapping={
                "context_recall": {
                    "comparison_basis": "x",
                    "expectation": {"status": "cannot_generate"},
                }
            }
        ),
        _context(
            metric_mapping={
                "context_recall": {
                    "comparison_basis": "x",
                    "expectation": {"status": "not_applicable", "value": {"x": 1}},
                }
            }
        ),
        _context(reference_spec={**SPEC, "pack_id": "RP_SCN_002"}),  # identity kept
    ],
)
def test_evaluation_context_shape(evaluation_context: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        QuestionCase.model_validate(_question(evaluation_context=evaluation_context))


def _metric(score: float | None = 1.0, status: str = "scored") -> dict[str, Any]:
    return {"status": status, "score": score, "rationale": "checked"}


def test_evaluation_metrics_are_keyed_by_any_metric_id() -> None:
    result = EvaluationResult.model_validate(
        _evaluation(
            verdict="pass",
            metrics={
                "relevance": _metric(),
                "custom_01": _metric(0.5),
                "mrr": _metric(1 / 3),  # code-computed reciprocal rank (continuous)
                "context_recall": _metric(None, "evidence_unavailable"),
            },
        )
    )
    assert result.metrics["custom_01"].score == 0.5
    assert result.metrics["context_recall"].status is MetricStatus.EVIDENCE_UNAVAILABLE


@pytest.mark.parametrize("status", [s.value for s in MetricStatus if s.value != "scored"])
def test_non_scored_statuses_never_carry_a_score(status: str) -> None:
    ok = EvaluationResult.model_validate(
        _evaluation(verdict="inconclusive", metrics={"m": _metric(None, status)})
    )
    assert ok.metrics["m"].score is None
    with pytest.raises(ValidationError, match="score is set exactly when status is scored"):
        EvaluationResult.model_validate(
            _evaluation(verdict="pass", metrics={"m": _metric(0.0, status)})
        )


def test_metric_score_iff_scored_and_bounded() -> None:
    with pytest.raises(ValidationError, match="score is set exactly when status is scored"):
        EvaluationResult.model_validate(
            _evaluation(verdict="pass", metrics={"relevance": _metric(None)})
        )
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(
            _evaluation(verdict="pass", metrics={"relevance": _metric(1.5)})
        )


def _legacy(**overrides: Any) -> dict[str, Any]:
    data = {k: v for k, v in _evaluation(verdict="pass").items() if k != "schema_version"}
    return {**data, **overrides}


def test_legacy_profile_evaluations_keep_their_original_metric_ids() -> None:
    """Never relabelled: completion / tool_selection stay as judged under the old rubrics."""
    agent = EvaluationResult.model_validate(
        _legacy(
            mode="ungrounded",
            evaluation_profile="agent",
            metrics={"relevance": _metric(), "completion": _metric(), "tool_selection": _metric()},
        )
    )
    assert agent.schema_version == 1 and agent.legacy_evaluation_profile == "agent"
    assert set(agent.metrics) == {"relevance", "completion", "tool_selection"}
    nl2sql = EvaluationResult.model_validate(
        _legacy(metrics={"semantic_correctness": _metric(), "rules_adherence": _metric()})
    )
    assert nl2sql.legacy_evaluation_profile == "nl2sql"


def test_former_voc_profile_reads_as_agent() -> None:
    result = EvaluationResult.model_validate(
        _legacy(
            metrics={"answer_relevance": _metric(), "answer_completeness": _metric()},
            evaluation_profile="voc",
        )
    )
    assert result.legacy_evaluation_profile == "agent"
    assert set(result.metrics) == {"relevance", "completion"}


def test_old_check_fields_are_gone() -> None:
    for field in ("checks", "check_rationale", "grounded_checks", "clarification_evaluation"):
        with pytest.raises(ValidationError, match="Extra inputs"):
            EvaluationResult.model_validate(_evaluation(verdict="pass", **{field: {}}))
    with pytest.raises(ValidationError, match="Extra inputs"):  # no profile on new results
        EvaluationResult.model_validate(_evaluation(verdict="pass", evaluation_profile="agent"))


SAMPLE_CHATBOT = Path(__file__).resolve().parents[2] / "data" / "products" / "sample_chatbot.yaml"


def test_sample_chatbot_is_a_non_sql_product() -> None:
    product = load_product(SAMPLE_CHATBOT)
    assert product.expected_outputs == ["Short text answer"]


def test_voc_product_description_loads() -> None:
    product = load_product(SAMPLE_CHATBOT.with_name("voice_of_customer_agent.yaml"))
    assert [s.name for s in product.data_sources] == ["Complaints", "CSATResult"]
    assert len(product.supported) == 10 and len(product.not_supported) == 9
    assert len(product.typical_user_questions) == 8 and len(product.expected_outputs) == 8
    assert product.business_problem and product.known_limitations


def test_previous_product_schema_is_refused_with_the_new_names() -> None:
    old = {"product_id": "p", "product_version": "1", "name": "n", "description": "d"}
    for key, message in (
        ("product_type", "'product_type' was removed"),
        ("not_supported_yet", "'not_supported_yet' is now 'not_supported'"),
        ("known_gaps", "'known_gaps' is now 'known_limitations'"),
        ("response_artifacts_expected", "is now 'expected_outputs'"),
    ):
        with pytest.raises(ValidationError, match=message):
            ProductDescription.model_validate({**old, key: ["x"]})
    minimal = ProductDescription.model_validate(old)  # everything else is optional
    assert minimal.data_sources == [] and minimal.business_problem is None
