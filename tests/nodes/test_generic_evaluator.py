"""C08: generic evaluation (all products, both modes) through the evaluator router."""

from __future__ import annotations

import copy
import logging
from datetime import UTC, datetime
from typing import Any

import pytest

from synthetic_user.core.config import AppConfig
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.models.evaluation import CriterionStatus, EvaluationResult, Verdict
from synthetic_user.models.execution import ExecutionRecord
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator import EvaluationCase, Evaluator
from synthetic_user.nodes.evaluators.generic import (
    check_generic_evaluation,
    expected_goal_status,
)
from synthetic_user.prompts.response_evaluation import (
    EvaluationFocus,
    build_response_evaluation_prompt,
)

CRITERIA = ["states the nightly hotel limit that applies", "says whether taxes are included"]
ANSWER = "For Paris the nightly hotel limit is 220 EUR including taxes."
NA = {"status": "not_applicable", "rationale": "n/a"}


def ok(rationale: str = "fine", evidence: str | None = None) -> dict[str, Any]:
    return {"status": "pass", "rationale": rationale, "evidence": evidence}


def bad(rationale: str, evidence: str | None = None) -> dict[str, Any]:
    return {"status": "fail", "rationale": rationale, "evidence": evidence}


def finding(category: str, evidence: str | None = None, severity: str = "medium") -> dict[str, Any]:
    return {
        "category": category,
        "description": f"{category} problem",
        "expected_behavior": "expected",
        "observed_behavior": "observed",
        "evidence": evidence,
        "severity": severity,
        "evidence_status": "supported",
    }


def answer_draft(**overrides: Any) -> dict[str, Any]:
    draft = {
        "relevance": ok("Answers the hotel question.", "nightly hotel limit"),
        "goal_completeness": ok("Both criteria met."),
        "criteria": [{"criterion": c, "status": "met", "rationale": "stated"} for c in CRITERIA],
        "persona_fit": ok(),
        "capability_consistency": ok(),
        "clarification_handling": NA,
        "clarification": None,
        "safety_and_honesty": ok(),
        "findings": [],
        "overall_verdict": "pass",
    }
    draft.update(overrides)
    return draft


@pytest.fixture
def ungrounded(sample_scenarios: list[Scenario]) -> tuple[Scenario, QuestionCase]:
    scenario = sample_scenarios[0].model_copy(
        update={
            "mode": GenerationMode.UNGROUNDED,
            "reference_index_id": None,
            "reference_cheat_sheet": None,
            "success_criteria": CRITERIA,
        }
    )
    question = QuestionCase.model_validate(
        {
            "question_id": "Q_001",
            "scenario_id": scenario.id,
            "persona_id": scenario.persona_id,
            "mode": "ungrounded",
            "utterance": "How much can I spend per night on a hotel in Paris?",
            "intended_goal": scenario.business_goal,
            "intent": {"task_type": "explanation"},
            "question_characteristics": {"ambiguity_level": "low"},
            "expectations": {
                "clarification": "not_expected",
                "acceptable_behaviors": ["states the limit"],
                "success_criteria": CRITERIA,
            },
        }
    )
    return scenario, question


def _execution(response: dict[str, Any]) -> ExecutionRecord:
    return ExecutionRecord.model_validate(
        {
            "execution_id": "EXE_001",
            "question_id": "Q_001",
            "product_id": "sample_policy_chatbot",
            "adapter": "fixture",
            "submitted_question": "How much can I spend per night on a hotel in Paris?",
            "started_at": datetime(2026, 10, 18, tzinfo=UTC),
            "response": response,
        }
    )


async def _evaluate(
    app_config: AppConfig,
    persona: Persona,
    product: ProductDescription,
    case: tuple[Scenario, QuestionCase],
    response: dict[str, Any],
    *drafts: dict[str, Any],
) -> tuple[EvaluationResult, FakeLLMClient]:
    scenario, question = case
    llm = FakeLLMClient(list(drafts))
    evaluator = Evaluator(llm, app_config, logging.getLogger("t"), index=None)
    result = await evaluator.evaluate(
        EvaluationCase(question, scenario, None, _execution(response)), product, persona
    )
    return result, llm


TEXT = {"outcome": "answered", "text": ANSWER}


async def test_relevant_answer_passes(
    app_config: AppConfig,
    sample_persona: Persona,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    result, llm = await _evaluate(
        app_config, sample_persona, sample_chatbot, ungrounded, TEXT, answer_draft()
    )
    assert result.verdict is Verdict.PASS and result.evaluation_status == "evaluated"
    assert result.checks is not None and result.checks.relevance == "pass"
    assert result.check_rationale["relevance"] == "Answers the hotel question."
    assert [c.status for c in result.criteria_results] == ["met", "met"]
    assert result.mode == "ungrounded" and not result.plugins_applied
    assert len(llm.calls) == 1


async def test_irrelevant_answer_fails_with_bug(
    app_config: AppConfig,
    sample_persona: Persona,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    draft = answer_draft(
        relevance=bad("Talks about something else.", "220 EUR"),
        findings=[finding("relevance", "220 EUR", "high")],
    )
    result, _ = await _evaluate(app_config, sample_persona, sample_chatbot, ungrounded, TEXT, draft)
    assert result.verdict is Verdict.FAIL
    assert [f.category for f in result.findings] == ["relevance"]
    assert result.findings[0].evidence == "220 EUR" and result.findings[0].source == "llm"
    assert len(result.bug_candidates) == 1


async def test_partial_goal_fails_with_criterion_detail(
    app_config: AppConfig,
    sample_persona: Persona,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    partial = answer_draft(
        goal_completeness=bad("Taxes not mentioned."),
        criteria=[
            {"criterion": CRITERIA[0], "status": "met", "rationale": "stated"},
            {"criterion": CRITERIA[1], "status": "not_met", "rationale": "taxes missing"},
        ],
        findings=[finding("goal_completeness")],
    )
    text = {"outcome": "answered", "text": "For Paris the nightly hotel limit is 220 EUR."}
    result, _ = await _evaluate(
        app_config, sample_persona, sample_chatbot, ungrounded, text, partial
    )
    assert result.verdict is Verdict.FAIL
    assert [(c.criterion, c.status) for c in result.criteria_results] == [
        (CRITERIA[0], CriterionStatus.MET),
        (CRITERIA[1], CriterionStatus.NOT_MET),
    ]
    assert result.checks is not None and result.checks.goal_completeness == "fail"


async def test_unjustified_denial_fails_with_bug(
    app_config: AppConfig,
    sample_persona: Persona,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    denial = {
        "relevance": NA,
        "goal_completeness": NA,
        "criteria": [],
        "persona_fit": NA,
        "capability_consistency": bad("Hotel limits are supported.", "I cannot help"),
        "clarification_handling": NA,
        "safety_and_honesty": NA,
        "findings": [finding("capability_consistency", "I cannot help", "high")],
        "overall_verdict": "fail",
    }
    response = {"outcome": "denied", "message": "I cannot help with that."}
    result, _ = await _evaluate(
        app_config, sample_persona, sample_chatbot, ungrounded, response, denial
    )
    # C10: a denial gets its verdict from capability_consistency only
    assert result.verdict is Verdict.FAIL and result.evaluation_status == "evaluated"
    assert [f.category for f in result.findings] == ["capability_consistency"]
    assert result.checks is not None and result.checks.capability_consistency == "fail"
    assert result.llm_verdict is Verdict.FAIL
    assert [(b.product_id, b.question_id) for b in result.bug_candidates] == [
        ("sample_policy_chatbot", "Q_001")
    ]


async def test_error_and_timeout_are_recorded_only_without_llm(
    app_config: AppConfig,
    sample_persona: Persona,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    for response in ({"outcome": "error", "error_type": "Boom"}, {"outcome": "timeout"}):
        result, llm = await _evaluate(
            app_config, sample_persona, sample_chatbot, ungrounded, response
        )
        assert result.verdict is None and result.checks is None and not llm.calls


async def test_invalid_draft_is_retried_with_reasons(
    app_config: AppConfig,
    sample_persona: Persona,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    wrong = answer_draft(
        criteria=[{"criterion": CRITERIA[0], "status": "met", "rationale": "x"}],
        relevance=ok("made up", "this quote is not in the answer"),
    )
    result, llm = await _evaluate(
        app_config, sample_persona, sample_chatbot, ungrounded, TEXT, wrong, answer_draft()
    )
    assert len(llm.calls) == 2 and result.verdict is Verdict.PASS
    retry = llm.calls[1].user
    assert "criteria must list each success criterion once" in retry
    assert "is not in the response" in retry


async def test_judge_failure_is_inconclusive_with_notes(
    app_config: AppConfig,
    sample_persona: Persona,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    broken = answer_draft(relevance=bad("no finding given"))
    result, _ = await _evaluate(
        app_config, sample_persona, sample_chatbot, ungrounded, TEXT, broken, broken, broken
    )
    assert result.verdict is Verdict.INCONCLUSIVE and result.checks is None
    assert "generic evaluation failed after 3 attempt(s)" in result.notes[0]


def test_goal_status_from_criteria() -> None:
    met, partly, unver = CriterionStatus.MET, CriterionStatus.PARTLY, CriterionStatus.UNVERIFIABLE
    assert expected_goal_status([met, met]) == "pass"
    assert expected_goal_status([met, partly]) == "fail"
    assert expected_goal_status([met, unver]) == "inconclusive"


def test_focus_rules_enforced() -> None:
    from synthetic_user.models.evaluation import GenericEvaluationDraft

    draft = GenericEvaluationDraft.model_validate(answer_draft())
    errors = check_generic_evaluation(draft, EvaluationFocus.DENIAL, [], ANSWER)
    assert any("denial: relevance must be not_applicable" in e for e in errors)
    clar = copy.deepcopy(answer_draft())
    errors = check_generic_evaluation(
        GenericEvaluationDraft.model_validate(clar), EvaluationFocus.CLARIFICATION, [], ANSWER
    )
    assert any("clarification detail is required" in e for e in errors)


def test_prompt_never_claims_ground_truth_when_ungrounded(
    sample_persona: Persona,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    scenario, question = ungrounded
    system, user = build_response_evaluation_prompt(
        sample_persona,
        sample_chatbot,
        scenario,
        question,
        _execution(TEXT).response,
        EvaluationFocus.ANSWER,
    )
    assert "There is NO ground truth" in user and "unverifiable" in user
    for item in sample_chatbot.not_supported_yet + sample_chatbot.known_gaps:
        assert item in user
    for criterion in CRITERIA:
        assert criterion in user
    assert ANSWER in user and "SQL" not in system


# --- C10: combination and bug creation rules -----------------------------------------------------


def test_combination_rules() -> None:
    from synthetic_user.nodes.evaluator import combine, combine_opinions

    assert combine([Verdict.PASS, Verdict.PASS]) is Verdict.PASS
    assert combine([Verdict.PASS, Verdict.INCONCLUSIVE]) is Verdict.INCONCLUSIVE
    assert combine([Verdict.INCONCLUSIVE, Verdict.FAIL, Verdict.PASS]) is Verdict.FAIL
    assert combine_opinions([None, Verdict.PASS, None]) is Verdict.PASS
    assert combine_opinions([None]) is None


async def test_failed_check_without_supported_evidence_is_inconclusive(
    app_config: AppConfig,
    sample_persona: Persona,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    weak = {**finding("persona_fit"), "evidence_status": "weak"}
    draft = answer_draft(persona_fit=bad("A bit formal."), findings=[weak], overall_verdict="fail")
    result, _ = await _evaluate(app_config, sample_persona, sample_chatbot, ungrounded, TEXT, draft)
    assert result.verdict is Verdict.INCONCLUSIVE
    assert result.llm_verdict is Verdict.FAIL  # recorded, not decisive
    assert not result.bug_candidates


async def test_llm_opinion_is_not_decisive(
    app_config: AppConfig,
    sample_persona: Persona,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    result, _ = await _evaluate(
        app_config,
        sample_persona,
        sample_chatbot,
        ungrounded,
        TEXT,
        answer_draft(overall_verdict="fail"),
    )
    assert result.verdict is Verdict.PASS and result.llm_verdict is Verdict.FAIL


async def test_bug_rules_severity_evidence_and_order(
    app_config: AppConfig,
    sample_persona: Persona,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    draft = answer_draft(
        relevance=bad("Off topic.", "220 EUR"),
        safety_and_honesty=bad("Overclaims.", "220 EUR"),
        findings=[
            finding("relevance", "220 EUR", "medium"),
            finding("safety_and_honesty", "220 EUR", "critical"),
            {**finding("relevance", None, "high"), "evidence_status": "weak"},
            finding("relevance", None, "low"),
        ],
        overall_verdict="fail",
    )
    result, _ = await _evaluate(app_config, sample_persona, sample_chatbot, ungrounded, TEXT, draft)
    assert result.verdict is Verdict.FAIL
    (bug,) = result.bug_candidates
    # only supported findings with severity >= medium, most severe first
    assert bug.finding_ids == ["F_002", "F_001"]
    assert bug.status == "needs_review" and bug.product_id == "sample_policy_chatbot"
    assert (bug.question_id, bug.scenario_id, bug.execution_id) == ("Q_001", "SCN_001", "EXE_001")
    assert bug.reproduction_question == "How much can I spend per night on a hotel in Paris?"
    assert bug.title.startswith("safety_and_honesty:")


async def test_findings_matching_known_gaps_are_flagged(
    app_config: AppConfig,
    sample_persona: Persona,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    gap = {
        **finding("goal_completeness"),
        "description": "Gives outdated hotel limits for recently changed countries",
    }
    draft = answer_draft(
        goal_completeness=bad("Outdated limit."),
        criteria=[
            {"criterion": CRITERIA[0], "status": "partly", "rationale": "outdated"},
            {"criterion": CRITERIA[1], "status": "met", "rationale": "ok"},
        ],
        findings=[gap],
        overall_verdict="fail",
    )
    result, _ = await _evaluate(app_config, sample_persona, sample_chatbot, ungrounded, TEXT, draft)
    assert result.findings[0].matches_known_gap is True
