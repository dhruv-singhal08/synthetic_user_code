"""Evaluation of answered responses with the 5 KPI metrics (design v3 section 11)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pytest

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import ArtifactIntegrityError
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.models.evaluation import (
    EvaluationDraft,
    EvaluationResult,
    MetricName,
    MetricStatus,
    Verdict,
)
from synthetic_user.models.execution import AnsweredResponse, ExecutionRecord
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import (
    DocumentPackDraft,
    DocumentPick,
    Reference,
    ReferencePack,
)
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator import EvaluationCase, Evaluator
from synthetic_user.nodes.evaluators.generic import compute_metrics, failing_metrics
from synthetic_user.prompts.response_evaluation import build_response_evaluation_prompt
from synthetic_user.references.pack import build_pack

CRITERIA = ["states the nightly hotel limit that applies", "says whether taxes are included"]
ANSWER = "For Paris the nightly hotel limit is USD 250, taxes included. Book via the travel tool."
LIMIT = "the nightly hotel limit is USD 250"
TAXES = "taxes included"
EXTRA = "Book via the travel tool."
POLICY_LIMIT = "The nightly hotel limit is USD 250 in Tier 1 cities"
TEXT = {"outcome": "answered", "text": ANSWER}

Case = tuple[Scenario, QuestionCase] | tuple[Scenario, QuestionCase, ReferencePack]


def score(value: float | None, rationale: str = "fine", evidence: str | None = None) -> dict:
    return {"score": value, "rationale": rationale, "evidence": evidence}


def claim(text: str, status: str, quote: str | None = None) -> dict[str, Any]:
    return {"text": text, "status": status, "reason": "checked", "reference_quote": quote}


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


def criteria(*statuses: str) -> list[dict[str, Any]]:
    return [
        {"criterion": c, "status": s, "rationale": "checked"}
        for c, s in zip(CRITERIA, statuses, strict=True)
    ]


def draft(**overrides: Any) -> dict[str, Any]:
    """An ungrounded judge reply that passes everything."""
    base: dict[str, Any] = {
        "relevancy": score(1, "Answers the hotel question.", LIMIT),
        "correctness": None,
        "criteria": criteria("met", "met"),
        "claims": [claim(LIMIT, "unverified"), claim(TAXES, "unverified")],
        "findings": [],
        "overall_verdict": "pass",
    }
    base.update(overrides)
    return base


def grounded_draft(**overrides: Any) -> dict[str, Any]:
    base = draft(
        correctness=score(1, "Matches the hotels section."),
        claims=[
            claim(LIMIT, "supported", POLICY_LIMIT),
            claim(TAXES, "supported", TAXES),
        ],
    )
    base.update(overrides)
    return base


@pytest.fixture
def ungrounded(sample_scenarios: list[Scenario]) -> tuple[Scenario, QuestionCase]:
    scenario = sample_scenarios[0].model_copy(
        update={
            "mode": GenerationMode.UNGROUNDED,
            "reference_id": None,
            "reference_pack_id": None,
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
            "question_characteristics": {
                "style": "precise",
                "ambiguity_level": "low",
                "complexity": "simple",
            },
            "expectations": {
                "expected_behavior": "State the nightly hotel limit for Paris and whether "
                "taxes are included.",
                "success_criteria": CRITERIA,
            },
        }
    )
    return scenario, question


@pytest.fixture
def grounded(
    ungrounded: tuple[Scenario, QuestionCase], policy_reference: Reference
) -> tuple[Scenario, QuestionCase, ReferencePack]:
    """The same hotel question, grounded in the sample travel policy documents."""
    scenario, question = ungrounded
    draft = DocumentPackDraft(
        documents=[DocumentPick(name="Hotel policy", why="Paris is a Tier 1 city")]
    )
    pack, rejected = build_pack(draft, policy_reference, scenario.id)
    assert pack is not None and not rejected
    scenario = scenario.model_copy(
        update={
            "mode": GenerationMode.GROUNDED,
            "reference_id": policy_reference.reference_id,
            "reference_pack_id": pack.pack_id,
        }
    )
    question = question.model_copy(
        update={"mode": GenerationMode.GROUNDED, "reference_pack_id": pack.pack_id}
    )
    return scenario, question, pack


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
    product: ProductDescription,
    case: Case,
    response: dict[str, Any],
    *drafts: dict[str, Any],
    reference: Reference | None = None,
) -> tuple[EvaluationResult, FakeLLMClient]:
    scenario, question = case[0], case[1]
    pack = case[2] if len(case) == 3 else None
    llm = FakeLLMClient(list(drafts))
    evaluator = Evaluator(llm, app_config, logging.getLogger("t"), reference=reference)
    result = await evaluator.evaluate(
        EvaluationCase(question, scenario, pack, _execution(response)), product
    )
    return result, llm


def _scores(result: EvaluationResult) -> dict[str, float | str | None]:
    assert result.metrics is not None
    return {
        name.value: (m.score if m.status is MetricStatus.SCORED else m.status.value)
        for name, m in result.metrics.by_name().items()
    }


# --- ungrounded ----------------------------------------------------------------------------


async def test_ungrounded_answer_passes_with_metric_scores(
    app_config: AppConfig,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    result, llm = await _evaluate(app_config, sample_chatbot, ungrounded, TEXT, draft())
    assert result.verdict is Verdict.PASS and result.mode == "ungrounded"
    assert _scores(result) == {
        "relevancy": 1.0,
        "correctness": "not_applicable",
        "groundedness": "not_applicable",
        "hallucination_rate": 0.0,
        "completeness": 1.0,
    }
    assert [c.status for c in result.claims] == ["unverified", "unverified"]
    assert len(llm.calls) == 1
    assert "labelled unverified (plausible but" in llm.calls[0].user
    assert "Reference pack" not in llm.calls[0].user


async def test_fabricated_claim_is_a_hallucination_and_a_bug(
    app_config: AppConfig,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    reply = draft(
        claims=[claim(LIMIT, "fabricated"), claim(TAXES, "unverified")],
        findings=[finding("hallucination_rate", LIMIT, "high")],
        overall_verdict="fail",
    )
    result, _ = await _evaluate(app_config, sample_chatbot, ungrounded, TEXT, reply)
    assert result.verdict is Verdict.FAIL
    assert _scores(result)["hallucination_rate"] == 0.5
    assert result.metrics is not None and result.metrics.hallucination_rate.evidence == LIMIT
    assert result.bug_candidates[0].title.startswith("hallucination_rate:")


async def test_irrelevant_answer_fails(
    app_config: AppConfig,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    reply = draft(
        relevancy=score(0, "Talks about booking.", EXTRA),
        findings=[finding("relevancy", EXTRA, "high")],
    )
    result, _ = await _evaluate(app_config, sample_chatbot, ungrounded, TEXT, reply)
    assert result.verdict is Verdict.FAIL and _scores(result)["relevancy"] == 0.0
    assert len(result.bug_candidates) == 1


async def test_completeness_from_criteria(
    app_config: AppConfig,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    half = draft(criteria=criteria("met", "not_met"), findings=[finding("completeness")])
    result, _ = await _evaluate(app_config, sample_chatbot, ungrounded, TEXT, half)
    assert _scores(result)["completeness"] == 0.5
    assert result.verdict is Verdict.PASS  # only a 0 fails the question
    none_met = draft(
        criteria=criteria("not_met", "not_met"), findings=[finding("completeness", LIMIT)]
    )
    result, _ = await _evaluate(app_config, sample_chatbot, ungrounded, TEXT, none_met)
    assert _scores(result)["completeness"] == 0.0 and result.verdict is Verdict.FAIL
    partly = draft(criteria=criteria("partly", "unverifiable"))
    result, _ = await _evaluate(app_config, sample_chatbot, ungrounded, TEXT, partly)
    assert _scores(result)["completeness"] == 0.5  # unverifiable is left out
    unverifiable = draft(criteria=criteria("unverifiable", "unverifiable"))
    result, _ = await _evaluate(app_config, sample_chatbot, ungrounded, TEXT, unverifiable)
    assert _scores(result)["completeness"] == "inconclusive"
    assert result.verdict is Verdict.INCONCLUSIVE


async def test_undecided_relevancy_or_weak_evidence_is_inconclusive(
    app_config: AppConfig,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    result, _ = await _evaluate(
        app_config, sample_chatbot, ungrounded, TEXT, draft(relevancy=score(None, "unclear"))
    )
    assert result.verdict is Verdict.INCONCLUSIVE
    weak = {**finding("relevancy", EXTRA), "evidence_status": "weak"}
    reply = draft(relevancy=score(0, "off topic"), findings=[weak], overall_verdict="fail")
    result, _ = await _evaluate(app_config, sample_chatbot, ungrounded, TEXT, reply)
    assert result.verdict is Verdict.INCONCLUSIVE and result.llm_verdict is Verdict.FAIL
    assert not result.bug_candidates


@pytest.mark.parametrize(
    "response",
    [
        {"outcome": "clarification", "message": "Which city?"},
        {"outcome": "denied", "message": "I cannot help with that."},
        {"outcome": "error", "error_type": "Boom"},
        {"outcome": "timeout"},
    ],
)
async def test_non_answers_are_recorded_only_without_llm(
    app_config: AppConfig,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
    response: dict[str, Any],
) -> None:
    result, llm = await _evaluate(app_config, sample_chatbot, ungrounded, response)
    assert result.verdict is None and result.evaluation_status == "recorded_only"
    assert result.metrics is None and not llm.calls


# --- grounded -----------------------------------------------------------------------------


async def test_grounded_answer_supported_by_pack_passes(
    app_config: AppConfig,
    sample_chatbot: ProductDescription,
    grounded: Case,
    policy_reference: Reference,
) -> None:
    result, llm = await _evaluate(
        app_config, sample_chatbot, grounded, TEXT, grounded_draft(), reference=policy_reference
    )
    assert result.verdict is Verdict.PASS and result.mode == "grounded"
    assert _scores(result) == {
        "relevancy": 1.0,
        "correctness": 1.0,
        "groundedness": 1.0,
        "hallucination_rate": 0.0,
        "completeness": 1.0,
    }
    assert "Cheat sheet of this scenario (AUTHORITATIVE" in llm.calls[0].user


async def test_extra_statements_lower_groundedness_but_do_not_fail(
    app_config: AppConfig,
    sample_chatbot: ProductDescription,
    grounded: Case,
    policy_reference: Reference,
) -> None:
    reply = grounded_draft(
        claims=[
            claim(LIMIT, "supported", POLICY_LIMIT),
            claim(TAXES, "supported", TAXES),
            claim(EXTRA, "not_in_reference"),
            claim("For Paris", "not_in_reference"),
        ]
    )
    result, _ = await _evaluate(
        app_config, sample_chatbot, grounded, TEXT, reply, reference=policy_reference
    )
    assert _scores(result)["groundedness"] == 0.5
    assert _scores(result)["hallucination_rate"] == 0.0
    assert result.verdict is Verdict.PASS


async def test_contradicted_claim_fails_correctness_and_hallucination(
    app_config: AppConfig,
    sample_chatbot: ProductDescription,
    grounded: Case,
    policy_reference: Reference,
) -> None:
    wrong = "For Paris the nightly hotel limit is USD 400."
    reply = grounded_draft(
        relevancy=score(1, "Answers the hotel question."),
        correctness=score(0, "Wrong limit.", "USD 400"),
        claims=[claim("the nightly hotel limit is USD 400", "contradicted", POLICY_LIMIT)],
        criteria=criteria("not_met", "not_met"),
        findings=[
            finding("correctness", "USD 400", "high"),
            finding("hallucination_rate", "USD 400", "high"),
            finding("completeness", "USD 400"),
        ],
        overall_verdict="fail",
    )
    result, _ = await _evaluate(
        app_config,
        sample_chatbot,
        grounded,
        {"outcome": "answered", "text": wrong},
        reply,
        reference=policy_reference,
    )
    assert result.verdict is Verdict.FAIL
    assert _scores(result)["groundedness"] == 0.0
    assert _scores(result)["hallucination_rate"] == 1.0
    assert result.metrics is not None
    assert set(failing_metrics(result.metrics)) == {
        MetricName.CORRECTNESS,
        MetricName.HALLUCINATION_RATE,
        MetricName.COMPLETENESS,
    }


async def test_invalid_drafts_are_retried_with_reasons(
    app_config: AppConfig,
    sample_chatbot: ProductDescription,
    grounded: Case,
    policy_reference: Reference,
) -> None:
    bad = grounded_draft(
        correctness=None,  # required when grounded
        criteria=criteria("met", "met")[:1],  # one criterion missing
        claims=[
            claim("a sentence the answer never says", "supported", POLICY_LIMIT),
            claim(LIMIT, "supported", "a passage that is not in the policy"),
            claim(TAXES, "unverified"),  # ungrounded status in a grounded run
        ],
    )
    result, llm = await _evaluate(
        app_config,
        sample_chatbot,
        grounded,
        TEXT,
        bad,
        grounded_draft(relevancy=score(0, "off")),  # 0 without a relevancy finding
        grounded_draft(),
        reference=policy_reference,
    )
    assert len(llm.calls) == 3 and result.verdict is Verdict.PASS
    first, second = llm.calls[1].user, llm.calls[2].user
    for reason in (
        "correctness is required for a grounded question",
        "criteria must list each success criterion once",
        "is not copied from the response",
        "reference_quote is not in the cheat sheet or the usage instructions",
        "status 'unverified' must be one of",
    ):
        assert reason in first
    assert "a finding with category 'relevancy' is required" in second


def test_scores_are_three_levels_only() -> None:
    with pytest.raises(ValueError, match="score must be 0, 0.5 or 1"):
        EvaluationDraft.model_validate(draft(relevancy=score(0.7)))


async def test_ungrounded_draft_with_correctness_is_retried(
    app_config: AppConfig,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    result, llm = await _evaluate(
        app_config,
        sample_chatbot,
        ungrounded,
        TEXT,
        draft(correctness=score(1)),
        draft(),
    )
    assert len(llm.calls) == 2 and result.verdict is Verdict.PASS
    assert "correctness must be null" in llm.calls[1].user


async def test_judge_failure_is_inconclusive_with_notes(
    app_config: AppConfig,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    broken = draft(relevancy=score(0, "no finding given"))
    result, _ = await _evaluate(
        app_config, sample_chatbot, ungrounded, TEXT, broken, broken, broken
    )
    assert result.verdict is Verdict.INCONCLUSIVE and result.metrics is None
    assert "evaluation failed after 3 attempt(s)" in result.notes[0]


async def test_grounded_case_needs_its_reference(
    app_config: AppConfig,
    sample_chatbot: ProductDescription,
    grounded: Case,
    policy_reference: Reference,
) -> None:
    with pytest.raises(ArtifactIntegrityError, match="grounded but no reference was given"):
        await _evaluate(app_config, sample_chatbot, grounded, TEXT)
    other = policy_reference.model_copy(update={"reference_id": "other-00000000"})
    with pytest.raises(ArtifactIntegrityError, match="was built from reference"):
        await _evaluate(app_config, sample_chatbot, grounded, TEXT, reference=other)


# --- prompt -------------------------------------------------------------------------------


def test_prompt_shows_only_the_evaluation_contract(
    sample_chatbot: ProductDescription,
    grounded: tuple[Scenario, QuestionCase, ReferencePack],
    policy_reference: Reference,
) -> None:
    scenario, question, pack = grounded
    response = _execution(TEXT).response
    assert isinstance(response, AnsweredResponse)
    _system, user = build_response_evaluation_prompt(
        sample_chatbot, scenario, question, response, (pack, policy_reference)
    )
    for shown in (
        question.utterance,
        scenario.business_goal,
        question.expectations.expected_behavior,
        POLICY_LIMIT,
        "Limits and amounts in the summaries are exact",  # usage instructions
        "consistent with the summaries",  # document correctness rule
        "labelled supported (a summary states it)",
        "they are\n  partial, so this is NOT an error",
        "Framing sentences that state no fact",
    ):
        assert shown in user
    for hidden in (
        scenario.title,
        scenario.capability_rationale,
        '"task_type"',
        "intended_goal",
        "Alcohol is never reimbursable",  # the full reference text is not sent
        "persona",
        "acceptable_variations",
        "$defs",  # compact contract, not the JSON Schema
    ):
        assert hidden not in user


# --- metrics, bugs ------------------------------------------------------------------------


def test_no_claims_means_not_applicable() -> None:
    metrics = compute_metrics(EvaluationDraft.model_validate(draft(claims=[])), grounded=False)
    assert metrics.groundedness.status is MetricStatus.NOT_APPLICABLE
    assert metrics.hallucination_rate.status is MetricStatus.NOT_APPLICABLE


async def test_bug_rules_severity_evidence_and_order(
    app_config: AppConfig,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    reply = draft(
        relevancy=score(0, "Off topic.", EXTRA),
        claims=[claim(LIMIT, "fabricated")],
        findings=[
            finding("relevancy", EXTRA, "medium"),
            finding("hallucination_rate", LIMIT, "critical"),
            {**finding("relevancy", None, "high"), "evidence_status": "weak"},
            finding("relevancy", None, "low"),
        ],
        overall_verdict="fail",
    )
    result, _ = await _evaluate(app_config, sample_chatbot, ungrounded, TEXT, reply)
    assert result.verdict is Verdict.FAIL
    (bug,) = result.bug_candidates
    assert bug.finding_ids == ["F_002", "F_001"]  # supported, >= medium, most severe first
    assert bug.status == "needs_review" and bug.product_id == "sample_policy_chatbot"
    assert bug.reproduction_question == "How much can I spend per night on a hotel in Paris?"
    assert bug.title.startswith("hallucination_rate:")


async def test_findings_matching_known_gaps_are_flagged(
    app_config: AppConfig,
    sample_chatbot: ProductDescription,
    ungrounded: tuple[Scenario, QuestionCase],
) -> None:
    gap = {
        **finding("completeness", LIMIT),
        "description": "Gives outdated hotel limits for recently changed countries",
    }
    reply = draft(criteria=criteria("not_met", "met"), findings=[gap])
    result, _ = await _evaluate(app_config, sample_chatbot, ungrounded, TEXT, reply)
    assert result.findings[0].matches_known_gap is True


async def test_reference_quote_may_come_from_the_usage_instructions(
    app_config: AppConfig,
    sample_chatbot: ProductDescription,
    grounded: tuple[Scenario, QuestionCase, ReferencePack],
    policy_reference: Reference,
) -> None:
    from_instructions = "Limits and amounts in the summaries are exact"
    result, llm = await _evaluate(
        app_config,
        sample_chatbot,
        grounded,
        TEXT,
        grounded_draft(claims=[claim(LIMIT, "supported", from_instructions)]),
        reference=policy_reference,
    )
    assert len(llm.calls) == 1 and result.verdict is Verdict.PASS
