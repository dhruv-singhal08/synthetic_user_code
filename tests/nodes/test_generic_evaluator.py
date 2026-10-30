"""NL2SQL evaluation (semantic correctness, rules adherence, schema faithfulness) and the
Agent evaluator (relevance, completion, telemetry-dependent optional metrics)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pytest

from synthetic_user.core.config import AppConfig
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.models.evaluation import (
    AgentEvaluationDraft,
    EvaluationDraft,
    EvaluationResult,
    MetricName,
    MetricStatus,
    Verdict,
)
from synthetic_user.models.evaluation_profile import AgentMetric, EvaluationProfile
from synthetic_user.models.execution import AnsweredResponse, ExecutionRecord
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.scenario import Scenario
from synthetic_user.no_sql import NO_SQL_REASON
from synthetic_user.nodes.evaluator import EvaluationCase, Evaluator
from synthetic_user.nodes.evaluators.agent import (
    check_agent_evaluation,
    compute_agent_metrics,
    plan_agent_metrics,
)
from synthetic_user.nodes.evaluators.generic import (
    compute_metrics,
    failing_metrics,
    without_ignored_sql_workarounds,
)
from synthetic_user.prompts.response_evaluation import build_response_evaluation_prompt

CRITERIA = ["names the employees ranked by total card spend", "covers the requested time period"]
JOIN = "c.employee_id = e.employee_id"
BAD_JOIN = "c.merchant_id = e.employee_id"
SQL = (
    "SELECT e.full_name, SUM(c.amount) AS total_spend FROM card_transactions c "
    f"JOIN employees e ON {JOIN} GROUP BY e.full_name ORDER BY total_spend DESC LIMIT 10"
)
ANSWER = {
    "outcome": "answered",
    "text": "Here is the SQL query for the top spenders.",
    "artifacts": [{"kind": "sql", "content": SQL}],
}


def score(value: float | None, rationale: str = "fine", evidence: str | None = None) -> dict:
    return {"score": value, "rationale": rationale, "evidence": evidence}


def finding(
    category: str, evidence: str | None = None, severity: str = "medium", **extra: Any
) -> dict[str, Any]:
    return {
        "category": category,
        "description": f"{category} problem",
        "expected_behavior": "expected",
        "observed_behavior": "observed",
        "evidence": evidence,
        "severity": severity,
        "evidence_status": "supported",
        **extra,
    }


def criteria(*statuses: str) -> list[dict[str, Any]]:
    return [
        {"criterion": c, "status": s, "rationale": "checked"}
        for c, s in zip(CRITERIA, statuses, strict=True)
    ]


def draft(**overrides: Any) -> dict[str, Any]:
    """A judge reply that passes all three NL2SQL metrics."""
    base: dict[str, Any] = {
        "semantic_correctness": score(1, "Ranks employees by spend.", "SUM(c.amount)"),
        "rules_adherence": score(1, "Uses the documented employee relationship."),
        "schema_faithfulness": score(1, "Every identifier is allowed."),
        "criteria": criteria("met", "met"),
        "findings": [],
        "overall_verdict": "pass",
    }
    base.update(overrides)
    return base


@pytest.fixture
def case(
    sample_scenarios: list[Scenario], sample_packs: list[ReferencePack]
) -> tuple[Scenario, QuestionCase, ReferencePack]:
    scenario, pack = sample_scenarios[0], sample_packs[0]
    question = QuestionCase.model_validate(
        {
            "question_id": "Q_001",
            "scenario_id": scenario.id,
            "persona_id": scenario.persona_id,
            "mode": "grounded",
            "reference_pack_id": pack.pack_id,
            "utterance": "Who were our top 10 card spenders last quarter?",
            "intended_goal": scenario.business_goal,
            "intent": {"task_type": "ranking"},
            "question_characteristics": {
                "style": "precise",
                "ambiguity_level": "low",
                "complexity": "simple",
            },
            "expectations": {
                "expected_behavior": "Return SQL ranking employees by card spend.",
                "success_criteria": CRITERIA,
            },
        }
    )
    return scenario, question, pack


def _execution(response: dict[str, Any]) -> ExecutionRecord:
    return ExecutionRecord.model_validate(
        {
            "execution_id": "EXE_001",
            "question_id": "Q_001",
            "product_id": "sample_nl2sql",
            "adapter": "fixture",
            "submitted_question": "Who were our top 10 card spenders last quarter?",
            "started_at": datetime(2026, 10, 18, tzinfo=UTC),
            "response": response,
        }
    )


async def _evaluate(
    settings: AppConfig,
    product: ProductDescription,
    case: tuple[Scenario, QuestionCase, ReferencePack],
    reference: Reference,
    response: dict[str, Any],
    *drafts: dict[str, Any],
) -> tuple[EvaluationResult, FakeLLMClient]:
    scenario, question, pack = case
    llm = FakeLLMClient(list(drafts))
    evaluator = Evaluator(llm, settings, logging.getLogger("t"), reference=reference)
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


# --- NL2SQL --------------------------------------------------------------------------------


async def test_correct_sql_passes_with_three_metrics_and_tagged_criteria(
    app_config: AppConfig, sample_product: ProductDescription, case: Any, sample_reference: Any
) -> None:
    result, llm = await _evaluate(
        app_config, sample_product, case, sample_reference, ANSWER, draft()
    )
    assert result.verdict is Verdict.PASS and result.evaluation_profile == "nl2sql"
    assert _scores(result) == {
        "semantic_correctness": 1.0,
        "rules_adherence": 1.0,
        "schema_faithfulness": 1.0,
    }
    assert [c.tag for c in result.criteria_results] == ["CRIT-01", "CRIT-02"]
    assert "three narrow NL2SQL comparisons" in llm.calls[0].system


async def test_rule_violation_fails_with_bug_and_trace_tags(
    app_config: AppConfig, sample_product: ProductDescription, case: Any, sample_reference: Any
) -> None:
    bad = {**ANSWER, "artifacts": [{"kind": "sql", "content": SQL.replace(JOIN, BAD_JOIN)}]}
    reply = draft(
        rules_adherence=score(0, "Undocumented relationship.", BAD_JOIN),
        findings=[finding("rules_adherence", BAD_JOIN, "high", related_tags=["JOIN-01"])],
        overall_verdict="fail",
    )
    result, _ = await _evaluate(app_config, sample_product, case, sample_reference, bad, reply)
    assert result.verdict is Verdict.FAIL
    assert _scores(result)["rules_adherence"] == 0.0
    (bug,) = result.bug_candidates
    assert bug.finding_ids == ["F_001"] and bug.product_id == "sample_nl2sql"
    assert "JOIN-01" in result.findings[0].related_tags


async def test_sub_perfect_score_without_finding_is_retried(
    app_config: AppConfig, sample_product: ProductDescription, case: Any, sample_reference: Any
) -> None:
    missing = draft(schema_faithfulness=score(0.5, "Some identifiers unknown."))
    fixed = draft(
        schema_faithfulness=score(0.5, "Some identifiers unknown."),
        findings=[finding("schema_faithfulness", "c.amount")],
    )
    result, llm = await _evaluate(
        app_config, sample_product, case, sample_reference, ANSWER, missing, fixed
    )
    assert len(llm.calls) == 2
    assert "category 'schema_faithfulness' is required" in llm.calls[1].user
    assert result.verdict is Verdict.FAIL


async def test_metric_boundaries_are_enforced(
    app_config: AppConfig, sample_product: ProductDescription, case: Any, sample_reference: Any
) -> None:
    # A schema allow-list problem reported under semantic correctness is sent back.
    wrong_bucket = draft(
        semantic_correctness=score(0.5, "Uses an undocumented column in the select."),
        findings=[finding("semantic_correctness", "SUM(c.amount)")],
    )
    result, llm = await _evaluate(
        app_config,
        sample_product,
        case,
        sample_reference,
        ANSWER,
        wrong_bucket,
        draft(),
    )
    assert "belong only to schema faithfulness" in llm.calls[1].user
    assert result.verdict is Verdict.PASS


async def test_sql_output_alias_is_never_a_schema_penalty(
    app_config: AppConfig, sample_product: ProductDescription, case: Any, sample_reference: Any
) -> None:
    alias_penalty = draft(
        schema_faithfulness=score(0.5, "total_spend is not an allowed column."),
        findings=[finding("schema_faithfulness", "total_spend")],
    )
    result, llm = await _evaluate(
        app_config, sample_product, case, sample_reference, ANSWER, alias_penalty
    )
    assert len(llm.calls) == 1  # repaired in code, no retry needed
    assert _scores(result)["schema_faithfulness"] == 1.0 and result.verdict is Verdict.PASS


async def test_answer_without_sql_fails_semantic_correctness_only(
    app_config: AppConfig, sample_product: ProductDescription, case: Any, sample_reference: Any
) -> None:
    text_only = {"outcome": "answered", "text": "I cannot build that query."}
    reply = draft(
        semantic_correctness=score(0, "No SQL."),
        findings=[finding("semantic_correctness", "I cannot build that query.")],
    )
    result, _ = await _evaluate(
        app_config, sample_product, case, sample_reference, text_only, reply
    )
    assert result.verdict is Verdict.FAIL and NO_SQL_REASON in result.notes
    assert _scores(result) == {
        "semantic_correctness": 0.0,
        "rules_adherence": "not_applicable",
        "schema_faithfulness": "not_applicable",
    }


@pytest.mark.parametrize(
    ("response", "no_sql_note"),
    [
        ({"outcome": "clarification", "message": "Which period?"}, False),
        ({"outcome": "denied", "message": "I cannot help with that."}, False),
        ({"outcome": "error", "error_type": "Boom"}, False),
        ({"outcome": "timeout"}, True),
    ],
)
async def test_non_answers_are_recorded_only_without_llm(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Any,
    response: dict[str, Any],
    no_sql_note: bool,
) -> None:
    result, llm = await _evaluate(app_config, sample_product, case, sample_reference, response)
    assert result.verdict is None and result.evaluation_status == "recorded_only"
    assert result.metrics is None and not llm.calls
    assert (NO_SQL_REASON in result.notes) is no_sql_note


async def test_judge_validation_failure_is_inconclusive_with_notes(
    app_config: AppConfig, sample_product: ProductDescription, case: Any, sample_reference: Any
) -> None:
    broken = draft(semantic_correctness=score(0, "no finding given"))
    result, _ = await _evaluate(
        app_config, sample_product, case, sample_reference, ANSWER, broken, broken, broken
    )
    assert result.verdict is Verdict.INCONCLUSIVE and result.metrics is None
    assert "evaluation failed after 3 attempt(s)" in result.notes[0]


async def test_judge_provider_outage_is_inconclusive_not_a_crash(
    app_config: AppConfig, sample_product: ProductDescription, case: Any, sample_reference: Any
) -> None:
    result, _ = await _evaluate(app_config, sample_product, case, sample_reference, ANSWER)
    assert result.verdict is Verdict.INCONCLUSIVE
    assert "evaluation provider call failed" in result.notes[0]


async def test_dummy_cg_id_filter_is_hidden_from_judge_only(
    app_config: AppConfig, sample_product: ProductDescription, case: Any, sample_reference: Any
) -> None:
    sql = SQL.replace("GROUP BY", "WHERE c.cg_id IN ('DUMMY') AND c.amount > 0 GROUP BY")
    response = {**ANSWER, "artifacts": [{"kind": "sql", "content": sql}]}
    settings = app_config.model_copy(
        update={
            "evaluation": app_config.evaluation.model_copy(
                update={"ignore_dummy_cg_id_filter": True}
            )
        }
    )
    result, llm = await _evaluate(
        settings, sample_product, case, sample_reference, response, draft()
    )
    assert "DUMMY" not in llm.calls[0].user and "c.amount > 0" in llm.calls[0].user
    assert any("cg_id IN ('DUMMY')" in note for note in result.notes)


def test_dummy_filter_removal_keeps_other_predicates() -> None:
    def judged(sql: str) -> str:
        response = AnsweredResponse(artifacts=[{"kind": "sql", "content": sql}])  # type: ignore[list-item]
        cleaned, _ = without_ignored_sql_workarounds(response)
        content = cleaned.artifacts[0].content
        assert isinstance(content, str)
        return content

    assert judged("SELECT 1 FROM t WHERE cg_id IN ('DUMMY')") == "SELECT 1 FROM t "
    assert judged("SELECT 1 FROM t WHERE a = 1 AND t.cg_id IN ('DUMMY')") == (
        "SELECT 1 FROM t WHERE a = 1"
    )
    assert judged("SELECT 1 FROM t WHERE cg_id IN ('X')") == "SELECT 1 FROM t WHERE cg_id IN ('X')"


def test_rules_and_schema_not_applicable_without_a_pack() -> None:
    parsed = EvaluationDraft.model_validate(draft())
    metrics = compute_metrics(parsed, rules_applicable=False, schema_applicable=False)
    statuses = {name.value: m.status.value for name, m in metrics.by_name().items()}
    assert statuses == {
        "semantic_correctness": "scored",
        "rules_adherence": "not_applicable",
        "schema_faithfulness": "not_applicable",
    }
    assert failing_metrics(metrics) == []


def test_scores_are_three_levels_only() -> None:
    with pytest.raises(ValueError, match="0, 0.5 or 1"):
        EvaluationDraft.model_validate(draft(semantic_correctness=score(0.7)))


def test_legacy_draft_names_are_migrated() -> None:
    parsed = EvaluationDraft.model_validate(
        {
            "correctness": score(1),
            "relevancy": score(1),
            "criteria": [],
            "findings": [],
            "overall_verdict": "pass",
        }
    )
    assert parsed.semantic_correctness.score == 1 and parsed.rules_adherence.score is None


async def test_findings_matching_known_gaps_are_flagged(
    app_config: AppConfig, sample_product: ProductDescription, case: Any, sample_reference: Any
) -> None:
    gap = {
        **finding("semantic_correctness", "SUM(c.amount)"),
        "description": "Ignores currency_code when summing amounts across currencies",
    }
    reply = draft(semantic_correctness=score(0.5, "Mixes currencies."), findings=[gap])
    result, _ = await _evaluate(app_config, sample_product, case, sample_reference, ANSWER, reply)
    assert result.findings[0].matches_known_gap is True


def test_prompt_shows_tagged_cheat_sheet_and_not_the_scenario_goal(
    sample_product: ProductDescription, case: Any, sample_reference: Reference
) -> None:
    scenario, question, pack = case
    response = AnsweredResponse.model_validate(ANSWER)
    system, user = build_response_evaluation_prompt(
        sample_product, scenario, question, response, (pack, sample_reference)
    )
    assert "TABLE-01" in user and "CRIT-01" in user and "JOIN-01" in user
    assert scenario.business_goal not in user
    assert "semantic_correctness" in user and "relevancy" not in user


# --- Agent ---------------------------------------------------------------------------------


def test_optional_agent_metrics_need_exposed_telemetry() -> None:
    enabled = list(AgentMetric)
    plain = AnsweredResponse(text="Complaints rose 10% in March.")
    plan = plan_agent_metrics(enabled, plain, None)
    assert plan.applicable == (MetricName.RELEVANCE, MetricName.COMPLETION)
    assert set(plan.not_applicable) == {
        MetricName.TOOL_SELECTION,
        MetricName.FAITHFULNESS,
        MetricName.CONTEXT_PRECISION,
        MetricName.CONTEXT_RECALL,
    }
    traced = AnsweredResponse.model_validate(
        {
            "text": "Complaints rose 10% in March.",
            "artifacts": [
                {"kind": "tool_calls", "content": [{"name": "aggregate", "input": {}}]},
                {"kind": "citations", "content": [{"source": "complaints"}]},
            ],
        }
    )
    plan = plan_agent_metrics(enabled, traced, None, has_expected_context=True)
    assert set(plan.applicable) == {MetricName(m.value) for m in AgentMetric}


def test_agent_metrics_cannot_disable_required_ones() -> None:
    with pytest.raises(ValueError, match="cannot disable required"):
        plan_agent_metrics([AgentMetric.TOOL_SELECTION], AnsweredResponse(text="x"), None)


def test_agent_draft_must_match_applicable_metrics() -> None:
    applicable = (MetricName.RELEVANCE, MetricName.COMPLETION)
    reply = AgentEvaluationDraft.model_validate(
        {
            "relevance": score(1),
            "completion": score(0.5),
            "tool_selection": score(1),
            "criteria": [],
            "findings": [],
            "overall_verdict": "pass",
        }
    )
    errors = check_agent_evaluation(reply, ["answers"], "the answer", applicable)
    assert any("tool_selection: must be omitted" in e for e in errors)
    assert any("category 'completion' is required" in e for e in errors)
    assert any("missing ['answers']" in e for e in errors)


async def test_agent_profile_evaluates_relevance_and_completion(
    app_config: AppConfig, sample_chatbot: ProductDescription, case: Any
) -> None:
    scenario, question, _ = case
    scenario = scenario.model_copy(
        update={
            "mode": "ungrounded",
            "reference_id": None,
            "reference_pack_id": None,
            "evaluation_profile": EvaluationProfile.AGENT,
        }
    )
    question = question.model_copy(
        update={
            "mode": "ungrounded",
            "reference_pack_id": None,
            "evaluation_profile": EvaluationProfile.AGENT,
        }
    )
    reply = {
        "relevance": score(1),
        "completion": score(1),
        "criteria": criteria("met", "met"),
        "findings": [],
        "overall_verdict": "pass",
    }
    llm = FakeLLMClient([reply])
    evaluator = Evaluator(
        llm,
        app_config,
        logging.getLogger("t"),
        reference=None,
        profile=EvaluationProfile.AGENT,
        agent_metrics=list(AgentMetric),
    )
    answer = {"outcome": "answered", "text": "Here are the top spenders."}
    result = await evaluator.evaluate(
        EvaluationCase(question, scenario, None, _execution(answer)), sample_chatbot
    )
    assert "Agent response" in llm.calls[0].system
    assert result.verdict is Verdict.PASS and result.evaluation_profile == "agent"
    assert _scores(result) == {
        "relevance": 1.0,
        "completion": 1.0,
        "tool_selection": "not_applicable",
        "faithfulness": "not_applicable",
        "context_precision": "not_applicable",
        "context_recall": "not_applicable",
    }


def test_compute_agent_metrics_marks_unobservable_metrics() -> None:
    plan = plan_agent_metrics(
        [AgentMetric.RELEVANCE, AgentMetric.COMPLETION, AgentMetric.FAITHFULNESS],
        AnsweredResponse(text="x"),
        None,
    )
    reply = AgentEvaluationDraft.model_validate(
        {
            "relevance": score(1),
            "completion": score(None),
            "criteria": [],
            "findings": [],
            "overall_verdict": "inconclusive",
        }
    )
    metrics = compute_agent_metrics(reply, plan)
    assert metrics.completion is not None and metrics.completion.status == "inconclusive"
    assert metrics.faithfulness is not None and metrics.faithfulness.status == "not_applicable"
