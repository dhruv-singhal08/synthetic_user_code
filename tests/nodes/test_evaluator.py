"""Evaluation against the frozen metric configuration: deterministic applicability, the two
independent judge groups, response classification / outcome routing, MRR, citations, tool
trace, custom metrics, the SQL parser evidence and the provisional verdict policy."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import LLMProviderError
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.metrics.compiler import custom_metric_id
from synthetic_user.metrics.registry import builtin_metric_config, freeze
from synthetic_user.models.evaluation import (
    EvaluationResult,
    EvidenceStatus,
    Finding,
    FindingSource,
    MetricStatus,
    Severity,
    Verdict,
)
from synthetic_user.models.execution import ExecutionRecord
from synthetic_user.models.metrics import (
    CustomMetricInput,
    EvidenceKind,
    ExpectationField,
    ExpectationFieldType,
    JudgeGroup,
    MetricCategory,
    MetricDefinition,
    MetricOrigin,
    MetricSelection,
    ResolvedMetricConfig,
)
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator import EvaluationCase, Evaluator
from synthetic_user.nodes.evaluator.no_sql import NO_SQL_REASON
from synthetic_user.nodes.evaluator.retrieval_judge import reciprocal_rank
from synthetic_user.nodes.evaluator.verdict import SeverityVerdictPolicy
from tests_support import evaluation_context, failed_check, pack_spec, scripted_retrieval_reply

CRITERIA = ["names the employees ranked by total card spend", "covers the requested time period"]
JOIN = "c.employee_id = e.employee_id"
SQL = (
    "SELECT e.full_name, SUM(c.amount) AS total_spend FROM card_transactions c "
    f"JOIN employees e ON {JOIN} GROUP BY e.full_name ORDER BY total_spend DESC LIMIT 10"
)
ANSWER: dict[str, Any] = {
    "outcome": "answered",
    "text": "Here is the SQL query for the top spenders.",
    "artifacts": [{"kind": "sql", "content": SQL}],
}
SQL_METRICS = (
    "relevance",
    "correctness_completeness",
    "semantic_correctness",
    "rules_adherence",
    "schema_faithfulness",
)
CHUNKS = [
    {
        "chunk_id": "c1",
        "rank": 1,
        "text": "January complaints: 120 about delays.",
        "source_id": "d1",
    },
    {"chunk_id": "c2", "rank": 2, "text": "Office party menu.", "source_id": "d2"},
    {"chunk_id": "c3", "rank": 3, "text": "February complaints: 90 about fees.", "source_id": "d1"},
]


def score(value: float | None, rationale: str = "fine", **extra: Any) -> dict[str, Any]:
    return {"score": value, "rationale": rationale, **extra}


def finding(category: str, evidence: str | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "category": category,
        "description": f"{category} problem",
        "expected_behavior": "expected",
        "observed_behavior": "observed",
        "evidence": evidence,
        "severity": "high",
        "evidence_status": "supported",
        **extra,
    }


def general(metrics: dict[str, Any], **extra: Any) -> dict[str, Any]:
    reply: dict[str, Any] = {
        "metrics": metrics,
        "findings": [],
        "overall_verdict": "pass",
        "goal_completion": {"status": "completed", "reason": "the goal is met"},
    }
    reply.update(extra)
    return reply


JUSTIFIED = {"status": "justified_decline", "reason": "The contract lists it as not supported."}


def all_pass(*metric_ids: str, classify: bool = True) -> dict[str, Any]:
    metrics = {}
    for metric_id in metric_ids:
        metrics[metric_id] = score(1)
        if metric_id == "correctness_completeness":
            metrics[metric_id]["factual_verification"] = "unverified"
    extra = {"response_classification": "answered"} if classify else {}
    return general(metrics, **extra)


@pytest.fixture
def case(
    sample_scenarios: list[Scenario], sample_packs: list[ReferencePack]
) -> tuple[Scenario, QuestionCase, ReferencePack]:
    scenario, pack = sample_scenarios[0], sample_packs[0]
    question = _question(scenario, pack)
    return scenario, question, pack


def _question(
    scenario: Scenario, pack: ReferencePack | None, **metric_expectations: Any
) -> QuestionCase:
    return QuestionCase.model_validate(
        {
            "question_id": "Q_001",
            "scenario_id": scenario.id,
            "persona_id": scenario.persona_id,
            "mode": scenario.mode,
            "reference_pack_id": pack.pack_id if pack else None,
            "utterance": "Who were our top 10 card spenders last quarter?",
            "intended_goal": scenario.business_goal,
            "intent": {"task_type": "ranking"},
            "question_characteristics": {
                "style": "precise",
                "ambiguity_level": "low",
                "complexity": "simple",
            },
            "evaluation_context": evaluation_context(
                CRITERIA,
                mapping={
                    metric_id: {"comparison_basis": "test", "expectation": expectation}
                    for metric_id, expectation in metric_expectations.items()
                },
                reference_spec=pack_spec(pack) if pack else None,
            ),
        }
    )


def _execution(response: dict[str, Any], context: dict[str, Any] | None = None) -> ExecutionRecord:
    return ExecutionRecord.model_validate(
        {
            "execution_id": "EXE_001",
            "question_id": "Q_001",
            "product_id": "sample_nl2sql",
            "adapter": "fixture",
            "submitted_question": "Who were our top 10 card spenders last quarter?",
            "started_at": datetime(2026, 10, 18, tzinfo=UTC),
            "response": response,
            **({"context": context} if context else {}),
        }
    )


def _context(chunks: list[dict[str, Any]] = CHUNKS, **extra: Any) -> dict[str, Any]:
    return {
        "availability": "available",
        "source_mode": "response_field",
        "chunks": chunks,
        "rank_exposed": bool(chunks) and all(c.get("rank") for c in chunks),
        **extra,
    }


def dispatch(
    general_reply: dict[str, Any] | Callable[[str, str], Any] | None = None,
    retrieval_reply: dict[str, Any] | Callable[[str, str], Any] | None = None,
) -> Callable[[str, str], Any]:
    """One FakeLLM responder for both judges, chosen by the system prompt."""

    def respond(system: str, user: str) -> Any:
        reply = retrieval_reply if "Retrieval and Grounding Judge" in system else general_reply
        assert reply is not None, f"unexpected judge call: {system[:60]}"
        return reply(system, user) if callable(reply) else reply

    return respond


async def evaluate(
    settings: AppConfig,
    product: ProductDescription,
    case: tuple[Scenario, QuestionCase, ReferencePack | None],
    reference: Reference | None,
    response: dict[str, Any],
    replies: list[Any],
    *,
    metrics: tuple[str, ...] | ResolvedMetricConfig = SQL_METRICS,
    context: dict[str, Any] | None = None,
) -> tuple[EvaluationResult, FakeLLMClient]:
    scenario, question, _ = case  # the question carries its own cheat sheet (D240)
    config = (
        metrics
        if isinstance(metrics, ResolvedMetricConfig)
        else builtin_metric_config(list(metrics))
    )
    llm = FakeLLMClient(replies)
    evaluator = Evaluator(
        llm, settings, logging.getLogger("t"), reference=reference, metric_config=config
    )
    result = await evaluator.evaluate(
        EvaluationCase(question, scenario, _execution(response, context)), product
    )
    return result, llm


def statuses(result: EvaluationResult) -> dict[str, Any]:
    return {
        m: (r.score if r.status is MetricStatus.SCORED else r.status.value)
        for m, r in result.metrics.items()
    }


# --- general judge: SQL metrics, parser evidence, classification -------------------------------


async def test_correct_sql_passes_all_selected_metrics_in_one_general_call(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    result, llm = await evaluate(
        app_config, sample_product, case, sample_reference, ANSWER, [all_pass(*SQL_METRICS)]
    )
    assert len(llm.calls) == 1 and "General Judge" in llm.calls[0].system
    assert statuses(result) == dict.fromkeys(SQL_METRICS, 1.0)
    assert result.verdict is Verdict.PASS and result.verdict_policy == "severity-v1 (fail at high+)"
    assert result.verdict_reason == "Passed: every scored metric met its expectations"
    assert "choose the lower level when" in llm.calls[0].user  # severity defined for the judge
    assert (result.response_classification, result.classification_source) == (
        "answered",
        "general_judge",
    )
    relevance = result.metrics["relevance"]
    assert relevance.comparison_basis == "question_vs_response"
    assert relevance.verification_method == "response_review"
    assert [g.guidance_id for g in relevance.guidance_applied] == ["METRIC:relevance"]
    assert result.limitations[0].startswith("Static SQL analysis only")
    prompt = llm.calls[0].user
    # parser output is supporting evidence: base identifiers, aliases kept apart, joins
    assert '"base_tables": [' in prompt and '"card_transactions"' in prompt
    sql_section = prompt.split("SQL analysis (parser output", 1)[1].split("Traceability", 1)[0]
    assert '"output_aliases"' in sql_section and '"total_spend"' in sql_section
    assert JOIN in prompt
    assert "Also classify the response kind" in prompt
    # factual correctness is never asserted without trusted evidence
    details = result.metrics["correctness_completeness"].details
    assert (
        details["factual_verification"] == "unverified"
        and "not (fully) verified" in details["limitation"]
    )


async def test_rule_violation_fails_with_bug(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    reply = all_pass(*SQL_METRICS)
    reply["metrics"]["rules_adherence"] = score(
        0, "wrong join", evidence=JOIN, checks=[failed_check("rules_adherence", "SQL:1")]
    )
    reply["findings"] = [finding("rules_adherence", JOIN)]
    result, _ = await evaluate(app_config, sample_product, case, sample_reference, ANSWER, [reply])
    assert result.verdict is Verdict.FAIL
    assert [b.finding_ids for b in result.bug_candidates] == [["F_001"]]
    assert result.findings[0].source == "llm"


async def test_answer_without_sql_scores_zero_with_deterministic_finding(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    text_only = {"outcome": "answered", "text": "Top spenders: Ann, Bob."}
    reply = all_pass("relevance", "correctness_completeness")
    result, llm = await evaluate(
        app_config, sample_product, case, sample_reference, text_only, [reply]
    )
    assert (
        "semantic_correctness"
        not in llm.calls[0].user.split("## METRICS", 1)[1].split("## RULES")[0]
    )
    assert statuses(result) == {
        "relevance": 1.0,
        "correctness_completeness": 1.0,
        "semantic_correctness": 0.0,
        "rules_adherence": "not_applicable",
        "schema_faithfulness": "not_applicable",
    }
    assert result.verdict is Verdict.FAIL and NO_SQL_REASON in result.notes
    assert result.findings[0].source == "deterministic"
    assert result.findings[0].description == NO_SQL_REASON


async def test_unable_to_answer_without_sql_is_not_a_sql_failure(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    decline = {"outcome": "answered", "text": "I cannot forecast spend."}
    reply = all_pass("relevance", "correctness_completeness")
    reply["response_classification"] = "unable_to_answer"
    reply["goal_completion"] = JUSTIFIED
    result, _ = await evaluate(app_config, sample_product, case, sample_reference, decline, [reply])
    assert result.product_outcome == "answered"  # the executor's outcome is never overwritten
    assert result.response_classification == "unable_to_answer"
    assert statuses(result)["semantic_correctness"] == "not_applicable"
    assert result.verdict is Verdict.PASS  # a justified limitation can be correct


async def test_typed_adapter_classification_needs_no_judge_classification(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    typed = {**ANSWER, "response_classification": "answered"}
    result, llm = await evaluate(
        app_config,
        sample_product,
        case,
        sample_reference,
        typed,
        [all_pass(*SQL_METRICS, classify=False)],
    )
    assert "Also classify the response kind" not in llm.calls[0].user
    assert (result.response_classification, result.classification_source) == ("answered", "adapter")


async def test_substantive_denial_is_judged_on_answer_level_metrics_only(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    denial = {"outcome": "denied", "message": "Forecasting is not supported yet."}
    reply = all_pass("relevance", "correctness_completeness", classify=False)
    reply["goal_completion"] = JUSTIFIED
    result, llm = await evaluate(
        app_config, sample_product, case, sample_reference, denial, [reply]
    )
    assert len(llm.calls) == 1 and "typed denial" in llm.calls[0].user
    assert (result.product_outcome, result.response_classification) == (
        "denied",
        "unable_to_answer",
    )
    assert result.classification_source == "typed_outcome"
    assert statuses(result) == {
        "relevance": 1.0,
        "correctness_completeness": 1.0,
        "semantic_correctness": "not_applicable",
        "rules_adherence": "not_applicable",
        "schema_faithfulness": "not_applicable",
    }
    assert "no sql to assess" in result.metrics["semantic_correctness"].rationale.lower()
    assert result.verdict is Verdict.PASS and result.evaluation_status == "evaluated"


@pytest.mark.parametrize(
    "response",
    [
        {"outcome": "clarification", "message": "Which quarter?"},
        {"outcome": "error", "error_type": "HttpError"},
        {"outcome": "timeout", "timeout_seconds": 30},
    ],
)
async def test_unresolved_clarification_error_and_timeout_never_invoke_judges(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
    response: dict[str, Any],
) -> None:
    result, llm = await evaluate(app_config, sample_product, case, sample_reference, response, [])
    assert llm.calls == [] and result.verdict is None and result.metrics == {}
    assert result.evaluation_status == "recorded_only"
    expected = "clarification" if response["outcome"] == "clarification" else None
    assert result.response_classification == expected
    if response["outcome"] == "timeout":
        assert result.notes == [NO_SQL_REASON]  # an SQL metric was selected


async def test_rules_and_schema_not_applicable_without_a_pack(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
) -> None:
    scenario = sample_scenarios[0].model_copy(
        update={"mode": "ungrounded", "reference_id": None, "reference_pack_id": None}
    )
    case = (scenario, _question(scenario, None), None)
    reply = all_pass("relevance", "correctness_completeness", "semantic_correctness")
    result, llm = await evaluate(app_config, sample_product, case, None, ANSWER, [reply])
    assert statuses(result)["rules_adherence"] == "not_applicable"
    assert statuses(result)["schema_faithfulness"] == "not_applicable"
    assert "Cheat sheet of this question" not in llm.calls[0].user


async def test_sub_perfect_score_without_finding_is_retried(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    bad = all_pass(*SQL_METRICS)
    bad["metrics"]["relevance"] = score(0.5, "partly")
    result, llm = await evaluate(
        app_config, sample_product, case, sample_reference, ANSWER, [bad, all_pass(*SQL_METRICS)]
    )
    assert len(llm.calls) == 2 and "relevance is scored 0.5" in llm.calls[1].user
    assert result.verdict is Verdict.PASS


async def test_evidence_must_be_an_exact_quote(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    bad = all_pass(*SQL_METRICS)
    bad["metrics"]["relevance"] = score(1, evidence="a quote that is not there")
    result, llm = await evaluate(
        app_config, sample_product, case, sample_reference, ANSWER, [bad, all_pass(*SQL_METRICS)]
    )
    assert "is not in the source" in llm.calls[1].user and result.verdict is Verdict.PASS


async def test_output_alias_is_never_a_schema_penalty(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    """Parser-based (not regex): an output alias / CTE / function cannot lower schema."""
    bad = all_pass(*SQL_METRICS)
    bad["metrics"]["schema_faithfulness"] = score(0.5, "unknown column", evidence="total_spend")
    bad["findings"] = [finding("schema_faithfulness", "total_spend")]
    result, llm = await evaluate(
        app_config, sample_product, case, sample_reference, ANSWER, [bad, all_pass(*SQL_METRICS)]
    )
    assert "is an SQL output alias, CTE name or function" in llm.calls[1].user
    assert statuses(result)["schema_faithfulness"] == 1.0


async def test_judge_failure_is_evaluation_error_never_a_score(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    result, _ = await evaluate(
        app_config, sample_product, case, sample_reference, ANSWER, ["no json"] * 3
    )
    assert set(statuses(result).values()) == {"evaluation_error"}
    assert result.metrics["relevance"].failure_source == "general judge"
    assert result.verdict is Verdict.INCONCLUSIVE
    assert any("general judge output invalid after 3 attempt(s)" in n for n in result.notes)
    outage, _ = await evaluate(
        app_config, sample_product, case, sample_reference, ANSWER, [LLMProviderError("down")]
    )
    assert outage.verdict is Verdict.INCONCLUSIVE
    assert any("provider call failed" in n for n in outage.notes)
    assert any("the General Judge call failed" in n for n in outage.notes)


async def test_unrequested_filter_is_judged_as_written_and_unresolved_never_passes(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    """D232: no predicate is hidden; an unrequested filter nothing establishes is reported as
    an unresolved finding: the verdict is inconclusive (never pass, never a bug)."""
    sql = f"{SQL.split(' GROUP BY')[0]} WHERE c.cg_id IN ('DUMMY') GROUP BY e.full_name"
    response = {"outcome": "answered", "artifacts": [{"kind": "sql", "content": sql}]}
    reply = all_pass(*SQL_METRICS)
    reply["findings"] = [
        finding(
            "semantic_correctness",
            "c.cg_id IN ('DUMMY')",
            severity="medium",
            evidence_status="weak",
            unresolved=True,
        )
    ]
    result, llm = await evaluate(
        app_config, sample_product, case, sample_reference, response, [reply]
    )
    judged_sql = llm.calls[0].user.split("Product response", 1)[1]
    assert "DUMMY" in judged_sql  # the SQL is judged exactly as generated
    assert '"unresolved": true' in llm.calls[0].user  # the option is explained to the judge
    assert result.findings[0].unresolved
    assert result.verdict is Verdict.INCONCLUSIVE and result.bug_candidates == []
    assert not any("Ignored configured SQL predicate" in n for n in result.notes)


# --- retrieval / grounding judge ---------------------------------------------------------------


RETRIEVAL = ("context_precision", "context_recall", "mrr", "faithfulness")


def _recall_case(case: Any) -> tuple[Scenario, QuestionCase, ReferencePack]:
    scenario, _q, pack = case
    question = _question(
        scenario,
        pack,
        context_recall={
            "status": "generated",
            "value": {"required_information": ["January complaints", "February complaints"]},
        },
    )
    return scenario, question, pack


async def test_two_independent_judges_and_mrr_computed_by_code(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    metrics = ("relevance", *RETRIEVAL)

    def retrieval(system: str, user: str) -> dict[str, Any]:
        reply = scripted_retrieval_reply(user, relevant={"c3"})  # first relevant at rank 3
        reply["required_information"][1]["status"] = "covered"
        return reply

    result, llm = await evaluate(
        app_config,
        sample_product,
        _recall_case(case),
        sample_reference,
        {"outcome": "answered", "text": "January had 120 complaints, February 90."},
        [dispatch(all_pass("relevance"), retrieval)] * 2,
        metrics=metrics,
        context=_context(),
    )
    systems = [c.system for c in llm.calls]
    assert sum("General Judge" in s for s in systems) == 1
    assert sum("Retrieval and Grounding Judge" in s for s in systems) == 1  # never merged
    retrieval_prompt = next(c.user for c in llm.calls if "Retrieval" in c.system)
    assert "per-chunk relevance labels" in retrieval_prompt
    assert '"mrr"' not in retrieval_prompt.split("## OUTPUT CONTRACT", 1)[1]
    general_prompt = next(c.user for c in llm.calls if "General Judge" in c.system)
    assert "Office party menu" not in general_prompt  # retrieval evidence stays separate
    mrr = result.metrics["mrr"]
    assert (mrr.score, mrr.details["first_relevant_rank"]) == (pytest.approx(1 / 3, abs=1e-6), 3)
    assert mrr.judge_group == "retrieval_grounding"
    recall = result.metrics["context_recall"]
    assert recall.details["disclosure"].startswith("Expectation-based coverage")
    assert [i["item"] for i in recall.details["required_information"]] == [
        "January complaints",
        "February complaints",
    ]


async def test_only_retrieval_metrics_make_one_call_and_leave_classification_unknown(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    result, llm = await evaluate(
        app_config,
        sample_product,
        case,
        sample_reference,
        {"outcome": "answered", "text": "January had 120 complaints."},
        [dispatch(None, lambda s, u: scripted_retrieval_reply(u))],
        metrics=("context_precision", "faithfulness"),
        context=_context(),
    )
    assert len(llm.calls) == 1 and "Retrieval" in llm.calls[0].system
    assert (result.response_classification, result.classification_source) == (
        "unknown",
        "unresolved",
    )
    assert any("classification unresolved" in n for n in result.notes)


async def test_missing_context_is_evidence_unavailable_and_makes_no_call(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    result, llm = await evaluate(
        app_config,
        sample_product,
        case,
        sample_reference,
        {"outcome": "answered", "text": "x"},
        [],
        metrics=RETRIEVAL,
    )
    assert llm.calls == []  # zero applicable LLM metrics -> zero judge calls
    assert statuses(result) == {
        "context_precision": "evidence_unavailable",
        "context_recall": "cannot_generate",  # this question has no required-information list
        "mrr": "evidence_unavailable",
        "faithfulness": "evidence_unavailable",
    }
    assert "No context source is configured" in result.metrics["mrr"].rationale
    assert result.verdict is Verdict.INCONCLUSIVE  # missing telemetry is never passing evidence


async def test_failed_context_is_reported_with_its_reason(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    failed = {"availability": "failed", "source_mode": "context_api", "error": "HTTP 500"}
    result, _ = await evaluate(
        app_config,
        sample_product,
        case,
        sample_reference,
        {"outcome": "answered", "text": "x"},
        [],
        metrics=("faithfulness",),
        context=failed,
    )
    assert result.metrics["faithfulness"].rationale == "Retrieved context failed: HTTP 500"


async def test_empty_retrieval_differs_from_missing_telemetry(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    result, llm = await evaluate(
        app_config,
        sample_product,
        _recall_case(case),
        sample_reference,
        {"outcome": "answered", "text": "x"},
        [],
        metrics=("context_precision", "context_recall", "mrr"),
        context=_context([]),
    )
    assert llm.calls == []
    assert statuses(result) == {
        "context_precision": "not_applicable",  # precision undefined with zero chunks
        "context_recall": 0.0,  # nothing retrieved covers the checklist
        "mrr": 0.0,  # empty ranked results: reciprocal rank 0, inside the denominator
    }
    assert [f.category for f in result.findings] == ["context_recall"]  # MRR has no finding


async def test_unranked_chunks_never_give_an_mrr(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    unranked = [{k: v for k, v in c.items() if k != "rank"} for c in CHUNKS]
    result, llm = await evaluate(
        app_config,
        sample_product,
        case,
        sample_reference,
        {"outcome": "answered", "text": "x"},
        [],
        metrics=("mrr",),
        context=_context(unranked),
    )
    assert llm.calls == [] and statuses(result) == {"mrr": "evidence_unavailable"}
    assert result.metrics["mrr"].missing_requirement == "ranked_context"


def test_reciprocal_rank_rules() -> None:
    from synthetic_user.models.execution import RetrievedContext

    ranked = RetrievedContext.model_validate(_context())
    assert reciprocal_rank(ranked, {"c1": True, "c2": False, "c3": True}).value == 1.0
    assert reciprocal_rank(ranked, {"c1": False, "c2": False, "c3": True}).value == pytest.approx(
        1 / 3, abs=1e-6
    )
    none = reciprocal_rank(ranked, {"c1": False, "c2": False, "c3": False})
    assert none is not None and (none.value, none.first_relevant_rank) == (0.0, None)
    assert reciprocal_rank(ranked, {"c1": True}) is None  # an unlabelled chunk -> no rank
    unranked = RetrievedContext.model_validate(
        _context([{k: v for k, v in c.items() if k != "rank"} for c in CHUNKS])
    )
    assert reciprocal_rank(unranked, {"c1": True, "c2": True, "c3": True}) is None


async def test_recall_score_must_match_the_checklist(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    def inconsistent(system: str, user: str) -> dict[str, Any]:
        reply = scripted_retrieval_reply(user)
        reply["required_information"][1]["status"] = "not_covered"  # but score stays 1
        return reply

    result, llm = await evaluate(
        app_config,
        sample_product,
        _recall_case(case),
        sample_reference,
        {"outcome": "answered", "text": "x"},
        [dispatch(None, inconsistent)] * 3,
        metrics=("context_recall",),
        context=_context(),
    )
    assert "score 1 needs every checklist item covered" in llm.calls[1].user
    assert statuses(result) == {"context_recall": "evaluation_error"}


# --- citations -------------------------------------------------------------------------------


def _citation_case(case: Any, required: bool) -> tuple[Scenario, QuestionCase, ReferencePack]:
    scenario, _q, pack = case
    return (
        scenario,
        _question(
            scenario,
            pack,
            citation_correctness={"status": "generated", "value": {"citations_required": required}},
        ),
        pack,
    )


async def test_citations_without_mapping_are_evidence_unavailable(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    result, _ = await evaluate(
        app_config,
        sample_product,
        _citation_case(case, True),
        sample_reference,
        {"outcome": "answered", "text": "x [1]"},
        [],
        metrics=("citation_correctness",),
        context=_context(),
    )
    assert statuses(result) == {"citation_correctness": "evidence_unavailable"}
    assert "no citation mapping" in result.metrics["citation_correctness"].rationale


@pytest.mark.parametrize(("required", "expected"), [(True, 0.0), (False, "not_applicable")])
async def test_missing_citations_fail_only_when_required(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
    required: bool,
    expected: Any,
) -> None:
    result, llm = await evaluate(
        app_config,
        sample_product,
        _citation_case(case, required),
        sample_reference,
        {"outcome": "answered", "text": "x"},
        [],
        metrics=("citation_correctness",),
        context=_context(citations_exposed=True),
    )
    assert llm.calls == [] and statuses(result) == {"citation_correctness": expected}


async def test_citations_resolve_to_specific_chunks(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    citations = [
        {"chunk_id": "c1", "claim": "120 complaints in January", "resolved": True},
        {"chunk_id": "c9", "claim": "made up", "resolved": False},
    ]

    def retrieval(system: str, user: str) -> dict[str, Any]:
        reply = scripted_retrieval_reply(user)
        reply["citation_checks"][1]["support"] = "unsupported"
        reply["citation_checks"][1]["chunk_id"] = None
        reply["metrics"]["citation_correctness"] = score(0.5, "one unsupported citation")
        reply["findings"] = [finding("citation_correctness")]
        return reply

    result, llm = await evaluate(
        app_config,
        sample_product,
        _citation_case(case, True),
        sample_reference,
        {"outcome": "answered", "text": "January had 120 complaints [1] and more [2]."},
        [dispatch(None, retrieval)],
        metrics=("citation_correctness",),
        context=_context(citations=citations, citations_exposed=True),
    )
    assert '"resolved": false' in llm.calls[0].user  # unresolvable citation is flagged
    checks = result.metrics["citation_correctness"].details["citation_checks"]
    assert [c["support"] for c in checks] == ["supported", "unsupported"]
    assert result.verdict is Verdict.FAIL


# --- tool correctness ----------------------------------------------------------------------------


def _tool_case(tool_reference: Reference, sample_scenarios: list[Scenario]) -> Any:
    from synthetic_user.models.reference import CatalogPackDraft, TablePick
    from synthetic_user.references.pack import build_pack

    pack, _ = build_pack(
        CatalogPackDraft(tables=[TablePick(table="complaints_t", columns=["theme"], why="x")]),
        tool_reference,
        "SCN_001",
    )
    assert pack is not None
    scenario = sample_scenarios[0].model_copy(update={"reference_id": tool_reference.reference_id})
    question = _question(
        scenario,
        pack,
        tool_correctness={
            "status": "generated",
            "value": {"expected_tools": ["aggregate_feedback"]},
        },
    )
    return scenario, question, pack


async def test_tool_correctness_uses_the_exposed_trace_and_definitions(
    app_config: AppConfig,
    sample_product: ProductDescription,
    tool_reference: Reference,
    sample_scenarios: list[Scenario],
) -> None:
    trace = [
        {
            "order": 1,
            "name": "aggregate_feedback",
            "arguments": {"source": "complaints"},
            "error": "upstream timeout",
        }
    ]
    reply = general(
        {"tool_correctness": score(1, "right tool; the tool itself failed")},
        response_classification="answered",
    )
    result, llm = await evaluate(
        app_config,
        sample_product,
        _tool_case(tool_reference, sample_scenarios),
        tool_reference,
        {"outcome": "answered", "text": "The tool failed.", "tool_trace": trace},
        [reply],
        metrics=("tool_correctness",),
    )
    prompt = llm.calls[0].user
    assert "Authoritative tool definitions" in prompt and "Exposed tool trace" in prompt
    assert "a correctly targeted tool that returned an error is not wrong by itself" in prompt
    contract = prompt.split("and its comparison basis:\n", 1)[1].split("\n\n", 1)[0]
    assert '"question_specific_expectation"' in contract
    assert '"expected_tools": [\n        "aggregate_feedback"' in contract
    assert statuses(result) == {"tool_correctness": 1.0}


async def test_tool_correctness_without_trace_or_definitions(
    app_config: AppConfig,
    sample_product: ProductDescription,
    tool_reference: Reference,
    sample_scenarios: list[Scenario],
    case: Any,
    sample_reference: Reference,
) -> None:
    no_trace, llm = await evaluate(
        app_config,
        sample_product,
        _tool_case(tool_reference, sample_scenarios),
        tool_reference,
        {"outcome": "answered", "text": "x"},
        [],
        metrics=("tool_correctness",),
    )
    assert llm.calls == [] and statuses(no_trace) == {"tool_correctness": "evidence_unavailable"}
    assert "No tool-call trace" in no_trace.metrics["tool_correctness"].rationale
    # the sample reference has no tool definitions -> cannot_generate, never guessed
    no_definitions, _ = await evaluate(
        app_config,
        sample_product,
        case,
        sample_reference,
        {"outcome": "answered", "text": "x", "tool_trace": []},
        [],
        metrics=("tool_correctness",),
    )
    assert statuses(no_definitions) == {"tool_correctness": "cannot_generate"}


# --- custom metrics ----------------------------------------------------------------------------


def _custom(
    index: int, group: JudgeGroup, evidence: list[EvidenceKind], **extra: Any
) -> MetricDefinition:
    return MetricDefinition(
        metric_id=custom_metric_id(index),
        display_name=f"Custom {index}",
        category=MetricCategory.CUSTOM,
        version="1.0",
        origin=MetricOrigin.CUSTOM,
        description="d",
        rubric="r",
        score_anchors={"1": "a", "0.5": "b", "0": "c"},
        judge_group=group,
        required_execution_evidence=[EvidenceKind.ANSWER, *evidence],
        applicability_policy="always",
        evidence_requirements="answer",
        source_text=f"user text {index}",
        source_provenance="free text",
        **extra,
    )


async def test_custom_metrics_route_to_their_judge_and_need_or_skip_expectations(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    workflow = _custom(
        1,
        JudgeGroup.GENERAL,
        [],
        question_expectation_schema=[
            ExpectationField(
                name="expected_workflow", type=ExpectationFieldType.STRING_LIST, description="steps"
            )
        ],
        generation_instructions="list the workflow steps",
    )
    tone = _custom(2, JudgeGroup.GENERAL, [])
    grounded = _custom(3, JudgeGroup.RETRIEVAL_GROUNDING, [EvidenceKind.RETRIEVED_CONTEXT])
    selection = MetricSelection(
        custom_inputs=[
            CustomMetricInput(text=d.source_text or "", provenance="free text")
            for d in (workflow, tone, grounded)
        ]
    )
    config = freeze(selection, [workflow, tone, grounded], compiler_called=True, notes=[])
    scenario, _q, pack = case
    question = _question(
        scenario,
        pack,
        custom_01={"status": "generated", "value": {"expected_workflow": ["plan", "approve"]}},
    )
    general_reply = general(
        {"custom_01": score(1), "custom_02": score(1)}, response_classification="answered"
    )
    result, llm = await evaluate(
        app_config,
        sample_product,
        (scenario, question, pack),
        sample_reference,
        {"outcome": "answered", "text": "January had 120 complaints."},
        [dispatch(general_reply, lambda s, u: scripted_retrieval_reply(u))] * 2,
        metrics=config,
        context=_context(),
    )
    assert statuses(result) == {"custom_01": 1.0, "custom_02": 1.0, "custom_03": 1.0}
    general_prompt = next(c.user for c in llm.calls if "General Judge" in c.system)
    assert '"expected_workflow"' in general_prompt
    assert result.metrics["custom_03"].judge_group == "retrieval_grounding"


async def test_uncompiled_custom_metric_is_unavailable_but_others_run(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    from synthetic_user.metrics.compiler import _unavailable

    item = CustomMetricInput(text="Dialect check", provenance="free text item 1")
    selection = MetricSelection(selected_metrics=["relevance"], custom_inputs=[item])
    relevance = builtin_metric_config(["relevance"]).definitions[0]
    config = freeze(
        selection,
        [relevance, _unavailable(1, item, "no SQL dialect configured")],
        compiler_called=True,
        notes=[],
    )
    result, _ = await evaluate(
        app_config,
        sample_product,
        case,
        sample_reference,
        ANSWER,
        [all_pass("relevance")],
        metrics=config,
    )
    assert statuses(result) == {"relevance": 1.0, "custom_01": "unavailable"}
    assert result.verdict is Verdict.PASS  # an uncompiled metric never blocks the others


# --- verdict policy and serialization ------------------------------------------------------------


def _metric(score: float | None, status: MetricStatus = MetricStatus.SCORED) -> Any:
    from synthetic_user.models.evaluation import MetricResult

    return MetricResult(status=status, score=score, rationale="judged")


def _finding(
    category: str,
    severity: str,
    evidence_status: str = "supported",
    *,
    unresolved: bool = False,
) -> Finding:
    return Finding(
        finding_id="F_001",
        category=category,
        description=f"{severity} problem",
        expected_behavior="e",
        observed_behavior="o",
        severity=Severity(severity),
        evidence_status=EvidenceStatus(evidence_status),
        source=FindingSource.LLM,
        unresolved=unresolved,
    )


def test_verdict_policy_never_passes_without_a_usable_score() -> None:
    policy = SeverityVerdictPolicy()
    na = _metric(None, MetricStatus.NOT_APPLICABLE)
    missing = _metric(None, MetricStatus.EVIDENCE_UNAVAILABLE)
    decision = policy.decide({"a": na, "b": missing}, [], {})
    assert decision.verdict is Verdict.INCONCLUSIVE and "could be scored" in decision.reason
    assert policy.decide({"a": na, "b": _metric(1.0)}, [], {}).verdict is Verdict.PASS


@pytest.mark.parametrize(
    ("score", "finding", "verdict", "reason"),
    [
        # Only a proven problem at or above the fail line fails the response.
        (0.5, ("a", "high"), Verdict.FAIL, "Failed: [high] a: high problem"),
        (0.0, ("a", "critical"), Verdict.FAIL, "Failed: [critical] a: critical problem"),
        # Not too strict: a limited problem passes, with its reason listed.
        (0.5, ("a", "medium"), Verdict.PASS, "Passed with minor issues: [medium] a: medium"),
        (0.5, ("a", "low"), Verdict.PASS, "Passed with minor issues: [low] a: low problem"),
        (0.5, None, Verdict.PASS, "Passed with minor issues: a scored 0.5"),
        # A serious problem that is not proven, or a 0 without a serious finding: a person looks.
        (0.5, ("a", "high", "weak"), Verdict.INCONCLUSIVE, "not proven (weak evidence)"),
        (0.0, ("a", "medium"), Verdict.INCONCLUSIVE, "a scored 0 but no finding explains"),
        # A finding on a metric scored 1 never fails it (the judgement disagrees with itself).
        (1.0, ("a", "high"), Verdict.PASS, "Passed with minor issues: [high] a: high problem"),
    ],
)
def test_verdict_follows_finding_severity_with_a_reason(
    score: float, finding: tuple[str, ...] | None, verdict: Verdict, reason: str
) -> None:
    findings = [_finding(*finding)] if finding else []
    decision = SeverityVerdictPolicy().decide({"a": _metric(score)}, findings, {})
    assert decision.verdict is verdict
    assert reason in decision.reason


def test_unresolved_and_undecided_metrics_are_inconclusive_never_fail() -> None:
    policy = SeverityVerdictPolicy()
    unresolved = _finding("a", "critical", unresolved=True)
    decision = policy.decide({"a": _metric(0.5)}, [unresolved], {})
    assert decision.verdict is Verdict.INCONCLUSIVE and "unresolved - [critical]" in decision.reason
    error = _metric(None, MetricStatus.EVALUATION_ERROR)
    decision = policy.decide({"a": _metric(1.0), "b": error}, [], {})
    assert decision.verdict is Verdict.INCONCLUSIVE and "b evaluation_error" in decision.reason


def test_fail_line_is_configurable() -> None:
    medium = [_finding("a", "medium")]
    strict = SeverityVerdictPolicy(Severity.MEDIUM).decide({"a": _metric(0.5)}, medium, {})
    assert strict.verdict is Verdict.FAIL
    lenient = SeverityVerdictPolicy(Severity.CRITICAL)
    assert lenient.decide({"a": _metric(0.5)}, [_finding("a", "high")], {}).verdict is Verdict.PASS
    assert lenient.name == "severity-v1 (fail at critical+)"


async def test_statuses_and_reasons_survive_serialization(
    app_config: AppConfig,
    sample_product: ProductDescription,
    case: Any,
    sample_reference: Reference,
) -> None:
    result, _ = await evaluate(
        app_config,
        sample_product,
        case,
        sample_reference,
        {"outcome": "answered", "text": "x"},
        [],
        metrics=("mrr", "faithfulness"),
    )
    again = EvaluationResult.model_validate_json(result.model_dump_json())
    assert again == result
    assert {m: r.missing_requirement for m, r in again.metrics.items()} == {
        "mrr": "retrieved_context",
        "faithfulness": "grounding_evidence",
    }


def test_quotes_match_whatever_the_sql_spacing() -> None:
    """A judge re-spaces SQL it reads JSON-encoded; the quote check must not loop on that
    (live failure: 3 attempts rejected the same concat(...) quote)."""
    from synthetic_user.nodes.evaluator.judge_common import quoted

    quote = "concat(d.first_nm,' ',d.lst_nm) as cardmember_name, d.email_ad_tx as email_address"
    for sql in (
        "select concat(d.first_nm, ' ', d.lst_nm) as cardmember_name, d.email_ad_tx as "
        "email_address from t d",
        "SELECT\n  CONCAT(d.first_nm,' ',d.lst_nm) AS cardmember_name\n  ,d.email_ad_tx AS "
        "email_address\nFROM t d",
    ):
        assert quoted(quote, sql)
    assert quoted("cardmember_name,\\n  d.email_ad_tx", "cardmember_name,\n d.email_ad_tx")
    # still exact otherwise: other characters, order and invented text are rejected
    assert not quoted("concat(d.first_nm,' ',d.last_nm)", "concat(d.first_nm, ' ', d.lst_nm)")
    assert not quoted("d.email_ad_tx as cardmember_name", "d.email_ad_tx as email_address")
