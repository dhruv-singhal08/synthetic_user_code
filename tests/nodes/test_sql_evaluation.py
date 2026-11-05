"""Bounded, evidence-driven SQL evaluation through the Evaluator (section 18, 21-44).

Offline: the FakeLLMClient plays the General Judge, so these tests pin the EVIDENCE each call
receives (reconciliation, exact catalog lookups, rules with ids), the bounded one-batch lookup,
the metric/finding plumbing, the verdict and the stored record. They do not measure how well a
real model judges (that needs live runs). Catalog: the SYNTHETIC fixture (D229).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from synthetic_user.core.config import AppConfig
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.metrics.compiler import custom_metric_id
from synthetic_user.metrics.registry import builtin_metric_config, freeze, resolve_metric_config
from synthetic_user.models.evaluation import EvaluationResult, MetricStatus, Verdict
from synthetic_user.models.execution import ExecutionRecord
from synthetic_user.models.lookup import IdentifierStatus, LookupOrigin
from synthetic_user.models.metrics import (
    CustomMetricInput,
    EvaluationTarget,
    EvidenceKind,
    JudgeGroup,
    MetricCategory,
    MetricDefinition,
    MetricOrigin,
    MetricSelection,
    ResolvedMetricConfig,
)
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import CatalogPackDraft, Reference, ReferencePack
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator import EvaluationCase, Evaluator
from synthetic_user.nodes.question.context import fixed_context
from synthetic_user.references.loader import load_reference
from synthetic_user.references.pack import build_pack
from synthetic_user.web.results_csv import build_rows
from tests_support import checks_for, failed_check, goal_for

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "references" / "sql_eval_catalog.yaml"
SQL_METRICS = ("semantic_correctness", "schema_faithfulness", "rules_adherence")
CRITERION = "returns the requested measure per billing month"
LPC_PACK = {
    "tables": [
        {
            "table": "v_lpc_details",
            "columns": ["bill_cycle_yr_mo_no", "net_lpc_charge_usd_am"],
            "why": "net LPC by billing month",
        }
    ]
}
NET_SQL = (
    "SELECT bill_cycle_yr_mo_no AS billing_month, SUM(net_lpc_charge_usd_am) AS net_lpc_usd "
    "FROM gcsinsight.v_lpc_details WHERE bill_cycle_yr_mo_no BETWEEN 202501 AND 202512 "
    "GROUP BY bill_cycle_yr_mo_no ORDER BY billing_month"
)
Reply = dict[str, Any]


@pytest.fixture(scope="module")
def reference() -> Reference:
    return load_reference(FIXTURE, max_chars=200_000)


def make_case(
    reference: Reference, pack_draft: dict[str, Any] = LPC_PACK
) -> tuple[Scenario, QuestionCase, ReferencePack]:
    pack, rejected = build_pack(CatalogPackDraft.model_validate(pack_draft), reference, "SCN_001")
    assert pack is not None and not rejected
    scenario = Scenario(
        id="SCN_001",
        persona_id="analyst",
        product_id="sample_nl2sql",
        mode=GenerationMode.GROUNDED,
        reference_id=reference.reference_id,
        reference_pack_id=pack.pack_id,
        signature="sig",
        business_goal="Track late payment charges per billing month",
        title="Net LPC by month",
        success_criteria=[CRITERION],
        task_type="trend",
        capability="supported",
        capability_rationale="trend over a single table",
        question_intent="net LPC per billing month in 2025",
    )
    question = QuestionCase.model_validate(
        {
            "question_id": "Q_005",
            "scenario_id": scenario.id,
            "persona_id": scenario.persona_id,
            "mode": scenario.mode,
            "reference_pack_id": pack.pack_id,
            "utterance": "What were net late payment charges by billing month in 2025?",
            "intended_goal": scenario.business_goal,
            "intent": {"task_type": "trend"},
            "question_characteristics": {
                "style": "precise",
                "ambiguity_level": "low",
                "complexity": "simple",
            },
            # Every criterion mapped to every criteria metric; rules / schema mapped by code.
            "evaluation_context": fixed_context(
                [CRITERION], pack, builtin_metric_config(list(SQL_METRICS)).definitions
            ).model_dump(mode="json", by_alias=True),
        }
    )
    return scenario, question, pack


def execution_for(question: QuestionCase, response: dict[str, Any]) -> ExecutionRecord:
    return ExecutionRecord.model_validate(
        {
            "execution_id": "EXE_001",
            "question_id": question.question_id,
            "product_id": "sample_nl2sql",
            "adapter": "fixture",
            "submitted_question": question.utterance,
            "started_at": datetime(2026, 11, 3, tzinfo=UTC),
            "response": response,
        }
    )


def asked_metrics(user: str) -> list[str]:
    line = user.split("on exactly these metric(s): ", 1)[1].split("\n", 1)[0]
    return [m.strip().rstrip(".") for m in line.split(",")]


def judge(
    scores: dict[str, float] | None = None,
    findings: list[dict[str, Any]] | None = None,
) -> Callable[[str, str], Reply]:
    """A General Judge reply: every asked metric 1 unless overridden; a passed check for every
    requirement the contract assigns, plus a failed derived check for a score below 1."""

    def respond(system: str, user: str) -> Reply:
        metrics = {}
        for metric_id in asked_metrics(user):
            value = (scores or {}).get(metric_id, 1)
            checks = checks_for(user, metric_id, evidence="SQL:1")
            if value < 1:
                checks.append(failed_check(metric_id, "SQL:1"))
            metrics[metric_id] = {"score": value, "rationale": "r", "checks": checks}
            if metric_id == "correctness_completeness":
                metrics[metric_id]["factual_verification"] = "unverified"
        reply: Reply = {
            "metrics": metrics,
            "findings": findings or [],
            "overall_verdict": "pass",
            "goal_completion": goal_for(user),
        }
        if "response_classification" in user:
            reply["response_classification"] = "answered"
        return reply

    return respond


def finding(category: str, evidence: str, **extra: Any) -> dict[str, Any]:
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


def request(*lookups: dict[str, Any]) -> Reply:
    return {"evidence_requests": [{"reason": "gap", **lookup} for lookup in lookups]}


async def evaluate(
    settings: AppConfig,
    product: ProductDescription,
    reference: Reference,
    sql: str | list[str] | None,
    replies: list[Any],
    *,
    metrics: tuple[str, ...] | ResolvedMetricConfig = SQL_METRICS,
    pack_draft: dict[str, Any] = LPC_PACK,
    text: str = "Here is the query.",
) -> tuple[EvaluationResult, FakeLLMClient]:
    scenario, question, pack = make_case(reference, pack_draft)
    sqls = [sql] if isinstance(sql, str) else (sql or [])
    response = {
        "outcome": "answered",
        "text": text,
        "artifacts": [{"kind": "sql", "content": s} for s in sqls],
    }
    execution = execution_for(question, response)
    config = (
        metrics
        if isinstance(metrics, ResolvedMetricConfig)
        else builtin_metric_config(list(metrics))
    )
    llm = FakeLLMClient(replies)
    evaluator = Evaluator(
        llm, settings, logging.getLogger("t"), reference=reference, metric_config=config
    )
    result = await evaluator.evaluate(EvaluationCase(question, scenario, execution), product)
    return result, llm


def scores(result: EvaluationResult) -> dict[str, Any]:
    return {
        m: (r.score if r.status is MetricStatus.SCORED else r.status.value)
        for m, r in result.metrics.items()
    }


def section(user: str, title: str) -> str:
    return user.split(title, 1)[1].split("\n\n", 1)[0] if title in user else ""


# --- metric evaluation 21-30 --------------------------------------------------------------------


async def test_net_versus_gross_is_semantic_and_rules_not_schema(  # 21
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    gross = NET_SQL.replace("net_lpc_charge_usd_am", "lpc_charge_usd_am")
    reply = judge(
        {"semantic_correctness": 0, "rules_adherence": 0},
        [
            finding("semantic_correctness", "SUM(lpc_charge_usd_am)"),
            finding("rules_adherence", "SUM(lpc_charge_usd_am)"),
        ],
    )
    result, llm = await evaluate(app_config, sample_product, reference, gross, [reply])
    user = llm.calls[0].user
    # the gross column is valid (schema) and described from the catalog before the call
    assert '"v_lpc_details.lpc_charge_usd_am"' in section(user, "Catalog reconciliation")
    assert "Gross late payment charge in USD." in user
    assert "Late payment charge questions use the net amount" in user  # the pack's table rule
    assert scores(result) == {
        "semantic_correctness": 0.0,
        "schema_faithfulness": 1.0,
        "rules_adherence": 0.0,
    }
    assert result.verdict is Verdict.FAIL and len(result.bug_candidates) == 1


async def test_correct_monthly_grouping_passes_in_one_call(  # 22 + 39
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    result, llm = await evaluate(app_config, sample_product, reference, NET_SQL, [judge()])
    assert result.verdict is Verdict.PASS and len(llm.calls) == 1
    checks = result.reference_checks
    assert checks is not None and checks.general_judge_calls == 1
    assert checks.judge_lookup_requests == 0
    assert all(e.origin is LookupOrigin.DETERMINISTIC for e in checks.lookup_trace)
    (rec,) = checks.artifacts
    assert {i.status for i in rec.identifiers} == {IdentifierStatus.VALID_IN_PACK}
    assert '"group_by"' in llm.calls[0].user


async def test_incorrect_market_grouping_fails_semantic_only(  # 23
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    sql = "SELECT mkt_cd, SUM(net_lpc_charge_usd_am) FROM v_lpc_details GROUP BY mkt_cd"
    reply = judge({"semantic_correctness": 0}, [finding("semantic_correctness", "GROUP BY mkt_cd")])
    result, llm = await evaluate(app_config, sample_product, reference, sql, [reply])
    assert scores(result)["semantic_correctness"] == 0.0
    assert scores(result)["schema_faithfulness"] == 1.0
    assert '"v_lpc_details.mkt_cd"' in section(llm.calls[0].user, "Catalog reconciliation")
    assert result.verdict is Verdict.FAIL


async def test_valid_extra_cg_id_is_valid_and_described(  # 24
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    sql = NET_SQL.replace("WHERE ", "WHERE cg_id = 'C1' AND ")
    result, llm = await evaluate(app_config, sample_product, reference, sql, [judge()])
    user = llm.calls[0].user
    assert "a valid_extra identifier is never a schema error" in " ".join(user.split())
    assert "AUTHORITATIVE allowed" not in user  # the cheat sheet is no longer an allow-list
    assert "Client group identifier." in user  # looked up by code before the call
    checks = result.reference_checks
    assert checks is not None
    extra = checks.artifacts[0].with_status(IdentifierStatus.VALID_EXTRA)
    assert [i.canonical for i in extra] == ["v_lpc_details.cg_id"]


async def test_unjustified_dummy_filter_is_unresolved_not_pass(  # 25
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    sql = NET_SQL.replace("WHERE ", "WHERE cg_id IN ('DUMMY') AND ")
    reply = judge(
        findings=[
            finding(
                "semantic_correctness",
                "cg_id IN ('DUMMY')",
                severity="medium",
                evidence_status="weak",
                unresolved=True,
            )
        ]
    )
    result, llm = await evaluate(app_config, sample_product, reference, sql, [reply])
    assert "cg_id IN ('DUMMY')" in llm.calls[0].user  # judged as generated, never hidden
    assert result.findings[0].unresolved and result.bug_candidates == []
    assert result.verdict is Verdict.INCONCLUSIVE


async def test_defra_plus_epa_rule_is_retrieved_even_when_not_selected(  # 26
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    pack = {"tables": [{"table": "v_emissions", "columns": ["defra_emissions_kg"], "why": "w"}]}
    sql = "SELECT SUM(defra_emissions_kg + epa_emissions_kg) AS total FROM v_emissions"
    reply = judge(
        {"rules_adherence": 0},
        [finding("rules_adherence", "SUM(defra_emissions_kg + epa_emissions_kg)")],
    )
    result, llm = await evaluate(
        app_config, sample_product, reference, sql, [reply], pack_draft=pack
    )
    lookups = section(llm.calls[0].user, "Catalog lookups done by code")
    assert '"rule_id": "R1"' in lookups
    assert "DEFRA and EPA emissions must never be added together into one total." in lookups
    assert scores(result)["rules_adherence"] == 0.0 and result.verdict is Verdict.FAIL


async def test_documented_delinquency_join_is_valid(  # 27
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    sql = (
        "SELECT l.bill_cycle_yr_mo_no, d.dlq_bucket_cd, SUM(l.net_lpc_charge_usd_am) "
        "FROM v_lpc_details l JOIN v_acct_delinquency d ON l.acct_id = d.acct_id "
        "AND l.bill_cycle_yr_mo_no = d.bill_cycle_yr_mo_no GROUP BY 1, 2"
    )
    result, llm = await evaluate(app_config, sample_product, reference, sql, [judge()])
    checks = result.reference_checks
    assert checks is not None
    statuses = {(j.left, j.right): j.status.value for j in checks.artifacts[0].joins}
    assert statuses[("v_lpc_details.acct_id", "v_acct_delinquency.acct_id")] == (
        "documented_verified"
    )
    assert "Days-past-due bucket." in llm.calls[0].user
    assert result.verdict is Verdict.PASS


async def test_catalog_facts_are_never_accepted_as_quoted_evidence(
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    """Live failure: the judge quoted the reconciliation reason ("table ... is not in the
    catalog") as evidence; the prompt says where evidence comes from and the check refuses it."""
    sql = "SELECT SUM(net_lpc_charge_usd_am) FROM no_such_table"
    bad = judge(
        {"schema_faithfulness": 0},
        [finding("schema_faithfulness", "table 'no_such_table' is not in the catalog")],
    )
    good = judge(
        {"schema_faithfulness": 0},
        [
            finding(
                "schema_faithfulness",
                "no_such_table",
                observed_behavior="table 'no_such_table' is not in the catalog",
            )
        ],
    )
    result, llm = await evaluate(app_config, sample_product, reference, sql, [bad, good])
    prompt = " ".join(llm.calls[0].user.split())
    assert "evidence is still copied ONLY from the product response" in prompt
    assert "is not in the source" in llm.calls[1].user  # rejected, never silently accepted
    assert result.findings[0].evidence == "no_such_table"
    assert result.verdict is Verdict.FAIL


UNDOCUMENTED_JOIN = "v_cardmember.acct_id = v_lpc_details.acct_id"
CARDMEMBER_SQL = (
    "SELECT c.cm_name, SUM(l.net_lpc_charge_usd_am) FROM v_lpc_details l "
    "JOIN v_cardmember c ON c.acct_id = l.acct_id GROUP BY 1"
)


def decided(base: Callable[[str, str], Reply], **decision: Any) -> Callable[[str, str], Reply]:
    def respond(system: str, user: str) -> Reply:
        reply = base(system, user)
        reply["join_decisions"] = [{"join": UNDOCUMENTED_JOIN, "reason": "r", **decision}]
        return reply

    return respond


async def test_undocumented_join_needs_a_decision_the_schema_score_agrees_with(
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    """Live finding (D238): an undocumented join passed schema_faithfulness unaddressed.
    The judge must decide every undocumented join; code checks only score/decision agreement."""
    not_allowed = decided(judge(), allowed=False, basis="usage_instructions")
    fixed = decided(
        judge(
            {"schema_faithfulness": 0.5},
            [finding("schema_faithfulness", "JOIN v_cardmember c ON c.acct_id = l.acct_id")],
        ),
        allowed=False,
        basis="usage_instructions",
    )
    result, llm = await evaluate(
        app_config, sample_product, reference, CARDMEMBER_SQL, [judge(), not_allowed, fixed]
    )
    first = " ".join(llm.calls[0].user.split())
    assert "For EVERY undocumented join add one join_decisions entry" in first
    assert "Tables can only be combined over the listed relationships" in first  # default text
    assert '"join_decisions"' in llm.calls[0].user  # in the output contract
    assert "join_decisions must decide every undocumented join" in llm.calls[1].user
    assert f"join_decisions says {UNDOCUMENTED_JOIN} is not allowed" in llm.calls[2].user
    assert scores(result)["schema_faithfulness"] == 0.5 and result.verdict is Verdict.FAIL
    checks = result.reference_checks
    assert checks is not None and checks.join_decisions[0]["basis"] == "usage_instructions"


async def test_approved_or_unestablished_join_decisions(
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    approved = decided(judge(), allowed=True, basis="evaluation_guidance")
    result, llm = await evaluate(app_config, sample_product, reference, CARDMEMBER_SQL, [approved])
    assert len(llm.calls) == 1 and scores(result)["schema_faithfulness"] == 1.0
    unknown = decided(judge(), allowed=False, basis="not_established")
    flagged = decided(
        judge(
            findings=[
                finding(
                    "schema_faithfulness",
                    "JOIN v_cardmember c ON c.acct_id = l.acct_id",
                    severity="low",
                    evidence_status="weak",
                    unresolved=True,
                )
            ]
        ),
        allowed=False,
        basis="not_established",
    )
    result, llm = await evaluate(
        app_config, sample_product, reference, CARDMEMBER_SQL, [unknown, flagged]
    )
    assert "add an unresolved finding with category 'schema_faithfulness'" in llm.calls[1].user
    assert result.verdict is Verdict.INCONCLUSIVE and result.bug_candidates == []


async def test_invalid_join_is_a_schema_finding(  # 28
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    sql = (
        "SELECT l.net_lpc_charge_usd_am FROM v_lpc_details l "
        "JOIN v_acct_delinquency d ON l.account_no = d.acct_id"
    )
    reply = judge(
        {"schema_faithfulness": 0.5},
        [finding("schema_faithfulness", "l.account_no = d.acct_id")],
    )
    result, llm = await evaluate(app_config, sample_product, reference, sql, [reply])
    checks = result.reference_checks
    assert checks is not None
    (rec,) = checks.artifacts
    assert [i.identifier for i in rec.with_status(IdentifierStatus.INVALID)] == [
        "v_lpc_details.account_no"
    ]
    assert rec.joins[0].status.value == "unresolved"
    assert result.verdict is Verdict.FAIL


async def test_optional_documented_join_may_be_omitted(  # 29
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    pack = {
        **LPC_PACK,
        "tables": [
            *LPC_PACK["tables"],
            {"table": "v_acct_delinquency", "columns": ["acct_id"], "why": "optional"},
        ],
        "joins": [{"from": "v_lpc_details.acct_id", "to": "v_acct_delinquency.acct_id"}],
    }
    result, _ = await evaluate(
        app_config, sample_product, reference, NET_SQL, [judge()], pack_draft=pack
    )
    checks = result.reference_checks
    assert checks is not None
    assert "v_acct_delinquency" in checks.artifacts[0].pack_items_not_used
    assert result.verdict is Verdict.PASS  # an unused cheat-sheet item never fails by itself


async def test_contradictory_rules_become_an_unresolved_finding(  # 30
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    pack = {"tables": [{"table": "v_cardmember", "columns": ["cm_name"], "why": "names"}]}
    sql = "SELECT cm_name FROM v_cardmember"
    reply = judge(
        findings=[
            finding(
                "rules_adherence",
                "cm_name",
                severity="medium",
                evidence_status="weak",
                unresolved=True,
                description="R2 and R3 contradict each other on showing cardmember names",
            )
        ]
    )
    result, llm = await evaluate(
        app_config, sample_product, reference, sql, [reply], pack_draft=pack
    )
    user = llm.calls[0].user
    assert "Cardmember names may be shown" in user and "Cardmember names must never" in user
    assert "never rewrite rules or choose between contradictory ones" in " ".join(user.split())
    assert result.verdict is Verdict.INCONCLUSIVE and result.bug_candidates == []


# --- custom metrics 31-38 ----------------------------------------------------------------------


def custom(index: int, evidence: list[EvidenceKind]) -> MetricDefinition:
    return MetricDefinition(
        metric_id=custom_metric_id(index),
        display_name=f"Custom {index}",
        category=MetricCategory.CUSTOM,
        version="1.0",
        origin=MetricOrigin.CUSTOM,
        description="d",
        rubric=f"rubric of custom {index}",
        score_anchors={"1": "a", "0.5": "b", "0": "c"},
        judge_group=JudgeGroup.GENERAL,
        required_execution_evidence=evidence,
        applicability_policy="always",
        evidence_requirements="as listed",
        source_text=f"user text {index}",
        source_provenance="free text",
    )


def config_with(*customs: MetricDefinition, builtins: tuple[str, ...] = SQL_METRICS) -> Any:
    selection = MetricSelection(
        selected_metrics=list(builtins),
        custom_inputs=[
            CustomMetricInput(text=c.source_text or "", provenance="t") for c in customs
        ],
    )
    defaults = builtin_metric_config(list(builtins)).definitions if builtins else []
    definitions = [*defaults, *customs]
    return freeze(selection, definitions, compiler_called=True, notes=[])


async def test_sql_custom_metric_shares_the_sql_evaluation(  # 31 + 35 + 36 + 43
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    efficiency = custom(1, [EvidenceKind.SQL])
    assert efficiency.evaluation_target is EvaluationTarget.SQL
    sql = NET_SQL.replace("WHERE ", "WHERE cg_id = 'C1' AND ")
    reply = judge({"custom_01": 0.5}, [finding("custom_01", "cg_id = 'C1'")])
    with_custom, llm = await evaluate(
        app_config, sample_product, reference, sql, [reply], metrics=config_with(efficiency)
    )
    assert len(llm.calls) == 1  # one call: defaults and custom share the same evidence
    assert asked_metrics(llm.calls[0].user) == [
        "semantic_correctness",
        "rules_adherence",
        "schema_faithfulness",
        "custom_01",
    ]
    assert "Client group identifier." in llm.calls[0].user  # lookup evidence reaches it
    assert scores(with_custom)["custom_01"] == 0.5
    baseline, _ = await evaluate(app_config, sample_product, reference, sql, [judge()])
    assert {m: scores(with_custom)[m] for m in SQL_METRICS} == scores(baseline)


async def test_non_sql_custom_metric_keeps_its_route(  # 32
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    tone = custom(1, [EvidenceKind.ANSWER])
    assert tone.evaluation_target is EvaluationTarget.NON_SQL
    result, llm = await evaluate(
        app_config,
        sample_product,
        reference,
        NET_SQL,
        [judge()],
        metrics=config_with(tone, builtins=("relevance",)),
    )
    user = llm.calls[0].user
    assert "SQL analysis" not in user and "Catalog reconciliation" not in user
    assert "AUTHORITATIVE allowed" in user  # non-SQL prompt unchanged
    assert result.reference_checks is None


async def test_metric_targeting_sql_and_text_gets_both(  # 33
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    both = custom(1, [EvidenceKind.ANSWER, EvidenceKind.SQL])
    assert both.evaluation_target is EvaluationTarget.BOTH
    _, llm = await evaluate(
        app_config,
        sample_product,
        reference,
        NET_SQL,
        [judge()],
        metrics=config_with(both, builtins=()),
        text="Net LPC per month is in the query.",
    )
    user = llm.calls[0].user
    assert "Catalog reconciliation" in user and "Net LPC per month is in the query." in user


async def test_ambiguous_custom_target_keeps_existing_routing(  # 34
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    vague = custom(1, [])
    assert vague.evaluation_target is EvaluationTarget.UNKNOWN
    result, llm = await evaluate(
        app_config,
        sample_product,
        reference,
        NET_SQL,
        [judge()],
        metrics=config_with(vague, builtins=()),
    )
    assert asked_metrics(llm.calls[0].user) == ["custom_01"]  # still the General Judge
    assert "Catalog reconciliation" not in llm.calls[0].user  # not forced through SQL
    assert scores(result) == {"custom_01": 1.0}


async def test_custom_metric_result_is_serialized_in_its_columns(  # 37
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    efficiency = custom(1, [EvidenceKind.SQL])
    reply = judge({"custom_01": 0.5}, [finding("custom_01", "ORDER BY billing_month")])
    config = config_with(efficiency)
    result, _ = await evaluate(
        app_config, sample_product, reference, NET_SQL, [reply], metrics=config
    )
    scenario, question, pack = make_case(reference)
    execution = execution_for(
        question, {"outcome": "answered", "artifacts": [{"kind": "sql", "content": NET_SQL}]}
    )
    (row,) = build_rows(
        "RUN_X",
        [scenario],
        [question],
        executions=[execution],
        evaluations=[result],
        packs=[pack],
        metric_ids=config.selected_metric_ids,
    )
    assert row["custom_01"] == "0.5" and row["custom_01_status"] == "scored"
    assert row["semantic_correctness"] == "1"
    assert "reference_checks" not in row and "lookup_trace" not in row  # no new CSV columns


def test_compiler_still_compiles_sql_custom_metrics(  # 38
    sample_product: ProductDescription, reference: Reference
) -> None:
    reply = {
        "metrics": [
            {
                "input_index": 1,
                "compilable": True,
                "display_name": "SQL date-filter correctness",
                "definition": "date filters in the generated SQL match the requested period",
                "rubric": "compare the SQL date predicates with the question's period",
                "score_anchors": {"1": "all", "0.5": "some", "0": "none"},
                "evidence_requirements": "the SQL artifact",
                "applicability": "SQL answers",
                "judge_group": "general",
                "required_execution_evidence": ["sql"],
                "required_reference_assets": [],
                "needs_question_expectations": False,
                "expectation_fields": [],
                "generation_instructions": None,
            }
        ]
    }
    llm = FakeLLMClient([reply])
    config = asyncio.run(
        resolve_metric_config(
            MetricSelection(
                custom_inputs=[CustomMetricInput(text="date filters are right", provenance="t")]
            ),
            sample_product,
            reference,
            llm=llm,
            logger=logging.getLogger("t"),
            max_retries=1,
            sql_dialect=None,
        )
    )
    definition = config.definition("custom_01")
    assert definition is not None and definition.evaluation_target is EvaluationTarget.SQL
    assert "The evidence decides what a metric is routed to" in llm.calls[0].user


# --- agent boundaries 39-44 --------------------------------------------------------------------


async def test_unresolved_evidence_triggers_one_batched_lookup(  # 40 + 42
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    first = request(
        {"tool": "lookup_schema", "table": "v_cardmember", "column": "cm_name"},
        {"tool": "lookup_schema", "table": "no_such_table"},
        {"tool": "lookup_rules", "rule_ids": ["R9"]},
    )
    result, llm = await evaluate(app_config, sample_product, reference, NET_SQL, [first, judge()])
    assert len(llm.calls) == 2
    assert "Optional evidence lookup (one batch)" in llm.calls[0].user
    second = llm.calls[1].user
    assert "Results of the lookups you requested" in second
    assert "Optional evidence lookup" not in second  # no further requests offered
    results = section(second, "Results of the lookups you requested")
    assert "Cardmember full name." in results
    assert '"status": "not_found"' in results and '"not_found": [\n      "R9"' in results
    checks = result.reference_checks
    assert checks is not None
    assert checks.judge_lookup_requests == 3 and checks.general_judge_calls == 2
    requested = [e for e in checks.lookup_trace if e.origin is LookupOrigin.JUDGE_REQUEST]
    assert len(requested) == 3 and all(e.result.model_dump() for e in requested)
    assert result.verdict is Verdict.PASS


async def test_no_second_lookup_round_and_bounded_requests(  # 41
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    lookup = {"tool": "lookup_schema", "table": "v_cardmember"}
    too_many = request(*([lookup] * 11))
    again = request(lookup)
    result, llm = await evaluate(
        app_config,
        sample_product,
        reference,
        NET_SQL,
        [too_many, request(lookup), again, again, again],
    )
    assert "at most 10 evidence_requests" in llm.calls[1].user
    # after the one lookup batch, further requests are refused until retries run out
    assert all("evidence_requests are not available" in c.user for c in llm.calls[3:5])
    assert len(llm.calls) == 5 and set(scores(result).values()) == {"evaluation_error"}
    checks = result.reference_checks
    assert checks is not None and checks.general_judge_calls == 2


async def test_lookups_can_be_disabled(  # config switch
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    lookup = app_config.evaluation.reference_lookup.model_copy(update={"enabled": False})
    settings = app_config.model_copy(
        update={"evaluation": app_config.evaluation.model_copy(update={"reference_lookup": lookup})}
    )
    _, llm = await evaluate(settings, sample_product, reference, NET_SQL, [judge()])
    assert "Optional evidence lookup" not in llm.calls[0].user
    assert "Catalog reconciliation" in llm.calls[0].user  # deterministic part always runs


async def test_identical_evidence_is_looked_up_once(  # 43
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    sql = NET_SQL.replace("WHERE ", "WHERE cg_id = 'C1' AND ")
    result, llm = await evaluate(app_config, sample_product, reference, [sql, sql], [judge()])
    checks = result.reference_checks
    assert checks is not None and len(checks.artifacts) == 2
    keys = [e.request.key() for e in checks.lookup_trace]
    assert len(keys) == len(set(keys))  # duplicates across artifacts answered once
    assert llm.calls[0].user.count("Client group identifier.") == 1


async def test_result_schema_stays_backward_compatible(  # 44
    app_config: AppConfig, sample_product: ProductDescription, reference: Reference
) -> None:
    result, _ = await evaluate(app_config, sample_product, reference, NET_SQL, [judge()])
    data = json.loads(result.model_dump_json())
    assert EvaluationResult.model_validate(data) == result
    old = {k: v for k, v in data.items() if k != "reference_checks"}
    old["findings"] = [{k: v for k, v in f.items() if k != "unresolved"} for f in old["findings"]]
    legacy = EvaluationResult.model_validate(old)
    assert legacy.reference_checks is None and legacy.metrics == result.metrics


async def test_document_and_ungrounded_runs_have_no_catalog_step(
    app_config: AppConfig,
    sample_product: ProductDescription,
    reference: Reference,
) -> None:
    """No data catalog -> no index: the judge sees the parser output only (unchanged)."""
    scenario, question, _ = make_case(reference)
    scenario = scenario.model_copy(
        update={"mode": "ungrounded", "reference_id": None, "reference_pack_id": None}
    )
    data = question.model_dump(mode="json", by_alias=True)
    data["evaluation_context"]["reference_spec"] = None
    for mapping in data["evaluation_context"]["metric_mapping"].values():
        mapping["reference_requirements"] = []
    question = QuestionCase.model_validate(
        {**data, "mode": "ungrounded", "reference_pack_id": None}
    )
    execution = execution_for(
        question, {"outcome": "answered", "artifacts": [{"kind": "sql", "content": NET_SQL}]}
    )
    llm = FakeLLMClient([judge()])
    evaluator = Evaluator(
        llm,
        app_config,
        logging.getLogger("t"),
        reference=None,
        metric_config=builtin_metric_config(["semantic_correctness"]),
    )
    result = await evaluator.evaluate(EvaluationCase(question, scenario, execution), sample_product)
    assert "SQL analysis" in llm.calls[0].user
    assert "Catalog reconciliation" not in llm.calls[0].user
    assert result.reference_checks is None
