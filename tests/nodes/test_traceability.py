"""Evidence-backed metric decisions (reforming): every mapped requirement gets a check-level
decision, every cited evidence / guidance id resolves, the score agrees with the checks, and
the evaluation explains coverage, guidance, limitations and unresolved requirements without
changing the verdict policy."""

from __future__ import annotations

import logging
from typing import Any

from nodes.test_evaluator import ANSWER, CRITERIA, _execution, all_pass, finding, score
from synthetic_user.core.config import AppConfig
from synthetic_user.core.guidance import TYPED_GUIDANCE, parse_evaluation_guidance
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.metrics.registry import builtin_metric_config
from synthetic_user.models.evaluation import EvaluationResult, MetricStatus, Verdict
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.run import EvaluationGuidance, GuidanceItem, GuidanceType
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator import EvaluationCase, Evaluator
from synthetic_user.nodes.question.context import fixed_context

METRICS = ("relevance", "semantic_correctness", "rules_adherence", "schema_faithfulness")


def _mapped_question(scenario: Scenario, pack: ReferencePack) -> QuestionCase:
    """A grounded question whose Metric Mapping assigns C1, C2 to the criteria metrics and the
    pack's rules / schema to rules_adherence / schema_faithfulness (as generation does)."""
    context = fixed_context(CRITERIA, pack, builtin_metric_config(list(METRICS)).definitions)
    return QuestionCase.model_validate(
        {
            "question_id": "Q_001",
            "scenario_id": scenario.id,
            "persona_id": scenario.persona_id,
            "mode": scenario.mode,
            "reference_pack_id": pack.pack_id,
            "utterance": "Who were our top 10 card spenders last quarter?",
            "intended_goal": scenario.business_goal,
            "intent": {"task_type": "ranking"},
            "question_characteristics": {
                "style": "precise",
                "ambiguity_level": "low",
                "complexity": "simple",
            },
            "evaluation_context": context.model_dump(mode="json", by_alias=True),
        }
    )


def check(rid: str, source: str, status: str = "passed", **extra: Any) -> dict[str, Any]:
    return {
        "requirement_id": rid,
        "source": source,
        "status": status,
        "evidence_ids": ["SQL:1"],
        "reason": f"{rid} {status}",
        **extra,
    }


def good_checks() -> dict[str, list[dict[str, Any]]]:
    return {
        "relevance": [check("C1", "success_criteria"), check("C2", "success_criteria")],
        "semantic_correctness": [check("C1", "success_criteria"), check("C2", "success_criteria")],
        "rules_adherence": [
            check("TABLE-RULE-01.01", "reference_spec"),
            check("RULE-R1", "reference_spec"),
        ],
        "schema_faithfulness": [check("reference_spec.schema", "reference_spec")],
    }


def reply(checks: dict[str, list[dict[str, Any]]] | None = None, **extra: Any) -> dict[str, Any]:
    out = all_pass(*METRICS)
    for metric_id, items in (checks or good_checks()).items():
        out["metrics"][metric_id]["checks"] = items
    out.update(extra)
    return out


async def run(
    settings: AppConfig,
    product: ProductDescription,
    reference: Reference,
    scenario: Scenario,
    pack: ReferencePack,
    replies: list[Any],
    guidance: EvaluationGuidance | None = None,
) -> tuple[EvaluationResult, FakeLLMClient]:
    llm = FakeLLMClient(replies)
    evaluator = Evaluator(
        llm,
        settings,
        logging.getLogger("t"),
        reference=reference,
        metric_config=builtin_metric_config(list(METRICS)),
        evaluation_guidance=guidance,
    )
    case = EvaluationCase(_mapped_question(scenario, pack), scenario, _execution(ANSWER))
    return await evaluator.evaluate(case, product), llm


async def test_every_score_is_explained_by_its_checks(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
) -> None:
    result, llm = await run(
        app_config,
        sample_product,
        sample_reference,
        sample_scenarios[0],
        sample_packs[0],
        [reply()],
    )
    assert len(llm.calls) == 1  # no extra LLM call for traceability
    prompt = llm.calls[0].user
    # the judge sees the evaluation contract, the criteria ids and the evidence ids
    assert '"reference_requirements": [' in prompt and '"id": "RULE-R1"' in prompt
    assert '"id": "C1"' in prompt and '"evidence_id": "SQL:1"' in prompt
    assert '"evidence_id": "RECON:1"' in prompt and "POLICY:UNRESOLVED" in prompt
    rules = result.metrics["rules_adherence"]
    assert rules.comparison_basis == "rules_vs_sql"
    assert rules.verification_method == "static_sql_analysis_with_catalog_lookups"
    assert rules.assigned_reference_requirements == ["TABLE-RULE-01.01", "RULE-R1"]
    assert [c.check_id for m in result.metrics.values() for c in m.checks] == [
        f"CHECK:{n:03d}" for n in range(1, 8)
    ]
    assert rules.checks[1].requirement == (
        "Amounts in different currency_code values must not be summed directly."
    )
    assert rules.coverage is not None
    assert rules.coverage.model_dump() == {
        "assigned": 2,
        "evaluated": 2,
        "passed": 2,
        "failed": 0,
        "not_applicable": 0,
        "unresolved": 0,
        "derived": 0,
    }
    assert rules.observed_evidence_ids == ["SQL:1"]
    [rubric] = rules.guidance_applied
    assert rubric.guidance_id == "METRIC:rules_adherence"
    assert rubric.source.startswith("metrics/catalog.yaml (catalog ")
    assert result.verdict is Verdict.PASS and not result.failed_requirements
    assert result.reference_checks is not None
    assert result.reference_checks.additional_lookup_performed is False
    assert all(e.evidence_id and e.round == 0 for e in result.reference_checks.lookup_trace)


async def test_omitted_or_invented_ids_and_hidden_failures_are_retried(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
) -> None:
    bad_checks = good_checks()
    bad_checks["relevance"] = bad_checks["relevance"][:1]  # C2 silently omitted
    bad_checks["semantic_correctness"][0]["evidence_ids"] = ["SQL:9"]  # invented evidence
    bad_checks["rules_adherence"].append(check("R99", "derived", "failed"))  # invented rule
    bad_checks["schema_faithfulness"] = [
        check("reference_spec.schema", "reference_spec", "failed", guidance_ids=["G7"])
    ]  # a failed check hidden behind a score of 1, citing guidance that does not exist
    result, llm = await run(
        app_config,
        sample_product,
        sample_reference,
        sample_scenarios[0],
        sample_packs[0],
        [reply(bad_checks), reply()],
    )
    feedback = " ".join(llm.calls[1].user.split())
    assert "relevance: checks must decide every assigned requirement; missing ['C2']" in feedback
    assert "semantic_correctness: unknown evidence ids ['SQL:9']" in feedback
    assert "rules_adherence: derived check 'R99' names no catalog rule" in feedback
    assert "schema_faithfulness is scored 1 but a check failed" in feedback
    assert "schema_faithfulness: unknown guidance ids ['G7']" in feedback
    assert result.verdict is Verdict.PASS  # the corrected reply


async def test_a_score_below_one_names_its_failed_requirement(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
) -> None:
    failing = good_checks()
    failing["semantic_correctness"][1] = check("C2", "success_criteria", "failed")
    out = reply(failing, findings=[finding("semantic_correctness", "LIMIT 10")])
    out["metrics"]["semantic_correctness"] = score(
        0.5, "period missing", checks=failing["semantic_correctness"]
    )
    result, _ = await run(
        app_config,
        sample_product,
        sample_reference,
        sample_scenarios[0],
        sample_packs[0],
        [out],
    )
    semantic = result.metrics["semantic_correctness"]
    assert semantic.coverage is not None and semantic.coverage.failed == 1
    assert result.failed_requirements == ["semantic_correctness: C2"]
    assert result.verdict is Verdict.FAIL
    assert "C2" in result.findings[0].related_tags  # the finding links its failed criterion


async def test_unresolved_checks_are_visible_and_the_verdict_unchanged(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
) -> None:
    """User decision: an unresolved check never passes silently, but only an unresolved
    FINDING makes the verdict inconclusive (the policy is unchanged)."""
    checks = good_checks()
    checks["rules_adherence"][1] = check("RULE-R1", "reference_spec", "unresolved")
    result, _ = await run(
        app_config,
        sample_product,
        sample_reference,
        sample_scenarios[0],
        sample_packs[0],
        [reply(checks)],
    )
    rules = result.metrics["rules_adherence"]
    assert rules.coverage is not None
    assert (rules.coverage.evaluated, rules.coverage.unresolved) == (1, 1)
    assert result.unresolved_requirements == ["rules_adherence: RULE-R1"]
    assert result.verdict is Verdict.PASS
    assert (result.verdict_reason or "").endswith(
        "Unresolved checks (verdict unchanged): rules_adherence: RULE-R1."
    )
    assert "rules_adherence: RULE-R1 unresolved - RULE-R1 unresolved" in result.limitations


async def test_guidance_is_cited_by_id_and_resolved_to_its_exact_text(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
) -> None:
    text = "Rankings may use any valid alias.\n\nTenant DUMMY filters are approved."
    parsed = parse_evaluation_guidance(text, source=TYPED_GUIDANCE, max_chars=1000)
    assert parsed is not None
    guidance = parsed.model_copy(
        update={
            "items": [
                GuidanceItem(
                    guidance_id="G1",
                    type=GuidanceType.INTERPRETATION_RULE,
                    metrics=["semantic_correctness"],
                    text="Rankings may use any valid alias.",
                    location="typed in the UI line 1",
                ),
                GuidanceItem(
                    guidance_id="G2",
                    type=GuidanceType.EXCEPTION,
                    text="Tenant DUMMY filters are approved.",
                    location="typed in the UI line 3",
                ),
            ]
        }
    )
    checks = good_checks()
    checks["semantic_correctness"][0]["guidance_ids"] = ["G1"]
    out = reply(checks)
    out["metrics"]["relevance"]["guidance_applied"] = [
        {"guidance_id": "POLICY:DECLINE", "applicability": "n/a", "influence": "none really"}
    ]
    result, llm = await run(
        app_config,
        sample_product,
        sample_reference,
        sample_scenarios[0],
        sample_packs[0],
        [out],
        guidance,
    )
    prompt = llm.calls[0].user
    assert "G1 [interpretation_rule; metrics: semantic_correctness]" in prompt
    assert "G2 [exception; metrics: all] (typed in the UI line 3):" in prompt
    applied = {g.guidance_id: g for g in result.metrics["semantic_correctness"].guidance_applied}
    assert list(applied) == ["METRIC:semantic_correctness", "G1"]
    assert applied["G1"].instruction == "Rankings may use any valid alias."
    assert applied["G1"].guidance_type == "interpretation_rule"
    assert applied["G1"].source == "evaluation guidance (typed in the UI line 1)"
    assert "CHECK:" in applied["G1"].influence
    policy = result.metrics["relevance"].guidance_applied[-1]
    assert (policy.guidance_id, policy.source) == ("POLICY:DECLINE", "general_prompt.yaml RULES")


async def test_metrics_decided_by_code_still_decide_every_mapped_requirement(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
) -> None:
    """A typed denial: the SQL metrics are not applicable, and their mapped requirements are
    recorded as not_applicable checks with the reason (never silently omitted)."""
    scenario, pack = sample_scenarios[0], sample_packs[0]
    denial = {"outcome": "denied", "message": "I cannot do that."}
    question = _mapped_question(scenario, pack)
    reply_ = all_pass("relevance", classify=False)
    reply_["metrics"]["relevance"]["checks"] = [
        {**check("C1", "success_criteria"), "evidence_ids": ["RESPONSE"]},
        {**check("C2", "success_criteria"), "evidence_ids": ["RESPONSE"]},
    ]
    reply_["goal_completion"] = {"status": "justified_decline", "reason": "contract"}
    llm = FakeLLMClient([reply_])
    evaluator = Evaluator(
        llm,
        app_config,
        logging.getLogger("t"),
        reference=sample_reference,
        metric_config=builtin_metric_config(list(METRICS)),
    )
    result = await evaluator.evaluate(
        EvaluationCase(question, scenario, _execution(denial)), sample_product
    )
    rules = result.metrics["rules_adherence"]
    assert rules.status is MetricStatus.NOT_APPLICABLE
    assert [(c.requirement_id, c.status) for c in rules.checks] == [
        ("TABLE-RULE-01.01", "not_applicable"),
        ("RULE-R1", "not_applicable"),
    ]
    assert rules.checks[0].reason.startswith("decided by code (not_applicable)")
    assert rules.guidance_applied == []  # no judge decided it
    assert result.limitations == []  # nothing static about SQL was judged
