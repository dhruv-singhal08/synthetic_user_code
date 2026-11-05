"""Regression tests of the targeted fixes (2026-11-05, D251-D254).

1-2 rules: only applicable rules are scoring checks, every available rule stays traceable;
3 response-level criteria are never forced into an SQL metric (coverage gap instead);
4 one invalid metric keeps the other metrics' valid results; one shared error otherwise;
5 a table the catalog does not list is a catalog mismatch, not a proven-absent table;
6 a derived defect explains a score below 1 despite complete mapped coverage;
7-8 an accepted question reply is usable, invalid enum values are fed back with clear reasons;
9 YTD YoY and rolling-12-month YoY stay distinct in the prompts;
10 run.log, the stored evaluation and the CSV report agree.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from nodes.test_question_generator import Env, batch, q, with_metric_expectations
from nodes.test_traceability import METRICS, check, good_checks, reply, run
from synthetic_user.core.config import AppConfig
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.metrics.registry import builtin_metric_config
from synthetic_user.models.evaluation import MetricStatus, RequirementStatus
from synthetic_user.models.lookup import IdentifierDiagnosis, IdentifierStatus
from synthetic_user.models.metrics import EvidenceKind
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import (
    EvaluationContextDraft,
    MetricMappingDraft,
    SuccessCriterionDraft,
)
from synthetic_user.models.reference import CatalogPackDraft, Reference, ReferencePack
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator import traceability
from synthetic_user.nodes.evaluator.evaluator import SHARED_ERROR
from synthetic_user.nodes.question.context import build_context, context_request, mapping_problems
from synthetic_user.nodes.question.prompt import QuestionPromptOptions, build_question_prompt
from synthetic_user.references.catalog_index import CatalogIndex
from synthetic_user.references.loader import load_reference
from synthetic_user.references.pack import build_pack
from synthetic_user.sql.parser import SqlglotParser
from synthetic_user.sql.reconcile import reconcile
from synthetic_user.web.results_csv import _evaluation_cells

SCENARIO_PROMPT_FILE = (
    Path(__file__).resolve().parents[2] / "src/synthetic_user/nodes/scenario/prompt.yaml"
)
Fixture = tuple[AppConfig, ProductDescription, Reference, Scenario, ReferencePack]


@pytest.fixture
def grounded(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
) -> Fixture:
    return app_config, sample_product, sample_reference, sample_scenarios[0], sample_packs[0]


def attempts(settings: AppConfig) -> int:
    return settings.llm.max_validation_retries + 1


# --- FIX 1: only applicable rules are scoring checks ------------------------------------------


async def test_irrelevant_rules_do_not_become_mandatory_checks(grounded: Fixture) -> None:
    checks = good_checks()
    checks["rules_adherence"] = [check("RULE-R1", "reference_spec")]  # the applicable rule
    out = reply(checks)
    out["metrics"]["rules_adherence"]["not_applicable_rules"] = ["TABLE-RULE-01.01"]
    result, llm = await run(*grounded, [out])
    assert len(llm.calls) == 1  # accepted at once: no check needed for the unrelated rule
    rules = result.metrics["rules_adherence"]
    assert rules.status is MetricStatus.SCORED and rules.score == 1
    assert [c.requirement_id for c in rules.checks] == ["RULE-R1"]
    assert rules.not_applicable_rules == ["TABLE-RULE-01.01"]
    assert rules.coverage is not None
    assert (rules.coverage.assigned, rules.coverage.available_rules) == (1, 2)
    assert rules.coverage.applicable_rules == 1
    assert "1 of 2 available rules applied (1 not applicable)" in (rules.score_explanation or "")
    # the full Reference Spec stays evidence: every rule is in the judge's cheat sheet
    assert "TABLE-RULE-01.01" in llm.calls[0].user and "RULE-R1" in llm.calls[0].user


async def test_applicable_rules_remain_traceable(grounded: Fixture) -> None:
    settings = grounded[0]
    checks = good_checks()
    checks["rules_adherence"] = [check("RULE-R1", "reference_spec", "failed")]  # one dropped
    silent = reply(checks)
    silent["metrics"]["rules_adherence"]["score"] = 0
    fixed = reply(
        {
            **checks,
            "rules_adherence": [
                check("RULE-R1", "reference_spec", "failed"),
                check("TABLE-RULE-01.01", "reference_spec", "unresolved"),
            ],
        }
    )
    fixed["metrics"]["rules_adherence"]["score"] = 0
    fixed["findings"] = [
        {
            "category": "rules_adherence",
            "description": "sums several currencies",
            "expected_behavior": "amounts per currency",
            "observed_behavior": "one total",
            "evidence": "SUM(c.amount)",
            "severity": "high",
            "evidence_status": "supported",
        }
    ]
    silent["findings"] = fixed["findings"]
    result, llm = await run(*grounded, [silent, fixed])
    assert len(llm.calls) == 2  # a silently dropped rule is never accepted
    assert "missing ['TABLE-RULE-01.01']" in llm.calls[1].user
    assert result.failed_requirements == ["rules_adherence: RULE-R1"]
    assert result.unresolved_requirements == ["rules_adherence: TABLE-RULE-01.01"]
    rules = result.metrics["rules_adherence"]
    assert [(c.requirement_id, c.requirement_source) for c in rules.checks] == [
        ("RULE-R1", "reference_spec"),
        ("TABLE-RULE-01.01", "reference_spec"),
    ]
    assert attempts(settings) >= 2


# --- FIX 2: response-level criteria are never forced into an SQL metric ------------------------


def test_response_level_criteria_are_not_forced_into_sql_metrics(
    sample_packs: list[ReferencePack],
) -> None:
    config = builtin_metric_config(["relevance", "semantic_correctness", "rules_adherence"])
    pack = sample_packs[0]
    request = context_request(config.definitions, None, pack)
    assert request.evidence_kinds == [EvidenceKind.ANSWER, EvidenceKind.SQL]
    criteria = [
        SuccessCriterionDraft(id="C1", requirement="SQL sums net spend", observed_in="sql"),
        SuccessCriterionDraft(
            id="C2", requirement="states that SQL cannot be executed", observed_in="answer"
        ),
    ]
    accepted = EvaluationContextDraft(
        success_criteria=criteria,
        metric_mapping={"semantic_correctness": MetricMappingDraft(criteria=["C1"])},
    )
    assert mapping_problems(accepted, request) == []  # C2 unmapped is not a rejection
    context = build_context(accepted, request, None, pack)
    assert [c.id for c in context.success_criteria] == ["C1", "C2"]  # every criterion kept
    assert [c.id for c in context.unmapped_criteria()] == ["C2"]
    assert context.success_criteria[1].observed_in is EvidenceKind.ANSWER

    forced = accepted.model_copy(
        update={
            "metric_mapping": {"semantic_correctness": MetricMappingDraft(criteria=["C1", "C2"])}
        }
    )
    [problem] = mapping_problems(forced, request)
    assert problem.startswith(
        "semantic_correctness: C2 is observed in the answer, but semantic_correctness evaluates "
        "only the sql"
    )

    summary = traceability.coverage_gaps(
        _question_with(context, pack),
        {"semantic_correctness": _scored()},
        {d.metric_id: d for d in config.definitions},
    )
    assert summary == [
        "coverage_gap: C2 (states that SQL cannot be executed) is not mapped to any selected "
        "metric (none observes it): only the goal-completion assessment reads it, no metric "
        "score checks it."
    ]


def _question_with(context: Any, pack: ReferencePack) -> Any:
    from synthetic_user.models.question import QuestionCase

    return QuestionCase.model_validate(
        {
            "question_id": "Q_001",
            "scenario_id": "SCN_001",
            "persona_id": "p",
            "mode": "grounded",
            "reference_pack_id": pack.pack_id,
            "utterance": "u",
            "intended_goal": "g",
            "intent": {"task_type": "ranking"},
            "question_characteristics": {
                "style": "precise",
                "ambiguity_level": "low",
                "complexity": "simple",
            },
            "evaluation_context": context.model_dump(mode="json", by_alias=True),
        }
    )


def _scored() -> Any:
    from synthetic_user.models.evaluation import MetricResult

    return MetricResult(status=MetricStatus.SCORED, score=1.0, rationale="ok")


# --- FIX 3: one invalid metric keeps the other metrics -----------------------------------------


async def test_invalid_judge_output_for_one_metric_preserves_other_scores(
    grounded: Fixture,
) -> None:
    settings = grounded[0]
    checks = good_checks()
    checks["rules_adherence"] = [check("RULE-R1", "reference_spec")]  # TABLE-RULE never decided
    bad = reply(checks)
    result, llm = await run(*grounded, [bad] * attempts(settings))
    assert len(llm.calls) == attempts(settings)  # the existing retries only, no extra call
    assert {m: r.status.value for m, r in result.metrics.items()} == {
        "relevance": "scored",
        "semantic_correctness": "scored",
        "rules_adherence": "evaluation_error",
        "schema_faithfulness": "scored",
    }
    rules = result.metrics["rules_adherence"]
    assert "missing ['TABLE-RULE-01.01']" in rules.rationale and rules.checks == []
    semantic = result.metrics["semantic_correctness"]
    assert [c.status for c in semantic.checks] == [RequirementStatus.PASSED] * 2  # kept as is
    assert any("rules_adherence only" in n for n in result.notes)
    # the verdict still uses the existing deterministic policy: the errored metric cannot be
    # settled, the valid ones pass -> inconclusive, naming only rules_adherence
    assert result.verdict is not None and result.verdict.value == "inconclusive"
    reason = result.verdict_reason or ""
    assert "Rules Adherence" in reason or "rules_adherence" in reason
    assert "Semantic Correctness" not in reason and "semantic_correctness" not in reason


async def test_a_shared_problem_is_one_error_named_once(grounded: Fixture) -> None:
    settings = grounded[0]
    bad = reply()
    bad.pop("goal_completion")
    result, _ = await run(*grounded, [bad] * attempts(settings))
    assert {r.status for r in result.metrics.values()} == {MetricStatus.EVALUATION_ERROR}
    shared = SHARED_ERROR.format(judge="General Judge", metrics=", ".join(METRICS))
    assert {r.rationale for r in result.metrics.values()} == {shared}
    [note] = [n for n in result.notes if "general judge output invalid" in n]
    assert note.endswith(f"[affects: {', '.join(METRICS)}]")
    assert note.count("goal_completion") == 1  # deduplicated across the attempts
    assert "(attempts 1, 2, 3)" in note
    assert not any("goal_completion" in line for line in result.limitations)


# --- FIX 4: catalog mismatch diagnostics -------------------------------------------------------

SQL_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "references" / "sql_eval_catalog.yaml"
)


def test_catalog_mismatches_are_reported_accurately() -> None:
    reference = load_reference(SQL_FIXTURE, max_chars=200_000)
    index = CatalogIndex.for_reference(reference)
    assert index is not None
    draft = CatalogPackDraft.model_validate(
        {
            "tables": [
                {"table": "v_lpc_details", "columns": ["net_lpc_charge_usd_am"], "why": "lpc"}
            ],
            "rules": [],
        }
    )
    pack, _ = build_pack(draft, reference, "SCN_001")
    assert pack is not None
    parser = SqlglotParser()
    sql = "SELECT d.net_lpc_charge_usd_am FROM acsinsight.lpc_details d"
    rec = reconcile(parser.parse(sql, dialect=None), artifact=1, index=index, pack=pack)
    table, column = rec.identifiers
    # status preserved; the diagnosis says what the evidence supports, no more
    assert (table.status, table.diagnosis) == (
        IdentifierStatus.INVALID,
        IdentifierDiagnosis.CATALOG_MISMATCH,
    )
    assert table.identifier == "acsinsight.lpc_details"
    assert table.expected_catalog_tables == ["v_lpc_details"]  # listed, never assumed the same
    assert "database existence was not verified" in table.reason
    assert (column.status, column.diagnosis) == (
        IdentifierStatus.UNRESOLVED,
        IdentifierDiagnosis.UNRESOLVED_IDENTIFIER,
    )
    confirmed = reconcile(
        parser.parse("SELECT no_such_column FROM v_lpc_details", dialect=None),
        artifact=1,
        index=index,
        pack=pack,
    )
    assert confirmed.identifiers[1].diagnosis is IdentifierDiagnosis.CONFIRMED_INVALID_IDENTIFIER

    from synthetic_user.models.lookup import ReferenceCheckRecord

    summary = traceability.summarise(
        {}, {}, reference_checks=ReferenceCheckRecord(catalog_version="v", artifacts=[rec])
    )
    assert (
        "SQL artifact 1: table acsinsight.lpc_details is not listed in the catalog (catalog "
        "mismatch; cheat-sheet table(s) the SQL does not read: v_lpc_details). Database "
        "existence was not verified; similarly named tables are not treated as equivalent."
    ) in summary.limitations


# --- FIX 5: derived defects explain a reduced score --------------------------------------------


async def test_derived_defects_explain_reduced_scores_despite_complete_mapped_coverage(
    grounded: Fixture,
) -> None:
    checks = good_checks()
    defect = {
        "requirement_ids": ["METRIC:semantic_correctness"],
        "source": "derived",
        "status": "failed",
        "evidence_ids": ["SQL:1"],
        "reason": "an unrequested LIMIT 10 changes the population",
    }
    checks["semantic_correctness"] = [*checks["semantic_correctness"], defect]
    out = reply(checks)
    out["metrics"]["semantic_correctness"]["score"] = 0.5
    out["findings"] = [
        {
            "category": "semantic_correctness",
            "description": "unrequested limit",
            "expected_behavior": "every spender",
            "observed_behavior": "top 10 only",
            "evidence": "LIMIT 10",
            "severity": "high",
            "evidence_status": "supported",
        }
    ]
    result, llm = await run(*grounded, [out])
    assert len(llm.calls) == 1
    semantic = result.metrics["semantic_correctness"]
    assert semantic.coverage is not None
    assert (semantic.coverage.passed, semantic.coverage.assigned) == (2, 2)  # mapped: complete
    assert (semantic.coverage.derived, semantic.coverage.derived_failed) == (1, 1)
    derived = semantic.checks[-1]
    assert derived.observed_evidence_ids == ["SQL:1"]
    assert derived.score_impact == "reduced Semantic Correctness to 0.5"
    assert len(semantic.checks) == 3  # the finding does not add the same defect again
    assert semantic.score_explanation == (
        "2/2 mapped requirements passed; 1 additional defect outside the mapped requirements "
        "reduced Semantic Correctness to 0.5 despite the mapped requirements passing: CHECK:003 "
        "[SQL:1] an unrequested LIMIT 10 changes the population."
    )


async def test_a_finding_derived_check_cites_the_evidence_id_of_its_quote(
    grounded: Fixture,
) -> None:
    out = reply()
    out["metrics"]["semantic_correctness"]["score"] = 0.5
    out["findings"] = [
        {
            "category": "semantic_correctness",
            "description": "unrequested limit",
            "expected_behavior": "every spender",
            "observed_behavior": "top 10 only",
            "evidence": "LIMIT 10",
            "severity": "high",
            "evidence_status": "supported",
        }
    ]
    result, _ = await run(*grounded, [out])
    derived = result.metrics["semantic_correctness"].checks[-1]
    assert (derived.requirement_source, derived.observed_evidence_ids) == ("derived", ["SQL:1"])
    assert derived.score_impact == "reduced Semantic Correctness to 0.5"


# --- FIX 6: accepted questions are usable; invalid enums are fed back --------------------------


@pytest.fixture
def env(
    sample_persona: Persona,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
    sample_reference: Reference,
    sample_packs: list[ReferencePack],
    app_config: AppConfig,
) -> Env:
    return Env(
        sample_persona,
        sample_product,
        sample_scenarios,
        sample_reference,
        sample_packs,
        app_config,
    )


async def test_accepted_questions_cannot_silently_disappear(env: Env) -> None:
    bad = q("Who spends most?", style="direct, task-oriented, conversational")
    good = q("Who spends most?")
    llm = FakeLLMClient([batch(bad), batch(good)])
    result = await env.run(llm, count=1)
    assert len(llm.calls) == 2  # the enum problem was fed back instead of being found later
    assert [x.utterance for x in result.questions] == ["Who spends most?"]
    assert not result.rejected


async def test_invalid_enum_values_are_rejected_with_clear_reasons(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    bad = q("Who spends most?", style="direct, task-oriented, conversational")
    llm = FakeLLMClient([batch(bad)] * attempts(env.settings))
    with caplog.at_level(logging.WARNING):
        result = await env.run(llm, count=1)
    assert not result.questions
    [reason] = result.rejected[0].reasons
    assert reason.startswith("question_characteristics.style: Input should be 'precise'")
    assert "use exactly the field names and the enum values the output contract" in reason
    feedback = llm.calls[1].user
    assert "question 1: question_characteristics.style" in feedback
    assert "SCN_001: question 1 discarded" in caplog.text  # never dropped silently


async def test_criteria_observed_in_is_required_when_criteria_are_mapped(env: Env) -> None:
    question = with_metric_expectations(q("Who spends most?"), {})
    for criterion in question["evaluation_context"]["success_criteria"]:
        criterion.pop("observed_in")
    llm = FakeLLMClient([batch(question)] * attempts(env.settings))
    result = await env.run(llm, count=1, metrics=("semantic_correctness",))
    assert not result.questions
    assert "C1: observed_in is required (answer | sql)" in result.rejected[0].reasons


# --- FIX 7: time windows ----------------------------------------------------------------------


def test_ytd_yoy_and_rolling_12_month_yoy_remain_distinct(
    sample_persona: Persona,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
) -> None:
    scenario_rules = " ".join(SCENARIO_PROMPT_FILE.read_text(encoding="utf-8").split())
    assert (
        "Year-to-date versus the same period of the prior year (YTD YoY) and the last 12 months "
        "versus the 12 months before (rolling-12-month YoY) are DIFFERENT comparisons"
    ) in scenario_rules
    assert "it never replaces a period the goal explicitly names" in scenario_rules
    _, user = build_question_prompt(
        sample_persona,
        sample_product,
        sample_scenarios[0],
        None,
        QuestionPromptOptions(
            count=1,
            ambiguity_level=sample_persona_level(),
            language="en",
            allow_vague_questions=False,
        ),
    )
    text = " ".join(user.split())
    assert "never copy a scenario criterion the utterance contradicts or does not request" in text
    assert (
        "YTD versus prior YTD and rolling 12 months versus the prior 12 months are distinct "
        "comparisons"
    ) in text
    assert "it never replaces an explicitly requested period" in text


def sample_persona_level() -> Any:
    from synthetic_user.models.base import AmbiguityLevel

    return AmbiguityLevel.LOW


def test_question_prompt_shortens_long_value_lists(
    sample_persona: Persona,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
    sample_reference: Reference,
    sample_packs: list[ReferencePack],
) -> None:
    pack = sample_packs[0]
    table = pack.tables[0]
    column = table.columns[0].model_copy(update={"values": [f"v{i}" for i in range(20)]})
    long_pack = pack.model_copy(
        update={"tables": [table.model_copy(update={"columns": [column]}), *pack.tables[1:]]}
    )
    options = QuestionPromptOptions(
        count=1,
        ambiguity_level=sample_persona_level(),
        language="en",
        allow_vague_questions=False,
        max_listed_values=3,
    )
    _, user = build_question_prompt(
        sample_persona,
        sample_product,
        sample_scenarios[0],
        (long_pack, sample_reference),
        options,
    )
    assert "[20 values: v0, v1, v2, ...]" in user and "v19" not in user
    assert "v19" in long_pack.render()  # the stored pack (evaluation evidence) is unchanged


# --- 10: run.log, the stored evaluation and the report agree ------------------------------------


async def test_run_log_structured_output_and_report_agree(
    grounded: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    logged: list[list[tuple[str, Any]]] = []
    monkeypatch.setattr(
        traceability, "record_evaluation", lambda subject, detail, sections: logged.append(sections)
    )
    checks = good_checks()
    checks["rules_adherence"] = [check("RULE-R1", "reference_spec")]
    out = reply(checks)
    out["metrics"]["rules_adherence"]["not_applicable_rules"] = ["TABLE-RULE-01.01"]
    result, _ = await run(*grounded, [out])
    [sections] = logged
    decisions = dict(sections)["E. METRIC DECISIONS"]
    verdict = dict(sections)["F. FINAL VERDICT"]
    cells = _evaluation_cells(result, list(METRICS))
    for metric_id, stored in result.metrics.items():
        log = decisions[metric_id]
        assert log["score"] == stored.score
        assert log["score_explanation"] == stored.score_explanation
        assert log["not_applicable_rules"] == stored.not_applicable_rules
        assert len(log["mapped_requirement_checks"]) + len(log["derived_checks"]) == len(
            stored.checks
        )
        if stored.score_explanation:
            assert cells[f"{metric_id}_coverage"].startswith(stored.score_explanation)
    assert "rules not applicable: TABLE-RULE-01.01" in cells["rules_adherence_checks"]
    assert verdict["verdict"] == result.verdict.value if result.verdict else True
    assert verdict["limitations"] == result.limitations
    assert cells["final_verdict"] == verdict["verdict"]


# --- live report: SQL:1 refused when no SQL metric was selected ---------------------------------


async def test_sql_artifact_id_is_citable_without_sql_metrics(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
) -> None:
    """Live report: with only relevance + correctness selected, the response view
    labelled the SQL artifact SQL:1 but validation refused SQL:1 (it was registered only for
    parsed SQL), so both metrics ended in evaluation_error after 3 attempts."""
    from nodes.test_evaluator import ANSWER, _execution, _question, all_pass
    from synthetic_user.nodes.evaluator import EvaluationCase, Evaluator

    scenario, pack = sample_scenarios[0], sample_packs[0]
    metrics = ("relevance", "correctness")
    out = all_pass(*metrics)
    out["metrics"]["relevance"]["checks"] = []
    out["metrics"]["correctness"]["checks"] = [
        {
            "requirement_ids": ["C1"],
            "source": "success_criteria",
            "status": "passed",
            "evidence_ids": ["SQL:1", "RESPONSE"],
            "reason": "the SQL ranks the employees",
        }
    ]
    llm = FakeLLMClient([out])
    evaluator = Evaluator(
        llm,
        app_config,
        logging.getLogger("t"),
        reference=sample_reference,
        metric_config=builtin_metric_config(list(metrics)),
    )
    question = _question(scenario, pack)
    data = question.model_dump(mode="json", by_alias=True)
    data["evaluation_context"]["metric_mapping"] = {
        "relevance": {"criteria": [], "comparison_basis": "question_vs_response"},
        "correctness": {
            "criteria": ["C1"],
            "comparison_basis": "criteria_vs_response",
        },
    }
    question = type(question).model_validate(data)
    result = await evaluator.evaluate(
        EvaluationCase(question, scenario, _execution(ANSWER)), sample_product
    )
    assert len(llm.calls) == 1, llm.calls[-1].user[-600:]
    assert '"evidence_id": "SQL:1"' in llm.calls[0].user  # what the judge is shown ...
    assert result.metrics["correctness"].status is MetricStatus.SCORED  # ... resolves
    assert "SQL:1" in result.metrics["correctness"].observed_evidence_ids
