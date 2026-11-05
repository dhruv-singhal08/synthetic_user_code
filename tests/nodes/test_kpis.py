from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from typing import Any

from synthetic_user.metrics.compiler import _unavailable
from synthetic_user.metrics.registry import builtin_metric_config, freeze
from synthetic_user.models.evaluation import EvaluationResult
from synthetic_user.models.execution import ExecutionRecord
from synthetic_user.models.kpi import RunKpis
from synthetic_user.models.metrics import CustomMetricInput, MetricAvailability, MetricSelection
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator.kpis import compute_kpis, consolidate_findings, kpis_csv
from synthetic_user.nodes.evaluator.no_sql import NO_SQL_REASON

SQL_CONFIG = builtin_metric_config(
    ["semantic_correctness", "rules_adherence", "schema_faithfulness"]
)

NOW = datetime(2026, 10, 18, tzinfo=UTC)


def _execution(i: int, response: dict[str, Any], latency: float | None) -> ExecutionRecord:
    return ExecutionRecord.model_validate(
        {
            "execution_id": f"EXE_{i:03d}",
            "question_id": f"Q_{i:03d}",
            "product_id": "p",
            "adapter": "fixture",
            "submitted_question": "q",
            "started_at": NOW,
            "latency_ms": latency,
            "response": response,
        }
    )


def _metric(
    score: float | None, status: str = "scored", rationale: str = "r", **extra: Any
) -> dict[str, Any]:
    return {"status": status, "score": score, "rationale": rationale, **extra}


def _finding(
    i: int,
    severity: str,
    known: bool,
    category: str = "semantic_correctness",
    description: str = "d",
) -> dict[str, Any]:
    return {
        "finding_id": f"F_{i:03d}",
        "category": category,
        "description": description,
        "expected_behavior": "e",
        "observed_behavior": "o",
        "severity": severity,
        "evidence_status": "supported",
        "source": "llm",
        "matches_known_gap": known,
    }


def _evaluation(
    i: int,
    scenario_id: str,
    verdict: str | None,
    metrics: dict[str, Any] | None = None,
    bug_findings: list[dict[str, Any]] | None = None,
    outcome: str = "answered",
    notes: list[str] | None = None,
) -> EvaluationResult:
    findings = bug_findings or []
    bugs = (
        [
            {
                "bug_id": f"BUG_{i:03d}",
                "title": "t",
                "description": "d",
                "reproduction_question": "q",
                "product_id": "p",
                "question_id": f"Q_{i:03d}",
                "scenario_id": scenario_id,
                "execution_id": f"EXE_{i:03d}",
                "finding_ids": [f["finding_id"] for f in findings],
            }
        ]
        if findings
        else []
    )
    return EvaluationResult.model_validate(
        {
            "schema_version": 2,
            "metric_config_id": SQL_CONFIG.config_id,
            "evaluation_id": f"EVAL_{i:03d}",
            "question_id": f"Q_{i:03d}",
            "scenario_id": scenario_id,
            "execution_id": f"EXE_{i:03d}",
            "product_outcome": outcome,
            "mode": "grounded",
            "verdict": verdict,
            "evaluation_status": "evaluated" if verdict else "recorded_only",
            "metrics": metrics or {},
            "findings": findings,
            "bug_candidates": bugs,
            "notes": notes or [],
        }
    )


def _metrics(semantic: float | None, rules: float | None, schema: float) -> dict[str, Any]:
    return {
        "semantic_correctness": _metric(semantic)
        if semantic is not None
        else _metric(None, "inconclusive"),
        "rules_adherence": _metric(rules) if rules is not None else _metric(None, "not_applicable"),
        "schema_faithfulness": _metric(schema),
    }


def test_kpis_from_evaluations(sample_scenarios: list[Scenario]) -> None:
    answered = {"outcome": "answered", "text": "x"}
    executions = [
        _execution(1, answered, 100),
        _execution(2, answered, 300),
        _execution(3, answered, 200),
        _execution(4, {"outcome": "clarification", "message": "which?"}, 50),
        _execution(5, {"outcome": "timeout"}, None),
    ]
    evaluations = [
        _evaluation(1, "SCN_001", "pass", _metrics(1, 1, 1)),
        _evaluation(
            2,
            "SCN_001",
            "fail",
            _metrics(0, 1, 0.5),
            [_finding(1, "high", known=False), _finding(2, "low", known=True)],
        ),
        _evaluation(
            3,
            "SCN_003",
            "inconclusive",
            _metrics(None, None, 1),
            [_finding(3, "medium", known=True)],
        ),
        _evaluation(4, "SCN_003", None, outcome="clarification"),
        _evaluation(5, "SCN_003", None, outcome="timeout"),
    ]
    kpis = compute_kpis(
        run_id="RUN_1",
        product_id="p",
        mode=GenerationMode.GROUNDED,
        scenarios=sample_scenarios,
        executions=executions,
        evaluations=evaluations,
        metric_config=SQL_CONFIG,
        now=NOW,
    )
    assert (kpis.questions_asked, kpis.judged) == (5, 3)
    assert kpis.selected_metrics == SQL_CONFIG.selected_metric_ids
    assert kpis.category_groups == {"SQL Quality": SQL_CONFIG.selected_metric_ids}
    assert kpis.verdicts == {"pass": 1, "fail": 1, "inconclusive": 1}
    assert (kpis.accuracy, kpis.task_success_rate, kpis.inconclusive_rate) == (0.333,) * 3
    assert kpis.availability == 0.8  # one timeout of five
    assert (kpis.latency_p50_ms, kpis.latency_p95_ms) == (100.0, 300.0)
    by = {m.metric: m for m in kpis.metrics}
    assert list(by) == ["semantic_correctness", "rules_adherence", "schema_faithfulness"]
    semantic = by["semantic_correctness"]
    assert (semantic.mean, semantic.pass_rate, semantic.n, semantic.inconclusive) == (
        0.5,
        0.5,
        2,
        1,
    )
    rules = by["rules_adherence"]
    assert (rules.mean, rules.n, rules.not_applicable) == (1.0, 2, 1)
    schema = by["schema_faithfulness"]
    assert (schema.mean, schema.pass_rate) == (0.833, 0.667)  # pass = score 1
    assert not any(m.lower_is_better for m in kpis.metrics)
    # BUG_002 has an unknown-gap finding -> new (severity = its worst, high); BUG_003 known
    assert (kpis.new_bugs, kpis.new_bugs_by_severity["high"], kpis.known_gap_bugs) == (1, 1, 1)
    by_task = {g.group: g for g in kpis.by_task_type}
    assert by_task["ranking"].questions == 2 and by_task["ranking"].pass_rate == 0.5
    assert by_task["other"].judged == 1
    assert {g.group for g in kpis.by_capability} == {"supported", "unsupported"}

    rows = list(csv.reader(io.StringIO(kpis_csv(kpis))))
    assert rows[0] == ["kpi", "value", "n", "excluded", "note"]
    table = {r[0]: r for r in rows[1:]}
    assert table["Semantic Correctness (mean)"][1:4] == [
        "0.5",
        "2",
        "0 not applicable, 0 cannot generate, 0 evidence unavailable, 1 inconclusive, "
        "0 evaluation error, 0 unavailable",
    ]
    assert table["Accuracy"][1:3] == ["0.333", "3"]
    assert table["Availability"][1] == "0.8"
    assert table["New bugs"][4] == "high 1"
    assert table["Questions asked"][4] == "recorded only: clarification 1, timeout 1"
    assert "same value" in table["Accuracy"][4]  # task_success_rate is the same formula


def test_empty_run_has_no_rates(sample_scenarios: list[Scenario]) -> None:
    kpis = compute_kpis(
        run_id="RUN_1",
        product_id="p",
        mode=GenerationMode.UNGROUNDED,
        scenarios=sample_scenarios,
        executions=[],
        evaluations=[],
        metric_config=SQL_CONFIG,
        now=NOW,
    )
    assert kpis.accuracy is None and kpis.availability is None
    # With nothing evaluated the selected metrics are still reported (empty).
    assert [m.metric for m in kpis.metrics] == SQL_CONFIG.selected_metric_ids
    assert all(m.mean is None and m.n == 0 for m in kpis.metrics)
    assert kpis_csv(kpis).count("\n") == 15  # header + accuracy + 3x2 metric rows + 7 others


def test_findings_are_consolidated_into_stable_themes() -> None:
    evaluations = [
        _evaluation(
            1,
            "SCN_001",
            "fail",
            _metrics(1, 0, 1),
            [_finding(1, "high", False, "rules_adherence", "Uses an invalid join on merchant_id")],
        ),
        _evaluation(
            2,
            "SCN_001",
            "fail",
            _metrics(1, 0, 1),
            [_finding(2, "medium", False, "rules_adherence", "The join key is wrong")],
        ),
        _evaluation(
            3,
            "SCN_003",
            "fail",
            _metrics(1, 1, 0.5),
            [_finding(3, "low", True, "schema_faithfulness", "Uses an undocumented column")],
        ),
    ]
    summary = consolidate_findings(evaluations)
    assert (summary.total_findings, summary.affected_questions) == (3, 3)
    joins, schema = summary.issues
    assert (joins.issue_id, joins.theme, joins.severity) == (
        "ISSUE_001",
        "joins_and_relationships",
        "high",
    )
    assert joins.affected_question_ids == ["Q_001", "Q_002"] and joins.occurrences == 2
    assert (schema.theme, schema.known_gap) == ("schema_and_identifiers", True)


def test_timeouts_of_a_sql_product_become_one_no_sql_issue() -> None:
    """No-SQL is decided by the evaluator (selected SQL metrics), never by a product keyword."""
    executions = [
        _execution(1, {"outcome": "timeout"}, None),
        _execution(2, {"outcome": "answered", "text": "no query"}, 10),
    ]
    evaluations = [
        _evaluation(1, "SCN_001", None, outcome="timeout", notes=[NO_SQL_REASON]),
        _evaluation(2, "SCN_001", "fail", _metrics(0, None, 1), notes=[NO_SQL_REASON]),
    ]
    summary = consolidate_findings(evaluations, executions, SQL_CONFIG)
    (issue,) = summary.issues
    assert issue.theme == "no_sql_generated" and issue.affected_question_ids == ["Q_001", "Q_002"]
    assert "availability" in issue.categories


def test_old_evaluations_and_kpis_stay_readable() -> None:
    legacy = _evaluation(1, "SCN_001", "pass", None).model_dump(mode="json")
    for key in ("schema_version", "metric_config_id"):
        legacy.pop(key)
    legacy["metrics"] = {
        "relevancy": _metric(1),
        "correctness": _metric(0.5),
        "groundedness": _metric(None, "not_applicable"),
        "hallucination_rate": _metric(0),
        "completeness": _metric(1),
    }
    legacy["claims"] = [{"text": "t", "status": "supported", "reason": "r"}]
    migrated = EvaluationResult.model_validate(legacy)
    assert migrated.schema_version == 1 and migrated.legacy_evaluation_profile == "nl2sql"
    assert set(migrated.metrics) == {"semantic_correctness"}
    assert migrated.metrics["semantic_correctness"].score == 0.5

    old_kpis = compute_kpis(
        run_id="RUN_1",
        product_id="p",
        mode=GenerationMode.GROUNDED,
        scenarios=[],
        executions=[],
        evaluations=[migrated],
        now=NOW,
    ).model_dump(mode="json")
    assert [m["metric"] for m in old_kpis["metrics"]] == ["semantic_correctness"]
    old_kpis.pop("accuracy")
    old_kpis["task_success_rate"] = 0.5
    old_kpis["metrics"] = [
        {**old_kpis["metrics"][0], "metric": "correctness"},
        {**old_kpis["metrics"][0], "metric": "hallucination_rate"},
    ]
    kpis = RunKpis.model_validate(old_kpis)
    assert kpis.accuracy == 0.5
    assert [m.metric for m in kpis.metrics] == ["semantic_correctness"]


def test_legacy_agent_runs_report_original_metric_ids() -> None:
    legacy = _evaluation(1, "SCN_001", "pass").model_dump(mode="json")
    for key in ("schema_version", "metric_config_id"):
        legacy.pop(key)
    legacy.update(
        evaluation_profile="agent",
        metrics={"relevance": _metric(1), "completion": _metric(0.5)},
    )
    kpis = compute_kpis(
        run_id="RUN_1",
        product_id="p",
        mode=GenerationMode.GROUNDED,
        scenarios=[],
        executions=[],
        evaluations=[EvaluationResult.model_validate(legacy)],
        now=NOW,
    )
    by = {m.metric: m for m in kpis.metrics}
    assert set(by) == {"relevance", "completion"}  # never relabelled correctness
    assert by["completion"].display_name == "Completion (legacy)"
    assert by["completion"].origin == "legacy"


def test_status_taxonomy_counts_and_reasons() -> None:
    config = builtin_metric_config(["relevance", "context_recall", "tool_correctness"])
    evaluations = [
        _evaluation(
            i,
            "SCN_001",
            "pass",
            {
                "relevance": _metric(1),
                "context_recall": _metric(
                    None,
                    "evidence_unavailable",
                    "Retrieved context not_requested",
                    missing_requirement="retrieved_context",
                ),
                "tool_correctness": _metric(
                    None, "cannot_generate", "the reference defines no tools (agent_context)"
                ),
            },
        )
        for i in (1, 2)
    ] + [
        _evaluation(
            3,
            "SCN_001",
            "inconclusive",
            {
                "relevance": _metric(
                    None, "evaluation_error", "general judge provider call failed"
                ),
                "context_recall": _metric(
                    1.0,
                    details={
                        "disclosure": "expectation-based coverage",
                        "required_information": [],
                    },
                ),
                "tool_correctness": _metric(None, "not_applicable", "typed denial"),
            },
        )
    ]
    kpis = compute_kpis(
        run_id="RUN_1",
        product_id="p",
        mode=GenerationMode.GROUNDED,
        scenarios=[],
        executions=[],
        evaluations=evaluations,
        metric_config=config,
        now=NOW,
    )
    by = {m.metric: m for m in kpis.metrics}
    assert (by["relevance"].n, by["relevance"].evaluation_error) == (2, 1)
    recall = by["context_recall"]
    assert (recall.n, recall.evidence_unavailable, recall.mean) == (1, 2, 1.0)
    assert recall.reasons == {"evidence_unavailable: Retrieved context not_requested": 2}
    assert any("not exhaustive corpus recall" in note for note in recall.notes)
    tools = by["tool_correctness"]
    assert (tools.n, tools.cannot_generate, tools.not_applicable, tools.mean) == (0, 2, 1, None)
    assert kpis.category_groups == {
        "Answer Quality / Task Fulfilment": ["relevance"],
        "Retrieval Quality": ["context_recall"],
        "Agent": ["tool_correctness"],
    }


def test_mrr_per_question_and_run_level_with_exclusions() -> None:
    config = builtin_metric_config(["mrr"])

    def rr(score: float | None, first: int | None, status: str = "scored") -> dict[str, Any]:
        details = (
            {"reciprocal_rank": score, "first_relevant_rank": first} if score is not None else {}
        )
        missing = (
            {"missing_requirement": "ranked_context"} if status == "evidence_unavailable" else {}
        )
        return {"mrr": _metric(score, status, details=details, **missing)}

    evaluations = [
        _evaluation(1, "SCN_001", "pass", rr(1.0, 1)),  # first relevant at rank 1
        _evaluation(2, "SCN_001", "pass", rr(0.333333, 3)),  # rank 3
        _evaluation(3, "SCN_001", "pass", rr(0.0, None)),  # no relevant chunk
        _evaluation(4, "SCN_001", "pass", rr(0.0, None)),  # empty ranked results
        _evaluation(5, "SCN_001", "pass", rr(None, None, "evidence_unavailable")),  # no rank
        _evaluation(6, "SCN_001", None, outcome="timeout"),
    ]
    kpis = compute_kpis(
        run_id="RUN_1",
        product_id="p",
        mode=GenerationMode.GROUNDED,
        scenarios=[],
        executions=[],
        evaluations=evaluations,
        metric_config=config,
        now=NOW,
    )
    assert kpis.mrr is not None
    assert (kpis.mrr.value, kpis.mrr.denominator) == (0.333, 4)
    assert (kpis.mrr.excluded_missing_rank, kpis.mrr.excluded_other) == (1, 1)
    assert [r.reciprocal_rank for r in kpis.mrr.per_question] == [
        1.0,
        0.333333,
        0.0,
        0.0,
        None,
        None,
    ]
    assert [r.first_relevant_rank for r in kpis.mrr.per_question][:2] == [1, 3]
    assert "1 / rank of the first relevant chunk" in kpis.mrr.definition
    table = {r[0]: r for r in csv.reader(io.StringIO(kpis_csv(kpis)))}
    assert table["MRR (run level)"][1:4] == ["0.333", "4", "1 missing rank, 1 other"]


def test_compiler_failures_reported_and_custom_findings_consolidated() -> None:
    selection = MetricSelection(
        selected_metrics=["relevance"],
        custom_inputs=[
            CustomMetricInput(text="Workflow adherence", provenance="free text item 1"),
            CustomMetricInput(text="Dialect check", provenance="m.xlsx: Sheet1 row 2"),
        ],
    )
    relevance = builtin_metric_config(["relevance"]).definitions[0]
    workflow = _unavailable(1, selection.custom_inputs[0], "x").model_copy(
        update={
            "availability": MetricAvailability.AVAILABLE,
            "unavailable_reason": None,
            "display_name": "Workflow adherence",
        }
    )
    dialect = _unavailable(2, selection.custom_inputs[1], "no SQL dialect is configured")
    config = freeze(selection, [relevance, workflow, dialect], compiler_called=True, notes=[])
    evaluations = [
        _evaluation(
            1,
            "SCN_001",
            "fail",
            {
                "relevance": _metric(1),
                "custom_01": _metric(0),
                "custom_02": _metric(None, "unavailable"),
            },
            [_finding(1, "high", False, "custom_01", "Skipped the approval step")],
        )
    ]
    kpis = compute_kpis(
        run_id="RUN_1",
        product_id="p",
        mode=GenerationMode.GROUNDED,
        scenarios=[],
        executions=[],
        evaluations=evaluations,
        metric_config=config,
        now=NOW,
    )
    assert [(f.metric_id, f.reason) for f in kpis.compiler_failures] == [
        ("custom_02", "no SQL dialect is configured")
    ]
    assert kpis.compiler_failures[0].source_text == "Dialect check"
    by = {m.metric: m for m in kpis.metrics}
    assert (
        by["custom_01"].display_name == "Workflow adherence" and by["custom_01"].origin == "custom"
    )
    assert (by["custom_02"].unavailable, by["custom_02"].evaluation_error) == (1, 0)
    assert by["custom_02"].notes
    # findings of a custom metric consolidate without a fixed fallback map (no KeyError)
    summary = consolidate_findings(evaluations, [], config)
    (issue,) = summary.issues
    assert issue.theme == "other_custom_01" and issue.title == "Workflow adherence issues"
