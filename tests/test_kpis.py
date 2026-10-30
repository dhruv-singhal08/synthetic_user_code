from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from typing import Any

from synthetic_user.kpis import compute_kpis, consolidate_findings, kpis_csv
from synthetic_user.models.evaluation import EvaluationResult, MetricName
from synthetic_user.models.execution import ExecutionRecord
from synthetic_user.models.kpi import RunKpis
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario

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


def _metric(score: float | None, status: str = "scored") -> dict[str, Any]:
    return {"status": status, "score": score, "rationale": "r"}


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
            "evaluation_id": f"EVAL_{i:03d}",
            "question_id": f"Q_{i:03d}",
            "scenario_id": scenario_id,
            "execution_id": f"EXE_{i:03d}",
            "product_outcome": outcome,
            "mode": "grounded",
            "verdict": verdict,
            "evaluation_status": "evaluated" if verdict else "recorded_only",
            "metrics": metrics,
            "findings": findings,
            "bug_candidates": bugs,
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
        _execution(4, {"outcome": "denied", "message": "no"}, 50),
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
        _evaluation(4, "SCN_003", None, outcome="denied"),
        _evaluation(5, "SCN_003", None, outcome="timeout"),
    ]
    kpis = compute_kpis(
        run_id="RUN_1",
        product_id="p",
        mode=GenerationMode.GROUNDED,
        scenarios=sample_scenarios,
        executions=executions,
        evaluations=evaluations,
        now=NOW,
    )
    assert (kpis.questions_asked, kpis.judged) == (5, 3)
    assert kpis.verdicts == {"pass": 1, "fail": 1, "inconclusive": 1}
    assert (kpis.accuracy, kpis.task_success_rate, kpis.inconclusive_rate) == (0.333,) * 3
    assert kpis.availability == 0.8  # one timeout of five
    assert (kpis.latency_p50_ms, kpis.latency_p95_ms) == (100.0, 300.0)
    by = {m.metric: m for m in kpis.metrics}
    assert list(by) == [
        MetricName.SEMANTIC_CORRECTNESS,
        MetricName.RULES_ADHERENCE,
        MetricName.SCHEMA_FAITHFULNESS,
    ]
    semantic = by[MetricName.SEMANTIC_CORRECTNESS]
    assert (semantic.mean, semantic.pass_rate, semantic.n, semantic.inconclusive) == (
        0.5,
        0.5,
        2,
        1,
    )
    rules = by[MetricName.RULES_ADHERENCE]
    assert (rules.mean, rules.n, rules.not_applicable) == (1.0, 2, 1)
    schema = by[MetricName.SCHEMA_FAITHFULNESS]
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
        "0 not applicable, 1 inconclusive",
    ]
    assert table["Accuracy"][1:3] == ["0.333", "3"]
    assert table["Availability"][1] == "0.8"
    assert table["New bugs"][4] == "high 1"
    assert table["Questions asked"][4] == "recorded only: denied 1, timeout 1"


def test_empty_run_has_no_rates(sample_scenarios: list[Scenario]) -> None:
    kpis = compute_kpis(
        run_id="RUN_1",
        product_id="p",
        mode=GenerationMode.UNGROUNDED,
        scenarios=sample_scenarios,
        executions=[],
        evaluations=[],
        now=NOW,
    )
    assert kpis.accuracy is None and kpis.availability is None
    # With nothing evaluated the three NL2SQL metrics are still reported (empty).
    assert [m.metric for m in kpis.metrics] == [
        MetricName.SEMANTIC_CORRECTNESS,
        MetricName.RULES_ADHERENCE,
        MetricName.SCHEMA_FAITHFULNESS,
    ]
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
    executions = [
        _execution(1, {"outcome": "timeout"}, None),
        _execution(2, {"outcome": "answered", "text": "no query"}, 10),
    ]
    executions = [e.model_copy(update={"product_id": "sample_nl2sql"}) for e in executions]
    evaluations = [_evaluation(1, "SCN_001", None, outcome="timeout")]
    summary = consolidate_findings(evaluations, executions)
    (issue,) = summary.issues
    assert issue.theme == "no_sql_generated" and issue.affected_question_ids == ["Q_001", "Q_002"]
    assert "availability" in issue.categories


def test_old_evaluations_and_kpis_stay_readable() -> None:
    legacy = _evaluation(1, "SCN_001", "pass", None).model_dump(mode="json")
    legacy["metrics"] = {
        "relevancy": _metric(1),
        "correctness": _metric(0.5),
        "groundedness": _metric(None, "not_applicable"),
        "hallucination_rate": _metric(0),
        "completeness": _metric(1),
    }
    legacy["claims"] = [{"text": "t", "status": "supported", "reason": "r"}]
    migrated = EvaluationResult.model_validate(legacy)
    assert migrated.metrics is not None
    assert set(migrated.metrics.by_name()) == {
        MetricName.SEMANTIC_CORRECTNESS,
        MetricName.RULES_ADHERENCE,
        MetricName.SCHEMA_FAITHFULNESS,
    }
    assert migrated.metrics.semantic_correctness is not None
    assert migrated.metrics.semantic_correctness.score == 0.5

    old_kpis = compute_kpis(
        run_id="RUN_1",
        product_id="p",
        mode=GenerationMode.GROUNDED,
        scenarios=[],
        executions=[],
        evaluations=[],
        now=NOW,
    ).model_dump(mode="json")
    old_kpis.pop("accuracy")
    old_kpis["task_success_rate"] = 0.5
    old_kpis["metrics"] = [
        {**old_kpis["metrics"][0], "metric": "correctness"},
        {**old_kpis["metrics"][0], "metric": "hallucination_rate"},
    ]
    kpis = RunKpis.model_validate(old_kpis)
    assert kpis.accuracy == 0.5
    assert [m.metric for m in kpis.metrics] == [MetricName.SEMANTIC_CORRECTNESS]
