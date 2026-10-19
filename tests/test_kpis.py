from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from typing import Any

from synthetic_user.kpis import compute_kpis, kpis_csv
from synthetic_user.models.evaluation import EvaluationResult, MetricName
from synthetic_user.models.execution import ExecutionRecord
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


def _finding(i: int, severity: str, known: bool) -> dict[str, Any]:
    return {
        "finding_id": f"F_{i:03d}",
        "category": "correctness",
        "description": "d",
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


def _metrics(
    relevancy: float, correctness: float | None, hallucination: float, complete: float
) -> dict[str, Any]:
    return {
        "relevancy": _metric(relevancy),
        "correctness": _metric(correctness)
        if correctness is not None
        else _metric(None, "inconclusive"),
        "groundedness": _metric(None, "not_applicable"),
        "hallucination_rate": _metric(hallucination),
        "completeness": _metric(complete),
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
        _evaluation(1, "SCN_001", "pass", _metrics(1, 1, 0, 1)),
        _evaluation(
            2,
            "SCN_001",
            "fail",
            _metrics(1, 0, 0.5, 0.5),
            [_finding(1, "high", known=False), _finding(2, "low", known=True)],
        ),
        _evaluation(
            3,
            "SCN_003",
            "inconclusive",
            _metrics(0.5, None, 0, 1),
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
    assert (kpis.task_success_rate, kpis.inconclusive_rate) == (0.333, 0.333)
    assert kpis.availability == 0.8  # one timeout of five
    assert (kpis.latency_p50_ms, kpis.latency_p95_ms) == (100.0, 300.0)
    by = {m.metric: m for m in kpis.metrics}
    correctness = by[MetricName.CORRECTNESS]
    assert (correctness.mean, correctness.pass_rate, correctness.n) == (0.5, 0.5, 2)
    assert correctness.inconclusive == 1
    hallucination = by[MetricName.HALLUCINATION_RATE]
    assert hallucination.lower_is_better and hallucination.pass_rate == 0.667  # share at 0
    assert MetricName.GROUNDEDNESS not in by  # computed, not reported (D206)
    # BUG_002 has an unknown-gap finding -> new (severity = its worst, high); BUG_003 known
    assert (kpis.new_bugs, kpis.new_bugs_by_severity["high"], kpis.known_gap_bugs) == (1, 1, 1)
    by_task = {g.group: g for g in kpis.by_task_type}
    assert by_task["ranking"].questions == 2 and by_task["ranking"].pass_rate == 0.5
    assert by_task["other"].judged == 1
    assert {g.group for g in kpis.by_capability} == {"supported", "unsupported"}

    rows = list(csv.reader(io.StringIO(kpis_csv(kpis))))
    assert rows[0] == ["kpi", "value", "n", "excluded", "note"]
    table = {r[0]: r for r in rows[1:]}
    assert table["Correctness (mean)"][1:4] == ["0.5", "2", "0 not applicable, 1 inconclusive"]
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
    assert kpis.task_success_rate is None and kpis.availability is None
    assert all(m.mean is None and m.n == 0 for m in kpis.metrics)
    assert kpis_csv(kpis).count("\n") == 17  # header + 16 KPI rows
