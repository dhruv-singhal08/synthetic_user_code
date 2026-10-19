"""The single results table: one row per question, columns filled in as steps complete.

Scenarios without an accepted question still get one row (question columns blank). Only what
evaluation uses is shown (design v3); the full detail stays in the JSON artifacts.

Cell format (readable in Excel): a cell holding several items lists them as bullet points,
one per line ("• item"); a single item is written as is. Metric cells hold the score (0..1),
"inconclusive", or stay empty when the metric does not apply.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence

from synthetic_user.models.evaluation import (
    HALLUCINATED,
    EvaluationResult,
    MetricName,
    MetricStatus,
    Verdict,
)
from synthetic_user.models.execution import (
    AnsweredResponse,
    ClarificationResponse,
    DeniedResponse,
    ErrorResponse,
    ExecutionRecord,
    ResponseArtifact,
)
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import ReferencePack, render_join
from synthetic_user.models.scenario import Scenario

BULLET = "• "

SCENARIO_COLUMNS = ["scenario_id", "mode", "scenario_title", "business_goal"]
QUESTION_COLUMNS = [
    "question_id",
    "question",
    "expected_behavior",
    "success_criteria",
]
# The scenario's cheat sheet: tables (with columns) or documents; joins and rules.
REFERENCE_COLUMNS = ["cheat_sheet", "cheat_sheet_rules"]
EXECUTION_COLUMNS = [
    "product_outcome",
    "answer_text",
    "artifacts",
    "product_message",
    "latency_ms",
]
# Each reported metric with the reason behind its score, then the verdict and its reason
# (D206; groundedness is not reported for now).
EVALUATION_COLUMNS = [
    "relevancy",
    "relevancy_reason",
    "correctness",
    "correctness_reason",
    "hallucination_rate",
    "hallucination_reason",
    "completeness",
    "completeness_reason",
    "final_verdict",
    "verdict_reason",
    "bug_ids",
    "evaluation_notes",
]
COLUMNS = [
    "run_id",
    *SCENARIO_COLUMNS,
    *QUESTION_COLUMNS,
    *REFERENCE_COLUMNS,
    *EXECUTION_COLUMNS,
    *EVALUATION_COLUMNS,
]


def bullets(items: Sequence[str]) -> str:
    """Several items -> one bullet per line; one item -> as is; none -> empty."""
    items = [i for i in items if i]
    if len(items) <= 1:
        return items[0] if items else ""
    return "\n".join(f"{BULLET}{item}" for item in items)


def build_rows(
    run_id: str,
    scenarios: Sequence[Scenario],
    questions: Sequence[QuestionCase] = (),
    executions: Sequence[ExecutionRecord] = (),
    evaluations: Sequence[EvaluationResult] = (),
    packs: Sequence[ReferencePack] = (),
) -> list[dict[str, str]]:
    execution_by_question = {e.question_id: e for e in executions}
    evaluation_by_execution = {e.execution_id: e for e in evaluations}
    pack_by_id = {p.pack_id: p for p in packs}
    rows: list[dict[str, str]] = []
    for scenario in scenarios:
        pack = pack_by_id.get(scenario.reference_pack_id or "")
        base = {"run_id": run_id, **_scenario_cells(scenario), **_pack_cells(pack)}
        scenario_questions = [q for q in questions if q.scenario_id == scenario.id]
        if not scenario_questions:
            rows.append(_complete(base))
            continue
        for question in scenario_questions:
            row = {**base, **_question_cells(question)}
            execution = execution_by_question.get(question.question_id)
            if execution is not None:
                row.update(_execution_cells(execution))
                evaluation = evaluation_by_execution.get(execution.execution_id)
                if evaluation is not None:
                    row.update(_evaluation_cells(evaluation))
            rows.append(_complete(row))
    return rows


def to_csv(rows: Sequence[dict[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _complete(row: dict[str, str]) -> dict[str, str]:
    return {column: row.get(column, "") for column in COLUMNS}


def _scenario_cells(s: Scenario) -> dict[str, str]:
    return {
        "scenario_id": s.id,
        "mode": s.mode.value,
        "scenario_title": s.title,
        "business_goal": s.business_goal,
    }


def _pack_cells(pack: ReferencePack | None) -> dict[str, str]:
    if pack is None:
        return {}
    items = [
        f"{t.name} ({', '.join(c.name for c in t.columns)})" if t.columns else t.name
        for t in pack.tables
    ] + [d.name for d in pack.documents]
    rules = [
        *(f"join {render_join(j, j.verified)}" for j in pack.joins),
        *(f"{t.name}: {rule}" for t in pack.tables for rule in t.rules),
        *(f"{r.id}: {r.rule}" for r in pack.rules),
    ]
    return {"cheat_sheet": bullets(items), "cheat_sheet_rules": bullets(rules)}


def _question_cells(q: QuestionCase) -> dict[str, str]:
    e = q.expectations
    return {
        "question_id": q.question_id,
        "question": q.utterance,
        "expected_behavior": e.expected_behavior,
        "success_criteria": bullets(e.success_criteria),
    }


def _artifact(a: ResponseArtifact) -> str:
    content = a.content if isinstance(a.content, str) else json.dumps(a.content, indent=1)
    return f"{a.kind}:\n{content.strip()}"


def _execution_cells(e: ExecutionRecord) -> dict[str, str]:
    latency = e.latency_ms
    response = e.response
    text, message, artifacts = "", "", ""
    if isinstance(response, AnsweredResponse):
        text = response.text or ""
        artifacts = bullets([_artifact(a) for a in response.artifacts])
    elif isinstance(response, ClarificationResponse | DeniedResponse):
        message = response.message
    elif isinstance(response, ErrorResponse):
        message = response.message or response.error_type
    return {
        "product_outcome": e.outcome.value,
        "answer_text": text,
        "artifacts": artifacts,
        "product_message": message,
        "latency_ms": f"{latency:.0f}" if latency is not None else "",
    }


def _metric_cell(ev: EvaluationResult, name: MetricName) -> str:
    if ev.metrics is None:
        return ""
    result = ev.metrics.by_name()[name]
    if result.status is MetricStatus.INCONCLUSIVE:
        return "inconclusive"
    return f"{result.score:g}" if result.score is not None else ""


def _judged_reason(ev: EvaluationResult, name: MetricName) -> str:
    """The judge's reason for a 0 / 0.5 / 1 score, with the quote it relied on."""
    if ev.metrics is None:
        return ""
    result = ev.metrics.by_name()[name]
    if result.status is MetricStatus.NOT_APPLICABLE:
        return f"not applicable: {result.rationale}"
    return f"{result.rationale} - '{result.evidence}'" if result.evidence else result.rationale


def _hallucination_reason(ev: EvaluationResult) -> str:
    if ev.metrics is None:
        return ""
    bad = [c for c in ev.claims if c.status in HALLUCINATED]
    if bad:
        return bullets([f"'{c.text}' ({c.status.value}): {c.reason}" for c in bad])
    if ev.claims:
        return f"No hallucinated statements ({len(ev.claims)} checked)"
    return ev.metrics.hallucination_rate.rationale


def _verdict_reason(ev: EvaluationResult) -> str:
    if ev.verdict is None:
        return ""
    if ev.findings:
        return bullets(
            [
                f"{f.finding_id} [{f.severity.value}] {f.category}: {f.description}"
                for f in ev.findings
            ]
        )
    return "All metrics passed" if ev.verdict is Verdict.PASS else ""


def _evaluation_cells(ev: EvaluationResult) -> dict[str, str]:
    return {
        "relevancy": _metric_cell(ev, MetricName.RELEVANCY),
        "relevancy_reason": _judged_reason(ev, MetricName.RELEVANCY),
        "correctness": _metric_cell(ev, MetricName.CORRECTNESS),
        "correctness_reason": _judged_reason(ev, MetricName.CORRECTNESS),
        "hallucination_rate": _metric_cell(ev, MetricName.HALLUCINATION_RATE),
        "hallucination_reason": _hallucination_reason(ev),
        "completeness": _metric_cell(ev, MetricName.COMPLETENESS),
        "completeness_reason": bullets(
            [f"{c.status.value}: {c.criterion}" for c in ev.criteria_results]
        ),
        "final_verdict": ev.verdict.value if ev.verdict else "recorded_only",
        "verdict_reason": _verdict_reason(ev),
        "bug_ids": bullets([b.bug_id for b in ev.bug_candidates]),
        "evaluation_notes": bullets(list(ev.notes)),
    }
