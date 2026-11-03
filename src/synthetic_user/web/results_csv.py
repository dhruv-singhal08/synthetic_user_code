"""The single results table: one row per question, columns filled in as steps complete.

Scenarios without an accepted question still get one row (question columns blank). Only what
evaluation uses is shown (design v3); the full detail stays in the JSON artifacts.

Cell format (readable in Excel): a cell holding several items lists them as bullet points,
one per line ("• item"); a single item is written as is. success_criteria is one JSON cell
(D240). Each selected metric has three
columns: ``<id>`` (score 0..1, or empty when not scored), ``<id>_status`` (scored /
not_applicable / cannot_generate / evidence_unavailable / inconclusive / evaluation_error /
unavailable) and ``<id>_reason``. Metric columns follow the run's frozen metric selection
(legacy runs: the metric ids their evaluations recorded).
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from typing import Any

from synthetic_user.models.evaluation import (
    EvaluationResult,
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
from synthetic_user.models.metrics import CHEAT_SHEET_KEY, GENERAL_CRITERIA_KEY
from synthetic_user.models.question import (
    MetricExpectationStatus,
    QuestionCase,
    SuccessCriteria,
)
from synthetic_user.models.reference import ReferencePack, render_join
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator.no_sql import NO_SQL_REASON

BULLET = "• "

SCENARIO_COLUMNS = ["scenario_id", "mode", "scenario_title", "business_goal", "capability"]
# success_criteria is ONE structured (JSON) cell (D240): "general" (the question's criteria),
# one key per metric with a question-specific expectation and "cheat_sheet" (the scenario's
# tables (with columns) or documents, and its joins and rules).
QUESTION_COLUMNS = ["question_id", "question", "expected_behavior", "success_criteria"]
EXECUTION_COLUMNS = [
    "product_outcome",
    "response_classification",
    "answer_text",
    "artifacts",
    "tool_calls",
    "retrieved_context",
    "product_message",
    "latency_ms",
]
VERDICT_COLUMNS = ["final_verdict", "verdict_reason", "bug_ids", "evaluation_notes"]


def metric_columns(metric_ids: Sequence[str]) -> list[str]:
    return [col for m in metric_ids for col in (m, f"{m}_status", f"{m}_reason")]


def columns_for(metric_ids: Sequence[str]) -> list[str]:
    return [
        "run_id",
        *SCENARIO_COLUMNS,
        *QUESTION_COLUMNS,
        *EXECUTION_COLUMNS,
        *metric_columns(metric_ids),
        *VERDICT_COLUMNS,
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
    metric_ids: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """One row per question (scenario rows without questions). ``metric_ids`` defaults to the
    ids the evaluations recorded (legacy runs)."""
    if metric_ids is None:
        metric_ids = list(dict.fromkeys(m for ev in evaluations for m in ev.metrics))
    columns = columns_for(metric_ids)
    execution_by_question = {e.question_id: e for e in executions}
    evaluation_by_execution = {e.execution_id: e for e in evaluations}
    pack_by_id = {p.pack_id: p for p in packs}
    rows: list[dict[str, str]] = []
    for scenario in scenarios:
        pack = pack_by_id.get(scenario.reference_pack_id or "")
        base = {"run_id": run_id, **_scenario_cells(scenario)}
        scenario_questions = [q for q in questions if q.scenario_id == scenario.id]
        if not scenario_questions:
            # Before questions exist the row still shows the scenario's cheat sheet.
            cheat_sheet = {CHEAT_SHEET_KEY: _pack_view(pack)} if pack is not None else {}
            rows.append(_complete({**base, "success_criteria": _json(cheat_sheet)}, columns))
            continue
        for question in scenario_questions:
            row = {**base, **_question_cells(question)}
            execution = execution_by_question.get(question.question_id)
            if execution is not None:
                row.update(_execution_cells(execution))
                evaluation = evaluation_by_execution.get(execution.execution_id)
                if evaluation is not None:
                    row.update(_evaluation_cells(evaluation, metric_ids))
            rows.append(_complete(row, columns))
    return rows


def to_csv(rows: Sequence[dict[str, str]], columns: Sequence[str] | None = None) -> str:
    fieldnames = list(columns) if columns is not None else (list(rows[0]) if rows else [])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _complete(row: dict[str, str], columns: Sequence[str]) -> dict[str, str]:
    return {column: row.get(column, "") for column in columns}


def _scenario_cells(s: Scenario) -> dict[str, str]:
    return {
        "scenario_id": s.id,
        "mode": s.mode.value,
        "scenario_title": s.title,
        "business_goal": s.business_goal,
        "capability": s.capability.value,
    }


def _pack_view(pack: ReferencePack) -> dict[str, list[str]]:
    items = [
        f"{t.name} ({', '.join(c.name for c in t.columns)})" if t.columns else t.name
        for t in pack.tables
    ] + [d.name for d in pack.documents]
    rules = [
        *(f"join {render_join(j, j.verified)}" for j in pack.joins),
        *(f"{t.name}: {rule}" for t in pack.tables for rule in t.rules),
        *(f"{r.id}: {r.rule}" for r in pack.rules),
    ]
    return {"items": items, "rules": rules}


def _criteria_view(criteria: SuccessCriteria) -> dict[str, Any]:
    """The structured success_criteria as shown: a generated metric expectation shows its value,
    any other its status and reason; the cheat sheet shows its items and rules."""
    view: dict[str, Any] = {GENERAL_CRITERIA_KEY: list(criteria.general)}
    for metric_id, x in criteria.metrics.items():
        view[metric_id] = (
            x.value
            if x.status is MetricExpectationStatus.GENERATED
            else {"status": x.status.value, "reason": x.reason}
        )
    if criteria.cheat_sheet is not None:
        view[CHEAT_SHEET_KEY] = _pack_view(criteria.cheat_sheet)
    return view


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1) if value else ""


def _question_cells(q: QuestionCase) -> dict[str, str]:
    e = q.expectations
    return {
        "question_id": q.question_id,
        "question": q.utterance,
        "expected_behavior": e.expected_behavior,
        "success_criteria": _json(_criteria_view(e.success_criteria)),
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
    tools = ""
    if isinstance(response, AnsweredResponse) and response.tool_trace is not None:
        tools = (
            bullets(
                [
                    f"{c.order}. {c.name}{f' [{c.operation}]' if c.operation else ''}"
                    f"({json.dumps(c.arguments, ensure_ascii=False)})"
                    + (f" error: {c.error}" if c.error else "")
                    for c in response.tool_trace
                ]
            )
            or "(exposed, no calls)"
        )
    context = e.context
    context_cell = context.availability.value
    if context.availability.value == "available":
        context_cell = f"{len(context.chunks)} chunk(s) via {context.source_mode.value}" + (
            f", {len(context.citations)} citation(s)" if context.citations_exposed else ""
        )
    elif context.error:
        context_cell += f": {context.error}"
    return {
        "product_outcome": e.outcome.value,
        "answer_text": text,
        "artifacts": artifacts,
        "tool_calls": tools,
        "retrieved_context": context_cell,
        "product_message": message,
        "latency_ms": f"{latency:.0f}" if latency is not None else "",
    }


def _metric_cells(ev: EvaluationResult, metric_id: str) -> dict[str, str]:
    result = ev.metrics.get(metric_id)
    if result is None:
        return {}
    reason = result.rationale
    if result.status is MetricStatus.SCORED and result.evidence:
        reason = f"{reason} - '{result.evidence}'"
    if result.missing_requirement and result.status is not MetricStatus.SCORED:
        reason = f"{reason} (missing: {result.missing_requirement})"
    return {
        metric_id: f"{result.score:g}" if result.score is not None else "",
        f"{metric_id}_status": result.status.value,
        f"{metric_id}_reason": reason,
    }


def _verdict_reason(ev: EvaluationResult) -> str:
    if ev.verdict_reason is not None:  # D241: decided with its reason
        return ev.verdict_reason
    if ev.verdict is None:
        return NO_SQL_REASON if NO_SQL_REASON in ev.notes else ""
    if ev.findings:
        return bullets(
            [
                f"{f.finding_id} [{f.severity.value}] {f.category}: {f.description}"
                for f in ev.findings
            ]
        )
    if ev.verdict is Verdict.PASS:
        return "All scored metrics passed"
    return ""


def _evaluation_cells(ev: EvaluationResult, metric_ids: Sequence[str]) -> dict[str, str]:
    cells: dict[str, str] = {
        "response_classification": (
            ev.response_classification.value if ev.response_classification else ""
        ),
        "final_verdict": ev.verdict.value if ev.verdict else "recorded_only",
        "verdict_reason": _verdict_reason(ev),
        "bug_ids": bullets([b.bug_id for b in ev.bug_candidates]),
        "evaluation_notes": bullets(list(ev.notes)),
    }
    for metric_id in metric_ids:
        cells.update(_metric_cells(ev, metric_id))
    return cells
