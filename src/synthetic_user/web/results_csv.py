"""The single results table: one row per question, columns filled in as steps complete.

Scenarios without an accepted question still get one row (question columns blank). Only what
evaluation uses is shown (design v3); the full detail stays in the JSON artifacts.

Cell format (readable in Excel): a cell holding several items lists them as bullet points,
one per line ("• item"); a single item is written as is. The question's Evaluation Context is
three cells: success_criteria ("C1: ..."), reference_spec (the pack id and its requirement ids)
and metric_mapping (JSON); a legacy question shows its stored expectations instead. Each
selected metric has six columns: ``<id>`` (score 0..1, or empty when not scored),
``<id>_status`` (scored / not_applicable / cannot_generate / evidence_unavailable /
inconclusive / evaluation_error / unavailable), ``<id>_reason``, ``<id>_checks`` (one line per
check-level decision with its evidence ids), ``<id>_coverage`` (counts, comparison basis and
verification method) and ``<id>_guidance`` (the guidance applied and how it influenced the
decision). The verdict columns add the failed / unresolved requirements, the lookups and the
evaluation's limitations. Metric columns follow the run's frozen metric selection
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
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import ReferencePack, render_join
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator.no_sql import NO_SQL_REASON

BULLET = "• "

SCENARIO_COLUMNS = ["scenario_id", "mode", "scenario_title", "business_goal", "capability"]
# The question's Evaluation Context (reforming, Part 4): criteria, Reference Spec, Metric Mapping.
QUESTION_COLUMNS = [
    "question_id",
    "question",
    "success_criteria",
    "reference_spec",
    "metric_mapping",
]
EXECUTION_COLUMNS = [
    "product_outcome",
    "product_response_type",  # the product's own type, as returned (D242)
    "response_classification",
    "answer_text",
    "artifacts",
    "tool_calls",
    "retrieved_context",
    "product_message",
    "latency_ms",
]
VERDICT_COLUMNS = [
    "final_verdict",
    "verdict_reason",
    "goal_completion",  # the General Judge's independent goal check (D243)
    "failed_requirements",
    "unresolved_requirements",
    "evidence_lookups",
    "limitations",
    "bug_ids",
    "evaluation_notes",
]
METRIC_SUFFIXES = ("_status", "_reason", "_checks", "_coverage", "_guidance")


def metric_columns(metric_ids: Sequence[str]) -> list[str]:
    return [col for m in metric_ids for col in (m, *(f"{m}{s}" for s in METRIC_SUFFIXES))]


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
            spec = _json(_pack_view(pack)) if pack is not None else ""
            rows.append(_complete({**base, "reference_spec": spec}, columns))
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


def _mapping_view(q: QuestionCase) -> dict[str, Any]:
    """Each metric's mapping as shown: criteria, reference requirements, comparison basis and
    the question-specific expectation (its value when generated, else its status and reason)."""
    view: dict[str, Any] = {}
    for metric_id, m in q.context.metric_mapping.items():
        entry: dict[str, Any] = {
            "criteria": m.criteria,
            "reference_requirements": m.reference_requirements,
            "comparison_basis": m.comparison_basis,
        }
        if (x := m.expectation) is not None:
            entry["expectation"] = (
                x.value if x.value is not None else {"status": x.status.value, "reason": x.reason}
            )
        view[metric_id] = entry
    return view


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1) if value else ""


def _question_cells(q: QuestionCase) -> dict[str, str]:
    if q.evaluation_context is None:  # legacy question: display only
        return {
            "question_id": q.question_id,
            "question": q.utterance,
            "success_criteria": _json({"legacy_expectations": q.legacy_expectations}),
        }
    context = q.evaluation_context
    spec = context.reference_spec
    return {
        "question_id": q.question_id,
        "question": q.utterance,
        "success_criteria": bullets([f"{c.id}: {c.requirement}" for c in context.success_criteria]),
        "reference_spec": _json(
            {
                "pack_id": spec.pack_id,
                "reference_id": spec.reference_id,
                "requirements": [f"{r.id}: {r.text}" for r in spec.requirements()],
                **_pack_view(spec.pack),
            }
        )
        if spec is not None
        else "",
        "metric_mapping": _json(_mapping_view(q)),
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
        "product_response_type": response.product_response_type or "",
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
    coverage = result.coverage
    return {
        metric_id: f"{result.score:g}" if result.score is not None else "",
        f"{metric_id}_status": result.status.value,
        f"{metric_id}_reason": reason,
        f"{metric_id}_checks": bullets(
            [
                f"{c.check_id} {c.requirement_id} {c.status.value}"
                + (f" [{', '.join(c.observed_evidence_ids)}]" if c.observed_evidence_ids else "")
                + f": {c.reason}"
                for c in result.checks
            ]
        ),
        f"{metric_id}_coverage": (
            f"assigned {coverage.assigned}, evaluated {coverage.evaluated}, passed "
            f"{coverage.passed}, failed {coverage.failed}, not applicable "
            f"{coverage.not_applicable}, unresolved {coverage.unresolved}, derived "
            f"{coverage.derived}; basis {result.comparison_basis}; method "
            f"{result.verification_method}"
            if coverage is not None
            else ""
        ),
        f"{metric_id}_guidance": bullets(
            [f"{g.guidance_id} ({g.guidance_type}): {g.influence}" for g in result.guidance_applied]
        ),
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
        "goal_completion": (
            f"{ev.goal_completion.status.value}: {ev.goal_completion.reason}"
            if ev.goal_completion is not None
            else ""
        ),
        "failed_requirements": bullets(ev.failed_requirements),
        "unresolved_requirements": bullets(ev.unresolved_requirements),
        "evidence_lookups": _lookups_cell(ev),
        "limitations": bullets(ev.limitations),
        "bug_ids": bullets([b.bug_id for b in ev.bug_candidates]),
        "evaluation_notes": bullets(list(ev.notes)),
    }
    for metric_id in metric_ids:
        cells.update(_metric_cells(ev, metric_id))
    return cells


def _lookups_cell(ev: EvaluationResult) -> str:
    record = ev.reference_checks
    if record is None:
        return ""
    lines = [
        f"{e.evidence_id or '-'} round {e.round} {e.request.tool.value}: {e.status}"
        + (" (used)" if e.used_in_judgment else "")
        for e in record.lookup_trace
    ]
    head = f"additional_lookup_performed = {str(bool(record.additional_lookup_performed)).lower()}"
    return bullets([head, *lines])
