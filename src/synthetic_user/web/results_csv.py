"""The single results table: one row per question, columns filled in as steps complete.

Scenarios without an accepted question still get one row (question columns blank).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence

from synthetic_user.models.evaluation import CheckStatus, EvaluationResult
from synthetic_user.models.execution import (
    AnsweredResponse,
    ClarificationResponse,
    DeniedResponse,
    ErrorResponse,
    ExecutionRecord,
)
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.scenario import Scenario

SCENARIO_COLUMNS = [
    "scenario_id",
    "mode",
    "scenario_title",
    "business_goal",
    "success_criteria",
    "task_type",
    "capability",
    "capability_rationale",
    "cheat_tables",
    "cheat_columns",
    "cheat_joins",
    "cheat_metrics",
    "cheat_dimensions",
    "cheat_operations",
    "cheat_filters",
    "cheat_semantic_considerations",
]
QUESTION_COLUMNS = [
    "question_id",
    "question",
    "question_style",
    "ambiguity_level",
    "clarification_expected",
    "acceptable_behaviors",
    "unacceptable_behaviors",
    "question_success_criteria",
    "interpretation_notes",
    "schema_leak_warnings",
]
EXECUTION_COLUMNS = [
    "execution_id",
    "adapter",
    "product_outcome",
    "answer_text",
    "artifact_kinds",
    "generated_sql",
    "product_message",
    "latency_ms",
]
EVALUATION_COLUMNS = [
    "evaluation_id",
    "final_verdict",
    "llm_verdict",
    "evaluation_status",
    "plugins_applied",
    "criteria_results",
    "failed_checks",
    "inconclusive_checks",
    "findings",
    "bug_ids",
    "bug_titles",
    "evaluation_notes",
]
COLUMNS = ["run_id", *SCENARIO_COLUMNS, *QUESTION_COLUMNS, *EXECUTION_COLUMNS, *EVALUATION_COLUMNS]

_SEP = " | "


def build_rows(
    run_id: str,
    scenarios: Sequence[Scenario],
    questions: Sequence[QuestionCase] = (),
    executions: Sequence[ExecutionRecord] = (),
    evaluations: Sequence[EvaluationResult] = (),
) -> list[dict[str, str]]:
    execution_by_question = {e.question_id: e for e in executions}
    evaluation_by_execution = {e.execution_id: e for e in evaluations}
    rows: list[dict[str, str]] = []
    for scenario in scenarios:
        base = {"run_id": run_id, **_scenario_cells(scenario)}
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
    cells = {
        "scenario_id": s.id,
        "mode": s.mode.value,
        "scenario_title": s.title,
        "business_goal": s.business_goal,
        "success_criteria": _SEP.join(s.success_criteria),
        "task_type": s.task_type.value,
        "capability": s.capability.value,
        "capability_rationale": s.capability_rationale,
    }
    cs = s.reference_cheat_sheet
    if cs is None:
        return cells
    return {
        **cells,
        "cheat_tables": _SEP.join(cs.tables),
        "cheat_columns": _SEP.join(cs.columns),
        "cheat_joins": _SEP.join(
            f"{r.source} = {r.target} ({r.verification.value})" for r in cs.relationships
        ),
        "cheat_metrics": _SEP.join(f"{m.name}: {m.definition}" for m in cs.metrics),
        "cheat_dimensions": _SEP.join(cs.dimensions),
        "cheat_operations": _SEP.join(cs.operations),
        "cheat_filters": _SEP.join(cs.optional_filters),
        "cheat_semantic_considerations": _SEP.join(cs.semantic_considerations),
    }


def _question_cells(q: QuestionCase) -> dict[str, str]:
    return {
        "question_id": q.question_id,
        "question": q.utterance,
        "question_style": q.question_characteristics.style.value,
        "ambiguity_level": q.question_characteristics.ambiguity_level.value,
        "clarification_expected": q.expectations.clarification.value,
        "acceptable_behaviors": _SEP.join(q.expectations.acceptable_behaviors),
        "unacceptable_behaviors": _SEP.join(q.expectations.unacceptable_behaviors),
        "question_success_criteria": _SEP.join(q.expectations.success_criteria),
        "interpretation_notes": _SEP.join(q.interpretation_notes),
        "schema_leak_warnings": _SEP.join(q.schema_leak_warnings),
    }


def _execution_cells(e: ExecutionRecord) -> dict[str, str]:
    latency = e.latency_ms
    response = e.response
    text, message, sql = "", "", ""
    if isinstance(response, AnsweredResponse):
        text = response.text or ""
        artifact = response.first_artifact("sql")
        sql = artifact.content if artifact and isinstance(artifact.content, str) else ""
    elif isinstance(response, ClarificationResponse | DeniedResponse):
        message = response.message
    elif isinstance(response, ErrorResponse):
        message = response.message or response.error_type
    return {
        "execution_id": e.execution_id,
        "adapter": e.adapter,
        "product_outcome": e.outcome.value,
        "answer_text": text,
        "artifact_kinds": _SEP.join(a.kind for a in getattr(response, "artifacts", [])),
        "generated_sql": sql,
        "product_message": message,
        "latency_ms": f"{latency:.0f}" if latency is not None else "",
    }


def _evaluation_cells(ev: EvaluationResult) -> dict[str, str]:
    checks = {**(ev.checks.model_dump() if ev.checks else {}), **ev.grounded_checks}
    return {
        "evaluation_id": ev.evaluation_id,
        "final_verdict": ev.verdict.value if ev.verdict else "",
        "llm_verdict": ev.llm_verdict.value if ev.llm_verdict else "",
        "evaluation_status": ev.evaluation_status.value,
        "plugins_applied": _SEP.join(ev.plugins_applied),
        "criteria_results": _SEP.join(
            f"{c.status.value}: {c.criterion}" for c in ev.criteria_results
        ),
        "failed_checks": _SEP.join(k for k, v in checks.items() if v == CheckStatus.FAIL),
        "inconclusive_checks": _SEP.join(
            k for k, v in checks.items() if v == CheckStatus.INCONCLUSIVE
        ),
        "findings": _SEP.join(
            f"{f.finding_id} [{f.severity.value}/{f.source.value}] {f.category}: {f.description}"
            for f in ev.findings
        ),
        "bug_ids": _SEP.join(b.bug_id for b in ev.bug_candidates),
        "bug_titles": _SEP.join(b.title for b in ev.bug_candidates),
        "evaluation_notes": _SEP.join([*ev.notes, *ev.cheat_sheet_conflicts]),
    }
