"""The single results table: one row per question, columns filled in as steps complete.

Scenarios without an accepted question still get one row (question columns blank).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence

from synthetic_user.models.evaluation import CheckStatus, EvaluationResult
from synthetic_user.models.execution import ExecutionResult
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.scenario import Scenario

SCENARIO_COLUMNS = [
    "scenario_id",
    "scenario_title",
    "business_goal",
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
    "interpretation_notes",
    "schema_leak_warnings",
]
EXECUTION_COLUMNS = [
    "execution_id",
    "product_outcome",
    "generated_sql",
    "product_message",
    "latency_ms",
]
EVALUATION_COLUMNS = [
    "evaluation_id",
    "final_verdict",
    "llm_verdict",
    "evaluation_status",
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
    executions: Sequence[ExecutionResult] = (),
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
    cs = s.cheat_sheet
    return {
        "scenario_id": s.id,
        "scenario_title": s.title,
        "business_goal": s.business_goal,
        "task_type": s.task_type.value,
        "capability": s.capability.value,
        "capability_rationale": s.capability_rationale,
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
        "interpretation_notes": _SEP.join(q.interpretation_notes),
        "schema_leak_warnings": _SEP.join(q.schema_leak_warnings),
    }


def _execution_cells(e: ExecutionResult) -> dict[str, str]:
    latency = e.metadata.latency_ms
    return {
        "execution_id": e.execution_id,
        "product_outcome": e.outcome.value,
        "generated_sql": getattr(e, "sql", "") or "",
        "product_message": getattr(e, "message", None) or getattr(e, "error_type", "") or "",
        "latency_ms": f"{latency:.0f}" if latency is not None else "",
    }


def _evaluation_cells(ev: EvaluationResult) -> dict[str, str]:
    checks = ev.checks.model_dump() if ev.checks else {}
    return {
        "evaluation_id": ev.evaluation_id,
        "final_verdict": ev.verdict.value if ev.verdict else "",
        "llm_verdict": ev.llm_verdict.value if ev.llm_verdict else "",
        "evaluation_status": ev.evaluation_status.value,
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
