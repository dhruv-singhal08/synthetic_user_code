"""The single results table: one row per question, columns filled in as steps complete.

Scenarios without an accepted question still get one row (question columns blank).
"""

from __future__ import annotations

import csv
import io
import json
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
from synthetic_user.models.reference import ReferencePack
from synthetic_user.models.scenario import Scenario

# Only what evaluation uses (design v3); the full detail stays in the JSON artifacts.
SCENARIO_COLUMNS = ["scenario_id", "mode", "scenario_title", "business_goal"]
QUESTION_COLUMNS = [
    "question_id",
    "question",
    "expected_behavior",
    "success_criteria",
    "acceptable_variations",
]
REFERENCE_COLUMNS = ["core_refs", "reference_rules"]
EXECUTION_COLUMNS = [
    "product_outcome",
    "answer_text",
    "artifacts",
    "product_message",
    "latency_ms",
]
EVALUATION_COLUMNS = [
    "final_verdict",
    "criteria_results",
    "failed_checks",
    "inconclusive_checks",
    "findings",
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

_SEP = " | "


def build_rows(
    run_id: str,
    scenarios: Sequence[Scenario],
    questions: Sequence[QuestionCase] = (),
    executions: Sequence[ExecutionRecord] = (),
    evaluations: Sequence[EvaluationResult] = (),
    packs: Sequence[ReferencePack] = (),
) -> list[dict[str, str]]:
    execution_by_question = {e.question_id: e for e in executions}
    pack_by_id = {p.pack_id: p for p in packs}
    evaluation_by_execution = {e.execution_id: e for e in evaluations}
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
    return {
        "core_refs": _SEP.join(r.name for r in pack.core_refs),
        "reference_rules": _SEP.join(r.rule for r in pack.rules),
    }


def _question_cells(q: QuestionCase) -> dict[str, str]:
    e = q.expectations
    return {
        "question_id": q.question_id,
        "question": q.utterance,
        "expected_behavior": e.expected_behavior,
        "success_criteria": _SEP.join(e.success_criteria),
        "acceptable_variations": _SEP.join(e.acceptable_variations),
    }


def _execution_cells(e: ExecutionRecord) -> dict[str, str]:
    latency = e.latency_ms
    response = e.response
    text, message, artifacts = "", "", ""
    if isinstance(response, AnsweredResponse):
        text = response.text or ""
        artifacts = _SEP.join(
            f"{a.kind}: {a.content if isinstance(a.content, str) else json.dumps(a.content)}"
            for a in response.artifacts
        )
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


def _evaluation_cells(ev: EvaluationResult) -> dict[str, str]:
    checks = ev.checks.model_dump() if ev.checks else {}
    return {
        "final_verdict": ev.verdict.value if ev.verdict else "recorded_only",
        "criteria_results": _SEP.join(
            f"{c.status.value}: {c.criterion}" for c in ev.criteria_results
        ),
        "failed_checks": _SEP.join(k for k, v in checks.items() if v == CheckStatus.FAIL),
        "inconclusive_checks": _SEP.join(
            k for k, v in checks.items() if v == CheckStatus.INCONCLUSIVE
        ),
        "findings": _SEP.join(
            f"{f.finding_id} [{f.severity.value}] {f.category}: {f.description}"
            for f in ev.findings
        ),
        "bug_ids": _SEP.join(b.bug_id for b in ev.bug_candidates),
        "evaluation_notes": _SEP.join(ev.notes),
    }
