"""Evaluator benchmark: run the evaluator on human-labeled cases and report agreement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import TypeAdapter, ValidationError, model_validator

from synthetic_user.core.exceptions import InputAssetError, format_validation_error
from synthetic_user.models.base import AmbiguityLevel, NonEmptyStr, StrictModel
from synthetic_user.models.evaluation import Verdict
from synthetic_user.models.execution import EXECUTION_RESULT_ADAPTER
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import (
    ClarificationExpectation,
    Expectations,
    QuestionCase,
    QuestionCharacteristics,
    QuestionIntent,
)
from synthetic_user.models.reference import ReferenceCatalog
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.sql_evaluator import EvaluationCase, SqlEvaluator
from synthetic_user.references.snapshot import build_snapshot, snapshot_id_for


class BenchmarkCase(StrictModel):
    case_id: NonEmptyStr
    scenario_id: NonEmptyStr
    question: NonEmptyStr
    clarification_expectation: ClarificationExpectation = ClarificationExpectation.CONDITIONAL
    outcome: Literal["sql_generated", "clarification"]
    sql: str | None = None
    message: str | None = None
    label: Verdict
    rationale: NonEmptyStr

    @model_validator(mode="after")
    def _payload_matches_outcome(self) -> Self:
        if self.outcome == "sql_generated" and not self.sql:
            raise ValueError("sql is required for outcome sql_generated")
        if self.outcome == "clarification" and not self.message:
            raise ValueError("message is required for outcome clarification")
        return self


@dataclass(frozen=True)
class BenchmarkRow:
    case_id: str
    label: Verdict
    verdict: Verdict | None

    @property
    def agrees(self) -> bool:
        return self.verdict == self.label


@dataclass(frozen=True)
class BenchmarkReport:
    rows: list[BenchmarkRow]

    @property
    def agreed(self) -> int:
        return sum(1 for r in self.rows if r.agrees)

    @property
    def agreement(self) -> float:
        return self.agreed / len(self.rows) if self.rows else 0.0


def load_cases(path: Path) -> list[BenchmarkCase]:
    if not path.is_file():
        raise InputAssetError(f"Benchmark cases not found: {path}")
    cases: list[BenchmarkCase] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(BenchmarkCase.model_validate_json(line))
        except ValidationError as exc:
            raise InputAssetError(
                f"{path}:{number}: invalid benchmark case: {format_validation_error(exc)}"
            ) from exc
    return cases


def load_scenarios(path: Path) -> list[Scenario]:
    if not path.is_file():
        raise InputAssetError(f"Benchmark scenarios not found: {path}")
    try:
        return TypeAdapter(list[Scenario]).validate_json(path.read_bytes())
    except ValidationError as exc:
        raise InputAssetError(f"{path}: invalid scenarios: {format_validation_error(exc)}") from exc


def build_evaluation_cases(
    cases: Sequence[BenchmarkCase],
    scenarios: Sequence[Scenario],
    catalog: ReferenceCatalog,
    product: ProductDescription,
) -> list[EvaluationCase]:
    by_id = {s.id: s for s in scenarios}
    started = datetime.now(UTC)
    built: list[EvaluationCase] = []
    for i, case in enumerate(cases, start=1):
        scenario = by_id.get(case.scenario_id)
        if scenario is None:
            raise InputAssetError(f"{case.case_id}: unknown scenario_id {case.scenario_id!r}")
        question = QuestionCase(
            question_id=f"Q_{i:03d}",
            scenario_id=scenario.id,
            persona_id=scenario.persona_id,
            reference_snapshot_id=snapshot_id_for(scenario.id),
            utterance=case.question,
            intended_goal=scenario.business_goal,
            intent=QuestionIntent(task_type=scenario.task_type),
            question_characteristics=QuestionCharacteristics(ambiguity_level=AmbiguityLevel.MEDIUM),
            expectations=Expectations(
                clarification=case.clarification_expectation,
                acceptable_behaviors=["answers the business question correctly"],
            ),
        )
        payload = (
            {"sql": case.sql} if case.outcome == "sql_generated" else {"message": case.message}
        )
        execution = EXECUTION_RESULT_ADAPTER.validate_python(
            {
                "execution_id": f"EXE_{i:03d}",
                "question_id": question.question_id,
                "product_id": product.product_id,
                "submitted_question": case.question,
                "started_at": started,
                "outcome": case.outcome,
                **payload,
            }
        )
        snapshot = build_snapshot(catalog, scenario.cheat_sheet, snapshot_id_for(scenario.id))
        built.append(EvaluationCase(question, scenario, snapshot, execution))
    return built


async def run_benchmark(
    evaluator: SqlEvaluator,
    cases: Sequence[BenchmarkCase],
    scenarios: Sequence[Scenario],
    catalog: ReferenceCatalog,
    product: ProductDescription,
) -> BenchmarkReport:
    evaluation_cases = build_evaluation_cases(cases, scenarios, catalog, product)
    results = await evaluator.evaluate_many(evaluation_cases, product)
    return BenchmarkReport(
        rows=[
            BenchmarkRow(case_id=c.case_id, label=c.label, verdict=r.verdict)
            for c, r in zip(cases, results, strict=True)
        ]
    )
