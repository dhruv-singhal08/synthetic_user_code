"""Evaluator benchmark: run the evaluator on labeled cases and report agreement.

Cases (tests/fixtures/benchmark/labeled_cases.jsonl) cover both modes (taken from each case's
scenario) and several product types (each case names its product_id). A case states what the
product answered (answered text and/or SQL, clarification or denial message) and the expected
verdict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import TypeAdapter, ValidationError, model_validator

from synthetic_user.core.exceptions import InputAssetError, format_validation_error
from synthetic_user.models.base import AmbiguityLevel, NonEmptyStr, StrictModel
from synthetic_user.models.evaluation import Verdict
from synthetic_user.models.execution import ExecutionRecord
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import (
    ClarificationExpectation,
    Expectations,
    QuestionCase,
    QuestionCharacteristics,
    QuestionIntent,
)
from synthetic_user.models.reference import ReferenceIndex
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator import EvaluationCase, Evaluator
from synthetic_user.references.snapshot import build_snapshot, snapshot_id_for

# Benchmark executions come from labeled cases, not from a product adapter.
BENCHMARK_ADAPTER = "benchmark"


class BenchmarkCase(StrictModel):
    case_id: NonEmptyStr
    scenario_id: NonEmptyStr
    product_id: NonEmptyStr
    question: NonEmptyStr
    clarification_expectation: ClarificationExpectation = ClarificationExpectation.CONDITIONAL
    outcome: Literal["answered", "clarification", "denied"]
    text: str | None = None
    sql: str | None = None
    message: str | None = None
    label: Verdict
    rationale: NonEmptyStr

    @model_validator(mode="after")
    def _payload_matches_outcome(self) -> Self:
        if self.outcome == "answered" and not (self.text or self.sql):
            raise ValueError("text and/or sql is required for outcome answered")
        if self.outcome != "answered" and not self.message:
            raise ValueError(f"message is required for outcome {self.outcome}")
        return self

    def response(self) -> dict[str, Any]:
        if self.outcome != "answered":
            return {"outcome": self.outcome, "message": self.message}
        artifacts = [{"kind": "sql", "content": self.sql}] if self.sql else []
        return {"outcome": "answered", "text": self.text, "artifacts": artifacts}


@dataclass(frozen=True)
class BenchmarkRow:
    case_id: str
    mode: GenerationMode
    product_id: str
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
    index: ReferenceIndex | None,
) -> list[EvaluationCase]:
    """One EvaluationCase per labeled case; grounded scenarios get a snapshot from ``index``."""
    by_id = {s.id: s for s in scenarios}
    started = datetime.now(UTC)
    built: list[EvaluationCase] = []
    for i, case in enumerate(cases, start=1):
        scenario = by_id.get(case.scenario_id)
        if scenario is None:
            raise InputAssetError(f"{case.case_id}: unknown scenario_id {case.scenario_id!r}")
        if scenario.product_id != case.product_id:
            raise InputAssetError(
                f"{case.case_id}: product_id {case.product_id!r} differs from its scenario's "
                f"{scenario.product_id!r}"
            )
        snapshot = None
        if scenario.reference_cheat_sheet is not None:
            if index is None:
                raise InputAssetError(f"{case.case_id}: grounded case needs a reference index")
            snapshot = build_snapshot(
                index, scenario.reference_cheat_sheet, snapshot_id_for(scenario.id)
            )
        question = QuestionCase(
            question_id=f"Q_{i:03d}",
            scenario_id=scenario.id,
            persona_id=scenario.persona_id,
            mode=scenario.mode,
            reference_snapshot_id=snapshot.snapshot_id if snapshot else None,
            utterance=case.question,
            intended_goal=scenario.business_goal,
            intent=QuestionIntent(task_type=scenario.task_type),
            question_characteristics=QuestionCharacteristics(ambiguity_level=AmbiguityLevel.MEDIUM),
            expectations=Expectations(
                clarification=case.clarification_expectation,
                acceptable_behaviors=["answers the business question correctly"],
                success_criteria=list(scenario.success_criteria),
            ),
        )
        execution = ExecutionRecord.model_validate(
            {
                "execution_id": f"EXE_{i:03d}",
                "question_id": question.question_id,
                "product_id": case.product_id,
                "adapter": BENCHMARK_ADAPTER,
                "submitted_question": case.question,
                "started_at": started,
                "response": case.response(),
            }
        )
        built.append(EvaluationCase(question, scenario, snapshot, execution))
    return built


async def run_benchmark(
    evaluator: Evaluator,
    cases: Sequence[BenchmarkCase],
    scenarios: Sequence[Scenario],
    index: ReferenceIndex | None,
    products: Mapping[str, ProductDescription],
    persona: Persona,
) -> BenchmarkReport:
    """Evaluate every case with its own product description; rows keep the case order."""
    missing = sorted({c.product_id for c in cases} - set(products))
    if missing:
        raise InputAssetError(f"No product description loaded for product_id(s) {missing}")
    evaluation_cases = build_evaluation_cases(cases, scenarios, index)
    verdicts: dict[str, Verdict | None] = {}
    for product_id, product in products.items():
        selected = [
            (c, e)
            for c, e in zip(cases, evaluation_cases, strict=True)
            if c.product_id == product_id
        ]
        if not selected:
            continue
        results = await evaluator.evaluate_many([e for _, e in selected], product, persona)
        for (case, _), result in zip(selected, results, strict=True):
            verdicts[case.case_id] = result.verdict
    return BenchmarkReport(
        rows=[
            BenchmarkRow(
                case_id=c.case_id,
                mode=e.question.mode,
                product_id=c.product_id,
                label=c.label,
                verdict=verdicts[c.case_id],
            )
            for c, e in zip(cases, evaluation_cases, strict=True)
        ]
    )
