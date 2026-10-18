"""Evaluator benchmark: run the evaluator on labeled cases and report agreement.

Cases (tests/fixtures/benchmark/labeled_cases.jsonl) cover both modes (taken from each case's
scenario) and several product types (each case names its product_id). A case states what the
product answered (answered text and/or artifacts such as a query, clarification or denial
message), what the product was expected to do, and the expected verdict. Only answered
responses are judged (design v3 KPIs); clarification / denied cases are reported as not
judged and left out of the agreement. The scenarios file
holds {"scenarios": [...], "reference_packs": [...]} (packs for the grounded scenarios).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, ValidationError, model_validator

from synthetic_user.core.exceptions import InputAssetError, format_validation_error
from synthetic_user.models.base import AmbiguityLevel, NonEmptyStr, StrictModel
from synthetic_user.models.evaluation import Verdict
from synthetic_user.models.execution import ExecutionRecord
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import (
    Expectations,
    QuestionCase,
    QuestionCharacteristics,
    QuestionIntent,
)
from synthetic_user.models.reference import ReferencePack
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator import EvaluationCase, Evaluator

# Benchmark executions come from labeled cases, not from a product adapter.
BENCHMARK_ADAPTER = "benchmark"


DEFAULT_EXPECTED_BEHAVIOR = (
    "Answer the business question correctly, or ask a relevant clarification when the question "
    "is genuinely ambiguous; decline only what the product does not support."
)


class BenchmarkArtifact(StrictModel):
    kind: NonEmptyStr
    content: NonEmptyStr


class BenchmarkCase(StrictModel):
    case_id: NonEmptyStr
    scenario_id: NonEmptyStr
    product_id: NonEmptyStr
    question: NonEmptyStr
    expected_behavior: NonEmptyStr = DEFAULT_EXPECTED_BEHAVIOR
    outcome: Literal["answered", "clarification", "denied"]
    text: str | None = None
    artifacts: list[BenchmarkArtifact] = Field(default_factory=list)
    message: str | None = None
    label: Verdict
    rationale: NonEmptyStr

    @model_validator(mode="after")
    def _payload_matches_outcome(self) -> Self:
        if self.outcome == "answered" and not (self.text or self.artifacts):
            raise ValueError("text and/or artifacts are required for outcome answered")
        if self.outcome != "answered" and not self.message:
            raise ValueError(f"message is required for outcome {self.outcome}")
        return self

    def response(self) -> dict[str, Any]:
        if self.outcome != "answered":
            return {"outcome": self.outcome, "message": self.message}
        artifacts = [a.model_dump() for a in self.artifacts]
        return {"outcome": "answered", "text": self.text, "artifacts": artifacts}


class BenchmarkScenarios(StrictModel):
    scenarios: list[Scenario]
    reference_packs: list[ReferencePack] = Field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkRow:
    case_id: str
    mode: GenerationMode
    product_id: str
    label: Verdict
    verdict: Verdict | None

    @property
    def judged(self) -> bool:
        return self.verdict is not None

    @property
    def agrees(self) -> bool:
        return self.verdict == self.label


@dataclass(frozen=True)
class BenchmarkReport:
    rows: list[BenchmarkRow]

    @property
    def judged(self) -> list[BenchmarkRow]:
        return [r for r in self.rows if r.judged]

    @property
    def agreed(self) -> int:
        return sum(1 for r in self.judged if r.agrees)

    @property
    def agreement(self) -> float:
        """Agreement over judged cases (recorded-only outcomes are not judged)."""
        return self.agreed / len(self.judged) if self.judged else 0.0


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


def load_scenarios(path: Path) -> tuple[list[Scenario], list[ReferencePack]]:
    if not path.is_file():
        raise InputAssetError(f"Benchmark scenarios not found: {path}")
    try:
        data = BenchmarkScenarios.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        raise InputAssetError(f"{path}: invalid scenarios: {format_validation_error(exc)}") from exc
    return data.scenarios, data.reference_packs


def build_evaluation_cases(
    cases: Sequence[BenchmarkCase],
    scenarios: Sequence[Scenario],
    packs: Sequence[ReferencePack] = (),
) -> list[EvaluationCase]:
    """One EvaluationCase per labeled case; grounded scenarios use their reference pack."""
    by_id = {s.id: s for s in scenarios}
    pack_by_id = {p.pack_id: p for p in packs}
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
        pack = None
        if scenario.reference_pack_id is not None:
            pack = pack_by_id.get(scenario.reference_pack_id)
            if pack is None:
                raise InputAssetError(
                    f"{case.case_id}: grounded scenario {scenario.id} has no reference pack "
                    f"{scenario.reference_pack_id} in the scenarios file"
                )
        question = QuestionCase(
            question_id=f"Q_{i:03d}",
            scenario_id=scenario.id,
            persona_id=scenario.persona_id,
            mode=scenario.mode,
            reference_pack_id=scenario.reference_pack_id,
            utterance=case.question,
            intended_goal=scenario.business_goal,
            intent=QuestionIntent(task_type=scenario.task_type),
            question_characteristics=QuestionCharacteristics(ambiguity_level=AmbiguityLevel.MEDIUM),
            expectations=Expectations(
                expected_behavior=case.expected_behavior,
                success_criteria=list(scenario.success_criteria)[:4],
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
        built.append(EvaluationCase(question, scenario, pack, execution))
    return built


async def run_benchmark(
    evaluator: Evaluator,
    cases: Sequence[BenchmarkCase],
    scenarios: Sequence[Scenario],
    packs: Sequence[ReferencePack],
    products: Mapping[str, ProductDescription],
) -> BenchmarkReport:
    """Evaluate every case with its own product description; rows keep the case order."""
    missing = sorted({c.product_id for c in cases} - set(products))
    if missing:
        raise InputAssetError(f"No product description loaded for product_id(s) {missing}")
    evaluation_cases = build_evaluation_cases(cases, scenarios, packs)
    verdicts: dict[str, Verdict | None] = {}
    for product_id, product in products.items():
        selected = [
            (c, e)
            for c, e in zip(cases, evaluation_cases, strict=True)
            if c.product_id == product_id
        ]
        if not selected:
            continue
        results = await evaluator.evaluate_many([e for _, e in selected], product)
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
