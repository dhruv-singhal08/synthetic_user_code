"""The single orchestrator: index -> scenarios -> questions -> execute -> evaluate.

Run artifacts live in ``<artifacts_dir>/runs/<run_id>/``. Existing artifacts are reused;
a step is regenerated only when asked (from_step / force_steps), when its files are missing,
or when an earlier step was regenerated (regeneration always cascades downstream, and stale
downstream files are deleted). manifest.json is updated after every step. Cross-references
are validated whenever artifacts are loaded.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import Field, TypeAdapter

from synthetic_user import __version__
from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import ArtifactIntegrityError, ArtifactNotFoundError
from synthetic_user.core.identifiers import SequentialIdGenerator, new_run_id
from synthetic_user.llm.base import LLMClient
from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.evaluation import BugCandidate, EvaluationResult
from synthetic_user.models.execution import ExecutionResult
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase, RejectedQuestion
from synthetic_user.models.reference import ReferenceCatalog, ReferenceIndex, ReferenceSnapshot
from synthetic_user.models.run import RunManifest
from synthetic_user.models.scenario import RejectedScenario, Scenario
from synthetic_user.nodes.product_executor import ProductExecutor
from synthetic_user.nodes.question_generator import QuestionGenerator
from synthetic_user.nodes.reference_index import ReferenceIndexGenerator
from synthetic_user.nodes.scenario_generator import ScenarioGenerator
from synthetic_user.nodes.sql_evaluator import EvaluationCase, SqlEvaluator
from synthetic_user.product.base import ProductClient
from synthetic_user.references.reuse import IndexResult, load_or_build_reference_index
from synthetic_user.references.snapshot import build_snapshot, snapshot_id_for
from synthetic_user.storage.json_store import JsonArtifactStore
from synthetic_user.storage.runs import MANIFEST, run_artifact

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]
_EXECUTIONS: TypeAdapter[list[ExecutionResult]] = TypeAdapter(list[ExecutionResult])


class Step(StrEnum):
    INDEX = "index"
    SCENARIOS = "scenarios"
    QUESTIONS = "questions"
    EXECUTE = "execute"
    EVALUATE = "evaluate"


STEP_ORDER: tuple[Step, ...] = tuple(Step)

# Manifest artifact key -> file name, per run step (the index lives outside the run folder).
STEP_ARTIFACTS: dict[Step, dict[str, str]] = {
    Step.INDEX: {},
    Step.SCENARIOS: {
        "scenarios": "scenarios.json",
        "scenarios_rejected": "scenarios_rejected.json",
    },
    Step.QUESTIONS: {
        "snapshots": "snapshots.json",
        "questions": "questions.json",
        "questions_rejected": "questions_rejected.json",
    },
    Step.EXECUTE: {"executions": "executions.json"},
    Step.EVALUATE: {"evaluations": "evaluations.json", "bug_candidates": "bug_candidates.json"},
}


@dataclass(frozen=True)
class RunInputs:
    persona: Persona
    product: ProductDescription
    catalog: ReferenceCatalog


@dataclass(frozen=True)
class ScenarioStep:
    accepted: list[Scenario]
    rejected: list[RejectedScenario]  # includes duplicates (reasons say "duplicate of ...")
    reused: bool


@dataclass(frozen=True)
class QuestionStep:
    questions: list[QuestionCase]
    snapshots: list[ReferenceSnapshot]
    rejected: list[RejectedQuestion]
    reused: bool


@dataclass(frozen=True)
class ExecuteStep:
    executions: list[ExecutionResult]
    reused: bool


@dataclass(frozen=True)
class EvaluateStep:
    evaluations: list[EvaluationResult]
    bug_candidates: list[BugCandidate]
    reused: bool


class RunSummary(StrictModel):
    run_id: NonEmptyStr
    run_dir: NonEmptyStr
    reference_index_id: NonEmptyStr
    steps_run: list[Step] = Field(default_factory=list)
    steps_reused: list[Step] = Field(default_factory=list)
    scenarios_accepted: int
    scenarios_rejected: int
    questions: int
    questions_rejected: int
    outcomes: dict[str, int]
    verdicts: dict[str, int]
    bug_candidates: int


def steps_from(step: Step) -> set[Step]:
    return set(STEP_ORDER[STEP_ORDER.index(step) :])


def validate_run(
    *,
    scenarios: Sequence[Scenario],
    snapshots: Sequence[ReferenceSnapshot] = (),
    questions: Sequence[QuestionCase] = (),
    executions: Sequence[ExecutionResult] = (),
    evaluations: Sequence[EvaluationResult] = (),
) -> None:
    """Raise ArtifactIntegrityError if any cross-reference between run artifacts is broken."""
    problems: list[str] = []
    for kind, ids in (
        ("scenario", [s.id for s in scenarios]),
        ("snapshot", [s.snapshot_id for s in snapshots]),
        ("question", [q.question_id for q in questions]),
        ("execution", [e.execution_id for e in executions]),
        ("evaluation", [e.evaluation_id for e in evaluations]),
    ):
        dupes = sorted(i for i, n in Counter(ids).items() if n > 1)
        if dupes:
            problems.append(f"duplicate {kind} ids {dupes}")

    scenario_ids = {s.id for s in scenarios}
    snapshot_ids = {s.snapshot_id for s in snapshots}
    question_by_id = {q.question_id: q for q in questions}
    execution_by_id = {e.execution_id: e for e in executions}
    for q in questions:
        if q.scenario_id not in scenario_ids:
            problems.append(f"{q.question_id} -> unknown scenario {q.scenario_id}")
        if snapshots and q.reference_snapshot_id not in snapshot_ids:
            problems.append(f"{q.question_id} -> unknown snapshot {q.reference_snapshot_id}")
    for e in executions:
        if e.question_id not in question_by_id:
            problems.append(f"{e.execution_id} -> unknown question {e.question_id}")
    for ev in evaluations:
        question = question_by_id.get(ev.question_id)
        execution = execution_by_id.get(ev.execution_id)
        if question is None:
            problems.append(f"{ev.evaluation_id} -> unknown question {ev.question_id}")
        elif question.scenario_id != ev.scenario_id:
            problems.append(f"{ev.evaluation_id} scenario {ev.scenario_id} != question's")
        if ev.scenario_id not in scenario_ids:
            problems.append(f"{ev.evaluation_id} -> unknown scenario {ev.scenario_id}")
        if execution is None:
            problems.append(f"{ev.evaluation_id} -> unknown execution {ev.execution_id}")
        elif execution.question_id != ev.question_id:
            problems.append(f"{ev.evaluation_id} execution is for {execution.question_id}")
        for bug in ev.bug_candidates:
            chain = (bug.question_id, bug.scenario_id, bug.execution_id)
            if chain != (ev.question_id, ev.scenario_id, ev.execution_id):
                problems.append(f"{bug.bug_id} id chain does not match {ev.evaluation_id}")
    if problems:
        shown = "; ".join(problems[:10])
        more = f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""
        raise ArtifactIntegrityError(f"Run artifacts are inconsistent: {shown}{more}")


class SyntheticUserPipeline:
    def __init__(
        self,
        settings: AppConfig,
        llm: LLMClient,
        product_client: ProductClient,
        store: JsonArtifactStore,
        *,
        logger: Logger,
        id_generators: Callable[[], SequentialIdGenerator] = SequentialIdGenerator,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._product = product_client
        self._store = store
        self._log = logger
        self._ids = id_generators

    # --- steps -------------------------------------------------------------------------

    async def build_reference_index(
        self, catalog: ReferenceCatalog, *, force: bool = False
    ) -> IndexResult:
        generator = ReferenceIndexGenerator(self._llm, self._settings, self._log)
        return await load_or_build_reference_index(
            catalog, self._store, generator.generate, force=force
        )

    async def generate_scenarios(
        self,
        run_id: str,
        inputs: RunInputs,
        index: ReferenceIndex,
        *,
        count: int | None = None,
        force: bool = False,
    ) -> ScenarioStep:
        manifest = self._manifest_or_none(run_id)
        if not force and manifest is not None and self._has(run_id, Step.SCENARIOS):
            self._check_manifest(manifest, inputs, index_id=index.index_id)
            return ScenarioStep(
                accepted=self._load_scenarios(run_id),
                rejected=self._store.read_models(
                    self._file(run_id, "scenarios_rejected"), RejectedScenario
                ),
                reused=True,
            )

        generator = ScenarioGenerator(
            self._llm, self._settings, self._log, id_generator=self._ids()
        )
        result = await generator.generate(inputs.persona, inputs.product, index, count=count)
        rejected = [*result.rejected, *result.duplicates]
        self._store.write_models(self._file(run_id, "scenarios"), result.accepted, Scenario)
        self._store.write_models(
            self._file(run_id, "scenarios_rejected"), rejected, RejectedScenario
        )
        self._save_manifest(
            RunManifest(
                run_id=run_id,
                created_at=manifest.created_at if manifest else datetime.now(UTC),
                persona_id=inputs.persona.id,
                product_id=inputs.product.product_id,
                catalog_id=index.source_catalog_id,
                catalog_version=index.source_catalog_version,
                reference_index_id=index.index_id,
                llm_provider=self._llm.provider_name,
                llm_model=self._llm.model_name,
                pipeline_version=__version__,
                effective_config=self._settings.effective_config(),
                artifacts={},
            ),
            Step.SCENARIOS,
        )
        return ScenarioStep(accepted=result.accepted, rejected=rejected, reused=False)

    async def generate_questions(
        self, run_id: str, inputs: RunInputs, *, count: int | None = None, force: bool = False
    ) -> QuestionStep:
        manifest = self._require_manifest(run_id, inputs)
        scenarios = self._load_scenarios(run_id)
        if not force and self._has(run_id, Step.QUESTIONS):
            questions = self._load_questions(run_id)
            snapshots = self._load_snapshots(run_id, scenarios, inputs.catalog)
            validate_run(scenarios=scenarios, snapshots=snapshots, questions=questions)
            rejected = self._store.read_models(
                self._file(run_id, "questions_rejected"), RejectedQuestion
            )
            return QuestionStep(questions, snapshots, rejected, reused=True)

        generator = QuestionGenerator(
            self._llm, self._settings, self._log, id_generator=self._ids()
        )
        result = await generator.generate(
            inputs.persona, inputs.product, scenarios, inputs.catalog, count=count
        )
        self._store.write_models(
            self._file(run_id, "snapshots"), result.snapshots, ReferenceSnapshot
        )
        self._store.write_models(self._file(run_id, "questions"), result.questions, QuestionCase)
        self._store.write_models(
            self._file(run_id, "questions_rejected"), result.rejected, RejectedQuestion
        )
        self._save_manifest(manifest, Step.QUESTIONS)
        return QuestionStep(result.questions, result.snapshots, result.rejected, reused=False)

    async def execute_questions(
        self, run_id: str, inputs: RunInputs, *, force: bool = False
    ) -> ExecuteStep:
        manifest = self._require_manifest(run_id, inputs)
        scenarios = self._load_scenarios(run_id)
        questions = self._load_questions(run_id)
        validate_run(scenarios=scenarios, questions=questions)
        if not force and self._has(run_id, Step.EXECUTE):
            executions = self._load_executions(run_id)
            validate_run(scenarios=scenarios, questions=questions, executions=executions)
            return ExecuteStep(executions, reused=True)

        executor = ProductExecutor(
            self._product, self._settings, self._log, id_generator=self._ids()
        )
        executions = await executor.execute(questions)
        self._store.write_with(self._file(run_id, "executions"), executions, _EXECUTIONS)
        self._save_manifest(manifest, Step.EXECUTE)
        return ExecuteStep(executions, reused=False)

    async def evaluate_executions(
        self, run_id: str, inputs: RunInputs, *, force: bool = False
    ) -> EvaluateStep:
        manifest = self._require_manifest(run_id, inputs)
        scenarios = self._load_scenarios(run_id)
        questions = self._load_questions(run_id)
        snapshots = self._load_snapshots(run_id, scenarios, inputs.catalog)
        executions = self._load_executions(run_id)
        validate_run(
            scenarios=scenarios, snapshots=snapshots, questions=questions, executions=executions
        )
        if not force and self._has(run_id, Step.EVALUATE):
            evaluations = self._store.read_models(
                self._file(run_id, "evaluations"), EvaluationResult
            )
            validate_run(
                scenarios=scenarios,
                snapshots=snapshots,
                questions=questions,
                executions=executions,
                evaluations=evaluations,
            )
            bugs = [b for e in evaluations for b in e.bug_candidates]
            return EvaluateStep(evaluations, bugs, reused=True)

        scenario_by_id = {s.id: s for s in scenarios}
        question_by_id = {q.question_id: q for q in questions}
        snapshot_by_id = {s.snapshot_id: s for s in snapshots}
        cases = [
            EvaluationCase(
                question_by_id[e.question_id],
                scenario_by_id[question_by_id[e.question_id].scenario_id],
                snapshot_by_id[question_by_id[e.question_id].reference_snapshot_id],
                e,
            )
            for e in executions
        ]
        evaluator = SqlEvaluator(
            self._llm,
            self._settings,
            self._log,
            catalog=inputs.catalog,
            id_generator=self._ids(),
        )
        evaluations = await evaluator.evaluate_many(cases, inputs.product)
        bugs = [b for e in evaluations for b in e.bug_candidates]
        self._store.write_models(self._file(run_id, "evaluations"), evaluations, EvaluationResult)
        self._store.write_models(self._file(run_id, "bug_candidates"), bugs, BugCandidate)
        self._save_manifest(manifest, Step.EVALUATE)
        return EvaluateStep(evaluations, bugs, reused=False)

    # --- full run ----------------------------------------------------------------------

    async def run(
        self,
        inputs: RunInputs,
        *,
        run_id: str | None = None,
        from_step: Step | None = None,
        force_steps: Iterable[Step] = (),
        scenario_count: int | None = None,
        question_count: int | None = None,
    ) -> RunSummary:
        run_id = run_id or new_run_id()
        regenerate: set[Step] = steps_from(from_step) if from_step else set()
        for step in force_steps:
            regenerate |= steps_from(step)
        ran: list[Step] = []
        reused: list[Step] = []

        def note(step: Step, was_reused: bool) -> None:
            (reused if was_reused else ran).append(step)

        index_result = await self.build_reference_index(
            inputs.catalog, force=Step.INDEX in regenerate
        )
        note(Step.INDEX, index_result.reused)
        manifest = self._manifest_or_none(run_id)
        cascade = (
            manifest is not None and manifest.reference_index_id != index_result.index.index_id
        )
        if cascade:
            self._log.info("Reference index changed; regenerating the run from scenarios")

        scenario_step = await self.generate_scenarios(
            run_id,
            inputs,
            index_result.index,
            count=scenario_count,
            force=cascade or Step.SCENARIOS in regenerate,
        )
        note(Step.SCENARIOS, scenario_step.reused)
        cascade = cascade or not scenario_step.reused

        question_step = await self.generate_questions(
            run_id, inputs, count=question_count, force=cascade or Step.QUESTIONS in regenerate
        )
        note(Step.QUESTIONS, question_step.reused)
        cascade = cascade or not question_step.reused

        execute_step = await self.execute_questions(
            run_id, inputs, force=cascade or Step.EXECUTE in regenerate
        )
        note(Step.EXECUTE, execute_step.reused)
        cascade = cascade or not execute_step.reused

        evaluate_step = await self.evaluate_executions(
            run_id, inputs, force=cascade or Step.EVALUATE in regenerate
        )
        note(Step.EVALUATE, evaluate_step.reused)

        verdicts = Counter(
            e.verdict.value if e.verdict else "recorded_only" for e in evaluate_step.evaluations
        )
        outcomes = Counter(e.outcome.value for e in execute_step.executions)
        return RunSummary(
            run_id=run_id,
            run_dir=str(self._store.path_for(run_artifact(run_id, ""))),
            reference_index_id=index_result.index.index_id,
            steps_run=ran,
            steps_reused=reused,
            scenarios_accepted=len(scenario_step.accepted),
            scenarios_rejected=len(scenario_step.rejected),
            questions=len(question_step.questions),
            questions_rejected=len(question_step.rejected),
            outcomes={
                o: outcomes.get(o, 0)
                for o in ("sql_generated", "clarification", "denied", "error", "timeout")
            },
            verdicts={
                v: verdicts.get(v, 0) for v in ("pass", "fail", "inconclusive", "recorded_only")
            },
            bug_candidates=len(evaluate_step.bug_candidates),
        )

    # --- artifact helpers --------------------------------------------------------------

    def _file(self, run_id: str, key: str) -> Path:
        name = next(files[key] for files in STEP_ARTIFACTS.values() if key in files)
        return run_artifact(run_id, name)

    def _has(self, run_id: str, step: Step) -> bool:
        return all(self._store.exists(self._file(run_id, k)) for k in STEP_ARTIFACTS[step])

    def _manifest_or_none(self, run_id: str) -> RunManifest | None:
        try:
            return self._store.read_model(run_artifact(run_id, MANIFEST), RunManifest)
        except ArtifactNotFoundError:
            return None

    def _require_manifest(self, run_id: str, inputs: RunInputs) -> RunManifest:
        manifest = self._store.read_model(run_artifact(run_id, MANIFEST), RunManifest)
        self._check_manifest(manifest, inputs)
        return manifest

    @staticmethod
    def _check_manifest(
        manifest: RunManifest, inputs: RunInputs, index_id: str | None = None
    ) -> None:
        expected = (
            manifest.persona_id,
            manifest.product_id,
            manifest.catalog_id,
            manifest.catalog_version,
        )
        actual = (
            inputs.persona.id,
            inputs.product.product_id,
            inputs.catalog.catalog_id,
            inputs.catalog.catalog_version,
        )
        if expected != actual:
            raise ArtifactIntegrityError(
                f"Configured inputs {actual} do not match run {manifest.run_id} {expected}"
            )
        if index_id is not None and manifest.reference_index_id != index_id:
            raise ArtifactIntegrityError(
                f"Run {manifest.run_id} was built from index {manifest.reference_index_id}, "
                f"not {index_id}; regenerate from the scenarios step"
            )

    def _save_manifest(self, manifest: RunManifest, step: Step) -> None:
        """Record ``step``'s artifacts; drop (and delete) artifacts of all later steps."""
        later = steps_from(step) - {step}
        stale = {key for s in later for key in STEP_ARTIFACTS[s]}
        for key in stale:
            self._store.delete(self._file(manifest.run_id, key))
        artifacts = {k: v for k, v in manifest.artifacts.items() if k not in stale}
        artifacts.update(STEP_ARTIFACTS[step])
        self._store.write_model(
            run_artifact(manifest.run_id, MANIFEST),
            manifest.model_copy(update={"artifacts": artifacts}),
        )

    def _load_scenarios(self, run_id: str) -> list[Scenario]:
        return self._store.read_models(self._file(run_id, "scenarios"), Scenario)

    def _load_questions(self, run_id: str) -> list[QuestionCase]:
        return self._store.read_models(self._file(run_id, "questions"), QuestionCase)

    def _load_executions(self, run_id: str) -> list[ExecutionResult]:
        return self._store.read_with(self._file(run_id, "executions"), _EXECUTIONS)

    def _load_snapshots(
        self, run_id: str, scenarios: Sequence[Scenario], catalog: ReferenceCatalog
    ) -> list[ReferenceSnapshot]:
        """snapshots.json if present, else rebuilt from the catalog (deterministic)."""
        relative = self._file(run_id, "snapshots")
        if self._store.exists(relative):
            return self._store.read_models(relative, ReferenceSnapshot)
        return [build_snapshot(catalog, s.cheat_sheet, snapshot_id_for(s.id)) for s in scenarios]
