"""The single orchestrator: reference -> scenarios -> questions -> execute -> evaluate.

The "reference" step only checks the user's reference file + usage guide (never generated
here); without one the run is ungrounded. Grounded NL2SQL scenarios write reference_packs.json;
Agent questions carry their answer keys as expected behaviour and success criteria instead.
When a golden set is present, scenarios and questions are deterministic projections of its rows;
their canonical tool/source definitions were validated against Reference data during setup.
A person may edit an NL2SQL pack before questions, invalidating the later steps.

Run artifacts live in ``<artifacts_dir>/runs/<run_id>/``. Existing artifacts are reused;
a step is regenerated only when asked (from_step / force_steps), when its files are missing,
or when an earlier step was regenerated (regeneration always cascades downstream, and stale
downstream files are deleted). manifest.json is updated after every step. Cross-references
are validated whenever artifacts are loaded.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import Field, JsonValue

from synthetic_user import __version__
from synthetic_user.adapters.base import ProductAdapter
from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
)
from synthetic_user.core.identifiers import IdKind, SequentialIdGenerator, new_run_id
from synthetic_user.core.trace import RUN_LOG, trace_step
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.tracing import TracingLLMClient
from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.evaluation import BugCandidate, EvaluationResult
from synthetic_user.models.evaluation_profile import (
    AgentMetric,
    EvaluationProfile,
    EvaluationProfileSelection,
    resolve_evaluation_profile,
)
from synthetic_user.models.execution import ExecutionRecord, ProductOutcome
from synthetic_user.models.golden_set import GoldenSet
from synthetic_user.models.kpi import RunKpis
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase, RejectedQuestion
from synthetic_user.models.reference import Reference, ReferencePack, ReferencePackDraft
from synthetic_user.models.run import GenerationMode, RunManifest
from synthetic_user.models.scenario import RejectedScenario, Scenario
from synthetic_user.nodes.evaluator import EvaluationCase, Evaluator
from synthetic_user.nodes.evaluator.kpis import compute_kpis, kpis_csv
from synthetic_user.nodes.executor.executor import ProductExecutor
from synthetic_user.nodes.question.generator import QuestionGenerator
from synthetic_user.nodes.scenario.generator import ScenarioGenerator
from synthetic_user.references.pack import edit_pack
from synthetic_user.storage.json_store import JsonArtifactStore
from synthetic_user.storage.runs import MANIFEST, run_artifact

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]


class Step(StrEnum):
    REFERENCE = "reference"
    SCENARIOS = "scenarios"
    QUESTIONS = "questions"
    EXECUTE = "execute"
    EVALUATE = "evaluate"


STEP_ORDER: tuple[Step, ...] = tuple(Step)

# Manifest artifact key -> file name, per run step (the reference is an input, not an artifact).
STEP_ARTIFACTS: dict[Step, dict[str, str]] = {
    Step.REFERENCE: {},
    Step.SCENARIOS: {
        "scenarios": "scenarios.json",
        "scenarios_rejected": "scenarios_rejected.json",
        "reference_packs": "reference_packs.json",
    },
    Step.QUESTIONS: {
        "questions": "questions.json",
        "questions_rejected": "questions_rejected.json",
    },
    Step.EXECUTE: {"executions": "executions.json"},
    Step.EVALUATE: {
        "evaluations": "evaluations.json",
        "bug_candidates": "bug_candidates.json",
        "kpis": "kpis.json",
        "kpis_csv": "kpis.csv",
    },
}
# Raw product responses (when the adapter supplies them): runs/<run_id>/raw/<execution_id>.json
RAW_DIR = "raw"
# Artifact integrity problems quoted in one error message.
MAX_PROBLEMS_SHOWN = 10
# Written only for grounded NL2SQL scenarios.
OPTIONAL_ARTIFACTS = frozenset({"reference_packs"})


@dataclass(frozen=True)
class RunInputs:
    persona: Persona
    product: ProductDescription
    # The user's reference file (+ usage guide); None -> ungrounded run.
    reference: Reference | None = None
    # SME-authored Agent examples used with Reference data to guide new scenario/question synthesis.
    golden_set: GoldenSet | None = None
    evaluation_profile: EvaluationProfile | None = None

    @property
    def mode(self) -> GenerationMode:
        return GenerationMode.GROUNDED if self.reference else GenerationMode.UNGROUNDED

    @property
    def reference_id(self) -> str | None:
        return self.reference.reference_id if self.reference else None

    @property
    def profile(self) -> EvaluationProfile:
        return (
            self.evaluation_profile
            or resolve_evaluation_profile(self.product, EvaluationProfileSelection.AUTO)[0]
        )


@dataclass(frozen=True)
class ReferenceStep:
    mode: GenerationMode
    reference: Reference | None
    warnings: list[str]


@dataclass(frozen=True)
class ScenarioStep:
    accepted: list[Scenario]
    rejected: list[RejectedScenario]  # includes duplicates (reasons say "duplicate of ...")
    packs: list[ReferencePack]
    reused: bool


@dataclass(frozen=True)
class QuestionStep:
    questions: list[QuestionCase]
    rejected: list[RejectedQuestion]
    reused: bool


@dataclass(frozen=True)
class ExecuteStep:
    executions: list[ExecutionRecord]
    reused: bool


@dataclass(frozen=True)
class EvaluateStep:
    evaluations: list[EvaluationResult]
    bug_candidates: list[BugCandidate]
    kpis: RunKpis
    reused: bool


class RunSummary(StrictModel):
    run_id: NonEmptyStr
    run_dir: NonEmptyStr
    mode: GenerationMode
    reference_id: NonEmptyStr | None
    steps_run: list[Step] = Field(default_factory=list)
    steps_reused: list[Step] = Field(default_factory=list)
    scenarios_accepted: int
    scenarios_rejected: int
    questions: int
    questions_rejected: int
    outcomes: dict[str, int]
    verdicts: dict[str, int]
    bug_candidates: int
    kpis: RunKpis


def steps_from(step: Step) -> set[Step]:
    return set(STEP_ORDER[STEP_ORDER.index(step) :])


def validate_run(
    *,
    scenarios: Sequence[Scenario],
    packs: Sequence[ReferencePack] = (),
    questions: Sequence[QuestionCase] = (),
    executions: Sequence[ExecutionRecord] = (),
    evaluations: Sequence[EvaluationResult] = (),
) -> None:
    """Raise ArtifactIntegrityError if any cross-reference between run artifacts is broken."""
    problems: list[str] = []
    for kind, ids in (
        ("scenario", [s.id for s in scenarios]),
        ("reference pack", [p.pack_id for p in packs]),
        ("question", [q.question_id for q in questions]),
        ("execution", [e.execution_id for e in executions]),
        ("evaluation", [e.evaluation_id for e in evaluations]),
    ):
        dupes = sorted(i for i, n in Counter(ids).items() if n > 1)
        if dupes:
            problems.append(f"duplicate {kind} ids {dupes}")

    scenario_by_id = {s.id: s for s in scenarios}
    scenario_ids = set(scenario_by_id)
    pack_ids = {p.pack_id for p in packs}
    question_by_id = {q.question_id: q for q in questions}
    execution_by_id = {e.execution_id: e for e in executions}
    for scenario in scenarios if packs else ():
        if scenario.reference_pack_id is not None and scenario.reference_pack_id not in pack_ids:
            problems.append(f"{scenario.id} -> unknown reference pack {scenario.reference_pack_id}")
    for pack in packs:
        if pack.scenario_id not in scenario_ids:
            problems.append(f"{pack.pack_id} -> unknown scenario {pack.scenario_id}")
    for q in questions:
        owner = scenario_by_id.get(q.scenario_id)
        if owner is None:
            problems.append(f"{q.question_id} -> unknown scenario {q.scenario_id}")
        elif q.evaluation_profile is not owner.evaluation_profile:
            problems.append(
                f"{q.question_id} profile {q.evaluation_profile} != scenario's "
                f"{owner.evaluation_profile}"
            )
        elif q.reference_pack_id != owner.reference_pack_id:
            problems.append(
                f"{q.question_id} uses reference pack {q.reference_pack_id}, its scenario "
                f"{owner.reference_pack_id}"
            )
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
        shown = "; ".join(problems[:MAX_PROBLEMS_SHOWN])
        more = (
            f" (+{len(problems) - MAX_PROBLEMS_SHOWN} more)"
            if len(problems) > MAX_PROBLEMS_SHOWN
            else ""
        )
        raise ArtifactIntegrityError(f"Run artifacts are inconsistent: {shown}{more}")


class SyntheticUserPipeline:
    def __init__(
        self,
        settings: AppConfig,
        llm: LLMClient,
        product_adapter: ProductAdapter,
        store: JsonArtifactStore,
        *,
        logger: Logger,
        id_generators: Callable[[], SequentialIdGenerator] = SequentialIdGenerator,
    ) -> None:
        self._settings = settings
        self._llm = TracingLLMClient.wrap(llm)
        self._product = product_adapter
        self._store = store
        self._log = logger
        self._ids = id_generators

    async def aclose(self) -> None:
        """Close the product adapter (callers that create a pipeline call this when done)."""
        await self._product.aclose()

    # --- steps -------------------------------------------------------------------------

    def check_reference(self, inputs: RunInputs) -> ReferenceStep:
        """Report the reference in use (loaded and size-checked already). No LLM."""
        reference = inputs.reference
        warnings: list[str] = []
        if reference is None:
            self._log.info("No reference: mode ungrounded")
        else:
            self._log.info(
                "Reference %s (%s, %d chars): mode grounded",
                reference.reference_id,
                reference.kind.value,
                reference.rendered_chars(),
            )
            if reference.instructions is None:
                warnings.append(
                    f"no usage instructions: the default instructions for a "
                    f"{reference.kind.value} are used"
                )
        for warning in warnings:
            self._log.warning("Reference warning: %s", warning)
        return ReferenceStep(inputs.mode, reference, warnings)

    async def generate_scenarios(
        self,
        run_id: str,
        inputs: RunInputs,
        *,
        count: int | None = None,
        force: bool = False,
    ) -> ScenarioStep:
        manifest = self._manifest_or_none(run_id)
        if not force and manifest is not None and self._has(run_id, Step.SCENARIOS):
            self._check_manifest(manifest, inputs)
            return ScenarioStep(
                accepted=self._load_scenarios(run_id),
                rejected=self._store.read_models(
                    self._file(run_id, "scenarios_rejected"), RejectedScenario
                ),
                packs=self._load_packs(run_id),
                reused=True,
            )

        if inputs.golden_set is not None and inputs.reference is None:
            raise ArtifactIntegrityError("Golden-guided generation requires Reference data")
        generator = ScenarioGenerator(
            self._llm, self._settings, self._log, id_generator=self._ids()
        )
        with self._trace(run_id, Step.SCENARIOS):
            result = await generator.generate(
                inputs.persona,
                inputs.product,
                inputs.reference,
                count=count,
                evaluation_profile=inputs.profile,
                golden_set=inputs.golden_set,
            )
        rejected = [*result.rejected, *result.duplicates]
        self._store.write_models(self._file(run_id, "scenarios"), result.accepted, Scenario)
        self._store.write_models(
            self._file(run_id, "scenarios_rejected"), rejected, RejectedScenario
        )
        skip: set[str] = set()
        if result.packs:
            self._store.write_models(
                self._file(run_id, "reference_packs"), result.packs, ReferencePack
            )
        else:  # Agent and ungrounded runs do not have schema reference packs.
            self._store.delete(self._file(run_id, "reference_packs"))
            skip.add("reference_packs")
        self._save_manifest(
            RunManifest(
                run_id=run_id,
                created_at=manifest.created_at if manifest else datetime.now(UTC),
                persona_id=inputs.persona.id,
                product_id=inputs.product.product_id,
                mode=inputs.mode,
                reference_id=inputs.reference_id,
                evaluation_profile=inputs.profile,
                llm_provider=self._llm.provider_name,
                llm_model=self._llm.model_name,
                pipeline_version=__version__,
                effective_config=self._settings.effective_config(),
                artifacts={},
            ),
            Step.SCENARIOS,
            skip=skip,
        )
        return ScenarioStep(
            accepted=result.accepted, rejected=rejected, packs=result.packs, reused=False
        )

    def update_reference_pack(
        self, run_id: str, inputs: RunInputs, pack_id: str, draft: ReferencePackDraft
    ) -> ReferencePack:
        """A person's edit of one pack: quotes re-checked against the reference (InputAssetError
        if any is missing; nothing saved). Saved edits invalidate questions and later steps."""
        manifest = self._require_manifest(run_id, inputs)
        if inputs.reference is None:
            raise ArtifactIntegrityError(f"Run {run_id} is ungrounded: it has no reference packs")
        packs = self._load_packs(run_id)
        current = next((p for p in packs if p.pack_id == pack_id), None)
        if current is None:
            raise ArtifactNotFoundError(f"No reference pack {pack_id} in run {run_id}")
        updated = edit_pack(current, draft, inputs.reference)
        packs = [updated if p.pack_id == pack_id else p for p in packs]
        self._store.write_models(self._file(run_id, "reference_packs"), packs, ReferencePack)
        self._save_manifest(manifest, Step.SCENARIOS)
        self._log.info("Reference pack %s edited; later steps must be run again", pack_id)
        return updated

    async def generate_questions(
        self, run_id: str, inputs: RunInputs, *, count: int | None = None, force: bool = False
    ) -> QuestionStep:
        manifest = self._require_manifest(run_id, inputs)
        scenarios = self._load_scenarios(run_id)
        packs = self._load_packs(run_id)
        validate_run(scenarios=scenarios, packs=packs)
        if not force and self._has(run_id, Step.QUESTIONS):
            questions = self._load_questions(run_id)
            validate_run(scenarios=scenarios, packs=packs, questions=questions)
            rejected = self._store.read_models(
                self._file(run_id, "questions_rejected"), RejectedQuestion
            )
            return QuestionStep(questions, rejected, reused=True)

        golden_question_limit = (
            self._settings.generation.question.agent_golden_max_questions_per_scenario
        )
        if inputs.golden_set is not None and count is not None and count > golden_question_limit:
            raise ArtifactIntegrityError(
                "Golden-guided runs accept at most "
                f"{golden_question_limit} new questions per scenario"
            )
        resolved_count = (
            golden_question_limit if inputs.golden_set is not None and count is None else count
        )
        generator = QuestionGenerator(
            self._llm, self._settings, self._log, id_generator=self._ids()
        )
        with self._trace(run_id, Step.QUESTIONS):
            result = await generator.generate(
                inputs.persona,
                inputs.product,
                scenarios,
                inputs.reference,
                packs,
                count=resolved_count,
                evaluation_profile=inputs.profile,
                golden_set=inputs.golden_set,
            )
        self._store.write_models(self._file(run_id, "questions"), result.questions, QuestionCase)
        self._store.write_models(
            self._file(run_id, "questions_rejected"), result.rejected, RejectedQuestion
        )
        self._save_manifest(manifest, Step.QUESTIONS)
        return QuestionStep(result.questions, result.rejected, reused=False)

    async def execute_questions(
        self,
        run_id: str,
        inputs: RunInputs,
        *,
        force: bool = False,
        start_from_question: int | None = None,
    ) -> ExecuteStep:
        manifest = self._require_manifest(run_id, inputs)
        scenarios = self._load_scenarios(run_id)
        questions = self._load_questions(run_id)
        validate_run(scenarios=scenarios, questions=questions)
        if start_from_question is not None and not 1 <= start_from_question <= len(questions):
            raise ArtifactIntegrityError(
                f"start_from_question must be between 1 and {len(questions)}"
            )
        if not force and start_from_question is None and self._has(run_id, Step.EXECUTE):
            executions = self._load_executions(run_id)
            validate_run(scenarios=scenarios, questions=questions, executions=executions)
            return ExecuteStep(executions, reused=True)

        raw_dir = run_artifact(run_id, RAW_DIR)
        preserved: list[ExecutionRecord] = []
        selected_questions = questions
        if start_from_question is None:
            for stale in sorted(self._store.path_for(raw_dir).glob("*.json")):
                stale.unlink()  # raw responses of a previous execute step
        else:
            previous = self._load_executions(run_id)
            boundary = start_from_question - 1
            if len(previous) < boundary:
                raise ArtifactIntegrityError(
                    f"Cannot resume from question {start_from_question}: only "
                    f"{len(previous)} execution record(s) exist"
                )
            preserved = previous[:boundary]
            for index, record in enumerate(preserved):
                expected = questions[index].question_id
                if record.question_id != expected:
                    raise ArtifactIntegrityError(
                        f"Cannot resume: execution {record.execution_id} is for "
                        f"{record.question_id}, expected {expected}"
                    )
            for stale_record in previous[boundary:]:
                if stale_record.raw_response_path:
                    self._store.delete(run_artifact(run_id, stale_record.raw_response_path))
            selected_questions = questions[boundary:]
            self._log.info(
                "Resuming execution from question %d: preserving %d, retrying %d",
                start_from_question,
                len(preserved),
                len(selected_questions),
            )

        def write_raw(execution_id: str, body: JsonValue) -> str:
            relative = raw_dir / f"{execution_id}.json"
            self._store.write_text(relative, json.dumps(body, indent=2, ensure_ascii=False))
            return f"{RAW_DIR}/{execution_id}.json"

        ids = self._ids()
        for _ in preserved:
            ids.next(IdKind.EXECUTION)
        executor = ProductExecutor(
            self._product,
            self._settings,
            self._log,
            llm=self._llm,
            id_generator=ids,
        )
        scenario_by_id = {scenario.id: scenario for scenario in scenarios}
        pack_by_id = {pack.pack_id: pack for pack in self._load_packs(run_id)}
        with self._trace(run_id, Step.EXECUTE):
            retried = await executor.execute_many(
                selected_questions,
                product_id=inputs.product.product_id,
                persona=inputs.persona,
                scenarios=scenario_by_id,
                packs=pack_by_id,
                raw_writer=write_raw,
            )
        executions = [*preserved, *retried]
        self._store.write_models(self._file(run_id, "executions"), executions, ExecutionRecord)
        self._save_manifest(manifest, Step.EXECUTE)
        return ExecuteStep(executions, reused=False)

    async def evaluate_executions(
        self,
        run_id: str,
        inputs: RunInputs,
        *,
        force: bool = False,
        evaluation_profile: EvaluationProfile | None = None,
        agent_metrics: Sequence[AgentMetric] | None = None,
    ) -> EvaluateStep:
        manifest = self._require_manifest(run_id, inputs)
        scenarios = self._load_scenarios(run_id)
        questions = self._load_questions(run_id)
        packs = self._load_packs(run_id)
        executions = self._load_executions(run_id)
        validate_run(scenarios=scenarios, packs=packs, questions=questions, executions=executions)
        if not force and self._has(run_id, Step.EVALUATE):
            evaluations = self._store.read_models(
                self._file(run_id, "evaluations"), EvaluationResult
            )
            validate_run(
                scenarios=scenarios,
                packs=packs,
                questions=questions,
                executions=executions,
                evaluations=evaluations,
            )
            bugs = [b for e in evaluations for b in e.bug_candidates]
            kpis = self._store.read_model(self._file(run_id, "kpis"), RunKpis)
            return EvaluateStep(evaluations, bugs, kpis, reused=True)

        scenario_by_id = {s.id: s for s in scenarios}
        question_by_id = {q.question_id: q for q in questions}
        pack_by_id = {p.pack_id: p for p in packs}
        cases = []
        for e in executions:
            question = question_by_id[e.question_id]
            pack_id = question.reference_pack_id
            cases.append(
                EvaluationCase(
                    question,
                    scenario_by_id[question.scenario_id],
                    pack_by_id[pack_id] if pack_id is not None else None,
                    e,
                )
            )
        evaluator = Evaluator(
            self._llm,
            self._settings,
            self._log,
            reference=inputs.reference,
            profile=evaluation_profile or inputs.profile,
            agent_metrics=agent_metrics,
            id_generator=self._ids(),
        )
        with self._trace(run_id, Step.EVALUATE):
            evaluations = await evaluator.evaluate_many(cases, inputs.product)
        bugs = [b for e in evaluations for b in e.bug_candidates]
        kpis = compute_kpis(
            run_id=run_id,
            product_id=inputs.product.product_id,
            mode=inputs.mode,
            scenarios=scenarios,
            executions=executions,
            evaluations=evaluations,
        )
        self._store.write_models(self._file(run_id, "evaluations"), evaluations, EvaluationResult)
        self._store.write_models(self._file(run_id, "bug_candidates"), bugs, BugCandidate)
        self._store.write_model(self._file(run_id, "kpis"), kpis)
        self._store.write_text(self._file(run_id, "kpis_csv"), kpis_csv(kpis))
        self._save_manifest(manifest, Step.EVALUATE)
        return EvaluateStep(evaluations, bugs, kpis, reused=False)

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

        self.check_reference(inputs)
        reference_id = inputs.reference_id
        manifest = self._manifest_or_none(run_id)
        cascade = manifest is not None and (
            manifest.reference_id != reference_id
            or manifest.mode is not inputs.mode
            or (
                manifest.evaluation_profile is not None
                and manifest.evaluation_profile is not inputs.profile
            )
        )
        # "reused" = same reference (and mode) as the existing run; the check itself is cheap.
        note(Step.REFERENCE, manifest is not None and not cascade)
        if cascade:
            self._log.info("Reference or mode changed; regenerating the run from scenarios")

        scenario_step = await self.generate_scenarios(
            run_id,
            inputs,
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
            run_id,
            inputs,
            force=cascade or Step.EVALUATE in regenerate,
            evaluation_profile=inputs.profile,
        )
        note(Step.EVALUATE, evaluate_step.reused)

        verdicts = Counter(
            e.verdict.value if e.verdict else "recorded_only" for e in evaluate_step.evaluations
        )
        outcomes = Counter(e.outcome.value for e in execute_step.executions)
        return RunSummary(
            run_id=run_id,
            run_dir=str(self._store.path_for(run_artifact(run_id, ""))),
            mode=inputs.mode,
            reference_id=reference_id,
            steps_run=ran,
            steps_reused=reused,
            scenarios_accepted=len(scenario_step.accepted),
            scenarios_rejected=len(scenario_step.rejected),
            questions=len(question_step.questions),
            questions_rejected=len(question_step.rejected),
            outcomes={o.value: outcomes.get(o, 0) for o in ProductOutcome},
            verdicts={
                v: verdicts.get(v, 0) for v in ("pass", "fail", "inconclusive", "recorded_only")
            },
            bug_candidates=len(evaluate_step.bug_candidates),
            kpis=evaluate_step.kpis,
        )

    # --- artifact helpers --------------------------------------------------------------

    def _file(self, run_id: str, key: str) -> Path:
        name = next(files[key] for files in STEP_ARTIFACTS.values() if key in files)
        return run_artifact(run_id, name)

    def _has(self, run_id: str, step: Step) -> bool:
        required = set(STEP_ARTIFACTS[step]) - OPTIONAL_ARTIFACTS
        return all(self._store.exists(self._file(run_id, k)) for k in required)

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
    def _check_manifest(manifest: RunManifest, inputs: RunInputs) -> None:
        expected = (manifest.persona_id, manifest.product_id, manifest.mode)
        actual = (inputs.persona.id, inputs.product.product_id, inputs.mode)
        if expected != actual:
            raise ArtifactIntegrityError(
                f"Configured inputs {actual} do not match run {manifest.run_id} {expected}"
            )
        if manifest.reference_id != inputs.reference_id:
            raise ArtifactIntegrityError(
                f"Run {manifest.run_id} was built from reference {manifest.reference_id}, "
                f"not {inputs.reference_id} (file or usage guide changed); regenerate from the "
                "scenarios step"
            )
        if (
            manifest.evaluation_profile is not None
            and manifest.evaluation_profile is not inputs.profile
        ):
            raise ArtifactIntegrityError(
                f"Run {manifest.run_id} was built for {manifest.evaluation_profile.value}, not "
                f"{inputs.profile.value}; regenerate from the scenarios step"
            )

    @contextmanager
    def _trace(self, run_id: str, step: Step) -> Iterator[None]:
        """Append this step's LLM / product calls (prompts and outputs) to run.log."""
        if not self._settings.trace.enabled:
            yield
            return
        with trace_step(
            self._store.path_for(run_artifact(run_id, RUN_LOG)),
            run_id,
            step.value,
            max_field_chars=self._settings.trace.max_field_chars,
        ):
            yield

    def _save_manifest(
        self, manifest: RunManifest, step: Step, skip: set[str] | frozenset[str] = frozenset()
    ) -> None:
        """Record ``step``'s artifacts (minus ``skip``: not written in this mode); drop (and
        delete) artifacts of all later steps."""
        later = steps_from(step) - {step}
        stale = {key for s in later for key in STEP_ARTIFACTS[s]}
        for key in stale:
            self._store.delete(self._file(manifest.run_id, key))
        artifacts = {k: v for k, v in manifest.artifacts.items() if k not in stale}
        artifacts = {k: v for k, v in artifacts.items() if k not in skip}
        artifacts.update({k: v for k, v in STEP_ARTIFACTS[step].items() if k not in skip})
        self._store.write_model(
            run_artifact(manifest.run_id, MANIFEST),
            manifest.model_copy(update={"artifacts": artifacts}),
        )

    def _load_scenarios(self, run_id: str) -> list[Scenario]:
        return self._store.read_models(self._file(run_id, "scenarios"), Scenario)

    def _load_questions(self, run_id: str) -> list[QuestionCase]:
        return self._store.read_models(self._file(run_id, "questions"), QuestionCase)

    def _load_executions(self, run_id: str) -> list[ExecutionRecord]:
        return self._store.read_models(self._file(run_id, "executions"), ExecutionRecord)

    def _load_packs(self, run_id: str) -> list[ReferencePack]:
        relative = self._file(run_id, "reference_packs")
        if not self._store.exists(relative):
            return []
        return self._store.read_models(relative, ReferencePack)
