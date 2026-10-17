"""Command-line entry point (``synthetic-user``).

Every run-based command goes through SyntheticUserPipeline. Explicit step commands always
regenerate their step (and drop stale downstream artifacts); ``run`` reuses what exists.

Exit codes: 0 ok (fail verdicts are results, not errors); 1 error (SyntheticUserError or
unexpected).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, NoReturn, TypeVar

import typer
from dotenv import find_dotenv, load_dotenv
from pydantic import TypeAdapter

from synthetic_user.adapters.base import ProductAdapter, UnavailableAdapter
from synthetic_user.adapters.fixture import fixture_template
from synthetic_user.adapters.registry import create_adapter
from synthetic_user.benchmark import load_cases, load_scenarios, run_benchmark
from synthetic_user.core.config import AppConfig, LLMProvider, load_adapter_file, load_config
from synthetic_user.core.exceptions import (
    ArtifactIntegrityError,
    ConfigError,
    SyntheticUserError,
)
from synthetic_user.core.identifiers import RUN_ID_PATTERN, SequentialIdGenerator, new_run_id
from synthetic_user.core.loaders import load_persona, load_product
from synthetic_user.core.logging import get_run_logger, setup_logging
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.factory import create_llm_client
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.base import StrictModel
from synthetic_user.models.evaluation import BugCandidate, EvaluationResult
from synthetic_user.models.execution import ExecutionRecord, ProductOutcome
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import ReferenceIndex
from synthetic_user.models.run import RunManifest
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator import EvaluationCase, Evaluator
from synthetic_user.pipeline import RunInputs, RunSummary, Step, SyntheticUserPipeline
from synthetic_user.prompts.smoke import SMOKE_SYSTEM, SMOKE_USER
from synthetic_user.references.integrity import validate_index
from synthetic_user.references.loader import load_optional_reference_index, load_reference_index
from synthetic_user.references.snapshot import snapshots_for
from synthetic_user.storage.json_store import JsonArtifactStore
from synthetic_user.storage.runs import run_artifact

T = TypeVar("T")

app = typer.Typer(no_args_is_help=True, add_completion=False)
_log = logging.getLogger(__name__)

EXIT_ERROR = 1

RunIdOption = Annotated[
    str | None,
    typer.Option("--run-id", help="Run id (RUN_YYYYMMDD_HHMMSS_xxxx). Default: new."),
]


ReferenceIndexOption = Annotated[
    Path | None,
    typer.Option(
        "--reference-index",
        help="Pre-built reference index JSON -> grounded. Default: inputs.reference_index_file; "
        "none -> ungrounded.",
    ),
]

CommandConfigOption = Annotated[
    Path | None,
    typer.Option("--config", help="Config YAML (same as the global --config)."),
]
CommandLogLevelOption = Annotated[
    str | None,
    typer.Option("--log-level", help="Log level (same as the global --log-level)."),
]


@dataclass(frozen=True)
class CliState:
    config_path: Path | None


def _state(ctx: typer.Context, config: Path | None, log_level: str | None) -> CliState:
    """Command-level --config / --log-level override the global ones."""
    state: CliState = ctx.obj
    if log_level is not None:
        _guard(lambda: setup_logging(log_level))
    return CliState(config_path=config) if config is not None else state


class SmokeReply(StrictModel):
    status: Literal["ok"]


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Config YAML. Overrides CONFIG_PATH and ./config/app.yaml."),
    ] = None,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="DEBUG, INFO, WARNING, ERROR. Overrides LOG_LEVEL."),
    ] = None,
) -> None:
    """Synthetic User for automated testing of conversational products."""
    load_dotenv(find_dotenv(usecwd=True), override=False)
    _guard(lambda: setup_logging(log_level or os.environ.get("LOG_LEVEL", "INFO")))
    ctx.obj = CliState(config_path=config)


# --- smoke --------------------------------------------------------------------------------


@app.command("smoke-llm")
def smoke_llm(
    ctx: typer.Context,
    config: CommandConfigOption = None,
    log_level: CommandLogLevelOption = None,
) -> None:
    """Send a tiny JSON request to the configured LLM; print provider, model, latency."""
    state = _state(ctx, config, log_level)
    _guard(lambda: asyncio.run(_smoke(state)))


async def _smoke(state: CliState) -> None:
    settings = load_config(state.config_path)
    client = _smoke_client(settings)
    started = time.perf_counter()
    await generate_structured(
        client,
        SMOKE_SYSTEM,
        SMOKE_USER,
        SmokeReply,
        max_retries=settings.llm.max_validation_retries,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    typer.echo(
        f"provider={client.provider_name} model={client.model_name} "
        f"latency_ms={latency_ms:.0f} status=ok"
    )


def _smoke_client(settings: AppConfig) -> LLMClient:
    client = create_llm_client(settings)
    if settings.llm.provider is LLMProvider.FAKE and isinstance(client, FakeLLMClient):
        # The bare fake provider has no model behind it; script the expected reply.
        client.queue({"status": "ok"})
    return client


# --- step commands ------------------------------------------------------------------------


@app.command("validate-index")
def validate_index_command(
    ctx: typer.Context,
    reference_index: Annotated[
        Path, typer.Option("--reference-index", help="Pre-built reference index JSON to check.")
    ],
    config: CommandConfigOption = None,
    log_level: CommandLogLevelOption = None,
) -> None:
    """Load a pre-built reference index and list warnings (exit 1 on load errors)."""
    _state(ctx, config, log_level)
    _guard(lambda: _validate_index(reference_index))


def _validate_index(path: Path) -> None:
    index = load_reference_index(path)
    warnings = validate_index(index)
    typer.echo(
        f"index_id={index.index_id} versioned_id={index.versioned_id} "
        f"tables={len(index.tables)} columns={sum(len(t.columns) for t in index.tables)} "
        f"relationships={len(index.relationships)} warnings={len(warnings)}"
    )
    for warning in warnings:
        typer.echo(f"WARNING {warning}")


@app.command("generate-scenarios")
def generate_scenarios(
    ctx: typer.Context,
    config: CommandConfigOption = None,
    log_level: CommandLogLevelOption = None,
    run_id: RunIdOption = None,
    persona: Annotated[
        Path | None, typer.Option("--persona", help="Persona YAML. Default: inputs.persona_path.")
    ] = None,
    product: Annotated[
        Path | None, typer.Option("--product", help="Product YAML. Default: inputs.product_path.")
    ] = None,
    reference_index: ReferenceIndexOption = None,
    count: Annotated[
        int | None,
        typer.Option("--count", min=1, help="Scenarios to request. Default: scenario_count."),
    ] = None,
) -> None:
    """Generate scenarios (+ cheat sheets when grounded) into artifacts/runs/<run_id>/."""
    state = _state(ctx, config, log_level)
    _guard(
        lambda: asyncio.run(
            _generate_scenarios(state, run_id, persona, product, reference_index, count)
        )
    )


async def _generate_scenarios(
    state: CliState,
    run_id: str | None,
    persona_path: Path | None,
    product_path: Path | None,
    index_path: Path | None,
    count: int | None,
) -> None:
    settings = load_config(state.config_path)
    run_id = _valid_run_id(run_id or new_run_id())
    inputs = _inputs(settings, persona_path, product_path, index_path)
    pipeline, store = _pipeline(settings, run_id)
    step = await pipeline.generate_scenarios(run_id, inputs, count=count, force=True)
    duplicates = sum(1 for r in step.rejected if r.reasons[0].startswith("duplicate of"))
    typer.echo(
        f"run_id={run_id} mode={inputs.mode} accepted={len(step.accepted)} "
        f"rejected={len(step.rejected) - duplicates} duplicates={duplicates} "
        f"dir={_run_dir(store, run_id)}"
    )


@app.command("generate-questions")
def generate_questions(
    ctx: typer.Context,
    run_id: Annotated[str, typer.Option("--run-id", help="Run with scenarios.json.")],
    config: CommandConfigOption = None,
    log_level: CommandLogLevelOption = None,
    count: Annotated[
        int | None,
        typer.Option("--count", min=1, help="Questions per scenario. Default: question_count."),
    ] = None,
    reference_index: ReferenceIndexOption = None,
) -> None:
    """Build snapshots and generate questions for a run's scenarios (always regenerates)."""
    state = _state(ctx, config, log_level)
    _guard(lambda: asyncio.run(_generate_questions(state, run_id, count, reference_index)))


async def _generate_questions(
    state: CliState, run_id: str, count: int | None, index_path: Path | None
) -> None:
    settings = load_config(state.config_path)
    run_id = _valid_run_id(run_id)
    pipeline, store = _pipeline(settings, run_id)
    inputs = _inputs(settings, reference_index_path=index_path)
    step = await pipeline.generate_questions(run_id, inputs, count=count, force=True)
    scenarios = len(
        {q.scenario_id for q in step.questions} | {r.scenario_id for r in step.rejected}
    )
    leaks = sum(1 for q in step.questions if q.schema_leak_warnings)
    typer.echo(
        f"run_id={run_id} mode={inputs.mode} scenarios={scenarios} "
        f"snapshots={len(step.snapshots)} "
        f"questions={len(step.questions)} rejected={len(step.rejected)} "
        f"with_leak_warnings={leaks} dir={_run_dir(store, run_id)}"
    )


@app.command("execute")
def execute(
    ctx: typer.Context,
    run_id: Annotated[str, typer.Option("--run-id", help="Run with questions.json.")],
    config: CommandConfigOption = None,
    log_level: CommandLogLevelOption = None,
    reference_index: ReferenceIndexOption = None,
    adapter_config: Annotated[
        Path | None,
        typer.Option(
            "--adapter-config",
            help="<product_id>.adapter.yaml to use instead of product_adapter in the config.",
        ),
    ] = None,
) -> None:
    """Submit a run's questions through the product adapter (always regenerates)."""
    state = _state(ctx, config, log_level)
    _guard(lambda: asyncio.run(_execute(state, run_id, reference_index, adapter_config)))


async def _execute(
    state: CliState, run_id: str, index_path: Path | None, adapter_config: Path | None
) -> None:
    settings = load_config(state.config_path)
    if adapter_config is not None:
        settings = settings.model_copy(
            update={"product_adapter": load_adapter_file(adapter_config)}
        )
    run_id = _valid_run_id(run_id)
    pipeline, store = _pipeline(settings, run_id, with_product=True)
    try:
        inputs = _inputs(settings, reference_index_path=index_path)
        step = await pipeline.execute_questions(run_id, inputs, force=True)
    finally:
        await pipeline.aclose()
    outcomes = {o: sum(1 for e in step.executions if e.outcome == o) for o in _OUTCOMES}
    typer.echo(
        f"run_id={run_id} executions={len(step.executions)} "
        + " ".join(f"{k}={v}" for k, v in outcomes.items())
        + f" dir={_run_dir(store, run_id)}"
    )


@app.command("export-fixture-template")
def export_fixture_template(
    ctx: typer.Context,
    run_id: Annotated[str, typer.Option("--run-id", help="Run with questions.json.")],
    config: CommandConfigOption = None,
    log_level: CommandLogLevelOption = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output JSON. Default: runs/<run_id>/fixture_template.json."),
    ] = None,
) -> None:
    """Write a fixture template (question_id, utterance, outcome, text, artifacts[]) to fill in
    with real product answers; use it with product_adapter type fixture."""
    state = _state(ctx, config, log_level)
    _guard(lambda: _export_fixture_template(state, run_id, out))


def _export_fixture_template(state: CliState, run_id: str, out: Path | None) -> None:
    settings = load_config(state.config_path)
    run_id = _valid_run_id(run_id)
    store = JsonArtifactStore(settings.paths.artifacts_dir)
    manifest = store.read_model(run_artifact(run_id, "manifest.json"), RunManifest)
    questions = store.read_models(run_artifact(run_id, "questions.json"), QuestionCase)
    template = fixture_template(
        manifest.product_id, [(q.question_id, q.utterance) for q in questions]
    )
    text = json.dumps(template, indent=2, ensure_ascii=False) + "\n"
    if out is None:
        path = store.write_text(run_artifact(run_id, FIXTURE_TEMPLATE), text)
    else:
        path = JsonArtifactStore(out.parent).write_text(out.name, text)
    typer.echo(f"run_id={run_id} questions={len(questions)} template={path}")
    typer.echo(
        "Fill in each answer (outcome answered + text and/or artifacts, or clarification / "
        "denied + message, error + error_type, timeout), then set product_adapter "
        "{type: fixture, options: {path: <this file>}} or pass it via --adapter-config."
    )


@app.command("evaluate")
def evaluate(
    ctx: typer.Context,
    config: CommandConfigOption = None,
    log_level: CommandLogLevelOption = None,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Run with questions/executions/scenarios.")
    ] = None,
    questions: Annotated[Path | None, typer.Option("--questions", help="questions.json")] = None,
    executions: Annotated[Path | None, typer.Option("--executions", help="executions.json")] = None,
    scenarios: Annotated[Path | None, typer.Option("--scenarios", help="scenarios.json")] = None,
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", help="File mode output dir. Default: the executions file's dir."),
    ] = None,
    reference_index: ReferenceIndexOption = None,
) -> None:
    """Evaluate executions; writes evaluations.json and bug_candidates.json."""
    state = _state(ctx, config, log_level)
    _guard(
        lambda: asyncio.run(
            _evaluate(state, run_id, questions, executions, scenarios, out_dir, reference_index)
        )
    )


async def _evaluate(
    state: CliState,
    run_id: str | None,
    questions_path: Path | None,
    executions_path: Path | None,
    scenarios_path: Path | None,
    out_dir: Path | None,
    index_path: Path | None,
) -> None:
    settings = load_config(state.config_path)
    files = (questions_path, executions_path, scenarios_path)
    if (run_id is None) == all(f is None for f in files):
        raise ConfigError("Pass either --run-id or all of --questions --executions --scenarios")

    if run_id is not None:
        run_id = _valid_run_id(run_id)
        pipeline, store = _pipeline(settings, run_id)
        inputs = _inputs(settings, reference_index_path=index_path)
        step = await pipeline.evaluate_executions(run_id, inputs, force=True)
        questions = store.read_models(run_artifact(run_id, "questions.json"), QuestionCase)
        _print_evaluation(
            step.evaluations, step.bug_candidates, len(questions), _run_dir(store, run_id)
        )
        return

    if questions_path is None or executions_path is None or scenarios_path is None:
        raise ConfigError("File mode needs all of --questions --executions --scenarios")
    product = load_product(settings.inputs.product_path)
    persona = load_persona(settings.inputs.persona_path)
    index_file = index_path or settings.inputs.reference_index_file
    index = load_optional_reference_index(index_file, _log)
    question_list = _read_file(questions_path, TypeAdapter(list[QuestionCase]))
    execution_items = _read_file(executions_path, TypeAdapter(list[ExecutionRecord]))
    scenario_list = _read_file(scenarios_path, TypeAdapter(list[Scenario]))
    out_store = JsonArtifactStore(out_dir or executions_path.parent)

    cases = _pair_cases(question_list, execution_items, scenario_list, index)
    evaluator = Evaluator(
        create_llm_client(settings),
        settings,
        get_run_logger("synthetic_user.nodes.evaluator", new_run_id()),
        index=index,
        id_generator=SequentialIdGenerator(),
    )
    results = await evaluator.evaluate_many(cases, product, persona)
    bugs = [b for r in results for b in r.bug_candidates]
    out_store.write_models("evaluations.json", results, EvaluationResult)
    out_store.write_models("bug_candidates.json", bugs, BugCandidate)
    _print_evaluation(results, bugs, len(question_list), out_store.path_for("."))


# --- full run -----------------------------------------------------------------------------


@app.command("run")
def run(
    ctx: typer.Context,
    config: CommandConfigOption = None,
    log_level: CommandLogLevelOption = None,
    run_id: RunIdOption = None,
    from_step: Annotated[
        Step | None,
        typer.Option("--from-step", help="Regenerate this step and everything after it."),
    ] = None,
    force_step: Annotated[
        list[Step] | None,
        typer.Option("--force-step", help="Regenerate this step (and its downstream). Repeatable."),
    ] = None,
    scenario_count: Annotated[
        int | None,
        typer.Option(
            "--scenario-count", min=1, help="Default: generation.scenario.scenario_count."
        ),
    ] = None,
    question_count: Annotated[
        int | None,
        typer.Option(
            "--question-count",
            min=1,
            help="Per scenario. Default: generation.question.question_count.",
        ),
    ] = None,
    reference_index: ReferenceIndexOption = None,
) -> None:
    """Full pipeline: index check -> scenarios -> questions -> execute -> evaluate."""
    state = _state(ctx, config, log_level)
    _guard(
        lambda: asyncio.run(
            _run(
                state,
                run_id,
                from_step,
                force_step or [],
                scenario_count,
                question_count,
                reference_index,
            )
        )
    )


async def _run(
    state: CliState,
    run_id: str | None,
    from_step: Step | None,
    force_steps: list[Step],
    scenario_count: int | None,
    question_count: int | None,
    index_path: Path | None,
) -> None:
    settings = load_config(state.config_path)
    run_id = _valid_run_id(run_id or new_run_id())
    inputs = _inputs(settings, reference_index_path=index_path)
    pipeline, _store = _pipeline(settings, run_id, with_product=True)
    try:
        summary = await pipeline.run(
            inputs,
            run_id=run_id,
            from_step=from_step,
            force_steps=force_steps,
            scenario_count=scenario_count,
            question_count=question_count,
        )
    finally:
        await pipeline.aclose()
    _print_summary(summary)


def _print_summary(summary: RunSummary) -> None:
    typer.echo(f"run_id={summary.run_id}")
    typer.echo(f"mode={summary.mode} reference_index={summary.reference_index_id or '-'}")
    typer.echo(
        f"steps_run={','.join(summary.steps_run) or '-'} "
        f"steps_reused={','.join(summary.steps_reused) or '-'}"
    )
    typer.echo(
        f"scenarios accepted={summary.scenarios_accepted} rejected={summary.scenarios_rejected}"
    )
    typer.echo(f"questions={summary.questions} rejected={summary.questions_rejected}")
    typer.echo("outcomes " + " ".join(f"{k}={v}" for k, v in summary.outcomes.items()))
    typer.echo("verdicts " + " ".join(f"{k}={v}" for k, v in summary.verdicts.items()))
    typer.echo(f"bug_candidates={summary.bug_candidates}")
    typer.echo(f"run_folder={summary.run_dir}")


@app.command("serve")
def serve(
    ctx: typer.Context,
    config: CommandConfigOption = None,
    log_level: CommandLogLevelOption = None,
    host: Annotated[str, typer.Option("--host", help="Bind address (local only by default).")] = (
        "127.0.0.1"
    ),
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8000,
) -> None:
    """Start the web UI + API (FastAPI) at http://HOST:PORT."""
    state = _state(ctx, config, log_level)

    def start() -> None:
        import uvicorn

        from synthetic_user.web.server import create_app

        settings = load_config(state.config_path)
        typer.echo(f"Synthetic User UI on http://{host}:{port}  (Ctrl+C to stop)")
        uvicorn.run(create_app(settings), host=host, port=port, log_level="warning")

    _guard(start)


# --- benchmark ----------------------------------------------------------------------------


@app.command("benchmark-evaluator")
def benchmark_evaluator(
    ctx: typer.Context,
    config: CommandConfigOption = None,
    log_level: CommandLogLevelOption = None,
    cases: Annotated[
        Path | None,
        typer.Option(
            "--cases", help="Labeled cases (JSONL). Default: inputs.benchmark_cases_path."
        ),
    ] = None,
    scenarios: Annotated[
        Path | None,
        typer.Option(
            "--scenarios", help="Scenarios for the cases. Default: inputs.benchmark_scenarios_path."
        ),
    ] = None,
) -> None:
    """LIVE ONLY: run the evaluator on human-labeled cases and report agreement."""
    state = _state(ctx, config, log_level)
    _guard(lambda: asyncio.run(_benchmark(state, cases, scenarios)))


async def _benchmark(state: CliState, cases_path: Path | None, scenarios_path: Path | None) -> None:
    settings = load_config(state.config_path)
    cases_path = cases_path or settings.inputs.benchmark_cases_path
    scenarios_path = scenarios_path or settings.inputs.benchmark_scenarios_path
    if cases_path is None or scenarios_path is None:
        raise ConfigError(
            "Set inputs.benchmark_cases_path and inputs.benchmark_scenarios_path "
            "or pass --cases and --scenarios"
        )
    if settings.llm.provider is LLMProvider.FAKE:
        raise ConfigError("benchmark-evaluator is live only: set llm.provider to a real provider")
    index_file = settings.inputs.reference_index_file
    index = load_reference_index(index_file) if index_file is not None else None
    product_paths = settings.inputs.benchmark_product_paths or [settings.inputs.product_path]
    products = {p.product_id: p for p in (load_product(path) for path in product_paths)}
    persona = load_persona(settings.inputs.persona_path)
    evaluator = Evaluator(
        create_llm_client(settings),
        settings,
        get_run_logger("synthetic_user.benchmark", new_run_id()),
        index=index,
        id_generator=SequentialIdGenerator(),
    )
    report = await run_benchmark(
        evaluator, load_cases(cases_path), load_scenarios(scenarios_path), index, products, persona
    )
    for row in report.rows:
        mark = "ok  " if row.agrees else "DIFF"
        typer.echo(
            f"{mark} {row.case_id} {row.mode} {row.product_id} label={row.label} "
            f"verdict={row.verdict}"
        )
    for group in sorted({(r.mode, r.product_id) for r in report.rows}):
        rows = [r for r in report.rows if (r.mode, r.product_id) == group]
        typer.echo(f"  {group[0]} {group[1]}: {sum(r.agrees for r in rows)}/{len(rows)}")
    typer.echo(f"agreement={report.agreement:.2f} ({report.agreed}/{len(report.rows)})")


# --- helpers ------------------------------------------------------------------------------

_OUTCOMES = tuple(o.value for o in ProductOutcome)
FIXTURE_TEMPLATE = "fixture_template.json"
_VERDICTS = ("pass", "fail", "inconclusive", "recorded_only")


def _valid_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ConfigError(f"--run-id must look like RUN_YYYYMMDD_HHMMSS_xxxx, got {run_id!r}")
    return run_id


def _inputs(
    settings: AppConfig,
    persona_path: Path | None = None,
    product_path: Path | None = None,
    reference_index_path: Path | None = None,
) -> RunInputs:
    """--reference-index, else inputs.reference_index_file; neither -> ungrounded (logged)."""
    index_path = reference_index_path or settings.inputs.reference_index_file
    return RunInputs(
        persona=load_persona(persona_path or settings.inputs.persona_path),
        product=load_product(product_path or settings.inputs.product_path),
        reference_index=load_optional_reference_index(index_path, _log),
    )


def _require_index(settings: AppConfig, index_path: Path | None) -> ReferenceIndex:
    path = index_path or settings.inputs.reference_index_file
    if path is None:
        raise ConfigError(
            "This command needs a reference index: pass --reference-index or set "
            "inputs.reference_index_file"
        )
    return load_reference_index(path)


def _pipeline(
    settings: AppConfig, run_id: str, *, with_product: bool = False
) -> tuple[SyntheticUserPipeline, JsonArtifactStore]:
    store = JsonArtifactStore(settings.paths.artifacts_dir)
    product: ProductAdapter = create_adapter(settings) if with_product else UnavailableAdapter()
    pipeline = SyntheticUserPipeline(
        settings,
        create_llm_client(settings),
        product,
        store,
        logger=get_run_logger("synthetic_user.pipeline", run_id),
    )
    return pipeline, store


def _run_dir(store: JsonArtifactStore, run_id: str) -> Path:
    return store.path_for(run_artifact(run_id, ""))


def _print_evaluation(
    results: list[EvaluationResult], bugs: list[BugCandidate], questions: int, where: Path
) -> None:
    counts = {v: sum(1 for r in results if (r.verdict or "recorded_only") == v) for v in _VERDICTS}
    typer.echo(
        f"evaluated={len(results)} "
        + " ".join(f"{k}={v}" for k, v in counts.items())
        + f" bug_candidates={len(bugs)} unexecuted={questions - len(results)} dir={where}"
    )


def _read_file(path: Path, adapter: TypeAdapter[T]) -> T:
    return JsonArtifactStore(path.parent).read_with(path.name, adapter)


def _pair_cases(
    questions: list[QuestionCase],
    executions: list[ExecutionRecord],
    scenarios: list[Scenario],
    index: ReferenceIndex | None,
) -> list[EvaluationCase]:
    by_question = {q.question_id: q for q in questions}
    by_scenario = {s.id: s for s in scenarios}
    snapshots = (
        {snap.snapshot_id: snap for snap in snapshots_for(index, scenarios)} if index else {}
    )
    cases: list[EvaluationCase] = []
    for execution in executions:
        question = by_question.get(execution.question_id)
        if question is None:
            raise ArtifactIntegrityError(
                f"{execution.execution_id} refers to unknown question {execution.question_id}"
            )
        scenario = by_scenario.get(question.scenario_id)
        if scenario is None:
            raise ArtifactIntegrityError(
                f"{question.question_id} refers to unknown scenario {question.scenario_id}"
            )
        snapshot = None
        if question.reference_snapshot_id is not None:
            snapshot = snapshots.get(question.reference_snapshot_id)
            if snapshot is None:
                raise ArtifactIntegrityError(
                    f"{question.question_id} is grounded: pass the reference index it was built "
                    "from (--reference-index)"
                )
        cases.append(EvaluationCase(question, scenario, snapshot, execution))
    return cases


def _guard(action: Callable[[], object]) -> None:
    """Map errors to messages + exit 1. Messages never include secret values."""
    try:
        action()
    except typer.Exit:
        raise
    except SyntheticUserError as exc:
        _fail(f"Error: {exc}")
    except Exception as exc:  # last-resort handler for the CLI boundary
        _log.debug("Unexpected error", exc_info=True)
        _fail(
            f"Unexpected error: {type(exc).__name__}: {exc}\n"
            "Re-run with --log-level DEBUG for details."
        )


def _fail(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=EXIT_ERROR)


if __name__ == "__main__":
    app()
