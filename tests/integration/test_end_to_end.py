"""End-to-end: scripted FakeLLM + fixture product client, everything in tmp_path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import TypeAdapter
from typer.testing import CliRunner

from synthetic_user.cli import app
from synthetic_user.core.config import AppConfig, load_config
from synthetic_user.core.exceptions import ArtifactIntegrityError
from synthetic_user.core.loaders import load_catalog, load_persona, load_product
from synthetic_user.core.logging import get_run_logger
from synthetic_user.llm.scripted import ScriptedLLMClient
from synthetic_user.models.evaluation import BugCandidate, EvaluationResult
from synthetic_user.models.execution import ExecutionResult
from synthetic_user.models.question import QuestionCase, RejectedQuestion
from synthetic_user.models.reference import ReferenceSnapshot
from synthetic_user.models.run import RunManifest
from synthetic_user.models.scenario import RejectedScenario, Scenario
from synthetic_user.pipeline import RunInputs, Step, SyntheticUserPipeline, validate_run
from synthetic_user.product.fixture_client import FixtureProductClient
from synthetic_user.storage.json_store import JsonArtifactStore

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "RUN_20261014_120000_abcd"


@pytest.fixture
def fake_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """config/app.fake.yaml with every path made absolute and artifacts in tmp_path."""
    monkeypatch.chdir(tmp_path)
    for var in (
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "LLM_PROVIDER",
        "SAFECHAIN_MODEL_ID",
        "SAFECHAIN_CONFIG_PATH",
        "CONFIG_PATH",
        "LOG_LEVEL",
    ):
        monkeypatch.delenv(var, raising=False)
    data = yaml.safe_load((REPO_ROOT / "config" / "app.fake.yaml").read_text(encoding="utf-8"))
    data["llm"]["fake_script_path"] = str(REPO_ROOT / data["llm"]["fake_script_path"])
    data["product"]["fixture_path"] = str(REPO_ROOT / data["product"]["fixture_path"])
    data["inputs"] = {k: str(REPO_ROOT / v) for k, v in data["inputs"].items()}
    data["paths"]["artifacts_dir"] = str(tmp_path / "artifacts")
    path = tmp_path / "app.fake.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def settings(fake_config: Path) -> AppConfig:
    return load_config(fake_config, env={})


@pytest.fixture
def inputs(settings: AppConfig) -> RunInputs:
    return RunInputs(
        persona=load_persona(settings.inputs.persona_path),
        product=load_product(settings.inputs.product_path),
        catalog=load_catalog(settings.inputs.catalog_path),
    )


def _pipeline(
    settings: AppConfig,
) -> tuple[SyntheticUserPipeline, ScriptedLLMClient, FixtureProductClient]:
    assert settings.llm.fake_script_path and settings.product.fixture_path
    llm = ScriptedLLMClient.from_file(settings.llm.fake_script_path)
    product = FixtureProductClient.from_file(settings.product.fixture_path)
    pipeline = SyntheticUserPipeline(
        settings,
        llm,
        product,
        JsonArtifactStore(settings.paths.artifacts_dir),
        logger=get_run_logger("test", RUN_ID),
    )
    return pipeline, llm, product


def _load(run_dir: Path, name: str, adapter: TypeAdapter[Any]) -> Any:
    return adapter.validate_json((run_dir / name).read_bytes())


async def test_full_run_files_types_refs_and_outcomes(
    settings: AppConfig, inputs: RunInputs
) -> None:
    pipeline, _llm, product = _pipeline(settings)
    summary = await pipeline.run(inputs, run_id=RUN_ID)
    run_dir = Path(summary.run_dir)

    # all files exist and load typed
    scenarios = _load(run_dir, "scenarios.json", TypeAdapter(list[Scenario]))
    _load(run_dir, "scenarios_rejected.json", TypeAdapter(list[RejectedScenario]))
    snapshots = _load(run_dir, "snapshots.json", TypeAdapter(list[ReferenceSnapshot]))
    questions = _load(run_dir, "questions.json", TypeAdapter(list[QuestionCase]))
    _load(run_dir, "questions_rejected.json", TypeAdapter(list[RejectedQuestion]))
    executions = _load(run_dir, "executions.json", TypeAdapter(list[ExecutionResult]))
    evaluations = _load(run_dir, "evaluations.json", TypeAdapter(list[EvaluationResult]))
    bugs = _load(run_dir, "bug_candidates.json", TypeAdapter(list[BugCandidate]))
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_bytes())
    assert set(manifest.artifacts.values()) == {p.name for p in run_dir.iterdir()} - {
        "manifest.json"
    }
    assert manifest.reference_index_id == summary.reference_index_id

    # cross-references valid
    validate_run(
        scenarios=scenarios,
        snapshots=snapshots,
        questions=questions,
        executions=executions,
        evaluations=evaluations,
    )

    # >=1 fail with bug, 1 clarification evaluation, 1 recorded_only
    failed = [e for e in evaluations if e.verdict == "fail"]
    assert failed and all(e.bug_candidates for e in failed)
    assert len([e for e in evaluations if e.clarification_evaluation is not None]) == 1
    assert len([e for e in evaluations if e.evaluation_status == "recorded_only"]) == 1
    assert [b.bug_id for b in bugs] == [b.bug_id for e in evaluations for b in e.bug_candidates]

    assert summary.steps_run == list(Step)
    assert summary.verdicts == {"pass": 1, "fail": 1, "inconclusive": 0, "recorded_only": 1}
    assert summary.outcomes["sql_generated"] == 1
    assert len(product.asked) == len(questions) == 3


async def test_rerun_reuses_everything(settings: AppConfig, inputs: RunInputs) -> None:
    pipeline, llm, _ = _pipeline(settings)
    await pipeline.run(inputs, run_id=RUN_ID)
    first_calls = len(llm.calls)

    pipeline2, llm2, product2 = _pipeline(settings)
    summary = await pipeline2.run(inputs, run_id=RUN_ID)
    assert summary.steps_reused == list(Step)
    assert llm2.calls == [] and product2.asked == []
    assert first_calls > 0


async def test_from_step_regenerates_downstream_only(
    settings: AppConfig, inputs: RunInputs
) -> None:
    pipeline, _, _ = _pipeline(settings)
    await pipeline.run(inputs, run_id=RUN_ID)
    pipeline2, llm2, product2 = _pipeline(settings)
    summary = await pipeline2.run(inputs, run_id=RUN_ID, from_step=Step.EXECUTE)
    assert summary.steps_run == [Step.EXECUTE, Step.EVALUATE]
    assert len(product2.asked) == 3
    assert all("You review SQL" in c.system or "You judge" in c.system for c in llm2.calls)


async def test_force_step_cascades(settings: AppConfig, inputs: RunInputs) -> None:
    pipeline, _, _ = _pipeline(settings)
    await pipeline.run(inputs, run_id=RUN_ID)
    pipeline2, _, _ = _pipeline(settings)
    summary = await pipeline2.run(inputs, run_id=RUN_ID, force_steps=[Step.QUESTIONS])
    assert summary.steps_run == [Step.QUESTIONS, Step.EXECUTE, Step.EVALUATE]


async def test_broken_cross_reference_detected(settings: AppConfig, inputs: RunInputs) -> None:
    pipeline, _, _ = _pipeline(settings)
    summary = await pipeline.run(inputs, run_id=RUN_ID)
    path = Path(summary.run_dir) / "executions.json"
    executions = json.loads(path.read_text(encoding="utf-8"))
    executions[0]["question_id"] = "Q_999"
    path.write_text(json.dumps(executions), encoding="utf-8")

    pipeline2, _, _ = _pipeline(settings)
    with pytest.raises(ArtifactIntegrityError, match="unknown question Q_999"):
        await pipeline2.run(inputs, run_id=RUN_ID)


async def test_regenerating_a_step_removes_stale_downstream(
    settings: AppConfig, inputs: RunInputs
) -> None:
    pipeline, _, _ = _pipeline(settings)
    summary = await pipeline.run(inputs, run_id=RUN_ID)
    run_dir = Path(summary.run_dir)
    await pipeline.generate_questions(RUN_ID, inputs, force=True)
    assert not (run_dir / "executions.json").exists()
    assert not (run_dir / "evaluations.json").exists()
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_bytes())
    assert "executions" not in manifest.artifacts and "evaluations" not in manifest.artifacts


def test_cli_run_offline(fake_config: Path) -> None:
    result = CliRunner().invoke(
        app, ["--config", str(fake_config), "--log-level", "WARNING", "run", "--run-id", RUN_ID]
    )
    assert result.exit_code == 0, result.output
    assert "verdicts pass=1 fail=1 inconclusive=0 recorded_only=1" in result.output
    assert "bug_candidates=1" in result.output
    assert "run_folder=" in result.output

    again = CliRunner().invoke(
        app, ["--config", str(fake_config), "run", "--run-id", RUN_ID, "--from-step", "evaluate"]
    )
    assert again.exit_code == 0, again.output
    assert "steps_run=evaluate" in again.output


def test_cli_step_by_step_offline(fake_config: Path) -> None:
    runner = CliRunner()
    base = ["--config", str(fake_config), "--log-level", "WARNING"]
    for args in (
        ["build-index"],
        ["generate-scenarios", "--run-id", RUN_ID],
        ["generate-questions", "--run-id", RUN_ID],
        ["execute", "--run-id", RUN_ID],
        ["evaluate", "--run-id", RUN_ID],
    ):
        result = runner.invoke(app, [*base, *args])
        assert result.exit_code == 0, (args, result.output)
    assert (
        "sql_generated=1 clarification=1 denied=1"
        in runner.invoke(app, [*base, "execute", "--run-id", RUN_ID]).output
    )


def test_config_option_after_command(fake_config: Path) -> None:
    """Card 17: `synthetic-user run --config config/app.fake.yaml` must work."""
    result = CliRunner().invoke(
        app, ["run", "--config", str(fake_config), "--log-level", "WARNING", "--run-id", RUN_ID]
    )
    assert result.exit_code == 0, result.output
    assert "bug_candidates=1" in result.output
