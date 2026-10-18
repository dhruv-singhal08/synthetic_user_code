"""End-to-end: scripted FakeLLM + fixture product client, everything in tmp_path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import TypeAdapter
from typer.testing import CliRunner

from synthetic_user.adapters.fixture import FixtureAdapter
from synthetic_user.adapters.registry import create_adapter
from synthetic_user.cli import app
from synthetic_user.core.config import AppConfig, load_config
from synthetic_user.core.exceptions import ArtifactIntegrityError, InputAssetError
from synthetic_user.core.loaders import load_persona, load_product
from synthetic_user.core.logging import get_run_logger
from synthetic_user.llm.scripted import ScriptedLLMClient
from synthetic_user.models.evaluation import BugCandidate, EvaluationResult
from synthetic_user.models.execution import ExecutionRecord
from synthetic_user.models.question import QuestionCase, RejectedQuestion
from synthetic_user.models.reference import CoreRef, ReferencePack, ReferencePackDraft
from synthetic_user.models.run import RunManifest
from synthetic_user.models.scenario import RejectedScenario, Scenario
from synthetic_user.pipeline import RunInputs, Step, SyntheticUserPipeline, validate_run
from synthetic_user.references.loader import load_reference
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
    data["product_adapter"] = {
        "type": "fixture",
        "options": {
            "path": str(REPO_ROOT / "data/fixtures/product_responses/sample_responses.json")
        },
    }
    data["inputs"] = {
        k: ([str(REPO_ROOT / x) for x in v] if isinstance(v, list) else str(REPO_ROOT / v))
        for k, v in data["inputs"].items()
        if v is not None
    }
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
        reference=load_reference(
            settings.inputs.reference_file,
            max_chars=settings.reference.max_chars,
            guide_path=settings.inputs.reference_guide_file,
        ),
    )


def _pipeline(
    settings: AppConfig,
) -> tuple[SyntheticUserPipeline, ScriptedLLMClient, FixtureAdapter]:
    assert settings.llm.fake_script_path
    llm = ScriptedLLMClient.from_file(settings.llm.fake_script_path)
    product = create_adapter(settings)
    assert isinstance(product, FixtureAdapter)
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
    packs = _load(run_dir, "reference_packs.json", TypeAdapter(list[ReferencePack]))
    questions = _load(run_dir, "questions.json", TypeAdapter(list[QuestionCase]))
    _load(run_dir, "questions_rejected.json", TypeAdapter(list[RejectedQuestion]))
    executions = _load(run_dir, "executions.json", TypeAdapter(list[ExecutionRecord]))
    evaluations = _load(run_dir, "evaluations.json", TypeAdapter(list[EvaluationResult]))
    bugs = _load(run_dir, "bug_candidates.json", TypeAdapter(list[BugCandidate]))
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_bytes())
    assert set(manifest.artifacts.values()) == {p.name for p in run_dir.iterdir()} - {
        "manifest.json"
    }
    assert manifest.reference_id == summary.reference_id == inputs.reference_id
    assert [p.pack_id for p in packs] == [s.reference_pack_id for s in scenarios]

    # cross-references valid
    validate_run(
        scenarios=scenarios,
        packs=packs,
        questions=questions,
        executions=executions,
        evaluations=evaluations,
    )

    # the wrong join fails reference_alignment with a bug, 1 clarification evaluation, the
    # justified denial passes
    failed = [e for e in evaluations if e.verdict == "fail"]
    assert failed and all(e.bug_candidates for e in failed)
    assert [f.category for e in failed for f in e.findings] == ["reference_alignment"]
    assert len([e for e in evaluations if e.clarification_evaluation is not None]) == 1
    denied = [e for e in evaluations if e.product_outcome == "denied"]
    assert [(e.verdict, e.evaluation_status) for e in denied] == [("pass", "evaluated")]
    assert [b.bug_id for b in bugs] == [b.bug_id for e in evaluations for b in e.bug_candidates]

    assert summary.steps_run == list(Step)
    assert summary.verdicts == {"pass": 2, "fail": 1, "inconclusive": 0, "recorded_only": 0}
    assert summary.outcomes["answered"] == 1
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
    assert llm2.calls and all("You evaluate how well" in c.system for c in llm2.calls)


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
    assert "verdicts pass=2 fail=1 inconclusive=0 recorded_only=0" in result.output
    assert "bug_candidates=1" in result.output
    assert "run_folder=" in result.output

    again = CliRunner().invoke(
        app, ["--config", str(fake_config), "run", "--run-id", RUN_ID, "--from-step", "evaluate"]
    )
    assert again.exit_code == 0, again.output
    assert "steps_run=evaluate" in again.output


def test_cli_step_by_step_offline(fake_config: Path, settings: AppConfig) -> None:
    reference_file = settings.inputs.reference_file
    runner = CliRunner()
    base = ["--config", str(fake_config), "--log-level", "WARNING"]
    for args in (
        ["check-reference", "--reference", str(reference_file)],
        ["generate-scenarios", "--run-id", RUN_ID],
        ["generate-questions", "--run-id", RUN_ID],
        ["execute", "--run-id", RUN_ID],
        ["evaluate", "--run-id", RUN_ID],
    ):
        result = runner.invoke(app, [*base, *args])
        assert result.exit_code == 0, (args, result.output)
    assert (
        "answered=1 clarification=1 denied=1"
        in runner.invoke(app, [*base, "execute", "--run-id", RUN_ID]).output
    )


def test_config_option_after_command(fake_config: Path) -> None:
    """Card 17: `synthetic-user run --config config/app.fake.yaml` must work."""
    result = CliRunner().invoke(
        app, ["run", "--config", str(fake_config), "--log-level", "WARNING", "--run-id", RUN_ID]
    )
    assert result.exit_code == 0, result.output
    assert "bug_candidates=1" in result.output


async def test_ungrounded_run_has_no_reference_packs(settings: AppConfig) -> None:
    """Ungrounded runs get scenarios without packs (no reference_packs.json), questions,
    executions and evaluations without reference_alignment."""
    inputs = RunInputs(
        persona=load_persona(settings.inputs.persona_path),
        product=load_product(REPO_ROOT / "data" / "products" / "sample_chatbot.yaml"),
    )
    pipeline, _llm, _product = _pipeline(settings)
    scenarios = await pipeline.generate_scenarios(RUN_ID, inputs, count=2, force=True)
    assert [s.mode for s in scenarios.accepted] == ["ungrounded", "ungrounded"]
    questions = await pipeline.generate_questions(RUN_ID, inputs, force=True)
    assert len(questions.questions) == 2 and not scenarios.packs
    assert all(q.mode == "ungrounded" and q.reference_pack_id is None for q in questions.questions)
    run_dir = Path(settings.paths.artifacts_dir) / "runs" / RUN_ID
    assert not (run_dir / "reference_packs.json").exists()
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_bytes())
    assert "reference_packs" not in manifest.artifacts and manifest.mode == "ungrounded"

    again = await pipeline.generate_questions(RUN_ID, inputs)  # reused without packs
    assert again.reused and len(again.questions) == 2
    executed = await pipeline.execute_questions(RUN_ID, inputs, force=True)
    assert len(executed.executions) == 2
    evaluated = await pipeline.evaluate_executions(RUN_ID, inputs, force=True)
    # the offline nl2sql fixture has no answers for chatbot questions -> error, recorded only
    assert [e.evaluation_status for e in evaluated.evaluations] == ["recorded_only"] * 2
    assert all(e.mode == "ungrounded" for e in evaluated.evaluations)


def test_cli_export_fixture_template_then_execute_with_it(
    fake_config: Path, tmp_path: Path
) -> None:
    """C07: testers paste real answers into the exported template for any product."""
    data = yaml.safe_load(fake_config.read_text(encoding="utf-8"))
    del data["inputs"]["reference_file"], data["inputs"]["reference_guide_file"]
    data["inputs"]["product_path"] = str(REPO_ROOT / "data/products/sample_chatbot.yaml")
    config = tmp_path / "chatbot.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    runner = CliRunner()
    base = ["--config", str(config), "--log-level", "WARNING"]
    for args in (
        ["generate-scenarios", "--run-id", RUN_ID, "--count", "2"],
        ["generate-questions", "--run-id", RUN_ID],
        ["export-fixture-template", "--run-id", RUN_ID],
    ):
        result = runner.invoke(app, [*base, *args])
        assert result.exit_code == 0, (args, result.output)
    run_dir = tmp_path / "artifacts" / "runs" / RUN_ID
    template = json.loads((run_dir / "fixture_template.json").read_text(encoding="utf-8"))
    assert template["product_id"] == "sample_policy_chatbot"
    assert [a["outcome"] for a in template["answers"]] == ["answered", "answered"]

    template["answers"][0]["text"] = "220 EUR per night including taxes."
    template["answers"][1].update(outcome="denied", message="I can't approve reports.")
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps(template), encoding="utf-8")
    adapter_file = tmp_path / "sample_policy_chatbot.adapter.yaml"
    adapter_file.write_text(
        yaml.safe_dump({"type": "fixture", "options": {"path": str(answers)}}), encoding="utf-8"
    )
    result = runner.invoke(
        app, [*base, "execute", "--run-id", RUN_ID, "--adapter-config", str(adapter_file)]
    )
    assert result.exit_code == 0, result.output
    assert "answered=1 clarification=0 denied=1" in result.output


async def test_pack_edit_invalidates_later_steps(settings: AppConfig, inputs: RunInputs) -> None:
    pipeline, _, _ = _pipeline(settings)
    summary = await pipeline.run(inputs, run_id=RUN_ID)
    run_dir = Path(summary.run_dir)
    draft = ReferencePackDraft(
        core_refs=[CoreRef(name="employees.full_name", quote="Employee full name.", why="ranking")]
    )
    edited = pipeline.update_reference_pack(RUN_ID, inputs, "RP_SCN_001", draft)
    assert edited.edited and [r.name for r in edited.core_refs] == ["employees.full_name"]
    packs = TypeAdapter(list[ReferencePack]).validate_json(
        (run_dir / "reference_packs.json").read_bytes()
    )
    assert packs[0] == edited and not packs[1].edited
    for stale in ("questions.json", "executions.json", "evaluations.json"):
        assert not (run_dir / stale).exists()
    with pytest.raises(InputAssetError, match="quote not found"):
        pipeline.update_reference_pack(
            RUN_ID,
            inputs,
            "RP_SCN_002",
            ReferencePackDraft(core_refs=[CoreRef(name="x", quote="not in the file", why="y")]),
        )
    # the next run regenerates questions and later steps from the edited pack
    pipeline2, llm2, _ = _pipeline(settings)
    again = await pipeline2.run(inputs, run_id=RUN_ID)
    assert again.steps_run == [Step.QUESTIONS, Step.EXECUTE, Step.EVALUATE]
    assert "Employee full name." in llm2.calls[0].user
