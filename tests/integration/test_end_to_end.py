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
from synthetic_user.models.kpi import RunKpis
from synthetic_user.models.metrics import (
    CustomMetricInput,
    MetricSelection,
    ResolvedMetricConfig,
)
from synthetic_user.models.question import QuestionCase, RejectedQuestion
from synthetic_user.models.reference import CatalogPackDraft, ReferencePack, TablePick
from synthetic_user.models.run import RunManifest
from synthetic_user.models.scenario import RejectedScenario, Scenario
from synthetic_user.pipeline import RunInputs, Step, SyntheticUserPipeline, validate_run
from synthetic_user.references.loader import load_reference
from synthetic_user.storage.json_store import JsonArtifactStore

REPO_ROOT = Path(__file__).resolve().parents[2]
STEP_NAMES = ("scenarios", "questions", "execute", "evaluate")
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
    pipeline, llm, product = _pipeline(settings)
    summary = await pipeline.run(inputs, run_id=RUN_ID)
    run_dir = Path(summary.run_dir)

    # the frozen metric configuration exists before scenarios and is used by every step
    config = ResolvedMetricConfig.model_validate_json((run_dir / "metric_config.json").read_bytes())
    assert config.selected_metric_ids == list(settings.evaluation.default_metrics)
    assert not config.compiler_called and summary.metric_config_id == config.config_id

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
        "manifest.json",
        "run.log",
    }

    # run log: every LLM / product call of every step, prompts in and outputs out (D202)
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    for step in STEP_NAMES:
        assert f"step {step} STARTED" in log and f"step {step} FINISHED" in log
    assert "LLM CALL  scenario batch  _RawScenarioBatch attempt 1" in log
    assert "VALIDATION _RawScenarioBatch attempt 1: ACCEPTED" in log
    for scenario in scenarios:
        assert f"LLM CALL  {scenario.id}  _RawQuestionBatch attempt 1" in log
    for question in questions:
        assert f"PRODUCT CALL  {question.question_id}  fixture adapter" in log
    for evaluation in evaluations:
        judged = f"LLM CALL  {evaluation.question_id} general  GeneralJudgeDraft"
        assert (judged in log) is (evaluation.verdict is not None)
        assert evaluation.metric_config_id == config.config_id

    # normal-path call budget: compiler 0, scenarios 1, questions 1 per accepted scenario,
    # evaluation <= 2 per evaluable response (only the general judge has metrics here)
    calls = [c.system for c in llm.calls if "MOCK of" not in c.system]
    assert sum("You design realistic test scenarios" in c for c in calls) == 1
    assert sum("You role-play" in c for c in calls) == len(scenarios)
    assert not any("You compile user-written evaluation metrics" in c for c in calls)
    assert sum("General Judge" in c for c in calls) == 2
    assert not any("Retrieval and Grounding Judge" in c for c in calls)
    assert log.count(">>> SYSTEM PROMPT") == log.count(">>> OUTPUT") == log.count("LLM CALL")
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

    # the wrong join fails rules adherence with a bug; the substantive denial is judged
    # (answer-level metrics only); the unresolved clarification is recorded only
    failed = [e for e in evaluations if e.verdict == "fail"]
    assert failed and all(e.bug_candidates for e in failed)
    # the planted wrong join: a rules finding and, as an unlisted relationship, a schema one (D238)
    assert [f.category for e in failed for f in e.findings] == [
        "rules_adherence",
        "schema_faithfulness",
        "correctness",  # correctness checks the Reference Spec too (D256)
    ]
    recorded = [e.product_outcome for e in evaluations if e.verdict is None]
    assert recorded == ["clarification"]
    by_outcome = {e.product_outcome.value: e for e in evaluations}
    assert by_outcome["answered"].response_classification == "answered"
    assert by_outcome["answered"].classification_source == "general_judge"
    denial = by_outcome["denied"]
    assert (denial.verdict, denial.response_classification) == ("pass", "unable_to_answer")
    assert {m: r.status.value for m, r in denial.metrics.items()} == {
        "relevance": "scored",
        "correctness": "scored",
        "semantic_correctness": "not_applicable",
        "rules_adherence": "not_applicable",
        "schema_faithfulness": "not_applicable",
    }
    assert by_outcome["clarification"].response_classification == "clarification"
    assert [b.bug_id for b in bugs] == [b.bug_id for e in evaluations for b in e.bug_candidates]

    # run KPIs (kpis.json / kpis.csv)
    kpis = RunKpis.model_validate_json((run_dir / "kpis.json").read_bytes())
    assert kpis == summary.kpis
    assert (kpis.judged, kpis.accuracy, kpis.task_success_rate, kpis.availability) == (
        2,
        0.5,
        0.5,
        1.0,
    )
    means = {m.metric: m.mean for m in kpis.metrics}
    assert means == {
        "relevance": 1.0,
        "correctness": 0.75,  # 0.5 on the planted join (it checks the spec), 1 on the denial
        "semantic_correctness": 1.0,
        "rules_adherence": 0.0,
        "schema_faithfulness": 0.0,  # the planted unlisted join (D238)
    }
    assert kpis.selected_metrics == config.selected_metric_ids
    assert kpis.response_classifications == {
        "answered": 1,
        "clarification": 1,
        "unable_to_answer": 1,
    }
    assert (kpis.new_bugs, kpis.new_bugs_by_severity["high"], kpis.known_gap_bugs) == (1, 1, 0)
    kpi_csv = (run_dir / "kpis.csv").read_text(encoding="utf-8")
    assert "Rules Adherence (mean),0,1" in kpi_csv and "Accuracy,0.5,2" in kpi_csv
    judged = by_outcome["answered"]
    assert judged.metrics["correctness"].details["factual_verification"] == ("unverified")
    # every score is explained: basis, assigned requirements, checks with evidence, coverage
    semantic = judged.metrics["semantic_correctness"]
    assert semantic.comparison_basis == "criteria_vs_sql"
    assert semantic.verification_method == "static_sql_analysis"
    assert semantic.assigned_criteria == ["C1", "C2"]
    assert [(c.requirement_id, c.status) for c in semantic.checks] == [
        ("C1", "passed"),
        ("C2", "passed"),
    ]
    assert semantic.checks[0].requirement == "names the employees ranked by total card spend"
    assert semantic.coverage is not None and semantic.coverage.evaluated == 2
    assert [g.guidance_id for g in semantic.guidance_applied] == ["METRIC:semantic_correctness"]
    rules = judged.metrics["rules_adherence"]
    assert rules.assigned_reference_requirements == []
    assert rules.available_rules == ["TABLE-RULE-01.01", "RULE-R1"]
    assert [(c.requirement_id, c.requirement_source, c.status) for c in rules.checks][-1] == (
        "JOIN-01",
        "derived",
        "failed",
    )
    assert rules.coverage is not None and (rules.coverage.assigned, rules.coverage.derived) == (
        2,
        1,
    )
    schema = judged.metrics["schema_faithfulness"]
    assert [(c.requirement_id, c.status) for c in schema.checks] == [
        ("reference_spec.schema", "failed")
    ]
    assert "RECON:1" in schema.observed_evidence_ids
    assert judged.failed_requirements == [
        "correctness: reference_spec.schema",
        "rules_adherence: JOIN-01",
        "schema_faithfulness: reference_spec.schema",
    ]
    assert any("not executed against a database" in item for item in judged.limitations)
    assert judged.reference_checks is not None
    assert judged.reference_checks.additional_lookup_performed is False
    # run.log reconstructs every evaluation: input, evidence, invocation, lookups, decisions
    for section in ("A. EVALUATION INPUT", "B. EVIDENCE PREPARATION", "C. JUDGE INVOCATIONS"):
        assert section in log
    for section in ("D. OPTIONAL EVIDENCE LOOKUPS", "E. METRIC DECISIONS", "F. FINAL VERDICT"):
        assert section in log
    assert f"EVALUATION TRACE  {judged.question_id}" in log
    # the question carries its Evaluation Context (no expected_behavior)
    question = next(q for q in questions if q.question_id == judged.question_id)
    context = question.context
    assert [c.id for c in context.success_criteria] == ["C1", "C2"]
    assert context.reference_spec is not None
    assert context.reference_spec.pack_id == question.reference_pack_id
    assert set(context.metric_mapping) == set(config.selected_metric_ids)

    assert summary.steps_run == list(Step)
    assert summary.verdicts == {"pass": 1, "fail": 1, "inconclusive": 0, "recorded_only": 1}
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
    assert llm2.calls and all("General Judge" in c.system for c in llm2.calls)


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
    assert "kpi accuracy=0.5 judged=2" in result.output
    assert "kpi rules_adherence mean=0 pass_rate=0 n=1" in result.output
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
    draft = CatalogPackDraft(
        tables=[TablePick(table="employees", columns=["full_name"], why="ranking")]
    )
    edited = pipeline.update_reference_pack(RUN_ID, inputs, "RP_SCN_001", draft)
    assert edited.edited and edited.item_names() == ["employees", "employees.full_name"]
    assert edited.tables[0].description  # the catalog's own text, copied by code
    packs = TypeAdapter(list[ReferencePack]).validate_json(
        (run_dir / "reference_packs.json").read_bytes()
    )
    assert packs[0] == edited and not packs[1].edited
    for stale in ("questions.json", "executions.json", "evaluations.json"):
        assert not (run_dir / stale).exists()
    with pytest.raises(InputAssetError, match="unknown table 'x'"):
        pipeline.update_reference_pack(
            RUN_ID,
            inputs,
            "RP_SCN_002",
            CatalogPackDraft(tables=[TablePick(table="x", why="y")]),
        )
    # the next run regenerates questions and later steps from the edited pack
    pipeline2, llm2, _ = _pipeline(settings)
    again = await pipeline2.run(inputs, run_id=RUN_ID)
    assert again.steps_run == [Step.QUESTIONS, Step.EXECUTE, Step.EVALUATE]
    assert "Employee full name." in llm2.calls[0].user


async def test_rerun_appends_to_the_run_log_and_it_can_be_off(
    settings: AppConfig, inputs: RunInputs
) -> None:
    pipeline, _llm, _product = _pipeline(settings)
    summary = await pipeline.run(inputs, run_id=RUN_ID)
    log_file = Path(summary.run_dir) / "run.log"
    pipeline, _llm, _product = _pipeline(settings)
    await pipeline.generate_questions(RUN_ID, inputs, force=True)
    assert log_file.read_text(encoding="utf-8").count("step questions STARTED") == 2

    off = settings.model_copy(
        update={"trace": settings.trace.model_copy(update={"enabled": False})}
    )
    size = log_file.stat().st_size
    pipeline, _llm, _product = _pipeline(off)
    await pipeline.execute_questions(RUN_ID, inputs, force=True)
    assert log_file.stat().st_size == size


async def test_execute_resumes_from_a_question_keeping_earlier_records(
    settings: AppConfig, inputs: RunInputs
) -> None:
    pipeline, _, _ = _pipeline(settings)
    await pipeline.run(inputs, run_id=RUN_ID)
    run_dir = settings.paths.artifacts_dir / "runs" / RUN_ID
    first = _load(run_dir, "executions.json", TypeAdapter(list[ExecutionRecord]))

    pipeline2, _, product2 = _pipeline(settings)
    step = await pipeline2.execute_questions(RUN_ID, inputs, start_from_question=2)
    assert len(product2.asked) == 2  # only questions 2 and 3 are sent again
    assert [e.execution_id for e in step.executions] == ["EXE_001", "EXE_002", "EXE_003"]
    assert step.executions[0] == first[0]  # question 1 kept as it was
    with pytest.raises(ArtifactIntegrityError, match="start_from_question must be between"):
        await pipeline2.execute_questions(RUN_ID, inputs, start_from_question=9)


async def test_changing_the_metric_selection_invalidates_the_run(
    settings: AppConfig, inputs: RunInputs
) -> None:
    pipeline, _, _ = _pipeline(settings)
    await pipeline.run(inputs, run_id=RUN_ID)
    other = RunInputs(
        persona=inputs.persona,
        product=inputs.product,
        reference=inputs.reference,
        metric_selection=MetricSelection(selected_metrics=["relevance"]),
    )
    with pytest.raises(ArtifactIntegrityError, match="different metric selection"):
        await pipeline.generate_questions(RUN_ID, other)
    # a full run with the new selection regenerates everything from scenarios
    pipeline2, _, _ = _pipeline(settings)
    summary = await pipeline2.run(other, run_id=RUN_ID)
    assert summary.steps_run == list(Step) and summary.selected_metrics == ["relevance"]


async def test_custom_metrics_compile_once_per_run_and_never_block_others(
    settings: AppConfig, inputs: RunInputs
) -> None:
    custom = RunInputs(
        persona=inputs.persona,
        product=inputs.product,
        reference=inputs.reference,
        metric_selection=MetricSelection(
            selected_metrics=list(settings.evaluation.default_metrics),
            custom_inputs=[
                CustomMetricInput(text="Uses ANSI SQL only", provenance="free text item 1")
            ],
        ),
    )
    pipeline, llm, _ = _pipeline(settings)
    summary = await pipeline.run(custom, run_id=RUN_ID)
    compiler_calls = [c for c in llm.calls if "You compile user-written evaluation" in c.system]
    assert len(compiler_calls) == 1  # one structured compilation call for the whole run
    assert summary.selected_metrics[-1] == "custom_01"
    (failure,) = summary.kpis.compiler_failures
    assert failure.source_text == "Uses ANSI SQL only" and "SQL dialect" in failure.reason
    # the built-in metrics were evaluated as usual; the custom one is reported unavailable
    assert summary.verdicts == {"pass": 1, "fail": 1, "inconclusive": 0, "recorded_only": 1}
    run_dir = Path(summary.run_dir)
    evaluations = _load(run_dir, "evaluations.json", TypeAdapter(list[EvaluationResult]))
    judged = [e for e in evaluations if e.verdict is not None]
    assert all(e.metrics["custom_01"].status == "unavailable" for e in judged)
    # the frozen configuration is reused: a rerun makes no compiler call
    pipeline2, llm2, _ = _pipeline(settings)
    await pipeline2.run(custom, run_id=RUN_ID)
    assert llm2.calls == []


async def test_changing_evaluation_guidance_reruns_only_evaluation(
    settings: AppConfig, inputs: RunInputs
) -> None:
    """Guidance is not part of the frozen metric selection (D237): changing it re-evaluates,
    never regenerates; the same guidance is reused; manifest + evaluations record it."""
    from dataclasses import replace

    from synthetic_user.core.guidance import TYPED_GUIDANCE, parse_evaluation_guidance

    pipeline, _, _ = _pipeline(settings)
    await pipeline.run(inputs, run_id=RUN_ID)
    guidance = parse_evaluation_guidance(
        "Approved: tenant filters.", source=TYPED_GUIDANCE, max_chars=1000
    )
    guided = replace(inputs, evaluation_guidance=guidance)

    pipeline2, llm2, product2 = _pipeline(settings)
    summary = await pipeline2.run(guided, run_id=RUN_ID)
    assert summary.steps_run == [Step.EVALUATE] and product2.asked == []
    # ONE structuring call for this guidance version, then the judges with its items
    structuring, *judges = llm2.calls
    assert "You structure evaluation guidance" in structuring.system
    assert judges and all("## EVALUATION GUIDANCE" in c.user for c in judges)
    assert all("G1 [exception; metrics: all]" in c.user for c in judges)
    run_dir = settings.paths.artifacts_dir / "runs" / RUN_ID
    manifest = _load(run_dir, "manifest.json", TypeAdapter(RunManifest))
    assert guidance is not None and manifest.evaluation_guidance is not None
    assert manifest.evaluation_guidance.sha256 == guidance.sha256
    assert [i.guidance_id for i in manifest.evaluation_guidance.items] == ["G1"]
    assert manifest.evaluation_guidance.items[0].text == "Approved: tenant filters."
    evaluations = _load(run_dir, "evaluations.json", TypeAdapter(list[EvaluationResult]))
    assert {e.evaluation_guidance_sha256 for e in evaluations} == {guidance.sha256}

    pipeline3, llm3, _ = _pipeline(settings)
    again = await pipeline3.run(guided, run_id=RUN_ID)
    assert again.steps_reused == list(Step) and llm3.calls == []


async def test_first_evaluation_with_guidance_is_not_logged_as_a_change(
    settings: AppConfig, inputs: RunInputs, caplog: pytest.LogCaptureFixture
) -> None:
    from dataclasses import replace

    from synthetic_user.core.guidance import TYPED_GUIDANCE, parse_evaluation_guidance

    guidance = parse_evaluation_guidance("Approved: x.", source=TYPED_GUIDANCE, max_chars=100)
    pipeline, _, _ = _pipeline(settings)
    with caplog.at_level("INFO"):
        await pipeline.run(replace(inputs, evaluation_guidance=guidance), run_id=RUN_ID)
    assert "Evaluation guidance applied to every judge call" in caplog.text  # captured
    assert "Evaluation guidance changed" not in caplog.text


async def test_rerunning_questions_passes_the_earlier_ones_as_diversity_context(
    settings: AppConfig, inputs: RunInputs
) -> None:
    """D239: re-running the questions step (same scenarios) gives each scenario's call its
    earlier questions; a repeat of one of them is a near-duplicate, never silently kept."""
    pipeline, _, _ = _pipeline(settings)
    await pipeline.run(inputs, run_id=RUN_ID)
    first = _load(
        settings.paths.artifacts_dir / "runs" / RUN_ID,
        "questions.json",
        TypeAdapter(list[QuestionCase]),
    )
    pipeline2, llm2, _ = _pipeline(settings)
    step = await pipeline2.generate_questions(RUN_ID, inputs, force=True)
    prompts = [c.user for c in llm2.calls if "You role-play" in c.system]
    assert prompts and all("Questions already generated for this scenario" in p for p in prompts)
    assert all(q.utterance in " ".join(prompts) for q in first)
    # the offline script repeats its questions: rejected as near-duplicates with the reason
    assert not step.questions
    assert all(r.reasons[0].startswith("near-duplicate of earlier question") for r in step.rejected)
