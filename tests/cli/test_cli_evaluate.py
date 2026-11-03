from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import TypeAdapter
from typer.testing import CliRunner

from synthetic_user import cli
from synthetic_user.cli import app
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.models.execution import ExecutionRecord
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.run import GenerationMode, RunManifest
from synthetic_user.models.scenario import Scenario
from synthetic_user.storage.json_store import JsonArtifactStore
from synthetic_user.storage.runs import run_artifact, save_run_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "RUN_20261013_120000_abcd"
TEST_METRICS = ("relevance", "correctness_completeness", "semantic_correctness")


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
    data = yaml.safe_load((REPO_ROOT / "config" / "app.yaml").read_text(encoding="utf-8"))
    data["paths"]["artifacts_dir"] = str(tmp_path / "artifacts")
    data["evaluation"]["default_metrics"] = list(TEST_METRICS)
    data["inputs"] = {
        key: (
            [str(REPO_ROOT / x) for x in value]
            if isinstance(value, list)
            else str(REPO_ROOT / value)
        )
        for key, value in data["inputs"].items()
    }
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _questions(scenarios: list[Scenario], pack: ReferencePack | None = None) -> list[QuestionCase]:
    """Grounded questions; ``pack`` is the cheat sheet copied into success_criteria (D240)."""
    return [
        QuestionCase.model_validate(
            {
                "question_id": f"Q_00{i}",
                "scenario_id": scenarios[0].id,
                "persona_id": "finance_controller",
                "mode": "grounded",
                "reference_pack_id": scenarios[0].reference_pack_id,
                "utterance": text,
                "intended_goal": scenarios[0].business_goal,
                "intent": {"task_type": "ranking"},
                "question_characteristics": {
                    "style": "precise",
                    "ambiguity_level": "low",
                    "complexity": "simple",
                },
                "expectations": {
                    "expected_behavior": "Return a query ranking the top spenders.",
                    "success_criteria": {
                        "general": ["names the top spenders"],
                        "cheat_sheet": pack.model_dump(mode="json") if pack else None,
                    },
                },
            }
        )
        for i, text in enumerate(
            ["Top card spenders?", "Biggest spenders by currency?", "Unasked"], 1
        )
    ]


def _executions() -> list[ExecutionRecord]:
    common = {
        "product_id": "sample_nl2sql",
        "adapter": "fixture",
        "submitted_question": "q",
        "started_at": datetime(2026, 10, 13, tzinfo=UTC),
    }
    return [
        ExecutionRecord.model_validate(
            {
                "execution_id": "EXE_001",
                "question_id": "Q_001",
                "response": {
                    "outcome": "answered",
                    "artifacts": [
                        {"kind": "sql", "content": "SELECT e.full_name FROM employees e"}
                    ],
                },
                **common,
            }
        ),
        ExecutionRecord.model_validate(
            {
                "execution_id": "EXE_002",
                "question_id": "Q_002",
                "response": {
                    "outcome": "answered",
                    "artifacts": [{"kind": "sql", "content": "SELEC broken"}],
                },
                **common,
            }
        ),
    ]


@pytest.fixture
def judge(judge_factory: Callable[..., Any]) -> Any:
    return judge_factory(fail_on="SELEC broken")


def _llm(monkeypatch: pytest.MonkeyPatch, judge: Any) -> FakeLLMClient:
    llm = FakeLLMClient([judge] * 10)
    monkeypatch.setattr(cli, "create_llm_client", lambda settings: llm)
    return llm


def _write(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_evaluate_file_mode(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
    judge: Any,
) -> None:
    llm = _llm(monkeypatch, judge)
    q = _write(
        tmp_path / "questions.json",
        [x.model_dump(mode="json") for x in _questions(sample_scenarios, sample_packs[0])],
    )
    e = _write(
        tmp_path / "executions.json",
        TypeAdapter(list[ExecutionRecord]).dump_python(_executions(), mode="json"),
    )
    s = _write(tmp_path / "scenarios.json", [x.model_dump(mode="json") for x in sample_scenarios])
    out = tmp_path / "out"

    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "evaluate",
            "--questions",
            str(q),
            "--executions",
            str(e),
            "--scenarios",
            str(s),
            "--out-dir",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        "evaluated=2 pass=1 fail=1 inconclusive=0 recorded_only=0 bug_candidates=1 unexecuted=1"
        in result.output
    )
    evaluations = json.loads((out / "evaluations.json").read_text(encoding="utf-8"))
    assert [x["verdict"] for x in evaluations] == ["pass", "fail"]
    bugs = json.loads((out / "bug_candidates.json").read_text(encoding="utf-8"))
    assert [(b["bug_id"], b["question_id"], b["execution_id"]) for b in bugs] == [
        ("BUG_001", "Q_002", "EXE_002")
    ]
    assert [x["metrics"]["semantic_correctness"]["score"] for x in evaluations] == [1.0, 0.0]
    kpis = json.loads((out / "kpis.json").read_text(encoding="utf-8"))
    assert kpis["accuracy"] == kpis["task_success_rate"] == 0.5 and kpis["judged"] == 2
    assert (out / "kpis.csv").read_text(encoding="utf-8").startswith("kpi,value,n")
    # one General Judge call per response (no retrieval metric selected); the SQL parser output
    # and the cheat sheet are evidence in that same call
    assert len(llm.calls) == 2
    assert all("General Judge" in c.system for c in llm.calls)
    # the cheat sheet is expected evidence next to the catalog reconciliation (D230)
    assert all('"table": "card_transactions"' in c.user for c in llm.calls)
    assert all("Catalog reconciliation of the SQL" in c.user for c in llm.calls)
    assert all("SQL analysis (parser output" in c.user for c in llm.calls)
    assert "parse failed" in llm.calls[1].user  # "SELEC broken": explicit, not a fail by itself
    config = json.loads((out / "metric_config.json").read_text(encoding="utf-8"))
    assert config["selected_metric_ids"] == list(TEST_METRICS)


def test_grounded_question_without_its_cheat_sheet_is_refused(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_scenarios: list[Scenario],
    judge: Any,
) -> None:
    _llm(monkeypatch, judge)
    q = _write(
        tmp_path / "questions.json",
        [x.model_dump(mode="json") for x in _questions(sample_scenarios)],
    )
    e = _write(
        tmp_path / "executions.json",
        TypeAdapter(list[ExecutionRecord]).dump_python(_executions(), mode="json"),
    )
    s = _write(tmp_path / "scenarios.json", [x.model_dump(mode="json") for x in sample_scenarios])
    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "evaluate",
            *("--questions", str(q), "--executions", str(e), "--scenarios", str(s)),
        ],
    )
    assert result.exit_code == 1
    assert "success_criteria has no cheat_sheet" in result.output


def test_evaluate_run_mode_updates_manifest(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
    sample_reference: Reference,
    judge: Any,
    seed_config: Callable[..., Any],
) -> None:
    _llm(monkeypatch, judge)
    store = JsonArtifactStore(tmp_path / "artifacts")
    store.write_models(run_artifact(RUN_ID, "scenarios.json"), sample_scenarios, Scenario)
    store.write_models(run_artifact(RUN_ID, "reference_packs.json"), sample_packs, ReferencePack)
    store.write_models(
        run_artifact(RUN_ID, "questions.json"),
        _questions(sample_scenarios, sample_packs[0]),
        QuestionCase,
    )
    store.write_with(
        run_artifact(RUN_ID, "executions.json"), _executions(), TypeAdapter(list[ExecutionRecord])
    )
    seed_config(store, RUN_ID, TEST_METRICS)
    save_run_manifest(
        store,
        RunManifest(
            run_id=RUN_ID,
            created_at=datetime(2026, 10, 13, tzinfo=UTC),
            persona_id="finance_controller",
            product_id="sample_nl2sql",
            mode=GenerationMode.GROUNDED,
            reference_id=sample_reference.reference_id,
            llm_provider="fake",
            llm_model="fake",
            pipeline_version="0.1.0",
            effective_config={},
            artifacts={"scenarios": "scenarios.json"},
        ),
    )
    result = CliRunner().invoke(app, ["--config", str(config_path), "evaluate", "--run-id", RUN_ID])
    assert result.exit_code == 0, result.output
    run_dir = tmp_path / "artifacts" / "runs" / RUN_ID
    assert (run_dir / "evaluations.json").is_file()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert {"scenarios", "evaluations", "bug_candidates"} <= set(manifest["artifacts"])


def test_evaluate_legacy_run_without_metric_config_explains_migration(
    config_path: Path,
    tmp_path: Path,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
    sample_reference: Reference,
) -> None:
    store = JsonArtifactStore(tmp_path / "artifacts")
    store.write_models(run_artifact(RUN_ID, "scenarios.json"), sample_scenarios, Scenario)
    store.write_models(run_artifact(RUN_ID, "reference_packs.json"), sample_packs, ReferencePack)
    store.write_models(
        run_artifact(RUN_ID, "questions.json"),
        _questions(sample_scenarios, sample_packs[0]),
        QuestionCase,
    )
    store.write_with(
        run_artifact(RUN_ID, "executions.json"), _executions(), TypeAdapter(list[ExecutionRecord])
    )
    save_run_manifest(
        store,
        RunManifest.model_validate(
            {
                "run_id": RUN_ID,
                "created_at": "2026-10-13T00:00:00Z",
                "persona_id": "finance_controller",
                "product_id": "sample_nl2sql",
                "mode": "grounded",
                "reference_id": sample_reference.reference_id,
                "evaluation_profile": "nl2sql",  # a profile-era manifest
                "llm_provider": "fake",
                "llm_model": "fake",
                "pipeline_version": "0.1.0",
                "effective_config": {},
            }
        ),
    )
    result = CliRunner().invoke(app, ["--config", str(config_path), "evaluate", "--run-id", RUN_ID])
    assert result.exit_code == 1
    assert "created before run-level metric selection" in result.output


def test_evaluate_needs_exactly_one_mode(config_path: Path, tmp_path: Path) -> None:
    neither = CliRunner().invoke(app, ["--config", str(config_path), "evaluate"])
    assert neither.exit_code == 1 and "Pass either --run-id" in neither.output
    partial = CliRunner().invoke(
        app, ["--config", str(config_path), "evaluate", "--questions", str(tmp_path / "q.json")]
    )
    assert partial.exit_code == 1 and "File mode needs all of" in partial.output


def test_execution_for_unknown_question(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_scenarios: list[Scenario],
    judge: Any,
) -> None:
    _llm(monkeypatch, judge)
    q = _write(tmp_path / "questions.json", [])
    e = _write(
        tmp_path / "executions.json",
        TypeAdapter(list[ExecutionRecord]).dump_python(_executions(), mode="json"),
    )
    s = _write(tmp_path / "scenarios.json", [x.model_dump(mode="json") for x in sample_scenarios])
    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "evaluate",
            "--questions",
            str(q),
            "--executions",
            str(e),
            "--scenarios",
            str(s),
        ],
    )
    assert result.exit_code == 1
    assert "refers to unknown question Q_001" in result.output


def test_benchmark_is_live_only(config_path: Path) -> None:
    result = CliRunner().invoke(app, ["--config", str(config_path), "benchmark-evaluator"])
    assert result.exit_code == 1
    assert "live only" in result.output
