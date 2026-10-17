from __future__ import annotations

import json
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
from synthetic_user.models.run import GenerationMode, RunManifest
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluators.plugins.sql.models import SqlChecks
from synthetic_user.references.snapshot import snapshot_id_for
from synthetic_user.storage.json_store import JsonArtifactStore
from synthetic_user.storage.runs import run_artifact, save_run_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "RUN_20261013_120000_abcd"
ALL_PASS = dict.fromkeys(SqlChecks.model_fields, "pass")


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


def _questions(scenarios: list[Scenario]) -> list[QuestionCase]:
    return [
        QuestionCase.model_validate(
            {
                "question_id": f"Q_00{i}",
                "scenario_id": scenarios[0].id,
                "persona_id": "finance_controller",
                "mode": "grounded",
                "reference_snapshot_id": snapshot_id_for(scenarios[0].id),
                "utterance": text,
                "intended_goal": scenarios[0].business_goal,
                "intent": {"task_type": "ranking"},
                "question_characteristics": {"ambiguity_level": "low"},
                "expectations": {
                    "clarification": "conditional",
                    "acceptable_behaviors": ["x"],
                    "success_criteria": ["names the top spenders"],
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


def _judge(system: str, user: str) -> dict[str, Any]:
    """Generic judge approves (criteria copied from the prompt); SQL review approves."""
    if "Success criteria (for goal_completeness)" not in user:
        return {"checks": ALL_PASS, "findings": [], "verdict": "pass"}
    ok = {"status": "pass", "rationale": "fine"}
    block = user.split("Success criteria (for goal_completeness):", 1)[1]
    criteria = json.loads(block.split("Product response", 1)[0])
    return {
        "relevance": ok,
        "goal_completeness": ok,
        "criteria": [{"criterion": c, "status": "met", "rationale": "fine"} for c in criteria],
        "persona_fit": ok,
        "capability_consistency": ok,
        "clarification_handling": {"status": "not_applicable", "rationale": "n/a"},
        "safety_and_honesty": ok,
        "overall_verdict": "pass",
    }


def _llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLMClient:
    llm = FakeLLMClient([_judge] * 10)
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
) -> None:
    llm = _llm(monkeypatch)
    q = _write(
        tmp_path / "questions.json",
        [x.model_dump(mode="json") for x in _questions(sample_scenarios)],
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
    # 2 generic judgements + 1 SQL review; the syntax-error SQL never reaches the SQL LLM
    sql_reviews = [c for c in llm.calls if "Success criteria (for goal_completeness)" not in c.user]
    assert len(llm.calls) == 3 and len(sql_reviews) == 1


def test_evaluate_run_mode_updates_manifest(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_scenarios: list[Scenario],
) -> None:
    _llm(monkeypatch)
    store = JsonArtifactStore(tmp_path / "artifacts")
    store.write_models(run_artifact(RUN_ID, "scenarios.json"), sample_scenarios, Scenario)
    store.write_models(
        run_artifact(RUN_ID, "questions.json"), _questions(sample_scenarios), QuestionCase
    )
    store.write_with(
        run_artifact(RUN_ID, "executions.json"), _executions(), TypeAdapter(list[ExecutionRecord])
    )
    save_run_manifest(
        store,
        RunManifest(
            run_id=RUN_ID,
            created_at=datetime(2026, 10, 13, tzinfo=UTC),
            persona_id="finance_controller",
            product_id="sample_nl2sql",
            mode=GenerationMode.GROUNDED,
            catalog_id="sample_card_spend",
            catalog_version="1.0",
            reference_index_id="IDX_1",
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
) -> None:
    _llm(monkeypatch)
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
