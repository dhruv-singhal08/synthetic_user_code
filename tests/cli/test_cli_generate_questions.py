from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from synthetic_user import cli
from synthetic_user.cli import app
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.models.reference import ReferenceIndex
from synthetic_user.models.run import RunManifest
from synthetic_user.models.scenario import Scenario
from synthetic_user.storage.json_store import JsonArtifactStore
from synthetic_user.storage.runs import run_artifact, save_run_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "RUN_20261013_120000_abcd"


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
    data["inputs"] = {key: str(REPO_ROOT / value) for key, value in data["inputs"].items()}
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _seed_run(
    tmp_path: Path, scenarios: list[Scenario], index: ReferenceIndex, persona_id: str
) -> None:
    store = JsonArtifactStore(tmp_path / "artifacts")
    store.write_models(run_artifact(RUN_ID, "scenarios.json"), scenarios[:2], Scenario)
    save_run_manifest(
        store,
        RunManifest(
            run_id=RUN_ID,
            created_at=datetime(2026, 10, 13, tzinfo=UTC),
            persona_id=persona_id,
            product_id="sample_nl2sql",
            catalog_id="sample_card_spend",
            catalog_version="1.0",
            reference_index_id=index.index_id,
            llm_provider="fake",
            llm_model="fake",
            pipeline_version="0.1.0",
            effective_config={},
            artifacts={"scenarios": "scenarios.json"},
        ),
    )


def _q(utterance: str) -> dict[str, Any]:
    return {
        "utterance": utterance,
        "intent": {"task_type": "ranking"},
        "question_characteristics": {"style": "precise", "ambiguity_level": "low"},
        "expectations": {"clarification": "not_expected", "acceptable_behaviors": ["answers"]},
    }


def test_writes_snapshots_questions_and_rejections(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_scenarios: list[Scenario],
    sample_index: ReferenceIndex,
) -> None:
    _seed_run(tmp_path, sample_scenarios, sample_index, "finance_controller")
    responses = [
        {"questions": [_q("Who are our biggest card spenders?")]},
        {"questions": [_q("How is spend per category trending?"), _q("Extra one?")]},
    ]
    monkeypatch.setattr(cli, "create_llm_client", lambda settings: FakeLLMClient(responses))

    result = CliRunner().invoke(
        app,
        ["--config", str(config_path), "generate-questions", "--run-id", RUN_ID, "--count", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "questions=2 rejected=1" in result.output

    run_dir = tmp_path / "artifacts" / "runs" / RUN_ID
    snapshots = json.loads((run_dir / "snapshots.json").read_text(encoding="utf-8"))
    assert [s["snapshot_id"] for s in snapshots] == ["SNAP_SCN_001", "SNAP_SCN_002"]
    questions = json.loads((run_dir / "questions.json").read_text(encoding="utf-8"))
    assert [(x["question_id"], x["scenario_id"]) for x in questions] == [
        ("Q_001", "SCN_001"),
        ("Q_002", "SCN_002"),
    ]
    rejected = json.loads((run_dir / "questions_rejected.json").read_text(encoding="utf-8"))
    assert rejected[0]["reasons"][0].startswith("surplus")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]) == {
        "scenarios",
        "snapshots",
        "questions",
        "questions_rejected",
    }


def test_inputs_must_match_run(
    config_path: Path,
    tmp_path: Path,
    sample_scenarios: list[Scenario],
    sample_index: ReferenceIndex,
) -> None:
    _seed_run(tmp_path, sample_scenarios, sample_index, "someone_else")
    result = CliRunner().invoke(
        app, ["--config", str(config_path), "generate-questions", "--run-id", RUN_ID]
    )
    assert result.exit_code == 1
    assert "do not match run" in result.output


def test_unknown_run(config_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["--config", str(config_path), "generate-questions", "--run-id", RUN_ID]
    )
    assert result.exit_code == 1
    assert "Artifact not found" in result.output
