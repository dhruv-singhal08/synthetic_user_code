from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from synthetic_user import cli
from synthetic_user.cli import app
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.models.reference import ReferenceIndex
from synthetic_user.storage.json_store import JsonArtifactStore

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "RUN_20261013_120000_abcd"


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_MODEL", "CONFIG_PATH", "LOG_LEVEL"):
        monkeypatch.delenv(var, raising=False)
    data = yaml.safe_load((REPO_ROOT / "config" / "app.yaml").read_text(encoding="utf-8"))
    data["paths"]["artifacts_dir"] = str(tmp_path / "artifacts")
    data["inputs"] = {key: str(REPO_ROOT / value) for key, value in data["inputs"].items()}
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def stored_index(tmp_path: Path, sample_index: ReferenceIndex) -> ReferenceIndex:
    JsonArtifactStore(tmp_path / "artifacts").write_reference_index(sample_index)
    return sample_index


def _script(monkeypatch: pytest.MonkeyPatch, *responses: Any) -> None:
    monkeypatch.setattr(cli, "create_llm_client", lambda settings: FakeLLMClient(list(responses)))


def _fields(output: str) -> dict[str, str]:
    return dict(p.split("=", 1) for p in output.strip().splitlines()[-1].split())


def test_writes_scenarios_rejected_and_manifest(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stored_index: ReferenceIndex,
    scenario_drafts: list[dict[str, Any]],
) -> None:
    bad = json.loads(json.dumps(scenario_drafts[1]))
    bad["cheat_sheet"]["tables"].append("budgets")
    _script(
        monkeypatch, {"scenarios": [scenario_drafts[0], bad, scenario_drafts[2]]}, {"scenarios": []}
    )

    result = CliRunner().invoke(
        app,
        ["--config", str(config_path), "generate-scenarios", "--run-id", RUN_ID, "--count", "3"],
    )
    assert result.exit_code == 0, result.output
    fields = _fields(result.output)
    assert (fields["accepted"], fields["rejected"], fields["duplicates"]) == ("2", "1", "0")

    run_dir = tmp_path / "artifacts" / "runs" / RUN_ID
    scenarios = json.loads((run_dir / "scenarios.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in scenarios] == ["SCN_001", "SCN_002"]
    rejected = json.loads((run_dir / "scenarios_rejected.json").read_text(encoding="utf-8"))
    assert rejected[0]["reasons"] == ["table 'budgets' is not in the reference index"]

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == RUN_ID
    assert manifest["reference_index_id"] == stored_index.index_id
    assert manifest["catalog_id"] == "sample_card_spend"
    assert manifest["artifacts"] == {
        "scenarios": "scenarios.json",
        "scenarios_rejected": "scenarios_rejected.json",
    }
    assert "secrets" not in manifest["effective_config"]


def test_explicit_inputs_and_new_run_id(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_index: ReferenceIndex,
    scenario_drafts: list[dict[str, Any]],
) -> None:
    index_file = tmp_path / "my_index.json"
    index_file.write_text(sample_index.model_dump_json(), encoding="utf-8")
    _script(monkeypatch, {"scenarios": [scenario_drafts[1]]})
    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "generate-scenarios",
            "--persona",
            str(REPO_ROOT / "data" / "personas" / "finance_controller.yaml"),
            "--product",
            str(REPO_ROOT / "data" / "products" / "sample_product.yaml"),
            "--index",
            str(index_file),
            "--count",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    run_id = _fields(result.output)["run_id"]
    assert run_id.startswith("RUN_")
    assert (tmp_path / "artifacts" / "runs" / run_id / "scenarios.json").is_file()


def test_no_index_tells_user_to_build_one(config_path: Path) -> None:
    result = CliRunner().invoke(app, ["--config", str(config_path), "generate-scenarios"])
    assert result.exit_code == 1
    assert "run `synthetic-user build-index` first" in result.output


def test_bad_run_id(config_path: Path, stored_index: ReferenceIndex) -> None:
    result = CliRunner().invoke(
        app, ["--config", str(config_path), "generate-scenarios", "--run-id", "../evil"]
    )
    assert result.exit_code == 1
    assert "--run-id must look like" in result.output
