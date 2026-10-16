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

REPO_ROOT = Path(__file__).resolve().parents[2]


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


@pytest.fixture
def scripted_llm(monkeypatch: pytest.MonkeyPatch, sample_draft: dict[str, Any]) -> list[int]:
    created: list[int] = []

    def factory(settings: object) -> FakeLLMClient:
        created.append(1)
        return FakeLLMClient([sample_draft])

    monkeypatch.setattr(cli, "create_llm_client", factory)
    return created


def _fields(output: str) -> dict[str, str]:
    line = output.strip().splitlines()[-1]
    return dict(part.split("=", 1) for part in line.split())


def test_build_then_reuse_then_force(
    config_path: Path, tmp_path: Path, scripted_llm: list[int]
) -> None:
    runner = CliRunner()
    first = runner.invoke(app, ["--config", str(config_path), "build-index"])
    assert first.exit_code == 0, first.output
    built = _fields(first.output)
    assert built["reused"] == "false"
    assert built["tables"] == "4"
    stored = Path(built["path"])
    assert stored.parent == tmp_path / "artifacts" / "reference_indexes" / "sample_card_spend_v1.0"

    second = runner.invoke(app, ["--config", str(config_path), "build-index"])
    assert second.exit_code == 0, second.output
    assert _fields(second.output)["reused"] == "true"
    assert _fields(second.output)["index_id"] == built["index_id"]
    assert len(scripted_llm) == 1

    forced = runner.invoke(app, ["--config", str(config_path), "build-index", "--force"])
    assert forced.exit_code == 0, forced.output
    assert _fields(forced.output)["reused"] == "false"
    assert len(scripted_llm) == 2


def test_out_and_explicit_catalog(
    config_path: Path, tmp_path: Path, scripted_llm: list[int]
) -> None:
    out = tmp_path / "exports" / "index.json"
    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "build-index",
            "--catalog",
            str(REPO_ROOT / "data" / "references" / "sample_catalog.yaml"),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert Path(_fields(result.output)["path"]) == out
    assert json.loads(out.read_text(encoding="utf-8"))["source_catalog_id"] == "sample_card_spend"


def test_missing_catalog_exits_1(config_path: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["--config", str(config_path), "build-index", "--catalog", str(tmp_path / "x.yaml")]
    )
    assert result.exit_code == 1
    assert "Catalog file not found" in result.output


def test_integrity_failure_exits_1(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, sample_draft: dict[str, Any]
) -> None:
    sample_draft["tables"][0]["important_columns"].append("amount_usd")
    monkeypatch.setattr(
        cli, "create_llm_client", lambda settings: FakeLLMClient([sample_draft] * 3)
    )
    result = CliRunner().invoke(app, ["--config", str(config_path), "build-index"])
    assert result.exit_code == 1
    assert "amount_usd" in result.output
