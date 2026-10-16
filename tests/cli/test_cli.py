from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from synthetic_user import cli
from synthetic_user.cli import app

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "app.yaml"
SECRET = "sk-must-never-be-printed"


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the repo .env (real keys) out of CLI tests.
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


def _gemini_config(tmp_path: Path) -> Path:
    data = yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))
    data["llm"]["provider"] = "gemini"
    path = tmp_path / "gemini.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_smoke_llm_with_fake_provider() -> None:
    result = CliRunner().invoke(app, ["--config", str(REPO_CONFIG), "smoke-llm"])
    assert result.exit_code == 0, result.output
    assert "provider=fake" in result.output
    assert "model=fake" in result.output
    assert "latency_ms=" in result.output
    assert "status=ok" in result.output


def test_bad_config_path_exits_nonzero(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["--config", str(tmp_path / "nope.yaml"), "smoke-llm"])
    assert result.exit_code == 1
    assert "Config file not found" in result.output


def test_bad_log_level_exits_nonzero() -> None:
    result = CliRunner().invoke(
        app, ["--config", str(REPO_CONFIG), "--log-level", "chatty", "smoke-llm"]
    )
    assert result.exit_code == 1
    assert "Unknown log level" in result.output


ALL_COMMANDS = (
    "smoke-llm",
    "build-index",
    "generate-scenarios",
    "generate-questions",
    "execute",
    "evaluate",
    "run",
    "benchmark-evaluator",
)


@pytest.mark.parametrize("command", ["execute", "generate-questions"])
def test_run_commands_require_existing_run(command: str) -> None:
    result = CliRunner().invoke(
        app, ["--config", str(REPO_CONFIG), command, "--run-id", "RUN_20260101_000000_abcd"]
    )
    assert result.exit_code == 1
    assert "not found" in result.output


def test_unexpected_error_suggests_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(settings: object) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "create_llm_client", explode)
    result = CliRunner().invoke(app, ["--config", str(REPO_CONFIG), "smoke-llm"])
    assert result.exit_code == 1
    assert "Unexpected error: RuntimeError: kaboom" in result.output
    assert "--log-level DEBUG" in result.output


def test_secrets_never_printed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # gemini package is not installed in the dev env -> ConfigError path, with key present.
    monkeypatch.setattr("synthetic_user.llm.factory.importlib.util.find_spec", lambda name: None)
    result = CliRunner().invoke(
        app,
        ["--config", str(_gemini_config(tmp_path)), "--log-level", "DEBUG", "smoke-llm"],
        env={"GOOGLE_API_KEY": SECRET},
    )
    assert result.exit_code == 1
    assert "poetry install --with gemini" in result.output
    assert SECRET not in result.output


def test_help_lists_all_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ALL_COMMANDS:
        assert name in result.output
