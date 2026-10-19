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
from synthetic_user.models.reference import Reference

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "RUN_20261013_120000_abcd"
SAMPLE_REFERENCE = REPO_ROOT / "data" / "references" / "sample_card_spend_catalog.yaml"
POLICY = REPO_ROOT / "data" / "references" / "sample_travel_policy_docs.yaml"
CHATBOT = REPO_ROOT / "data" / "products" / "sample_chatbot.yaml"


def _config(tmp_path: Path, *, with_index: bool) -> Path:
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
    if not with_index:
        del data["inputs"]["reference_file"], data["inputs"]["reference_guide_file"]
    path = tmp_path / ("app.yaml" if with_index else "app.ungrounded.yaml")
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    return _config(tmp_path, with_index=True)


@pytest.fixture
def ungrounded_config(tmp_path: Path) -> Path:
    return _config(tmp_path, with_index=False)


def _script(monkeypatch: pytest.MonkeyPatch, *responses: Any) -> None:
    monkeypatch.setattr(cli, "create_llm_client", lambda settings: FakeLLMClient(list(responses)))


def _fields(output: str) -> dict[str, str]:
    return dict(p.split("=", 1) for p in output.strip().splitlines()[-1].split())


def _ungrounded(draft: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in draft.items() if k != "reference_pack"}


def test_writes_scenarios_rejected_and_manifest(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_reference: Reference,
    scenario_drafts: list[dict[str, Any]],
) -> None:
    bad = json.loads(json.dumps(scenario_drafts[1]))
    bad["reference_pack"] = {"tables": [{"table": "budgets", "why": "x"}]}
    _script(
        monkeypatch, {"scenarios": [scenario_drafts[0], bad, scenario_drafts[2]]}, {"scenarios": []}
    )

    result = CliRunner().invoke(
        app,
        ["--config", str(config_path), "generate-scenarios", "--run-id", RUN_ID, "--count", "3"],
    )
    assert result.exit_code == 0, result.output
    fields = _fields(result.output)
    assert (fields["mode"], fields["accepted"], fields["rejected"], fields["duplicates"]) == (
        "grounded",
        "2",
        "1",
        "0",
    )

    run_dir = tmp_path / "artifacts" / "runs" / RUN_ID
    scenarios = json.loads((run_dir / "scenarios.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in scenarios] == ["SCN_001", "SCN_002"]
    rejected = json.loads((run_dir / "scenarios_rejected.json").read_text(encoding="utf-8"))
    assert rejected[0]["reasons"][0] == (
        "reference pack names no table / document that is in the reference"
    )
    packs = json.loads((run_dir / "reference_packs.json").read_text(encoding="utf-8"))
    assert [p["pack_id"] for p in packs] == ["RP_SCN_001", "RP_SCN_002"]
    assert fields["reference_packs"] == "2"

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == RUN_ID
    assert manifest["mode"] == "grounded"
    assert manifest["reference_id"] == sample_reference.reference_id
    assert manifest["artifacts"] == {
        "scenarios": "scenarios.json",
        "scenarios_rejected": "scenarios_rejected.json",
        "reference_packs": "reference_packs.json",
    }
    assert "secrets" not in manifest["effective_config"]


def test_explicit_inputs_reference_and_new_run_id(
    ungrounded_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario_drafts: list[dict[str, Any]],
) -> None:
    _script(monkeypatch, {"scenarios": [scenario_drafts[1]]})
    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(ungrounded_config),
            "generate-scenarios",
            "--persona",
            str(REPO_ROOT / "data" / "personas" / "finance_controller.yaml"),
            "--product",
            str(REPO_ROOT / "data" / "products" / "sample_product.yaml"),
            "--reference",
            str(SAMPLE_REFERENCE),
            "--count",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    fields = _fields(result.output)
    assert fields["run_id"].startswith("RUN_") and fields["mode"] == "grounded"
    assert (tmp_path / "artifacts" / "runs" / fields["run_id"] / "scenarios.json").is_file()
    assert "instructions: default" in result.output  # none given, none in the config


def test_no_index_is_ungrounded_with_chatbot_product(
    ungrounded_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario_drafts: list[dict[str, Any]],
) -> None:
    _script(monkeypatch, {"scenarios": [_ungrounded(scenario_drafts[1])]})
    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(ungrounded_config),
            "generate-scenarios",
            "--run-id",
            RUN_ID,
            "--product",
            str(CHATBOT),
            "--count",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "mode ungrounded" in result.output  # logged
    fields = _fields(result.output)
    assert (fields["mode"], fields["accepted"]) == ("ungrounded", "1")
    run_dir = tmp_path / "artifacts" / "runs" / RUN_ID
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["mode"], manifest["reference_id"]) == ("ungrounded", None)
    assert manifest["product_id"] == "sample_policy_chatbot"
    assert not (run_dir / "reference_packs.json").exists()
    scenario = json.loads((run_dir / "scenarios.json").read_text(encoding="utf-8"))[0]
    assert scenario["mode"] == "ungrounded" and scenario["reference_pack_id"] is None


def test_missing_reference_file_is_an_error(config_path: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "generate-scenarios",
            "--reference",
            str(tmp_path / "missing.json"),
        ],
    )
    assert result.exit_code == 1
    assert "Reference not found" in result.output


def test_check_reference_prints_id_and_size(config_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "check-reference",
            "--reference",
            str(POLICY),
            "--reference-guide",
            str(POLICY.with_name("sample_travel_policy_instructions.md")),
        ],
    )
    assert result.exit_code == 0, result.output
    fields = _fields(result.output)
    assert fields["kind"] == "document_collection" and fields["documents"] == "4"
    assert fields["max_chars"] == "200000"
    assert fields["reference_id"].startswith("sample_travel_policy_docs-")
    assert fields["instructions"] == "sample_travel_policy_instructions.md"


def test_bad_run_id(config_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["--config", str(config_path), "generate-scenarios", "--run-id", "../evil"]
    )
    assert result.exit_code == 1
    assert "--run-id must look like" in result.output


def test_convert_reference_writes_a_valid_catalog(config_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "catalog.yaml"
    source = REPO_ROOT / "tests" / "fixtures" / "references" / "refidx_sample.json"
    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "convert-reference",
            "--input",
            str(source),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "tables=4 relationships=2 rules=1" in result.output
    check = CliRunner().invoke(
        app, ["--config", str(config_path), "check-reference", "--reference", str(out)]
    )
    assert check.exit_code == 0 and "kind=data_catalog" in check.output
    bad = CliRunner().invoke(
        app,
        ["--config", str(config_path), "convert-reference", "--input", str(out), "--out", str(out)],
    )
    assert bad.exit_code == 1 and "Cannot read reference index" in bad.output
