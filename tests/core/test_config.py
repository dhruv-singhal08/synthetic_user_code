from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from synthetic_user.core.config import (
    DEFAULT_CONFIG_PATH,
    AmbiguityLevel,
    LLMProvider,
    ProductAdapterType,
    load_config,
    resolve_config_path,
)
from synthetic_user.core.exceptions import ConfigError

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "app.yaml"


def _minimal() -> dict[str, Any]:
    return {
        "product_adapter": {"type": "fixture", "options": {"path": "fixtures/responses.json"}},
        "paths": {"artifacts_dir": "artifacts"},
        "inputs": {
            "persona_path": "persona.yaml",
            "product_path": "product.yaml",
        },
    }


def _write(tmp_path: Path, data: object, name: str = "app.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- config load -------------------------------------------------------------------------


def test_repo_config_loads_with_documented_defaults() -> None:
    cfg = load_config(REPO_CONFIG, env={})
    assert cfg.llm.provider is LLMProvider.FAKE
    assert cfg.llm.max_validation_retries == 2
    assert cfg.product_adapter.type is ProductAdapterType.FIXTURE
    assert cfg.product.timeout_seconds == 300
    assert cfg.product.max_concurrency == 1
    assert cfg.reference.max_chars == 200_000
    assert cfg.generation.question.ambiguity_level is AmbiguityLevel.LOW
    assert cfg.secrets.google_api_key is None


def test_minimal_config_fills_section_defaults(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _minimal()), env={})
    assert cfg.llm.provider is LLMProvider.FAKE
    assert cfg.llm.temperature is None
    assert cfg.generation.scenario.scenario_count == 10
    assert cfg.inputs.reference_file is None  # optional: absent -> ungrounded
    assert cfg.inputs.reference_guide_file is None


# --- env override / path order ---------------------------------------------------------


def test_cli_path_beats_env_path(tmp_path: Path) -> None:
    cli = tmp_path / "cli.yaml"
    assert resolve_config_path(cli, {"CONFIG_PATH": "env.yaml"}) == cli


def test_env_path_beats_default() -> None:
    assert resolve_config_path(None, {"CONFIG_PATH": "env.yaml"}) == Path("env.yaml")


def test_default_path_when_nothing_set() -> None:
    assert resolve_config_path(None, {}) == DEFAULT_CONFIG_PATH
    assert resolve_config_path(None, {"CONFIG_PATH": "  "}) == DEFAULT_CONFIG_PATH


def test_config_path_env_is_used_by_load_config(tmp_path: Path) -> None:
    data = _minimal()
    data["reference"] = {"max_chars": 5000}
    path = _write(tmp_path, data, "from_env.yaml")
    cfg = load_config(env={"CONFIG_PATH": str(path)})
    assert cfg.reference.max_chars == 5000


def test_api_key_comes_from_env_as_secret(tmp_path: Path) -> None:
    data = _minimal()
    data["llm"] = {"provider": "gemini", "model": "some-model"}
    cfg = load_config(_write(tmp_path, data), env={"GOOGLE_API_KEY": "sk-test-123"})
    assert cfg.secrets.google_api_key is not None
    assert cfg.secrets.google_api_key.get_secret_value() == "sk-test-123"
    assert "sk-test-123" not in repr(cfg)
    assert "sk-test-123" not in cfg.model_dump_json()


def test_gemini_model_env_overrides_yaml_model(tmp_path: Path) -> None:
    data = _minimal()
    data["llm"] = {"provider": "gemini", "model": "yaml-model"}
    env = {"GOOGLE_API_KEY": "k", "GEMINI_MODEL": "env-model"}
    assert load_config(_write(tmp_path, data), env=env).llm.model == "env-model"


def test_gemini_model_env_ignored_for_other_providers(tmp_path: Path) -> None:
    data = _minimal()
    data["llm"] = {"provider": "fake", "model": "fake"}
    cfg = load_config(_write(tmp_path, data), env={"GEMINI_MODEL": "env-model"})
    assert cfg.llm.model == "fake"


def test_llm_provider_env_switches_to_safechain(tmp_path: Path) -> None:
    data = _minimal()
    data["llm"] = {"provider": "gemini", "model": "gem"}
    env = {
        "LLM_PROVIDER": "SafeChain",
        "SAFECHAIN_MODEL_ID": "3",
        "SAFECHAIN_CONFIG_PATH": "./cfg/sc.yaml",
        "GEMINI_MODEL": "ignored",
    }
    llm = load_config(_write(tmp_path, data), env=env).llm
    assert (llm.provider, llm.model) == (LLMProvider.SAFECHAIN, "3")
    assert llm.safechain_config_path == Path("./cfg/sc.yaml")


def test_llm_provider_env_switches_to_gemini(tmp_path: Path) -> None:
    env = {"LLM_PROVIDER": "gemini", "GEMINI_MODEL": "gem-x", "GOOGLE_API_KEY": "k"}
    llm = load_config(_write(tmp_path, _minimal()), env=env).llm
    assert (llm.provider, llm.model) == (LLMProvider.GEMINI, "gem-x")


def test_llm_provider_switch_requires_model_env(tmp_path: Path) -> None:
    env = {"LLM_PROVIDER": "safechain", "SAFECHAIN_CONFIG_PATH": "x.yaml"}
    with pytest.raises(ConfigError, match="SAFECHAIN_MODEL_ID"):
        load_config(_write(tmp_path, _minimal()), env=env)


def test_safechain_requires_config_path(tmp_path: Path) -> None:
    data = _minimal()
    data["llm"] = {"provider": "safechain", "model": "1"}
    with pytest.raises(ConfigError, match="safechain_config_path"):
        load_config(_write(tmp_path, data), env={})


def test_unknown_llm_provider_env_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="provider"):
        load_config(_write(tmp_path, _minimal()), env={"LLM_PROVIDER": "openai"})


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"GEMINI_API_KEY": "gem-key"}, "gem-key"),
        ({"GOOGLE_API_KEY": "goog-key", "GEMINI_API_KEY": "gem-key"}, "goog-key"),
        ({"GOOGLE_API_KEY": "", "GEMINI_API_KEY": "gem-key"}, "gem-key"),
    ],
)
def test_gemini_api_key_alias(tmp_path: Path, env: dict[str, str], expected: str) -> None:
    data = _minimal()
    data["llm"] = {"provider": "gemini", "model": "m"}
    cfg = load_config(_write(tmp_path, data), env=env)
    assert cfg.secrets.google_api_key is not None
    assert cfg.secrets.google_api_key.get_secret_value() == expected


def test_gemini_without_api_key_is_rejected(tmp_path: Path) -> None:
    data = _minimal()
    data["llm"] = {"provider": "gemini", "model": "some-model"}
    with pytest.raises(ConfigError, match="GOOGLE_API_KEY"):
        load_config(_write(tmp_path, data), env={})


def test_secrets_in_yaml_are_rejected(tmp_path: Path) -> None:
    data = _minimal()
    data["secrets"] = {"google_api_key": "sk-leaked"}
    with pytest.raises(ConfigError, match="environment") as exc_info:
        load_config(_write(tmp_path, data), env={})
    assert "sk-leaked" not in str(exc_info.value)


# --- missing / invalid config ----------------------------------------------------------


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml", env={})


def test_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("llm: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path, env={})


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_config(_write(tmp_path, ["a", "b"]), env={})


def test_empty_file_reports_missing_sections(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="paths"):
        load_config(path, env={})


def test_unknown_key_is_forbidden(tmp_path: Path) -> None:
    data = _minimal()
    data["llm"] = {"provider": "fake", "modle": "typo"}
    with pytest.raises(ConfigError, match=r"llm\.modle"):
        load_config(_write(tmp_path, data), env={})


def test_unknown_provider_is_rejected(tmp_path: Path) -> None:
    data = _minimal()
    data["llm"] = {"provider": "openai"}
    with pytest.raises(ConfigError, match=r"llm\.provider"):
        load_config(_write(tmp_path, data), env={})


def test_removed_v2_sections_are_rejected(tmp_path: Path) -> None:
    for section in ({"sql": {"dialect": "bigquery"}}, {"evaluation": {"plugins": ["sql"]}}):
        data = {**_minimal(), **section}
        with pytest.raises(ConfigError, match="Extra inputs are not permitted"):
            load_config(_write(tmp_path, data), env={})


def test_product_adapter_is_required(tmp_path: Path) -> None:
    data = _minimal()
    del data["product_adapter"]
    with pytest.raises(ConfigError, match="product_adapter"):
        load_config(_write(tmp_path, data), env={})


def test_missing_product_adapter_file(tmp_path: Path) -> None:
    data = _minimal()
    data["product_adapter"] = str(tmp_path / "missing.adapter.yaml")
    with pytest.raises(ConfigError, match="Product adapter file"):
        load_config(_write(tmp_path, data), env={})


def test_retries_must_be_non_negative(tmp_path: Path) -> None:
    data = _minimal()
    data["llm"] = {"max_validation_retries": -1}
    with pytest.raises(ConfigError, match="max_validation_retries"):
        load_config(_write(tmp_path, data), env={})


@pytest.mark.parametrize(
    ("name", "env"),
    [
        ("app.yaml", {}),
        ("app.fake.yaml", {}),
        ("app.gemini.yaml", {"GOOGLE_API_KEY": "k", "GEMINI_MODEL": "gemini-test"}),
        ("app.safechain.yaml", {}),
    ],
)
def test_every_shipped_config_loads(name: str, env: dict[str, str]) -> None:
    """The new evaluation / web / question-policy keys are accepted in every config."""
    cfg = load_config(REPO_CONFIG.parent / name, env=env)
    assert cfg.generation.question.max_success_criteria == 6
    assert set(cfg.evaluation.default_metrics) >= {"relevance", "correctness"}
    assert cfg.generation.question.max_parallel_scenarios == 5
    assert cfg.web.max_step_count == 100


def test_safechain_config_keeps_its_run_policy() -> None:
    cfg = load_config(REPO_CONFIG.parent / "app.safechain.yaml", env={})
    assert cfg.product.clarification_policy == "ask"
    # No predicate is hidden from the judge any more (D232); lookups are bounded (D231).
    assert cfg.evaluation.catalog_namespaces is None
    assert cfg.evaluation.reference_lookup.enabled
    assert cfg.evaluation.reference_lookup.max_requests == 10


@pytest.mark.parametrize(
    ("section", "key", "message"),
    [
        ("evaluation", "agent_metrics", "there are no evaluation profiles"),
        ("evaluation", "ignore_dummy_cg_id_filter", "unresolved finding"),
        ("evaluation", "ignored_sql_predicates", "unresolved finding"),
    ],
)
def test_removed_evaluation_keys_are_refused_with_their_replacement(
    tmp_path: Path, section: str, key: str, message: str
) -> None:
    data = _minimal()
    data[section] = {key: True}
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, data), env={})


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("max_topup_rounds", "one LLM call"),
        ("agent_golden_examples", "golden set was removed"),
    ],
)
def test_removed_scenario_keys_are_refused(tmp_path: Path, key: str, message: str) -> None:
    data = _minimal()
    data["generation"] = {"scenario": {key: 1}}
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, data), env={})


def test_removed_question_keys_are_refused(tmp_path: Path) -> None:
    data = _minimal()
    data["generation"] = {"question": {"near_duplicate_threshold": 0.8}}
    with pytest.raises(ConfigError, match="replaced by diversity_check"):
        load_config(_write(tmp_path, data), env={})


def test_default_metrics_must_be_builtin_ids(tmp_path: Path) -> None:
    data = _minimal()
    data["evaluation"] = {"default_metrics": ["completion"]}
    with pytest.raises(ConfigError, match="default_metrics must be distinct built-in"):
        load_config(_write(tmp_path, data), env={})
