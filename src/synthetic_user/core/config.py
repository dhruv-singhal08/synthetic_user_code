"""Application settings loaded from YAML, with secrets taken only from the environment.

Config path order: ``--config`` > ``CONFIG_PATH`` env var > ``./config/app.yaml``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from sqlglot.dialects.dialect import Dialect

from synthetic_user.core.exceptions import ConfigError, format_validation_error
from synthetic_user.core.loaders import read_yaml_mapping
from synthetic_user.models.base import AmbiguityLevel

DEFAULT_CONFIG_PATH = Path("config/app.yaml")
CONFIG_PATH_ENV = "CONFIG_PATH"
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"  # alias; GOOGLE_API_KEY wins if both set
GEMINI_MODEL_ENV = "GEMINI_MODEL"
LLM_PROVIDER_ENV = "LLM_PROVIDER"  # overrides llm.provider
SAFECHAIN_MODEL_ENV = "SAFECHAIN_MODEL_ID"  # SafeChain YAML models key, e.g. "1"
# Not CONFIG_PATH: that is this app's config; the adapter hands this path to SafeChain (D117).
SAFECHAIN_CONFIG_ENV = "SAFECHAIN_CONFIG_PATH"
# A header value made only of ${ENV_VAR} references (safe to record in the manifest).
_ENV_ONLY = re.compile(r"(?:[^$]*\$\{[A-Za-z_][A-Za-z0-9_]*\})+[^$]*")


class LLMProvider(StrEnum):
    GEMINI = "gemini"
    SAFECHAIN = "safechain"
    FAKE = "fake"


class ProductAdapterType(StrEnum):
    FIXTURE = "fixture"
    HTTP_JSON = "http_json"
    PYTHON = "python"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LLMSettings(_Section):
    provider: LLMProvider = LLMProvider.FAKE
    model: str = Field(default="fake", min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_validation_retries: int = Field(default=2, ge=0)
    max_concurrency: int = Field(default=4, ge=1)
    # provider "fake" only: JSON rule script so the full pipeline runs offline.
    fake_script_path: Path | None = None
    # provider "safechain" only: SafeChain's own YAML (idaas + models).
    safechain_config_path: Path | None = None

    @model_validator(mode="after")
    def _safechain_requires_config(self) -> Self:
        if self.provider is LLMProvider.SAFECHAIN and self.safechain_config_path is None:
            raise ValueError(
                f"llm.safechain_config_path (or {SAFECHAIN_CONFIG_ENV}) is required "
                "when llm.provider is 'safechain'"
            )
        return self


class ProductSettings(_Section):
    timeout_seconds: float = Field(default=30.0, gt=0)  # per product call (executor)


class ProductAdapterSettings(_Section):
    """Which adapter talks to the product. Options are validated by adapters/registry.py."""

    type: ProductAdapterType
    options: dict[str, Any] = Field(default_factory=dict)
    # Set by load_config when product_adapter names a <product_id>.adapter.yaml file.
    file: Path | None = None


class EvaluationSettings(_Section):
    # Artifact plugins for grounded evaluation (nodes/evaluators/plugins/registry.py).
    plugins: list[str] = Field(default_factory=lambda: ["sql"])


class SQLSettings(_Section):
    # Spanner has no sqlglot dialect; bigquery is the closest supported default.
    dialect: str = "bigquery"

    @field_validator("dialect")
    @classmethod
    def _known_dialect(cls, value: str) -> str:
        try:
            Dialect.get_or_raise(value)
        except ValueError as exc:
            raise ValueError(f"unknown sqlglot dialect: {value!r}") from exc
        return value


class ScenarioGenerationSettings(_Section):
    scenario_count: int = Field(default=10, ge=1)
    allow_multi_table: bool = True
    include_unsupported_scenarios: bool = True
    diversity_preference: str = Field(default="balanced", min_length=1)
    max_topup_rounds: int = Field(default=1, ge=0)


class QuestionGenerationSettings(_Section):
    question_count: int = Field(default=1, ge=1)
    ambiguity_level: AmbiguityLevel = AmbiguityLevel.LOW
    language: str = Field(default="en", min_length=1)
    allow_vague_questions: bool = False
    reject_schema_leaks: bool = True


class GenerationSettings(_Section):
    scenario: ScenarioGenerationSettings = Field(default_factory=ScenarioGenerationSettings)
    question: QuestionGenerationSettings = Field(default_factory=QuestionGenerationSettings)


class PathsSettings(_Section):
    artifacts_dir: Path


class InputsSettings(_Section):
    persona_path: Path
    product_path: Path
    # The user's pre-built reference index JSON. None -> ungrounded mode.
    reference_index_file: Path | None = None
    benchmark_cases_path: Path | None = None
    benchmark_scenarios_path: Path | None = None
    # Product descriptions for benchmark cases (by product_id). Empty -> product_path only.
    benchmark_product_paths: list[Path] = Field(default_factory=list)


class SecretsSettings(_Section):
    google_api_key: SecretStr | None = None


class AppConfig(_Section):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    product: ProductSettings = Field(default_factory=ProductSettings)
    product_adapter: ProductAdapterSettings
    sql: SQLSettings = Field(default_factory=SQLSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    paths: PathsSettings
    inputs: InputsSettings
    secrets: SecretsSettings = Field(default_factory=SecretsSettings)

    @model_validator(mode="after")
    def _provider_has_credentials(self) -> Self:
        if self.llm.provider is LLMProvider.GEMINI and self.secrets.google_api_key is None:
            raise ValueError(
                f"{GOOGLE_API_KEY_ENV} (or {GEMINI_API_KEY_ENV}) must be set "
                "when llm.provider is 'gemini'"
            )
        return self

    def effective_config(self) -> dict[str, Any]:
        """JSON-safe settings for the run manifest. Never includes secrets: literal adapter
        header values are redacted (``${ENV_VAR}`` templates are kept, they hold no secret)."""
        dumped = self.model_dump(mode="json", exclude={"secrets"})
        headers = dumped["product_adapter"]["options"].get("headers")
        if isinstance(headers, dict):
            dumped["product_adapter"]["options"]["headers"] = {
                name: value if isinstance(value, str) and _ENV_ONLY.fullmatch(value) else "***"
                for name, value in headers.items()
            }
        return dumped


def resolve_config_path(cli_path: Path | None, env: Mapping[str, str]) -> Path:
    """Apply the path order: --config > CONFIG_PATH > ./config/app.yaml."""
    if cli_path is not None:
        return cli_path
    env_path = env.get(CONFIG_PATH_ENV, "").strip()
    if env_path:
        return Path(env_path)
    return DEFAULT_CONFIG_PATH


def load_config(cli_path: Path | None = None, env: Mapping[str, str] | None = None) -> AppConfig:
    """Load and validate settings. Raises ConfigError on any problem."""
    environ: Mapping[str, str] = os.environ if env is None else env
    path = resolve_config_path(cli_path, environ)
    raw = read_yaml_mapping(path, ConfigError, kind="Config file")
    if "secrets" in raw:
        raise ConfigError(f"{path}: secrets must come from the environment, not the config file")
    data = {
        **_apply_env_overrides(raw, environ),
        "secrets": _secrets_from_env(environ),
    }
    if isinstance(data.get("product_adapter"), str):
        data["product_adapter"] = _read_adapter_file(Path(data["product_adapter"]))
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config {path}: {format_validation_error(exc)}") from exc


def load_adapter_file(path: Path) -> ProductAdapterSettings:
    """A ``<product_id>.adapter.yaml`` file (e.g. ``execute --adapter-config``) -> settings."""
    try:
        return ProductAdapterSettings.model_validate(_read_adapter_file(path))
    except ValidationError as exc:
        raise ConfigError(
            f"Invalid product adapter file {path}: {format_validation_error(exc)}"
        ) from exc


def _read_adapter_file(path: Path) -> dict[str, Any]:
    """``product_adapter: config/products/<product_id>.adapter.yaml`` -> {type, options, file}."""
    raw = read_yaml_mapping(path, ConfigError, kind="Product adapter file")
    if "file" in raw:
        raise ConfigError(f"{path}: 'file' is set by the loader, not in the adapter file")
    return {**raw, "file": str(path)}


def _apply_env_overrides(raw: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """LLM_PROVIDER overrides llm.provider. Then the active provider's model variable
    (GEMINI_MODEL / SAFECHAIN_MODEL_ID) overrides llm.model, and SAFECHAIN_CONFIG_PATH
    overrides llm.safechain_config_path. Switching provider by env requires its model variable,
    so a model name meant for another provider is never sent.
    """
    llm = raw.get("llm", {})
    if not isinstance(llm, dict):
        return raw
    llm = dict(llm)
    env_provider = env.get(LLM_PROVIDER_ENV, "").strip().lower()
    switched = bool(env_provider) and env_provider != llm.get("provider", LLMProvider.FAKE.value)
    if env_provider:
        llm["provider"] = env_provider
    model_env = {
        LLMProvider.GEMINI.value: GEMINI_MODEL_ENV,
        LLMProvider.SAFECHAIN.value: SAFECHAIN_MODEL_ENV,
    }.get(str(llm.get("provider")))
    model = env.get(model_env, "").strip() if model_env else ""
    if model:
        llm["model"] = model
    elif switched and model_env:
        raise ConfigError(f"{LLM_PROVIDER_ENV}={env_provider} needs {model_env} to be set")
    safechain_config = env.get(SAFECHAIN_CONFIG_ENV, "").strip()
    if safechain_config:
        llm["safechain_config_path"] = safechain_config
    return {**raw, "llm": llm}


def _secrets_from_env(env: Mapping[str, str]) -> dict[str, str | None]:
    google_api_key = (
        env.get(GOOGLE_API_KEY_ENV, "").strip() or env.get(GEMINI_API_KEY_ENV, "").strip()
    )
    return {"google_api_key": google_api_key or None}
