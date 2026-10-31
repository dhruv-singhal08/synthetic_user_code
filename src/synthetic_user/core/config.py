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
    model_validator,
)

from synthetic_user.core.exceptions import ConfigError, format_validation_error
from synthetic_user.core.loaders import read_yaml_mapping
from synthetic_user.core.trace import MAX_FIELD_CHARS
from synthetic_user.models.base import AmbiguityLevel
from synthetic_user.models.evaluation_profile import (
    DEFAULT_AGENT_METRICS,
    AgentMetric,
    normalize_agent_metrics,
)
from synthetic_user.models.question import (
    MAX_DELIVERABLES,
    MAX_SUCCESS_CRITERIA,
    QuestionComplexity,
)

DEFAULT_CONFIG_PATH = Path("config/app.yaml")
CONFIG_PATH_ENV = "CONFIG_PATH"
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"  # alias; GOOGLE_API_KEY wins if both set
GEMINI_MODEL_ENV = "GEMINI_MODEL"
LLM_PROVIDER_ENV = "LLM_PROVIDER"  # overrides llm.provider
SAFECHAIN_MODEL_ENV = "SAFECHAIN_MODEL_ID"  # SafeChain YAML models key, e.g. "1"
# Not CONFIG_PATH: that is this app's config; the adapter hands this path to SafeChain (D117).
SAFECHAIN_CONFIG_ENV = "SAFECHAIN_CONFIG_PATH"
DEFAULT_ARTIFACT_MAX_CHARS = 8_000
# A header value made only of ${ENV_VAR} references (safe to record in the manifest).
_ENV_ONLY = re.compile(r"(?:[^$]*\$\{[A-Za-z_][A-Za-z0-9_]*\})+[^$]*")


class LLMProvider(StrEnum):
    GEMINI = "gemini"
    SAFECHAIN = "safechain"
    FAKE = "fake"


class ProductAdapterType(StrEnum):
    FIXTURE = "fixture"
    HTTP_JSON = "http_json"
    HTTP_SSE = "http_sse"
    PYTHON = "python"


class ClarificationPolicy(StrEnum):
    """How a single-turn product call should handle missing details."""

    ASK = "ask"
    ASSUME = "assume"


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
    # Questions in flight at once. Separate from llm.max_concurrency: a local product often
    # answers one request at a time, and queued requests would use up their timeout (D184).
    max_concurrency: int = Field(default=1, ge=1, le=32)
    # In ask mode the executor answers product clarifications as the persona and continues with
    # one stable session id. In assume mode it appends a transport-only instruction so the
    # product answers with reasonable assumptions. The generated/stored question is unchanged.
    clarification_policy: ClarificationPolicy = ClarificationPolicy.ASSUME
    # SSL on / off for http_json and http_sse products. false = the product's TLS certificate and
    # host name are NOT checked (a test server with a self-signed certificate only). A product
    # contract can also turn it off for itself with ``ssl_verify: false``.
    ssl_verify: bool = True


class ProductAdapterSettings(_Section):
    """Which adapter talks to the product. Options are validated by adapters/registry.py."""

    type: ProductAdapterType
    options: dict[str, Any] = Field(default_factory=dict)
    # Set by load_config when product_adapter names a <product_id>.adapter.yaml file.
    file: Path | None = None


class ReferenceSettings(_Section):
    # The whole reference goes into the prompts, so its size is capped (characters).
    max_chars: int = Field(default=200_000, ge=1_000)


class EvaluationSettings(_Section):
    # Temporary product workaround: omit only the exact cg_id IN ('DUMMY') predicate from the
    # judge's copy of SQL. The original product response and SQL remain stored unchanged.
    ignore_dummy_cg_id_filter: bool = False
    # Per response artifact included in an evaluator prompt. The stored response is unchanged.
    artifact_max_chars: int = Field(default=DEFAULT_ARTIFACT_MAX_CHARS, ge=1_000)
    # Relevance and completion are mandatory. The other Agent metrics can be disabled here and
    # overridden per UI run. Enabled optional metrics become not_applicable when the response
    # does not expose the tool/retrieval information they require.
    agent_metrics: list[AgentMetric] = Field(default_factory=lambda: list(DEFAULT_AGENT_METRICS))

    @model_validator(mode="after")
    def _valid_agent_metrics(self) -> Self:
        normalized = normalize_agent_metrics(self.agent_metrics)
        object.__setattr__(self, "agent_metrics", normalized)
        return self


class TraceSettings(_Section):
    # Append every LLM / product call (prompt in, output out) to runs/<run_id>/run.log (D202).
    enabled: bool = True
    # Per prompt / response field. The marker records the original character count.
    max_field_chars: int = Field(default=MAX_FIELD_CHARS, ge=1_000)


class WebSettings(_Section):
    """Web server defaults and request safety limits; CLI host/port flags override these."""

    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=8_000, ge=1, le=65_535)
    max_upload_bytes: int = Field(default=10_000_000, ge=1)
    max_persona_queries: int = Field(default=50, ge=1)
    recent_runs_limit: int = Field(default=20, ge=1)
    max_step_count: int = Field(default=100, ge=1)
    max_reference_instructions_chars: int = Field(default=50_000, ge=1)
    max_product_url_chars: int = Field(default=2_000, ge=1)


class ScenarioGenerationSettings(_Section):
    scenario_count: int = Field(default=10, ge=1)
    include_unsupported_scenarios: bool = True
    diversity_preference: str = Field(default="balanced", min_length=1)
    max_topup_rounds: int = Field(default=1, ge=0)
    # Bounded representative SME examples sent to scenario generation when uploaded.
    agent_golden_examples: int = Field(default=12, ge=1, le=50)


class QuestionGenerationSettings(_Section):
    question_count: int = Field(default=1, ge=1)
    # Golden-guided runs default to this value and reject larger per-scenario batches.
    agent_golden_max_questions_per_scenario: int = Field(default=2, ge=1)
    # Most relevant SME examples sent to each question-generation call when uploaded.
    agent_golden_examples: int = Field(default=6, ge=1, le=20)
    # Minimum operational breadth for supported golden-guided Agent questions. Complexity
    # overrides can raise these per run; unsupported scenarios are exempt.
    agent_min_tool_calls: int = Field(default=1, ge=0, le=12)
    agent_min_data_sources: int = Field(default=1, ge=0, le=2)
    # Hard policy: persona preferences can shape a question only inside these limits.
    max_deliverables: int = Field(default=2, ge=1, le=MAX_DELIVERABLES)
    max_sources: int = Field(default=1, ge=0, le=2)
    max_success_criteria: int = Field(default=2, ge=1, le=MAX_SUCCESS_CRITERIA)
    max_words: int = Field(default=40, ge=5, le=200)
    allowed_complexities: list[QuestionComplexity] = Field(
        default_factory=lambda: [QuestionComplexity.SIMPLE, QuestionComplexity.MEDIUM],
        min_length=1,
    )
    allow_follow_ups: bool = False
    allow_cross_source_comparison: bool = False
    ambiguity_level: AmbiguityLevel = AmbiguityLevel.LOW
    language: str = Field(default="en", min_length=1)
    allow_vague_questions: bool = False
    reject_reference_leaks: bool = True
    near_duplicate_threshold: float = Field(default=0.8, gt=0.0, le=1.0)


class GenerationSettings(_Section):
    scenario: ScenarioGenerationSettings = Field(default_factory=ScenarioGenerationSettings)
    question: QuestionGenerationSettings = Field(default_factory=QuestionGenerationSettings)


class PathsSettings(_Section):
    artifacts_dir: Path


class InputsSettings(_Section):
    persona_path: Path
    product_path: Path
    # The user's reference file (.json / .txt / .md). None -> ungrounded mode.
    reference_file: Path | None = None
    # Optional usage guide (.md / .txt) telling the LLM how to read the reference.
    reference_guide_file: Path | None = None
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
    reference: ReferenceSettings = Field(default_factory=ReferenceSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    paths: PathsSettings
    inputs: InputsSettings
    secrets: SecretsSettings = Field(default_factory=SecretsSettings)
    trace: TraceSettings = Field(default_factory=TraceSettings)
    web: WebSettings = Field(default_factory=WebSettings)

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
