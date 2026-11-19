"""Build the configured LLMClient."""

from __future__ import annotations

import importlib.util

from synthetic_user.core.config import SAFECHAIN_CONFIG_ENV, AppConfig, LLMProvider
from synthetic_user.core.exceptions import ConfigError
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.llm.gemini_adapter import INSTALL_HINT as GEMINI_INSTALL_HINT
from synthetic_user.llm.gemini_adapter import GeminiClient
from synthetic_user.llm.retrying import RetryingLLMClient
from synthetic_user.llm.safechain_adapter import INSTALL_HINT as SAFECHAIN_INSTALL_HINT
from synthetic_user.llm.safechain_adapter import SafeChainClient
from synthetic_user.llm.scripted import ScriptedLLMClient


def create_llm_client(settings: AppConfig) -> LLMClient:
    """The configured client; a real provider (Gemini, SafeChain) waits and retries a failed
    call (llm.provider_retries / provider_retry_wait_seconds, D264), the offline ones do not."""
    llm = settings.llm
    if llm.provider == LLMProvider.FAKE:
        if llm.fake_script_path is not None:
            return ScriptedLLMClient.from_file(llm.fake_script_path, model_name=llm.model)
        return FakeLLMClient(model_name=llm.model)
    if llm.provider == LLMProvider.GEMINI:
        _require_package("langchain_google_genai", "gemini", GEMINI_INSTALL_HINT)
        api_key = settings.secrets.google_api_key
        if api_key is None:
            raise ConfigError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) must be set when llm.provider is 'gemini'"
            )
        return _retrying(
            GeminiClient(
                model=llm.model,
                api_key=api_key,
                timeout_seconds=llm.timeout_seconds,
                temperature=llm.temperature,
            ),
            settings,
        )
    if llm.provider == LLMProvider.SAFECHAIN:
        _require_package("safechain", "safechain", SAFECHAIN_INSTALL_HINT)
        config_path = llm.safechain_config_path
        if config_path is None or not config_path.is_file():
            raise ConfigError(
                f"SafeChain config file not found: {config_path} "
                f"(set llm.safechain_config_path or {SAFECHAIN_CONFIG_ENV})"
            )
        return _retrying(
            SafeChainClient(
                model=llm.model, config_path=config_path, timeout_seconds=llm.timeout_seconds
            ),
            settings,
        )
    raise ConfigError(f"Unknown llm.provider: {llm.provider!r}")


def _retrying(client: LLMClient, settings: AppConfig) -> LLMClient:
    return RetryingLLMClient.wrap(
        client,
        retries=settings.llm.provider_retries,
        wait_seconds=settings.llm.provider_retry_wait_seconds,
    )


def _require_package(module: str, provider: str, install_hint: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise ConfigError(
            f"llm.provider '{provider}' needs package '{module}', which is not installed; "
            f"run `{install_hint}`"
        )
