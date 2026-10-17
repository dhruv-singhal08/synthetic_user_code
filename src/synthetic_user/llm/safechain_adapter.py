"""SafeChain (Amex) via ``safechain==1.5.0``. The SDK is imported lazily on first call.

SafeChain reads its own YAML (IDaaS + ``models`` keyed by id, e.g. ``"1"`` -> gpt-5.2) from the
``CONFIG_PATH`` environment variable, which is also this app's config-path variable. The app
config is loaded before any LLM call, so the adapter points ``CONFIG_PATH`` at the SafeChain
YAML just before importing the SDK (D117). IDaaS credentials (the env var names in that YAML,
e.g. CIBIS_CONSUMER_SECRET) are read by SafeChain from the environment; never touched here.

Temperature and SDK retries come from the SafeChain YAML ``model_kwargs``, not ``llm.temperature``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from synthetic_user.core.exceptions import LLMProviderError
from synthetic_user.llm.json_utils import extract_json_object

INSTALL_HINT = "poetry install  (installs safechain==1.5.0; Amex Artifactory only)"
SAFECHAIN_CONFIG_ENV = "CONFIG_PATH"  # name SafeChain itself reads

_log = logging.getLogger(__name__)


class SafeChainClient:
    def __init__(
        self,
        *,
        model: str,
        config_path: Path,
        timeout_seconds: float,
    ) -> None:
        self._model_id = model
        self._config_path = config_path
        self._timeout_seconds = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "safechain"

    @property
    def model_name(self) -> str:
        return self._model_id

    async def generate_json(self, system: str, user: str) -> dict[str, Any]:
        return extract_json_object(await self.generate_text(system, user))

    async def generate_text(self, system: str, user: str) -> str:
        try:
            return await asyncio.wait_for(self._invoke(system, user), self._timeout_seconds)
        except LLMProviderError:
            raise
        except Exception as exc:  # SDK raises many unrelated types; normalize them all.
            hint = (
                " - SafeChain read a config without `models`: remove CONFIG_PATH from .env "
                f"(SafeChain must read {self._config_path}) and check that file has `models:`"
                if "No models defined" in str(exc)
                else ""
            )
            raise LLMProviderError(
                f"SafeChain call failed (model id {self._model_id}): "
                f"{type(exc).__name__}: {exc}{hint}"
            ) from exc

    async def _invoke(self, system: str, user: str) -> str:
        # The model is built per call (as in the proven Amex smoke script): the SDK's async
        # session must belong to the running event loop, and the web UI runs several loops.
        sdk = _import_sdk(self._config_path)
        model = sdk.LCELRetry(await sdk.amodel(self._model_id), log_func=_log)
        prompt = sdk.ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{user}")])
        chain = prompt | model | sdk.StrOutputParser()
        result = await chain.ainvoke({"system": system, "user": user})
        if not isinstance(result, str):
            raise LLMProviderError(
                f"SafeChain returned unsupported content type: {type(result).__name__}"
            )
        return result


def _import_sdk(config_path: Path) -> Any:
    if not config_path.is_file():
        raise LLMProviderError(f"SafeChain config file not found: {config_path}")
    os.environ[SAFECHAIN_CONFIG_ENV] = str(config_path.resolve())
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from safechain.lcel.model import amodel
        from safechain.lcel.retry import LCELRetry
    except ImportError as exc:
        raise LLMProviderError(f"safechain is not installed; run `{INSTALL_HINT}`") from exc
    return SimpleNamespace(
        amodel=amodel,
        LCELRetry=LCELRetry,
        ChatPromptTemplate=ChatPromptTemplate,
        StrOutputParser=StrOutputParser,
    )
