"""Wait and try again when a real LLM provider call fails (D264).

The provider SDKs retry quickly on their own; a reasoning model behind a busy gateway still
fails now and then (timeout, connection reset, 429 / 5xx). ``RetryingLLMClient`` then waits
``wait_seconds`` (``asyncio.sleep``: other calls keep running) and repeats the same request, at
most ``retries`` times, before the last error goes on exactly as before. Only
``LLMProviderError`` is retried, and not one marked ``retryable=False`` (a missing package or
config file: waiting cannot fix it). Invalid output is not a provider failure: the bounded
validation retries of ``llm/structured.py`` handle it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from synthetic_user.core.exceptions import LLMProviderError
from synthetic_user.llm.base import LLMClient

_log = logging.getLogger(__name__)
_T = TypeVar("_T")


class RetryingLLMClient:
    def __init__(self, llm: LLMClient, *, retries: int, wait_seconds: float) -> None:
        self._llm = llm
        self._retries = retries
        self._wait_seconds = wait_seconds

    @staticmethod
    def wrap(llm: LLMClient, *, retries: int, wait_seconds: float) -> LLMClient:
        """``llm`` itself when retrying is off (``retries`` 0)."""
        if retries <= 0:
            return llm
        return RetryingLLMClient(llm, retries=retries, wait_seconds=wait_seconds)

    @property
    def provider_name(self) -> str:
        return self._llm.provider_name

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    async def generate_json(self, system: str, user: str) -> dict[str, Any]:
        return await self._with_retries(lambda: self._llm.generate_json(system, user))

    async def generate_text(self, system: str, user: str) -> str:
        return await self._with_retries(lambda: self._llm.generate_text(system, user))

    async def _with_retries(self, call: Callable[[], Awaitable[_T]]) -> _T:
        attempt = 0
        while True:
            try:
                return await call()
            except LLMProviderError as exc:
                if not exc.retryable or attempt >= self._retries:
                    raise
                attempt += 1
                _log.warning(
                    "LLM call failed (%s); waiting %.0f s, then retry %d of %d",
                    exc,
                    self._wait_seconds,
                    attempt,
                    self._retries,
                )
                await asyncio.sleep(self._wait_seconds)
