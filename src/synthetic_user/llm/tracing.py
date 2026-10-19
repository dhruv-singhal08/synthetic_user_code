"""LLMClient wrapper that writes every call to the active run log (core/trace.py, D202).

Calls pass through unchanged; with no active trace nothing is recorded.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from synthetic_user.core.exceptions import LLMOutputValidationError, SyntheticUserError
from synthetic_user.core.trace import record_llm_call, tracing
from synthetic_user.llm.base import LLMClient

T = TypeVar("T")


class TracingLLMClient:
    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    @staticmethod
    def wrap(llm: LLMClient) -> LLMClient:
        return llm if isinstance(llm, TracingLLMClient) else TracingLLMClient(llm)

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    async def generate_json(self, system: str, user: str) -> dict[str, Any]:
        return await self._call("generate_json", self._inner.generate_json, system, user)

    async def generate_text(self, system: str, user: str) -> str:
        return await self._call("generate_text", self._inner.generate_text, system, user)

    async def _call(
        self, method: str, call: Callable[[str, str], Awaitable[T]], system: str, user: str
    ) -> T:
        if not tracing():
            return await call(system, user)
        start = time.perf_counter()
        base = {
            "method": method,
            "model": f"{self.provider_name}/{self.model_name}",
            "system": system,
            "user": user,
        }
        try:
            output = await call(system, user)
        except LLMOutputValidationError as exc:
            record_llm_call(
                **base,
                latency_ms=_ms(start),
                output=exc.raw_output,
                error="; ".join(exc.errors) or str(exc),
            )
            raise
        except SyntheticUserError as exc:
            record_llm_call(**base, latency_ms=_ms(start), output=None, error=str(exc))
            raise
        record_llm_call(**base, latency_ms=_ms(start), output=output, error=None)
        return output


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
