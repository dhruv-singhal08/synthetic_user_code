"""FakeLLMClient: scripted, offline LLM for tests."""

from __future__ import annotations

import copy
import json
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from synthetic_user.core.exceptions import LLMProviderError
from synthetic_user.llm.json_utils import extract_json_object

ResponseValue = dict[str, Any] | str
FakeResponse = ResponseValue | Exception | Callable[[str, str], ResponseValue]


@dataclass(frozen=True)
class FakeCall:
    method: str
    system: str
    user: str


class FakeLLMClient:
    """Pops one queued response per call.

    Queue items: a dict (JSON object), a str (raw text; parsed like a real adapter for
    generate_json), an Exception (raised), or a callable ``(system, user) -> dict | str``.
    """

    def __init__(
        self,
        responses: Iterable[FakeResponse] = (),
        *,
        provider_name: str = "fake",
        model_name: str = "fake",
    ) -> None:
        self._queue: deque[FakeResponse] = deque(responses)
        self._provider_name = provider_name
        self._model_name = model_name
        self.calls: list[FakeCall] = []

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def queue(self, *responses: FakeResponse) -> None:
        self._queue.extend(responses)

    @property
    def remaining(self) -> int:
        return len(self._queue)

    async def generate_json(self, system: str, user: str) -> dict[str, Any]:
        value = self._next("generate_json", system, user)
        if isinstance(value, str):
            return extract_json_object(value)
        return copy.deepcopy(value)

    async def generate_text(self, system: str, user: str) -> str:
        value = self._next("generate_text", system, user)
        return value if isinstance(value, str) else json.dumps(value)

    def _next(self, method: str, system: str, user: str) -> ResponseValue:
        self.calls.append(FakeCall(method, system, user))
        if not self._queue:
            raise LLMProviderError(
                f"FakeLLMClient queue is empty: call #{len(self.calls)} ({method}) "
                "has no scripted response"
            )
        item = self._queue.popleft()
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(system, user)
        return item
