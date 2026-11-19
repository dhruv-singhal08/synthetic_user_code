"""A failed real-provider LLM call waits and is tried again (D264)."""

from __future__ import annotations

from typing import Any

import pytest

from synthetic_user.core.exceptions import LLMProviderError
from synthetic_user.llm import retrying
from synthetic_user.llm.factory import create_llm_client
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.llm.retrying import RetryingLLMClient


class Flaky:
    """Fails ``failures`` times with ``error``, then answers."""

    provider_name = "flaky"
    model_name = "m"

    def __init__(self, failures: int, error: LLMProviderError | None = None) -> None:
        self.failures = failures
        self.error = error or LLMProviderError("SafeChain call failed: ReadTimeout")
        self.calls = 0

    async def generate_text(self, system: str, user: str) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return "ok"

    async def generate_json(self, system: str, user: str) -> dict[str, Any]:
        return {"answer": await self.generate_text(system, user)}


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The waits asked for (none actually slept in tests)."""
    asked: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        asked.append(seconds)

    monkeypatch.setattr(retrying.asyncio, "sleep", fake_sleep)
    return asked


async def test_a_failed_call_waits_then_succeeds(waits: list[float]) -> None:
    flaky = Flaky(failures=2)
    client = RetryingLLMClient(flaky, retries=2, wait_seconds=90)
    assert await client.generate_json("s", "u") == {"answer": "ok"}
    assert flaky.calls == 3 and waits == [90, 90]
    assert (client.provider_name, client.model_name) == ("flaky", "m")


async def test_after_the_last_retry_the_error_goes_on_unchanged(waits: list[float]) -> None:
    flaky = Flaky(failures=5)
    with pytest.raises(LLMProviderError, match="ReadTimeout"):
        await RetryingLLMClient(flaky, retries=2, wait_seconds=60).generate_text("s", "u")
    assert flaky.calls == 3 and waits == [60, 60]


async def test_a_setup_error_is_not_retried(waits: list[float]) -> None:
    flaky = Flaky(failures=1, error=LLMProviderError("safechain is not installed", retryable=False))
    with pytest.raises(LLMProviderError, match="not installed"):
        await RetryingLLMClient(flaky, retries=2, wait_seconds=90).generate_text("s", "u")
    assert flaky.calls == 1 and waits == []


def test_zero_retries_and_offline_clients_are_not_wrapped(app_config: Any) -> None:
    fake = FakeLLMClient()
    assert RetryingLLMClient.wrap(fake, retries=0, wait_seconds=90) is fake
    assert not isinstance(create_llm_client(app_config), RetryingLLMClient)  # provider fake
