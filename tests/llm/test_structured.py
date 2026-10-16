from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from synthetic_user.core.exceptions import LLMOutputValidationError, LLMProviderError
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.llm.structured import generate_structured

SYSTEM = "You are a test."
USER = "Give me an item."


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    count: int


def _no_zero(item: Item) -> list[str]:
    return ["count must be > 0"] if item.count == 0 else []


async def test_success_first_try() -> None:
    llm = FakeLLMClient([{"name": "a", "count": 1}])
    item = await generate_structured(llm, SYSTEM, USER, Item, max_retries=2)
    assert item == Item(name="a", count=1)
    assert len(llm.calls) == 1
    assert llm.calls[0].user == USER


async def test_json_error_then_success() -> None:
    llm = FakeLLMClient(["not json at all", {"name": "a", "count": 1}])
    item = await generate_structured(llm, SYSTEM, USER, Item, max_retries=2)
    assert item.name == "a"
    assert len(llm.calls) == 2
    retry = llm.calls[1]
    assert retry.system == SYSTEM
    assert retry.user.startswith(USER)
    assert "previous output invalid:" in retry.user
    assert "no JSON object" in retry.user


async def test_validation_error_then_success() -> None:
    llm = FakeLLMClient([{"name": "a"}, {"name": "a", "count": 2}])
    item = await generate_structured(llm, SYSTEM, USER, Item, max_retries=1)
    assert item.count == 2
    assert "count" in llm.calls[1].user


async def test_post_validate_error_then_success() -> None:
    llm = FakeLLMClient([{"name": "a", "count": 0}, {"name": "a", "count": 3}])
    item = await generate_structured(llm, SYSTEM, USER, Item, max_retries=1, post_validate=_no_zero)
    assert item.count == 3
    assert "count must be > 0" in llm.calls[1].user


async def test_retry_feedback_is_not_cumulative() -> None:
    llm = FakeLLMClient(["bad", "bad", {"name": "a", "count": 1}])
    await generate_structured(llm, SYSTEM, USER, Item, max_retries=2)
    assert llm.calls[2].user.count("previous output invalid:") == 1


async def test_exhaustion_raises_typed_error() -> None:
    llm = FakeLLMClient(["bad", {"name": "a"}, {"name": "a", "count": 0}])
    with pytest.raises(LLMOutputValidationError) as exc_info:
        await generate_structured(llm, SYSTEM, USER, Item, max_retries=2, post_validate=_no_zero)
    err = exc_info.value
    assert err.attempts == 3
    assert len(err.errors) == 3
    assert err.errors[0].startswith("attempt 1:")
    assert "count must be > 0" in err.errors[2]
    assert llm.remaining == 0


async def test_zero_retries_means_one_attempt() -> None:
    llm = FakeLLMClient(["bad", {"name": "a", "count": 1}])
    with pytest.raises(LLMOutputValidationError) as exc_info:
        await generate_structured(llm, SYSTEM, USER, Item, max_retries=0)
    assert exc_info.value.attempts == 1
    assert len(llm.calls) == 1


async def test_provider_error_is_not_retried() -> None:
    llm = FakeLLMClient([LLMProviderError("down"), {"name": "a", "count": 1}])
    with pytest.raises(LLMProviderError, match="down"):
        await generate_structured(llm, SYSTEM, USER, Item, max_retries=2)
    assert len(llm.calls) == 1


async def test_negative_retries_rejected() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        await generate_structured(FakeLLMClient(), SYSTEM, USER, Item, max_retries=-1)
