from __future__ import annotations

import pytest

from synthetic_user.core.exceptions import LLMProviderError
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.fake import FakeCall, FakeLLMClient


def test_satisfies_protocol() -> None:
    assert isinstance(FakeLLMClient(), LLMClient)


async def test_records_calls_in_order() -> None:
    llm = FakeLLMClient([{"a": 1}, "hello"])
    await llm.generate_json("sys1", "user1")
    await llm.generate_text("sys2", "user2")
    assert llm.calls == [
        FakeCall("generate_json", "sys1", "user1"),
        FakeCall("generate_text", "sys2", "user2"),
    ]


async def test_queue_item_kinds() -> None:
    llm = FakeLLMClient()
    llm.queue(
        {"a": 1},
        '```json\n{"b": 2}\n```',
        lambda system, user: {"echo": user},
        RuntimeError("boom"),
    )
    assert await llm.generate_json("s", "u") == {"a": 1}
    assert await llm.generate_json("s", "u") == {"b": 2}
    assert await llm.generate_json("s", "ping") == {"echo": "ping"}
    with pytest.raises(RuntimeError, match="boom"):
        await llm.generate_json("s", "u")


async def test_generate_text_serialises_dicts() -> None:
    llm = FakeLLMClient([{"a": 1}])
    assert await llm.generate_text("s", "u") == '{"a": 1}'


async def test_returned_dicts_are_copies() -> None:
    scripted = {"a": [1]}
    llm = FakeLLMClient([scripted])
    result = await llm.generate_json("s", "u")
    result["a"].append(2)
    assert scripted == {"a": [1]}


async def test_empty_queue_gives_clear_error() -> None:
    llm = FakeLLMClient([{"a": 1}])
    await llm.generate_json("s", "u")
    with pytest.raises(LLMProviderError, match=r"queue is empty: call #2 \(generate_text\)"):
        await llm.generate_text("s", "u")


def test_names_are_configurable() -> None:
    llm = FakeLLMClient(provider_name="p", model_name="m")
    assert (llm.provider_name, llm.model_name) == ("p", "m")
