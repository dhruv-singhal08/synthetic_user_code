from __future__ import annotations

import pytest

from synthetic_user.core.exceptions import LLMOutputValidationError
from synthetic_user.llm.json_utils import extract_json_object


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Sure! Here it is:\n```json\n{"a": 1}\n```\nHope that helps.',
        'prefix {"a": 1} suffix',
        '  \n{"a": 1}\n  ',
    ],
)
def test_extracts_object(text: str) -> None:
    assert extract_json_object(text) == {"a": 1}


def test_takes_outermost_object_with_nesting() -> None:
    assert extract_json_object('x {"a": {"b": [1, {"c": 2}]}} y') == {"a": {"b": [1, {"c": 2}]}}


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "no JSON object"),
        ("no braces here", "no JSON object"),
        ("} backwards {", "no JSON object"),
        ('{"a": 1,}', "invalid JSON"),
        ("{'a': 1}", "invalid JSON"),
        ('{"a": 1} and {"b": 2}', "invalid JSON"),
    ],
)
def test_rejects_without_repair(text: str, reason: str) -> None:
    with pytest.raises(LLMOutputValidationError) as exc_info:
        extract_json_object(text)
    assert exc_info.value.attempts == 1
    assert reason in exc_info.value.errors[0]


def test_rejects_non_object_json() -> None:
    with pytest.raises(LLMOutputValidationError):
        extract_json_object("[1, 2, 3]")
