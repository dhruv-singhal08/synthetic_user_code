"""Extract a JSON object from raw LLM text. Strict: no "repair" of malformed JSON."""

from __future__ import annotations

import json
import re
from typing import Any

from synthetic_user.core.exceptions import LLMOutputValidationError

_FENCE = re.compile(r"```[A-Za-z0-9_-]*")


def extract_json_object(text: str) -> dict[str, Any]:
    """Strip ``` fences, take the outermost ``{...}``, and ``json.loads`` it.

    Raises LLMOutputValidationError (attempts=1) if no JSON object can be parsed.
    """
    cleaned = _FENCE.sub("", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end < start:
        raise _invalid("no JSON object found in LLM output", text)
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        error = f"invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
        raise _invalid(error, text) from exc
    if not isinstance(data, dict):
        raise _invalid(f"expected a JSON object, got {type(data).__name__}", text)
    return data


def _invalid(error: str, text: str) -> LLMOutputValidationError:
    return LLMOutputValidationError(
        "LLM output is not a JSON object", attempts=1, errors=[error], raw_output=text
    )
