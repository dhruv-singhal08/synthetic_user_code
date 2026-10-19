"""Compact output contracts: a JSON example instead of the full JSON Schema (D204).

The model sees one short example with placeholders; code still validates the reply against
the Pydantic model, so nothing is less strict. A test keeps every example's keys equal to
its model's fields.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from enum import StrEnum
from typing import Any


def choices(values: Iterable[StrEnum | str]) -> str:
    return "<" + " | ".join(str(v) for v in values) + ">"


def render_contract(example: dict[str, Any], notes: str = "") -> str:
    return (
        "Return ONE JSON object shaped exactly like this example. <...> is a placeholder to "
        "fill; <a | b> means one of those values. Use real JSON types (numbers, true / false, "
        "null), no extra keys."
        + (f"\n{notes}" if notes else "")
        + "\n"
        + json.dumps(example, indent=2, ensure_ascii=False)
    )
