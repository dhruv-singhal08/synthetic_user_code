"""Compact output contracts: a JSON example instead of the full JSON Schema (D204).

The model sees one short example with placeholders; code still validates the reply against
the Pydantic model, so nothing is less strict. A test keeps every example's keys equal to
its model's fields. The introducing sentence is in ``common.yaml``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from enum import StrEnum
from typing import Any

from synthetic_user.prompts.templates import prompts_beside

PROMPTS = prompts_beside(__file__, "common.yaml")


def choices(values: Iterable[StrEnum | str]) -> str:
    return "<" + " | ".join(str(v) for v in values) + ">"


def render_contract(example: dict[str, Any], notes: str = "") -> str:
    return (
        PROMPTS.text("output_contract")
        + (f"\n{notes}" if notes else "")
        + "\n"
        + json.dumps(example, indent=2, ensure_ascii=False)
    )
