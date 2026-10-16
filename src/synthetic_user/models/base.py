"""Shared Pydantic base and field types for all domain models."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# Fully qualified column reference "table.column".
QualifiedColumn = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, pattern=r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$"
    ),
]


class AmbiguityLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def find_duplicates(names: list[str]) -> list[str]:
    """Return names that appear more than once, in first-seen order."""
    seen: set[str] = set()
    dupes: list[str] = []
    for name in names:
        if name in seen and name not in dupes:
            dupes.append(name)
        seen.add(name)
    return dupes


def table_of(qualified: str) -> str:
    """``"t.c"`` -> ``"t"``."""
    return qualified.split(".", 1)[0]
