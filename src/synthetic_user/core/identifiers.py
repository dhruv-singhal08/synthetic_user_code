"""Identifier formats.

Run ids: ``RUN_YYYYMMDD_HHMMSS_xxxx`` (UTC timestamp + 4 lowercase hex chars).
Entity ids: ``<PREFIX>_NNN`` from an injected, per-run SequentialIdGenerator.
"""

from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime
from enum import StrEnum

RUN_ID_PATTERN = re.compile(r"^RUN_\d{8}_\d{6}_[0-9a-f]{4}$")
INDEX_ID_PATTERN = re.compile(r"^IDX_\d{8}_\d{6}_[0-9a-f]{4}$")
_TOKEN_PATTERN = re.compile(r"^[0-9a-f]{4}$")


class IdKind(StrEnum):
    SCENARIO = "SCN"
    QUESTION = "Q"
    EXECUTION = "EXE"
    EVALUATION = "EVAL"
    FINDING = "F"
    BUG = "BUG"


def new_run_id(*, now: datetime | None = None, token: str | None = None) -> str:
    """Build a run id. ``now`` and ``token`` are injectable for deterministic tests."""
    return _timestamped_id("RUN", now, token)


def new_index_id(*, now: datetime | None = None, token: str | None = None) -> str:
    """Reference index id ``IDX_YYYYMMDD_HHMMSS_xxxx``; sorts chronologically."""
    return _timestamped_id("IDX", now, token)


def _timestamped_id(prefix: str, now: datetime | None, token: str | None) -> str:
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(2) if token is None else token
    if not _TOKEN_PATTERN.fullmatch(suffix):
        raise ValueError(f"id token must be 4 lowercase hex chars, got {suffix!r}")
    return f"{prefix}_{timestamp}_{suffix}"


class SequentialIdGenerator:
    """Per-run counters producing SCN_001, Q_001, EXE_001, EVAL_001, F_001, BUG_001, ..."""

    def __init__(self) -> None:
        self._counters: dict[IdKind, int] = dict.fromkeys(IdKind, 0)

    def next(self, kind: IdKind) -> str:
        self._counters[kind] += 1
        return f"{kind.value}_{self._counters[kind]:03d}"

    def peek(self, kind: IdKind) -> str:
        """The id ``next`` would return, without using it."""
        return f"{kind.value}_{self._counters[kind] + 1:03d}"
