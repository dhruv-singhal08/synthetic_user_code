from __future__ import annotations

from datetime import UTC, datetime

import pytest

from synthetic_user.core.identifiers import (
    RUN_ID_PATTERN,
    IdKind,
    SequentialIdGenerator,
    new_run_id,
)


def test_run_id_format_is_deterministic_with_injected_inputs() -> None:
    now = datetime(2026, 10, 13, 9, 5, 7, tzinfo=UTC)
    assert new_run_id(now=now, token="a1b2") == "RUN_20261013_090507_a1b2"


def test_run_id_default_matches_pattern() -> None:
    assert RUN_ID_PATTERN.fullmatch(new_run_id())


@pytest.mark.parametrize("token", ["", "abc", "abcde", "ABCD", "zz11"])
def test_run_id_rejects_bad_token(token: str) -> None:
    with pytest.raises(ValueError, match="4 lowercase hex"):
        new_run_id(token=token)


@pytest.mark.parametrize(
    ("kind", "first", "second"),
    [
        (IdKind.SCENARIO, "SCN_001", "SCN_002"),
        (IdKind.QUESTION, "Q_001", "Q_002"),
        (IdKind.EXECUTION, "EXE_001", "EXE_002"),
        (IdKind.EVALUATION, "EVAL_001", "EVAL_002"),
        (IdKind.FINDING, "F_001", "F_002"),
        (IdKind.BUG, "BUG_001", "BUG_002"),
    ],
)
def test_sequential_id_formats(kind: IdKind, first: str, second: str) -> None:
    gen = SequentialIdGenerator()
    assert gen.next(kind) == first
    assert gen.next(kind) == second


def test_counters_are_independent_per_kind() -> None:
    gen = SequentialIdGenerator()
    assert gen.next(IdKind.SCENARIO) == "SCN_001"
    assert gen.next(IdKind.SCENARIO) == "SCN_002"
    assert gen.next(IdKind.QUESTION) == "Q_001"


def test_generators_do_not_share_state() -> None:
    first, second = SequentialIdGenerator(), SequentialIdGenerator()
    first.next(IdKind.BUG)
    assert second.next(IdKind.BUG) == "BUG_001"


def test_ids_grow_past_three_digits() -> None:
    gen = SequentialIdGenerator()
    ids = [gen.next(IdKind.FINDING) for _ in range(1000)]
    assert ids[998] == "F_999"
    assert ids[999] == "F_1000"
