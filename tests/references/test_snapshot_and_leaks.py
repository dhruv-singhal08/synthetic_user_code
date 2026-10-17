from __future__ import annotations

import pytest

from synthetic_user.models.reference import ReferenceIndex, VerificationStatus
from synthetic_user.models.scenario import ReferenceCheatSheet, Scenario
from synthetic_user.references.schema_leak import find_schema_leaks, index_identifiers
from synthetic_user.references.snapshot import build_snapshot, snapshot_id_for


def test_snapshot_is_deterministic(
    sample_index: ReferenceIndex, sample_scenarios: list[Scenario]
) -> None:
    cheat = sample_scenarios[0].reference_cheat_sheet
    first = build_snapshot(sample_index, cheat, "SNAP_SCN_001")
    second = build_snapshot(sample_index, cheat, "SNAP_SCN_001")
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


def test_snapshot_has_all_columns_and_touching_relationships(
    sample_index: ReferenceIndex, sample_scenarios: list[Scenario]
) -> None:
    snapshot = build_snapshot(
        sample_index, sample_scenarios[0].reference_cheat_sheet, "SNAP_SCN_001"
    )
    assert [t.name for t in snapshot.tables] == ["card_transactions", "employees"]
    assert snapshot.tables[0] == sample_index.get_table("card_transactions")
    # employees touches both index relationships; verification is the index's original.
    assert [(r.source_table, r.target_table, r.verification) for r in snapshot.relationships] == [
        ("card_transactions", "employees", VerificationStatus.VERIFIED),
        ("employees", "departments", VerificationStatus.UNKNOWN),
    ]
    assert snapshot.missing_references == []
    assert snapshot.source_index_id == sample_index.versioned_id


def test_single_table_snapshot_relationships(
    sample_index: ReferenceIndex, sample_scenarios: list[Scenario]
) -> None:
    snapshot = build_snapshot(sample_index, sample_scenarios[1].reference_cheat_sheet, "S")
    assert [t.name for t in snapshot.tables] == ["card_transactions"]
    assert len(snapshot.relationships) == 1


def test_missing_refs_recorded(sample_index: ReferenceIndex) -> None:
    cheat = ReferenceCheatSheet(
        tables=["card_transactions", "budgets"],
        columns=["card_transactions.amount", "card_transactions.fx_rate", "budgets.limit"],
    )
    snapshot = build_snapshot(sample_index, cheat, "S")
    assert [t.name for t in snapshot.tables] == ["card_transactions"]
    assert snapshot.missing_references == ["budgets", "card_transactions.fx_rate"]


def test_snapshot_id_for() -> None:
    assert snapshot_id_for("SCN_001") == "SNAP_SCN_001"


# --- schema leaks ---------------------------------------------------------------------------


def test_identifiers_are_underscored_names(sample_index: ReferenceIndex) -> None:
    identifiers = index_identifiers(sample_index)
    assert "card_transactions" in identifiers
    assert "currency_code" in identifiers
    assert "amount" not in identifiers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Show total spend from card_transactions last month", ["card_transactions"]),
        ("Group it by CURRENCY_CODE please", ["currency_code"]),
        ("What did we spend on card transactions?", []),
        ("Total amount by department?", []),
        ("my_card_transactions_v2 report", []),
        ("Split by department_id and merchant_category", ["department_id", "merchant_category"]),
    ],
)
def test_find_schema_leaks(sample_index: ReferenceIndex, text: str, expected: list[str]) -> None:
    assert find_schema_leaks(text, index_identifiers(sample_index)) == expected
