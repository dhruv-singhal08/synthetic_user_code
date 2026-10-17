"""Deterministic per-scenario reference snapshot: the slice of the reference index a cheat
sheet points at."""

from __future__ import annotations

from collections.abc import Sequence

from synthetic_user.models.reference import ReferenceIndex, ReferenceSnapshot
from synthetic_user.models.scenario import ReferenceCheatSheet, Scenario


def snapshot_id_for(scenario_id: str) -> str:
    return f"SNAP_{scenario_id}"


def snapshots_for(index: ReferenceIndex, scenarios: Sequence[Scenario]) -> list[ReferenceSnapshot]:
    """One snapshot per grounded scenario (ungrounded scenarios have no cheat sheet)."""
    return [
        build_snapshot(index, s.reference_cheat_sheet, snapshot_id_for(s.id))
        for s in scenarios
        if s.reference_cheat_sheet is not None
    ]


def build_snapshot(
    index: ReferenceIndex, cheat_sheet: ReferenceCheatSheet, snapshot_id: str
) -> ReferenceSnapshot:
    """The cheat sheet's tables (all columns, rules) plus every top-level index relationship
    touching them, with the index's original verification. Unknown tables / "t.c" refs are
    listed in ``missing_references`` (tables as "t", columns as "t.c"), never dropped silently.
    """
    table_names = list(dict.fromkeys(cheat_sheet.tables))
    tables = [t for name in table_names if (t := index.get_table(name)) is not None]
    present = {t.name for t in tables}

    missing: list[str] = [name for name in table_names if name not in present]
    for ref in _column_refs(cheat_sheet):
        table, column = ref.split(".", 1)
        if table in present and not index.has_column(table, column) and ref not in missing:
            missing.append(ref)

    relationships = [
        r for r in index.relationships if r.source_table in present or r.target_table in present
    ]
    return ReferenceSnapshot(
        snapshot_id=snapshot_id,
        source_index_id=index.versioned_id,
        tables=tables,
        relationships=relationships,
        missing_references=missing,
    )


def _column_refs(cheat_sheet: ReferenceCheatSheet) -> list[str]:
    return [
        *cheat_sheet.columns,
        *cheat_sheet.dimensions,
        *(c for m in cheat_sheet.metrics for c in m.columns),
        *(r.source for r in cheat_sheet.relationships),
        *(r.target for r in cheat_sheet.relationships),
    ]
