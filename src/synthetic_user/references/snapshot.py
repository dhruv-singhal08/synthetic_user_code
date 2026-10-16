"""Deterministic per-scenario reference snapshot built from the catalog (not the LLM index)."""

from __future__ import annotations

from synthetic_user.models.reference import ReferenceCatalog, ReferenceSnapshot
from synthetic_user.models.scenario import ReferenceCheatSheet


def snapshot_id_for(scenario_id: str) -> str:
    return f"SNAP_{scenario_id}"


def build_snapshot(
    catalog: ReferenceCatalog, cheat_sheet: ReferenceCheatSheet, snapshot_id: str
) -> ReferenceSnapshot:
    """All columns of the cheat sheet's tables plus every catalog relationship touching them,
    with the catalog's original verification. Unknown tables / "t.c" refs are listed in
    ``missing_references`` (tables as "t", columns as "t.c"), never dropped silently.
    """
    table_names = list(dict.fromkeys(cheat_sheet.tables))
    tables = [t for name in table_names if (t := catalog.get_table(name)) is not None]
    present = {t.name for t in tables}

    missing: list[str] = [name for name in table_names if name not in present]
    for ref in _column_refs(cheat_sheet):
        table, column = ref.split(".", 1)
        if table in present and not catalog.has_column(table, column) and ref not in missing:
            missing.append(ref)

    relationships = [
        r for r in catalog.relationships if r.source_table in present or r.target_table in present
    ]
    return ReferenceSnapshot(
        snapshot_id=snapshot_id,
        source_catalog_id=catalog.catalog_id,
        source_catalog_version=catalog.catalog_version,
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
