"""Pure checks of a scenario cheat sheet against the reference index.

A cheat sheet may only use tables in the index, and only columns the index mentions for that
table (important/dimension/time/metric columns, relationship endpoints).
"""

from __future__ import annotations

from synthetic_user.models.reference import ReferenceIndex, VerificationStatus
from synthetic_user.models.scenario import CheatRelationship, ReferenceCheatSheet


def index_columns(index: ReferenceIndex) -> dict[str, set[str]]:
    """Table -> bare column names the index mentions for it."""
    columns: dict[str, set[str]] = {t.name: set() for t in index.tables}
    for table in index.tables:
        refs = [
            *table.important_columns,
            *table.dimensions,
            *table.time_columns,
            *(c for m in table.metrics for c in m.columns),
            *(r.source_column for r in table.relationships),
        ]
        columns[table.name].update(_bare(ref) for ref in refs)
        for rel in table.relationships:
            columns.setdefault(rel.target_table, set()).add(_bare(rel.target_column))
    return columns


def check_cheat_sheet_against_index(
    cheat_sheet: ReferenceCheatSheet, index: ReferenceIndex
) -> list[str]:
    """Errors for tables not in the index and "t.c" refs whose column the index doesn't
    list for table t. Empty list means valid."""
    known_tables = {t.name for t in index.tables}
    allowed = index_columns(index)
    errors = [
        f"table {t!r} is not in the reference index"
        for t in cheat_sheet.tables
        if t not in known_tables
    ]
    refs = [
        ("columns", cheat_sheet.columns),
        ("dimensions", cheat_sheet.dimensions),
        *((f"metric {m.name!r}", m.columns) for m in cheat_sheet.metrics),
        ("relationships", [r.source for r in cheat_sheet.relationships]),
        ("relationships", [r.target for r in cheat_sheet.relationships]),
    ]
    for field, values in refs:
        for ref in values:
            table, column = ref.split(".", 1)
            if table not in known_tables:
                continue  # already reported (or caught by the table-list check)
            if column not in allowed.get(table, set()):
                errors.append(
                    f"{field}: {ref!r} - column {column!r} is not on table {table!r} "
                    "in the reference index"
                )
    return errors


def align_relationship_verification(
    cheat_sheet: ReferenceCheatSheet, index: ReferenceIndex
) -> tuple[ReferenceCheatSheet, list[str]]:
    """Copy verification from the index; never promote.

    - Relationship in the index with a different status -> index status + note.
    - Join with no index relationship -> "unknown" + note.
    Notes are appended to semantic_considerations and also returned.
    """
    known = _index_relationships(index)
    notes: list[str] = []
    relationships: list[CheatRelationship] = []
    for rel in cheat_sheet.relationships:
        pair = frozenset({rel.source, rel.target})
        indexed = known.get(pair)
        if indexed is None:
            notes.append(
                f"join {rel.source} = {rel.target} has no relationship in the reference "
                "index; verification unknown"
            )
            rel = rel.model_copy(update={"verification": VerificationStatus.UNKNOWN})
        elif rel.verification is not indexed:
            notes.append(
                f"join {rel.source} = {rel.target} was marked {rel.verification.value} but the "
                f"reference index says {indexed.value}; set to {indexed.value}"
            )
            rel = rel.model_copy(update={"verification": indexed})
        relationships.append(rel)
    updated = cheat_sheet.model_copy(
        update={
            "relationships": relationships,
            "semantic_considerations": [*cheat_sheet.semantic_considerations, *notes],
        }
    )
    return updated, notes


def _index_relationships(index: ReferenceIndex) -> dict[frozenset[str], VerificationStatus]:
    known: dict[frozenset[str], VerificationStatus] = {}
    for table in index.tables:
        for rel in table.relationships:
            source = f"{table.name}.{_bare(rel.source_column)}"
            target = f"{rel.target_table}.{_bare(rel.target_column)}"
            known[frozenset({source, target})] = rel.verification
    return known


def _bare(ref: str) -> str:
    return ref.rpartition(".")[2]
