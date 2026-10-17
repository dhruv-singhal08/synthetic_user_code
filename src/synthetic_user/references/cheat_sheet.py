"""Pure checks of a scenario cheat sheet against the reference index.

A cheat sheet may only use tables in the index, and only columns that table has in the index.
"""

from __future__ import annotations

from synthetic_user.models.reference import ReferenceIndex, VerificationStatus
from synthetic_user.models.scenario import CheatRelationship, ReferenceCheatSheet


def index_columns(index: ReferenceIndex) -> dict[str, set[str]]:
    """Table -> its column names in the index."""
    return {t.name: {c.name for c in t.columns} for t in index.tables}


def check_cheat_sheet_against_index(
    cheat_sheet: ReferenceCheatSheet, index: ReferenceIndex
) -> list[str]:
    """Errors for tables not in the index and "t.c" refs whose column table t does not have.
    Empty list means valid."""
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
    notes: list[str] = []
    relationships: list[CheatRelationship] = []
    for rel in cheat_sheet.relationships:
        indexed = index.verification(rel.source, rel.target)
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
