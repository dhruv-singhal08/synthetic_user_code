"""Pure checks of LLM-built reference data against the authoritative catalog.

Column references inside an IndexedTable are either bare (``amount``) or qualified with that
same table (``card_transactions.amount``). Anything else is an error.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from synthetic_user.models.reference import (
    ColumnMetadata,
    IndexedTable,
    IndexRelationship,
    ReferenceCatalog,
    ReferenceIndex,
    Relationship,
    TableMetadata,
    VerificationStatus,
)

TIME_TYPES = frozenset({"DATE", "DATETIME", "TIME", "TIMESTAMP"})


def check_index_against_catalog(
    index: ReferenceIndex,
    catalog: ReferenceCatalog,
    expected_tables: Collection[str] | None = None,
) -> list[str]:
    """Errors for an index that is not grounded in ``catalog``. Empty list means valid."""
    errors: list[str] = []
    if (index.source_catalog_id, index.source_catalog_version) != (
        catalog.catalog_id,
        catalog.catalog_version,
    ):
        errors.append(
            f"index built from {index.source_catalog_id} v{index.source_catalog_version}, "
            f"not {catalog.catalog_id} v{catalog.catalog_version}"
        )
    return errors + check_tables_against_catalog(index.tables, catalog, expected_tables)


def check_tables_against_catalog(
    tables: Sequence[IndexedTable],
    catalog: ReferenceCatalog,
    expected_tables: Collection[str] | None = None,
) -> list[str]:
    """Errors for unknown/missing/unexpected tables, unknown or misplaced columns,
    bad metric/time columns, and relationships that point at nothing.

    ``expected_tables`` defaults to every catalog table.
    """
    expected = set(expected_tables) if expected_tables is not None else _names(catalog)
    errors: list[str] = []
    seen: set[str] = set()
    for table in tables:
        seen.add(table.name)
        catalog_table = catalog.get_table(table.name)
        if catalog_table is None:
            errors.append(f"unknown table {table.name!r} (not in catalog)")
            continue
        if table.name not in expected:
            errors.append(f"table {table.name!r} was not requested in this call")
        errors.extend(_check_table(table, catalog_table, catalog))
    for name in sorted(expected - seen):
        errors.append(f"missing table {name!r} (every requested catalog table must be indexed)")
    return errors


def enforce_relationship_verification(
    index: ReferenceIndex, catalog: ReferenceCatalog
) -> ReferenceIndex:
    """Return a copy where "verified" survives only if the catalog has the same verified
    relationship (either direction). Others are downgraded to the catalog's status if the
    catalog knows the relationship, else to "inferred"; each downgrade is recorded in
    ``uncertainties``.
    """
    uncertainties = list(index.uncertainties)
    tables: list[IndexedTable] = []
    for table in index.tables:
        relationships: list[IndexRelationship] = []
        for rel in table.relationships:
            if rel.verification is VerificationStatus.VERIFIED:
                match = _catalog_match(table.name, rel, catalog)
                if match is None or match.verification is not VerificationStatus.VERIFIED:
                    downgraded = match.verification if match else VerificationStatus.INFERRED
                    uncertainties.append(
                        f"relationship {_describe(table.name, rel)} was marked verified but is "
                        f"not verified in the catalog; downgraded to {downgraded.value}"
                    )
                    rel = rel.model_copy(update={"verification": downgraded})
            relationships.append(rel)
        tables.append(table.model_copy(update={"relationships": relationships}))
    return index.model_copy(update={"tables": tables, "uncertainties": uncertainties})


# --- helpers -----------------------------------------------------------------------------


def _check_table(
    table: IndexedTable, catalog_table: TableMetadata, catalog: ReferenceCatalog
) -> list[str]:
    errors: list[str] = []
    for field, refs in (
        ("important_columns", table.important_columns),
        ("dimensions", table.dimensions),
        ("time_columns", table.time_columns),
    ):
        errors.extend(_check_ref(table.name, field, ref, catalog) for ref in refs)
    for metric in table.metrics:
        errors.extend(
            _check_ref(table.name, f"metric {metric.name!r}", ref, catalog)
            for ref in metric.columns
        )
    for ref in table.time_columns:
        column = _column(catalog_table, _bare(table.name, ref))
        if column is not None and column.data_type and column.data_type.upper() not in TIME_TYPES:
            errors.append(
                f"{table.name}: time_columns {ref!r} has non-time type {column.data_type}"
            )
    for rel in table.relationships:
        errors.extend(_check_relationship(table.name, rel, catalog))
    return [e for e in errors if e]


def _check_ref(table: str, field: str, ref: str, catalog: ReferenceCatalog) -> str:
    """Empty string if ``ref`` is a column of ``table``; otherwise an error message."""
    owner, _, column = ref.rpartition(".")
    if owner and owner != table:
        if catalog.has_column(owner, column):
            return f"{table}: {field} {ref!r} is a column of another table ({owner})"
        return f"{table}: {field} {ref!r} is not a catalog column"
    if catalog.has_column(table, column):
        return ""
    owners = [t.name for t in catalog.tables if t.has_column(column)]
    if owners:
        return f"{table}: {field} {ref!r} is not on this table (found on {', '.join(owners)})"
    return f"{table}: {field} {ref!r} is not a catalog column"


def _check_relationship(table: str, rel: IndexRelationship, catalog: ReferenceCatalog) -> list[str]:
    errors: list[str] = []
    source_error = _check_ref(table, "relationship source_column", rel.source_column, catalog)
    if source_error:
        errors.append(source_error)
    if catalog.get_table(rel.target_table) is None:
        errors.append(f"{table}: relationship target_table {rel.target_table!r} is not in catalog")
    elif not catalog.has_column(rel.target_table, _bare(rel.target_table, rel.target_column)):
        errors.append(
            f"{table}: relationship target {rel.target_table}.{rel.target_column} "
            "is not a catalog column"
        )
    return errors


def _catalog_match(
    table: str, rel: IndexRelationship, catalog: ReferenceCatalog
) -> Relationship | None:
    source = (table, _bare(table, rel.source_column))
    target = (rel.target_table, _bare(rel.target_table, rel.target_column))
    for candidate in catalog.relationships:
        ends = {
            (candidate.source_table, candidate.source_column),
            (candidate.target_table, candidate.target_column),
        }
        if ends == {source, target} and source != target:
            return candidate
    return None


def _describe(table: str, rel: IndexRelationship) -> str:
    return (
        f"{table}.{_bare(table, rel.source_column)} -> "
        f"{rel.target_table}.{_bare(rel.target_table, rel.target_column)}"
    )


def _bare(table: str, ref: str) -> str:
    owner, _, column = ref.rpartition(".")
    return column if owner == table else ref


def _column(table: TableMetadata, name: str) -> ColumnMetadata | None:
    return next((c for c in table.columns if c.name == name), None)


def _names(catalog: ReferenceCatalog) -> set[str]:
    return {t.name for t in catalog.tables}
