"""Pure checks of a loaded reference index. Problems are WARNINGS: listed, never fixed.

The loader already rejected unknown fields and missing required ones; this module reports
content problems the tool can live with: duplicate tables/columns, relationships pointing to
unknown tables/columns, column lists naming unknown columns, a table-level relationship
missing from the top-level list, scope counts that disagree with the content, and an
integrity_report that did not pass.
"""

from __future__ import annotations

from synthetic_user.models.base import find_duplicates
from synthetic_user.models.reference import IndexRelationship, ReferenceIndex


def validate_index(index: ReferenceIndex) -> list[str]:
    """Warnings for ``index`` (empty list = clean), in a stable order."""
    warnings: list[str] = []
    names = index.table_names()
    dupes = find_duplicates(names)
    if dupes:
        warnings.append(f"duplicate table names {dupes}")
    columns = {t.name: {c.name for c in t.columns} for t in index.tables}

    for table in index.tables:
        dupe_columns = find_duplicates([c.name for c in table.columns])
        if dupe_columns:
            warnings.append(f"{table.name}: duplicate column names {dupe_columns}")
        for field, refs in (
            ("primary_key_columns", table.primary_key_columns),
            ("important_columns", table.important_columns),
            ("sensitive_columns", table.sensitive_columns),
            ("dimensions", [d.name for d in table.dimensions]),
            ("time_columns", [c.name for c in table.time_columns]),
        ):
            for ref in refs:
                if ref not in columns[table.name]:
                    warnings.append(f"{table.name}.{field}: unknown column {ref!r}")

    top_level = {_key(r) for r in index.relationships}
    for where, rel in [
        *(("relationships", r) for r in index.relationships),
        *((f"{t.name}.relationships", r) for t in index.tables for r in t.relationships),
    ]:
        warnings.extend(
            f"{where} {rel.source} -> {rel.target}: {p}" for p in _dangling(rel, columns)
        )
    for table in index.tables:
        for rel in table.relationships:
            if _key(rel) not in top_level:
                warnings.append(
                    f"{table.name}.relationships {rel.source} -> {rel.target}: "
                    "missing from the top-level relationships list"
                )

    warnings.extend(_scope_warnings(index, columns))
    report = index.extras.get("integrity_report")
    if isinstance(report, dict) and report.get("status") not in (None, "passed"):
        warnings.append(
            f"integrity_report.status is {report.get('status')!r} (issues: {report.get('issues')})"
        )
    return warnings


def _dangling(rel: IndexRelationship, columns: dict[str, set[str]]) -> list[str]:
    problems: list[str] = []
    for table, column in (
        (rel.source_table, rel.source_column),
        (rel.target_table, rel.target_column),
    ):
        if table not in columns:
            problems.append(f"unknown table {table!r}")
        elif column not in columns[table]:
            problems.append(f"unknown column {table}.{column}")
    return problems


def _scope_warnings(index: ReferenceIndex, columns: dict[str, set[str]]) -> list[str]:
    scope = index.extras.get("scope")
    if not isinstance(scope, dict):
        return []
    warnings: list[str] = []
    if "table_count" in scope and scope["table_count"] != len(index.tables):
        warnings.append(f"scope.table_count={scope['table_count']} but {len(index.tables)} tables")
    total = sum(len(t.columns) for t in index.tables)
    if "column_count" in scope and scope["column_count"] != total:
        warnings.append(f"scope.column_count={scope['column_count']} but {total} columns")
    listed = scope.get("tables")
    if isinstance(listed, list) and sorted(map(str, listed)) != sorted(columns):
        warnings.append(f"scope.tables {sorted(map(str, listed))} != tables {sorted(columns)}")
    for table in index.tables:
        declared = table.extras.get("column_count")
        if isinstance(declared, int) and declared != len(table.columns):
            warnings.append(
                f"{table.name}.column_count={declared} but {len(table.columns)} columns"
            )
    return warnings


def _key(rel: IndexRelationship) -> tuple[str, str]:
    return (rel.source, rel.target)
