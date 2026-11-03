"""Deterministic reconciliation of a parsed SQL artifact with the cheat sheet and the catalog.

Runs BEFORE the judge, from the parser output only (never a regex, never execution):

- every base table / column -> ``valid_in_pack`` (catalog + cheat sheet), ``valid_extra``
  (catalog only: NOT a schema error by itself), ``invalid`` (exact lookup proves it is absent)
  or ``unresolved`` (ambiguous unqualified column, unknown table, namespace not permitted);
- cheat-sheet tables / columns the SQL does not use (listed; never a failure by itself);
- equality join keys -> documented (verified / not verified), undocumented or unresolved;
- rule ids the judge may need that the cheat sheet did not carry (rules of extra tables and
  unselected catalog-wide rules): candidates only, applicability is the judge's call.

``evidence_requests`` turns the obvious gaps into exact lookups so the judge does not have to
discover them (extra columns, extra tables, unselected rules, alternatives for invalid columns,
relationships between the joined tables).
"""

from __future__ import annotations

from itertools import combinations

from synthetic_user.models.lookup import (
    ArtifactReconciliation,
    IdentifierCheck,
    IdentifierStatus,
    JoinCheck,
    JoinStatus,
    LookupRequest,
    LookupTool,
)
from synthetic_user.models.reference import CatalogTable, ReferenceKind, ReferencePack
from synthetic_user.references.catalog_index import CatalogIndex
from synthetic_user.sql.parser import SqlColumnRef, SqlStructure


def reconcile(
    structure: SqlStructure,
    *,
    artifact: int,
    index: CatalogIndex,
    pack: ReferencePack | None,
) -> ArtifactReconciliation:
    if not structure.parsed:
        return ArtifactReconciliation(
            artifact=artifact, parsed=False, diagnostics=list(structure.diagnostics)
        )
    pack_tables, pack_columns = _pack_items(pack)
    checks: list[IdentifierCheck] = []
    sql_tables: dict[str, CatalogTable] = {}
    unresolved_tables = False
    for name in structure.base_tables:
        table, reason = index.resolve_table(name)
        if table is None:
            assert reason is not None
            namespace = "namespace" in reason
            unresolved_tables = True
            checks.append(
                IdentifierCheck(
                    identifier=name,
                    kind="table",
                    status=IdentifierStatus.UNRESOLVED if namespace else IdentifierStatus.INVALID,
                    reason=reason,
                )
            )
            continue
        sql_tables[table.name.casefold()] = table
        in_pack = table.name.casefold() in pack_tables
        checks.append(
            IdentifierCheck(
                identifier=name,
                kind="table",
                status=IdentifierStatus.VALID_IN_PACK if in_pack else IdentifierStatus.VALID_EXTRA,
                canonical=table.name,
                reason="in the catalog and the cheat sheet"
                if in_pack
                else "in the catalog; not selected in the cheat sheet",
            )
        )
    used_columns: set[tuple[str, str]] = set()
    for ref in structure.base_columns:
        check = _column_check(ref, index, sql_tables, pack_columns, unresolved=unresolved_tables)
        checks.append(check)
        if check.canonical is not None:
            table_name, _, column_name = check.canonical.partition(".")
            used_columns.add((table_name.casefold(), column_name.casefold()))
    not_used = [name for key, name in pack_tables.items() if key not in sql_tables] + [
        f"{table}.{column}"
        for (table_key, column_key), (table, column) in pack_columns.items()
        if (table_key, column_key) not in used_columns
    ]
    extra_tables = [t.name for key, t in sql_tables.items() if key not in pack_tables]
    pack_rule_ids = {r.id.casefold() for r in pack.rules} if pack is not None else set()
    rules_not_in_pack = [
        rule_id
        for key in extra_tables
        for rule_id in index.table_rule_ids(sql_tables[key.casefold()])
    ] + [rid for rid in index.global_rule_ids() if rid.casefold() not in pack_rule_ids]
    return ArtifactReconciliation(
        artifact=artifact,
        parsed=True,
        diagnostics=list(structure.diagnostics),
        identifiers=checks,
        extra_tables=extra_tables,
        pack_items_not_used=not_used,
        joins=[_join_check(key.left, key.right, index) for key in structure.join_keys],
        rules_not_in_pack=rules_not_in_pack,
    )


def evidence_requests(
    reconciliations: list[ArtifactReconciliation], index: CatalogIndex
) -> list[LookupRequest]:
    """Exact lookups for the gaps code can see; duplicates are removed by the session."""
    requests: list[LookupRequest] = []
    for rec in reconciliations:
        for check in rec.with_status(IdentifierStatus.VALID_EXTRA):
            if check.kind == "column" and check.canonical is not None:
                table, _, column = check.canonical.partition(".")
                requests.append(
                    LookupRequest(
                        tool=LookupTool.LOOKUP_SCHEMA,
                        table=table,
                        column=column,
                        reason="column used by the SQL but not selected in the cheat sheet",
                    )
                )
        for check in rec.with_status(IdentifierStatus.INVALID):
            if check.kind == "column":
                table, _, column = (check.canonical or check.identifier).rpartition(".")
                requests.append(
                    LookupRequest(
                        tool=LookupTool.LOOKUP_COLUMN_ALTERNATIVES,
                        column=column or check.identifier,
                        excluded_table=table or None,
                        reason="column absent from the catalog table it is read from",
                    )
                )
        if rec.extra_tables or rec.rules_not_in_pack:
            requests.append(
                LookupRequest(
                    tool=LookupTool.LOOKUP_RULES,
                    tables=list(rec.extra_tables),
                    rule_ids=[r for r in rec.rules_not_in_pack if "#R" not in r],
                    reason="rules of tables the SQL reads that the cheat sheet did not carry, "
                    "and catalog-wide rules it did not select",
                )
            )
        joined = sorted(
            {
                check.canonical
                for check in rec.identifiers
                if check.kind == "table" and check.canonical is not None
            }
        )
        if rec.joins and len(joined) >= 2:
            requests.extend(
                LookupRequest(
                    tool=LookupTool.LOOKUP_RELATIONSHIPS,
                    pair=[a, b],
                    reason="documented relationships between tables the SQL combines",
                )
                for a, b in combinations(joined, 2)
            )
    return requests


def _pack_items(
    pack: ReferencePack | None,
) -> tuple[dict[str, str], dict[tuple[str, str], tuple[str, str]]]:
    if pack is None or pack.kind is not ReferenceKind.DATA_CATALOG:
        return {}, {}
    tables = {t.name.casefold(): t.name for t in pack.tables}
    columns = {
        (t.name.casefold(), c.name.casefold()): (t.name, c.name)
        for t in pack.tables
        for c in t.columns
    }
    return tables, columns


def _column_check(
    ref: SqlColumnRef,
    index: CatalogIndex,
    sql_tables: dict[str, CatalogTable],
    pack_columns: dict[tuple[str, str], tuple[str, str]],
    *,
    unresolved: bool,
) -> IdentifierCheck:
    written = _written(ref)
    if ref.table is not None:
        table, reason = index.resolve_table(ref.table)
        if table is None:
            return IdentifierCheck(
                identifier=written,
                kind="column",
                status=IdentifierStatus.UNRESOLVED,
                reason=f"its table cannot be checked: {reason}",
            )
        candidates = [table]
    else:
        candidates = [t for t in sql_tables.values() if index.column(t, ref.column) is not None]
        if len(candidates) > 1:
            return IdentifierCheck(
                identifier=written,
                kind="column",
                status=IdentifierStatus.UNRESOLVED,
                reason="unqualified and present in several tables the SQL reads: "
                + ", ".join(t.name for t in candidates),
            )
        if not candidates:
            return IdentifierCheck(
                identifier=written,
                kind="column",
                status=IdentifierStatus.UNRESOLVED if unresolved else IdentifierStatus.INVALID,
                reason="not in any catalog table the SQL reads"
                + (" (some of its tables could not be resolved)" if unresolved else ""),
            )
    table = candidates[0]
    column = index.column(table, ref.column)
    if column is None:
        return IdentifierCheck(
            identifier=written,
            kind="column",
            status=IdentifierStatus.INVALID,
            canonical=None,
            reason=f"column {ref.column!r} is not in catalog table {table.name}",
        )
    canonical = f"{table.name}.{column.name}"
    in_pack = (table.name.casefold(), column.name.casefold()) in pack_columns
    return IdentifierCheck(
        identifier=written,
        kind="column",
        status=IdentifierStatus.VALID_IN_PACK if in_pack else IdentifierStatus.VALID_EXTRA,
        canonical=canonical,
        reason="in the catalog and the cheat sheet"
        if in_pack
        else "in the catalog; not selected in the cheat sheet",
    )


def _written(ref: SqlColumnRef) -> str:
    return f"{ref.table}.{ref.column}" if ref.table else ref.column


def _join_check(left: SqlColumnRef, right: SqlColumnRef, index: CatalogIndex) -> JoinCheck:
    ends: list[str] = []
    for ref in (left, right):
        table = index.resolve_table(ref.table)[0] if ref.table else None
        column = index.column(table, ref.column) if table is not None else None
        if table is None or column is None:
            return JoinCheck(
                left=_written(left), right=_written(right), status=JoinStatus.UNRESOLVED
            )
        ends.append(f"{table.name}.{column.name}")
    pair = {ends[0].casefold(), ends[1].casefold()}
    for position, rel in enumerate(index.catalog.relationships, start=1):
        if {rel.from_.casefold(), rel.to.casefold()} == pair:
            return JoinCheck(
                left=ends[0],
                right=ends[1],
                status=JoinStatus.DOCUMENTED_VERIFIED
                if rel.verified
                else JoinStatus.DOCUMENTED_UNVERIFIED,
                provenance=index.provenance(f"relationships[{position}]"),
            )
    return JoinCheck(left=ends[0], right=ends[1], status=JoinStatus.UNDOCUMENTED)
