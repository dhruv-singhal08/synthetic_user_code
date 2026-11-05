"""Deterministic in-memory index of a ``data_catalog`` and its four read-only exact lookups.

Built ONCE per catalog version (the reference id: file stem + content hash) from the already
validated reference; the YAML is never re-read per lookup. Matching is exact and
case-insensitive (quotes / backticks / brackets around a name are ignored). A qualified name
(``namespace.table``) resolves to ``table`` only when the namespace policy permits it:
``namespaces=None`` accepts any qualifier (the previous behaviour), a list accepts only those
qualifiers. Nothing is ever inferred: no similarity, aliases, cardinality or join validity
beyond what the reference states. Results are cached per index (same catalog version).

Rule ids are stable and positional: table rule n of table T is ``T#R<n>``; catalog-wide rule n
is ``R<n>`` (the ids the cheat sheet already uses). The rule text is the user's own wording.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from synthetic_user.models.lookup import (
    ColumnAlternativesLookup,
    ColumnFact,
    LookupOrigin,
    LookupRequest,
    LookupResult,
    LookupStatus,
    LookupTool,
    LookupTraceEntry,
    Provenance,
    RelationshipFact,
    RelationshipsLookup,
    RuleFact,
    RulesLookup,
    SchemaLookup,
)
from synthetic_user.models.reference import (
    CatalogColumn,
    CatalogTable,
    DataCatalog,
    Reference,
)

_QUOTES = "`\"[]'"


def normal_name(name: str) -> str:
    return name.strip().strip(_QUOTES).casefold()


def table_rule_id(table: str, index: int) -> str:
    return f"{table}#R{index}"


class CatalogIndex:
    def __init__(
        self,
        catalog: DataCatalog,
        *,
        catalog_version: str,
        namespaces: Sequence[str] | None = None,
        global_rules_label: str = "rules",
    ) -> None:
        self.catalog = catalog
        self.catalog_version = catalog_version
        self._namespaces = (
            None if namespaces is None else frozenset(normal_name(n) for n in namespaces)
        )
        self._global_label = global_rules_label
        self._tables = {t.name.casefold(): t for t in catalog.tables}
        self._columns = {
            (t.name.casefold(), c.name.casefold()): c for t in catalog.tables for c in t.columns
        }
        self._by_column: dict[str, list[CatalogTable]] = {}
        for table in catalog.tables:
            for column in table.columns:
                self._by_column.setdefault(column.name.casefold(), []).append(table)
        self._global_rules = catalog.rule_ids()
        self._cache: dict[str, LookupResult] = {}

    @classmethod
    def for_reference(
        cls, reference: Reference | None, *, namespaces: Sequence[str] | None = None
    ) -> CatalogIndex | None:
        """The index of a data-catalog reference; None for documents or no reference."""
        if reference is None or not isinstance(reference.content, DataCatalog):
            return None
        return cls(
            reference.content,
            catalog_version=reference.reference_id,
            namespaces=namespaces,
            global_rules_label="global_rules" if reference.source_format else "rules",
        )

    # --- resolution ------------------------------------------------------------------------

    def resolve_table(self, name: str) -> tuple[CatalogTable | None, str | None]:
        """(table, None) or (None, reason). A permitted qualifier is dropped; the last name
        part must then match a catalog table exactly (case-insensitive)."""
        parts = [p for p in (normal_name(x) for x in name.split(".")) if p]
        if not parts:
            return None, f"{name!r} is not a table name"
        *qualifiers, base = parts
        if qualifiers and self._namespaces is not None:
            qualifier = ".".join(qualifiers)
            if qualifier not in self._namespaces and qualifiers[-1] not in self._namespaces:
                return None, (
                    f"namespace {qualifier!r} is not permitted by evaluation.catalog_namespaces"
                )
        table = self._tables.get(base)
        if table is None:
            return None, f"table {base!r} is not in the catalog"
        return table, None

    def column(self, table: CatalogTable, name: str) -> CatalogColumn | None:
        return self._columns.get((table.name.casefold(), normal_name(name)))

    def tables_with_column(self, name: str) -> list[CatalogTable]:
        return list(self._by_column.get(normal_name(name), []))

    def table_rule_ids(self, table: CatalogTable) -> list[str]:
        return [table_rule_id(table.name, i) for i in range(1, len(table.rules) + 1)]

    def global_rule_ids(self) -> list[str]:
        return list(self._global_rules)

    def provenance(self, path: str) -> Provenance:
        return Provenance(catalog_version=self.catalog_version, path=path)

    def column_fact(self, table: CatalogTable, column: CatalogColumn) -> ColumnFact:
        return ColumnFact(
            table=table.name,
            column=column.name,
            type=column.type,
            description=column.description,
            values=list(column.values),
            table_description=table.description,
            provenance=self.provenance(f"tables[{table.name}].columns[{column.name}]"),
        )

    # --- the four lookups --------------------------------------------------------------------

    def lookup_schema(self, table_name: str, column_name: str | None = None) -> SchemaLookup:
        table, reason = self.resolve_table(table_name)
        if table is None:
            return SchemaLookup(
                table_name=table_name,
                column_name=column_name,
                status=LookupStatus.NOT_FOUND,
                reason=reason,
            )
        found = SchemaLookup(
            table_name=table_name,
            column_name=column_name,
            status=LookupStatus.FOUND,
            table=table.name,
            table_description=table.description,
            provenance=self.provenance(f"tables[{table.name}]"),
        )
        if column_name is None:
            return found
        column = self.column(table, column_name)
        if column is None:
            return found.model_copy(
                update={
                    "status": LookupStatus.NOT_FOUND,
                    "reason": f"column {normal_name(column_name)!r} is not in table {table.name}",
                    "provenance": self.provenance(f"tables[{table.name}].columns"),
                }
            )
        fact = self.column_fact(table, column)
        return found.model_copy(update={"column": fact, "provenance": fact.provenance})

    def lookup_rules(self, tables: Sequence[str] = (), rule_ids: Sequence[str] = ()) -> RulesLookup:
        rules: list[RuleFact] = []
        not_found: list[str] = []
        seen: set[str] = set()

        def add(fact: RuleFact) -> None:
            if fact.rule_id not in seen:
                seen.add(fact.rule_id)
                rules.append(fact)

        for name in tables:
            table, _ = self.resolve_table(name)
            if table is None:
                not_found.append(name)
                continue
            for fact in self._table_rules(table):
                add(fact)
        for rule_id in rule_ids:
            rule = self._rule(rule_id)
            if rule is None:
                not_found.append(rule_id)
            else:
                add(rule)
        return RulesLookup(
            tables=list(tables), rule_ids=list(rule_ids), rules=rules, not_found=not_found
        )

    def lookup_relationships(
        self, tables: Sequence[str] = (), pair: Sequence[str] | None = None
    ) -> RelationshipsLookup:
        names = list(tables) + [n for n in (pair or []) if n not in tables]
        resolved: dict[str, str] = {}
        not_found: list[str] = []
        for name in names:
            table, _ = self.resolve_table(name)
            if table is None:
                not_found.append(name)
            else:
                resolved[name] = table.name.casefold()
        wanted = {resolved[n] for n in tables if n in resolved}
        pair_set = {resolved[n] for n in (pair or []) if n in resolved}
        facts: list[RelationshipFact] = []
        for index, rel in enumerate(self.catalog.relationships, start=1):
            ends = {rel.from_.split(".", 1)[0].casefold(), rel.to.split(".", 1)[0].casefold()}
            if pair is not None:
                if len(pair_set) != 2 or ends != pair_set:
                    continue
            elif not ends & wanted:
                continue
            facts.append(
                RelationshipFact(
                    from_endpoint=rel.from_,
                    to_endpoint=rel.to,
                    verified=rel.verified,
                    description=rel.description,
                    provenance=self.provenance(f"relationships[{index}]"),
                )
            )
        return RelationshipsLookup(
            tables=list(tables) or list(pair or []),
            pair=list(pair) if pair is not None else None,
            relationships=facts,
            not_found=not_found,
        )

    def lookup_column_alternatives(
        self, column_name: str, excluded_table: str | None = None
    ) -> ColumnAlternativesLookup:
        excluded = None
        if excluded_table is not None:
            table, _ = self.resolve_table(excluded_table)
            excluded = table.name.casefold() if table is not None else None
        alternatives: list[ColumnFact] = []
        for table in self.tables_with_column(column_name):
            if table.name.casefold() == excluded:
                continue
            column = self.column(table, column_name)
            if column is not None:
                alternatives.append(self.column_fact(table, column))
        return ColumnAlternativesLookup(
            column_name=column_name, excluded_table=excluded_table, alternatives=alternatives
        )

    # --- dispatch + cache --------------------------------------------------------------------

    def run(self, request: LookupRequest) -> tuple[LookupResult, bool]:
        """(result, cached). Identical requests are answered once per catalog version."""
        key = request.key()
        if key in self._cache:
            return self._cache[key], True
        result = self._dispatch(request)
        self._cache[key] = result
        return result, False

    def _dispatch(self, request: LookupRequest) -> LookupResult:
        tool = request.tool
        if tool is LookupTool.LOOKUP_SCHEMA:
            assert request.table is not None  # checked by LookupRequest
            return self.lookup_schema(request.table, request.column)
        if tool is LookupTool.LOOKUP_RULES:
            return self.lookup_rules(request.tables, request.rule_ids)
        if tool is LookupTool.LOOKUP_RELATIONSHIPS:
            return self.lookup_relationships(request.tables, request.pair)
        assert request.column is not None  # checked by LookupRequest
        return self.lookup_column_alternatives(request.column, request.excluded_table)

    # --- rules ---------------------------------------------------------------------------------

    def _table_rules(self, table: CatalogTable) -> Iterable[RuleFact]:
        for index, text in enumerate(table.rules, start=1):
            yield RuleFact(
                rule_id=table_rule_id(table.name, index),
                scope="table",
                table=table.name,
                text=text,
                provenance=self.provenance(f"tables[{table.name}].rules[{index}]"),
            )

    def _rule(self, rule_id: str) -> RuleFact | None:
        key = rule_id.strip()
        for gid, text in self._global_rules.items():
            if gid.casefold() == key.casefold():
                index = int(gid[1:])
                return RuleFact(
                    rule_id=gid,
                    scope="catalog",
                    text=text,
                    provenance=self.provenance(f"{self._global_label}[{index}]"),
                )
        table_name, sep, number = key.rpartition("#R")
        if not sep or not number.isdigit():
            return None
        table, _ = self.resolve_table(table_name)
        if table is None:
            return None
        facts = list(self._table_rules(table))
        index = int(number)
        return facts[index - 1] if 1 <= index <= len(facts) else None


class LookupSession:
    """The lookups of ONE evaluation: runs requests through the shared index and keeps the
    ordered, auditable trace (origin, request, result, cache hit)."""

    def __init__(self, index: CatalogIndex) -> None:
        self.index = index
        self.trace: list[LookupTraceEntry] = []

    def run(self, request: LookupRequest, origin: LookupOrigin) -> LookupTraceEntry:
        """Run one lookup and trace it with its evidence id (LOOKUP:<n>, in trace order) and
        round (0 = precomputed by code, 1 = the judge's requested batch)."""
        result, cached = self.index.run(request)
        entry = LookupTraceEntry(
            origin=origin,
            request=request,
            result=result,
            cached=cached,
            evidence_id=f"LOOKUP:{len(self.trace) + 1}",
            round=0 if origin is LookupOrigin.DETERMINISTIC else 1,
        )
        self.trace.append(entry)
        return entry

    def run_all(
        self, requests: Sequence[LookupRequest], origin: LookupOrigin
    ) -> list[LookupTraceEntry]:
        """Run a batch once each (duplicate requests in the batch are answered once)."""
        done: dict[str, LookupTraceEntry] = {}
        for request in requests:
            key = request.key()
            if key not in done:
                done[key] = self.run(request, origin)
        return list(done.values())
