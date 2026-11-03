"""Pluggable SQL AST parser: supporting evidence for the SQL metric rubrics (never a judge).

``SqlParser`` is the interface; ``SqlglotParser`` is the default implementation. The parser
extracts base tables, base columns (aliases resolved per scope), CTE names, output aliases,
functions, joins with predicates, filters, projections, aggregations, grouping, ordering and
limits, and keeps the raw SQL. Base database identifiers are kept apart from CTE names, output
aliases and functions so Schema Faithfulness never compares those.

The dialect comes only from explicit configuration (``evaluation.sql_dialect``); it is never
guessed. A parse failure is explicit (``parsed=False`` + diagnostics) and never fails a metric
by itself: the judge still sees the original SQL. SQL is never executed.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import Field

from synthetic_user.models.base import NonEmptyStr, StrictModel

if TYPE_CHECKING:
    from sqlglot import exp

PARSER_UNAVAILABLE = "SQL parser unavailable: the sqlglot package is not installed"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")  # terminal colour codes in parser error messages


class SqlColumnRef(StrictModel):
    column: NonEmptyStr
    table: str | None = None  # the resolved BASE table; None when it cannot be resolved
    qualifier: str | None = None  # the qualifier as written (alias or table name)


class SqlJoin(StrictModel):
    kind: NonEmptyStr  # e.g. "JOIN", "LEFT JOIN"
    target: NonEmptyStr  # the joined table / derived source as written
    predicate: str | None = None


class SqlStructure(StrictModel):
    raw_sql: str
    dialect: str | None = None
    parsed: bool
    diagnostics: list[NonEmptyStr] = Field(default_factory=list)
    statement_count: int = Field(default=0, ge=0)
    normalized_sql: str | None = None
    base_tables: list[NonEmptyStr] = Field(default_factory=list)
    table_aliases: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    ctes: list[NonEmptyStr] = Field(default_factory=list)
    base_columns: list[SqlColumnRef] = Field(default_factory=list)
    output_aliases: list[NonEmptyStr] = Field(default_factory=list)
    functions: list[NonEmptyStr] = Field(default_factory=list)
    joins: list[SqlJoin] = Field(default_factory=list)
    filters: list[NonEmptyStr] = Field(default_factory=list)
    projections: list[NonEmptyStr] = Field(default_factory=list)
    aggregations: list[NonEmptyStr] = Field(default_factory=list)
    group_by: list[NonEmptyStr] = Field(default_factory=list)
    order_by: list[NonEmptyStr] = Field(default_factory=list)
    limits: list[NonEmptyStr] = Field(default_factory=list)


class SchemaComparison(StrictModel):
    """Deterministic comparison of base identifiers with the cheat sheet (evidence only)."""

    unknown_tables: list[NonEmptyStr] = Field(default_factory=list)
    unknown_columns: list[NonEmptyStr] = Field(default_factory=list)
    unresolved_columns: list[NonEmptyStr] = Field(default_factory=list)


class SqlParser(Protocol):
    name: str

    def parse(self, sql: str, *, dialect: str | None) -> SqlStructure: ...

    def remove_predicates(
        self, sql: str, predicates: Sequence[str], *, dialect: str | None
    ) -> tuple[str, bool]: ...


def _sqlglot() -> Any:
    try:
        return importlib.import_module("sqlglot")
    except ImportError:
        return None


class SqlglotParser:
    name = "sqlglot"

    def parse(self, sql: str, *, dialect: str | None) -> SqlStructure:
        sqlglot = _sqlglot()
        if sqlglot is None:
            return SqlStructure(
                raw_sql=sql, dialect=dialect, parsed=False, diagnostics=[PARSER_UNAVAILABLE]
            )
        try:
            statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
        except (sqlglot.errors.ParseError, sqlglot.errors.TokenError, ValueError) as exc:
            plain = _ANSI.sub("", str(exc))
            message = " ".join(plain.split())[:500] or type(exc).__name__
            return SqlStructure(
                raw_sql=sql, dialect=dialect, parsed=False, diagnostics=[f"parse error: {message}"]
            )
        if not statements:
            return SqlStructure(
                raw_sql=sql, dialect=dialect, parsed=False, diagnostics=["no SQL statement found"]
            )
        exp = sqlglot.exp
        kinds = tuple(getattr(exp, n) for n in ("Query", "DML", "DDL") if hasattr(exp, n))
        odd = [type(s).__name__ for s in statements if not isinstance(s, kinds)]
        if odd:
            return SqlStructure(
                raw_sql=sql,
                dialect=dialect,
                parsed=False,
                diagnostics=[f"not a recognizable SQL statement (parsed as {', '.join(odd)})"],
            )
        collector = _Collector(dialect)
        for statement in statements:
            collector.collect(statement)
        return collector.result(sql, len(statements))

    def remove_predicates(
        self, sql: str, predicates: Sequence[str], *, dialect: str | None
    ) -> tuple[str, bool]:
        """Drop configured predicates (exact AST match) from AND / OR chains and WHERE clauses.

        Used only for the judge's copy of the SQL; the stored response is never changed. When
        the SQL or a predicate cannot be parsed, nothing is removed.
        """
        sqlglot = _sqlglot()
        if sqlglot is None or not predicates or not sql.strip():
            return sql, False
        exp = sqlglot.exp
        try:
            targets = {_key(sqlglot.parse_one(p, read=dialect), dialect) for p in predicates}
            statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
        except (sqlglot.errors.ParseError, sqlglot.errors.TokenError, ValueError):
            return sql, False
        changed = False

        def drop(node: Any) -> Any:
            nonlocal changed
            if isinstance(node, exp.And | exp.Or):
                if _key(node.left, dialect) in targets:
                    changed = True
                    return node.right
                if _key(node.right, dialect) in targets:
                    changed = True
                    return node.left
            if isinstance(node, exp.Where) and _key(node.this, dialect) in targets:
                changed = True
                return None
            return node

        rewritten = [s.transform(drop) for s in statements]
        if not changed:
            return sql, False
        return ";\n".join(s.sql(dialect=dialect) for s in rewritten), True


def _key(node: Any, dialect: str | None) -> str:
    inner = node.unnest() if hasattr(node, "unnest") else node
    return " ".join(str(inner.sql(dialect=dialect)).casefold().split())


class _Collector:
    def __init__(self, dialect: str | None) -> None:
        self.dialect = dialect
        self.base_tables: dict[str, None] = {}
        self.aliases: dict[str, str] = {}
        self.ctes: dict[str, None] = {}
        self.columns: list[SqlColumnRef] = []
        self.output_aliases: dict[str, None] = {}
        self.functions: dict[str, None] = {}
        self.joins: list[SqlJoin] = []
        self.filters: list[str] = []
        self.projections: list[str] = []
        self.aggregations: dict[str, None] = {}
        self.group_by: list[str] = []
        self.order_by: list[str] = []
        self.limits: list[str] = []
        self.diagnostics: list[str] = []
        self.normalized: list[str] = []

    def sql(self, node: Any) -> str:
        return str(node.sql(dialect=self.dialect))

    def collect(self, statement: exp.Expression) -> None:
        sqlglot = _sqlglot()
        exp = sqlglot.exp
        self.normalized.append(self.sql(statement))
        for cte in statement.find_all(exp.CTE):
            if cte.alias:
                self.ctes[cte.alias] = None
        cte_names = {name.casefold() for name in self.ctes}
        for table in statement.find_all(exp.Table):
            if not table.name or table.name.casefold() in cte_names:
                continue
            qualified = ".".join(p for p in (table.catalog, table.db, table.name) if p)
            self.base_tables[qualified] = None
            if table.alias:
                self.aliases[table.alias] = qualified
        for select in statement.find_all(exp.Select):
            for projection in select.expressions:
                if isinstance(projection, exp.Alias) and projection.alias:
                    self.output_aliases[projection.alias] = None
            self._clauses(select, exp, top=select is statement or select.parent is statement)
        for join in statement.find_all(exp.Join):
            side = " ".join(p for p in (join.side, join.kind) if p)
            on = join.args.get("on")
            using = join.args.get("using")
            predicate = (
                self.sql(on)
                if on is not None
                else ("USING (" + ", ".join(self.sql(u) for u in using) + ")" if using else None)
            )
            self.joins.append(
                SqlJoin(
                    kind=f"{side} JOIN".strip().upper(),
                    target=self.sql(join.this),
                    predicate=predicate,
                )
            )
        for func in statement.find_all(exp.Func):
            name = func.name if isinstance(func, exp.Anonymous) else func.sql_name()
            if name:
                self.functions[str(name).upper()] = None
            if isinstance(func, exp.AggFunc):
                self.aggregations[self.sql(func)] = None
        self._resolve_columns(statement, exp)

    def _clauses(self, select: Any, exp: Any, *, top: bool) -> None:
        if top:
            self.projections.extend(self.sql(p) for p in select.expressions)
        where = select.args.get("where")
        if where is not None:
            self.filters.append(self.sql(where.this))
        having = select.args.get("having")
        if having is not None:
            self.filters.append(f"HAVING {self.sql(having.this)}")
        group = select.args.get("group")
        if group is not None:
            self.group_by.extend(self.sql(g) for g in group.expressions)
        order = select.args.get("order")
        if order is not None:
            self.order_by.extend(self.sql(o) for o in order.expressions)
        limit = select.args.get("limit")
        if limit is not None:
            value = limit.args.get("expression") or limit.this
            if value is not None:
                self.limits.append(self.sql(value))

    def _resolve_columns(self, statement: Any, exp: Any) -> None:
        """Resolve each column reference to a base table per scope (CTE / subquery aware)."""
        try:
            scope_module = importlib.import_module("sqlglot.optimizer.scope")
            scopes = list(scope_module.traverse_scope(statement))
        except Exception as exc:  # scope analysis is best effort; the parse itself succeeded
            self.diagnostics.append(f"column scope resolution failed: {type(exc).__name__}")
            scopes = []
        seen: set[tuple[str, str | None]] = set()
        aliases = {alias.casefold() for alias in self.output_aliases}
        for scope in scopes:
            base_sources = {
                name: source
                for name, source in scope.sources.items()
                if isinstance(source, exp.Table)
                and source.name.casefold() not in {c.casefold() for c in self.ctes}
            }
            for column in scope.columns:
                if not column.name or isinstance(column.this, exp.Star):
                    continue
                qualifier = column.table or None
                table: str | None = None
                if qualifier:
                    source = base_sources.get(qualifier)
                    if source is None:
                        continue  # a CTE / derived-table column, not a base identifier
                    table = source.name
                elif column.name.casefold() in aliases and len(base_sources) != 1:
                    continue  # a reference to an output alias (e.g. ORDER BY total)
                elif len(base_sources) == 1:
                    table = next(iter(base_sources.values())).name
                elif not base_sources:
                    continue  # selects only from CTEs / subqueries
                key = (column.name.casefold(), table.casefold() if table else None)
                if key in seen:
                    continue
                seen.add(key)
                self.columns.append(
                    SqlColumnRef(column=column.name, table=table, qualifier=qualifier)
                )

    def result(self, sql: str, count: int) -> SqlStructure:
        return SqlStructure(
            raw_sql=sql,
            dialect=self.dialect,
            parsed=True,
            diagnostics=self.diagnostics,
            statement_count=count,
            normalized_sql=";\n".join(self.normalized),
            base_tables=list(self.base_tables),
            table_aliases=self.aliases,
            ctes=list(self.ctes),
            base_columns=self.columns,
            output_aliases=list(self.output_aliases),
            functions=list(self.functions),
            joins=self.joins,
            filters=self.filters,
            projections=self.projections,
            aggregations=list(self.aggregations),
            group_by=self.group_by,
            order_by=self.order_by,
            limits=self.limits,
        )


def compare_with_schema(
    structure: SqlStructure, allowed: Mapping[str, Iterable[str]]
) -> SchemaComparison | None:
    """Base identifiers not in the allowed tables/columns (case-insensitive; the last name part
    of a qualified table is compared). None when the SQL was not parsed."""
    if not structure.parsed:
        return None
    tables = {name.casefold(): {c.casefold() for c in cols} for name, cols in allowed.items()}
    unknown_tables = [
        t for t in structure.base_tables if t.rsplit(".", 1)[-1].casefold() not in tables
    ]
    unknown_columns: list[str] = []
    unresolved: list[str] = []
    for ref in structure.base_columns:
        if ref.table is None:
            if not any(ref.column.casefold() in cols for cols in tables.values()):
                unresolved.append(ref.column)
            continue
        columns = tables.get(ref.table.rsplit(".", 1)[-1].casefold())
        if columns is not None and ref.column.casefold() not in columns:
            unknown_columns.append(f"{ref.table}.{ref.column}")
    return SchemaComparison(
        unknown_tables=unknown_tables,
        unknown_columns=unknown_columns,
        unresolved_columns=unresolved,
    )
