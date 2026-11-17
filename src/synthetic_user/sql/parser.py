"""Pluggable SQL AST parser: supporting evidence for the SQL metric rubrics (never a judge).

``SqlParser`` is the interface; ``SqlglotParser`` is the default implementation. The parser
extracts base tables, base columns (aliases resolved per scope), CTE names, output aliases,
functions, joins with predicates and their equality join keys (resolved to base tables),
filters, projections, aggregations, grouping, ordering and limits, the columns compared with a
text value (D261), and keeps the raw SQL. Base
database identifiers are kept apart from CTE names, output aliases and functions so Schema
Faithfulness never compares those. Identifier validity is decided elsewhere, against the
catalog (``sql/reconcile.py``); the parser never compares with a reference.

The dialect comes only from explicit configuration (``evaluation.sql_dialect``); it is never
guessed. A parse failure is explicit (``parsed=False`` + diagnostics) and never fails a metric
by itself: the judge still sees the original SQL. SQL is never executed.
"""

from __future__ import annotations

import importlib
import re
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


class SqlValueFilter(StrictModel):
    """A column compared with text values (=, <>, IN, LIKE / ILIKE, also inside a function such
    as UPPER(col)), anywhere in the statement: its catalog values are looked up (D261)."""

    column: SqlColumnRef
    values: list[NonEmptyStr] = Field(default_factory=list)  # the literals as written


class SqlJoin(StrictModel):
    kind: NonEmptyStr  # e.g. "JOIN", "LEFT JOIN"
    target: NonEmptyStr  # the joined table / derived source as written
    predicate: str | None = None


class SqlJoinKey(StrictModel):
    """One ``a.x = b.y`` equality of an ON clause between two base-table columns."""

    left: SqlColumnRef
    right: SqlColumnRef


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
    join_keys: list[SqlJoinKey] = Field(default_factory=list)
    filters: list[NonEmptyStr] = Field(default_factory=list)
    projections: list[NonEmptyStr] = Field(default_factory=list)
    aggregations: list[NonEmptyStr] = Field(default_factory=list)
    group_by: list[NonEmptyStr] = Field(default_factory=list)
    order_by: list[NonEmptyStr] = Field(default_factory=list)
    limits: list[NonEmptyStr] = Field(default_factory=list)
    value_filters: list[SqlValueFilter] = Field(default_factory=list)


class SqlParser(Protocol):
    name: str

    def parse(self, sql: str, *, dialect: str | None) -> SqlStructure: ...


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


def _qualified(table: Any) -> str:
    """The base table name as written, qualifiers included (``ns.table``)."""
    return ".".join(p for p in (table.catalog, table.db, table.name) if p)


def _single_column(side: Any, exp: Any) -> Any:
    """The column of a comparison side: the column itself, or the only column inside a function
    (``UPPER(col)``, ``TRIM(col)``, ``CAST(col AS ...)``); None otherwise."""
    if isinstance(side, exp.Column):
        return side
    if isinstance(side, exp.Func):
        columns = list(side.find_all(exp.Column))
        return columns[0] if len(columns) == 1 else None
    return None


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
        self.join_keys: list[SqlJoinKey] = []
        self.filters: list[str] = []
        self.projections: list[str] = []
        self.aggregations: dict[str, None] = {}
        self.group_by: list[str] = []
        self.order_by: list[str] = []
        self.limits: list[str] = []
        self.value_filters: dict[tuple[str, str | None], SqlValueFilter] = {}
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
            qualified = _qualified(table)
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
            if on is not None:
                self._join_keys(on, exp)
        for func in statement.find_all(exp.Func):
            name = func.name if isinstance(func, exp.Anonymous) else func.sql_name()
            if name:
                self.functions[str(name).upper()] = None
            if isinstance(func, exp.AggFunc):
                self.aggregations[self.sql(func)] = None
        self._resolve_columns(statement, exp)
        self._value_filters(statement, exp)

    def _join_keys(self, on: Any, exp: Any) -> None:
        """``a.x = b.y`` equalities between columns of base tables (CTE / derived sides and
        USING joins are skipped: they are not base identifiers or have no written sides)."""
        for eq in on.find_all(exp.EQ):
            left, right = eq.this, eq.expression
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                continue
            refs = [self._base_ref(left), self._base_ref(right)]
            if refs[0] is not None and refs[1] is not None:
                self.join_keys.append(SqlJoinKey(left=refs[0], right=refs[1]))

    def _base_ref(self, column: Any) -> SqlColumnRef | None:
        qualifier = column.table or None
        if qualifier is None:
            return SqlColumnRef(column=column.name, table=None, qualifier=None)
        aliases = {alias.casefold(): table for alias, table in self.aliases.items()}
        table = aliases.get(qualifier.casefold())
        if table is None:
            table = next(
                (
                    t
                    for t in self.base_tables
                    if t.rsplit(".", 1)[-1].casefold() == qualifier.casefold()
                ),
                None,
            )
        if table is None:
            return None  # a CTE / derived-table qualifier
        return SqlColumnRef(column=column.name, table=table, qualifier=qualifier)

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
                    table = _qualified(source)
                elif column.name.casefold() in aliases and len(base_sources) != 1:
                    continue  # a reference to an output alias (e.g. ORDER BY total)
                elif len(base_sources) == 1:
                    table = _qualified(next(iter(base_sources.values())))
                elif not base_sources:
                    continue  # selects only from CTEs / subqueries
                key = (column.name.casefold(), table.casefold() if table else None)
                if key in seen:
                    continue
                seen.add(key)
                self.columns.append(
                    SqlColumnRef(column=column.name, table=table, qualifier=qualifier)
                )

    def _value_filters(self, statement: Any, exp: Any) -> None:
        """Columns compared with text literals; an unqualified column takes the base table its
        scope resolved it to when that is unambiguous. CTE / derived columns are skipped."""
        for node in statement.find_all(exp.EQ, exp.NEQ, exp.Like, exp.ILike, exp.In):
            if isinstance(node, exp.In):
                sides = [(node.this, list(node.expressions))]
            else:
                sides = [(node.this, [node.expression]), (node.expression, [node.this])]
            for side, others in sides:
                column = _single_column(side, exp)
                texts = [
                    str(o.this)
                    for o in others
                    if isinstance(o, exp.Literal) and o.is_string and str(o.this).strip()
                ]
                ref = self._base_ref(column) if column is not None and texts else None
                if ref is None:
                    continue
                if ref.table is None:
                    name = ref.column.casefold()
                    tables = {c.table for c in self.columns if c.column.casefold() == name}
                    if len(tables) == 1:
                        ref = ref.model_copy(update={"table": tables.pop()})
                key = (ref.column.casefold(), ref.table.casefold() if ref.table else None)
                found = self.value_filters.setdefault(key, SqlValueFilter(column=ref))
                found.values.extend(t for t in texts if t not in found.values)

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
            join_keys=self.join_keys,
            filters=self.filters,
            projections=self.projections,
            aggregations=list(self.aggregations),
            group_by=self.group_by,
            order_by=self.order_by,
            limits=self.limits,
            value_filters=list(self.value_filters.values()),
        )
