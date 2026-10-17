"""SQL plugin Stage A: static analysis of product SQL with the SQLGlot AST (never executed).

Walks the AST (no regex on SQL structure) for tables, CTEs, aliases, columns, joins with their
equality pairs, filters, date expressions, aggregations, group/order/limit, identifier issues
(unknown column on an index table vs table not in the index = unverifiable) and join checks
against index relationships. Deterministic findings (parse failure, non-SELECT, unknown
columns, joins contradicting a verified relationship) override the LLM in plugin.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from synthetic_user.models.evaluation import Severity
from synthetic_user.models.reference import ReferenceIndex, VerificationStatus
from synthetic_user.nodes.evaluators.plugins.sql.models import JoinInfo, SqlAnalysis

_DATE_NODES: tuple[type[exp.Expression], ...] = (
    exp.CurrentDate,
    exp.CurrentTimestamp,
    exp.DateAdd,
    exp.DateSub,
    exp.DateDiff,
    exp.DateTrunc,
    exp.TimestampTrunc,
    exp.DatetimeTrunc,
    exp.TimestampSub,
    exp.DatetimeSub,
    exp.TsOrDsAdd,
    exp.Extract,
    exp.StrToDate,
    exp.Interval,
)


@dataclass(frozen=True)
class RuleFinding:
    category: str
    description: str
    expected_behavior: str
    observed_behavior: str
    sql_evidence: str
    severity: Severity


@dataclass
class StageAResult:
    analysis: SqlAnalysis
    parse_error: str | None = None
    non_select: str | None = None  # statement type, e.g. "Delete"
    findings: list[RuleFinding] = field(default_factory=list)

    @property
    def unverifiable_tables(self) -> list[str]:
        return self.analysis.unverifiable_tables

    @property
    def blocks_llm(self) -> bool:
        """Parse failures and non-SELECT statements are decided without the LLM."""
        return self.parse_error is not None or self.non_select is not None


def analyze_sql(sql: str, dialect: str, index: ReferenceIndex) -> StageAResult:
    """Stage A for one SQL string: analysis + deterministic findings."""

    def failed(problem: str) -> StageAResult:
        analysis = SqlAnalysis(dialect=dialect, parse_ok=False, parse_errors=[problem])
        return StageAResult(analysis, parse_error=problem)

    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except SqlglotError as exc:
        return failed(str(exc).splitlines()[0])
    if not statements:
        return failed("empty SQL")
    if len(statements) > 1:
        return failed(f"{len(statements)} statements; expected exactly one")

    statement = statements[0]
    if not isinstance(statement, exp.Query):
        kind = type(statement).__name__
        analysis = SqlAnalysis(dialect=dialect, parse_ok=True, statement_type=kind)
        return StageAResult(analysis, non_select=kind)

    schema = _Schema(index)
    cte_names = {c.alias_or_name.lower() for c in statement.find_all(exp.CTE)}
    derived = {s.alias_or_name.lower() for s in statement.find_all(exp.Subquery) if s.alias}
    aliases: dict[str, str] = {}
    unverifiable: list[str] = []
    for table in statement.find_all(exp.Table):
        name = table.name.lower()
        if name in cte_names:
            continue
        aliases[table.alias_or_name.lower()] = name
        aliases[name] = name
        if not schema.has_table(name) and table.name not in unverifiable:
            unverifiable.append(table.name)

    unknown = _unknown_columns(statement, schema, aliases, cte_names | derived, dialect)
    contradictions = _join_contradictions(statement, index, schema, aliases, dialect)
    analysis = _inventory(statement, dialect, index, aliases, sorted(cte_names), unverifiable)
    analysis = analysis.model_copy(
        update={
            "unknown_columns": [f.sql_evidence for f in unknown],
            "join_issues": [*(f.description for f in contradictions), *analysis.join_issues],
        }
    )
    return StageAResult(analysis, findings=[*unknown, *contradictions])


class _Schema:
    """Case-insensitive view of the reference index."""

    def __init__(self, index: ReferenceIndex) -> None:
        self.columns = {t.name.lower(): {c.name.lower() for c in t.columns} for t in index.tables}

    def has_table(self, table: str) -> bool:
        return table in self.columns

    def has_column(self, table: str, column: str) -> bool:
        return column in self.columns.get(table, set())


def _inventory(
    statement: exp.Query,
    dialect: str,
    index: ReferenceIndex,
    aliases: dict[str, str],
    ctes: list[str],
    unverifiable: list[str],
) -> SqlAnalysis:
    def sql(node: Any) -> str:  # any sqlglot node
        return str(node.sql(dialect=dialect))

    def unique(items: list[str]) -> list[str]:
        return list(dict.fromkeys(i for i in items if i))

    time_columns = {
        (t.name.lower(), c.name.lower()) for t in index.tables for c in t.time_columns
    } | {
        (t.name.lower(), c.name.lower())
        for t in index.tables
        for c in t.columns
        if c.data_type and c.data_type.upper().startswith(("DATE", "TIME"))
    }
    sources = set(aliases.values())

    joins: list[JoinInfo] = []
    join_issues: list[str] = []
    for join in statement.find_all(exp.Join):
        condition = join.args.get("on")
        pairs: list[tuple[str, str]] = []
        equalities = list(condition.find_all(exp.EQ)) if condition is not None else []
        for eq in equalities:
            if not (isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Column)):
                continue
            pairs.append((sql(eq.left), sql(eq.right)))
            lt = aliases.get(eq.left.table.lower(), eq.left.table.lower())
            rt = aliases.get(eq.right.table.lower(), eq.right.table.lower())
            left, right = f"{lt}.{eq.left.name}", f"{rt}.{eq.right.name}"
            both_indexed = index.get_table(lt) is not None and index.get_table(rt) is not None
            if lt != rt and both_indexed and index.verification(left, right) is None:
                join_issues.append(f"join {left} = {right} has no relationship in the index")
        side = " ".join(str(p) for p in (join.side, join.kind) if p)
        joins.append(
            JoinInfo(
                kind=f"{side} JOIN".strip(),
                table=sql(join.this),
                condition=sql(condition) if condition is not None else None,
                equality_pairs=pairs,
            )
        )

    filters = [sql(w.this) for w in statement.find_all(exp.Where)]
    filters += [sql(h.this) for h in statement.find_all(exp.Having)]
    dates = [sql(n) for n in statement.find_all(*_DATE_NODES)]
    for where in statement.find_all(exp.Where):
        for column in where.find_all(exp.Column):
            name = column.name.lower()
            tables = (
                {aliases.get(column.table.lower(), column.table.lower())}
                if column.table
                else sources
            )
            if any((t, name) in time_columns for t in tables):
                parent = column.parent
                dates.append(sql(parent) if parent is not None else sql(column))

    limit = statement.find(exp.Limit)
    limit_value = limit.args.get("expression") if limit is not None else None
    return SqlAnalysis(
        dialect=dialect,
        parse_ok=True,
        statement_type=type(statement).__name__,
        tables=unique(
            [sql(t) for t in statement.find_all(exp.Table) if t.name.lower() not in ctes]
        ),
        ctes=ctes,
        aliases={k: v for k, v in aliases.items() if k != v},
        columns=unique([sql(c) for c in statement.find_all(exp.Column)]),
        joins=joins,
        filters=unique(filters),
        date_expressions=unique(dates),
        aggregations=unique([sql(a) for a in statement.find_all(exp.AggFunc)]),
        group_by=unique([sql(e) for g in statement.find_all(exp.Group) for e in g.expressions]),
        order_by=unique([sql(e) for o in statement.find_all(exp.Order) for e in o.expressions]),
        limit=sql(limit_value) if isinstance(limit_value, exp.Expression) else None,
        unverifiable_tables=unverifiable,
        join_issues=unique(join_issues),
    )


def _unknown_columns(
    statement: exp.Query,
    schema: _Schema,
    aliases: dict[str, str],
    non_index_sources: set[str],
    dialect: str,
) -> list[RuleFinding]:
    select_aliases = {a.alias.lower() for a in statement.find_all(exp.Alias)}
    index_sources = {t for t in set(aliases.values()) if schema.has_table(t)}
    only_index_sources = not non_index_sources and index_sources == set(aliases.values())
    findings: list[RuleFinding] = []
    seen: set[str] = set()
    for column in statement.find_all(exp.Column):
        name = column.name.lower()
        qualifier = column.table.lower()
        if qualifier:
            table = aliases.get(qualifier)
            if table is None or not schema.has_table(table) or schema.has_column(table, name):
                continue
            problem = f"column {column.name!r} does not exist on table {table!r}"
        else:
            if (
                not only_index_sources
                or name in select_aliases
                or any(schema.has_column(t, name) for t in index_sources)
            ):
                continue
            problem = f"column {column.name!r} does not exist on {sorted(index_sources)}"
        evidence = column.sql(dialect=dialect)
        if evidence in seen:
            continue
        seen.add(evidence)
        findings.append(
            RuleFinding(
                category="column_selection",
                description=problem,
                expected_behavior="use only columns that exist in the reference index",
                observed_behavior=f"SQL references {evidence}",
                sql_evidence=evidence,
                severity=Severity.HIGH,
            )
        )
    return findings


def _join_contradictions(
    statement: exp.Query,
    index: ReferenceIndex,
    schema: _Schema,
    aliases: dict[str, str],
    dialect: str,
) -> list[RuleFinding]:
    verified = [r for r in index.relationships if r.verification is VerificationStatus.VERIFIED]
    findings: list[RuleFinding] = []
    for join in statement.find_all(exp.Join):
        condition = join.args.get("on")
        if condition is None:
            continue
        for eq in condition.find_all(exp.EQ):
            left, right = eq.left, eq.right
            if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
                continue
            lt, rt = aliases.get(left.table.lower()), aliases.get(right.table.lower())
            if lt is None or rt is None or lt == rt:
                continue
            if not (schema.has_table(lt) and schema.has_table(rt)):
                continue
            between = [
                r for r in verified if {r.source_table.lower(), r.target_table.lower()} == {lt, rt}
            ]
            used = {(lt, left.name.lower()), (rt, right.name.lower())}
            if not between or any(
                used
                == {
                    (r.source_table.lower(), r.source_column.lower()),
                    (r.target_table.lower(), r.target_column.lower()),
                }
                for r in between
            ):
                continue
            expected = "; ".join(
                f"{r.source_table}.{r.source_column} = {r.target_table}.{r.target_column}"
                for r in between
            )
            evidence = eq.sql(dialect=dialect)
            findings.append(
                RuleFinding(
                    category="join_correctness",
                    description=f"join {lt} to {rt} contradicts the verified relationship",
                    expected_behavior=f"join on {expected}",
                    observed_behavior=f"joins on {evidence}",
                    sql_evidence=evidence,
                    severity=Severity.HIGH,
                )
            )
    return findings
