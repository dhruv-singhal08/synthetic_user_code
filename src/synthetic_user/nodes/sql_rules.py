"""Deterministic SQL rules that override the LLM evaluator (card 16).

Deliberately minimal (the full Stage A analyzer of card 14 is skipped, D085): it parses with
sqlglot and walks the AST only to detect
- parse failures (incl. multiple statements),
- non-SELECT statements,
- unknown columns on catalog tables,
- joins that contradict a verified catalog relationship,
- tables that are not in the catalog ("unverifiable").
No regex is used on SQL structure. This is not a semantic correctness check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from synthetic_user.models.evaluation import Severity
from synthetic_user.models.reference import ReferenceCatalog, VerificationStatus


@dataclass(frozen=True)
class RuleFinding:
    category: str
    description: str
    expected_behavior: str
    observed_behavior: str
    sql_evidence: str
    severity: Severity


@dataclass
class SqlRuleResult:
    parse_error: str | None = None
    non_select: str | None = None  # statement type, e.g. "Delete"
    findings: list[RuleFinding] = field(default_factory=list)
    unverifiable_tables: list[str] = field(default_factory=list)

    @property
    def blocks_llm(self) -> bool:
        """Parse failures and non-SELECT statements are decided without the LLM."""
        return self.parse_error is not None or self.non_select is not None


def apply_sql_rules(sql: str, dialect: str, catalog: ReferenceCatalog) -> SqlRuleResult:
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except SqlglotError as exc:
        return SqlRuleResult(parse_error=str(exc).splitlines()[0])
    if not statements:
        return SqlRuleResult(parse_error="empty SQL")
    if len(statements) > 1:
        return SqlRuleResult(parse_error=f"{len(statements)} statements; expected exactly one")

    statement = statements[0]
    if not isinstance(statement, exp.Query):
        return SqlRuleResult(non_select=type(statement).__name__)

    schema = _Schema(catalog)
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

    result = SqlRuleResult(unverifiable_tables=unverifiable)
    result.findings.extend(
        _unknown_columns(statement, schema, aliases, cte_names | derived, dialect)
    )
    result.findings.extend(_join_contradictions(statement, catalog, schema, aliases, dialect))
    return result


class _Schema:
    """Case-insensitive view of the catalog."""

    def __init__(self, catalog: ReferenceCatalog) -> None:
        self.columns = {t.name.lower(): {c.name.lower() for c in t.columns} for t in catalog.tables}

    def has_table(self, table: str) -> bool:
        return table in self.columns

    def has_column(self, table: str, column: str) -> bool:
        return column in self.columns.get(table, set())


def _unknown_columns(
    statement: exp.Query,
    schema: _Schema,
    aliases: dict[str, str],
    non_catalog_sources: set[str],
    dialect: str,
) -> list[RuleFinding]:
    select_aliases = {a.alias.lower() for a in statement.find_all(exp.Alias)}
    catalog_sources = {t for t in set(aliases.values()) if schema.has_table(t)}
    only_catalog_sources = not non_catalog_sources and catalog_sources == set(aliases.values())
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
                not only_catalog_sources
                or name in select_aliases
                or any(schema.has_column(t, name) for t in catalog_sources)
            ):
                continue
            problem = f"column {column.name!r} does not exist on {sorted(catalog_sources)}"
        evidence = column.sql(dialect=dialect)
        if evidence in seen:
            continue
        seen.add(evidence)
        findings.append(
            RuleFinding(
                category="column_selection",
                description=problem,
                expected_behavior="use only columns that exist in the catalog",
                observed_behavior=f"SQL references {evidence}",
                sql_evidence=evidence,
                severity=Severity.HIGH,
            )
        )
    return findings


def _join_contradictions(
    statement: exp.Query,
    catalog: ReferenceCatalog,
    schema: _Schema,
    aliases: dict[str, str],
    dialect: str,
) -> list[RuleFinding]:
    verified = [r for r in catalog.relationships if r.verification is VerificationStatus.VERIFIED]
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
