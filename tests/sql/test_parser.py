"""The pluggable SQL AST parser: supporting evidence only, never executes SQL."""

from __future__ import annotations

from synthetic_user.sql.parser import SqlglotParser, compare_with_schema

PARSER = SqlglotParser()
SQL = """WITH monthly AS (
  SELECT e.dept_id, SUM(t.amount) AS total
  FROM card_transactions t JOIN employees e ON t.emp_id = e.id
  WHERE t.yr = 2024 GROUP BY e.dept_id
)
SELECT d.name, m.total, COUNT(DISTINCT t2.card_id) AS cards
FROM monthly m
LEFT JOIN departments d ON d.id = m.dept_id
JOIN card_transactions t2 ON t2.dept = d.id
GROUP BY d.name, m.total
HAVING COUNT(*) > 1
ORDER BY total DESC LIMIT 10"""


def test_extracts_tables_ctes_aliases_joins_and_clauses() -> None:
    s = PARSER.parse(SQL, dialect=None)
    assert s.parsed and s.statement_count == 1 and not s.diagnostics
    assert s.ctes == ["monthly"]
    assert set(s.base_tables) == {"card_transactions", "employees", "departments"}  # no CTE
    assert s.table_aliases["t"] == "card_transactions" and s.table_aliases["d"] == "departments"
    assert set(s.output_aliases) == {"total", "cards"}
    assert {"SUM", "COUNT"} <= set(s.functions)
    assert any(j.kind == "LEFT JOIN" and j.predicate == "d.id = m.dept_id" for j in s.joins)
    assert set(s.filters) == {"t.yr = 2024", "HAVING COUNT(*) > 1"}
    assert set(s.group_by) == {"e.dept_id", "d.name", "m.total"}
    assert s.order_by == ["total DESC"] and s.limits == ["10"]
    assert "COUNT(DISTINCT t2.card_id)" in s.aggregations
    # base columns resolve through aliases; CTE and output-alias references are excluded
    resolved = {(c.table, c.column) for c in s.base_columns}
    assert ("card_transactions", "amount") in resolved and ("employees", "id") in resolved
    assert ("departments", "name") in resolved and ("card_transactions", "card_id") in resolved
    assert not any(c.column == "total" for c in s.base_columns)


def test_schema_comparison_flags_only_base_identifiers() -> None:
    s = PARSER.parse(SQL, dialect=None)
    allowed = {
        "card_transactions": ["amount", "emp_id", "yr", "card_id", "dept"],
        "employees": ["id", "dept_id"],
        "departments": ["id"],
    }
    comparison = compare_with_schema(s, allowed)
    assert comparison is not None
    assert comparison.unknown_columns == ["departments.name"]
    assert comparison.unknown_tables == []


def test_parse_failure_is_explicit_and_never_raises() -> None:
    for sql in ("SELEC broken", "SELECT FROM WHERE (", ""):
        s = PARSER.parse(sql, dialect=None)
        assert not s.parsed and s.diagnostics, sql
        assert s.raw_sql == sql
        assert compare_with_schema(s, {}) is None  # parser-dependent checks inconclusive
    assert "\x1b" not in PARSER.parse("SELECT FROM WHERE (", dialect=None).diagnostics[0]


def test_dialect_is_only_what_is_configured() -> None:
    s = PARSER.parse("SELECT `a` FROM `t`", dialect="bigquery")
    assert s.parsed and s.dialect == "bigquery" and s.base_tables == ["t"]


def test_configured_predicates_are_removed_from_the_judge_copy_only() -> None:
    sql = "SELECT a FROM t WHERE x = 1 AND cg_id IN ('DUMMY')"
    cleaned, removed = PARSER.remove_predicates(sql, ["cg_id IN ('DUMMY')"], dialect=None)
    assert removed and "DUMMY" not in cleaned and "x = 1" in cleaned
    only, _ = PARSER.remove_predicates(
        "SELECT a FROM t WHERE cg_id IN ('DUMMY')", ["cg_id IN ('DUMMY')"], dialect=None
    )
    assert "WHERE" not in only
    other, removed = PARSER.remove_predicates(
        "SELECT a FROM t WHERE cg_id IN ('REAL')", ["cg_id IN ('DUMMY')"], dialect=None
    )
    assert not removed and other == "SELECT a FROM t WHERE cg_id IN ('REAL')"
    assert PARSER.remove_predicates("SELEC broken", ["x = 1"], dialect=None) == (
        "SELEC broken",
        False,
    )
