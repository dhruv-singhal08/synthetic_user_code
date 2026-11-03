"""The pluggable SQL AST parser: supporting evidence only, never executes SQL."""

from __future__ import annotations

from synthetic_user.sql.parser import SqlglotParser

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


def test_join_keys_resolve_aliases_to_base_tables_and_skip_cte_sides() -> None:
    s = PARSER.parse(SQL, dialect=None)
    keys = {((k.left.table, k.left.column), (k.right.table, k.right.column)) for k in s.join_keys}
    assert (("card_transactions", "emp_id"), ("employees", "id")) in keys
    assert (("card_transactions", "dept"), ("departments", "id")) in keys
    # d.id = m.dept_id joins a CTE: not a base-table join key
    assert not any("dept_id" in (k.left.column, k.right.column) for k in s.join_keys)


def test_qualified_tables_keep_their_qualifier_on_tables_and_columns() -> None:
    s = PARSER.parse(
        "SELECT d.cg_id, x.amt FROM ns.tbl_a d JOIN tbl_b x ON x.k = d.k "
        "WHERE d.yr IN (SELECT yr FROM tbl_c)",
        dialect=None,
    )
    assert set(s.base_tables) == {"ns.tbl_a", "tbl_b", "tbl_c"}
    resolved = {(c.table, c.column) for c in s.base_columns}
    assert ("ns.tbl_a", "cg_id") in resolved and ("tbl_c", "yr") in resolved
    assert s.join_keys[0].right.table == "ns.tbl_a"


def test_parse_failure_is_explicit_and_never_raises() -> None:
    for sql in ("SELEC broken", "SELECT FROM WHERE (", ""):
        s = PARSER.parse(sql, dialect=None)
        assert not s.parsed and s.diagnostics, sql
        assert s.raw_sql == sql and not s.base_tables and not s.base_columns  # nothing invented
    assert "\x1b" not in PARSER.parse("SELECT FROM WHERE (", dialect=None).diagnostics[0]


def test_dialect_is_only_what_is_configured() -> None:
    s = PARSER.parse("SELECT `a` FROM `t`", dialect="bigquery")
    assert s.parsed and s.dialect == "bigquery" and s.base_tables == ["t"]
