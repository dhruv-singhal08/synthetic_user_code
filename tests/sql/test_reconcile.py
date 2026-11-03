"""SQLGlot output reconciled with the cheat sheet AND the full catalog (section 18, 11-20).

The cheat sheet is expected evidence, never an allow-list: an identifier it did not select is
``valid_extra`` when the catalog has it. Uses the SYNTHETIC fixture catalog (D229).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synthetic_user.models.lookup import (
    ArtifactReconciliation,
    IdentifierStatus,
    JoinStatus,
    LookupTool,
)
from synthetic_user.models.reference import CatalogPackDraft, Reference, ReferencePack
from synthetic_user.references.catalog_index import CatalogIndex
from synthetic_user.references.loader import load_reference
from synthetic_user.references.pack import build_pack
from synthetic_user.sql.parser import SqlglotParser
from synthetic_user.sql.reconcile import evidence_requests, reconcile

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "references" / "sql_eval_catalog.yaml"
PARSER = SqlglotParser()


@pytest.fixture(scope="module")
def reference() -> Reference:
    return load_reference(FIXTURE, max_chars=200_000)


@pytest.fixture(scope="module")
def index(reference: Reference) -> CatalogIndex:
    built = CatalogIndex.for_reference(reference)
    assert built is not None
    return built


@pytest.fixture(scope="module")
def pack(reference: Reference) -> ReferencePack:
    draft = CatalogPackDraft.model_validate(
        {
            "tables": [
                {
                    "table": "v_lpc_details",
                    "columns": ["bill_cycle_yr_mo_no", "net_lpc_charge_usd_am"],
                    "why": "net LPC by billing month",
                }
            ],
            "rules": [],
        }
    )
    built, _ = build_pack(draft, reference, "SCN_001")
    assert built is not None
    return built


def run(
    sql: str, index: CatalogIndex, pack: ReferencePack, dialect: str | None = None
) -> ArtifactReconciliation:
    return reconcile(PARSER.parse(sql, dialect=dialect), artifact=1, index=index, pack=pack)


def statuses(rec: ArtifactReconciliation) -> dict[str, str]:
    return {i.identifier: i.status.value for i in rec.identifiers}


def test_table_aliases_resolve_to_catalog_columns(index: CatalogIndex, pack: ReferencePack) -> None:
    rec = run("SELECT d.net_lpc_charge_usd_am FROM v_lpc_details d", index, pack)  # 11
    assert statuses(rec) == {
        "v_lpc_details": "valid_in_pack",
        "v_lpc_details.net_lpc_charge_usd_am": "valid_in_pack",
    }
    assert rec.pack_items_not_used == ["v_lpc_details.bill_cycle_yr_mo_no"]


def test_output_aliases_and_ctes_are_never_physical_identifiers(
    index: CatalogIndex, pack: ReferencePack
) -> None:
    rec = run(  # 12 + 13
        "WITH monthly AS (SELECT bill_cycle_yr_mo_no AS billing_month, "
        "SUM(net_lpc_charge_usd_am) AS net_lpc_usd FROM v_lpc_details GROUP BY 1) "
        "SELECT m.billing_month, m.net_lpc_usd FROM monthly m ORDER BY net_lpc_usd",
        index,
        pack,
    )
    names = set(statuses(rec))
    assert not names & {"monthly", "billing_month", "net_lpc_usd", "m.billing_month"}
    assert rec.with_status(IdentifierStatus.INVALID) == []
    assert rec.with_status(IdentifierStatus.UNRESOLVED) == []


def test_subquery_tables_are_valid_extra_and_bring_their_rules(
    index: CatalogIndex, pack: ReferencePack
) -> None:
    rec = run(  # 14
        "SELECT SUM(net_lpc_charge_usd_am) FROM v_lpc_details WHERE acct_id IN "
        "(SELECT acct_id FROM v_acct_delinquency WHERE dlq_bucket_cd = '90')",
        index,
        pack,
    )
    found = statuses(rec)
    assert found["v_acct_delinquency"] == "valid_extra"
    assert found["v_acct_delinquency.dlq_bucket_cd"] == "valid_extra"
    assert found["v_lpc_details.acct_id"] == "valid_extra"  # not selected, but it exists
    assert rec.extra_tables == ["v_acct_delinquency"]
    assert rec.rules_not_in_pack == ["R1", "R2", "R3"]  # unselected catalog rules: candidates


def test_multiple_joins_are_checked_against_documented_relationships(
    index: CatalogIndex, pack: ReferencePack
) -> None:
    rec = run(  # 15
        "SELECT l.net_lpc_charge_usd_am, d.dlq_bucket_cd, e.epa_emissions_kg, c.cm_name "
        "FROM v_lpc_details l "
        "JOIN v_acct_delinquency d ON l.acct_id = d.acct_id "
        "JOIN v_emissions e ON e.acct_id = l.acct_id "
        "JOIN v_cardmember c ON c.acct_id = l.acct_id",
        index,
        pack,
    )
    joins = {(j.left, j.right): j.status for j in rec.joins}
    assert joins[("v_lpc_details.acct_id", "v_acct_delinquency.acct_id")] is (
        JoinStatus.DOCUMENTED_VERIFIED
    )
    assert joins[("v_emissions.acct_id", "v_lpc_details.acct_id")] is (
        JoinStatus.DOCUMENTED_UNVERIFIED
    )
    assert joins[("v_cardmember.acct_id", "v_lpc_details.acct_id")] is JoinStatus.UNDOCUMENTED
    assert "v_cardmember#R1" in rec.rules_not_in_pack
    requests = evidence_requests([rec], index)
    pairs = [r.pair for r in requests if r.tool is LookupTool.LOOKUP_RELATIONSHIPS]
    assert len(pairs) == 6  # every pair of the four joined tables, looked up exactly
    (rules,) = [r for r in requests if r.tool is LookupTool.LOOKUP_RULES]
    assert set(rules.tables) == {"v_acct_delinquency", "v_emissions", "v_cardmember"}


def test_extra_key_on_a_documented_join_is_not_an_undocumented_join(
    index: CatalogIndex, pack: ReferencePack
) -> None:
    rec = run(
        "SELECT 1 FROM v_lpc_details l JOIN v_acct_delinquency d ON l.acct_id = d.acct_id "
        "AND l.bill_cycle_yr_mo_no = d.bill_cycle_yr_mo_no",
        index,
        pack,
    )
    assert [j.status for j in rec.joins] == [
        JoinStatus.DOCUMENTED_VERIFIED,
        JoinStatus.EXTRA_CONDITION,
    ]


def test_missing_selected_column_is_listed_not_failed(
    index: CatalogIndex, pack: ReferencePack
) -> None:
    rec = run("SELECT bill_cycle_yr_mo_no FROM v_lpc_details", index, pack)  # 16
    assert rec.pack_items_not_used == ["v_lpc_details.net_lpc_charge_usd_am"]
    assert rec.with_status(IdentifierStatus.INVALID) == []


def test_alternative_valid_table_is_valid_extra_and_described(
    index: CatalogIndex, pack: ReferencePack
) -> None:
    rec = run("SELECT DISTINCT bill_cycle_yr_mo_no FROM v_acct_delinquency", index, pack)  # 17
    assert statuses(rec)["v_acct_delinquency.bill_cycle_yr_mo_no"] == "valid_extra"
    assert rec.pack_items_not_used == [
        "v_lpc_details",
        "v_lpc_details.bill_cycle_yr_mo_no",
        "v_lpc_details.net_lpc_charge_usd_am",
    ]
    schema = [r for r in evidence_requests([rec], index) if r.tool is LookupTool.LOOKUP_SCHEMA]
    assert [(r.table, r.column) for r in schema] == [("v_acct_delinquency", "bill_cycle_yr_mo_no")]


def test_parser_failure_reconciles_nothing_and_requests_nothing(
    index: CatalogIndex, pack: ReferencePack
) -> None:
    rec = run("SELEC broken FROM", index, pack)  # 18
    assert not rec.parsed and rec.diagnostics and rec.identifiers == []
    assert evidence_requests([rec], index) == []


def test_unresolved_identifiers_are_not_called_invalid(
    reference: Reference, index: CatalogIndex, pack: ReferencePack
) -> None:
    ambiguous = run(  # 19: unqualified column present in both joined tables
        "SELECT acct_id FROM v_lpc_details l JOIN v_acct_delinquency d ON l.acct_id = d.acct_id",
        index,
        pack,
    )
    (check,) = [i for i in ambiguous.identifiers if i.identifier == "acct_id"]
    assert check.status is IdentifierStatus.UNRESOLVED and "several tables" in check.reason
    strict = CatalogIndex.for_reference(reference, namespaces=["gcsinsight"])
    assert strict is not None
    refused = run("SELECT cg_id FROM other_ns.v_lpc_details", strict, pack)
    assert {i.status for i in refused.identifiers} == {IdentifierStatus.UNRESOLVED}
    allowed = run("SELECT cg_id FROM gcsinsight.v_lpc_details", strict, pack)
    assert statuses(allowed)["gcsinsight.v_lpc_details.cg_id"] == "valid_extra"


def test_invalid_column_and_table_with_exact_alternatives(
    index: CatalogIndex, pack: ReferencePack
) -> None:
    rec = run("SELECT x.dlq_bucket_cd, y.z FROM v_lpc_details x, no_table y", index, pack)
    found = statuses(rec)
    assert found["v_lpc_details.dlq_bucket_cd"] == "invalid"
    assert found["no_table"] == "invalid" and found["no_table.z"] == "unresolved"
    (alt,) = [
        r
        for r in evidence_requests([rec], index)
        if r.tool is LookupTool.LOOKUP_COLUMN_ALTERNATIVES
    ]
    assert alt.column == "dlq_bucket_cd" and alt.excluded_table == "v_lpc_details"


def test_unsupported_dialect_construct_is_an_explicit_parse_failure(
    index: CatalogIndex, pack: ReferencePack
) -> None:
    sql = "SELECT TOP 5 acct_id FROM v_lpc_details"  # 20: T-SQL syntax
    assert run(sql, index, pack, dialect="tsql").parsed
    rec = run(sql, index, pack, dialect="postgres")
    assert not rec.parsed and "parse error" in rec.diagnostics[0]
    assert rec.identifiers == []  # never fabricated
