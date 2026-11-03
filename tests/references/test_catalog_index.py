"""The reference_data format and the exact catalog lookups (instruction section 18, 1-10).

Uses the SYNTHETIC fixture tests/fixtures/references/sql_eval_catalog.yaml, not the
authoritative reference (which was not available; see D229).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.models.lookup import LookupRequest, LookupStatus, LookupTool
from synthetic_user.models.reference import DataCatalog, Reference
from synthetic_user.references.catalog_index import CatalogIndex
from synthetic_user.references.loader import load_reference, parse_reference

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "references" / "sql_eval_catalog.yaml"
BOOTSTRAP = (
    Path(__file__).resolve().parents[2] / "data" / "references" / "insight_referance(1).yaml"
)


@pytest.fixture
def reference() -> Reference:
    return load_reference(FIXTURE, max_chars=200_000)


@pytest.fixture
def index(reference: Reference) -> CatalogIndex:
    built = CatalogIndex.for_reference(reference)
    assert built is not None
    return built


# --- reference_data format ----------------------------------------------------------------------


def test_reference_data_maps_one_to_one_onto_the_data_catalog(reference: Reference) -> None:
    catalog = reference.content
    assert isinstance(catalog, DataCatalog) and reference.source_format == "reference_data"
    assert [t.name for t in catalog.tables][:2] == ["v_lpc_details", "v_acct_delinquency"]
    lpc = catalog.table("v_lpc_details")
    assert lpc is not None and lpc.rules[0].startswith("Late payment charge questions")
    market = lpc.column("mkt_cd")
    assert market is not None and market.values == ["US", "UK", "IN"]
    assert catalog.rule_ids()["R1"].startswith("DEFRA and EPA")
    assert catalog.relationships[0].verified and not catalog.relationships[1].verified


def test_bootstrap_template_is_a_valid_reference_without_invented_content() -> None:
    reference = load_reference(BOOTSTRAP, max_chars=200_000)
    catalog = reference.content
    assert isinstance(catalog, DataCatalog)
    (table,) = catalog.tables
    assert table.name == "v_lpc_details" and table.description is None and table.rules == []
    assert [c.name for c in table.columns] == [
        "bill_cycle_yr_mo_no",
        "net_lpc_charge_usd_am",
        "lpc_charge_usd_am",
        "cg_id",
    ]
    assert all(c.type is None and c.description is None and not c.values for c in table.columns)
    assert catalog.relationships == [] and catalog.rules == []


def test_reference_data_refuses_unknown_fields_and_unquoted_booleans() -> None:
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    data["tables"]["v_lpc_details"]["owner"] = "someone"
    with pytest.raises(InputAssetError, match="owner"):
        parse_reference(yaml.safe_dump(data), source_name="x.yaml", max_chars=200_000)
    flags = "kind: reference_data\nname: n\ntables:\n  t:\n    columns:\n      f:\n"
    flags += "        categorical_values: [yes, no]\n"
    with pytest.raises(InputAssetError, match="quote it"):
        parse_reference(flags, source_name="x.yaml", max_chars=200_000)


# --- lookups 1-10 -----------------------------------------------------------------------------


def test_valid_selected_column_is_found_with_provenance(index: CatalogIndex) -> None:  # 1
    result = index.lookup_schema("v_lpc_details", "net_lpc_charge_usd_am")
    assert result.status is LookupStatus.FOUND and result.column is not None
    assert result.column.type == "NUMERIC" and result.column.description == (
        "Net late payment charge in USD."
    )
    assert result.provenance is not None
    assert result.provenance.path == "tables[v_lpc_details].columns[net_lpc_charge_usd_am]"
    assert result.provenance.catalog_version == index.catalog_version


def test_valid_extra_column_exists_whatever_the_cheat_sheet_says(index: CatalogIndex) -> None:
    result = index.lookup_schema("gcsinsight.v_lpc_details", "cg_id")  # 2 (+5)
    assert result.status is LookupStatus.FOUND and result.table == "v_lpc_details"


def test_invalid_column_and_unknown_table_are_not_found(index: CatalogIndex) -> None:
    column = index.lookup_schema("v_lpc_details", "no_such_col")  # 3
    assert column.status is LookupStatus.NOT_FOUND and column.table == "v_lpc_details"
    assert "not in table v_lpc_details" in (column.reason or "")
    table = index.lookup_schema("v_lpc_detail")  # 4: no similarity matching
    assert table.status is LookupStatus.NOT_FOUND and table.table is None


def test_schema_qualified_names_follow_the_namespace_policy(reference: Reference) -> None:  # 5
    permissive = CatalogIndex.for_reference(reference)
    strict = CatalogIndex.for_reference(reference, namespaces=["gcsinsight"])
    assert permissive is not None and strict is not None
    assert permissive.lookup_schema("other.v_lpc_details").status is LookupStatus.FOUND
    assert strict.lookup_schema("gcsinsight.v_lpc_details").status is LookupStatus.FOUND
    refused = strict.lookup_schema("other.v_lpc_details")
    assert refused.status is LookupStatus.NOT_FOUND and "namespace" in (refused.reason or "")


def test_identifiers_match_case_insensitively(index: CatalogIndex) -> None:  # 6
    result = index.lookup_schema("`V_LPC_DETAILS`", "CG_ID")
    assert result.status is LookupStatus.FOUND and result.column is not None
    assert result.column.column == "cg_id"


def test_same_column_name_in_several_tables_is_an_exact_reverse_lookup(
    index: CatalogIndex,
) -> None:  # 7
    result = index.lookup_column_alternatives("acct_id", excluded_table="v_lpc_details")
    tables = [a.table for a in result.alternatives]
    assert tables == ["v_acct_delinquency", "v_emissions", "v_cardmember"]
    assert "does not prove the same meaning" in result.note
    assert index.lookup_column_alternatives("acct").alternatives == []  # no fuzzy match


def test_documented_relationships_keep_their_verified_status(index: CatalogIndex) -> None:
    pair = index.lookup_relationships(pair=["v_lpc_details", "v_acct_delinquency"])  # 8
    (rel,) = pair.relationships
    assert rel.verified and rel.provenance.path == "relationships[1]"
    unverified = index.lookup_relationships(pair=["v_emissions", "v_lpc_details"])
    assert [r.verified for r in unverified.relationships] == [False]
    touching = index.lookup_relationships(tables=["v_lpc_details"])
    assert len(touching.relationships) == 2


def test_undocumented_relationship_is_never_inferred(index: CatalogIndex) -> None:  # 9
    result = index.lookup_relationships(pair=["v_cardmember", "v_lpc_details"])
    assert result.relationships == [] and "not established" in result.note


def test_rules_are_returned_unchanged_with_stable_ids(index: CatalogIndex) -> None:  # 10
    result = index.lookup_rules(tables=["v_lpc_details"], rule_ids=["R1", "r3", "R9"])
    by_id = {r.rule_id: r for r in result.rules}
    assert by_id["v_lpc_details#R1"].text == (
        "Late payment charge questions use the net amount (net_lpc_charge_usd_am) unless gross "
        "is explicitly requested."
    )
    assert by_id["R1"].scope == "catalog" and by_id["R1"].provenance.path == "global_rules[1]"
    assert by_id["R3"].text == "Cardmember names must never be shown in outputs."
    assert result.not_found == ["R9"]
    again = index.lookup_rules(rule_ids=["v_cardmember#R1"])
    assert again.rules[0].text.startswith("Do not output cardmember names")


def test_identical_requests_are_cached_per_catalog_version(index: CatalogIndex) -> None:
    request = LookupRequest(
        tool=LookupTool.LOOKUP_SCHEMA, table="v_lpc_details", column="cg_id", reason="a"
    )
    first, cached_first = index.run(request)
    second, cached_second = index.run(
        request.model_copy(update={"reason": "b", "table": "V_LPC_DETAILS"})
    )
    assert not cached_first and cached_second and first is second


def test_lookup_requests_are_validated_per_tool() -> None:
    with pytest.raises(ValueError, match="needs table"):
        LookupRequest(tool=LookupTool.LOOKUP_SCHEMA, reason="r")
    with pytest.raises(ValueError, match="exactly two"):
        LookupRequest(tool=LookupTool.LOOKUP_RELATIONSHIPS, pair=["a"], reason="r")
    with pytest.raises(ValueError, match="needs column"):
        LookupRequest(tool=LookupTool.LOOKUP_COLUMN_ALTERNATIVES, reason="r")


def test_document_references_have_no_catalog_index(policy_reference: Reference) -> None:
    assert CatalogIndex.for_reference(policy_reference) is None
    assert CatalogIndex.for_reference(None) is None
