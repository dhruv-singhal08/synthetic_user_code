"""Reference formats (D203): data_catalog / document_collection, instructions, cheat sheet."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import yaml

from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.models.reference import (
    DEFAULT_INSTRUCTIONS,
    CatalogPackDraft,
    DocumentPackDraft,
    DocumentPick,
    Join,
    Reference,
    ReferenceKind,
    ReferencePack,
    TablePick,
)
from synthetic_user.references.convert import refidx_to_catalog
from synthetic_user.references.leaks import find_leaks, pack_identifiers
from synthetic_user.references.loader import (
    load_optional_reference,
    load_reference,
    parse_reference,
)
from synthetic_user.references.pack import NO_VALID_ITEM, build_pack, edit_pack

LOG = logging.getLogger("test")
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "references"
CATALOG = {
    "kind": "data_catalog",
    "name": "Spend",
    "tables": [
        {
            "name": "tx",
            "description": "Transactions.",
            "rules": ["Amounts are in USD."],
            "columns": [{"name": "amount"}, {"name": "emp_id"}],
        },
        {"name": "emp", "columns": [{"name": "emp_id"}, {"name": "name"}]},
    ],
    "relationships": [{"from": "tx.emp_id", "to": "emp.emp_id"}],
    "rules": ["Never sum across currencies."],
}


def _yaml(data: object) -> str:
    return yaml.safe_dump(data, sort_keys=False)


# --- loader -----------------------------------------------------------------------------------


def test_catalog_reference_loads_with_instructions(sample_reference: Reference) -> None:
    assert sample_reference.kind is ReferenceKind.DATA_CATALOG
    assert sample_reference.source_name == "sample_card_spend_catalog.yaml"
    assert sample_reference.reference_id.startswith("sample_card_spend_catalog-")
    assert sample_reference.instructions_name == "sample_card_spend_instructions.md"
    text = sample_reference.render()
    assert "- card_transactions: Corporate card spend" in text
    assert "card_transactions.employee_id -> employees.employee_id (verified)" in text
    assert "employees.department_id -> departments.department_id (NOT verified)" in text
    assert "- R1: Amounts in different currency_code values" in text
    assert "values: USD, EUR, GBP, INR" in text


def test_document_reference_loads(policy_reference: Reference) -> None:
    assert policy_reference.kind is ReferenceKind.DOCUMENT_COLLECTION
    text = policy_reference.render()
    assert "### Hotel policy" in text and "USD 250 in Tier 1" in text


def test_json_and_yaml_give_the_same_reference() -> None:
    from_yaml = parse_reference(_yaml(CATALOG), source_name="spend.yaml", max_chars=10_000)
    from_json = parse_reference(json.dumps(CATALOG), source_name="spend.json", max_chars=10_000)
    assert from_yaml.reference_id == from_json.reference_id
    assert from_yaml.content == from_json.content


@pytest.mark.parametrize(
    ("name", "content", "match"),
    [
        ("ref.md", "# notes", "use a YAML or JSON file"),
        ("ref.yaml", "- a\n- b", "expected a mapping with 'kind'"),
        ("ref.yaml", "kind: spreadsheet\nname: x", "'kind' must be one of"),
        ("ref.json", "{not json", "not valid json"),
        ("ref.yaml", "kind: data_catalog\nname: x\ntables: []", "tables"),
        ("ref.yaml", "kind: document_collection\nname: x\ndocuments: [{name: a}]", "summary"),
        ("ref.yaml", "kind: data_catalog\nname: x\ntables: [{name: t}]\nextra: 1", "extra"),
    ],
)
def test_bad_references_are_refused(name: str, content: str, match: str) -> None:
    with pytest.raises(InputAssetError, match=match):
        parse_reference(content, source_name=name, max_chars=10_000)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda d: d["tables"].append({"name": "TX"}), "duplicate table names"),
        (lambda d: d["tables"][0].update(name="t.x"), "must not contain dots"),
        (lambda d: d["tables"][1]["columns"].append({"name": "name"}), "duplicate columns"),
        (lambda d: d["relationships"].append({"from": "tx.amount", "to": "x.y"}), "unknown table"),
        (lambda d: d["relationships"].append({"from": "tx.zz", "to": "emp.name"}), "not listed"),
        (lambda d: d["relationships"].append({"from": "tx", "to": "emp.name"}), "table.column"),
    ],
)
def test_catalog_consistency_is_checked(change: object, match: str) -> None:
    data = json.loads(json.dumps(CATALOG))
    change(data)  # type: ignore[operator]
    with pytest.raises(InputAssetError, match=match):
        parse_reference(_yaml(data), source_name="c.yaml", max_chars=10_000)


def test_duplicate_documents_and_size_limit_are_refused() -> None:
    docs = {
        "kind": "document_collection",
        "name": "d",
        "documents": [{"name": "A doc", "summary": "s"}, {"name": "a  DOC", "summary": "t"}],
    }
    with pytest.raises(InputAssetError, match="duplicate document names"):
        parse_reference(_yaml(docs), source_name="d.yaml", max_chars=10_000)
    big = {
        "kind": "document_collection",
        "name": "d",
        "documents": [{"name": "A", "summary": "x" * 2_000}],
    }
    with pytest.raises(InputAssetError, match="reference.max_chars"):
        parse_reference(_yaml(big), source_name="d.yaml", max_chars=1_000)


def test_instructions_are_optional_text_and_change_the_id() -> None:
    plain = parse_reference(_yaml(CATALOG), source_name="c.yaml", max_chars=10_000)
    assert plain.instructions is None
    assert plain.instructions_text() == DEFAULT_INSTRUCTIONS[ReferenceKind.DATA_CATALOG]
    guided = parse_reference(
        _yaml(CATALOG),
        source_name="c.yaml",
        max_chars=10_000,
        guide="Only R1 matters.",
        guide_name="i.md",
    )
    assert guided.instructions_text() == "Only R1 matters." and guided.instructions_name == "i.md"
    assert guided.reference_id != plain.reference_id
    with pytest.raises(InputAssetError, match="use a .md or .txt file"):
        parse_reference(
            _yaml(CATALOG), source_name="c.yaml", max_chars=10_000, guide="x", guide_name="g.pdf"
        )


def test_optional_reference_sets_the_mode(tmp_path: Path) -> None:
    assert load_optional_reference(None, LOG, max_chars=10_000) is None
    with pytest.raises(InputAssetError, match="without a reference file"):
        load_optional_reference(None, LOG, max_chars=10_000, guide_path=tmp_path / "g.md")
    path = tmp_path / "c.yaml"
    path.write_text(_yaml(CATALOG), encoding="utf-8")
    loaded = load_optional_reference(path, LOG, max_chars=10_000)
    assert loaded is not None and loaded == load_reference(path, max_chars=10_000)


# --- cheat sheet --------------------------------------------------------------------------------


def _catalog() -> Reference:
    return parse_reference(_yaml(CATALOG), source_name="c.yaml", max_chars=10_000)


def test_catalog_pack_copies_the_catalog_text_and_records_unknown_picks() -> None:
    draft = CatalogPackDraft(
        tables=[
            TablePick(table="TX", columns=["amount", "nope"], why="spend"),
            TablePick(table="ghost", why="x"),
            TablePick(table="tx", columns=["emp_id"], why="again"),  # merged into tx
        ],
        joins=[
            Join(**{"from": "emp.emp_id", "to": "tx.emp_id"}),
            Join(**{"from": "tx.amount", "to": "emp.name"}),
        ],
        rules=["r1", "R9"],
    )
    pack, rejected = build_pack(draft, _catalog(), "SCN_001")
    assert pack is not None and pack.pack_id == "RP_SCN_001" and pack.kind == "data_catalog"
    (table,) = pack.tables
    assert table.name == "tx" and table.description == "Transactions."
    assert [c.name for c in table.columns] == ["amount", "emp_id"]
    assert table.rules == ["Amounts are in USD."]  # table rules travel with the table
    assert [(j.from_, j.to, j.verified) for j in pack.joins] == [("tx.emp_id", "emp.emp_id", False)]
    assert [(r.id, r.rule) for r in pack.rules] == [("R1", "Never sum across currencies.")]
    assert [r.reason for r in rejected] == [
        "column 'nope' is not in table tx",
        "unknown table 'ghost'",
        "tx.amount -> emp.name is not a listed relationship",
        "unknown catalog rule id 'R9'",
    ]
    assert pack.rejected_items == rejected
    text = pack.render()
    assert "tx.emp_id -> emp.emp_id (NOT verified)" in text and "rule: Amounts are in USD." in text
    assert pack.draft().model_dump(by_alias=True)["joins"] == [
        {"from": "tx.emp_id", "to": "emp.emp_id"}
    ]


def test_document_pack_and_kind_mismatch(policy_reference: Reference) -> None:
    draft = DocumentPackDraft(
        documents=[
            DocumentPick(name="hotel POLICY", why="limits"),
            DocumentPick(name="Visa rules", why="x"),
        ]
    )
    pack, rejected = build_pack(draft, policy_reference, "SCN_002")
    assert pack is not None and [d.name for d in pack.documents] == ["Hotel policy"]
    assert "USD 250" in pack.documents[0].summary and "USD 250" in pack.render()
    assert [r.reason for r in rejected] == ["unknown document 'Visa rules'"]
    wrong, why = build_pack(
        CatalogPackDraft(tables=[TablePick(table="tx", why="x")]), policy_reference, "SCN_002"
    )
    assert wrong is None and "does not fit a document_collection" in why[0].reason


def test_pack_without_a_valid_table_is_none() -> None:
    pack, rejected = build_pack(
        CatalogPackDraft(tables=[TablePick(table="ghost", why="x")]), _catalog(), "SCN_001"
    )
    assert pack is None and rejected and NO_VALID_ITEM


def test_edit_pack_is_all_or_nothing(
    sample_packs: list[ReferencePack], sample_reference: Reference
) -> None:
    good = CatalogPackDraft(tables=[TablePick(table="employees", columns=["full_name"], why="w")])
    edited = edit_pack(sample_packs[0], good, sample_reference)
    assert edited.edited and edited.item_names() == ["employees", "employees.full_name"]
    bad = CatalogPackDraft(
        tables=[TablePick(table="employees", why="w"), TablePick(table="ghost", why="x")]
    )
    with pytest.raises(InputAssetError, match="not saved: unknown table 'ghost'"):
        edit_pack(sample_packs[0], bad, sample_reference)
    other = sample_reference.model_copy(update={"reference_id": "other-123"})
    with pytest.raises(InputAssetError, match="was built from reference"):
        edit_pack(sample_packs[0], good, other)


def test_identifiers_are_identifier_like_pack_names(sample_packs: list[ReferencePack]) -> None:
    assert pack_identifiers(sample_packs[0]) == [
        "card_transactions",
        "card_transactions.amount",
        "card_transactions.currency_code",
        "currency_code",
        "employees.full_name",
        "full_name",
    ]


def test_find_leaks_matches_whole_words_only() -> None:
    identifiers = ["card_transactions", "full_name"]
    assert find_leaks("Top spenders in card_transactions?", identifiers) == ["card_transactions"]
    assert find_leaks("show my full name", identifiers) == []


# --- converter ----------------------------------------------------------------------------------


def test_refidx_index_converts_to_a_valid_catalog() -> None:
    index = json.loads((FIXTURES / "refidx_sample.json").read_text(encoding="utf-8"))
    catalog = refidx_to_catalog(index)
    reference = parse_reference(_yaml(catalog), source_name="c.yaml", max_chars=200_000)
    assert reference.kind is ReferenceKind.DATA_CATALOG
    tables = {t["name"]: t for t in catalog["tables"]}
    assert tables["card_transactions"]["description"].startswith("Corporate card spend")
    assert "merchants" in tables and "description" not in tables["merchants"]  # "unknown" dropped
    currency = next(
        c for c in tables["card_transactions"]["columns"] if c["name"] == "currency_code"
    )
    assert currency["values"] == ["USD", "EUR", "GBP", "INR"]
    assert [r["verified"] for r in catalog["relationships"]] == [True, False]
    assert catalog["rules"] == [
        "Amounts in different currency_code values must not be summed directly."
    ]
    with pytest.raises(InputAssetError, match="no 'tables' list"):
        refidx_to_catalog({"index_id": "x"})
