from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.models.reference import (
    CoreRef,
    Reference,
    ReferenceFormat,
    ReferencePack,
    ReferencePackDraft,
    ReferenceRule,
)
from synthetic_user.references.leaks import find_leaks, pack_identifiers
from synthetic_user.references.loader import (
    load_optional_reference,
    load_reference,
    parse_reference,
)
from synthetic_user.references.pack import NO_VALID_CORE_REF, build_pack, edit_pack

LOG = logging.getLogger("test")


# --- loader -----------------------------------------------------------------------------------


def test_json_reference_is_reindented_and_ided(sample_reference: Reference) -> None:
    assert sample_reference.format is ReferenceFormat.JSON
    assert sample_reference.source_name == "sample_card_spend_index.json"
    assert sample_reference.reference_id.startswith("sample_card_spend_index-")
    assert sample_reference.guide_name == "sample_card_spend_guide.md"
    assert json.loads(sample_reference.text)["tables"][0]["name"] == "card_transactions"


def test_text_and_markdown_references(policy_reference: Reference, tmp_path: Path) -> None:
    assert policy_reference.format is ReferenceFormat.MARKDOWN
    assert policy_reference.contains("The nightly hotel limit is USD 250 in Tier 1 cities")
    path = tmp_path / "voc_tables.txt"
    path.write_text("complaints: one row per customer complaint\n", encoding="utf-8")
    reference = load_reference(path, max_chars=1_000)
    assert reference.format is ReferenceFormat.TEXT
    assert reference.guide is None and "not exhaustive" in reference.guide_text()


@pytest.mark.parametrize(
    ("name", "content", "match"),
    [
        ("ref.pdf", b"%PDF", "unsupported file type"),
        ("ref.json", b"{not json", "not valid JSON"),
        ("ref.txt", b"   \n", "empty"),
        ("ref.txt", b"\xff\xfe\x00", "not UTF-8"),
        ("ref.txt", b"x" * 2_000, "limit is 1,000"),
    ],
)
def test_bad_references_are_refused(name: str, content: bytes, match: str) -> None:
    with pytest.raises(InputAssetError, match=match):
        parse_reference(content, source_name=name, max_chars=1_000)


def test_guide_must_be_text_and_changes_the_id() -> None:
    plain = parse_reference("a b c", source_name="r.txt", max_chars=1_000)
    guided = parse_reference(
        "a b c", source_name="r.txt", max_chars=1_000, guide="Read it.", guide_name="g.md"
    )
    assert plain.reference_id != guided.reference_id
    assert guided.guide_text() == "Read it."
    with pytest.raises(InputAssetError, match="use a .md or .txt"):
        parse_reference("a", source_name="r.txt", max_chars=1_000, guide="x", guide_name="g.pdf")


def test_optional_reference_sets_the_mode(tmp_path: Path) -> None:
    assert load_optional_reference(None, LOG, max_chars=1_000) is None
    with pytest.raises(InputAssetError, match="usage guide was given without a reference"):
        load_optional_reference(None, LOG, max_chars=1_000, guide_path=tmp_path / "g.md")
    with pytest.raises(InputAssetError, match="Reference not found"):
        load_optional_reference(tmp_path / "missing.md", LOG, max_chars=1_000)


# --- pack -------------------------------------------------------------------------------------


def _draft(*core: CoreRef, rules: tuple[ReferenceRule, ...] = ()) -> ReferencePackDraft:
    return ReferencePackDraft(core_refs=list(core), rules=list(rules))


GOOD = CoreRef(name="employees.full_name", quote="Employee full name.", why="ranking")
BAD = CoreRef(name="employees.salary", quote="Annual salary in USD.", why="invented")


def test_build_pack_keeps_quoted_items_and_records_the_rest(sample_reference: Reference) -> None:
    rule_bad = ReferenceRule(rule="convert to USD", quote="All amounts are converted to USD.")
    pack, rejected = build_pack(_draft(GOOD, BAD, rules=(rule_bad,)), sample_reference, "SCN_004")
    assert pack is not None and pack.pack_id == "RP_SCN_004"
    assert [r.name for r in pack.core_refs] == ["employees.full_name"]
    assert pack.rules == []
    assert [r.reason for r in pack.rejected_items] == [r.reason for r in rejected]
    assert len(rejected) == 2 and all("quote not found" in r.reason for r in rejected)


def test_build_pack_without_valid_core_ref_is_none(sample_reference: Reference) -> None:
    pack, rejected = build_pack(_draft(BAD), sample_reference, "SCN_001")
    assert pack is None and len(rejected) == 1
    assert "no core_ref" in NO_VALID_CORE_REF


def test_edit_pack_is_all_or_nothing(
    sample_packs: list[ReferencePack], sample_reference: Reference
) -> None:
    pack = sample_packs[0]
    edited = edit_pack(pack, _draft(GOOD), sample_reference)
    assert edited.edited and [r.name for r in edited.core_refs] == ["employees.full_name"]
    with pytest.raises(InputAssetError, match=r"core_ref 2 \(employees.salary\): quote not found"):
        edit_pack(pack, _draft(GOOD, BAD), sample_reference)
    other = sample_reference.model_copy(update={"reference_id": "other-00000000"})
    with pytest.raises(InputAssetError, match="was built from reference"):
        edit_pack(pack, _draft(GOOD), other)


# --- leaks ------------------------------------------------------------------------------------


def test_identifiers_are_identifier_like_core_ref_names(sample_packs: list[ReferencePack]) -> None:
    assert pack_identifiers(sample_packs[0]) == [
        "card_transactions",
        "card_transactions.amount",
        "card_transactions.currency_code",
        "currency_code",
        "employees.full_name",
        "full_name",
    ]
    policy_pack = ReferencePack(
        pack_id="RP_SCN_001",
        scenario_id="SCN_001",
        reference_id="policy-1",
        core_refs=[CoreRef(name="1. Hotels", quote="Hotels", why="limit")],
    )
    assert pack_identifiers(policy_pack) == []


def test_find_leaks_matches_whole_words_only() -> None:
    identifiers = ["full_name", "card_transactions"]
    assert find_leaks("Show FULL_NAME from card_transactions", identifiers) == [
        "card_transactions",
        "full_name",
    ]
    assert find_leaks("Show employee names and full names", identifiers) == []
