from __future__ import annotations

from typing import Any

import pytest

from synthetic_user.models.reference import (
    IndexRelationship,
    ReferenceCatalog,
    ReferenceIndex,
    VerificationStatus,
)
from synthetic_user.references.integrity import (
    check_index_against_catalog,
    check_tables_against_catalog,
    enforce_relationship_verification,
)


def _table(index: ReferenceIndex, name: str) -> Any:
    return next(t for t in index.tables if t.name == name)


def _with_table(index: ReferenceIndex, name: str, **update: Any) -> ReferenceIndex:
    tables = [t.model_copy(update=update) if t.name == name else t for t in index.tables]
    return index.model_copy(update={"tables": tables})


def test_sample_index_is_clean(
    sample_index: ReferenceIndex, sample_catalog: ReferenceCatalog
) -> None:
    assert check_index_against_catalog(sample_index, sample_catalog) == []


def test_unknown_table(sample_index: ReferenceIndex, sample_catalog: ReferenceCatalog) -> None:
    extra = _table(sample_index, "departments").model_copy(update={"name": "budgets"})
    index = sample_index.model_copy(update={"tables": [*sample_index.tables, extra]})
    assert any(
        "unknown table 'budgets'" in e for e in check_index_against_catalog(index, sample_catalog)
    )


def test_missing_table(sample_index: ReferenceIndex, sample_catalog: ReferenceCatalog) -> None:
    index = sample_index.model_copy(
        update={"tables": [t for t in sample_index.tables if t.name != "merchants"]}
    )
    assert any(
        "missing table 'merchants'" in e for e in check_index_against_catalog(index, sample_catalog)
    )


def test_unexpected_table_for_batch(
    sample_index: ReferenceIndex, sample_catalog: ReferenceCatalog
) -> None:
    errors = check_tables_against_catalog(
        sample_index.tables, sample_catalog, expected_tables={"employees"}
    )
    assert any("'card_transactions' was not requested" in e for e in errors)


def test_unknown_column(sample_index: ReferenceIndex, sample_catalog: ReferenceCatalog) -> None:
    index = _with_table(sample_index, "card_transactions", important_columns=["amount_usd"])
    errors = check_index_against_catalog(index, sample_catalog)
    assert errors == ["card_transactions: important_columns 'amount_usd' is not a catalog column"]


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("department_name", "is not on this table (found on departments)"),
        ("departments.department_name", "is a column of another table (departments)"),
    ],
)
def test_column_on_wrong_table(
    sample_index: ReferenceIndex, sample_catalog: ReferenceCatalog, ref: str, expected: str
) -> None:
    index = _with_table(sample_index, "card_transactions", dimensions=[ref])
    errors = check_index_against_catalog(index, sample_catalog)
    assert len(errors) == 1 and expected in errors[0]


def test_qualified_same_table_column_ok(
    sample_index: ReferenceIndex, sample_catalog: ReferenceCatalog
) -> None:
    index = _with_table(
        sample_index, "card_transactions", dimensions=["card_transactions.currency_code"]
    )
    assert check_index_against_catalog(index, sample_catalog) == []


def test_bad_metric_column(sample_index: ReferenceIndex, sample_catalog: ReferenceCatalog) -> None:
    table = _table(sample_index, "card_transactions")
    metric = table.metrics[0].model_copy(update={"columns": ["amount", "fx_rate"]})
    index = _with_table(sample_index, "card_transactions", metrics=[metric])
    errors = check_index_against_catalog(index, sample_catalog)
    assert errors == ["card_transactions: metric 'total_spend' 'fx_rate' is not a catalog column"]


def test_time_column_must_be_time_typed(
    sample_index: ReferenceIndex, sample_catalog: ReferenceCatalog
) -> None:
    index = _with_table(sample_index, "card_transactions", time_columns=["amount"])
    errors = check_index_against_catalog(index, sample_catalog)
    assert errors == ["card_transactions: time_columns 'amount' has non-time type NUMERIC"]


def test_time_column_unknown(
    sample_index: ReferenceIndex, sample_catalog: ReferenceCatalog
) -> None:
    index = _with_table(sample_index, "card_transactions", time_columns=["posted_at"])
    assert "is not a catalog column" in check_index_against_catalog(index, sample_catalog)[0]


def test_relationship_to_unknown_target(
    sample_index: ReferenceIndex, sample_catalog: ReferenceCatalog
) -> None:
    rel = IndexRelationship(source_column="merchant_id", target_table="vendors", target_column="id")
    index = _with_table(sample_index, "card_transactions", relationships=[rel])
    errors = check_index_against_catalog(index, sample_catalog)
    assert errors == ["card_transactions: relationship target_table 'vendors' is not in catalog"]


def test_relationship_to_unknown_target_column(
    sample_index: ReferenceIndex, sample_catalog: ReferenceCatalog
) -> None:
    rel = IndexRelationship(
        source_column="merchant_id", target_table="merchants", target_column="id"
    )
    index = _with_table(sample_index, "card_transactions", relationships=[rel])
    assert (
        "merchants.id is not a catalog column"
        in check_index_against_catalog(index, sample_catalog)[0]
    )


def test_catalog_mismatch(sample_index: ReferenceIndex, sample_catalog: ReferenceCatalog) -> None:
    index = sample_index.model_copy(update={"source_catalog_version": "2.0"})
    assert "not sample_card_spend v1.0" in check_index_against_catalog(index, sample_catalog)[0]


# --- relationship verification -----------------------------------------------------------


def _rels(index: ReferenceIndex, table: str) -> list[IndexRelationship]:
    return list(_table(index, table).relationships)


def test_catalog_verified_relationship_kept(
    sample_index: ReferenceIndex, sample_catalog: ReferenceCatalog
) -> None:
    result = enforce_relationship_verification(sample_index, sample_catalog)
    assert _rels(result, "card_transactions")[0].verification is VerificationStatus.VERIFIED
    assert result.uncertainties == sample_index.uncertainties


def test_verified_in_reverse_direction_kept(
    sample_index: ReferenceIndex, sample_catalog: ReferenceCatalog
) -> None:
    rel = IndexRelationship(
        source_column="employee_id",
        target_table="card_transactions",
        target_column="employee_id",
        verification=VerificationStatus.VERIFIED,
    )
    index = _with_table(sample_index, "employees", relationships=[rel])
    result = enforce_relationship_verification(index, sample_catalog)
    assert _rels(result, "employees")[0].verification is VerificationStatus.VERIFIED


def test_verified_upgrade_of_unknown_catalog_relationship_downgraded(
    sample_index: ReferenceIndex, sample_catalog: ReferenceCatalog
) -> None:
    rel = _rels(sample_index, "employees")[0].model_copy(
        update={"verification": VerificationStatus.VERIFIED}
    )
    index = _with_table(sample_index, "employees", relationships=[rel])
    result = enforce_relationship_verification(index, sample_catalog)
    assert _rels(result, "employees")[0].verification is VerificationStatus.UNKNOWN
    assert result.uncertainties[-1] == (
        "relationship employees.department_id -> departments.department_id was marked verified "
        "but is not verified in the catalog; downgraded to unknown"
    )


def test_verified_claim_not_in_catalog_downgraded_to_inferred(
    sample_index: ReferenceIndex, sample_catalog: ReferenceCatalog
) -> None:
    rel = IndexRelationship(
        source_column="merchant_id",
        target_table="merchants",
        target_column="merchant_id",
        verification=VerificationStatus.VERIFIED,
    )
    index = _with_table(sample_index, "card_transactions", relationships=[rel])
    result = enforce_relationship_verification(index, sample_catalog)
    assert _rels(result, "card_transactions")[0].verification is VerificationStatus.INFERRED
    assert "downgraded to inferred" in result.uncertainties[-1]


def test_enforcement_is_pure(
    sample_index: ReferenceIndex, sample_catalog: ReferenceCatalog
) -> None:
    rel = _rels(sample_index, "employees")[0].model_copy(
        update={"verification": VerificationStatus.VERIFIED}
    )
    index = _with_table(sample_index, "employees", relationships=[rel])
    before = index.model_dump()
    enforce_relationship_verification(index, sample_catalog)
    assert index.model_dump() == before
