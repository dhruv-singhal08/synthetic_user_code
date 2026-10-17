from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.models.reference import ReferenceIndex, VerificationStatus
from synthetic_user.models.run import GenerationMode
from synthetic_user.references.integrity import validate_index
from synthetic_user.references.loader import (
    load_optional_reference_index,
    load_reference_index,
    parse_reference_index,
    resolve_mode,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCES = REPO_ROOT / "data" / "references"
SAMPLE = REFERENCES / "sample_card_spend_index.json"
AMEX_SAMPLE = REFERENCES / "reference_index_sample_v2.json"
LOG = logging.getLogger("test")


def _raw(path: Path = SAMPLE) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _write(tmp_path: Path, data: Any, name: str = "index.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --- loading --------------------------------------------------------------------------------------


def test_sample_index_loads_with_lookups() -> None:
    index = load_reference_index(SAMPLE)
    assert index.table_names() == ["card_transactions", "employees", "departments", "merchants"]
    assert {c.name for c in index.columns("employees")} == {
        "employee_id",
        "full_name",
        "department_id",
    }
    column = index.get_column("card_transactions", "currency_code")
    assert column is not None and column.data_type == "STRING" and column.description
    assert index.has_column("card_transactions", "merchant_category")
    assert not index.has_column("card_transactions", "colour") and index.columns("nope") == []
    assert (
        index.verification("employees.employee_id", "card_transactions.employee_id")
        is VerificationStatus.VERIFIED
    )
    assert (
        index.verification("employees.department_id", "departments.department_id")
        is VerificationStatus.UNKNOWN
    )
    assert index.verification("card_transactions.merchant_id", "merchants.merchant_id") is None
    assert [r.target_table for r in index.relationships_of("employees")] == [
        "employees",
        "departments",
    ]
    assert index.rules("employees") == {"sensitive_columns": ["full_name"]}
    assert index.global_semantic_rules["currency"]
    assert validate_index(index) == []


def test_amex_sample_structure_loads_and_keeps_extras() -> None:
    """The structure transcribed from the user's real file loads; nothing is dropped."""
    raw = _raw(AMEX_SAMPLE)
    index = load_reference_index(AMEX_SAMPLE)
    assert len(index.tables) == raw["scope"]["table_count"]
    assert index.extras["scope"] == raw["scope"]
    assert index.extras["integrity_report"] == raw["integrity_report"]
    assert index.extras["generation_mode"] == raw["generation_mode"]
    table, raw_table = index.tables[0], raw["tables"][0]
    assert table.extras == {
        k: raw_table[k] for k in ("business_domain", "grain_status", "column_count")
    }
    assert table.business_rules == raw_table["business_rules"]
    assert table.columns[0].extras["semantic_roles"] == raw_table["columns"][0]["semantic_roles"]
    assert index.relationships[0].extras["source"] == raw["relationships"][0]["source"]
    assert validate_index(index) == []


def test_versioned_id_changes_with_content(tmp_path: Path) -> None:
    first = load_reference_index(SAMPLE)
    data = _raw()
    data["tables"][0]["business_purpose"] = "edited"
    edited = load_reference_index(_write(tmp_path, data))
    assert first.index_id == edited.index_id
    assert first.versioned_id != edited.versioned_id
    assert first.versioned_id.startswith(f"{first.index_id}-")


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda d: d.update(colour="blue"), "colour"),
        (lambda d: d["tables"][0].update(owner="x"), "tables.0.owner"),
        (lambda d: d["tables"][1]["columns"][0].update(unit="usd"), "tables.1.columns.0.unit"),
        (lambda d: d["relationships"][0].update(weight=1), "relationships.0.weight"),
        (lambda d: d.pop("index_id"), "index_id"),
        (lambda d: d["tables"][0].pop("business_purpose"), "tables.0.business_purpose"),
        (lambda d: d["tables"][0]["columns"][0].pop("name"), "tables.0.columns.0.name"),
        (lambda d: d["tables"][0].update(columns=[]), "tables.0.columns"),
        (
            lambda d: d["relationships"][0].update(verification="sure"),
            "relationships.0.verification",
        ),
    ],
)
def test_unknown_or_missing_fields_name_path_and_field(
    tmp_path: Path, mutate: Any, field: str
) -> None:
    data = copy.deepcopy(_raw())
    mutate(data)
    path = _write(tmp_path, data)
    with pytest.raises(InputAssetError) as info:
        load_reference_index(path)
    message = str(info.value)
    assert str(path) in message and field in message


@pytest.mark.parametrize("where", ["top", "column"])
def test_reserved_keys_rejected(tmp_path: Path, where: str) -> None:
    data = _raw()
    target = data if where == "top" else data["tables"][0]["columns"][0]
    target["extras"] = {"sneaky": 1}
    with pytest.raises(InputAssetError, match="reserved field"):
        load_reference_index(_write(tmp_path, data))


def test_bad_files(tmp_path: Path) -> None:
    with pytest.raises(InputAssetError, match="not found"):
        load_reference_index(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(InputAssetError, match="not valid JSON"):
        load_reference_index(bad)
    with pytest.raises(InputAssetError, match="JSON object"):
        parse_reference_index("[1, 2]", source="upload")


# --- mode -----------------------------------------------------------------------------------------


def test_missing_index_means_ungrounded(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    assert resolve_mode(None, LOG) is GenerationMode.UNGROUNDED
    assert "mode ungrounded" in caplog.text
    assert load_optional_reference_index(None, LOG) is None


def test_given_index_means_grounded() -> None:
    assert resolve_mode(SAMPLE, LOG) is GenerationMode.GROUNDED
    index = load_optional_reference_index(SAMPLE, LOG)
    assert isinstance(index, ReferenceIndex)


# --- integrity warnings (listed, never fixed) -----------------------------------------------------


def test_warnings_for_dangling_relationship_and_duplicates(tmp_path: Path) -> None:
    data = _raw()
    data["relationships"].append(
        {
            "source_table": "card_transactions",
            "source_column": "merchant_id",
            "target_table": "vendors",
            "target_column": "vendor_id",
        }
    )
    data["relationships"].append(
        {
            "source_table": "card_transactions",
            "source_column": "merchant_ref",
            "target_table": "merchants",
            "target_column": "merchant_id",
        }
    )
    data["tables"][3]["columns"].append({"name": "merchant_id", "data_type": "STRING"})
    data["tables"].append(copy.deepcopy(data["tables"][2]))
    index = load_reference_index(_write(tmp_path, data))
    warnings = validate_index(index)
    assert any("unknown table 'vendors'" in w for w in warnings)
    assert any("unknown column card_transactions.merchant_ref" in w for w in warnings)
    assert any("duplicate table names ['departments']" in w for w in warnings)
    assert any("merchants: duplicate column names ['merchant_id']" in w for w in warnings)
    assert any("scope.table_count=4 but 5 tables" in w for w in warnings)
    # Not fixed: the index still holds exactly what the file said.
    assert len(index.relationships) == 4 and len(index.tables) == 5


def test_warnings_for_column_lists_and_table_level_relationships(tmp_path: Path) -> None:
    data = _raw()
    data["tables"][0]["important_columns"].append("ghost")
    data["tables"][0]["relationships"].append(
        {
            "source_table": "card_transactions",
            "source_column": "merchant_id",
            "target_table": "merchants",
            "target_column": "merchant_id",
        }
    )
    data["integrity_report"]["status"] = "failed"
    warnings = validate_index(load_reference_index(_write(tmp_path, data)))
    assert "card_transactions.important_columns: unknown column 'ghost'" in warnings
    assert any("missing from the top-level relationships list" in w for w in warnings)
    assert any("integrity_report.status is 'failed'" in w for w in warnings)
