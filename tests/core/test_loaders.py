from __future__ import annotations

from pathlib import Path

import pytest

from synthetic_user.core.config import load_config
from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.core.loaders import load_catalog, load_persona, load_product
from synthetic_user.models.reference import Cardinality, VerificationStatus

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"


def test_persona_yaml_loads() -> None:
    persona = load_persona(DATA / "personas" / "finance_controller.yaml")
    assert persona.id == "finance_controller"
    assert persona.typical_tasks


def test_product_yaml_loads() -> None:
    product = load_product(DATA / "products" / "sample_product.yaml")
    assert product.product_id == "sample_nl2sql"
    assert not set(product.supported) & set(product.not_supported_yet)


def test_catalog_yaml_loads_with_intended_properties() -> None:
    catalog = load_catalog(DATA / "references" / "sample_catalog.yaml")
    assert {t.name for t in catalog.tables} == {
        "card_transactions",
        "employees",
        "departments",
        "merchants",
    }
    assert catalog.has_column("card_transactions", "currency_code")
    assert catalog.has_column("card_transactions", "merchant_category")
    assert catalog.has_column("employees", "department_id")

    statuses = sorted(r.verification for r in catalog.relationships)
    assert statuses == [VerificationStatus.UNKNOWN, VerificationStatus.VERIFIED]
    unknown = next(r for r in catalog.relationships if r.verification is VerificationStatus.UNKNOWN)
    assert unknown.cardinality is Cardinality.UNKNOWN

    merchants = catalog.get_table("merchants")
    assert merchants is not None
    assert merchants.description is None
    assert all(c.description is None for c in merchants.columns)


def test_repo_config_input_paths_resolve_to_loadable_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    cfg = load_config(REPO_ROOT / "config" / "app.yaml", env={})
    load_persona(cfg.inputs.persona_path)
    load_product(cfg.inputs.product_path)
    load_catalog(cfg.inputs.catalog_path)


def test_missing_asset(tmp_path: Path) -> None:
    with pytest.raises(InputAssetError, match="Persona file not found"):
        load_persona(tmp_path / "missing.yaml")


def test_invalid_yaml_asset(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("tables: [", encoding="utf-8")
    with pytest.raises(InputAssetError, match="not valid YAML"):
        load_catalog(path)


def test_non_mapping_asset(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(InputAssetError, match="mapping"):
        load_product(path)


def test_validation_error_details_name_the_field(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text(
        "catalog_id: c\ncatalog_version: '1'\n"
        "tables:\n  - name: t\n    columns:\n      - name: a\n        nullable: maybe\n",
        encoding="utf-8",
    )
    with pytest.raises(InputAssetError) as exc_info:
        load_catalog(path)
    message = str(exc_info.value)
    assert "ReferenceCatalog" in message
    assert "tables.0.columns.0.nullable" in message


def test_duplicate_table_names_rejected(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text(
        "catalog_id: c\ncatalog_version: '1'\ntables:\n"
        "  - {name: t, columns: [{name: a}]}\n"
        "  - {name: t, columns: [{name: b}]}\n",
        encoding="utf-8",
    )
    with pytest.raises(InputAssetError, match="duplicate table names"):
        load_catalog(path)
