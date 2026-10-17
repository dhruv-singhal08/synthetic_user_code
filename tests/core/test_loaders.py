from __future__ import annotations

from pathlib import Path

import pytest

from synthetic_user.core.config import load_config
from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.core.loaders import load_persona, load_product
from synthetic_user.references.loader import load_reference_index

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


def test_repo_config_input_paths_resolve_to_loadable_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    cfg = load_config(REPO_ROOT / "config" / "app.yaml", env={})
    load_persona(cfg.inputs.persona_path)
    load_product(cfg.inputs.product_path)
    assert cfg.inputs.reference_index_file is not None
    load_reference_index(cfg.inputs.reference_index_file)


def test_missing_asset(tmp_path: Path) -> None:
    with pytest.raises(InputAssetError, match="Persona file not found"):
        load_persona(tmp_path / "missing.yaml")


def test_invalid_yaml_asset(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("tables: [", encoding="utf-8")
    with pytest.raises(InputAssetError, match="not valid YAML"):
        load_product(path)


def test_non_mapping_asset(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(InputAssetError, match="mapping"):
        load_product(path)


def test_validation_error_details_name_the_field(tmp_path: Path) -> None:
    path = tmp_path / "product.yaml"
    path.write_text(
        "product_id: p\nproduct_version: '1'\nname: n\ndescription: d\nproduct_type: chatbot\n"
        "supported: [a]\ncolour: blue\n",
        encoding="utf-8",
    )
    with pytest.raises(InputAssetError) as exc_info:
        load_product(path)
    message = str(exc_info.value)
    assert "ProductDescription" in message
    assert "colour" in message
