"""Load input assets (persona, product) from YAML into validated models.

The reference index is JSON and has its own loader (references/loader.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from synthetic_user.core.exceptions import (
    InputAssetError,
    SyntheticUserError,
    format_validation_error,
)
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription

ModelT = TypeVar("ModelT", bound=BaseModel)


def read_yaml_mapping(path: Path, error: type[SyntheticUserError], *, kind: str) -> dict[str, Any]:
    """Read a YAML file whose top level is a mapping. Empty file -> ``{}``.

    Raises ``error`` (e.g. ConfigError, InputAssetError) with the path and cause.
    """
    if not path.is_file():
        raise error(f"{kind} not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise error(f"Cannot read {kind.lower()} {path}: {exc}") from exc
    return parse_yaml_mapping(text, error, kind=kind, source=str(path))


def parse_yaml_mapping(
    text: str, error: type[SyntheticUserError], *, kind: str, source: str
) -> dict[str, Any]:
    """Parse YAML text whose top level is a mapping (``source`` names it in errors)."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise error(f"{kind} {source} is not valid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise error(f"{kind} {source} must contain a mapping at the top level")
    return data


def load_persona(path: Path) -> Persona:
    return _load_model(path, Persona, kind="Persona file")


def load_product(path: Path) -> ProductDescription:
    return _load_model(path, ProductDescription, kind="Product file")


def parse_asset(text: str, model: type[ModelT], *, kind: str, source: str) -> ModelT:
    """Validate YAML text (e.g. an uploaded file) as ``model``; InputAssetError with details."""
    data = parse_yaml_mapping(text, InputAssetError, kind=kind, source=source)
    return _validate(data, model, kind=kind, source=source)


def _load_model(path: Path, model: type[ModelT], *, kind: str) -> ModelT:
    data = read_yaml_mapping(path, InputAssetError, kind=kind)
    return _validate(data, model, kind=kind, source=str(path))


def _validate(data: dict[str, Any], model: type[ModelT], *, kind: str, source: str) -> ModelT:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise InputAssetError(
            f"{kind} {source} failed {model.__name__} validation: {format_validation_error(exc)}"
        ) from exc
