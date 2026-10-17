"""Build the configured ProductAdapter.

Config::

    product_adapter:
      type: fixture | http_json | python
      options: {...}

or ``product_adapter: config/products/<product_id>.adapter.yaml`` (same shape in the file).
``type: python`` imports ``options.class: "pkg.module:Class"`` and calls ``Class(**options.init)``,
so a team can plug in any custom adapter without editing this package.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from pydantic import Field, ValidationError

from synthetic_user.adapters.base import ProductAdapter
from synthetic_user.adapters.fixture import FixtureAdapter
from synthetic_user.adapters.http_json import HttpJsonAdapter, Transport, urllib_transport
from synthetic_user.core.config import AppConfig, ProductAdapterType
from synthetic_user.core.exceptions import ConfigError, format_validation_error
from synthetic_user.models.base import NonEmptyStr, StrictModel

_T = TypeVar("_T", bound=StrictModel)


class FixtureOptions(StrictModel):
    path: Path


class PythonOptions(StrictModel):
    class_path: NonEmptyStr = Field(alias="class", pattern=r"^[\w.]+:\w+$")
    init: dict[str, Any] = Field(default_factory=dict)


def create_adapter(
    settings: AppConfig,
    *,
    env: Mapping[str, str] | None = None,
    transport: Transport = urllib_transport,
) -> ProductAdapter:
    """Unknown type, invalid options or a bad import -> ConfigError."""
    config = settings.product_adapter
    environ: Mapping[str, str] = os.environ if env is None else env
    if config.type is ProductAdapterType.FIXTURE:
        return FixtureAdapter.from_file(_options(FixtureOptions, config.options, config.type).path)
    if config.type is ProductAdapterType.HTTP_JSON:
        return HttpJsonAdapter.from_options(config.options, env=environ, transport=transport)
    if config.type is ProductAdapterType.PYTHON:
        return _python_adapter(_options(PythonOptions, config.options, config.type))
    raise ConfigError(f"Unknown product_adapter.type: {config.type!r}")


def _python_adapter(options: PythonOptions) -> ProductAdapter:
    module_name, _, class_name = options.class_path.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(
            f"product_adapter class {options.class_path!r}: cannot import {module_name!r}: {exc}"
        ) from exc
    cls = getattr(module, class_name, None)
    if not isinstance(cls, type):
        raise ConfigError(
            f"product_adapter class {options.class_path!r}: {class_name!r} is not a class "
            f"in {module_name!r}"
        )
    try:
        adapter = cls(**options.init)
    except TypeError as exc:
        raise ConfigError(
            f"product_adapter class {options.class_path!r} rejected options.init: {exc}"
        ) from exc
    if not isinstance(adapter, ProductAdapter):
        raise ConfigError(
            f"product_adapter class {options.class_path!r} does not implement ProductAdapter "
            "(needs: name, async send(req), async aclose())"
        )
    return adapter


def _options(model: type[_T], options: Mapping[str, Any], kind: str) -> _T:
    try:
        return model.model_validate(options)
    except ValidationError as exc:
        raise ConfigError(
            f"Invalid product_adapter options for {kind}: {format_validation_error(exc)}"
        ) from exc
