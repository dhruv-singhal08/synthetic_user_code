"""The built-in metric registry, loaded from ``catalog.yaml`` (exactly 11 metrics).

Built-in definitions are data: their rubric and generation text lives in the YAML file, and the
judges / question generator read them through ``MetricDefinition``. The catalog version is
recorded in every frozen run configuration.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

from synthetic_user.core.exceptions import PromptTemplateError
from synthetic_user.models.metrics import MetricCategory, MetricDefinition, MetricOrigin

CATALOG_PATH = Path(__file__).resolve().parent / "catalog.yaml"
BUILTIN_VERSION = "1.0"
BUILTIN_METRIC_COUNT = 11


@cache
def _load() -> dict[str, Any]:
    try:
        data = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PromptTemplateError(f"cannot read metric catalog {CATALOG_PATH}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("metrics"), dict):
        raise PromptTemplateError(f"metric catalog {CATALOG_PATH} must have a 'metrics' mapping")
    return data


@cache
def builtin_definitions() -> tuple[MetricDefinition, ...]:
    """Every built-in metric, in catalog order (validated once per process)."""
    definitions = tuple(
        MetricDefinition.model_validate(
            {
                "metric_id": metric_id,
                "version": BUILTIN_VERSION,
                "origin": MetricOrigin.BUILTIN.value,
                **body,
            }
        )
        for metric_id, body in _load()["metrics"].items()
    )
    if len(definitions) != BUILTIN_METRIC_COUNT:
        raise PromptTemplateError(
            f"metric catalog must define exactly {BUILTIN_METRIC_COUNT} metrics, "
            f"got {len(definitions)}"
        )
    return definitions


def builtin_ids() -> tuple[str, ...]:
    return tuple(d.metric_id for d in builtin_definitions())


def builtin(metric_id: str) -> MetricDefinition | None:
    return next((d for d in builtin_definitions() if d.metric_id == metric_id), None)


def catalog_version() -> str:
    return str(_load()["catalog_version"])


def category_labels() -> dict[MetricCategory, str]:
    return {MetricCategory(key): str(label) for key, label in _load()["category_labels"].items()}
