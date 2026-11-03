"""Resolve a run's metric selection into the frozen ``ResolvedMetricConfig`` (once per run).

Built-in ids are validated against the catalog; legacy ids (``completion``, ``tool_selection``,
``answer_correctness``, ...) are refused with a migration message, never silently mapped.
The compiler is called only when custom metrics exist. The result is immutable: the pipeline
stores it as ``metric_config.json`` before scenario generation and every later step reads it.
"""

from __future__ import annotations

import logging

from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.llm.base import LLMClient
from synthetic_user.metrics.catalog import builtin_definitions, builtin_ids, catalog_version
from synthetic_user.metrics.compiler import COMPILER_VERSION, MetricCompiler
from synthetic_user.models.metrics import (
    MetricDefinition,
    MetricSelection,
    ResolvedMetricConfig,
    config_id_for,
)
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import Reference

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

# Metric ids of earlier versions: refused with the replacement, never silently relabelled.
LEGACY_METRIC_IDS = {
    "completion": "correctness_completeness (merged definition; old scores are not relabelled)",
    "answer_correctness": "correctness_completeness",
    "tool_selection": "tool_correctness (expanded definition; old scores are not relabelled)",
}


def validate_selected_ids(selected: list[str]) -> None:
    known = set(builtin_ids())
    problems: list[str] = []
    for metric_id in selected:
        if metric_id in known:
            continue
        if metric_id in LEGACY_METRIC_IDS:
            problems.append(f"{metric_id!r} was replaced by {LEGACY_METRIC_IDS[metric_id]}")
        else:
            problems.append(f"unknown built-in metric {metric_id!r}")
    if problems:
        raise InputAssetError(
            "Invalid metric selection: " + "; ".join(problems) + f". Built-in metrics: "
            f"{', '.join(builtin_ids())}"
        )


def freeze(
    selection: MetricSelection,
    definitions: list[MetricDefinition],
    *,
    compiler_called: bool,
    notes: list[str],
) -> ResolvedMetricConfig:
    payload = {
        "selection": selection.model_dump(mode="json"),
        "definitions": [d.model_dump(mode="json") for d in definitions],
        "catalog": catalog_version(),
        "compiler": COMPILER_VERSION,
    }
    return ResolvedMetricConfig(
        config_id=config_id_for(payload),
        compiler_version=COMPILER_VERSION,
        catalog_version=catalog_version(),
        selection_fingerprint=selection.fingerprint(),
        selected_metric_ids=[d.metric_id for d in definitions],
        definitions=definitions,
        custom_inputs=list(selection.custom_inputs),
        input_errors=list(selection.input_errors),
        compiler_called=compiler_called,
        compiler_notes=notes,
    )


def builtin_metric_config(metric_ids: list[str]) -> ResolvedMetricConfig:
    """A frozen configuration of built-in metrics only (never calls the compiler)."""
    validate_selected_ids(metric_ids)
    selection = MetricSelection(selected_metrics=metric_ids)
    chosen = set(metric_ids)
    definitions = [d for d in builtin_definitions() if d.metric_id in chosen]
    return freeze(selection, definitions, compiler_called=False, notes=[])


async def resolve_metric_config(
    selection: MetricSelection,
    product: ProductDescription,
    reference: Reference | None,
    *,
    llm: LLMClient,
    logger: Logger,
    max_retries: int,
    sql_dialect: str | None,
) -> ResolvedMetricConfig:
    validate_selected_ids(list(selection.selected_metrics))
    chosen = set(selection.selected_metrics)
    definitions: list[MetricDefinition] = [
        d for d in builtin_definitions() if d.metric_id in chosen
    ]
    notes: list[str] = []
    compiler_called = bool(selection.custom_inputs)
    if compiler_called:
        compiled, notes = await MetricCompiler(llm, logger, max_retries=max_retries).compile(
            selection.custom_inputs, product, reference, sql_dialect=sql_dialect
        )
        definitions.extend(compiled)
    else:
        logger.info("No custom metrics: metric compiler skipped")
    return freeze(selection, definitions, compiler_called=compiler_called, notes=notes)
