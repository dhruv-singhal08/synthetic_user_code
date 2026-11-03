"""Assemble the metric compiler prompt; the prompt text itself is in ``compiler_prompt.yaml``."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from synthetic_user.metrics.catalog import builtin_definitions
from synthetic_user.models.metrics import (
    CustomMetricInput,
    EvidenceKind,
    ExpectationFieldType,
    JudgeGroup,
    ReferenceAsset,
)
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import DataCatalog, Reference
from synthetic_user.prompts.contracts import choices, render_contract
from synthetic_user.prompts.templates import prompts_beside

PROMPTS = prompts_beside(__file__, "compiler_prompt.yaml")
SYSTEM = PROMPTS.text("system")
# Judge groups a compiled custom metric may use (no safe deterministic custom implementation).
COMPILABLE_JUDGE_GROUPS = (JudgeGroup.GENERAL, JudgeGroup.RETRIEVAL_GROUNDING)
UNIVERSAL_QUESTION_FIELDS = ("utterance", "expected_behavior", "success_criteria")
PRODUCT_FIELDS = {
    "name",
    "description",
    "data_sources",
    "supported",
    "not_supported",
    "known_limitations",
    "user_visible_capabilities",
    "expected_outputs",
}


def compiler_example() -> dict[str, Any]:
    example: dict[str, Any] = PROMPTS.data(
        "compiled_example",
        judge_groups=choices(COMPILABLE_JUDGE_GROUPS),
        evidence_kinds=choices(EvidenceKind),
        reference_assets=choices(ReferenceAsset),
        field_types=choices(ExpectationFieldType),
    )
    return example


def reference_summary(reference: Reference | None) -> dict[str, Any] | str:
    if reference is None:
        return PROMPTS.text("no_reference")
    content = reference.content
    summary: dict[str, Any] = {"kind": reference.kind.value, "name": content.name}
    if isinstance(content, DataCatalog):
        summary["tables"] = [t.name for t in content.tables]
        summary["catalog_rules"] = len(content.rules)
        summary["relationships"] = len(content.relationships)
    else:
        summary["documents"] = [d.name for d in content.documents]
    agent = reference.agent_context
    summary["tool_definitions"] = [t.name for t in agent.tools] if agent is not None else []
    return summary


def build_compiler_prompt(
    inputs: Sequence[CustomMetricInput],
    product: ProductDescription,
    reference: Reference | None,
    *,
    sql_dialect: str | None,
) -> tuple[str, str]:
    numbered = "\n".join(
        f"{index}. [{item.provenance}] {item.text}" for index, item in enumerate(inputs, start=1)
    )
    builtins = [
        {"id": d.metric_id, "name": d.display_name, "description": d.description}
        for d in builtin_definitions()
    ]
    user = PROMPTS.text(
        "user",
        count=len(inputs),
        custom_metrics=numbered,
        product=json.dumps(
            product.model_dump(mode="json", include=PRODUCT_FIELDS, exclude_none=True), indent=2
        ),
        reference=json.dumps(reference_summary(reference), indent=2),
        sql_dialect=sql_dialect or PROMPTS.text("no_sql_dialect"),
        builtins=json.dumps(builtins, indent=2),
        question_fields=", ".join(UNIVERSAL_QUESTION_FIELDS),
        evidence_kinds=", ".join(kind.value for kind in EvidenceKind),
        reference_assets=", ".join(asset.value for asset in ReferenceAsset),
        contract=render_contract(compiler_example()),
    )
    return SYSTEM, user
