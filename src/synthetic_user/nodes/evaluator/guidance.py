"""Evaluation Guidance ids: what a judge may cite as guidance, and how it is recorded.

Two sources, both resolvable (reforming, Part 5):

- the user's evaluation guidance, structured once when it was saved into verbatim items
  ``G1``, ``G2``, ... (``models.run.GuidanceItem``);
- built-in evaluator guidance: each selected metric's rubric (``METRIC:<metric id>``, from the
  frozen metric definition) and the few named policies in ``judge_common_prompt.yaml``
  (``POLICY:<name>``). Not every prompt sentence is a rule: only these ids exist.

A judge cites ids; code checks that each resolves and records the exact instruction (or its
source reference). Guidance is never merged into the question's Success Criteria.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from synthetic_user.models.metrics import MetricDefinition, MetricOrigin
from synthetic_user.models.run import EvaluationGuidance, GuidanceType
from synthetic_user.nodes.evaluator.judge_common import PROMPTS, guidance_items

METRIC_PREFIX = "METRIC:"


@dataclass(frozen=True)
class GuidanceRef:
    guidance_id: str
    guidance_type: str
    source: str
    instruction: str
    metrics: tuple[str, ...] = ()  # () = every metric
    user: bool = False  # True for the user's own guidance items


def metric_guidance_id(metric_id: str) -> str:
    return f"{METRIC_PREFIX}{metric_id}"


def guidance_registry(
    definitions: Sequence[MetricDefinition],
    guidance: EvaluationGuidance | None,
    *,
    catalog_version: str,
) -> dict[str, GuidanceRef]:
    """Every citable guidance id of one run, user items first."""
    registry: dict[str, GuidanceRef] = {}
    if guidance is not None:
        for item in guidance_items(guidance):
            registry[item.guidance_id] = GuidanceRef(
                guidance_id=item.guidance_id,
                guidance_type=item.type.value,
                source=f"evaluation guidance ({item.location})",
                instruction=item.text,
                metrics=tuple(item.metrics),
                user=True,
            )
    for definition in definitions:
        source = (
            PROMPTS.text("metric_guidance_source_builtin", catalog_version=catalog_version)
            if definition.origin is MetricOrigin.BUILTIN
            else PROMPTS.text(
                "metric_guidance_source_custom",
                provenance=definition.source_provenance or definition.metric_id,
            )
        )
        registry[metric_guidance_id(definition.metric_id)] = GuidanceRef(
            guidance_id=metric_guidance_id(definition.metric_id),
            guidance_type=GuidanceType.SCORING_POLICY.value,
            source=source,
            instruction=PROMPTS.text(
                "metric_guidance_instruction",
                display_name=definition.display_name,
                version=definition.version,
            ),
            metrics=(definition.metric_id,),
        )
    for guidance_id, body in PROMPTS.data("builtin_guidance").items():
        registry[guidance_id] = GuidanceRef(
            guidance_id=guidance_id,
            guidance_type=str(body["type"]),
            source=str(body["source"]),
            instruction=str(body["instruction"]),
        )
    return registry


def builtin_listing(registry: dict[str, GuidanceRef]) -> list[dict[str, str]]:
    """The built-in policy ids as the General Judge sees them (metric rubrics are named in the
    METRICS section; the user's items in the EVALUATION GUIDANCE section)."""
    return [
        {"id": ref.guidance_id, "type": ref.guidance_type, "instruction": ref.instruction}
        for ref in registry.values()
        if not ref.user and not ref.guidance_id.startswith(METRIC_PREFIX)
    ]
