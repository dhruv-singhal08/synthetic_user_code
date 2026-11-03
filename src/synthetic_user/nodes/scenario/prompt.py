"""Assemble the Node 2 (scenario) prompts; the prompt text itself is in ``prompt.yaml``.

COMMON (both modes): persona (prompts/persona.py) + product description + settings + the run's
selected metrics as coverage guidance only; business goal first; realistic tasks this persona
would do WITH THIS PRODUCT; capability with a rationale citing the product description;
2-5 checkable success_criteria; a one-line question_intent (never the wording); no questions.
GROUNDED adds the reference (data_catalog or document_collection, rendered by code, plus its
tool definitions when present) + usage instructions and asks for the scenario's cheat sheet
(reference_pack): picks by exact name / id that code checks and fills with the reference's
own text (D203).
UNGROUNDED has no reference and no pack, and must not invent data structures or numbers.
The output contract is a compact JSON example (D204).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from synthetic_user.models.metrics import MetricDefinition
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import Reference, ReferenceKind
from synthetic_user.models.scenario import CapabilityClassification, TaskType
from synthetic_user.prompts.contracts import choices, render_contract
from synthetic_user.prompts.persona import scenario_persona_block
from synthetic_user.prompts.templates import prompts_beside

PROMPTS = prompts_beside(__file__)
SYSTEM = PROMPTS.text("system")
GROUNDED_SYSTEM = f"{SYSTEM} {PROMPTS.text('system_grounded_suffix')}"
UNGROUNDED_SYSTEM = f"{SYSTEM} {PROMPTS.text('system_ungrounded_suffix')}"

# Fields of the product description shown to the scenario generator.
PRODUCT_FIELDS = {
    "name",
    "description",
    "business_problem",
    "data_sources",
    "supported",
    "not_supported",
    "known_limitations",
    "user_visible_capabilities",
    "typical_user_questions",
    "expected_outputs",
}


@dataclass(frozen=True)
class ScenarioPromptOptions:
    count: int
    include_unsupported_scenarios: bool
    diversity_preference: str
    max_request_steps: int = 3


def scenario_example(kind: ReferenceKind | None) -> dict[str, Any]:
    scenario: dict[str, Any] = PROMPTS.data(
        "scenario_example",
        task_types=choices(TaskType),
        capabilities=choices(CapabilityClassification),
    )
    if kind is not None:
        scenario["reference_pack"] = PROMPTS.data("pack_examples")[kind.value]
    return {"scenarios": [scenario]}


def build_scenario_prompt(
    persona: Persona,
    product: ProductDescription,
    reference: Reference | None,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str] = (),
    metrics: Sequence[MetricDefinition] = (),
) -> tuple[str, str]:
    """Grounded prompt when a reference is given, else the ungrounded one."""
    if reference is None:
        return build_ungrounded_scenario_prompt(
            persona, product, options, avoid_signatures, metrics
        )
    return build_grounded_scenario_prompt(
        persona, product, reference, options, avoid_signatures, metrics
    )


def build_grounded_scenario_prompt(
    persona: Persona,
    product: ProductDescription,
    reference: Reference,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str] = (),
    metrics: Sequence[MetricDefinition] = (),
) -> tuple[str, str]:
    label = PROMPTS.text(
        "grounded_label",
        kind=PROMPTS.data("kind_labels")[reference.kind.value],
        source_name=reference.source_name,
    )
    tools = (
        PROMPTS.text("grounded_tools", tools=reference.render_agent_context())
        if reference.agent_context is not None
        else ""
    )
    reference_block = PROMPTS.text(
        "grounded_reference",
        label=label,
        reference=reference.render(),
        tools=tools,
        instructions=reference.instructions_text(),
    )
    user = _core(
        persona,
        product,
        options,
        avoid_signatures,
        metrics,
        kind=reference.kind,
        reference=reference_block,
        constraints=PROMPTS.text(
            "grounded_constraints", pack_rules=PROMPTS.data("pack_rules")[reference.kind.value]
        ),
        prohibited=PROMPTS.text("grounded_prohibited"),
        missing=PROMPTS.text("grounded_missing"),
    )
    return GROUNDED_SYSTEM, user


def build_ungrounded_scenario_prompt(
    persona: Persona,
    product: ProductDescription,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str] = (),
    metrics: Sequence[MetricDefinition] = (),
) -> tuple[str, str]:
    user = _core(
        persona,
        product,
        options,
        avoid_signatures,
        metrics,
        kind=None,
        reference="",
        constraints=PROMPTS.text("ungrounded_constraints"),
        prohibited=PROMPTS.text("ungrounded_prohibited"),
        missing=PROMPTS.text("ungrounded_missing"),
    )
    return UNGROUNDED_SYSTEM, user


# --- common core --------------------------------------------------------------------------------


def _core(
    persona: Persona,
    product: ProductDescription,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str],
    metrics: Sequence[MetricDefinition],
    *,
    kind: ReferenceKind | None,
    reference: str,
    constraints: str,
    prohibited: str,
    missing: str,
) -> str:
    product_json = product.model_dump(mode="json", include=PRODUCT_FIELDS, exclude_none=True)
    unsupported = PROMPTS.text(
        "unsupported_included" if options.include_unsupported_scenarios else "unsupported_excluded"
    )
    avoid = "\n".join(f"- {s}" for s in avoid_signatures) or PROMPTS.text("no_signatures_to_avoid")
    metrics_block = (
        PROMPTS.text(
            "metrics_block",
            metrics="\n".join(
                PROMPTS.text("metric_item", name=m.display_name, description=m.description)
                for m in metrics
            ),
        )
        if metrics
        else ""
    )
    return PROMPTS.text(
        "core",
        count=options.count,
        product_name=product.name,
        parts=PROMPTS.text("parts_with_pack" if kind else "parts"),
        persona=scenario_persona_block(persona),
        product=json.dumps(product_json, indent=2),
        reference=reference,
        metrics=metrics_block,
        diversity_preference=repr(options.diversity_preference),
        avoid=avoid,
        max_request_steps=options.max_request_steps,
        unsupported=unsupported,
        constraints=constraints,
        metric_rule=PROMPTS.text("metric_rule") if metrics else "",
        contract=render_contract(scenario_example(kind)),
        prohibited=prohibited,
        missing=missing,
    )
