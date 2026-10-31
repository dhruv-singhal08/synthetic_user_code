"""Assemble the Node 2 (scenario) prompts; the prompt text itself is in ``prompt.yaml``.

COMMON (both modes): persona (prompts/persona.py) + product description + settings; business
goal first; realistic tasks this persona would do WITH THIS PRODUCT; capability with a
rationale citing the product description; 2-5 checkable success_criteria; a one-line
question_intent (never the wording); no user questions.
GROUNDED adds the reference (data_catalog or document_collection, rendered by code) + usage
instructions and asks for the scenario's cheat sheet (reference_pack): picks by exact name /
id that code checks and fills with the reference's own text (D203).
UNGROUNDED has no reference and no pack, and must not invent data structures or numbers.
The output contract is a compact JSON example (D204).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from synthetic_user.models.evaluation_profile import EvaluationProfile
from synthetic_user.models.golden_set import GoldenSet
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import DataCatalog, Reference, ReferenceKind
from synthetic_user.models.scenario import CapabilityClassification, TaskType
from synthetic_user.prompts.contracts import choices, render_contract
from synthetic_user.prompts.persona import scenario_persona_block
from synthetic_user.prompts.templates import prompts_beside
from synthetic_user.references.golden_set import golden_inspiration, golden_set_summary

PROMPTS = prompts_beside(__file__)
SYSTEM = PROMPTS.text("system")
GROUNDED_SYSTEM = f"{SYSTEM} {PROMPTS.text('system_grounded_suffix')}"
UNGROUNDED_SYSTEM = f"{SYSTEM} {PROMPTS.text('system_ungrounded_suffix')}"
AGENT_SYSTEM = f"{SYSTEM} {PROMPTS.text('system_agent_suffix')}"

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
# Fields of each golden-set example shown as inspiration.
GOLDEN_EXAMPLE_FIELDS = (
    "case_id",
    "scenario_type",
    "question",
    "expected_behavior",
    "success_criteria",
    "uploaded_criteria",
)


@dataclass(frozen=True)
class ScenarioPromptOptions:
    count: int
    include_unsupported_scenarios: bool
    diversity_preference: str
    max_request_steps: int = 3
    agent_golden_examples: int = 12
    agent_min_tool_calls: int = 1
    agent_min_data_sources: int = 1


def scenario_example(kind: ReferenceKind | None) -> dict[str, Any]:
    scenario: dict[str, Any] = PROMPTS.data(
        "scenario_example",
        task_types=choices(TaskType),
        capabilities=choices(CapabilityClassification),
    )
    if kind is not None:
        scenario["reference_pack"] = PROMPTS.data("pack_examples")[kind.value]
    return {"scenarios": [scenario]}


def _golden_scenario_guidance(golden_set: GoldenSet | None, options: ScenarioPromptOptions) -> str:
    if golden_set is None:
        return ""
    examples = [
        {field: example[field] for field in GOLDEN_EXAMPLE_FIELDS}
        for example in golden_inspiration(golden_set, limit=options.agent_golden_examples)
    ]
    return PROMPTS.text(
        "golden_guidance",
        summary=json.dumps(golden_set_summary(golden_set), indent=2),
        examples=json.dumps(examples, indent=2),
    )


def build_scenario_prompt(
    persona: Persona,
    product: ProductDescription,
    reference: Reference | None,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str] = (),
    evaluation_profile: EvaluationProfile = EvaluationProfile.NL2SQL,
    golden_set: GoldenSet | None = None,
) -> tuple[str, str]:
    """Grounded prompt when a reference is given, else the ungrounded one."""
    if reference is None:
        return build_ungrounded_scenario_prompt(persona, product, options, avoid_signatures)
    if evaluation_profile is EvaluationProfile.AGENT:
        return build_agent_scenario_prompt(
            persona, product, reference, options, avoid_signatures, golden_set=golden_set
        )
    return build_grounded_scenario_prompt(
        persona,
        product,
        reference,
        options,
        avoid_signatures,
        golden_set=golden_set,
    )


def build_agent_scenario_prompt(
    persona: Persona,
    product: ProductDescription,
    reference: Reference,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str] = (),
    *,
    golden_set: GoldenSet | None = None,
) -> tuple[str, str]:
    """Grounded Agent prompt without physical schema or a scenario-level cheat sheet.

    A data catalog may contain an operational description of tools and data sources. Agent
    generation receives only that description and catalog-wide behavioural rules; table,
    column, and join metadata remain exclusive to NL2SQL.
    """
    content = reference.content
    if reference.agent_context is not None:
        rendered = reference.render_agent_context()
    elif isinstance(content, DataCatalog):
        description = content.description or PROMPTS.text("agent_catalog_no_description")
        description = description.split("\n\nTABLE DETAILS", 1)[0].strip()
        context_lines = [PROMPTS.text("agent_catalog_title", name=content.name), description]
        if content.rules:
            context_lines += ["", PROMPTS.text("agent_catalog_rules_title")]
            context_lines += [f"- {rule}" for rule in content.rules]
        rendered = "\n".join(context_lines)
    else:
        rendered = reference.render()
    inspiration = ""
    if golden_set is not None:
        inspiration = PROMPTS.text(
            "agent_inspiration",
            summary=json.dumps(golden_set_summary(golden_set), indent=2),
            examples=json.dumps(
                golden_inspiration(golden_set, limit=options.agent_golden_examples), indent=2
            ),
        )
    golden_constraint = PROMPTS.text(
        "agent_golden_constraint" if golden_set is not None else "agent_no_golden_constraint"
    )
    user = _core(
        persona,
        product,
        options,
        avoid_signatures,
        kind=None,
        reference=PROMPTS.text("agent_reference", context=rendered, inspiration=inspiration),
        constraints=PROMPTS.text(
            "agent_constraints",
            golden_constraint=golden_constraint,
            min_data_sources=options.agent_min_data_sources,
            min_tool_calls=options.agent_min_tool_calls,
        ),
        prohibited=PROMPTS.text("agent_prohibited"),
        missing=PROMPTS.text("agent_missing"),
    )
    return AGENT_SYSTEM, user


def build_grounded_scenario_prompt(
    persona: Persona,
    product: ProductDescription,
    reference: Reference,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str] = (),
    *,
    golden_set: GoldenSet | None = None,
) -> tuple[str, str]:
    label = PROMPTS.text(
        "grounded_label",
        kind=PROMPTS.data("kind_labels")[reference.kind.value],
        source_name=reference.source_name,
    )
    reference_block = PROMPTS.text(
        "grounded_reference",
        label=label,
        reference=reference.render(),
        instructions=reference.instructions_text(),
        golden_guidance=_golden_scenario_guidance(golden_set, options),
    )
    user = _core(
        persona,
        product,
        options,
        avoid_signatures,
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
) -> tuple[str, str]:
    user = _core(
        persona,
        product,
        options,
        avoid_signatures,
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
    return PROMPTS.text(
        "core",
        count=options.count,
        product_name=product.name,
        parts=PROMPTS.text("parts_with_pack" if kind else "parts"),
        persona=scenario_persona_block(persona),
        product=json.dumps(product_json, indent=2),
        reference=reference,
        diversity_preference=repr(options.diversity_preference),
        avoid=avoid,
        max_request_steps=options.max_request_steps,
        unsupported=unsupported,
        constraints=constraints,
        contract=render_contract(scenario_example(kind)),
        prohibited=prohibited,
        missing=missing,
    )
