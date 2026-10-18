"""Prompts for Node 2 (scenarios): two builders sharing a common core.

COMMON (both modes): persona + product description + settings; business goal first; realistic
tasks this persona would do WITH THIS PRODUCT (any product type); capability with a rationale
citing the product description; 2-5 checkable success_criteria; no user questions.
GROUNDED adds the user's reference (any format) + usage guide and asks for a reference_pack
(core_refs + rules, each quoting the reference verbatim) in the same call.
UNGROUNDED has no reference and no pack, and must not invent data structures or numbers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import Reference
from synthetic_user.models.scenario import ScenarioBatchDraft

SYSTEM = (
    "You design realistic test scenarios for a synthetic user who will later test a "
    "conversational product. You plan what the user wants to achieve with THIS product; you "
    "never write the user's questions. You reply with a single JSON object and nothing else."
)
GROUNDED_SYSTEM = SYSTEM + (
    " You ground every scenario in the reference supplied by the user and quote it exactly; "
    "you never invent names, facts or rules it does not contain."
)
UNGROUNDED_SYSTEM = SYSTEM + (
    " No data schema is available: you never invent tables, fields, data structures or numbers."
)


@dataclass(frozen=True)
class ScenarioPromptOptions:
    count: int
    include_unsupported_scenarios: bool
    diversity_preference: str


def build_scenario_prompt(
    persona: Persona,
    product: ProductDescription,
    reference: Reference | None,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str] = (),
) -> tuple[str, str]:
    """Grounded prompt when a reference is given, else the ungrounded one."""
    if reference is None:
        return build_ungrounded_scenario_prompt(persona, product, options, avoid_signatures)
    return build_grounded_scenario_prompt(persona, product, reference, options, avoid_signatures)


def build_grounded_scenario_prompt(
    persona: Persona,
    product: ProductDescription,
    reference: Reference,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str] = (),
) -> tuple[str, str]:
    reference_block = f"""Usage guide (how to read and use the reference - follow it):
{reference.guide_text()}

Reference ({reference.format.value} file "{reference.source_name}"; AUTHORITATIVE):
<<<REFERENCE
{reference.text}
REFERENCE>>>

"""
    constraints = """\
- reference_pack: the answer key for this scenario, built in this same answer.
  - core_refs (1-6): what a correct answer MUST rely on (as the usage guide defines "core":
    e.g. tables / columns / metrics, policy sections, documents, facts). name = as the
    reference spells it; quote = a short passage (max ~300 characters) copied EXACTLY from the
    reference that shows it; why = why the goal needs it.
  - rules (0-5): constraints from the reference the answer must respect (e.g. a default
    currency, a calculation definition, an eligibility rule), each with an exact quote.
  - Only what the goal really needs. Things that merely could help are NOT core.
- Quotes are checked by code against the reference: copy them character for character
  (no paraphrase, no "..."). An item whose quote is not found is dropped.
- Base goals and success_criteria on what the reference actually contains."""
    prohibited = """\
- Names, facts, rules or relationships that are not in the reference.
- Paraphrased or shortened-with-ellipsis quotes."""
    missing = """\
If the reference lacks something a goal needs, choose another goal, or classify the capability
as unknown and say what is missing in capability_rationale."""
    user = _core(
        persona,
        product,
        options,
        avoid_signatures,
        task_extra="a reference_pack",
        reference=reference_block,
        constraints=constraints,
        prohibited=prohibited,
        missing=missing,
    )
    return GROUNDED_SYSTEM, user


def build_ungrounded_scenario_prompt(
    persona: Persona,
    product: ProductDescription,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str] = (),
) -> tuple[str, str]:
    constraints = """\
- There is NO reference: do NOT include reference_pack (leave it out).
- Do not invent data structures, tables, fields, columns, IDs or numbers; describe goals and
  success_criteria at business level only (what the user must learn or be able to do)."""
    prohibited = (
        "- reference_pack, internal names (tables, fields, IDs), code, schemas, made-up figures."
    )
    missing = """\
If the product description does not say whether something is supported, classify the
capability as unknown and say what is missing in capability_rationale."""
    user = _core(
        persona,
        product,
        options,
        avoid_signatures,
        task_extra=None,
        reference="",
        constraints=constraints,
        prohibited=prohibited,
        missing=missing,
    )
    return UNGROUNDED_SYSTEM, user


# --- common core --------------------------------------------------------------------------------


def _core(
    persona: Persona,
    product: ProductDescription,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str],
    *,
    task_extra: str | None,
    reference: str,
    constraints: str,
    prohibited: str,
    missing: str,
) -> str:
    persona_json = persona.model_dump(mode="json")
    product_json = product.model_dump(
        mode="json",
        include={
            "name",
            "description",
            "product_type",
            "supported",
            "not_supported_yet",
            "known_gaps",
            "user_visible_capabilities",
            "response_artifacts_expected",
        },
    )
    unsupported = (
        "Include some scenarios the product does NOT support (see not_supported_yet / "
        "known_gaps) and classify them as unsupported."
        if options.include_unsupported_scenarios
        else "Only produce scenarios the product supports."
    )
    avoid = "\n".join(f"- {s}" for s in avoid_signatures) or "- (none)"
    schema = json.dumps(ScenarioBatchDraft.model_json_schema(), indent=2)
    parts = "a title, task_type, capability classification with rationale, success_criteria, "
    parts += f"{task_extra}, and question guidance" if task_extra else "and question guidance"
    return f"""## TASK
Write {options.count} distinct scenario(s) this persona would realistically pursue WITH THIS
PRODUCT (product type: {product.product_type}). For each: the business goal FIRST, then
{parts}.

## INPUT
Persona:
{json.dumps(persona_json, indent=2)}

Product:
{json.dumps(product_json, indent=2)}

{reference}Settings: count={options.count}; diversity_preference={options.diversity_preference!r}.

Existing scenario signatures to AVOID repeating:
{avoid}

## CONSTRAINTS
- business_goal is what the persona wants to achieve, in business terms. Write it first.
- The task must be something this persona would really do with this product.
- capability is supported | unsupported | unknown. capability_rationale must cite the
  specific product description item (supported / not_supported_yet / known_gaps) it relies on.
- {unsupported}
- success_criteria: 2-5 short, checkable statements of what a goal-complete answer must
  contain (an evaluator will tick each one off), in business terms.
{constraints}
- Vary task types and goals; do not repeat an avoided signature.

## OUTPUT CONTRACT
Return one JSON object matching this JSON Schema:
{schema}

## PROHIBITED
- Writing the user's question or any utterance (questions are written later).
- ids, persona/product/reference ids, mode, signatures, timestamps.
{prohibited}
- Any text outside the JSON object.

## MISSING INFO
{missing} Fewer good scenarios are better than padded ones.
"""
