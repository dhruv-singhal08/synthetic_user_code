"""Prompts for Node 2 (scenarios): two builders sharing a common core.

COMMON (both modes): persona + product description + settings; business goal first; realistic
tasks this persona would do WITH THIS PRODUCT (any product type); capability with a rationale
citing the product description; 2-5 checkable success_criteria; no user questions.
GROUNDED adds the reference index and a reference_cheat_sheet built in the same call.
UNGROUNDED has no index and no cheat sheet, and must not invent data structures or numbers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import ReferenceIndex
from synthetic_user.models.scenario import ScenarioBatchDraft
from synthetic_user.references.cheat_sheet import index_columns

SYSTEM = (
    "You design realistic test scenarios for a synthetic user who will later test a "
    "conversational product. You plan what the user wants to achieve with THIS product; you "
    "never write the user's questions. You reply with a single JSON object and nothing else."
)
GROUNDED_SYSTEM = SYSTEM + (
    " You use ONLY the tables and columns in the reference index for the cheat sheet."
)
UNGROUNDED_SYSTEM = SYSTEM + (
    " No data schema is available: you never invent tables, fields, data structures or numbers."
)


@dataclass(frozen=True)
class ScenarioPromptOptions:
    count: int
    allow_multi_table: bool
    include_unsupported_scenarios: bool
    diversity_preference: str


def build_scenario_prompt(
    persona: Persona,
    product: ProductDescription,
    index: ReferenceIndex | None,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str] = (),
) -> tuple[str, str]:
    """Grounded prompt when an index is given, else the ungrounded one."""
    if index is None:
        return build_ungrounded_scenario_prompt(persona, product, options, avoid_signatures)
    return build_grounded_scenario_prompt(persona, product, index, options, avoid_signatures)


def build_grounded_scenario_prompt(
    persona: Persona,
    product: ProductDescription,
    index: ReferenceIndex,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str] = (),
) -> tuple[str, str]:
    allowed = {
        table: sorted(f"{table}.{c}" for c in columns)
        for table, columns in index_columns(index).items()
    }
    multi_table = (
        "Scenarios MAY use several tables when they are linked by a relationship in the index."
        if options.allow_multi_table
        else "Each scenario MUST use exactly ONE table."
    )
    reference = f"""Reference index (AUTHORITATIVE; table business_rules / calculated_fields are
domain rules the scenario must respect; never plan to expose sensitive_columns):
{json.dumps(_index_view(index), indent=2)}

Allowed columns per table (use these exact "table.column" strings):
{json.dumps(allowed, indent=2)}
"""
    constraints = f"""\
- reference_cheat_sheet: a plausible analysis path (tables, columns, joins, metrics,
  dimensions, operations, optional filters, semantic considerations), built in this same
  answer. It is NOT SQL and NOT a question.
- Use only tables from the reference index and only the allowed "table.column" strings.
  Every table referenced anywhere in the cheat sheet must be listed in
  reference_cheat_sheet.tables.
- For each join, copy the relationship's verification EXACTLY from the reference index.
  A join with no relationship in the index must be "unknown" and explained in
  semantic_considerations.
- {multi_table}"""
    prohibited = """- Tables or columns not in the allowed list; renamed or invented names.
- Marking a join more verified than the reference index does."""
    missing = """\
If the index lacks something a scenario needs (e.g. no relationship between two tables),
say so in semantic_considerations and use "unknown" rather than guessing."""
    user = _core(
        persona,
        product,
        options,
        avoid_signatures,
        task_extra="a reference cheat sheet (reference_cheat_sheet)",
        reference=reference,
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
- There is NO reference index: do NOT include reference_cheat_sheet (leave it out).
- Do not invent data structures, tables, fields, columns, IDs or numbers; describe goals and
  success_criteria at business level only (what the user must learn or be able to do)."""
    prohibited = (
        """- reference_cheat_sheet, table or column names, SQL, schemas, made-up figures."""
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
- ids, persona/product/index ids, mode, signatures, timestamps.
{prohibited}
- Any text outside the JSON object.

## MISSING INFO
{missing} Fewer good scenarios are better than padded ones.
"""


def _index_view(index: ReferenceIndex) -> dict[str, object]:
    """What the scenario LLM sees of the index (no provenance extras)."""
    return {
        "tables": [
            t.model_dump(
                mode="json",
                exclude={
                    "extras": True,
                    "relationships": True,
                    "columns": {"__all__": {"extras"}},
                    "metrics": {"__all__": {"extras"}},
                    "dimensions": {"__all__": {"extras"}},
                    "time_columns": {"__all__": {"extras"}},
                },
                exclude_defaults=True,
            )
            for t in index.tables
        ],
        "relationships": [
            r.model_dump(mode="json", exclude={"extras"}) for r in index.relationships
        ],
        "global_semantic_rules": index.global_semantic_rules,
        "uncertainties": [u.text() for u in index.uncertainties],
    }
