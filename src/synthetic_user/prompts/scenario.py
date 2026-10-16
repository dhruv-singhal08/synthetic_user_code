"""Prompt for Node 2: scenarios + reference cheat sheets in one call."""

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
    "You design realistic analytics scenarios for a synthetic business user who will later "
    "test an NL2SQL product. You use ONLY the tables and columns in the reference index. "
    "You reply with a single JSON object and nothing else."
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
    index: ReferenceIndex,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str] = (),
) -> tuple[str, str]:
    """Return (system, user) asking for ``options.count`` scenarios."""
    persona_json = persona.model_dump(mode="json")
    product_json = product.model_dump(
        mode="json",
        include={"name", "description", "supported", "not_supported_yet", "known_gaps"},
    )
    index_json = index.model_dump(mode="json", include={"tables", "uncertainties"})
    allowed = {
        table: sorted(f"{table}.{c}" for c in columns)
        for table, columns in index_columns(index).items()
    }
    multi_table = (
        "Scenarios MAY use several tables when they are linked by a relationship in the index."
        if options.allow_multi_table
        else "Each scenario MUST use exactly ONE table."
    )
    unsupported = (
        "Include some scenarios the product does NOT support (see not_supported_yet / "
        "known_gaps) and classify them as unsupported."
        if options.include_unsupported_scenarios
        else "Only produce scenarios the product supports."
    )
    avoid = "\n".join(f"- {s}" for s in avoid_signatures) or "- (none)"
    schema = json.dumps(ScenarioBatchDraft.model_json_schema(), indent=2)

    user = f"""## TASK
Write {options.count} distinct scenario(s) this persona would realistically pursue.
For each: the business goal FIRST, then a title, task_type, capability classification with
rationale, a reference cheat sheet, and question guidance.

## INPUT
Persona:
{json.dumps(persona_json, indent=2)}

Product:
{json.dumps(product_json, indent=2)}

Reference index:
{json.dumps(index_json, indent=2)}

Allowed columns per table (use these exact "table.column" strings):
{json.dumps(allowed, indent=2)}

Settings: count={options.count}; diversity_preference={options.diversity_preference!r}.

Existing scenario signatures to AVOID repeating
(format task_type|goal keywords|tables|metrics|dimensions|operations):
{avoid}

## CONSTRAINTS
- business_goal is what the persona wants to achieve, in business terms. Write it first.
- The cheat sheet is a plausible analysis path (tables, columns, joins, metrics, dimensions,
  operations, optional filters, semantic considerations). It is NOT SQL.
- Use only tables from the reference index and only the allowed "table.column" strings.
  Every table referenced anywhere in the cheat sheet must be listed in cheat_sheet.tables.
- For each join, copy the relationship's verification EXACTLY from the reference index.
  A join with no relationship in the index must be "unknown" and explained in
  semantic_considerations.
- {multi_table}
- {unsupported}
- capability is supported | unsupported | unknown. capability_rationale must cite the
  specific product description item (supported / not_supported_yet / known_gaps) it relies on.
- Vary task types, tables, metrics and dimensions; do not repeat an avoided signature.

## OUTPUT CONTRACT
Return one JSON object matching this JSON Schema:
{schema}

## PROHIBITED
- SQL of any kind.
- Writing the user's question or any utterance (questions are written later).
- Tables or columns not in the allowed list; renamed or invented names.
- Marking a join more verified than the reference index does.
- ids, persona/product/index ids, signatures, timestamps.
- Any text outside the JSON object.

## MISSING INFO
If the index lacks something a scenario needs (e.g. no relationship between two tables),
say so in semantic_considerations and use "unknown" rather than guessing. Fewer good
scenarios are better than padded ones.
"""
    return SYSTEM, user
