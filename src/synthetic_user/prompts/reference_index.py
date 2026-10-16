"""Prompt for Node 1: build a reference index from catalog tables."""

from __future__ import annotations

import json
from collections.abc import Sequence

from synthetic_user.models.reference import ReferenceIndexDraft, Relationship, TableMetadata

SYSTEM = (
    "You document a data catalog for analysts. You describe ONLY what the catalog input "
    "contains. You never invent tables, columns, or relationships. "
    "You reply with a single JSON object and nothing else."
)


def build_reference_index_prompt(
    tables: Sequence[TableMetadata], relationships: Sequence[Relationship]
) -> tuple[str, str]:
    """Return (system, user) for indexing ``tables``."""
    catalog_input = {
        "tables": [t.model_dump(mode="json", exclude_none=True) for t in tables],
        "relationships": [r.model_dump(mode="json") for r in relationships],
    }
    schema = json.dumps(ReferenceIndexDraft.model_json_schema(), indent=2)
    names = ", ".join(t.name for t in tables)
    user = f"""## TASK
Build a reference index entry for each of these catalog tables: {names}.
For each table give its business purpose, grain, key entities, important columns, metrics,
dimensions, time columns, and relationships to other tables.

## INPUT
Catalog (authoritative):
{json.dumps(catalog_input, indent=2)}

## CONSTRAINTS
- Output exactly one entry per input table, using the table name exactly as written.
- Every column you mention must exist on that table in INPUT. Write it as the bare column
  name (e.g. "amount"), exactly as written in INPUT.
- metrics[].columns, important_columns, dimensions, time_columns and
  relationships[].source_column refer to columns of the table they are listed under.
- relationships[].target_table / target_column must exist in INPUT.
- time_columns must be date/time columns.
- verification may be "verified" ONLY when INPUT lists the same relationship as "verified".
  A relationship you deduce yourself is "inferred". If unsure, use "unknown".
- cardinality: one_to_one | one_to_many | many_to_one | many_to_many | unknown.

## OUTPUT CONTRACT
Return one JSON object matching this JSON Schema:
{schema}

## PROHIBITED
- Inventing, renaming, translating, or pluralising tables or columns.
- Tables that are not in INPUT, or omitting a table that is in INPUT.
- Marking a relationship "verified" that INPUT does not mark "verified".
- ids, versions, timestamps, provider or model names.
- Any text outside the JSON object.

## MISSING INFO
If the catalog does not tell you something (e.g. a table has no description), write
"unknown" for that text field and add a short note to "uncertainties" naming the table
and what is missing. Never guess silently.
"""
    return SYSTEM, user
