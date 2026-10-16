"""Prompt for Node 3: natural-language questions for one scenario."""

from __future__ import annotations

import json
from dataclasses import dataclass

from synthetic_user.models.base import AmbiguityLevel
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionBatchDraft
from synthetic_user.models.reference import ReferenceSnapshot
from synthetic_user.models.scenario import Scenario

SYSTEM = (
    "You role-play a business user asking an NL2SQL assistant questions in their own words. "
    "You never use database table or column names and never write SQL. "
    "You reply with a single JSON object and nothing else."
)

_STYLE_MIX = {
    AmbiguityLevel.LOW: "Mostly precise questions; at most one vague or investigative one.",
    AmbiguityLevel.MEDIUM: "A mix of precise, vague and investigative questions.",
    AmbiguityLevel.HIGH: "Mostly vague or investigative questions that need interpretation "
    "or clarification.",
}


@dataclass(frozen=True)
class QuestionPromptOptions:
    count: int
    ambiguity_level: AmbiguityLevel
    language: str
    allow_vague_questions: bool


def build_question_prompt(
    persona: Persona,
    product: ProductDescription,
    scenario: Scenario,
    snapshot: ReferenceSnapshot,
    options: QuestionPromptOptions,
) -> tuple[str, str]:
    """Return (system, user) asking for ``options.count`` questions for ``scenario``."""
    persona_json = persona.model_dump(
        mode="json",
        include={"identity", "expertise", "behavior", "primary_motivation"},
    )
    product_json = product.model_dump(
        mode="json", include={"name", "supported", "not_supported_yet", "known_gaps"}
    )
    scenario_json = scenario.model_dump(
        mode="json",
        include={
            "business_goal",
            "title",
            "task_type",
            "capability",
            "capability_rationale",
            "question_guidance",
        },
    )
    cheat_json = scenario.cheat_sheet.model_dump(mode="json")
    snapshot_json = snapshot.model_dump(
        mode="json", include={"tables", "relationships", "missing_references"}
    )
    vague_rule = (
        "Vague questions are allowed."
        if options.allow_vague_questions
        else 'Do NOT write vague questions (style "vague" is not allowed).'
    )
    examples = "\n".join(f"- {u}" for u in persona.example_utterances) or "- (none)"
    schema = json.dumps(QuestionBatchDraft.model_json_schema(), indent=2)
    style_mix = _STYLE_MIX[options.ambiguity_level]

    user = f"""## TASK
Write {options.count} question(s) this persona would type into the product to pursue the
scenario's business goal. Language: {options.language}.

## INPUT
Persona (sound like this person):
{json.dumps(persona_json, indent=2)}

Product:
{json.dumps(product_json, indent=2)}

Scenario:
{json.dumps(scenario_json, indent=2)}

Reference cheat sheet (for YOUR understanding only - never quote its names):
{json.dumps(cheat_json, indent=2)}

Reference snapshot (for YOUR understanding only - never quote its names):
{json.dumps(snapshot_json, indent=2)}

## CONSTRAINTS
- Sound like the persona: their role, vocabulary, expertise and behaviour.
- Business language only. No table names, no column names, no snake_case identifiers, no SQL.
- Style mix for ambiguity level "{options.ambiguity_level.value}": {style_mix}
  Set question_characteristics.style to precise | vague | investigative and
  ambiguity_level to low | medium | high for each question. {vague_rule}
- For each question give: intent, interpretation_notes (a question may have several valid
  interpretations - list them), expectations.clarification (required | conditional |
  not_expected), acceptable_behaviors and unacceptable_behaviors for the product.
- Questions must differ meaningfully from each other.

## OUTPUT CONTRACT
Return one JSON object matching this JSON Schema:
{schema}

## PROHIBITED
- Table or column names, snake_case identifiers, SQL.
- Copying or lightly rewording the persona's example utterances:
{examples}
- ids, goals, schema-leak warnings, timestamps (code adds these).
- Any text outside the JSON object.

## MISSING INFO
If the snapshot lists missing_references or the scenario depends on unknown relationships,
reflect that in interpretation_notes and expectations (e.g. the product should ask for
clarification or state its assumption). Do not guess silently.
"""
    return SYSTEM, user
