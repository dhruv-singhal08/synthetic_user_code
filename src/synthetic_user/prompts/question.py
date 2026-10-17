"""Prompt for Node 3: natural-language questions for one scenario (any product type).

Both modes: persona + product + scenario. Grounded adds the scenario's cheat sheet and its
reference snapshot (for the model's understanding only - names are never quoted).
"""

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
    "You role-play a real user typing questions into a conversational product, in their own "
    "words. You never expose internal identifiers, IDs, code or query language unless the "
    "persona would naturally do so. You reply with a single JSON object and nothing else."
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
    # True when this persona would normally use technical identifiers (e.g. a data engineer).
    technical_persona: bool = False


def build_question_prompt(
    persona: Persona,
    product: ProductDescription,
    scenario: Scenario,
    snapshot: ReferenceSnapshot | None,
    options: QuestionPromptOptions,
) -> tuple[str, str]:
    """Return (system, user) asking for ``options.count`` questions for ``scenario``.

    ``snapshot`` is given for grounded scenarios and None for ungrounded ones.
    """
    persona_json = persona.model_dump(
        mode="json",
        include={"identity", "expertise", "behavior", "primary_motivation"},
    )
    product_json = product.model_dump(
        mode="json",
        include={
            "name",
            "product_type",
            "supported",
            "not_supported_yet",
            "known_gaps",
            "user_visible_capabilities",
        },
    )
    scenario_json = scenario.model_dump(
        mode="json",
        include={
            "business_goal",
            "title",
            "task_type",
            "capability",
            "capability_rationale",
            "success_criteria",
            "question_guidance",
        },
    )
    vague_rule = (
        "Vague questions are allowed."
        if options.allow_vague_questions
        else 'Do NOT write vague questions (style "vague" is not allowed).'
    )
    identifiers_rule = (
        "This persona is technical: internal names or identifiers are acceptable where they "
        "would naturally use them, but plain language is still preferred."
        if options.technical_persona
        else "Plain language only: no internal identifiers, IDs, field or table names, "
        "snake_case words, code or query language."
    )
    examples = "\n".join(f"- {u}" for u in persona.example_utterances) or "- (none)"
    schema = json.dumps(QuestionBatchDraft.model_json_schema(), indent=2)
    style_mix = _STYLE_MIX[options.ambiguity_level]

    if snapshot is not None and scenario.reference_cheat_sheet is not None:
        sheet = scenario.reference_cheat_sheet.model_dump(mode="json")
        reference = f"""
Reference cheat sheet (for YOUR understanding only - never quote its names):
{json.dumps(sheet, indent=2)}

Reference snapshot (for YOUR understanding only - never quote its names):
{json.dumps(_snapshot_view(snapshot), indent=2)}
"""
        missing = (
            "If the snapshot lists missing_references or the scenario depends on unknown "
            "relationships,\nreflect that in interpretation_notes and expectations (e.g. the "
            "product should ask for\nclarification or state its assumption). Do not guess "
            "silently."
        )
    else:
        reference = ""
        missing = (
            "No reference data is available: do not invent names, fields or numbers. If the "
            "goal could\nbe read several ways, list the readings in interpretation_notes and "
            "say in expectations\nwhether the product should ask for clarification."
        )

    user = f"""## TASK
Write {options.count} question(s) this persona would type into the product to pursue the
scenario's business goal. Language: {options.language}. The product type
({product.product_type}) is context only; ask what this persona would naturally ask.

## INPUT
Persona (sound like this person):
{json.dumps(persona_json, indent=2)}

Product:
{json.dumps(product_json, indent=2)}

Scenario:
{json.dumps(scenario_json, indent=2)}
{reference}
## CONSTRAINTS
- Sound like the persona: their role, vocabulary, expertise and behaviour.
- {identifiers_rule}
- Style mix for ambiguity level "{options.ambiguity_level.value}": {style_mix}
  Set question_characteristics.style to precise | vague | investigative and
  ambiguity_level to low | medium | high for each question. {vague_rule}
- For each question give: intent, interpretation_notes (a question may have several valid
  interpretations - list them), expectations.clarification (required | conditional |
  not_expected), acceptable_behaviors and unacceptable_behaviors for the product.
  expectations.success_criteria: copy the scenario's success_criteria, refined to this
  exact question if needed (leave empty to copy them unchanged).
- The intended goal is the scenario's business_goal (code copies it; do not restate it).
- Questions must differ meaningfully from each other.

## OUTPUT CONTRACT
Return one JSON object matching this JSON Schema:
{schema}

## PROHIBITED
- Copying or lightly rewording the persona's example utterances:
{examples}
- ids, goals, schema-leak warnings, timestamps (code adds these).
- Any text outside the JSON object.

## MISSING INFO
{missing}
"""
    return SYSTEM, user


def _snapshot_view(snapshot: ReferenceSnapshot) -> dict[str, object]:
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
            for t in snapshot.tables
        ],
        "relationships": [
            r.model_dump(mode="json", exclude={"extras"}) for r in snapshot.relationships
        ],
        "missing_references": snapshot.missing_references,
    }
