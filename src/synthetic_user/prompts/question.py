"""Prompt for Node 3: natural-language questions for one scenario (any product type).

Both modes: persona + product + scenario. Grounded adds the scenario's Reference Pack and the
usage guide (for the model's understanding only - reference names are never quoted).
Each question owns its expectations: expected_behavior (written), 1-4 success_criteria and
acceptable_variations (design v3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from synthetic_user.models.base import AmbiguityLevel
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import MAX_SUCCESS_CRITERIA, QuestionBatchDraft
from synthetic_user.models.reference import Reference, ReferencePack
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
    grounding: tuple[ReferencePack, Reference] | None,
    options: QuestionPromptOptions,
) -> tuple[str, str]:
    """Return (system, user) asking for ``options.count`` questions for ``scenario``.

    ``grounding`` (pack + reference) is given for grounded scenarios, None for ungrounded.
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

    if grounding is not None:
        pack, reference = grounding
        reference_block = f"""
Usage guide of the reference (for YOUR understanding):
{reference.guide_text()}

Reference pack of this scenario (for YOUR understanding only - never quote its names):
{json.dumps(pack.view(), indent=2)}
"""
        missing = (
            "If the pack shows the goal depends on something ambiguous, say in "
            "expected_behavior\nwhether the product should ask or state its assumption, and "
            "list the valid readings in\nacceptable_variations. Do not guess silently."
        )
    else:
        reference_block = ""
        missing = (
            "No reference data is available: do not invent names, fields or numbers. If the "
            "goal could\nbe read several ways, list the readings in acceptable_variations and "
            "say in expected_behavior\nwhether the product should ask for clarification."
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
{reference_block}
## CONSTRAINTS
- Sound like the persona: their role, vocabulary, expertise and behaviour.
- {identifiers_rule}
- Style mix for ambiguity level "{options.ambiguity_level.value}": {style_mix}
  Set question_characteristics.style to precise | vague | investigative and
  ambiguity_level to low | medium | high for each question. {vague_rule}
- For each question give its intent and expectations:
  - expected_behavior: 1-3 plain sentences describing what the product should do for THIS
    question (what it should return or do; whether asking a clarification or declining
    would be right or wrong). Not a label.
  - success_criteria: 1-{MAX_SUCCESS_CRITERIA} short, checkable must-haves of a goal-complete
    answer, refined from the scenario's success_criteria to this exact question. Include an
    important "must not" as a negative criterion (e.g. "does not mix currencies").
  - acceptable_variations: valid readings or behaviours the evaluator must NOT penalise
    (e.g. "calendar or fiscal quarter", "may add extra columns").
- The intended goal is the scenario's business_goal (code copies it; do not restate it).
- Questions must differ meaningfully from each other.

## OUTPUT CONTRACT
Return one JSON object matching this JSON Schema:
{schema}

## PROHIBITED
- Copying or lightly rewording the persona's example utterances:
{examples}
- ids, goals, leak warnings, timestamps (code adds these).
- Any text outside the JSON object.

## MISSING INFO
{missing}
"""
    return SYSTEM, user
