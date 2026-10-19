"""Prompt for Node 3: natural-language questions for one scenario (any product type).

Both modes: persona (read as prompts/persona.py says) + product + scenario. The persona alone
decides HOW the question is asked; the scenario says only WHAT the user wants (D204).
Grounded adds the scenario's cheat sheet (Reference Pack) and the usage instructions, for the
model's understanding only - reference names are never quoted. Each question owns its
expectations: expected_behavior (written) and 1-4 success_criteria (design v3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from synthetic_user.models.base import AmbiguityLevel
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import (
    MAX_SUCCESS_CRITERIA,
    QuestionComplexity,
    QuestionStyle,
)
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.scenario import Scenario, TaskType
from synthetic_user.prompts.contracts import choices, render_contract
from synthetic_user.prompts.persona import question_persona_block

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

QUESTION_EXAMPLE: dict[str, Any] = {
    "questions": [
        {
            "utterance": "<the question exactly as this persona would type it>",
            "intent": {"task_type": choices(TaskType)},
            "question_characteristics": {
                "style": choices(QuestionStyle),
                "ambiguity_level": choices(AmbiguityLevel),
                "complexity": choices(QuestionComplexity),
            },
            "expectations": {
                "expected_behavior": "<1-3 sentences: what the product should do for this "
                "question>",
                "success_criteria": [f"<1-{MAX_SUCCESS_CRITERIA} short, checkable must-haves>"],
            },
        }
    ]
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
    product_json = product.model_dump(
        mode="json",
        include={
            "name",
            "description",
            "data_sources",
            "supported",
            "not_supported",
            "known_limitations",
            "user_visible_capabilities",
        },
    )
    scenario_json = scenario.model_dump(
        mode="json",
        include={
            "business_goal",
            "title",
            "typical_task",
            "task_type",
            "capability",
            "capability_rationale",
            "success_criteria",
            "question_intent",
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
    ambiguity_rule = (
        f'level "{options.ambiguity_level.value}" - {_STYLE_MIX[options.ambiguity_level]} '
        f"{vague_rule}"
    )
    persona_block = question_persona_block(persona, ambiguity_rule=ambiguity_rule)

    if grounding is not None:
        pack, reference = grounding
        reference_block = f"""
Cheat sheet of this scenario (for YOUR understanding only - never use its names):
{pack.render()}

Usage instructions of the reference (for YOUR understanding):
{reference.instructions_text()}
"""
        missing = (
            "If the cheat sheet shows the goal depends on something ambiguous, say in "
            "expected_behavior\nwhether the product should ask or state its assumption. Do not "
            "guess silently."
        )
    else:
        reference_block = ""
        missing = (
            "No reference data is available: do not invent names, fields or numbers. If the "
            "goal could\nbe read several ways, say in expected_behavior which readings are fine "
            "and whether the\nproduct should ask for clarification."
        )

    user = f"""## TASK
Write {options.count} question(s) this persona would type into the product to pursue the
scenario's business goal. Language: {options.language}.

## INPUT
{persona_block}
Product:
{json.dumps(product_json, indent=2)}

Scenario (WHAT the user wants):
{json.dumps(scenario_json, indent=2)}
{reference_block}
## CONSTRAINTS
- The persona decides HOW the question is asked: wording, tone, length, complexity and
  ambiguity come only from "How to use the persona". The scenario only says WHAT is wanted.
- Do not reuse the wording of the scenario's business_goal, title or question_intent; write
  the question as this person would type it.
- {identifiers_rule}
- Set question_characteristics (style, ambiguity_level, complexity) to describe each
  question honestly, following the persona's rules.
- expected_behavior: 1-3 plain sentences: what the product should do for THIS question (what
  it should return or do; whether asking for clarification or declining would be right or
  wrong). If the question can fairly be read in more than one way, say which readings are
  acceptable.
- success_criteria: 1-{MAX_SUCCESS_CRITERIA} short, checkable must-haves of a goal-complete
  answer, refined from the scenario's success_criteria to this exact question. Include an
  important "must not" as a negative criterion (e.g. "does not mix currencies"). Word each
  criterion so that every acceptable reading passes it.
- The intended goal is the scenario's business_goal (code copies it; do not restate it).
- Questions must differ meaningfully from each other.

## OUTPUT CONTRACT
{render_contract(QUESTION_EXAMPLE)}

## PROHIBITED
- Copying or lightly rewording the persona's style examples.
- ids, goals, leak warnings, timestamps (code adds these).
- Any text outside the JSON object.

## MISSING INFO
{missing}
"""
    return SYSTEM, user
