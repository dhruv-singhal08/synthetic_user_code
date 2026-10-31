"""How the scenario and question prompts read the persona (persona schema v2, D200).

Each persona field is shown to the LLM together with an explicit rule for using it, so a
change in any value changes what is generated. ``id`` / ``version`` are never shown. The text
is in ``persona.yaml``.

Scenarios use: identity, expertise, behavior.traits, typical_tasks, primary_motivation,
decision_inputs, question_preferences.complexity.
Questions use: identity, expertise, behavior (style + traits), primary_motivation,
decision_inputs, question_preferences (complexity, follow_up_questions, ambiguity) and
example_utterances (style examples, never copied).
"""

from __future__ import annotations

import json
from typing import Any, Literal

from synthetic_user.models.persona import ComplexityPreference, Persona
from synthetic_user.prompts.templates import prompts_beside

PROMPTS = prompts_beside(__file__, "persona.yaml")
COMPLEXITY_MEANING: dict[ComplexityPreference, str] = {
    ComplexityPreference(level): meaning
    for level, meaning in PROMPTS.data("complexity_meaning").items()
}


def _identity(persona: Persona) -> dict[str, Any]:
    return persona.identity.model_dump(mode="json", exclude_none=True)


def _expertise(persona: Persona) -> dict[str, Any]:
    levels = {area: level.value for area, level in persona.expertise.levels().items()}
    view: dict[str, Any] = {"levels": levels}
    if persona.expertise.notes:
        view["notes"] = persona.expertise.notes
    return view


def _complexity_rule(persona: Persona, unit: Literal["scenarios", "questions"]) -> str:
    """``unit``: "scenarios" (one call writes them all, so they can be spread) or
    "questions" (one call per scenario, so each fits its scenario's goal)."""
    complexity = persona.question_preferences.complexity
    if complexity is not ComplexityPreference.MIXED:
        meaning = COMPLEXITY_MEANING[complexity]
        if unit == "questions":
            return PROMPTS.text(
                "complexity_fixed_questions", complexity=complexity.value, meaning=meaning
            )
        return PROMPTS.text(
            "complexity_fixed_scenarios",
            complexity=complexity.value,
            unit=unit.removesuffix("s"),
            meaning=meaning,
        )
    meanings = "; ".join(
        PROMPTS.text("complexity_meaning_item", complexity=level.value, meaning=meaning)
        for level, meaning in COMPLEXITY_MEANING.items()
    )
    return PROMPTS.text(f"complexity_mixed_{unit}", meanings=meanings)


def scenario_persona_block(persona: Persona) -> str:
    view = {
        "identity": _identity(persona),
        "expertise": _expertise(persona),
        "traits": persona.behavior.traits,
        "typical_tasks": persona.typical_tasks,
        "primary_motivation": persona.primary_motivation,
        "decision_inputs": persona.decision_inputs,
        "question_complexity": persona.question_preferences.complexity.value,
    }
    return PROMPTS.text(
        "scenario_block",
        persona=json.dumps(view, indent=2),
        expertise_rule=PROMPTS.text("expertise_rule"),
        complexity_rule=_complexity_rule(persona, "scenarios"),
    )


def question_persona_block(persona: Persona, *, ambiguity_rule: str) -> str:
    prefs = persona.question_preferences
    view = {
        "identity": _identity(persona),
        "expertise": _expertise(persona),
        "style": persona.behavior.style,
        "traits": persona.behavior.traits,
        "primary_motivation": persona.primary_motivation,
        "decision_inputs": persona.decision_inputs,
        "question_preferences": prefs.model_dump(mode="json", exclude_none=True),
    }
    examples = "\n".join(
        PROMPTS.text("style_example", utterance=utterance)
        for utterance in persona.example_utterances
    ) or PROMPTS.text("no_style_examples")
    return PROMPTS.text(
        "question_block",
        persona=json.dumps(view, indent=2),
        examples=examples,
        expertise_rule=PROMPTS.text("expertise_rule"),
        complexity_rule=_complexity_rule(persona, "questions"),
        follow_ups=PROMPTS.text("follow_ups_on" if prefs.follow_up_questions else "follow_ups_off"),
        ambiguity_rule=ambiguity_rule,
    )
