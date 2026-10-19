"""How the scenario and question prompts read the persona (persona schema v2, D200).

Each persona field is shown to the LLM together with an explicit rule for using it, so a
change in any value changes what is generated. ``id`` / ``version`` are never shown.

Scenarios use: identity, expertise, behavior.traits, typical_tasks, primary_motivation,
decision_inputs, question_preferences.complexity.
Questions use: identity, expertise, behavior (style + traits), primary_motivation,
decision_inputs, question_preferences (complexity, follow_up_questions, ambiguity) and
example_utterances (style examples, never copied).
"""

from __future__ import annotations

import json
from typing import Any

from synthetic_user.models.persona import ComplexityPreference, Persona

COMPLEXITY_MEANING = {
    ComplexityPreference.SIMPLE: "one fact, metric or item, with at most one filter or period",
    ComplexityPreference.MEDIUM: "a breakdown, ranking or comparison with two or three conditions",
    ComplexityPreference.COMPLEX: "several parts or dimensions at once: comparisons across "
    "segments or periods, root causes, trade-offs or impact",
}


def _identity(persona: Persona) -> dict[str, Any]:
    return persona.identity.model_dump(mode="json", exclude_none=True)


def _expertise(persona: Persona) -> dict[str, Any]:
    levels = {area: level.value for area, level in persona.expertise.levels().items()}
    view: dict[str, Any] = {"levels": levels}
    if persona.expertise.notes:
        view["notes"] = persona.expertise.notes
    return view


def _complexity_rule(persona: Persona, unit: str) -> str:
    """``unit``: "scenarios" (one call writes them all, so they can be spread) or
    "questions" (one call per scenario, so each fits its scenario's goal)."""
    complexity = persona.question_preferences.complexity
    meanings = "; ".join(f"{c.value} = {m}" for c, m in COMPLEXITY_MEANING.items())
    if complexity is not ComplexityPreference.MIXED:
        return (
            f"complexity is {complexity.value}: every {unit[:-1]} is {complexity.value} - "
            f"{COMPLEXITY_MEANING[complexity]}."
        )
    if unit == "scenarios":
        return f"complexity is mixed: spread the scenarios across all three ({meanings})."
    return f"complexity is mixed: use the one that fits the scenario's goal ({meanings})."


def _expertise_rule() -> str:
    return (
        "expertise.levels: per area, low | medium | high. In a high area use its specialist "
        "vocabulary and go deep; in a low area stay basic and in plain words. expertise.notes "
        "say what this person knows and does not know - respect them."
    )


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
    return f"""Persona (who pursues these goals):
{json.dumps(view, indent=2)}

How to use the persona:
- typical_tasks: base each scenario on one of them and copy its exact text into
  typical_task. Spread the scenarios over different typical tasks (use as many different
  ones as the count allows before repeating one). typical_task is null only for another task
  this persona would clearly do.
- primary_motivation: frame each business_goal as a step towards it.
- decision_inputs: the evidence this persona relies on. Prefer goals whose answer draws on
  one of these AND that the product can provide (a goal needing evidence the product does
  not have is an unsupported scenario).
- traits: let them shape what each goal looks for and its success_criteria (e.g. "explores
  root causes" -> a criterion about the cause, not only the number).
- {_expertise_rule()}
- question_complexity: {_complexity_rule(persona, "scenarios")}
"""


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
    follow_ups = (
        "follow_up_questions is true: this person digs deeper (why, impact, risk, what next). "
        "Each question is sent on its own with no conversation history, so write such a "
        "question self-contained: restate the context, then ask the deeper question."
        if prefs.follow_up_questions
        else "follow_up_questions is false: write first questions only - no 'why did that "
        "happen' or 'what about...' style follow-ups."
    )
    examples = "\n".join(f"  - {u}" for u in persona.example_utterances) or "  - (none given)"
    return f"""Persona (sound like this person):
{json.dumps(view, indent=2)}

Style examples - how this person writes (match their length, tone and vocabulary; NEVER copy
or lightly reword them, copies are rejected):
{examples}

How to use the persona:
- style: the tone and length of every question.
- traits: every question must reflect at least one trait (what they ask about and how).
- {_expertise_rule()}
- primary_motivation: the reason behind the question; it shapes what they ask for.
- decision_inputs: the kinds of evidence they ask for or refer to (only those that fit the
  scenario and the product).
- {_complexity_rule(persona, "questions")} Set question_characteristics.complexity to
  simple | medium | complex for each question.
- {follow_ups}
- ambiguity: {ambiguity_rule}
"""
