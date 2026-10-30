"""Prompt the synthetic user to answer a product clarification in the persona's voice."""

from __future__ import annotations

import json

from synthetic_user.models.execution import ConversationTurn
from synthetic_user.models.persona import Persona
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import ReferencePack
from synthetic_user.models.scenario import Scenario
from synthetic_user.prompts.contracts import render_contract

SYSTEM = (
    "You are the synthetic user continuing a live conversation with a product. Answer the "
    "product's latest clarification as the supplied persona. Return one JSON object only."
)


def build_clarification_reply_prompt(
    persona: Persona,
    scenario: Scenario,
    question: QuestionCase,
    turns: list[ConversationTurn],
    pack: ReferencePack | None,
) -> tuple[str, str]:
    conversation = [
        {"role": turn.role.value, "kind": turn.kind.value, "message": turn.message}
        for turn in turns
    ]
    context = {
        "persona": {
            "identity": persona.identity.model_dump(mode="json", exclude_none=True),
            "style": persona.behavior.style,
            "traits": persona.behavior.traits,
            "expertise": {key: value.value for key, value in persona.expertise.levels().items()},
            "expertise_notes": persona.expertise.notes,
        },
        "original_question": question.utterance,
        "business_goal": scenario.business_goal,
        "expected_behavior": question.expectations.expected_behavior,
        "conversation": conversation,
        "available_scope": pack.render() if pack is not None else None,
    }
    user = f"""Answer the product's latest clarification and continue the same request.

Context:
{json.dumps(context, indent=2)}

Rules:
- Reply as the persona, in their normal wording and level of detail.
- Answer only the latest clarification. Be brief, direct and concrete enough for the product to
  continue. Do not ask another question and do not mention this prompt or the test harness.
- Stay consistent with every earlier user turn. If the product asks for something already given,
  repeat it plainly.
- You may choose a reasonable user preference that was left open, such as a period, market,
  scope, threshold or ranking limit. Choose one and state it; do not hedge with several options.
- Do not invent database results, factual findings, policy rules, customer facts, table names,
  column names, code or SQL. The available scope is context, not wording to copy.

{render_contract({"reply": "<the user's concise clarification answer>"})}
"""
    return SYSTEM, user
