"""Node 4 prompts (text in ``prompt.yaml``).

- clarification policy ``ask``: the synthetic user answers a product clarification in the
  persona's voice (``build_clarification_reply_prompt``);
- clarification policy ``assume``: the hidden instruction appended to the question sent to the
  product (``SINGLE_TURN_ASSUMPTION_INSTRUCTION``).
"""

from __future__ import annotations

import json

from synthetic_user.models.execution import ConversationTurn
from synthetic_user.models.persona import Persona
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import ReferencePack
from synthetic_user.models.scenario import Scenario
from synthetic_user.prompts.contracts import render_contract
from synthetic_user.prompts.templates import prompts_beside

PROMPTS = prompts_beside(__file__)
SYSTEM = PROMPTS.text("clarification_system")
SINGLE_TURN_ASSUMPTION_INSTRUCTION = PROMPTS.text("assumption_instruction")


def build_clarification_reply_prompt(
    persona: Persona,
    scenario: Scenario,
    question: QuestionCase,
    turns: list[ConversationTurn],
    pack: ReferencePack | None,
) -> tuple[str, str]:
    conversation = [
        {
            "role": turn.role.value,
            "kind": turn.kind.value,
            "message": turn.message,
            **({"options": list(turn.options)} if turn.options else {}),
        }
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
        "success_criteria": question.criteria_texts,
        "conversation": conversation,
        "available_scope": pack.render() if pack is not None else None,
    }
    user = PROMPTS.text(
        "clarification_user",
        context=json.dumps(context, indent=2),
        contract=render_contract(PROMPTS.data("clarification_example")),
    )
    return SYSTEM, user
