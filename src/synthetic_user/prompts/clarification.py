"""Prompt for judging a clarification question returned by the product."""

from __future__ import annotations

import json

from synthetic_user.models.evaluation import ClarificationEvaluation
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import ReferenceSnapshot
from synthetic_user.models.scenario import Scenario
from synthetic_user.prompts.evaluation_context import evaluation_context

SYSTEM = (
    "You judge whether an NL2SQL product was right to ask the user a clarification question "
    "instead of answering. You reply with a single JSON object and nothing else."
)


def build_clarification_prompt(
    question: QuestionCase,
    scenario: Scenario,
    snapshot: ReferenceSnapshot,
    product: ProductDescription,
    clarification_message: str,
) -> tuple[str, str]:
    schema = json.dumps(ClarificationEvaluation.model_json_schema(), indent=2)
    user = f"""## TASK
The product replied to the user's question with a clarification request instead of SQL.
Judge its relevance and necessity.

## INPUT
{evaluation_context(question, scenario, snapshot, product)}

Clarification returned by the product:
"{clarification_message}"

## CONSTRAINTS
- relevance: relevant (about a real ambiguity in this question) | irrelevant | unclear.
- necessity: necessary (cannot answer reliably without it) | reasonable (could assume, but
  asking is sensible) | unnecessary (the question and metadata already settle it) | unclear.
- asks_for_given_info: true if it asks for something the user's question already states.
- Consider the question's expectations.clarification (required | conditional | not_expected)
  and interpretation_notes, but judge the actual clarification on its merits.
- reason: one or two sentences citing the question or metadata.

## OUTPUT CONTRACT
Return one JSON object matching this JSON Schema:
{schema}

## PROHIBITED
- Judging anything other than the clarification request.
- Any text outside the JSON object.

## MISSING INFO
If you cannot tell whether the clarification is needed, use "unclear".
"""
    return SYSTEM, user
