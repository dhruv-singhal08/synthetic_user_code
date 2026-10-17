"""Prompt for the reference_coverage grounded check (grounded answers no artifact plugin covers,
e.g. a chatbot scenario with a cheat sheet): does the answer cover the cheat-sheet entities and
metrics? No SQL logic is judged."""

from __future__ import annotations

import json

from synthetic_user.models.evaluation import ReferenceCoverageDraft
from synthetic_user.models.execution import AnsweredResponse
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.scenario import ReferenceCheatSheet

SYSTEM = (
    "You check whether a product's answer covers the business entities and metrics a "
    "reference plan expects. You judge coverage only, not technical correctness. You quote "
    "evidence exactly as written in the answer. You reply with a single JSON object and "
    "nothing else."
)


def coverage_items(sheet: ReferenceCheatSheet) -> list[str]:
    """Entities/metrics the answer should cover: metric names, dimensions and columns."""
    items = [m.name for m in sheet.metrics] + list(sheet.dimensions) + list(sheet.columns)
    return list(dict.fromkeys(items))


def build_reference_coverage_prompt(
    question: QuestionCase, sheet: ReferenceCheatSheet, response: AnsweredResponse
) -> tuple[str, str]:
    items = coverage_items(sheet)
    answer = {
        "text": response.text,
        "artifacts": [
            {
                "kind": a.kind,
                "content": (a.content if isinstance(a.content, str) else json.dumps(a.content))[
                    :1500
                ],
            }
            for a in response.artifacts
        ],
    }
    schema = json.dumps(ReferenceCoverageDraft.model_json_schema(), indent=2)
    user = f"""## TASK
Decide whether the answer covers the reference items below (the entities and metrics the
reference plan for this question uses).

## INPUT
Question: {question.utterance}

Reference items (copy them exactly into covered / missing):
{json.dumps(items, indent=2)}

Semantic considerations from the plan:
{json.dumps(sheet.semantic_considerations, indent=2)}

Answer:
{json.dumps(answer, indent=2)}

## RULES
- Put every reference item in exactly one of covered or missing. Items may be covered in
  business words (e.g. "spend" for amount).
- reference_coverage: pass if every item the question needs is covered, fail if a needed
  item is missing, inconclusive if you cannot tell; not_applicable only if no item applies.
- A fail needs a finding with category "reference_coverage".
- evidence must be copied exactly from the answer, or null.

## OUTPUT CONTRACT
Return one JSON object matching this JSON Schema:
{schema}

## PROHIBITED
- Judging SQL or calculation logic; inventing quotes; any text outside the JSON object.
"""
    return SYSTEM, user
