"""Shared INPUT block for the SQL and clarification evaluation prompts."""

from __future__ import annotations

import json

from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import ReferenceSnapshot
from synthetic_user.models.scenario import Scenario


def evaluation_context(
    question: QuestionCase,
    scenario: Scenario,
    snapshot: ReferenceSnapshot,
    product: ProductDescription,
) -> str:
    question_json = question.model_dump(
        mode="json",
        include={"utterance", "intended_goal", "intent", "expectations", "interpretation_notes"},
    )
    scenario_json = scenario.model_dump(
        mode="json",
        include={"business_goal", "title", "task_type", "capability", "success_criteria"},
    )
    sheet = scenario.reference_cheat_sheet
    snapshot_json = snapshot.model_dump(
        mode="json", include={"tables", "relationships", "missing_references"}
    )
    product_json = product.model_dump(
        mode="json", include={"name", "supported", "not_supported_yet", "known_gaps"}
    )
    return f"""Question asked by the user:
{json.dumps(question_json, indent=2)}

Scenario:
{json.dumps(scenario_json, indent=2)}

Reference cheat sheet (a plausible path - a REFERENCE, not a required structure):
{json.dumps(sheet.model_dump(mode="json") if sheet else None, indent=2)}

Reference snapshot (AUTHORITATIVE schema metadata; table business_rules and calculated_fields
are authoritative domain rules the SQL must follow, e.g. default currency or count definitions):
{json.dumps(snapshot_json, indent=2)}

Product under test:
{json.dumps(product_json, indent=2)}"""
