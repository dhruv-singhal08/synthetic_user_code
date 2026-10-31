"""Assemble the Agent judge prompt; the prompt text itself is in ``agent_prompt.yaml``.

Configurable, observable Agent response evaluation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from synthetic_user.core.config import DEFAULT_ARTIFACT_MAX_CHARS
from synthetic_user.models.evaluation import (
    CriterionStatus,
    EvidenceStatus,
    MetricName,
    Severity,
    Verdict,
)
from synthetic_user.models.evaluation_profile import AGENT_EXPECTATION_METRIC_MAP
from synthetic_user.models.execution import AnsweredResponse, ConversationTurn
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator.nl2sql_prompt import render_response
from synthetic_user.prompts.contracts import choices, render_contract
from synthetic_user.prompts.templates import prompts_beside

PROMPTS = prompts_beside(__file__, "agent_prompt.yaml")
SYSTEM = PROMPTS.text("system")
METRIC_RULES: dict[MetricName, str] = {
    MetricName(name): rule for name, rule in PROMPTS.data("metric_rules").items()
}
# The Agent judge sees the full product contract.
PRODUCT_CONTRACT_FIELDS = {
    "name",
    "description",
    "data_sources",
    "supported",
    "not_supported",
    "known_limitations",
    "user_visible_capabilities",
    "expected_outputs",
}


def evaluation_example(metrics: Sequence[MetricName]) -> dict[str, Any]:
    rest: dict[str, Any] = PROMPTS.data(
        "evaluation_example",
        criterion_statuses=choices(CriterionStatus),
        metrics=choices(metrics),
        severities=choices(Severity),
        evidence_statuses=choices(EvidenceStatus),
        verdicts=choices(Verdict),
    )
    return {**{metric.value: PROMPTS.data("score_example") for metric in metrics}, **rest}


def build_agent_evaluation_prompt(
    product: ProductDescription,
    scenario: Scenario,
    question: QuestionCase,
    response: AnsweredResponse,
    metrics: Sequence[MetricName],
    grounding: tuple[ReferencePack, Reference] | None = None,
    conversation: Sequence[ConversationTurn] = (),
    *,
    artifact_max_chars: int = DEFAULT_ARTIFACT_MAX_CHARS,
) -> tuple[str, str]:
    pack, reference = grounding if grounding is not None else (None, None)
    product_contract = product.model_dump(mode="json", include=PRODUCT_CONTRACT_FIELDS)
    expected = {
        "question": question.utterance,
        "conversation": [
            {"role": turn.role.value, "kind": turn.kind.value, "message": turn.message}
            for turn in conversation
        ],
        "expected_behavior": question.expectations.expected_behavior,
        "success_criteria": question.expectations.success_criteria,
        "expected_tools": question.expectations.expected_tools,
        "expected_journey": question.expectations.expected_journey,
        "expected_time_period": question.expectations.expected_time_period,
        "expected_data_sources": question.expectations.expected_data_sources,
        "expected_context": question.expectations.expected_context,
        "prohibited_behaviors": question.expectations.prohibited_behaviors,
        "reference_context": question.expectations.reference_context,
        "golden_set_criteria": question.expectations.golden_set_criteria,
        "metric_mapping": {
            field: [metric.value for metric in mapped if MetricName(metric.value) in metrics]
            for field, mapped in AGENT_EXPECTATION_METRIC_MAP.items()
        },
    }
    expected_context = None
    if pack is not None and reference is not None:
        expected_context = {
            "kind": pack.kind.value,
            "tables_or_sources": [table.model_dump(mode="json") for table in pack.tables],
            "rules": [rule.model_dump(mode="json") for rule in pack.rules],
            "documents": [document.model_dump(mode="json") for document in pack.documents],
            "usage_instructions": reference.instructions_text(),
        }
    metric_rules = "\n".join(
        PROMPTS.text("metric_rule", metric=metric.value, rule=METRIC_RULES[metric])
        for metric in metrics
    )
    user = PROMPTS.text(
        "user",
        product_contract=json.dumps(product_contract, indent=2),
        expected=json.dumps(expected, indent=2),
        reference_context=json.dumps(expected_context, indent=2),
        response=render_response(response, artifact_max_chars=artifact_max_chars),
        metric_rules=metric_rules,
        contract=render_contract(evaluation_example(metrics)),
    )
    _ = scenario
    return SYSTEM, user
