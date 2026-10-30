"""Prompt for configurable, observable Agent response evaluation."""

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
from synthetic_user.prompts.contracts import choices, render_contract
from synthetic_user.prompts.response_evaluation import render_response

SYSTEM = (
    "You evaluate an Agent response using only the requested observable comparisons. Keep metric "
    "boundaries strict, quote response evidence exactly, and never invent hidden tool calls, "
    "retrieved context, tool results, or ground-truth facts. Reply with one JSON object only."
)

METRIC_RULES = {
    MetricName.RELEVANCE: (
        "compare the user's request with the final answer. Score 1 when the response directly "
        "addresses the request or appropriately explains why it cannot; 0.5 when partly "
        "responsive; 0 when off-topic."
    ),
    MetricName.COMPLETION: (
        "compare expected_behavior, every success criterion, and prohibited_behaviors with the "
        "delivered answer and actions. Score 1 only when all requested outputs/actions are "
        "complete and no prohibited behavior occurs; 0.5 when some are complete; 0 when the "
        "central task is not completed."
    ),
    MetricName.TOOL_SELECTION: (
        "compare exposed tool calls with expected_tools, expected_data_sources, the question, "
        "and product capabilities. "
        "Check tool/operation, data source, arguments, filters, grouping, and time boundaries. "
        "Judge the selected action, not whether a correctly targeted tool later failed."
    ),
    MetricName.FAITHFULNESS: (
        "compare claims in the final answer with expected_context, expected_data_sources, and "
        "the exposed retrieved context or tool results. "
        "Score 1 when every material claim is supported, 0.5 when support is mixed, and 0 when "
        "material claims contradict or are unsupported by that evidence."
    ),
    MetricName.CONTEXT_PRECISION: (
        "compare exposed retrieved context with the question, expected_context, and "
        "expected_data_sources. Score 1 when the retrieved context "
        "is focused and relevant, 0.5 when it mixes relevant and irrelevant material, and 0 when "
        "it is predominantly irrelevant."
    ),
    MetricName.CONTEXT_RECALL: (
        "compare exposed retrieved context with expected_context, expected_data_sources, any "
        "reference scope, and the requested task. "
        "Score 1 when it covers all necessary expected information, 0.5 when important expected "
        "information is missing, and 0 when it misses the central required context."
    ),
}


def evaluation_example(metrics: Sequence[MetricName]) -> dict[str, Any]:
    def score() -> dict[str, Any]:
        return {
            "score": "<0 | 0.5 | 1 | null>",
            "rationale": "<why>",
            "evidence": "<one short exact fragment from the product response, or null>",
            "what_worked": ["<observable success; empty if none>"],
            "what_didnt": ["<observable gap; empty if none>"],
        }

    return {
        **{metric.value: score() for metric in metrics},
        "criteria": [
            {
                "criterion": "<a success criterion, copied exactly>",
                "status": choices(CriterionStatus),
                "rationale": "<why>",
                "evidence": "<one short exact fragment from the response, or null>",
            }
        ],
        "findings": [
            {
                "category": choices(metrics),
                "description": "<the observable problem>",
                "expected_behavior": "<what should have happened>",
                "observed_behavior": "<what happened>",
                "evidence": "<one short exact fragment from the response, or null>",
                "severity": choices(Severity),
                "confidence": "<0-1, or null>",
                "evidence_status": choices(EvidenceStatus),
                "matches_known_gap": "<true | false>",
                "related_tags": [],
            }
        ],
        "overall_verdict": choices(Verdict),
    }


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
    product_contract = product.model_dump(
        mode="json",
        include={
            "name",
            "description",
            "data_sources",
            "supported",
            "not_supported",
            "known_limitations",
            "user_visible_capabilities",
            "expected_outputs",
        },
    )
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
    metric_rules = "\n".join(f"- {metric.value}: {METRIC_RULES[metric]}" for metric in metrics)
    user = f"""## TASK
Evaluate this Agent response using exactly the applicable metrics listed below.

## INPUT
Product contract:
{json.dumps(product_contract, indent=2)}

Question and expected behaviour:
{json.dumps(expected, indent=2)}

Expected reference context, when supplied:
{json.dumps(expected_context, indent=2)}

Product response, including exposed structured artifacts:
{render_response(response, artifact_max_chars=artifact_max_chars)}

## APPLICABLE METRICS
{metric_rules}

## RULES
- Evaluate only the listed metrics. Optional metrics with missing telemetry were removed by code;
  do not infer hidden tool calls, retrieved context, or tool results.
- Include every success criterion exactly once and judge it against the final response/actions.
- Treat expected_journey, expected_time_period, expected_tools, expected_data_sources,
  expected_context, prohibited_behaviors, and golden_set_criteria as the
  SME-authored evaluation contract. Do not drop or reinterpret an uploaded criterion. Use
  telemetry-dependent criteria only for metrics whose required telemetry is exposed.
- reference_context contains canonical definitions copied from the separate reference upload.
  Use it to interpret tool/source/workflow names; do not require the final answer to repeat it.
- metric_mapping is authoritative: it states exactly which applicable metrics consume each part
  of the expected-behaviour contract. An empty mapped list means its metrics are disabled or N/A.
- Keep metrics independent. A score below 1 requires a finding in that metric category.
- Evidence is one short contiguous exact quote from the response text or artifacts, or null.
- related_tags must be []; code links criteria and determines known-gap matches.
- overall_verdict is advisory; code computes the final verdict from applicable metric results.

## OUTPUT CONTRACT
{render_contract(evaluation_example(metrics))}

## PROHIBITED
- SQL-specific syntax or schema grading; the NL2SQL profile owns those checks.
- Penalizing a correctly selected tool solely because the tool returned an error.
- Any metric field not listed under APPLICABLE METRICS.
- Any text outside the JSON object.
"""
    _ = scenario
    return SYSTEM, user
