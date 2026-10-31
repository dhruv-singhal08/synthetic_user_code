"""Assemble the NL2SQL judge prompt; the prompt text itself is in ``nl2sql_prompt.yaml``.

Three deliberately narrow NL2SQL quality comparisons.

Semantic correctness checks the requested result, rules adherence checks rules and joins, and schema
faithfulness checks only explicit database table/column identifiers against the cheat sheet.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from synthetic_user.core.config import DEFAULT_ARTIFACT_MAX_CHARS
from synthetic_user.models.evaluation import (
    NL2SQL_METRICS,
    CriterionStatus,
    EvidenceStatus,
    Severity,
    Verdict,
)
from synthetic_user.models.execution import AnsweredResponse, ConversationTurn
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator.no_sql import product_expects_sql
from synthetic_user.nodes.evaluator.trace_tags import trace_catalog
from synthetic_user.prompts.contracts import choices, render_contract
from synthetic_user.prompts.templates import prompts_beside

PROMPTS = prompts_beside(__file__, "nl2sql_prompt.yaml")
SYSTEM = PROMPTS.text("system")
# Added to evidence errors so the retry knows how to fix them.
COPY_HINT = PROMPTS.text("copy_hint")
IGNORED_DUMMY_FILTER_NOTE = PROMPTS.text("ignored_dummy_filter_note")
# The NL2SQL judge sees only the answer contract of the product description.
PRODUCT_CONTRACT_FIELDS = {"name", "description", "supported", "not_supported", "expected_outputs"}


def response_text(response: AnsweredResponse) -> str:
    """Everything the judge may quote from: the text and the artifact contents."""
    parts = [response.text or ""]
    for artifact in response.artifacts:
        content = artifact.content
        parts.append(content if isinstance(content, str) else json.dumps(content))
    return "\n".join(p for p in parts if p)


def evaluation_example() -> dict[str, Any]:
    example: dict[str, Any] = PROMPTS.data(
        "evaluation_example",
        criterion_statuses=choices(CriterionStatus),
        metrics=choices(NL2SQL_METRICS),
        severities=choices(Severity),
        evidence_statuses=choices(EvidenceStatus),
        verdicts=choices(Verdict),
    )
    return example


def build_response_evaluation_prompt(
    product: ProductDescription,
    scenario: Scenario,
    question: QuestionCase,
    response: AnsweredResponse,
    grounding: tuple[ReferencePack, Reference] | None = None,
    conversation: Sequence[ConversationTurn] = (),
    *,
    artifact_max_chars: int = DEFAULT_ARTIFACT_MAX_CHARS,
) -> tuple[str, str]:
    expectations = question.expectations
    sql_product = product_expects_sql(product)
    rubric_mode = PROMPTS.text("rubric_mode_sql" if sql_product else "rubric_mode_other")
    product_contract = product.model_dump(mode="json", include=PRODUCT_CONTRACT_FIELDS)
    contract = {
        "question": question.utterance,
        "conversation": [
            {"role": turn.role.value, "kind": turn.kind.value, "message": turn.message}
            for turn in conversation
        ],
        "expected_behavior": expectations.expected_behavior,
        "success_criteria": expectations.success_criteria,
    }
    reference_context: dict[str, Any] | None = None
    pack: ReferencePack | None = None
    if grounding is not None:
        pack, reference = grounding
        reference_context = {
            "kind": pack.kind.value,
            "tables": [
                {
                    "tag": f"TABLE-{table_index:02d}",
                    "allowed_table": table.name,
                    "allowed_columns": [
                        {
                            "tag": f"COLUMN-{table_index:02d}.{column_index:02d}",
                            "name": column.name,
                        }
                        for column_index, column in enumerate(table.columns, start=1)
                    ],
                    "table_rules": [
                        {
                            "tag": f"TABLE-RULE-{table_index:02d}.{rule_index:02d}",
                            "rule": rule,
                        }
                        for rule_index, rule in enumerate(table.rules, start=1)
                    ],
                }
                for table_index, table in enumerate(pack.tables, start=1)
            ],
            "allowed_joins": [
                {
                    "tag": f"JOIN-{join_index:02d}",
                    "from": join.from_,
                    "to": join.to,
                    "verified": join.verified,
                }
                for join_index, join in enumerate(pack.joins, start=1)
            ],
            "selected_rules": [
                {"tag": f"RULE-{rule.id}", **rule.model_dump(mode="json")} for rule in pack.rules
            ],
            "documents": [
                {"name": document.name, "summary": document.summary} for document in pack.documents
            ],
            "usage_instructions": reference.instructions_text(),
        }
    user = PROMPTS.text(
        "user",
        product_contract=json.dumps(product_contract, indent=2),
        expected=json.dumps(contract, indent=2),
        reference_context=json.dumps(reference_context, indent=2),
        trace_catalog=json.dumps(
            [item.as_dict() for item in trace_catalog(question, pack)], indent=2
        ),
        response=render_response(response, artifact_max_chars=artifact_max_chars),
        rubric_mode=rubric_mode,
        contract=render_contract(evaluation_example()),
    )
    _ = scenario  # kept in the stable evaluator interface; intentionally not a judge input
    return SYSTEM, user


def render_response(response: AnsweredResponse, *, artifact_max_chars: int) -> str:
    """Render the evaluator's response view with a configurable per-artifact bound."""
    artifacts: list[dict[str, Any]] = []
    for artifact in response.artifacts:
        content = (
            artifact.content if isinstance(artifact.content, str) else json.dumps(artifact.content)
        )
        if len(content) > artifact_max_chars:
            content = content[:artifact_max_chars] + PROMPTS.text("artifact_cut_marker")
        artifacts.append({"kind": artifact.kind, "content": content})
    return json.dumps({"text": response.text, "artifacts": artifacts}, indent=2)


def missing_finding_message(metric: str, score: float) -> str:
    """Retry feedback that allows BOTH fixes. Naming only "add a finding" made a live judge lower
    a correct score to 0 just to satisfy the check."""
    return PROMPTS.text(
        "missing_finding", metric=metric, score=f"{score:g}", metric_repr=repr(metric)
    )
