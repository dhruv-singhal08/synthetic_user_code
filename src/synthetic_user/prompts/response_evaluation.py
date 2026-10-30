"""Prompt for three deliberately narrow NL2SQL quality comparisons.

Semantic correctness checks the requested result, rules adherence checks rules and joins, and schema
faithfulness checks only explicit database table/column identifiers against the cheat sheet.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from synthetic_user.core.config import DEFAULT_ARTIFACT_MAX_CHARS
from synthetic_user.evaluation_trace import trace_catalog
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
from synthetic_user.no_sql import product_expects_sql
from synthetic_user.prompts.contracts import choices, render_contract

SYSTEM = (
    "You perform three narrow NL2SQL comparisons on a product response. Use only the "
    "input pair defined for each metric. Do not infer factual correctness or use a schema catalog "
    "as answer data. Quote evidence exactly as written in the response. Reply with a single JSON "
    "object and nothing else."
)


def response_text(response: AnsweredResponse) -> str:
    """Everything the judge may quote from: the text and the artifact contents."""
    parts = [response.text or ""]
    for artifact in response.artifacts:
        content = artifact.content
        parts.append(content if isinstance(content, str) else json.dumps(content))
    return "\n".join(p for p in parts if p)


def evaluation_example() -> dict[str, Any]:
    score = {
        "score": "<0 | 0.5 | 1 | null>",
        "rationale": "<why>",
        "evidence": "<one short fragment copied from the answer, or null>",
        "what_worked": ["<concise observable success; empty if none>"],
        "what_didnt": ["<concise observable gap; empty if none>"],
    }
    return {
        "semantic_correctness": score,
        "rules_adherence": score,
        "schema_faithfulness": score,
        "criteria": [
            {
                "criterion": "<a success criterion, copied exactly>",
                "status": choices(CriterionStatus),
                "rationale": "<why>",
                "evidence": "<one short fragment copied from the answer, or null>",
            }
        ],
        "findings": [
            {
                "category": choices(NL2SQL_METRICS),
                "description": "<the problem>",
                "expected_behavior": "<what should have happened>",
                "observed_behavior": "<what happened>",
                "evidence": "<one short fragment copied from the answer, or null>",
                "severity": choices(Severity),
                "confidence": "<0-1, or null>",
                "evidence_status": choices(EvidenceStatus),
                "matches_known_gap": "<true | false>",
                "related_tags": [],
            }
        ],
        "overall_verdict": choices(Verdict),
    }


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
    rubric_mode = "NL2SQL" if sql_product else "unsupported non-NL2SQL profile"
    product_contract = product.model_dump(
        mode="json",
        include={
            "name",
            "description",
            "supported",
            "not_supported",
            "expected_outputs",
        },
    )
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
    user = f"""## TASK
Evaluate the product's answer using exactly three metrics and report observable problems.

## INPUT
Product answer contract (defines what form a successful answer takes):
{json.dumps(product_contract, indent=2)}

What was asked and what is expected:
{json.dumps(contract, indent=2)}

Allowed schema names, rules, and joins from this question's cheat sheet
(null means schema faithfulness and rules adherence are not applicable):
{json.dumps(reference_context, indent=2)}

Traceability catalog (stable tags used by the report to connect expectations and reference items):
{json.dumps([item.as_dict() for item in trace_catalog(question, pack)], indent=2)}

Product answer:
{render_response(response, artifact_max_chars=artifact_max_chars)}

## SCORING
- rubric mode: {rubric_mode}.
- If the product timed out, refused, or returned no SQL query: Semantic Correctness = 0 with a
  Semantic Correctness finding; Rules Adherence = null; Schema Faithfulness = null. Never award
  schema faithfulness merely because no invalid identifier is visible.
- semantic_correctness: compare the clarified request, expected_behavior, and behavioural success
  criteria with the final SQL under the Product answer contract. A behavioural criterion describes
  the requested result: projections, calculations, aggregations, filters, grouping, ordering,
  limits, or supported response handling. Do not use table/column allow-list compliance,
  table/business rules, privacy/currency rules, or documented-join requirements in Semantic
  Correctness; those have their own metrics below. 1 = all requested behaviour is visibly
  delivered; 0.5 = only part is delivered;
  0 = none is delivered or no SQL query is generated; null only if the SQL exists but the
  requested behaviour genuinely cannot be assessed. For
  an SQL-generating product, statically inspect whether the query is plausibly valid and contains
  the projections, aggregations, joins, filters, grouping, ordering and limits needed to produce
  the requested result. A ranking needs a limit only when the clarified request or expected
  behaviour specifies one. Do not execute the query or verify data values. Do not require returned
  rows or values.
  Judge date semantics from the actual filter field and boundaries: filtering a documented
  billing-month field with calculated calendar-month boundaries is a billing-month filter. Do not
  require the SQL or prose to repeat the words "billing month". A current-year equality filter is
  a valid semantic implementation of year-to-date for historical transaction data unless the
  contract explicitly requires a different boundary or the response demonstrably includes future
  periods. Judge the result produced, not whether the answer uses your preferred SQL shape.
- rules_adherence: check direct, material compliance with applicable table rules, selected rules,
  and verified
  documented joins in the question's cheat sheet. Include a success criterion here when it is
  itself a rule or join requirement. Apply a rule or join only when the question or SQL concerns
  it; irrelevant or theoretical rules must not lower the score. 1 = all applicable rules and joins
  are followed; 0.5 = a mix of compliance and supported conflicts; 0 = applicable rules/joins are
  materially violated; null when no rules/joins apply or no SQL was generated. Do not use unknown
  table/column names or missing requested result behaviour in this metric.
  A sensitive identifier used only as an input to COUNT or COUNT(DISTINCT) is not exposed in the
  result. Do not penalize aggregate-only use under a rule that prohibits exposing identifiers;
  penalize projection, grouping, display, or return of raw identifier values instead.
- schema_faithfulness: compare only explicit database table/column identifiers with the allowed
  table and column names. Include a success criterion here only when it explicitly requires an
  allowed schema identifier. Do not use table rules, selected rules, joins, expected result
  behaviour, or general response quality in schema faithfulness. Compare
  names case-insensitively after removing quoting and table qualification. Values such as
  source=csat, group_by=month, metric=avg_score, filters and ordinary prose are not schema
  identifiers. 1 = every explicit table and column identifier is allowed; 0.5 = a mix of allowed
  and unallowed identifiers; 0 = none of the explicit identifiers is allowed; null when the cheat
  sheet is absent, no SQL was generated, or the SQL exposes nothing that can be compared. Missing
  requested behaviour does not reduce schema faithfulness. SQL output aliases, CTE names and
  function names are not database table/column
  identifiers and must never be checked against the allowed schema list.
- criteria: one entry per success criterion (copy each criterion text exactly), status met | partly
  | not_met | unverifiable. Judge semantic outcomes, not exact wording or a preferred
  implementation. Each criterion is displayed for traceability, but it affects only its matching
  metric: requested-result behaviour -> semantic_correctness; rule/join compliance ->
  rules_adherence; allowed table/column names -> schema_faithfulness.
  Treat a criterion as met when the response uses a reasonable equivalent that produces the same
  requested result; do not fail theoretical edge cases unless they are stated in the contract or
  materially change the result. Never guess whether reported facts or numbers are true.
  UNVERIFIABLE is neutral: it must not by itself lower a metric, appear in what_didnt, or create a
  finding.

## SCORE BREAKDOWN AND FINDINGS
- For every metric, what_worked lists the distinct observable parts that satisfied the comparison;
  use [] when none did. what_didnt lists the distinct observable gaps that kept the score below 1;
  use [] for a perfect score and for a comparison that is not applicable. Keep each item concise,
  specific and understandable without reading the rationale. For a 0.5 score, report both sides.
- A finding explains a problem in detail; category is semantic_correctness, rules_adherence or
  schema_faithfulness.
- For every finding, related_tags contains only exact tags from the Traceability catalog that the
  finding directly affects. Link a failed success criterion, table/column mismatch, material rule
  violation, missing/invalid join, or contradicted document as applicable. Do not attach nearby or
  merely topically similar reference items. Use [] when no catalog item is directly affected. For
  an invented or unlisted schema identifier, include SCHEMA-UNLISTED.
- Keep metric boundaries strict. An unknown, undocumented or disallowed table/column identifier is
  a schema_faithfulness issue only. A rule or join conflict is a rules_adherence issue only.
  Missing requested behaviour is a semantic_correctness issue only. An extra filter may affect
  semantic correctness when
  its observable effect changes requested scope and may separately affect schema faithfulness if
  its identifier is not allowed.
- Prefer the ordinary business interpretation when multiple implementations are reasonable. Do not
  lower a score for stylistic differences, redundant syntax, a different but equivalent SQL shape,
  or a hypothetical edge case that is not evidenced. Use null/inconclusive rather than inventing a
  failure when semantic equivalence genuinely cannot be determined.
- Required for every metric scored 0 or 0.5. No finding is needed for a null score.
- evidence (in scores, criteria and findings) is ONE short contiguous fragment copied
  character for character from the product answer (a few words is enough), or null - never
  reworded, never several pieces joined with "...". evidence_status: supported (the quote
  shows it), weak, or unsupported.
- matches_known_gap: always false here; code compares findings with known product gaps later.
- overall_verdict: your own opinion; recorded, not decisive.

## OUTPUT CONTRACT
{render_contract(evaluation_example())}

## PROHIBITED
- Using table/column allow-list problems for semantic correctness or rules adherence; using rules
  and joins for semantic correctness or schema faithfulness; using missing requested behaviour for
  the other two metrics.
- Inventing quotes, factual correctness, hallucination labels, table names or column names.
- Any text outside the JSON object.
"""
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
            content = content[:artifact_max_chars] + "\n[... cut: artifact too long]"
        artifacts.append({"kind": artifact.kind, "content": content})
    return json.dumps({"text": response.text, "artifacts": artifacts}, indent=2)
