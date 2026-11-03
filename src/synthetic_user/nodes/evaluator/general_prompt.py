"""Assemble the General Judge prompt; the prompt text itself is in ``general_prompt.yaml``.

Sections are included only when one of the call's metrics (or the correctness evidence) needs
them: question-specific metric expectations, the cheat sheet, authoritative tool definitions,
the SQL parser output and the exposed tool trace.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from synthetic_user.models.evaluation import CriterionStatus, FactualVerification, Verdict
from synthetic_user.models.execution import ConversationTurn, DeniedResponse, ToolCall
from synthetic_user.models.metrics import MetricDefinition
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import MetricExpectationStatus, QuestionCase
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.nodes.evaluator.judge_common import (
    EvaluableResponse,
    bounded,
    finding_example,
    metric_blocks,
    render_response,
    score_example,
)
from synthetic_user.nodes.evaluator.trace_tags import trace_catalog
from synthetic_user.prompts.contracts import choices, render_contract
from synthetic_user.prompts.templates import prompts_beside
from synthetic_user.sql.parser import SchemaComparison, SqlStructure

PROMPTS = prompts_beside(__file__, "general_prompt.yaml")
SYSTEM = PROMPTS.text("system")
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


@dataclass(frozen=True)
class SqlEvidence:
    structure: SqlStructure
    schema: SchemaComparison | None


@dataclass(frozen=True)
class GeneralJudgeInput:
    product: ProductDescription
    question: QuestionCase
    response: EvaluableResponse
    definitions: Sequence[MetricDefinition]
    conversation: Sequence[ConversationTurn]
    pack: ReferencePack | None
    reference: Reference | None
    sql: Sequence[SqlEvidence]
    tool_trace: Sequence[ToolCall] | None
    tool_definitions: bool
    needs_classification: bool
    sql_dialect: str | None
    artifact_max_chars: int


def cheat_sheet_view(pack: ReferencePack, reference: Reference | None) -> dict[str, Any]:
    """The pack as the judge sees it, with the stable trace tags of trace_tags.py."""
    return {
        "kind": pack.kind.value,
        "tables": [
            {
                "tag": f"TABLE-{ti:02d}",
                "allowed_table": table.name,
                "allowed_columns": [
                    {"tag": f"COLUMN-{ti:02d}.{ci:02d}", "name": column.name}
                    for ci, column in enumerate(table.columns, start=1)
                ],
                "table_rules": [
                    {"tag": f"TABLE-RULE-{ti:02d}.{ri:02d}", "rule": rule}
                    for ri, rule in enumerate(table.rules, start=1)
                ],
            }
            for ti, table in enumerate(pack.tables, start=1)
        ],
        "allowed_joins": [
            {"tag": f"JOIN-{ji:02d}", "from": j.from_, "to": j.to, "verified": j.verified}
            for ji, j in enumerate(pack.joins, start=1)
        ],
        "selected_rules": [
            {"tag": f"RULE-{rule.id}", **rule.model_dump(mode="json")} for rule in pack.rules
        ],
        "documents": [
            {"tag": f"DOCUMENT-{di:02d}", "name": d.name, "summary": d.summary}
            for di, d in enumerate(pack.documents, start=1)
        ],
        "usage_instructions": reference.instructions_text() if reference is not None else None,
    }


def judge_example(
    definitions: Sequence[MetricDefinition], *, needs_classification: bool
) -> dict[str, Any]:
    ids = [d.metric_id for d in definitions]
    metrics: dict[str, Any] = {}
    for definition in definitions:
        example = score_example()
        if definition.asks_factual_verification:
            example["factual_verification"] = PROMPTS.text(
                "factual_example", factual_values=choices(FactualVerification)
            )
        metrics[definition.metric_id] = example
    out: dict[str, Any] = dict(PROMPTS.data("judge_example")) if needs_classification else {}
    out.update(
        {
            "metrics": metrics,
            "criteria": [
                PROMPTS.data(
                    "criterion_example",
                    criterion_statuses=choices(CriterionStatus),
                    metric_ids=choices([*ids, "null"]),
                )
            ],
            "findings": [finding_example(ids)],
            "overall_verdict": PROMPTS.text("verdict_example", verdicts=choices(Verdict)),
        }
    )
    return out


def build_general_prompt(data: GeneralJudgeInput) -> tuple[str, str]:
    question = data.question
    definitions = list(data.definitions)
    expected = {
        "question": question.utterance,
        "conversation": [
            {"role": t.role.value, "kind": t.kind.value, "message": t.message}
            for t in data.conversation
        ],
        "expected_behavior": question.expectations.expected_behavior,
        "success_criteria": question.expectations.success_criteria,
    }
    sections: list[str] = []
    expectations = {
        d.metric_id: question.expectations.metric_expectations[d.metric_id].value
        for d in definitions
        if d.metric_id in question.expectations.metric_expectations
        and question.expectations.metric_expectations[d.metric_id].status
        is MetricExpectationStatus.GENERATED
    }
    if expectations:
        sections.append(
            PROMPTS.text(
                "section_metric_expectations", expectations=json.dumps(expectations, indent=2)
            )
        )
    if data.pack is not None:
        sections.append(
            PROMPTS.text(
                "section_cheat_sheet",
                cheat_sheet=json.dumps(cheat_sheet_view(data.pack, data.reference), indent=2),
            )
        )
    agent = data.reference.agent_context if data.reference is not None else None
    if data.tool_definitions and agent is not None and data.reference is not None:
        sections.append(
            PROMPTS.text("section_tool_definitions", tools=data.reference.render_agent_context())
        )
    if data.sql:
        dialect = (
            PROMPTS.text("dialect_known", dialect=data.sql_dialect)
            if data.sql_dialect
            else PROMPTS.text("dialect_unknown")
        )
        sql_view = [_sql_view(index, item) for index, item in enumerate(data.sql, start=1)]
        sections.append(
            PROMPTS.text(
                "section_sql",
                dialect=dialect,
                sql=bounded(json.dumps(sql_view, indent=2), data.artifact_max_chars),
            )
        )
    if data.tool_trace is not None:
        trace = [call.model_dump(mode="json") for call in data.tool_trace]
        sections.append(
            PROMPTS.text(
                "section_tool_trace",
                trace=bounded(json.dumps(trace, indent=2), data.artifact_max_chars),
            )
        )
    factual = any(d.asks_factual_verification for d in definitions)
    marked = [
        d.model_copy(update={"display_name": d.display_name + PROMPTS.text("factual_marker")})
        if d.asks_factual_verification
        else d
        for d in definitions
    ]
    user = PROMPTS.text(
        "user",
        metric_ids=", ".join(d.metric_id for d in definitions),
        classification_task=(
            PROMPTS.text("classification_task") if data.needs_classification else ""
        ),
        product=json.dumps(
            data.product.model_dump(mode="json", include=PRODUCT_CONTRACT_FIELDS), indent=2
        ),
        expected=json.dumps(expected, indent=2),
        sections="".join(sections),
        trace_catalog=json.dumps(
            [item.as_dict() for item in trace_catalog(question, data.pack)], indent=2
        ),
        response_kind=PROMPTS.text(
            "response_kind_denial"
            if isinstance(data.response, DeniedResponse)
            else "response_kind_answer"
        ),
        response=render_response(data.response, artifact_max_chars=data.artifact_max_chars),
        metric_rules=metric_blocks(marked),
        factual_rule=PROMPTS.text("factual_rule_on" if factual else "factual_rule_off"),
        contract=render_contract(
            judge_example(definitions, needs_classification=data.needs_classification)
        ),
    )
    return SYSTEM, user


def _sql_view(index: int, item: SqlEvidence) -> dict[str, Any]:
    s = item.structure
    view: dict[str, Any] = {"sql_artifact": index, "parsed": s.parsed}
    if s.diagnostics:
        view["diagnostics"] = s.diagnostics
    if not s.parsed:
        view["note"] = "parse failed: judge the original SQL; parser-dependent checks inconclusive"
        return view
    view.update(
        {
            "base_tables": s.base_tables,
            "table_aliases": s.table_aliases,
            "ctes": s.ctes,
            "base_columns": [
                f"{c.table}.{c.column}" if c.table else c.column for c in s.base_columns
            ],
            "output_aliases": s.output_aliases,
            "functions": s.functions,
            "joins": [j.model_dump(mode="json") for j in s.joins],
            "filters": s.filters,
            "projections": s.projections,
            "aggregations": s.aggregations,
            "group_by": s.group_by,
            "order_by": s.order_by,
            "limits": s.limits,
        }
    )
    if item.schema is not None:
        view["identifiers_not_in_cheat_sheet"] = item.schema.model_dump(mode="json")
    return view
