"""Assemble the General Judge prompt; the prompt text itself is in ``general_prompt.yaml``.

The judge always sees the question's success criteria (with their ids) and the evaluation
contract of each listed metric (its Metric Mapping: criteria, Reference Spec requirements,
comparison basis, question-specific expectation). Other sections are included only when one of
the call's metrics needs them: the cheat sheet, authoritative tool definitions, the SQL parser
output and the exposed tool trace. Every citable piece of evidence carries its evidence id
(evidence.py) and every guidance item its guidance id (guidance.py).

When SQL is reconciled with a data catalog (D230/D231) the cheat sheet is presented as expected
evidence (not an allow-list), followed by the catalog reconciliation, the lookups code made, and
either the optional one-batch lookup offer (first call) or the results of the judge's requested
lookups (second call).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synthetic_user.models.evaluation import (
    FactualVerification,
    GoalStatus,
    JoinBasis,
    RequirementSource,
    RequirementStatus,
    Verdict,
)
from synthetic_user.models.execution import (
    ConversationTurn,
    DeniedResponse,
    ResponseClassification,
    ToolCall,
)
from synthetic_user.models.lookup import (
    ArtifactReconciliation,
    IdentifierStatus,
    JoinStatus,
    LookupTool,
    LookupTraceEntry,
)
from synthetic_user.models.metrics import MetricDefinition
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.run import EvaluationGuidance
from synthetic_user.models.scenario import CapabilityClassification
from synthetic_user.nodes.evaluator.evidence import (
    SQL_CLAUSES,
    EvidenceRegistry,
    build_registry,
    sql_clause_id,
)
from synthetic_user.nodes.evaluator.guidance import GuidanceRef, builtin_listing
from synthetic_user.nodes.evaluator.judge_common import (
    EvaluableResponse,
    bounded,
    finding_example,
    guidance_block,
    metric_blocks,
    render_response,
    score_example,
    severity_rule,
)
from synthetic_user.nodes.evaluator.trace_tags import trace_catalog
from synthetic_user.prompts.contracts import choices, render_contract
from synthetic_user.prompts.templates import prompts_beside
from synthetic_user.sql.parser import SqlStructure

PROMPTS = prompts_beside(__file__, "general_prompt.yaml")
SYSTEM = PROMPTS.text("system")
# Version of the judge's prompt templates (recorded per judge invocation in run.log): sha256 of
# the two template files, so a prompt change is visible when comparing runs.
TEMPLATE_FINGERPRINT = hashlib.sha256(
    b"".join(
        (Path(__file__).parent / name).read_bytes()
        for name in ("general_prompt.yaml", "judge_common_prompt.yaml")
    )
).hexdigest()[:16]
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
    # Catalog reconciliation; None when the run has no data catalog (no pack / documents).
    reconciliation: ArtifactReconciliation | None


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
    # Exact lookups code made before the call (reconciled runs only).
    catalog_lookups: Sequence[LookupTraceEntry] = ()
    # > 0: the judge may return up to this many lookups instead of scoring (first call only).
    max_lookup_requests: int = 0
    # Second call: the results of the judge's requested lookups.
    requested_lookups: Sequence[LookupTraceEntry] | None = None
    # Optional run-level evaluation guidance (D237); None -> no guidance section.
    guidance: EvaluationGuidance | None = None
    # Every citable guidance id of the run (user items, METRIC:<id>, POLICY:<name>).
    guidance_refs: dict[str, GuidanceRef] = field(default_factory=dict)
    # The scenario's capability (and why) for the goal-completion assessment (D243).
    capability: CapabilityClassification = CapabilityClassification.UNKNOWN
    capability_rationale: str | None = None

    @property
    def reconciled(self) -> bool:
        return any(item.reconciliation is not None for item in self.sql)

    def registry(self) -> EvidenceRegistry:
        """Every evidence id this call may cite (the requested lookups only on the second call)."""
        return build_registry(
            self.response,
            sql=[(item.structure, item.reconciliation) for item in self.sql],
            tool_trace=self.tool_trace,
            conversation=self.conversation,
            lookups=[*self.catalog_lookups, *(self.requested_lookups or ())],
        )

    def assigned(self, metric_id: str) -> list[str]:
        """The requirement ids the question's Metric Mapping assigns to a metric."""
        mapping = self.question.context.metric_mapping.get(metric_id)
        if mapping is None:
            return []
        return [*mapping.criteria, *mapping.reference_requirements]

    def undocumented_joins(self) -> list[str]:
        """``left = right`` of every undocumented join (the judge must decide each, D238)."""
        return [
            f"{join.left} = {join.right}"
            for item in self.sql
            if item.reconciliation is not None
            for join in item.reconciliation.joins
            if join.status is JoinStatus.UNDOCUMENTED
        ]


def cheat_sheet_view(
    pack: ReferencePack, reference: Reference | None, *, expected: bool = False
) -> dict[str, Any]:
    """The pack as the judge sees it, with the stable trace tags of trace_tags.py.

    ``expected``: the pack is expected evidence next to a catalog reconciliation, so its items
    are not labelled "allowed" (an allow-list wording made valid extra identifiers look wrong).
    """
    table_key, columns_key, joins_key = (
        ("table", "columns", "joins")
        if expected
        else ("allowed_table", "allowed_columns", "allowed_joins")
    )
    return {
        "kind": pack.kind.value,
        "tables": [
            {
                "tag": f"TABLE-{ti:02d}",
                table_key: table.name,
                columns_key: [
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
        joins_key: [
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
    definitions: Sequence[MetricDefinition],
    *,
    needs_classification: bool,
    unresolved_findings: bool = False,
    lookup_offer: bool = False,
    join_decisions: bool = False,
    guidance: bool = False,
) -> dict[str, Any]:
    ids = [d.metric_id for d in definitions]
    metrics: dict[str, Any] = {}
    check = PROMPTS.data(
        "check_example",
        sources=choices(RequirementSource),
        statuses=choices(RequirementStatus),
    )
    for position, definition in enumerate(definitions):
        example = score_example()
        # The full check / guidance shape once; the other metrics point to it (shorter prompt).
        example["checks"] = [check] if position == 0 else [PROMPTS.text("checks_as_above")]
        example["guidance_applied"] = (
            [PROMPTS.data("guidance_applied_example")]
            if position == 0
            else [PROMPTS.text("guidance_as_above")]
        )
        if definition.asks_factual_verification:
            example["factual_verification"] = PROMPTS.text(
                "factual_example", factual_values=choices(FactualVerification)
            )
        metrics[definition.metric_id] = example
    out: dict[str, Any] = dict(PROMPTS.data("judge_example")) if needs_classification else {}
    out.update(
        {
            "metrics": metrics,
            "findings": [_finding_example(ids, unresolved=unresolved_findings, guidance=guidance)],
            "goal_completion": PROMPTS.data("goal_example", statuses=choices(GoalStatus)),
            "overall_verdict": PROMPTS.text("verdict_example", verdicts=choices(Verdict)),
        }
    )
    if lookup_offer:
        out["evidence_requests"] = []
    if join_decisions:
        out["join_decisions"] = [PROMPTS.data("join_decision_example", bases=choices(JoinBasis))]
    return out


def _finding_example(ids: Sequence[str], *, unresolved: bool, guidance: bool) -> dict[str, Any]:
    example = finding_example(ids, guidance=guidance)
    if unresolved:
        example["unresolved"] = PROMPTS.text("unresolved_example")
    return example


def build_general_prompt(data: GeneralJudgeInput) -> tuple[str, str]:
    question = data.question
    context = question.context
    definitions = list(data.definitions)
    expected = {
        "question": question.utterance,
        "conversation": [
            {
                "evidence_id": f"TURN:{n}",
                "role": t.role.value,
                "kind": t.kind.value,
                "message": t.message,
                **({"options": list(t.options)} if t.options else {}),
            }
            for n, t in enumerate(data.conversation, start=1)
        ],
        "success_criteria": [c.model_dump(mode="json") for c in context.success_criteria],
    }
    sections: list[str] = []
    if data.pack is not None:
        sections.append(
            PROMPTS.text(
                "section_cheat_sheet_expected" if data.reconciled else "section_cheat_sheet",
                cheat_sheet=json.dumps(
                    cheat_sheet_view(data.pack, data.reference, expected=data.reconciled),
                    indent=2,
                ),
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
        sections.extend(_catalog_sections(data))
    if data.tool_trace is not None:
        trace = [
            {"evidence_id": f"TOOL:{call.order}", **call.model_dump(mode="json")}
            for call in data.tool_trace
        ]
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
        mapping=json.dumps(_mapping_view(data), indent=2),
        sections="".join(sections),
        trace_catalog=json.dumps(
            [item.as_dict() for item in trace_catalog(question, data.pack)], indent=2
        ),
        response_kind=_response_kind(data.response),
        response=render_response(
            data.response, artifact_max_chars=data.artifact_max_chars, evidence_ids=True
        ),
        metric_rules=metric_blocks(marked),
        guidance=guidance_block(data.guidance),
        builtin_guidance=json.dumps(builtin_listing(data.guidance_refs), indent=2),
        severity_rule=severity_rule(),
        goal_rule=PROMPTS.text(
            "goal_rule",
            capability=data.capability.value,
            rationale=(
                PROMPTS.text("capability_rationale", rationale=data.capability_rationale)
                if data.capability_rationale
                else ""
            ),
            capability_rule=PROMPTS.data("capability_rule")[data.capability.value],
        ),
        factual_rule=PROMPTS.text("factual_rule_on" if factual else "factual_rule_off"),
        contract=render_contract(
            judge_example(
                definitions,
                needs_classification=data.needs_classification,
                unresolved_findings=data.reconciled,
                lookup_offer=data.max_lookup_requests > 0,
                join_decisions=bool(data.undocumented_joins()),
                guidance=bool(data.guidance),
            )
        ),
    )
    return SYSTEM, user


def _response_kind(response: EvaluableResponse) -> str:
    if isinstance(response, DeniedResponse):
        kind = PROMPTS.text("response_kind_denial")
    elif response.response_classification is ResponseClassification.UNABLE_TO_ANSWER:
        kind = PROMPTS.text("response_kind_unable")
    else:
        kind = PROMPTS.text("response_kind_answer")
    if response.product_response_type is None:
        return kind
    return PROMPTS.text("response_kind_typed", kind=kind, type=response.product_response_type)


def _mapping_view(data: GeneralJudgeInput) -> dict[str, Any]:
    """Each listed metric's evaluation contract, with the Reference Spec text of every
    assigned requirement (so the judge never has to guess what an id means)."""
    context = data.question.context
    view: dict[str, Any] = {}
    for definition in data.definitions:
        mapping = context.metric_mapping.get(definition.metric_id)
        if mapping is None:
            view[definition.metric_id] = {"comparison_basis": definition.basis}
            continue
        entry: dict[str, Any] = {
            "comparison_basis": mapping.comparison_basis,
            "criteria": mapping.criteria,
            "reference_requirements": [
                {"id": r, "text": context.requirement_text(r)}
                for r in mapping.reference_requirements
            ],
        }
        if (value := context.generated(definition.metric_id)) is not None:
            entry["question_specific_expectation"] = value
        view[definition.metric_id] = entry
    return view


def _sql_view(index: int, item: SqlEvidence) -> dict[str, Any]:
    s = item.structure
    view: dict[str, Any] = {
        "sql_artifact": index,
        "evidence_id": f"SQL:{index}",
        "parsed": s.parsed,
    }
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
            "joins": [
                {"id": sql_clause_id(index, "JOIN", i), **j.model_dump(mode="json")}
                for i, j in enumerate(s.joins, start=1)
            ],
            **{
                attribute: [
                    {"id": sql_clause_id(index, clause, i), "expr": expression}
                    for i, expression in enumerate(getattr(s, attribute), start=1)
                ]
                for clause, attribute in SQL_CLAUSES
            },
        }
    )
    return view


def _catalog_sections(data: GeneralJudgeInput) -> list[str]:
    """Reconciliation, code lookups, then the lookup offer or the requested results."""
    if not data.reconciled:
        return []
    limit = data.artifact_max_chars
    views = [_reconciliation_view(i.reconciliation) for i in data.sql if i.reconciliation]
    sections = [
        PROMPTS.text(
            "section_reconciliation", reconciliation=bounded(json.dumps(views, indent=2), limit)
        )
    ]
    if data.catalog_lookups:
        sections.append(
            PROMPTS.text(
                "section_catalog_evidence", lookups=bounded(_lookups(data.catalog_lookups), limit)
            )
        )
    if data.requested_lookups is not None:
        sections.append(
            PROMPTS.text(
                "section_requested_evidence",
                results=bounded(_lookups(data.requested_lookups), limit),
            )
        )
    elif data.max_lookup_requests > 0:
        sections.append(
            PROMPTS.text(
                "section_lookup_tools",
                max_requests=data.max_lookup_requests,
                request_example=json.dumps(
                    PROMPTS.data("request_example", tools=choices(LookupTool)), indent=2
                ),
            )
        )
    return sections


def _lookups(entries: Sequence[LookupTraceEntry]) -> str:
    """Lookup results as the judge sees them, each with its evidence id: nulls and empty lists
    dropped, and a found column's table-level duplicates (table, description, provenance) shown
    once, in the column. The stored lookup trace keeps the full results."""
    views: list[dict[str, Any]] = []
    for entry in entries:
        view = {
            "evidence_id": entry.evidence_id,
            **_prune(entry.result.model_dump(mode="json", exclude_none=True)),
        }
        if isinstance(view.get("column"), dict):
            for key in ("table", "table_description", "provenance"):
                view.pop(key, None)
        views.append(view)
    return json.dumps(views, indent=2)


def _prune(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _prune(v) for k, v in value.items() if v != []}
    if isinstance(value, list):
        return [_prune(v) for v in value]
    return value


def _reconciliation_view(rec: ArtifactReconciliation) -> dict[str, Any]:
    view: dict[str, Any] = {
        "sql_artifact": rec.artifact,
        "evidence_id": f"RECON:{rec.artifact}",
        "parsed": rec.parsed,
    }
    if not rec.parsed:
        view["note"] = (
            "parse failed: identifiers were not reconciled; judge the original SQL and treat "
            "identifier existence as inconclusive unless the SQL text itself settles it"
        )
        return view
    for status in (IdentifierStatus.VALID_IN_PACK, IdentifierStatus.VALID_EXTRA):
        view[status.value] = [i.canonical or i.identifier for i in rec.with_status(status)]
    for status in (IdentifierStatus.INVALID, IdentifierStatus.UNRESOLVED):
        view[status.value] = [
            {"identifier": i.identifier, "reason": i.reason} for i in rec.with_status(status)
        ]
    view["cheat_sheet_items_not_used"] = rec.pack_items_not_used
    view["joins"] = [j.model_dump(mode="json", exclude_none=True) for j in rec.joins]
    view["rules_not_in_cheat_sheet"] = rec.rules_not_in_pack
    return view
