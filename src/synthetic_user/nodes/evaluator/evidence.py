"""Evidence ids: every piece of evidence a judge may cite gets a stable id assigned by code.

The General Judge sees the ids next to the evidence itself (response, artifacts, parsed SQL
clauses, catalog reconciliation, lookups, tool calls, clarification turns) and cites them in
its check-level decisions; code verifies that every cited id resolves (no invented evidence).
Ids are deterministic for a given execution, so run.log and the results show the same ids.

- ``RESPONSE``: the final response text (or the denial message);
- ``ARTIFACT:<n>``: the n-th non-SQL artifact of the response;
- ``SQL:<k>``: the k-th SQL artifact as generated; ``SQL:<k>:<CLAUSE>:<i>``: the i-th item of a
  parsed clause (FILTER, PROJECTION, AGGREGATION, GROUP_BY, ORDER_BY, LIMIT, JOIN) - sqlglot
  output, supporting evidence only;
- ``RECON:<k>``: the catalog reconciliation of SQL artifact k;
- ``LOOKUP:<n>``: the n-th exact catalog lookup (precomputed by code or requested by the judge);
- ``TOOL:<order>``: an exposed tool call; ``TURN:<n>``: a clarification turn;
- ``CHUNK:<chunk id>``: a retrieved chunk (Retrieval/Grounding Judge).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from synthetic_user.models.execution import (
    AnsweredResponse,
    ConversationTurn,
    RetrievedContext,
    ToolCall,
)
from synthetic_user.models.lookup import ArtifactReconciliation, LookupOrigin, LookupTraceEntry
from synthetic_user.nodes.evaluator.judge_common import EvaluableResponse
from synthetic_user.sql.parser import SqlStructure

RESPONSE_ID = "RESPONSE"
# Parsed clause lists of an SQL artifact that get an evidence id per item.
SQL_CLAUSES = (
    ("PROJECTION", "projections"),
    ("FILTER", "filters"),
    ("AGGREGATION", "aggregations"),
    ("GROUP_BY", "group_by"),
    ("ORDER_BY", "order_by"),
    ("LIMIT", "limits"),
)


class EvidenceProvenance(StrEnum):
    """Where a piece of evidence comes from (the report's evidence provenance)."""

    PRODUCT_RESPONSE = "product_response"
    SQLGLOT = "sqlglot"
    CATALOG_RECONCILIATION = "catalog_reconciliation"
    PRECOMPUTED_LOOKUP = "precomputed_lookup"
    JUDGE_REQUESTED_LOOKUP = "judge_requested_lookup"
    TOOL_TRACE = "tool_trace"
    CONVERSATION = "conversation"
    RETRIEVED_CONTEXT = "retrieved_context"


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    provenance: EvidenceProvenance
    label: str  # a short human-readable description (never a whole payload)


@dataclass
class EvidenceRegistry:
    items: dict[str, EvidenceItem] = field(default_factory=dict)

    def add(self, evidence_id: str, provenance: EvidenceProvenance, label: str) -> str:
        self.items[evidence_id] = EvidenceItem(evidence_id, provenance, _short(label))
        return evidence_id

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self.items

    def ids(self) -> list[str]:
        return list(self.items)

    def unknown(self, ids: Sequence[str]) -> list[str]:
        return [i for i in ids if i not in self.items]

    def by_provenance(self, ids: Sequence[str]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for evidence_id in ids:
            item = self.items.get(evidence_id)
            if item is not None:
                grouped.setdefault(item.provenance.value, []).append(evidence_id)
        return grouped


def sql_clause_id(artifact: int, clause: str, index: int) -> str:
    return f"SQL:{artifact}:{clause}:{index}"


def lookup_label(entry: LookupTraceEntry) -> str:
    request = entry.request
    target = request.table or ", ".join(request.tables) or request.column or ""
    if request.column and request.table:
        target = f"{request.table}.{request.column}"
    return f"{request.tool.value} {target} -> {entry.status}".strip()


def build_registry(
    response: EvaluableResponse,
    *,
    sql: Sequence[tuple[SqlStructure, ArtifactReconciliation | None]] = (),
    tool_trace: Sequence[ToolCall] | None = None,
    conversation: Sequence[ConversationTurn] = (),
    lookups: Sequence[LookupTraceEntry] = (),
    context: RetrievedContext | None = None,
) -> EvidenceRegistry:
    """Every citable evidence id of one evaluation, in a stable order."""
    registry = EvidenceRegistry()
    registry.add(RESPONSE_ID, EvidenceProvenance.PRODUCT_RESPONSE, "final response")
    if isinstance(response, AnsweredResponse):
        # Numbered exactly as the judge's response view numbers them (render_response): every
        # SQL artifact is SQL:<k> whether or not an SQL metric parsed it.
        others = [a for a in response.artifacts if a.kind.casefold() != "sql"]
        for n, artifact in enumerate(others, start=1):
            registry.add(f"ARTIFACT:{n}", EvidenceProvenance.PRODUCT_RESPONSE, artifact.kind)
        queries = [
            a.content
            for a in response.artifacts
            if a.kind.casefold() == "sql" and isinstance(a.content, str)
        ]
        for k, query in enumerate(queries, start=1):
            registry.add(f"SQL:{k}", EvidenceProvenance.PRODUCT_RESPONSE, query)
    for k, (structure, reconciliation) in enumerate(sql, start=1):
        registry.add(f"SQL:{k}", EvidenceProvenance.PRODUCT_RESPONSE, structure.raw_sql)
        for clause, attribute in SQL_CLAUSES:
            for i, expression in enumerate(getattr(structure, attribute), start=1):
                registry.add(sql_clause_id(k, clause, i), EvidenceProvenance.SQLGLOT, expression)
        for i, join in enumerate(structure.joins, start=1):
            registry.add(
                sql_clause_id(k, "JOIN", i),
                EvidenceProvenance.SQLGLOT,
                f"{join.kind} {join.target} {join.predicate or ''}",
            )
        if reconciliation is not None:
            registry.add(
                f"RECON:{k}",
                EvidenceProvenance.CATALOG_RECONCILIATION,
                f"catalog reconciliation of SQL artifact {k}",
            )
    for turn_number, turn in enumerate(conversation, start=1):
        registry.add(f"TURN:{turn_number}", EvidenceProvenance.CONVERSATION, turn.message)
    for call in tool_trace or ():
        registry.add(f"TOOL:{call.order}", EvidenceProvenance.TOOL_TRACE, call.name)
    for entry in lookups:
        if entry.evidence_id is None:
            continue
        provenance = (
            EvidenceProvenance.PRECOMPUTED_LOOKUP
            if entry.origin is LookupOrigin.DETERMINISTIC
            else EvidenceProvenance.JUDGE_REQUESTED_LOOKUP
        )
        registry.add(entry.evidence_id, provenance, lookup_label(entry))
    if context is not None:
        for chunk in context.chunks:
            registry.add(
                f"CHUNK:{chunk.chunk_id}", EvidenceProvenance.RETRIEVED_CONTEXT, chunk.text
            )
    return registry


def _short(text: str, limit: int = 120) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."
