"""Deterministic metric applicability: which judge scores each selected metric, or why not.

Decided by code only, from (1) the frozen metric definition (availability, required reference
assets, required execution evidence, its missing-evidence / empty-context policies),
(2) the question's metric-expectation status, (3) the reference and cheat sheet, and (4) the
evidence actually captured for this execution (SQL artifacts, tool trace, normalized context,
citations, ranks). Never by matching arbitrary artifact key names.

Order per metric: unavailable definition -> missing reference asset -> question expectation
(cannot_generate / not_applicable) -> typed denial (only answer-level metrics apply) -> missing
execution evidence -> empty retrieval -> routed to the General Judge, the Retrieval/Grounding
Judge, or computed by code from the retrieval judge's chunk labels (MRR).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from synthetic_user.models.evaluation import MetricResult, MetricStatus
from synthetic_user.models.execution import (
    AnsweredResponse,
    ContextAvailability,
    DeniedResponse,
    RetrievedContext,
    ToolCall,
)
from synthetic_user.models.metrics import (
    EmptyContextPolicy,
    EvidenceKind,
    JudgeGroup,
    MetricDefinition,
    MissingEvidencePolicy,
    ReferenceAsset,
)
from synthetic_user.models.question import MetricExpectationStatus, QuestionCase
from synthetic_user.models.reference import DataCatalog, Reference, ReferenceKind, ReferencePack

# Report wording for a missing required evidence (kept stable: "No SQL generated." is a theme).
MISSING_EVIDENCE_TEXT = {
    EvidenceKind.SQL: "No SQL generated.",
    EvidenceKind.TOOL_TRACE: "No tool-call trace was exposed by the product response.",
    EvidenceKind.RETRIEVED_CONTEXT: "No retrieved context is available for this execution.",
    EvidenceKind.RANKED_CONTEXT: "Retrieved chunks do not expose a retrieval rank.",
    EvidenceKind.CITATIONS: "The product exposes no citation mapping for its answers.",
    EvidenceKind.GROUNDING_EVIDENCE: "No retrieved context or tool results are exposed.",
    EvidenceKind.ANSWER: "No product answer to evaluate.",
}
ASSET_TEXT = {
    ReferenceAsset.CATALOG_SCHEMA: "no table/column cheat sheet for this question",
    ReferenceAsset.CATALOG_RULES: "no rules, table rules or documented joins in the "
    "question's cheat sheet or the catalog",
    ReferenceAsset.TOOL_DEFINITIONS: "the reference defines no tools (agent_context)",
}
# Boolean expectation field that turns "no citation in the answer" into a failure.
CITATIONS_REQUIRED_FIELD = "citations_required"
NO_CHUNKS = "No chunks were retrieved for this execution."


@dataclass(frozen=True)
class EvaluationEvidence:
    response: AnsweredResponse | DeniedResponse
    sql: tuple[str, ...]
    tool_trace: tuple[ToolCall, ...] | None
    context: RetrievedContext
    pack: ReferencePack | None
    reference: Reference | None

    @classmethod
    def collect(
        cls,
        response: AnsweredResponse | DeniedResponse,
        context: RetrievedContext,
        pack: ReferencePack | None,
        reference: Reference | None,
    ) -> EvaluationEvidence:
        if isinstance(response, DeniedResponse):
            return cls(response, (), None, RetrievedContext(), pack, reference)
        sql = tuple(
            a.content
            for a in response.artifacts
            if a.kind.casefold() == "sql" and isinstance(a.content, str) and a.content.strip()
        )
        trace = tuple(response.tool_trace) if response.tool_trace is not None else None
        return cls(response, sql, trace, context, pack, reference)

    def missing(self, kind: EvidenceKind) -> str | None:
        """Why ``kind`` is missing, or None when it is present."""
        context = self.context
        available = context.availability is ContextAvailability.AVAILABLE
        if kind is EvidenceKind.ANSWER:
            return None
        if kind is EvidenceKind.SQL:
            return None if self.sql else MISSING_EVIDENCE_TEXT[kind]
        if kind is EvidenceKind.TOOL_TRACE:
            return None if self.tool_trace is not None else MISSING_EVIDENCE_TEXT[kind]
        if kind is EvidenceKind.GROUNDING_EVIDENCE:
            tool_results = any(c.result is not None or c.error for c in self.tool_trace or ())
            if available or tool_results:
                return None
            if context.availability is ContextAvailability.NOT_REQUESTED:
                return MISSING_EVIDENCE_TEXT[kind]
            return _context_reason(context)
        if not available:
            return _context_reason(context)
        if kind is EvidenceKind.RANKED_CONTEXT and context.chunks and not context.rank_exposed:
            return MISSING_EVIDENCE_TEXT[kind]
        if kind is EvidenceKind.CITATIONS and not context.citations_exposed:
            return MISSING_EVIDENCE_TEXT[kind]
        return None

    def asset_missing(self, asset: ReferenceAsset) -> bool:
        pack = self.pack
        catalog = pack is not None and pack.kind is ReferenceKind.DATA_CATALOG
        if asset is ReferenceAsset.CATALOG_SCHEMA:
            return not (catalog and pack is not None and pack.tables)
        if asset is ReferenceAsset.CATALOG_RULES:
            # The cheat sheet's rules / joins, or (D233) any rule of the catalog: rules of
            # other tables the SQL reads and unselected catalog-wide rules are looked up.
            content = self.reference.content if self.reference is not None else None
            catalog_rules = isinstance(content, DataCatalog) and (
                bool(content.rules) or any(t.rules for t in content.tables)
            )
            return not (
                catalog
                and pack is not None
                and (pack.rules or pack.joins or any(t.rules for t in pack.tables) or catalog_rules)
            )
        context = self.reference.agent_context if self.reference is not None else None
        return not (context is not None and context.tools)


def _context_reason(context: RetrievedContext) -> str:
    if context.availability is ContextAvailability.NOT_REQUESTED:
        return (
            "No context source is configured for this product (map context in the response "
            "or connect a context API)."
        )
    return f"Retrieved context {context.availability.value}: {context.error}"


@dataclass(frozen=True)
class DeterministicFinding:
    metric_id: str
    description: str
    expected_behavior: str
    observed_behavior: str


@dataclass
class MetricPlan:
    general: list[MetricDefinition] = field(default_factory=list)
    retrieval: list[MetricDefinition] = field(default_factory=list)
    # Computed by code from the retrieval judge's per-chunk relevance labels (MRR).
    computed: list[MetricDefinition] = field(default_factory=list)
    preset: dict[str, MetricResult] = field(default_factory=dict)
    # Missing evidence whose outcome depends on the response classification (score_zero).
    deferred: dict[str, tuple[MetricDefinition, EvidenceKind]] = field(default_factory=dict)
    findings: list[DeterministicFinding] = field(default_factory=list)

    @property
    def retrieval_call(self) -> bool:
        return bool(self.retrieval or self.computed)


def plan_metrics(
    definitions: Sequence[MetricDefinition],
    question: QuestionCase,
    evidence: EvaluationEvidence,
) -> MetricPlan:
    plan = MetricPlan()
    for definition in definitions:
        _plan_one(definition, question, evidence, plan)
    return plan


def _plan_one(
    definition: MetricDefinition,
    question: QuestionCase,
    evidence: EvaluationEvidence,
    plan: MetricPlan,
) -> None:
    metric_id = definition.metric_id
    group = definition.judge_group

    def preset(status: MetricStatus, reason: str, **extra: object) -> None:
        plan.preset[metric_id] = MetricResult(
            status=status,
            rationale=reason,
            judge_group=group,
            **extra,  # type: ignore[arg-type]
        )

    if not definition.available:
        preset(
            MetricStatus.UNAVAILABLE,
            definition.unavailable_reason or "metric could not be compiled",
            failure_source="metric_compiler",
        )
        return
    for asset in definition.required_reference_assets:
        if evidence.asset_missing(asset):
            status = (
                MetricStatus.CANNOT_GENERATE
                if definition.needs_expectations
                else MetricStatus.NOT_APPLICABLE
            )
            preset(status, ASSET_TEXT[asset], missing_requirement=f"reference: {asset.value}")
            return
    expectation = question.context.expectation(metric_id)
    if definition.needs_expectations:
        if expectation is None:
            preset(
                MetricStatus.CANNOT_GENERATE,
                "the question has no expectation for this metric",
                missing_requirement="question expectation",
            )
            return
        if expectation.status is MetricExpectationStatus.NOT_APPLICABLE:
            preset(
                MetricStatus.NOT_APPLICABLE,
                expectation.reason or "the question generator marked this metric not applicable",
            )
            return
        if expectation.status is MetricExpectationStatus.CANNOT_GENERATE:
            preset(
                MetricStatus.CANNOT_GENERATE,
                expectation.reason or "the question expectation could not be generated",
                missing_requirement="question expectation",
            )
            return
    if isinstance(evidence.response, DeniedResponse):
        beyond_answer = [
            k for k in definition.required_execution_evidence if k is not EvidenceKind.ANSWER
        ]
        if beyond_answer:
            preset(
                MetricStatus.NOT_APPLICABLE,
                "typed denial: the product returned no "
                + ", ".join(k.value for k in beyond_answer)
                + " to assess",
            )
            return
    for kind in definition.required_execution_evidence:
        reason = evidence.missing(kind)
        if reason is None:
            continue
        policy = definition.when_evidence_missing
        if policy is MissingEvidencePolicy.SCORE_ZERO:
            plan.deferred[metric_id] = (definition, kind)
        elif policy is MissingEvidencePolicy.NOT_APPLICABLE:
            preset(MetricStatus.NOT_APPLICABLE, reason, missing_requirement=kind.value)
        else:
            preset(
                MetricStatus.EVIDENCE_UNAVAILABLE,
                reason,
                missing_requirement=kind.value,
                failure_source="execution telemetry",
            )
        return
    context = evidence.context
    uses_chunks = {EvidenceKind.RETRIEVED_CONTEXT, EvidenceKind.RANKED_CONTEXT} & set(
        definition.required_execution_evidence
    )
    if uses_chunks and context.availability is ContextAvailability.AVAILABLE and not context.chunks:
        if definition.when_no_chunks is EmptyContextPolicy.NOT_APPLICABLE:
            preset(MetricStatus.NOT_APPLICABLE, NO_CHUNKS + " The metric is undefined.")
            return
        if definition.when_no_chunks is EmptyContextPolicy.SCORE_ZERO:
            preset(
                MetricStatus.SCORED,
                NO_CHUNKS,
                score=0.0,
                what_didnt=[NO_CHUNKS],
                details={"retrieved_chunks": 0},
            )
            if definition.counts_toward_verdict:
                plan.findings.append(
                    DeterministicFinding(
                        metric_id,
                        NO_CHUNKS,
                        "The product retrieves the information this question needs.",
                        "The product retrieved zero chunks.",
                    )
                )
            return
    if EvidenceKind.CITATIONS in definition.required_execution_evidence and not context.citations:
        required = bool(
            expectation is not None
            and expectation.value is not None
            and expectation.value.get(CITATIONS_REQUIRED_FIELD) is True
        )
        if not required:
            preset(MetricStatus.NOT_APPLICABLE, "The answer exposes no citations to verify.")
            return
        message = "Citations were required but the answer exposes none."
        preset(MetricStatus.SCORED, message, score=0.0, what_didnt=[message])
        plan.findings.append(
            DeterministicFinding(
                metric_id,
                message,
                "The answer cites the retrieved sources that support it.",
                "The answer contains no citation.",
            )
        )
        return
    if group is JudgeGroup.GENERAL:
        plan.general.append(definition)
    elif group is JudgeGroup.RETRIEVAL_GROUNDING:
        (plan.computed if definition.continuous_score else plan.retrieval).append(definition)
    else:
        preset(
            MetricStatus.EVALUATION_ERROR,
            "no deterministic implementation exists for this metric",
            failure_source="evaluator",
        )
