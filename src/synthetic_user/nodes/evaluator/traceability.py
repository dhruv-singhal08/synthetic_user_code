"""Evidence-backed metric decisions: check-level results, coverage, guidance and limitations.

For every selected metric the stored result explains (reforming, Part 10): what was compared
against what (comparison basis), the Success Criteria and Reference Spec requirements its Metric
Mapping assigned, the guidance applied (built-in rubric + cited items, each resolved to its exact
instruction or source), a numbered check per decision with the evidence ids inspected, the
coverage counts and the verification method. Code only - no LLM call:

- General Judge metrics: the judge's validated checks (every assigned requirement decided);
- Retrieval/Grounding metrics: code turns the judge's structured outputs into checks
  (required-information items, citation checks, chunk relevance labels);
- metrics code decided without a judge (not applicable, evidence unavailable, no SQL, ...):
  every assigned requirement gets a code check that states why (never silently omitted).

The evaluation also gets its explicit limitations (static SQL analysis, parse failures,
unresolved identifiers, not-found lookups, missing telemetry, unresolved checks, ...) and the
EVALUATION TRACE block of run.log (A input, B evidence, C judge invocations, D lookups, E metric
decisions, F final verdict).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from synthetic_user.core.trace import record_evaluation
from synthetic_user.models.evaluation import (
    CheckCoverage,
    CheckDraft,
    CoverageStatus,
    EvaluationResult,
    FactualVerification,
    GuidanceApplication,
    JudgedScore,
    MetricResult,
    MetricStatus,
    RequirementCheck,
    RequirementSource,
    RequirementStatus,
    RetrievalJudgeDraft,
    SupportStatus,
)
from synthetic_user.models.lookup import (
    IdentifierStatus,
    LookupOrigin,
    LookupTraceEntry,
    ReferenceCheckRecord,
)
from synthetic_user.models.metrics import EvidenceKind, JudgeGroup, MetricDefinition
from synthetic_user.models.question import QuestionCase
from synthetic_user.nodes.evaluator.evidence import lookup_label
from synthetic_user.nodes.evaluator.guidance import GuidanceRef, metric_guidance_id

# Limitation texts (shown in results, the UI and run.log).
STATIC_SQL = (
    "Static SQL analysis only: the SQL was not executed against a database and no returned "
    "values were verified."
)
NOT_EXHAUSTIVE = (
    "Reference evidence is not exhaustive: the Reference Spec, the catalog reconciliation and "
    "exact lookups were consulted, not every catalog entry."
)
TOOL_TRACE_LIMIT = (
    "Tool calls are judged from the trace the product exposes; calls it does not expose cannot "
    "be assessed."
)
_JUDGED = frozenset({MetricStatus.SCORED, MetricStatus.INCONCLUSIVE})
_FAILED_SUPPORT = {SupportStatus.PARTLY, SupportStatus.UNSUPPORTED}
_NOT_VERIFIED = {FactualVerification.UNVERIFIED.value, FactualVerification.PARTLY_VERIFIED.value}


@dataclass
class CheckNumbering:
    """CHECK:001, CHECK:002, ... across one evaluation."""

    next_number: int = 1

    def next(self) -> str:
        check_id = f"CHECK:{self.next_number:03d}"
        self.next_number += 1
        return check_id


@dataclass
class TraceInputs:
    """What the traceability of one evaluation reads (judge drafts and the run's registries)."""

    question: QuestionCase
    definitions: Mapping[str, MetricDefinition]
    guidance_refs: Mapping[str, GuidanceRef]
    general_scores: Mapping[str, JudgedScore] = field(default_factory=dict)
    retrieval: RetrievalJudgeDraft | None = None
    # Rule id -> its original text, from the catalog lookups (derived rule checks).
    rule_texts: Mapping[str, str] = field(default_factory=dict)
    # metric id -> guidance ids its exempted findings relied on.
    exemptions: Mapping[str, Sequence[str]] = field(default_factory=dict)


def trace_metrics(
    metrics: Mapping[str, MetricResult], inputs: TraceInputs
) -> dict[str, MetricResult]:
    numbering = CheckNumbering()
    return {
        metric_id: trace_metric(metric_id, result, inputs, numbering)
        for metric_id, result in metrics.items()
    }


def trace_metric(
    metric_id: str, result: MetricResult, inputs: TraceInputs, numbering: CheckNumbering
) -> MetricResult:
    definition = inputs.definitions.get(metric_id)
    context = inputs.question.context
    mapping = context.metric_mapping.get(metric_id)
    criteria = list(mapping.criteria) if mapping is not None else []
    references = list(mapping.reference_requirements) if mapping is not None else []
    judged = inputs.general_scores.get(metric_id)
    if judged is not None and result.status in _JUDGED:
        checks = [_from_draft(c, inputs, numbering) for c in judged.checks]
    else:
        checks = _code_checks([*criteria, *references], criteria, result, inputs, numbering)
    if (
        result.status in _JUDGED
        and inputs.retrieval is not None
        and definition is not None
        and definition.judge_group is JudgeGroup.RETRIEVAL_GROUNDING
    ):
        checks.extend(_retrieval_checks(definition, inputs.retrieval, numbering))
    guidance = _guidance(metric_id, result, judged, checks, inputs)
    evidence = list(
        dict.fromkeys(
            [
                *(i for c in checks for i in c.observed_evidence_ids),
                *(_evidence_id(ref) for ref in result.evidence_refs),
            ]
        )
    )
    return result.model_copy(
        update={
            "comparison_basis": mapping.comparison_basis
            if mapping is not None
            else (definition.basis if definition is not None else None),
            "verification_method": definition.method if definition is not None else None,
            "assigned_criteria": criteria,
            "assigned_reference_requirements": references,
            "guidance_applied": guidance,
            "checks": checks,
            "observed_evidence_ids": evidence,
            "coverage": coverage(checks, len(criteria) + len(references)),
        }
    )


def coverage(checks: Sequence[RequirementCheck], assigned: int) -> CheckCoverage:
    mapped = [c for c in checks if c.requirement_source is not RequirementSource.DERIVED]
    count = {status: sum(1 for c in checks if c.status is status) for status in RequirementStatus}
    return CheckCoverage(
        assigned=assigned,
        evaluated=sum(1 for c in mapped if c.status is not RequirementStatus.UNRESOLVED),
        passed=count[RequirementStatus.PASSED],
        failed=count[RequirementStatus.FAILED],
        not_applicable=count[RequirementStatus.NOT_APPLICABLE],
        unresolved=count[RequirementStatus.UNRESOLVED],
        derived=len(checks) - len(mapped),
    )


def _from_draft(
    draft: CheckDraft, inputs: TraceInputs, numbering: CheckNumbering
) -> RequirementCheck:
    requirement_id = draft.requirement_id.strip()
    return RequirementCheck(
        check_id=numbering.next(),
        requirement_source=draft.source,
        requirement_id=requirement_id,
        requirement=_requirement_text(requirement_id, inputs),
        status=draft.status,
        observed_evidence_ids=list(dict.fromkeys(draft.evidence_ids)),
        guidance_ids=list(dict.fromkeys(draft.guidance_ids)),
        reason=draft.reason,
    )


def _code_checks(
    assigned: Sequence[str],
    criteria: Sequence[str],
    result: MetricResult,
    inputs: TraceInputs,
    numbering: CheckNumbering,
) -> list[RequirementCheck]:
    """Every assigned requirement of a metric no judge decided, with the reason code knows."""
    if result.status is MetricStatus.NOT_APPLICABLE:
        status = RequirementStatus.NOT_APPLICABLE
    elif result.status is MetricStatus.SCORED and result.score is not None:
        status = RequirementStatus.PASSED if result.score >= 1 else RequirementStatus.FAILED
    else:
        status = RequirementStatus.UNRESOLVED
    return [
        RequirementCheck(
            check_id=numbering.next(),
            requirement_source=RequirementSource.SUCCESS_CRITERIA
            if rid in criteria
            else RequirementSource.REFERENCE_SPEC,
            requirement_id=rid,
            requirement=_requirement_text(rid, inputs),
            status=status,
            reason=f"decided by code ({result.status.value}): {result.rationale}",
        )
        for rid in assigned
    ]


def _retrieval_checks(
    definition: MetricDefinition, draft: RetrievalJudgeDraft, numbering: CheckNumbering
) -> list[RequirementCheck]:
    """The Retrieval/Grounding Judge's structured outputs as checks (no prompt change)."""
    evidence = set(definition.required_execution_evidence)
    checks: list[RequirementCheck] = []
    if any(f.name == "required_information" for f in definition.question_expectation_schema):
        for i, item in enumerate(draft.required_information, start=1):
            checks.append(
                RequirementCheck(
                    check_id=numbering.next(),
                    requirement_source=RequirementSource.DERIVED,
                    requirement_id=f"EXPECTATION:required_information:{i}",
                    requirement=item.item,
                    status=RequirementStatus.PASSED
                    if item.status is CoverageStatus.COVERED
                    else RequirementStatus.FAILED,
                    observed_evidence_ids=[f"CHUNK:{c}" for c in item.chunk_ids],
                    reason=f"{item.status.value}: {item.rationale}",
                )
            )
    if EvidenceKind.CITATIONS in evidence:
        for check in draft.citation_checks:
            checks.append(
                RequirementCheck(
                    check_id=numbering.next(),
                    requirement_source=RequirementSource.DERIVED,
                    requirement_id=f"CITATION:{check.citation}",
                    requirement="the citation is supported by the chunk it cites",
                    status=RequirementStatus.FAILED
                    if check.support in _FAILED_SUPPORT
                    else RequirementStatus.PASSED,
                    observed_evidence_ids=[f"CHUNK:{check.chunk_id}"] if check.chunk_id else [],
                    reason=f"{check.support.value}: {check.rationale}",
                )
            )
    if (
        EvidenceKind.RETRIEVED_CONTEXT in evidence
        and not definition.continuous_score
        and not definition.question_expectation_schema
        and EvidenceKind.CITATIONS not in evidence
        and EvidenceKind.ANSWER not in evidence
    ):
        for label in draft.chunk_relevance:  # context precision: one check per labelled chunk
            checks.append(
                RequirementCheck(
                    check_id=numbering.next(),
                    requirement_source=RequirementSource.DERIVED,
                    requirement_id=f"CHUNK:{label.chunk_id}",
                    requirement="the retrieved chunk is relevant to the question",
                    status=RequirementStatus.PASSED if label.relevant else RequirementStatus.FAILED,
                    observed_evidence_ids=[f"CHUNK:{label.chunk_id}"],
                    reason=label.rationale,
                )
            )
    return checks


def _guidance(
    metric_id: str,
    result: MetricResult,
    judged: JudgedScore | None,
    checks: Sequence[RequirementCheck],
    inputs: TraceInputs,
) -> list[GuidanceApplication]:
    """Built-in rubric (every judged metric), the judge's guidance_applied, guidance ids cited
    in checks and the guidance behind exempted findings - each resolved to its instruction."""
    refs = inputs.guidance_refs
    applied: dict[str, GuidanceApplication] = {}

    def add(guidance_id: str, influence: str, applicability: str | None = None) -> None:
        ref = refs.get(guidance_id)
        if ref is None or guidance_id in applied:
            return
        applied[guidance_id] = GuidanceApplication(
            guidance_id=guidance_id,
            guidance_type=ref.guidance_type,
            source=ref.source,
            instruction=ref.instruction,
            applicability=applicability,
            influence=influence,
        )

    if result.status in _JUDGED:
        add(
            metric_guidance_id(metric_id),
            "the metric's rubric and score anchors decided how the evidence was scored",
        )
    for item in judged.guidance_applied if judged is not None else []:
        add(item.guidance_id, item.influence, item.applicability)
    for check in checks:
        for guidance_id in check.guidance_ids:
            add(guidance_id, f"cited by {check.check_id} ({check.requirement_id})")
    for guidance_id in inputs.exemptions.get(metric_id, ()):
        add(guidance_id, "an approved exception: the difference it covers is not a defect")
    return list(applied.values())


def _requirement_text(requirement_id: str, inputs: TraceInputs) -> str | None:
    context = inputs.question.context
    if (text := context.requirement_text(requirement_id)) is not None:
        return text
    if (ref := inputs.guidance_refs.get(requirement_id)) is not None:
        return ref.instruction
    return inputs.rule_texts.get(requirement_id)


def _evidence_id(ref: str) -> str:
    """A metric's evidence ref as an evidence id (retrieval refs are raw chunk ids)."""
    if ":" in ref or ref == "RESPONSE":
        return ref
    if ref.startswith("tool-"):
        return f"TOOL:{ref.removeprefix('tool-')}"
    return f"CHUNK:{ref}"


def rule_texts(entries: Sequence[LookupTraceEntry]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for entry in entries:
        for rule in getattr(entry.result, "rules", []):
            texts[rule.rule_id] = rule.text
    return texts


# --- evaluation-level summary -----------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationSummary:
    limitations: list[str]
    failed: list[str]
    unresolved: list[str]


def summarise(
    metrics: Mapping[str, MetricResult],
    definitions: Mapping[str, MetricDefinition],
    *,
    sql_parse: Sequence[tuple[int, bool, Sequence[str]]] = (),
    reference_checks: ReferenceCheckRecord | None = None,
    notes: Sequence[str] = (),
) -> EvaluationSummary:
    """The evaluation's explicit limitations and its failed / unresolved requirements.
    ``sql_parse`` = (artifact number, parsed, diagnostics) per SQL artifact."""
    limitations: list[str] = []
    needed = [
        set(definition.required_execution_evidence)
        for metric_id, result in metrics.items()
        if result.status in _JUDGED and (definition := definitions.get(metric_id)) is not None
    ]
    if any(EvidenceKind.SQL in evidence for evidence in needed):
        limitations.append(STATIC_SQL)
    for artifact, parsed, diagnostics in sql_parse:
        if not parsed:
            detail = "; ".join(diagnostics) or "no diagnostics"
            limitations.append(
                f"SQL artifact {artifact} could not be parsed ({detail}): parser-dependent "
                "checks are inconclusive."
            )
    if reference_checks is not None:
        limitations.append(NOT_EXHAUSTIVE)
        for rec in reference_checks.artifacts:
            unresolved = [i.identifier for i in rec.with_status(IdentifierStatus.UNRESOLVED)]
            if unresolved:
                limitations.append(
                    f"SQL artifact {rec.artifact}: identifiers that could not be resolved in the "
                    f"catalog: {', '.join(unresolved)}."
                )
        for entry in reference_checks.lookup_trace:
            if entry.status != "found":
                limitations.append(
                    f"{entry.evidence_id or 'lookup'} ({lookup_label(entry)}): unavailable "
                    "or incomplete lookup result."
                )
    if any(EvidenceKind.TOOL_TRACE in evidence for evidence in needed):
        limitations.append(TOOL_TRACE_LIMIT)
    failed: list[str] = []
    unresolved_reqs: list[str] = []
    for metric_id, result in metrics.items():
        if result.status is MetricStatus.EVIDENCE_UNAVAILABLE:
            limitations.append(f"{metric_id}: {result.rationale}")
        state = result.details.get("factual_verification")
        if state in (
            FactualVerification.UNVERIFIED.value,
            FactualVerification.PARTLY_VERIFIED.value,
        ):
            limitations.append(
                f"{metric_id}: factual correctness {state} against trusted evidence."
            )
        for check in result.checks:
            label = f"{metric_id}: {check.requirement_id}"
            if check.status is RequirementStatus.FAILED:
                failed.append(label)
            elif check.status is RequirementStatus.UNRESOLVED:
                unresolved_reqs.append(label)
                limitations.append(f"{label} unresolved - {check.reason}")
    limitations.extend(notes)
    return EvaluationSummary(
        limitations=list(dict.fromkeys(limitations)),
        failed=list(dict.fromkeys(failed)),
        unresolved=list(dict.fromkeys(unresolved_reqs)),
    )


def mark_used_lookups(
    record: ReferenceCheckRecord | None, metrics: Mapping[str, MetricResult]
) -> ReferenceCheckRecord | None:
    """Whether each lookup was cited by a final metric decision; whether the judge requested
    its one optional batch."""
    if record is None:
        return None
    cited = {i for r in metrics.values() for i in r.observed_evidence_ids}
    trace = [
        entry.model_copy(update={"used_in_judgment": entry.evidence_id in cited})
        for entry in record.lookup_trace
    ]
    return record.model_copy(
        update={
            "lookup_trace": trace,
            "additional_lookup_performed": record.judge_lookup_requests > 0,
        }
    )


# --- run.log: the EVALUATION TRACE block ------------------------------------------------------


@dataclass
class InvocationRecord:
    """One judge call as the trace shows it (no prompt text: that is in the LLM CALL blocks)."""

    invocation_id: str
    judge: str
    metrics: list[str]
    attempts: list[int] = field(default_factory=list)
    evidence_supplied: list[str] = field(default_factory=list)
    guidance_supplied: list[str] = field(default_factory=list)
    template_fingerprint: str | None = None
    error: str | None = None


def log_evaluation(
    result: EvaluationResult,
    question: QuestionCase,
    *,
    selected: Sequence[str],
    evidence_preparation: dict[str, Any],
    invocations: Sequence[InvocationRecord],
) -> None:
    """Write the question's EVALUATION TRACE to run.log (no-op outside a traced step)."""
    context = question.context
    spec = context.reference_spec
    user_guidance = sorted(
        {g.guidance_id for m in result.metrics.values() for g in m.guidance_applied}
    )
    sections: list[tuple[str, Any]] = [
        (
            "A. EVALUATION INPUT",
            {
                "question_id": question.question_id,
                "scenario_id": question.scenario_id,
                "execution_id": result.execution_id,
                "mode": question.mode.value,
                "selected_metrics": list(selected),
                "success_criteria": [c.model_dump(mode="json") for c in context.success_criteria],
                "reference_spec": {
                    "pack_id": spec.pack_id,
                    "reference_id": spec.reference_id,
                    "requirements": [f"{r.id}: {r.text}" for r in spec.requirements()],
                }
                if spec is not None
                else None,
                "metric_mapping": {
                    k: v.model_dump(mode="json", exclude={"expectation"})
                    for k, v in context.metric_mapping.items()
                },
                "guidance_ids_applied": user_guidance,
            },
        ),
        ("B. EVIDENCE PREPARATION", evidence_preparation),
        (
            "C. JUDGE INVOCATIONS",
            [
                {
                    "invocation_id": inv.invocation_id,
                    "judge": inv.judge,
                    "question_id": question.question_id,
                    "assigned_requirements": {
                        m: [
                            *context.metric_mapping[m].criteria,
                            *context.metric_mapping[m].reference_requirements,
                        ]
                        for m in inv.metrics
                        if m in context.metric_mapping
                    },
                    "guidance_supplied": inv.guidance_supplied,
                    "evidence_supplied": inv.evidence_supplied,
                    "template_fingerprint": inv.template_fingerprint,
                    "attempts": inv.attempts,
                    "error": inv.error,
                }
                for inv in invocations
            ]
            or "no judge call (no applicable LLM-judged metric)",
        ),
        ("D. OPTIONAL EVIDENCE LOOKUPS", _lookup_section(result.reference_checks)),
        (
            "E. METRIC DECISIONS",
            {
                metric_id: {
                    "status": r.status.value,
                    "score": r.score,
                    "comparison_basis": r.comparison_basis,
                    "verification_method": r.verification_method,
                    "criteria_evaluated": r.assigned_criteria,
                    "reference_requirements_evaluated": r.assigned_reference_requirements,
                    "guidance_applied": [
                        f"{g.guidance_id} ({g.guidance_type}): {g.influence}"
                        for g in r.guidance_applied
                    ],
                    "checks": [
                        f"{c.check_id} {c.requirement_source.value} {c.requirement_id} "
                        f"{c.status.value} evidence={c.observed_evidence_ids}: {c.reason}"
                        for c in r.checks
                    ],
                    "evidence_ids": r.observed_evidence_ids,
                    "coverage": r.coverage.model_dump(mode="json") if r.coverage else None,
                    "rationale": r.rationale,
                }
                for metric_id, r in result.metrics.items()
            },
        ),
        (
            "F. FINAL VERDICT",
            {
                "metric_scores": {
                    m: (r.score if r.score is not None else r.status.value)
                    for m, r in result.metrics.items()
                },
                "verdict": result.verdict.value if result.verdict else "recorded_only",
                "verdict_reason": result.verdict_reason,
                "verdict_policy": result.verdict_policy,
                "failed_requirements": result.failed_requirements,
                "unresolved_requirements": result.unresolved_requirements,
                "applied_exceptions": [
                    f"{f.finding_id} {f.exempted_by_id or ''}: {f.exempted_by}"
                    for f in result.findings
                    if f.exempted_by is not None
                ],
                "limitations": result.limitations,
            },
        ),
    ]
    detail = f"{result.evaluation_id} · {result.verdict.value if result.verdict else 'recorded'}"
    record_evaluation(question.question_id, detail, sections)


def _lookup_section(record: ReferenceCheckRecord | None) -> Any:
    if record is None:
        return {"additional_lookup_performed": False, "note": "no catalog reconciliation"}
    return {
        "additional_lookup_performed": bool(record.additional_lookup_performed),
        "lookups": [
            {
                "evidence_id": e.evidence_id,
                "round": e.round,
                "origin": "precomputed"
                if e.origin is LookupOrigin.DETERMINISTIC
                else "judge_requested",
                "operation": e.request.tool.value,
                "arguments": e.request.model_dump(
                    mode="json", exclude={"tool", "reason"}, exclude_defaults=True
                ),
                "reason": e.request.reason,
                "status": e.status,
                "used_in_judgment": e.used_in_judgment,
            }
            for e in record.lookup_trace
        ],
    }
