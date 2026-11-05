"""Node 5: evaluate product executions against the run's frozen metric configuration.

Routing by the executor's typed outcome (never overwritten by an LLM label):
- answered      -> evaluated; response_classification from the adapter's typed field, else
                   from the General Judge's normal call, else ``unknown`` (no extra call);
- denied        -> evaluated as ``unable_to_answer`` (answer-level metrics only; no SQL / tool
                   / context evidence is fabricated);
- clarification -> terminal clarification: recorded only (classification ``clarification``);
- error/timeout -> recorded only (classification None). Never judged.

Per evaluable response, applicability is decided by code (applicability.py); then at most two
LLM calls run concurrently: the General Judge (general metrics) and the Retrieval/Grounding
Judge (retrieval metrics + chunk labels for MRR). A group without applicable metrics makes no
call; zero LLM metrics -> zero calls. MRR is computed by code. The final verdict comes from the
provisional ``VerdictPolicy`` (verdict.py); the judge's own opinion is stored as llm_verdict.
Verdict: ``verdict.py`` (D241) - decided from the findings' severity, always with a reason.
Bug candidates: one per root cause, from the verdict's failing findings (D244).

Traceability (reforming): every metric result carries its comparison basis, the requirements
the question's Metric Mapping assigned, check-level decisions with evidence ids, coverage, the
guidance applied and the verification method; the evaluation carries its limitations and its
failed / unresolved requirements, and run.log gets one EVALUATION TRACE block per question
(traceability.py). A question without an Evaluation Context (legacy) is refused. LLM calls run
under ``llm.max_concurrency``; ids are assigned afterwards in order.

SQL against a data catalog (D230/D231): each SQL artifact is parsed once, reconciled with the
cheat sheet AND the full catalog (exact lookups through one ``CatalogIndex`` per catalog
version), the obvious gaps are looked up before the call, and the General Judge may request
one batch of further lookups. The SQL is judged exactly as generated (no predicate is hidden).
The reconciliation and the lookup trace are stored on the evaluation (``reference_checks``).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import JsonValue

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import ArtifactIntegrityError
from synthetic_user.core.identifiers import IdKind, SequentialIdGenerator
from synthetic_user.core.trace import trace_subject
from synthetic_user.llm.base import LLMClient
from synthetic_user.models.evaluation import (
    EVALUATION_SCHEMA_VERSION,
    GOAL_PROBLEM_SEVERITY,
    GOAL_ROOT_CAUSE,
    ClassificationSource,
    EvaluationResult,
    EvaluationStatus,
    EvidenceStatus,
    FactualVerification,
    Finding,
    FindingSource,
    GoalCompletion,
    JudgedScore,
    MetricResult,
    MetricStatus,
    RequirementSource,
    RequirementStatus,
    RootCause,
    RootCauseMechanism,
    Severity,
)
from synthetic_user.models.execution import (
    AnsweredResponse,
    ConversationTurnKind,
    DeniedResponse,
    ExecutionRecord,
    ProductOutcome,
    ResponseClassification,
)
from synthetic_user.models.lookup import (
    IdentifierStatus,
    LookupOrigin,
    LookupTraceEntry,
    ReferenceCheckRecord,
)
from synthetic_user.models.metrics import (
    GOAL_COMPLETION_CATEGORY,
    EvidenceKind,
    JudgeGroup,
    MetricDefinition,
    MetricOrigin,
    ReferenceAsset,
    ResolvedMetricConfig,
)
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import LEGACY_QUESTION_MESSAGE, QuestionCase
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.run import EvaluationGuidance, GenerationMode
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator.applicability import (
    MISSING_EVIDENCE_TEXT,
    DeterministicFinding,
    EvaluationEvidence,
    MetricPlan,
    plan_metrics,
)
from synthetic_user.nodes.evaluator.bugs import (
    BugGroup,
    SubjectResolver,
    assign_bugs,
    bug_groups,
)
from synthetic_user.nodes.evaluator.evidence import SQL_CLAUSES, sql_clause_id
from synthetic_user.nodes.evaluator.general_judge import GeneralJudge, GeneralJudgement
from synthetic_user.nodes.evaluator.general_prompt import (
    TEMPLATE_FINGERPRINT,
    GeneralJudgeInput,
    SqlEvidence,
)
from synthetic_user.nodes.evaluator.guidance import guidance_registry
from synthetic_user.nodes.evaluator.judge_common import (
    evidence_texts,
    matches_known_gap,
    normalise,
    quoted,
    response_text,
)
from synthetic_user.nodes.evaluator.no_sql import NO_SQL_REASON
from synthetic_user.nodes.evaluator.retrieval_judge import (
    RetrievalJudge,
    RetrievalJudgement,
    reciprocal_rank,
)
from synthetic_user.nodes.evaluator.retrieval_prompt import (
    REQUIRED_INFORMATION_FIELD,
    RetrievalJudgeInput,
)
from synthetic_user.nodes.evaluator.trace_tags import (
    link_finding_tags,
    merge_finding_tags,
    trace_catalog,
)
from synthetic_user.nodes.evaluator.traceability import (
    InvocationRecord,
    TraceInputs,
    log_evaluation,
    mark_used_lookups,
    rule_texts,
    summarise,
    trace_metrics,
)
from synthetic_user.nodes.evaluator.verdict import SeverityVerdictPolicy, VerdictPolicy
from synthetic_user.references.catalog_index import CatalogIndex, LookupSession
from synthetic_user.sql.parser import SqlglotParser, SqlParser
from synthetic_user.sql.reconcile import evidence_requests, reconcile

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

CONTEXT_RECALL_DISCLOSURE = (
    "Expectation-based coverage of the question's required-information checklist; "
    "not exhaustive recall over the full corpus."
)
# A metric of a failed judge call: the error itself is recorded ONCE in the notes (D251).
SHARED_ERROR = (
    "Not scored: the {judge} call failed - one shared error for {metrics}, recorded in the "
    "evaluation notes."
)
UNKNOWN_AFTER_JUDGE_ERROR = (
    "Response classification unresolved: the General Judge call failed (see the error note)."
)
RECORDED_ONLY_REASON = (
    "Recorded only: a {outcome} response has no answer or denial to judge, so it gets no verdict."
)
UNKNOWN_CLASSIFICATION_NOTE = (
    "Response classification unresolved: the adapter exposes no typed response kind and no "
    "General Judge call ran for this response."
)


@dataclass(frozen=True)
class EvaluationCase:
    question: QuestionCase
    scenario: Scenario
    execution: ExecutionRecord

    @property
    def pack(self) -> ReferencePack | None:
        """The cheat sheet the question was generated with (its Reference Spec); grounded
        questions only. None for a legacy question (refused by the evaluator)."""
        context = self.question.evaluation_context
        return context.pack if context is not None else None


@dataclass
class _Judgements:
    plan: MetricPlan
    evidence: EvaluationEvidence
    general: GeneralJudgement | None = None
    retrieval: RetrievalJudgement | None = None
    notes: list[str] = field(default_factory=list)
    reference_checks: ReferenceCheckRecord | None = None
    sql: list[SqlEvidence] = field(default_factory=list)
    general_input: GeneralJudgeInput | None = None


class Evaluator:
    def __init__(
        self,
        llm: LLMClient,
        settings: AppConfig,
        logger: Logger,
        *,
        reference: Reference | None,
        metric_config: ResolvedMetricConfig,
        id_generator: SequentialIdGenerator | None = None,
        sql_parser: SqlParser | None = None,
        verdict_policy: VerdictPolicy | None = None,
        evaluation_guidance: EvaluationGuidance | None = None,
    ) -> None:
        self._settings = settings
        self._log = logger
        self._ids = id_generator or SequentialIdGenerator()
        self._reference = reference
        self._config = metric_config
        self._definitions = {d.metric_id: d for d in metric_config.definitions}
        retries = settings.llm.max_validation_retries
        self._general = GeneralJudge(llm, logger, max_retries=retries)
        self._retrieval = RetrievalJudge(llm, logger, max_retries=retries)
        self._parser: SqlParser = sql_parser or SqlglotParser()
        self._policy = verdict_policy or SeverityVerdictPolicy(settings.evaluation.fail_severity)
        # Built once per catalog version (the reference is fixed for the run); None for
        # document collections and ungrounded runs.
        self._catalog = CatalogIndex.for_reference(
            reference, namespaces=settings.evaluation.catalog_namespaces
        )
        # Run-level guidance for every judge call (no extra call); None -> prompts unchanged.
        self._guidance = evaluation_guidance
        # Every citable guidance id: the user's items, METRIC:<id> rubrics, POLICY:<name>.
        self._guidance_refs = guidance_registry(
            metric_config.definitions,
            evaluation_guidance,
            catalog_version=metric_config.catalog_version,
        )
        if evaluation_guidance is not None:
            logger.info(
                "Evaluation guidance applied to every judge call: %s (sha256 %s, %d chars)",
                evaluation_guidance.source,
                evaluation_guidance.sha256[:12],
                len(evaluation_guidance.text),
            )
        custom = [d for d in metric_config.definitions if d.origin is MetricOrigin.CUSTOM]
        if custom:
            logger.info(
                "Custom metric routing: %s",
                ", ".join(
                    f"{d.metric_id}={d.judge_group.value}/{d.evaluation_target.value}"
                    for d in custom
                ),
            )

    @property
    def metric_config(self) -> ResolvedMetricConfig:
        return self._config

    async def evaluate(self, case: EvaluationCase, product: ProductDescription) -> EvaluationResult:
        return (await self.evaluate_many([case], product))[0]

    async def evaluate_many(
        self, cases: Sequence[EvaluationCase], product: ProductDescription
    ) -> list[EvaluationResult]:
        for case in cases:
            _check_links(case, self._reference)
        semaphore = asyncio.Semaphore(self._settings.llm.max_concurrency)
        judgements = await asyncio.gather(*(self._judge(c, product, semaphore) for c in cases))
        assembled = [self._assemble(c, j, product) for c, j in zip(cases, judgements, strict=True)]
        # One bug per root cause across the batch (D244), from the failing findings only.
        results = assign_bugs(
            [
                (result, case.question, groups)
                for case, (result, groups) in zip(cases, assembled, strict=True)
            ],
            product.product_id,
            self._ids,
        )
        self._log.info(
            "Evaluated %d execution(s): %s",
            len(results),
            ", ".join(
                f"{v}={sum(1 for r in results if (r.verdict or 'recorded_only') == v)}"
                for v in ("pass", "fail", "inconclusive", "recorded_only")
            ),
        )
        return results

    # --- phase 1: judgements (concurrent, no ids) ----------------------------------------

    async def _judge(
        self,
        case: EvaluationCase,
        product: ProductDescription,
        semaphore: asyncio.Semaphore,
    ) -> _Judgements | None:
        response = case.execution.response
        if not isinstance(response, AnsweredResponse | DeniedResponse):
            return None  # clarification / error / timeout: recorded only
        evidence = EvaluationEvidence.collect(
            response, case.execution.context, case.pack, self._reference
        )
        plan = plan_metrics(self._config.definitions, case.question, evidence)
        result = _Judgements(plan=plan, evidence=evidence)
        conversation = [
            turn
            for turn in case.execution.turns
            if turn.kind
            in (ConversationTurnKind.CLARIFICATION, ConversationTurnKind.CLARIFICATION_ANSWER)
        ]
        artifact_max_chars = self._settings.evaluation.artifact_max_chars
        calls: list[Any] = []
        if plan.general:
            sql = self._sql_evidence(response, plan, case.pack)
            result.sql = sql
            session = (
                LookupSession(self._catalog)
                if self._catalog is not None and any(i.reconciliation for i in sql)
                else None
            )
            catalog_lookups = self._catalog_lookups(case, sql, session)
            lookup_settings = self._settings.evaluation.reference_lookup
            general_input = GeneralJudgeInput(
                product=product,
                question=case.question,
                response=response,
                definitions=plan.general,
                conversation=conversation,
                pack=case.pack,
                reference=self._reference,
                sql=sql,
                tool_trace=evidence.tool_trace,
                tool_definitions=any(
                    ReferenceAsset.TOOL_DEFINITIONS in d.required_reference_assets
                    for d in plan.general
                ),
                needs_classification=isinstance(response, AnsweredResponse)
                and response.response_classification is None,
                sql_dialect=self._settings.evaluation.sql_dialect,
                artifact_max_chars=artifact_max_chars,
                catalog_lookups=catalog_lookups,
                max_lookup_requests=lookup_settings.max_requests
                if session is not None and lookup_settings.enabled
                else 0,
                guidance=self._guidance,
                guidance_refs=self._guidance_refs,
                capability=case.scenario.capability,
                capability_rationale=case.scenario.capability_rationale,
            )
            result.general_input = general_input
            calls.append(self._run_general(case, general_input, semaphore, result, session))
        if plan.retrieval_call:
            retrieval_input = RetrievalJudgeInput(
                product=product,
                question=case.question,
                response=response,
                definitions=plan.retrieval,
                needs_labels=bool(plan.computed),
                context=evidence.context,
                tool_trace=evidence.tool_trace,
                artifact_max_chars=artifact_max_chars,
                guidance=self._guidance,
            )
            calls.append(self._run_retrieval(case, retrieval_input, semaphore, result))
        started = time.perf_counter()
        if calls:
            # The two judge groups are independent calls; they may run concurrently.
            await asyncio.gather(*calls)
        self._log.info(
            "%s: judged in %.0f ms (general calls %d, lookup requests %d)",
            case.question.question_id,
            (time.perf_counter() - started) * 1000,
            result.general.calls if result.general else 0,
            result.general.lookup_requests if result.general else 0,
        )
        return result

    async def _run_general(
        self,
        case: EvaluationCase,
        data: GeneralJudgeInput,
        semaphore: asyncio.Semaphore,
        result: _Judgements,
        session: LookupSession | None,
    ) -> None:
        async with semaphore:
            with trace_subject(f"{case.question.question_id} general"):
                result.general = await self._general.judge(data, session)
        if session is not None:
            result.reference_checks = ReferenceCheckRecord(
                catalog_version=session.index.catalog_version,
                artifacts=[i.reconciliation for i in data.sql if i.reconciliation is not None],
                lookup_trace=session.trace,
                judge_lookup_requests=result.general.lookup_requests,
                general_judge_calls=result.general.calls,
                join_decisions=[
                    d.model_dump(mode="json")
                    for d in (result.general.draft.join_decisions if result.general.draft else [])
                ],
            )

    async def _run_retrieval(
        self,
        case: EvaluationCase,
        data: RetrievalJudgeInput,
        semaphore: asyncio.Semaphore,
        result: _Judgements,
    ) -> None:
        async with semaphore:
            with trace_subject(f"{case.question.question_id} retrieval"):
                result.retrieval = await self._retrieval.judge(data)

    def _sql_evidence(
        self,
        response: AnsweredResponse | DeniedResponse,
        plan: MetricPlan,
        pack: ReferencePack | None,
    ) -> list[SqlEvidence]:
        """Parser output (+ catalog reconciliation when the run has a data catalog) for each
        SQL artifact, only when a general metric needs SQL. Each artifact is parsed once and
        judged exactly as generated."""
        needs_sql = any(EvidenceKind.SQL in d.required_execution_evidence for d in plan.general)
        if not needs_sql or not isinstance(response, AnsweredResponse):
            return []
        dialect = self._settings.evaluation.sql_dialect
        evidence: list[SqlEvidence] = []
        for artifact in response.artifacts:
            if artifact.kind.casefold() != "sql" or not isinstance(artifact.content, str):
                continue
            structure = self._parser.parse(artifact.content, dialect=dialect)
            reconciliation = (
                reconcile(structure, artifact=len(evidence) + 1, index=self._catalog, pack=pack)
                if self._catalog is not None
                else None
            )
            evidence.append(SqlEvidence(structure, reconciliation))
        return evidence

    def _catalog_lookups(
        self,
        case: EvaluationCase,
        sql: Sequence[SqlEvidence],
        session: LookupSession | None,
    ) -> list[LookupTraceEntry]:
        """Run the lookups code can already see are needed (one batch), and log the
        reconciliation (parse outcome, extra / invalid / unresolved identifiers)."""
        reconciliations = [i.reconciliation for i in sql if i.reconciliation is not None]
        if session is None or not reconciliations:
            return []
        for rec in reconciliations:
            counts = {status.value: len(rec.with_status(status)) for status in IdentifierStatus}
            self._log.info(
                "%s: SQL artifact %d parsed=%s identifiers %s extra_tables=%s",
                case.question.question_id,
                rec.artifact,
                rec.parsed,
                counts,
                rec.extra_tables,
            )
        return session.run_all(
            evidence_requests(reconciliations, session.index), LookupOrigin.DETERMINISTIC
        )

    # --- phase 2: assembly (sequential, assigns ids) --------------------------------------

    def _assemble(
        self,
        case: EvaluationCase,
        judgements: _Judgements | None,
        product: ProductDescription,
    ) -> tuple[EvaluationResult, list[BugGroup]]:
        """The evaluation (without bugs yet) and its failing findings grouped by root cause."""
        outcome = case.execution.outcome
        ids: dict[str, Any] = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "evaluation_id": self._ids.next(IdKind.EVALUATION),
            "question_id": case.question.question_id,
            "scenario_id": case.scenario.id,
            "execution_id": case.execution.execution_id,
            "product_outcome": outcome,
            "mode": case.question.mode,
            "metric_config_id": self._config.config_id,
            "evaluation_guidance_sha256": self._guidance.sha256 if self._guidance else None,
        }
        if judgements is None:
            clarification = outcome is ProductOutcome.CLARIFICATION
            notes = (
                [NO_SQL_REASON]
                if outcome is ProductOutcome.TIMEOUT
                and self._config.needs_evidence(EvidenceKind.SQL)
                else []
            )
            recorded = EvaluationResult(
                **ids,
                response_classification=(
                    ResponseClassification.CLARIFICATION if clarification else None
                ),
                classification_source=(
                    ClassificationSource.TYPED_OUTCOME if clarification else None
                ),
                verdict=None,
                verdict_reason=" ".join(
                    [RECORDED_ONLY_REASON.format(outcome=outcome.value), *notes]
                ),
                evaluation_status=EvaluationStatus.RECORDED_ONLY,
                notes=notes,
            )
            return recorded, []

        plan, general, retrieval = judgements.plan, judgements.general, judgements.retrieval
        notes = list(judgements.notes)
        classification, source, classification_note = self._classification(case, general)
        if classification is ResponseClassification.UNKNOWN:
            notes.append(
                UNKNOWN_AFTER_JUDGE_ERROR
                if general is not None and general.error
                else UNKNOWN_CLASSIFICATION_NOTE
            )
        results: dict[str, MetricResult] = dict(plan.preset)
        deterministic = list(plan.findings)
        judge_limits: list[str] = []
        if general is not None:
            affected = [d.metric_id for d in plan.general]
            if general.error:
                # ONE shared error naming the affected metrics (each metric's reason points to
                # it); the per-attempt details are in run.log.
                notes.append(f"{general.error} [affects: {', '.join(affected)}]")
                judge_limits.append(
                    f"General Judge error: {', '.join(affected)} not scored (see the notes)."
                )
            if general.invalid_metrics:
                invalid = list(general.invalid_metrics)
                notes.append(
                    f"General Judge output stayed invalid for {', '.join(invalid)} only; the "
                    "other metrics of the same reply were validated independently and kept."
                )
                judge_limits.append(
                    f"{', '.join(invalid)}: not scored (judge output invalid for this metric)."
                )
            self._judged(
                results,
                plan.general,
                general.draft.metrics if general.draft else None,
                SHARED_ERROR.format(judge="General Judge", metrics=", ".join(affected))
                if general.error
                else None,
                JudgeGroup.GENERAL,
                invalid=general.invalid_metrics,
            )
        if retrieval is not None:
            if retrieval.error:
                affected = [d.metric_id for d in [*plan.retrieval, *plan.computed]]
                notes.append(f"{retrieval.error} [affects: {', '.join(affected)}]")
            self._judged(
                results,
                plan.retrieval,
                retrieval.draft.metrics if retrieval.draft else None,
                SHARED_ERROR.format(
                    judge="Retrieval/Grounding Judge",
                    metrics=", ".join(d.metric_id for d in plan.retrieval),
                )
                if retrieval.error
                else None,
                JudgeGroup.RETRIEVAL_GROUNDING,
            )
            self._retrieval_details(results, plan, retrieval, judgements.evidence)
        for metric_id, (definition, kind) in plan.deferred.items():
            reason = MISSING_EVIDENCE_TEXT[kind]
            if classification is ResponseClassification.UNABLE_TO_ANSWER:
                results[metric_id] = MetricResult(
                    status=MetricStatus.NOT_APPLICABLE,
                    rationale=f"The product stated it cannot answer; {reason[0].lower()}"
                    f"{reason[1:]}",
                    judge_group=definition.judge_group,
                    missing_requirement=kind.value,
                )
                continue
            results[metric_id] = MetricResult(
                status=MetricStatus.SCORED,
                score=0.0,
                rationale=reason,
                what_didnt=[reason],
                judge_group=definition.judge_group,
                missing_requirement=kind.value,
            )
            deterministic.append(
                DeterministicFinding(
                    metric_id,
                    reason,
                    f"The response includes the required {kind.value} evidence.",
                    reason,
                )
            )
            if kind is EvidenceKind.SQL:
                notes.append(NO_SQL_REASON)
        metrics = {
            metric_id: results[metric_id]
            for metric_id in self._config.selected_metric_ids
            if metric_id in results
        }
        general_scores = general.draft.metrics if general and general.draft else {}
        failed_criteria = {
            metric_id: [
                rid.strip()
                for c in judged.checks
                if c.status is RequirementStatus.FAILED
                and c.source is RequirementSource.SUCCESS_CRITERIA
                for rid in c.requirement_ids
            ]
            for metric_id, judged in general_scores.items()
        }
        drafts = [
            *(general.draft.findings if general and general.draft else []),
            *(retrieval.draft.findings if retrieval and retrieval.draft else []),
        ]
        findings = [
            self._finding(f.model_dump(), FindingSource.LLM, case, failed_criteria, product)
            for f in drafts
        ] + [
            self._finding(
                {
                    "category": d.metric_id,
                    "description": d.description,
                    "expected_behavior": d.expected_behavior,
                    "observed_behavior": d.observed_behavior,
                    "severity": Severity.HIGH,
                    "evidence_status": EvidenceStatus.SUPPORTED,
                    "root_cause": RootCause(
                        mechanism=RootCauseMechanism.MISSING_OUTPUT, hypothesis=d.description
                    ),
                },
                FindingSource.DETERMINISTIC,
                case,
                failed_criteria,
                product,
            )
            for d in deterministic
        ]
        goal = self._goal(case, general)
        if goal is not None and goal.status in GOAL_PROBLEM_SEVERITY:
            # The goal is not met: a finding of its own (never a metric score), so the verdict
            # and the bug candidate see it like any other problem (D243).
            findings.append(
                self._finding(
                    {
                        "category": GOAL_COMPLETION_CATEGORY,
                        "description": goal.reason,
                        "expected_behavior": "; ".join(
                            f"{c.id}: {c.requirement}"
                            for c in case.question.context.success_criteria
                        ),
                        "observed_behavior": goal.evidence or goal.reason,
                        "evidence": goal.evidence,
                        "severity": GOAL_PROBLEM_SEVERITY[goal.status],
                        # The judge's quote was checked against the response.
                        "evidence_status": EvidenceStatus.SUPPORTED,
                        "root_cause": RootCause(
                            mechanism=GOAL_ROOT_CAUSE[goal.status], hypothesis=goal.reason
                        ),
                    },
                    FindingSource.LLM,
                    case,
                    failed_criteria,
                    product,
                )
            )
        lookups = judgements.reference_checks.lookup_trace if judgements.reference_checks else []
        exemptions: dict[str, list[str]] = {}
        by_metric: dict[str, list[tuple[str, str, bool, str | None]]] = {}
        for finding in findings:
            if finding.exempted_by_id is not None:
                exemptions.setdefault(finding.category, []).append(finding.exempted_by_id)
            elif finding.exempted_by is None:
                by_metric.setdefault(finding.category, []).append(
                    (
                        finding.finding_id,
                        finding.description,
                        finding.unresolved,
                        finding.evidence,
                    )
                )
        response = case.execution.response
        assert isinstance(response, AnsweredResponse | DeniedResponse)
        metrics = trace_metrics(
            metrics,
            TraceInputs(
                question=case.question,
                definitions=self._definitions,
                guidance_refs=self._guidance_refs,
                general_scores=general_scores,
                retrieval=retrieval.draft if retrieval is not None else None,
                rule_texts=rule_texts(lookups),
                exemptions=exemptions,
                findings=by_metric,
                evidence_texts=evidence_texts(response),
            ),
        )
        reference_checks = mark_used_lookups(judgements.reference_checks, metrics)
        summary = summarise(
            metrics,
            self._definitions,
            sql_parse=[
                (k, item.structure.parsed, item.structure.diagnostics)
                for k, item in enumerate(judgements.sql, start=1)
            ],
            reference_checks=reference_checks,
            notes=judge_limits,
            question=case.question,
        )
        decision = self._policy.decide(metrics, findings, self._definitions, goal)
        verdict = decision.verdict
        groups = bug_groups(
            findings, decision.failing, decision.confirmed, self._subject_resolver(case)
        )
        result = EvaluationResult(
            **ids,
            response_classification=classification,
            classification_source=source,
            classification_note=classification_note,
            verdict=verdict,
            verdict_reason=_with_unresolved(decision.reason, summary.unresolved),
            verdict_policy=self._policy.name,
            evaluation_status=EvaluationStatus.EVALUATED,
            metrics=metrics,
            findings=findings,
            llm_verdict=general.draft.overall_verdict if general and general.draft else None,
            goal_completion=goal,
            notes=list(dict.fromkeys(notes)),
            reference_checks=reference_checks,
            limitations=summary.limitations,
            failed_requirements=summary.failed,
            unresolved_requirements=summary.unresolved,
        )
        log_evaluation(
            result,
            case.question,
            selected=self._config.selected_metric_ids,
            evidence_preparation=_evidence_preparation(case, judgements),
            invocations=self._invocations(case, judgements),
        )
        return result, groups

    def _invocations(self, case: EvaluationCase, judgements: _Judgements) -> list[InvocationRecord]:
        records: list[InvocationRecord] = []
        general, data = judgements.general, judgements.general_input
        if general is not None and data is not None:
            for call in range(1, general.calls + 1):
                records.append(
                    InvocationRecord(
                        invocation_id=f"{case.question.question_id}:general:{call}",
                        judge="general",
                        metrics=[d.metric_id for d in data.definitions],
                        attempts=general.attempts[call - 1 : call],
                        evidence_supplied=general.evidence_ids
                        if call == general.calls
                        else data.registry().ids(),
                        guidance_supplied=[g for g, ref in self._guidance_refs.items() if ref.user],
                        template_fingerprint=TEMPLATE_FINGERPRINT,
                        error=general.error if call == general.calls else None,
                    )
                )
        if judgements.retrieval is not None:
            records.append(
                InvocationRecord(
                    invocation_id=f"{case.question.question_id}:retrieval:1",
                    judge="retrieval_grounding",
                    metrics=[
                        d.metric_id for d in [*judgements.plan.retrieval, *judgements.plan.computed]
                    ],
                    evidence_supplied=[
                        f"CHUNK:{c.chunk_id}" for c in judgements.evidence.context.chunks
                    ],
                    guidance_supplied=[g for g, ref in self._guidance_refs.items() if ref.user],
                    error=judgements.retrieval.error,
                )
            )
        return records

    def _subject_resolver(self, case: EvaluationCase) -> SubjectResolver:
        """A root-cause subject counts for grouping only when it is verifiable: found in the
        response, the question or the cheat sheet (a dotted name part by part), or a trace tag
        of this question. Catalog tables / columns are canonicalised (namespace dropped)."""
        response = case.execution.response
        assert isinstance(response, AnsweredResponse | DeniedResponse)
        pack = case.pack
        haystack = "\n".join(
            [
                response_text(response),
                case.question.utterance,
                pack.model_dump_json() if pack is not None else "",
            ]
        )
        catalog = self._catalog
        # A trace tag (TABLE-RULE-01.01, COLUMN-02.03) is numbered per pack: read the item.
        tagged = {item.tag.casefold(): item.text for item in trace_catalog(case.question, pack)}

        def resolve(subject: str) -> str | None:
            text = subject.strip().strip("`'\"[] ")
            text = tagged.get(text.casefold(), text)
            if not text:
                return None
            parts = [p for p in text.split(".") if p]
            dotted = len(parts) > 1 and " " not in text
            if not (quoted(text, haystack) or (dotted and all(quoted(p, haystack) for p in parts))):
                return None
            if catalog is not None and " " not in text:
                table, _ = catalog.resolve_table(text)
                if table is not None:
                    return table.name.casefold()
                if dotted:
                    table, _ = catalog.resolve_table(".".join(parts[:-1]))
                    if table is not None:
                        return f"{table.name}.{parts[-1]}".casefold()
            return normalise(text)

        return resolve

    @staticmethod
    def _goal(case: EvaluationCase, general: GeneralJudgement | None) -> GoalCompletion | None:
        draft = general.draft.goal_completion if general and general.draft else None
        if draft is None:
            return None
        return GoalCompletion(
            status=draft.status,
            reason=draft.reason,
            evidence=draft.evidence,
            capability=case.scenario.capability.value,
            product_response_type=case.execution.response.product_response_type,
        )

    def _classification(
        self, case: EvaluationCase, general: GeneralJudgement | None
    ) -> tuple[ResponseClassification, ClassificationSource, str | None]:
        response = case.execution.response
        if isinstance(response, DeniedResponse):
            return (
                ResponseClassification.UNABLE_TO_ANSWER,
                ClassificationSource.TYPED_OUTCOME,
                None,
            )
        assert isinstance(response, AnsweredResponse)
        if response.response_classification is not None:
            return response.response_classification, ClassificationSource.ADAPTER, None
        draft = general.draft if general is not None else None
        if draft is not None and draft.response_classification is not None:
            return (
                ResponseClassification(draft.response_classification),
                ClassificationSource.GENERAL_JUDGE,
                draft.classification_note,
            )
        return ResponseClassification.UNKNOWN, ClassificationSource.UNRESOLVED, None

    @staticmethod
    def _judged(
        results: dict[str, MetricResult],
        definitions: Sequence[MetricDefinition],
        scores: dict[str, JudgedScore] | None,
        error: str | None,
        group: JudgeGroup,
        *,
        invalid: Mapping[str, list[str]] | None = None,
    ) -> None:
        for definition in definitions:
            problems = (invalid or {}).get(definition.metric_id)
            if problems:
                results[definition.metric_id] = MetricResult(
                    status=MetricStatus.EVALUATION_ERROR,
                    rationale="Not scored: the judge's output for this metric stayed invalid "
                    "after the retries (the other metrics were kept): " + "; ".join(problems),
                    judge_group=group,
                    failure_source=f"{group.value} judge",
                )
                continue
            judged = scores.get(definition.metric_id) if scores is not None else None
            if judged is None:
                results[definition.metric_id] = MetricResult(
                    status=MetricStatus.EVALUATION_ERROR,
                    rationale=error or "the judge returned no result for this metric",
                    judge_group=group,
                    failure_source=f"{group.value} judge",
                )
                continue
            details: dict[str, Any] = {}
            if judged.factual_verification is not None:
                details["factual_verification"] = judged.factual_verification.value
                if judged.factual_verification in (
                    FactualVerification.UNVERIFIED,
                    FactualVerification.PARTLY_VERIFIED,
                ):
                    details["limitation"] = (
                        "Factual correctness was not (fully) verified against trusted evidence; "
                        "the score reflects observable completeness and consistency only."
                    )
            results[definition.metric_id] = MetricResult(
                status=MetricStatus.INCONCLUSIVE if judged.score is None else MetricStatus.SCORED,
                score=None if judged.score is None else float(judged.score),
                rationale=judged.rationale,
                evidence=judged.evidence,
                what_worked=judged.what_worked,
                what_didnt=judged.what_didnt,
                judge_group=group,
                evidence_refs=judged.evidence_ids,
                details=details,
            )

    @staticmethod
    def _retrieval_details(
        results: dict[str, MetricResult],
        plan: MetricPlan,
        retrieval: RetrievalJudgement,
        evidence: EvaluationEvidence,
    ) -> None:
        draft = retrieval.draft
        for definition in plan.retrieval:
            result = results.get(definition.metric_id)
            if result is None or draft is None:
                continue
            details: dict[str, JsonValue] = dict(result.details)
            checklist: list[JsonValue] = [
                item.model_dump(mode="json") for item in draft.required_information
            ]
            if checklist and REQUIRED_INFORMATION_FIELD in {
                f.name for f in definition.question_expectation_schema
            }:
                details["required_information"] = checklist
                details["disclosure"] = CONTEXT_RECALL_DISCLOSURE
            if EvidenceKind.CITATIONS in definition.required_execution_evidence:
                details["citation_checks"] = list[JsonValue](
                    c.model_dump(mode="json") for c in draft.citation_checks
                )
            if EvidenceKind.RETRIEVED_CONTEXT in definition.required_execution_evidence:
                details["retrieved_chunks"] = len(evidence.context.chunks)
            results[definition.metric_id] = result.model_copy(update={"details": details})
        for definition in plan.computed:
            if draft is None:
                results[definition.metric_id] = MetricResult(
                    status=MetricStatus.EVALUATION_ERROR,
                    rationale=retrieval.error or "no chunk relevance labels",
                    judge_group=JudgeGroup.RETRIEVAL_GROUNDING,
                    failure_source="retrieval_grounding judge",
                )
                continue
            labels = {label.chunk_id: label.relevant for label in draft.chunk_relevance}
            rank = reciprocal_rank(evidence.context, labels)
            if rank is None:
                results[definition.metric_id] = MetricResult(
                    status=MetricStatus.EVIDENCE_UNAVAILABLE,
                    rationale="Ranked chunks with relevance labels are required; "
                    "chunk order is never assumed to be the retrieval rank.",
                    judge_group=JudgeGroup.RETRIEVAL_GROUNDING,
                    missing_requirement=EvidenceKind.RANKED_CONTEXT.value,
                )
                continue
            rationale = (
                f"First relevant chunk at rank {rank.first_relevant_rank}: reciprocal rank "
                f"1/{rank.first_relevant_rank}."
                if rank.first_relevant_rank is not None
                else "No retrieved chunk is relevant: reciprocal rank 0."
            )
            results[definition.metric_id] = MetricResult(
                status=MetricStatus.SCORED,
                score=rank.value,
                rationale=rationale,
                judge_group=JudgeGroup.RETRIEVAL_GROUNDING,
                evidence_refs=rank.relevant_chunk_ids,
                details={
                    "reciprocal_rank": rank.value,
                    "first_relevant_rank": rank.first_relevant_rank,
                    "relevant_chunk_ids": list[JsonValue](rank.relevant_chunk_ids),
                    "chunks_labelled": len(labels),
                },
            )

    def _finding(
        self,
        data: dict[str, Any],
        source: FindingSource,
        case: EvaluationCase,
        failed_criteria: dict[str, list[str]],
        product: ProductDescription,
    ) -> Finding:
        exempted = data.get("exempted_by")
        if exempted is not None:
            # The judge cites a guidance item id; the stored finding keeps its exact text.
            ref = self._guidance_refs.get(str(exempted).strip())
            if ref is not None:
                data = {**data, "exempted_by": ref.instruction, "exempted_by_id": ref.guidance_id}
        draft = Finding.model_validate(
            {"finding_id": "F_000", "source": source, "related_tags": [], **data}
        )
        tags = merge_finding_tags(
            list(data.get("related_tags", [])),
            link_finding_tags(draft, case.question, case.pack, failed_criteria),
            case.question,
            case.pack,
        )
        known = draft.matches_known_gap or matches_known_gap(
            f"{draft.category} {draft.description} {draft.observed_behavior}",
            product.known_limitations,
        )
        return draft.model_copy(
            update={
                "finding_id": self._ids.next(IdKind.FINDING),
                "related_tags": tags,
                "matches_known_gap": known,
            }
        )


def _check_links(case: EvaluationCase, reference: Reference | None) -> None:
    question, scenario, pack, execution = (
        case.question,
        case.scenario,
        case.pack,
        case.execution,
    )
    if question.is_legacy:
        raise ArtifactIntegrityError(
            LEGACY_QUESTION_MESSAGE.format(question_id=question.question_id)
        )
    problems = []
    if execution.question_id != question.question_id:
        problems.append(f"execution is for {execution.question_id}, not {question.question_id}")
    if question.scenario_id != scenario.id:
        problems.append(f"question belongs to {question.scenario_id}, not {scenario.id}")
    pack_id = pack.pack_id if pack else None
    if pack is None and question.reference_pack_id is not None:
        problems.append(
            f"{question.question_id} uses reference pack {question.reference_pack_id} but its "
            "evaluation_context has no reference_spec (regenerate the questions)"
        )
    elif question.reference_pack_id != pack_id:
        problems.append(f"question uses reference pack {question.reference_pack_id}, not {pack_id}")
    if question.mode is GenerationMode.GROUNDED:
        if reference is None:
            problems.append(f"{question.question_id} is grounded but no reference was given")
        elif reference.reference_id != scenario.reference_id:
            problems.append(
                f"scenario {scenario.id} was built from reference {scenario.reference_id}, "
                f"not {reference.reference_id}"
            )
    if problems:
        raise ArtifactIntegrityError("Mismatched evaluation inputs: " + "; ".join(problems))


def _with_unresolved(reason: str, unresolved: list[str]) -> str:
    """The verdict reason plus the unresolved checks: visible, the verdict is unchanged."""
    if not unresolved:
        return reason
    return f"{reason} Unresolved checks (verdict unchanged): {'; '.join(unresolved)}."


def _evidence_preparation(case: EvaluationCase, judgements: _Judgements) -> dict[str, Any]:
    """Section B of the EVALUATION TRACE: what code prepared before the judges (ids, parse and
    reconciliation outcomes, tool calls, context) - no raw payloads beyond the SQL clauses."""
    sql: list[dict[str, Any]] = []
    for k, item in enumerate(judgements.sql, start=1):
        s = item.structure
        entry: dict[str, Any] = {
            "artifact_id": f"SQL:{k}",
            "parse_status": "parsed" if s.parsed else "parse_failed",
            "parser_warnings": s.diagnostics,
        }
        if s.parsed:
            entry["parsed_structure"] = {
                "base_tables": s.base_tables,
                "base_columns": [
                    f"{c.table}.{c.column}" if c.table else c.column for c in s.base_columns
                ],
                **{
                    attribute: [
                        f"{sql_clause_id(k, clause, i)} {expression}"
                        for i, expression in enumerate(getattr(s, attribute), start=1)
                    ]
                    for clause, attribute in SQL_CLAUSES
                },
                "joins": [
                    f"{sql_clause_id(k, 'JOIN', i)} {j.kind} {j.target} {j.predicate or ''}".strip()
                    for i, j in enumerate(s.joins, start=1)
                ],
            }
        rec = item.reconciliation
        if rec is not None:
            entry["catalog_reconciliation"] = {
                "evidence_id": f"RECON:{k}",
                **{
                    status.value: [i.identifier for i in rec.with_status(status)]
                    for status in IdentifierStatus
                },
                "joins": [f"{j.left} = {j.right}: {j.status.value}" for j in rec.joins],
                "rules_not_in_pack": rec.rules_not_in_pack,
            }
        sql.append(entry)
    evidence = judgements.evidence
    trace = evidence.tool_trace
    return {
        "sql_artifacts": sql or "none",
        "tool_calls": [
            {
                "evidence_id": f"TOOL:{c.order}",
                "name": c.name,
                "operation": c.operation,
                "arguments": c.arguments,
                "result_available": c.result is not None,
                "error": c.error,
            }
            for c in trace
        ]
        if trace is not None
        else "no tool trace exposed",
        "retrieved_chunks": [f"CHUNK:{c.chunk_id}" for c in evidence.context.chunks],
        "missing_evidence": {
            metric_id: result.rationale
            for metric_id, result in judgements.plan.preset.items()
            if result.status is MetricStatus.EVIDENCE_UNAVAILABLE
        },
    }
