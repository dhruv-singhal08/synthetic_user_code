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
Bug candidates: one per failed evaluation, from findings with severity >= medium and supported
evidence (never an ``unresolved`` finding). LLM calls run under ``llm.max_concurrency``; ids are
assigned afterwards in order.

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
from collections.abc import Sequence
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
    BugCandidate,
    ClassificationSource,
    CriterionResult,
    EvaluationResult,
    EvaluationStatus,
    EvidenceStatus,
    FactualVerification,
    Finding,
    FindingSource,
    JudgedScore,
    MetricResult,
    MetricStatus,
    Severity,
    Verdict,
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
    LookupResult,
    ReferenceCheckRecord,
)
from synthetic_user.models.metrics import (
    EvidenceKind,
    JudgeGroup,
    MetricDefinition,
    MetricOrigin,
    ReferenceAsset,
    ResolvedMetricConfig,
)
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
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
from synthetic_user.nodes.evaluator.general_judge import GeneralJudge, GeneralJudgement
from synthetic_user.nodes.evaluator.general_prompt import GeneralJudgeInput, SqlEvidence
from synthetic_user.nodes.evaluator.judge_common import matches_known_gap
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
    criterion_tag,
    link_finding_tags,
    merge_finding_tags,
)
from synthetic_user.nodes.evaluator.verdict import DEFAULT_VERDICT_POLICY, VerdictPolicy
from synthetic_user.references.catalog_index import CatalogIndex, LookupSession
from synthetic_user.sql.parser import SqlglotParser, SqlParser
from synthetic_user.sql.reconcile import evidence_requests, reconcile

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

MAX_BUG_TITLE_CHARS = 200
_SEVERITY_RANK = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
CONTEXT_RECALL_DISCLOSURE = (
    "Expectation-based coverage of the question's required-information checklist; "
    "not exhaustive recall over the full corpus."
)
UNKNOWN_AFTER_JUDGE_ERROR = (
    "Response classification unresolved: the General Judge call failed (see the error note)."
)
UNKNOWN_CLASSIFICATION_NOTE = (
    "Response classification unresolved: the adapter exposes no typed response kind and no "
    "General Judge call ran for this response."
)


@dataclass(frozen=True)
class EvaluationCase:
    question: QuestionCase
    scenario: Scenario
    pack: ReferencePack | None  # grounded questions only
    execution: ExecutionRecord


@dataclass
class _Judgements:
    plan: MetricPlan
    evidence: EvaluationEvidence
    general: GeneralJudgement | None = None
    retrieval: RetrievalJudgement | None = None
    notes: list[str] = field(default_factory=list)
    reference_checks: ReferenceCheckRecord | None = None


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
        verdict_policy: VerdictPolicy = DEFAULT_VERDICT_POLICY,
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
        self._policy = verdict_policy
        # Built once per catalog version (the reference is fixed for the run); None for
        # document collections and ungrounded runs.
        self._catalog = CatalogIndex.for_reference(
            reference, namespaces=settings.evaluation.catalog_namespaces
        )
        # Run-level guidance for every judge call (no extra call); None -> prompts unchanged.
        self._guidance = evaluation_guidance
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

    async def evaluate(self, case: EvaluationCase, product: ProductDescription) -> EvaluationResult:
        return (await self.evaluate_many([case], product))[0]

    async def evaluate_many(
        self, cases: Sequence[EvaluationCase], product: ProductDescription
    ) -> list[EvaluationResult]:
        for case in cases:
            _check_links(case, self._reference)
        semaphore = asyncio.Semaphore(self._settings.llm.max_concurrency)
        judgements = await asyncio.gather(*(self._judge(c, product, semaphore) for c in cases))
        results = [self._assemble(c, j, product) for c, j in zip(cases, judgements, strict=True)]
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
                guidance=self._guidance.text if self._guidance else None,
            )
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
                guidance=self._guidance.text if self._guidance else None,
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
    ) -> list[LookupResult]:
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
    ) -> EvaluationResult:
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
            return EvaluationResult(
                **ids,
                response_classification=(
                    ResponseClassification.CLARIFICATION if clarification else None
                ),
                classification_source=(
                    ClassificationSource.TYPED_OUTCOME if clarification else None
                ),
                verdict=None,
                evaluation_status=EvaluationStatus.RECORDED_ONLY,
                notes=notes,
            )

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
        if general is not None:
            if general.error:
                notes.append(general.error)
            self._judged(
                results,
                plan.general,
                general.draft.metrics if general.draft else None,
                general.error,
                JudgeGroup.GENERAL,
            )
        if retrieval is not None:
            if retrieval.error:
                notes.append(retrieval.error)
            self._judged(
                results,
                plan.retrieval,
                retrieval.draft.metrics if retrieval.draft else None,
                retrieval.error,
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
        criteria_results = (
            [
                CriterionResult(tag=criterion_tag(index), **result.model_dump())
                for index, result in enumerate(general.draft.criteria, start=1)
            ]
            if general is not None and general.draft is not None
            else []
        )
        drafts = [
            *(general.draft.findings if general and general.draft else []),
            *(retrieval.draft.findings if retrieval and retrieval.draft else []),
        ]
        findings = [
            self._finding(f.model_dump(), FindingSource.LLM, case, criteria_results, product)
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
                },
                FindingSource.DETERMINISTIC,
                case,
                criteria_results,
                product,
            )
            for d in deterministic
        ]
        verdict = self._policy.decide(metrics, findings, self._definitions)
        return EvaluationResult(
            **ids,
            response_classification=classification,
            classification_source=source,
            classification_note=classification_note,
            verdict=verdict,
            verdict_policy=self._policy.name,
            evaluation_status=EvaluationStatus.EVALUATED,
            metrics=metrics,
            criteria_results=criteria_results,
            findings=findings,
            bug_candidates=self._bug_candidates(verdict, findings, case, product),
            llm_verdict=general.draft.overall_verdict if general and general.draft else None,
            notes=list(dict.fromkeys(notes)),
            reference_checks=judgements.reference_checks,
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
    ) -> None:
        for definition in definitions:
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
        criteria_results: list[CriterionResult],
        product: ProductDescription,
    ) -> Finding:
        draft = Finding.model_validate(
            {"finding_id": "F_000", "source": source, "related_tags": [], **data}
        )
        tags = merge_finding_tags(
            list(data.get("related_tags", [])),
            link_finding_tags(draft, case.question, case.pack, criteria_results),
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

    def _bug_candidates(
        self,
        verdict: Verdict,
        findings: Sequence[Finding],
        case: EvaluationCase,
        product: ProductDescription,
    ) -> list[BugCandidate]:
        if verdict is not Verdict.FAIL:
            return []
        qualifying = sorted(
            (
                f
                for f in findings
                if _SEVERITY_RANK[f.severity] >= _SEVERITY_RANK[Severity.MEDIUM]
                and f.evidence_status is EvidenceStatus.SUPPORTED
                and not f.unresolved
            ),
            key=lambda f: -_SEVERITY_RANK[f.severity],
        )
        if not qualifying:
            return []
        lead = qualifying[0]
        return [
            BugCandidate(
                bug_id=self._ids.next(IdKind.BUG),
                title=f"{lead.category}: {lead.description}"[:MAX_BUG_TITLE_CHARS],
                description="; ".join(f.description for f in qualifying),
                reproduction_question=case.question.utterance,
                product_id=product.product_id,
                question_id=case.question.question_id,
                scenario_id=case.scenario.id,
                execution_id=case.execution.execution_id,
                finding_ids=[f.finding_id for f in qualifying],
            )
        ]


def _check_links(case: EvaluationCase, reference: Reference | None) -> None:
    question, scenario, pack, execution = (
        case.question,
        case.scenario,
        case.pack,
        case.execution,
    )
    problems = []
    if execution.question_id != question.question_id:
        problems.append(f"execution is for {execution.question_id}, not {question.question_id}")
    if question.scenario_id != scenario.id:
        problems.append(f"question belongs to {question.scenario_id}, not {scenario.id}")
    pack_id = pack.pack_id if pack else None
    if question.reference_pack_id != pack_id:
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
