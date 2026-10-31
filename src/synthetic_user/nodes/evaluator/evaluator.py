"""Node 5: evaluate product executions (routing + assembly).

Routing by outcome:
- answered -> one LLM judgement, then semantic correctness, rules adherence, schema
              faithfulness and the verdict computed by code (nodes/evaluators/generic.py).
              A scenario Reference Pack supplies only allowed table/column names, not
              factual answer ground truth.
- clarification / denied / error / timeout -> recorded_only, verdict None, no LLM call.

Artifact plugins (nodes/evaluators/plugins/) are disconnected for now: every product goes
through this one generic path. The LLM's own opinion is stored as llm_verdict and is never
decisive.
Bug candidates: one per failed evaluation, from its findings with severity >= medium and
evidence supported (most severe first), status needs_review, carrying question / scenario /
execution / finding ids, the reproduction utterance and the product_id.
LLM calls run concurrently (``llm.max_concurrency``); ids (EVAL/F/BUG) are assigned afterwards
in input order.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import ArtifactIntegrityError
from synthetic_user.core.identifiers import IdKind, SequentialIdGenerator
from synthetic_user.core.trace import trace_subject
from synthetic_user.evaluation_trace import criterion_tag, link_finding_tags, merge_finding_tags
from synthetic_user.llm.base import LLMClient
from synthetic_user.models.evaluation import (
    AgentEvaluationDraft,
    BugCandidate,
    CriterionResult,
    EvaluationDraft,
    EvaluationResult,
    EvaluationStatus,
    EvidenceStatus,
    Finding,
    FindingSource,
    Severity,
    Verdict,
)
from synthetic_user.models.evaluation_profile import AgentMetric, EvaluationProfile
from synthetic_user.models.execution import (
    AnsweredResponse,
    ConversationTurnKind,
    ExecutionRecord,
)
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario
from synthetic_user.no_sql import NO_SQL_REASON, no_sql_execution, product_expects_sql
from synthetic_user.nodes.evaluators.agent import (
    AgentEvaluator,
    AgentJudgement,
    compute_agent_metrics,
)
from synthetic_user.nodes.evaluators.generic import (
    GenericEvaluator,
    GenericJudgement,
    compute_metrics,
    matches_known_gap,
    metrics_verdict,
)

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

_SEVERITY_RANK = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}


@dataclass(frozen=True)
class EvaluationCase:
    question: QuestionCase
    scenario: Scenario
    pack: ReferencePack | None  # grounded questions only
    execution: ExecutionRecord


class Evaluator:
    def __init__(
        self,
        llm: LLMClient,
        settings: AppConfig,
        logger: Logger,
        *,
        reference: Reference | None,
        profile: EvaluationProfile = EvaluationProfile.NL2SQL,
        agent_metrics: Sequence[AgentMetric] | None = None,
        id_generator: SequentialIdGenerator | None = None,
    ) -> None:
        self._settings = settings
        self._log = logger
        self._ids = id_generator or SequentialIdGenerator()
        self._reference = reference
        self._profile = profile
        self._generic = GenericEvaluator(llm, settings, logger)
        self._agent = AgentEvaluator(llm, settings, logger, enabled_metrics=agent_metrics)

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
    ) -> GenericJudgement | AgentJudgement | None:
        response = case.execution.response
        if not isinstance(response, AnsweredResponse):
            return None  # clarification / denied / error / timeout: recorded only
        grounding = (
            (case.pack, self._reference)
            if case.pack is not None and self._reference is not None
            else None
        )
        async with semaphore:
            with trace_subject(case.question.question_id):
                evaluator = (
                    self._agent if self._profile is EvaluationProfile.AGENT else self._generic
                )
                return await evaluator.judge(
                    product,
                    case.scenario,
                    case.question,
                    response,
                    grounding,
                    [
                        turn
                        for turn in case.execution.turns
                        if turn.kind
                        in (
                            ConversationTurnKind.CLARIFICATION,
                            ConversationTurnKind.CLARIFICATION_ANSWER,
                        )
                    ],
                )

    # --- phase 2: assembly (sequential, assigns ids) --------------------------------------

    def _assemble(
        self,
        case: EvaluationCase,
        judgement: GenericJudgement | AgentJudgement | None,
        product: ProductDescription,
    ) -> EvaluationResult:
        ids: dict[str, Any] = {
            "evaluation_id": self._ids.next(IdKind.EVALUATION),
            "question_id": case.question.question_id,
            "scenario_id": case.scenario.id,
            "execution_id": case.execution.execution_id,
            "product_outcome": case.execution.outcome,
            "mode": case.question.mode,
            "evaluation_profile": self._profile,
        }
        if judgement is None:
            notes = (
                [NO_SQL_REASON]
                if self._profile is EvaluationProfile.NL2SQL
                and no_sql_execution(
                    case.execution,
                    sql_expected=product_expects_sql(product),
                )
                else []
            )
            return EvaluationResult(
                **ids,
                verdict=None,
                evaluation_status=EvaluationStatus.RECORDED_ONLY,
                notes=notes,
            )

        fields: dict[str, Any] = {}
        findings: list[Finding] = []
        verdict = Verdict.INCONCLUSIVE
        draft = judgement.draft
        if draft is not None:
            if self._profile is EvaluationProfile.AGENT:
                assert isinstance(judgement, AgentJudgement)
                assert isinstance(draft, AgentEvaluationDraft)
                assert judgement.plan is not None
                metrics = compute_agent_metrics(draft, judgement.plan)
                sql_generated = True
            else:
                assert isinstance(judgement, GenericJudgement)
                assert isinstance(draft, EvaluationDraft)
                schema_applicable = bool(
                    case.pack is not None
                    and any(table.name or table.columns for table in case.pack.tables)
                )
                rules_applicable = bool(
                    case.pack is not None
                    and (
                        case.pack.rules
                        or case.pack.joins
                        or any(table.rules for table in case.pack.tables)
                    )
                )
                sql_generated = not no_sql_execution(
                    case.execution,
                    sql_expected=product_expects_sql(product),
                )
                metrics = compute_metrics(
                    draft,
                    rules_applicable=rules_applicable,
                    schema_applicable=schema_applicable,
                    sql_generated=sql_generated,
                )
            criteria_results = [
                CriterionResult(tag=criterion_tag(index), **result.model_dump())
                for index, result in enumerate(draft.criteria, start=1)
            ]
            findings = [
                Finding(
                    finding_id=self._ids.next(IdKind.FINDING),
                    source=FindingSource.LLM,
                    **{
                        **f.model_dump(),
                        "related_tags": merge_finding_tags(
                            f.related_tags,
                            link_finding_tags(
                                f,
                                case.question,
                                case.pack,
                                criteria_results,
                            ),
                            case.question,
                            case.pack,
                        ),
                        "matches_known_gap": f.matches_known_gap
                        or matches_known_gap(
                            f"{f.category} {f.description} {f.observed_behavior}",
                            product.known_limitations,
                        ),
                    },
                )
                for f in draft.findings
            ]
            fields = {
                "metrics": metrics,
                "criteria_results": criteria_results,
            }
            verdict = Verdict.FAIL if not sql_generated else metrics_verdict(metrics, findings)
        notes = list(judgement.notes)
        if self._profile is EvaluationProfile.NL2SQL and no_sql_execution(
            case.execution, sql_expected=product_expects_sql(product)
        ):
            notes.append(NO_SQL_REASON)
        return EvaluationResult(
            **ids,
            verdict=verdict,
            evaluation_status=EvaluationStatus.EVALUATED,
            findings=findings,
            bug_candidates=self._bug_candidates(verdict, findings, case, product),
            llm_verdict=draft.overall_verdict if draft else None,
            notes=list(dict.fromkeys(notes)),
            **fields,
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
            ),
            key=lambda f: -_SEVERITY_RANK[f.severity],
        )
        if not qualifying:
            return []
        lead = qualifying[0]
        return [
            BugCandidate(
                bug_id=self._ids.next(IdKind.BUG),
                title=f"{lead.category}: {lead.description}"[:200],
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
