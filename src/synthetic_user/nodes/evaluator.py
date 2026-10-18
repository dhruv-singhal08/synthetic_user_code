"""Node 5: evaluate product executions (routing + assembly).

Routing by outcome (one LLM call per judged response, design v3):
- answered      -> all generic checks; grounded questions also get reference_alignment
                   against the scenario's Reference Pack.
- clarification -> generic checks with the clarification_handling focus.
- denied        -> verdict from capability_consistency only (was the denial justified?).
- error/timeout -> recorded_only, verdict None, no LLM call.

Artifact plugins (nodes/evaluators/plugins/) are disconnected for now: every product goes
through this one generic path.

Verdict: any failed check backed by a finding with evidence_status supported -> fail; else any
inconclusive applicable check (or a failed check without supported evidence) ->
inconclusive; else pass. not_applicable checks are ignored. The LLM's own opinion is stored
as llm_verdict and is never decisive.
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
from synthetic_user.llm.base import LLMClient
from synthetic_user.models.evaluation import (
    BugCandidate,
    EvaluationResult,
    EvaluationStatus,
    EvidenceStatus,
    Finding,
    FindingSource,
    GenericChecks,
    Severity,
    Verdict,
)
from synthetic_user.models.execution import ExecutionRecord
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluators.generic import (
    GenericEvaluator,
    GenericJudgement,
    generic_verdict,
    matches_known_gap,
)
from synthetic_user.prompts.response_evaluation import focus_for

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
        id_generator: SequentialIdGenerator | None = None,
    ) -> None:
        self._settings = settings
        self._log = logger
        self._ids = id_generator or SequentialIdGenerator()
        self._reference = reference
        self._generic = GenericEvaluator(llm, settings, logger)

    async def evaluate(
        self, case: EvaluationCase, product: ProductDescription, persona: Persona
    ) -> EvaluationResult:
        return (await self.evaluate_many([case], product, persona))[0]

    async def evaluate_many(
        self, cases: Sequence[EvaluationCase], product: ProductDescription, persona: Persona
    ) -> list[EvaluationResult]:
        for case in cases:
            _check_links(case, self._reference)
        semaphore = asyncio.Semaphore(self._settings.llm.max_concurrency)
        judgements = await asyncio.gather(
            *(self._judge(c, product, persona, semaphore) for c in cases)
        )
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
        persona: Persona,
        semaphore: asyncio.Semaphore,
    ) -> GenericJudgement | None:
        focus = focus_for(case.execution.response)
        if focus is None:
            return None  # error / timeout
        grounding = (
            (case.pack, self._reference)
            if case.pack is not None and self._reference is not None
            else None
        )
        async with semaphore:
            return await self._generic.judge(
                persona,
                product,
                case.scenario,
                case.question,
                case.execution.response,
                focus,
                grounding,
            )

    # --- phase 2: assembly (sequential, assigns ids) --------------------------------------

    def _assemble(
        self,
        case: EvaluationCase,
        judgement: GenericJudgement | None,
        product: ProductDescription,
    ) -> EvaluationResult:
        ids: dict[str, Any] = {
            "evaluation_id": self._ids.next(IdKind.EVALUATION),
            "question_id": case.question.question_id,
            "scenario_id": case.scenario.id,
            "execution_id": case.execution.execution_id,
            "product_outcome": case.execution.outcome,
            "mode": case.question.mode,
        }
        if judgement is None:
            return EvaluationResult(
                **ids, verdict=None, evaluation_status=EvaluationStatus.RECORDED_ONLY
            )

        fields: dict[str, Any] = {}
        findings: list[Finding] = []
        verdict = Verdict.INCONCLUSIVE
        draft = judgement.draft
        if draft is not None:
            checks = GenericChecks(**{n: j.status for n, j in draft.judgements().items()})
            findings = [
                Finding(
                    finding_id=self._ids.next(IdKind.FINDING),
                    source=FindingSource.LLM,
                    **{
                        **f.model_dump(),
                        "matches_known_gap": f.matches_known_gap
                        or matches_known_gap(
                            f"{f.category} {f.description} {f.observed_behavior}",
                            product.known_gaps,
                        ),
                    },
                )
                for f in draft.findings
            ]
            fields = {
                "checks": checks,
                "check_rationale": {n: j.rationale for n, j in draft.judgements().items()},
                "criteria_results": draft.criteria,
                "clarification_evaluation": draft.clarification,
            }
            # denied: only capability_consistency is judged, so the same rule yields the
            # verdict "was the denial justified?" (the other checks are not_applicable).
            verdict = generic_verdict(checks, findings)
        return EvaluationResult(
            **ids,
            verdict=verdict,
            evaluation_status=EvaluationStatus.EVALUATED,
            findings=findings,
            bug_candidates=self._bug_candidates(verdict, findings, case, product),
            llm_verdict=draft.overall_verdict if draft else None,
            notes=judgement.notes,
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
