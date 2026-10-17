"""Node 5: evaluate product executions (routing + assembly).

Routing by outcome:
- answered      -> generic checks (both modes). Grounded mode only: every configured artifact
                   plugin that applies (evaluation.plugins, e.g. "sql") also runs; if none
                   applies, the reference_coverage check runs instead. Ungrounded: plugins
                   never run.
- clarification -> generic checks with the clarification_handling focus.
- denied        -> verdict from capability_consistency only (was the denial justified?).
- error/timeout -> recorded_only, verdict None, no LLM call.

Verdict (generic + plugin results combined, see ``combine``): any failed check backed by a
finding with evidence_status supported -> fail; else any inconclusive applicable check (or a
failed check without supported evidence) -> inconclusive; else pass. not_applicable checks
are ignored. Plugin overrides (e.g. SQL parse failure) can only make the verdict worse. The
LLMs' own opinions are stored as llm_verdict (worst of them) and are never decisive.
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
from dataclasses import dataclass, field
from typing import Any

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import ArtifactIntegrityError
from synthetic_user.core.identifiers import IdKind, SequentialIdGenerator
from synthetic_user.llm.base import LLMClient
from synthetic_user.models.evaluation import (
    BugCandidate,
    CheckStatus,
    EvaluationResult,
    EvaluationStatus,
    EvidenceStatus,
    Finding,
    FindingSource,
    GenericChecks,
    Severity,
    Verdict,
)
from synthetic_user.models.execution import AnsweredResponse, ExecutionRecord
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import ReferenceIndex, ReferenceSnapshot
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluators.generic import (
    GenericEvaluator,
    GenericJudgement,
    generic_verdict,
)
from synthetic_user.nodes.evaluators.plugins.base import (
    ArtifactPlugin,
    GroundedContext,
    PluginResult,
    matches_known_gap,
)
from synthetic_user.nodes.evaluators.plugins.reference_coverage import ReferenceCoveragePlugin
from synthetic_user.nodes.evaluators.plugins.registry import create_plugins
from synthetic_user.prompts.response_evaluation import focus_for

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

_SEVERITY_RANK = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
_VERDICT_RANK = {Verdict.PASS: 0, Verdict.INCONCLUSIVE: 1, Verdict.FAIL: 2}


@dataclass(frozen=True)
class EvaluationCase:
    question: QuestionCase
    scenario: Scenario
    snapshot: ReferenceSnapshot | None  # grounded questions only
    execution: ExecutionRecord


@dataclass
class _Judgement:
    case: EvaluationCase
    generic: GenericJudgement | None = None
    plugins: list[tuple[str, PluginResult]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def worse(a: Verdict, b: Verdict) -> Verdict:
    return a if _VERDICT_RANK[a] >= _VERDICT_RANK[b] else b


def combine(verdicts: Sequence[Verdict]) -> Verdict:
    """Combined verdict of generic + plugin results: the worst (fail > inconclusive > pass).

    Each part applies the same check rule (generic_verdict / plugin checks_verdict), so this
    equals applying "failed check with supported evidence -> fail, else any inconclusive ->
    inconclusive, else pass" to all checks at once, plus plugin overrides."""
    result = Verdict.PASS
    for verdict in verdicts:
        result = worse(result, verdict)
    return result


def combine_opinions(opinions: Sequence[Verdict | None]) -> Verdict | None:
    """llm_verdict: the worst of the LLMs' own stated verdicts (None if none was given)."""
    given = [o for o in opinions if o is not None]
    return combine(given) if given else None


class Evaluator:
    def __init__(
        self,
        llm: LLMClient,
        settings: AppConfig,
        logger: Logger,
        *,
        index: ReferenceIndex | None,
        id_generator: SequentialIdGenerator | None = None,
    ) -> None:
        self._settings = settings
        self._log = logger
        self._ids = id_generator or SequentialIdGenerator()
        self._index = index
        self._generic = GenericEvaluator(llm, settings, logger)
        self._plugins: list[ArtifactPlugin] = create_plugins(llm, settings, logger)
        self._coverage = ReferenceCoveragePlugin(llm, settings, logger)

    async def evaluate(
        self, case: EvaluationCase, product: ProductDescription, persona: Persona
    ) -> EvaluationResult:
        return (await self.evaluate_many([case], product, persona))[0]

    async def evaluate_many(
        self, cases: Sequence[EvaluationCase], product: ProductDescription, persona: Persona
    ) -> list[EvaluationResult]:
        for case in cases:
            _check_links(case)
        semaphore = asyncio.Semaphore(self._settings.llm.max_concurrency)
        judgements = await asyncio.gather(
            *(self._judge(c, product, persona, semaphore) for c in cases)
        )
        results = [self._assemble(j, product) for j in judgements]
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
    ) -> _Judgement:
        judgement = _Judgement(case)
        focus = focus_for(case.execution.response)
        if focus is None:
            return judgement  # error / timeout
        async with semaphore:
            judgement.generic = await self._generic.judge(
                persona, product, case.scenario, case.question, case.execution.response, focus
            )
            for plugin in self._grounded_plugins(case):
                ctx = self._context(case, product)
                judgement.plugins.append((plugin.name, await plugin.evaluate(ctx)))
        return judgement

    def _grounded_plugins(self, case: EvaluationCase) -> list[ArtifactPlugin]:
        """Grounded mode only: configured plugins that apply, else reference_coverage."""
        if case.question.mode is not GenerationMode.GROUNDED:
            return []
        args = (case.execution.response, case.question, case.scenario)
        applicable = [p for p in self._plugins if p.applies(*args)]
        if applicable:
            return applicable
        return [self._coverage] if self._coverage.applies(*args) else []

    def _context(self, case: EvaluationCase, product: ProductDescription) -> GroundedContext:
        response = case.execution.response
        if (
            self._index is None
            or case.snapshot is None
            or not isinstance(response, AnsweredResponse)
        ):
            raise ArtifactIntegrityError(
                f"{case.question.question_id} is grounded but no reference index / snapshot "
                "is available for grounded checks"
            )
        return GroundedContext(
            question=case.question,
            scenario=case.scenario,
            snapshot=case.snapshot,
            index=self._index,
            product=product,
            response=response,
        )

    # --- phase 2: assembly (sequential, assigns ids) --------------------------------------

    def _assemble(self, judgement: _Judgement, product: ProductDescription) -> EvaluationResult:
        case = judgement.case
        ids: dict[str, Any] = {
            "evaluation_id": self._ids.next(IdKind.EVALUATION),
            "question_id": case.question.question_id,
            "scenario_id": case.scenario.id,
            "execution_id": case.execution.execution_id,
            "product_outcome": case.execution.outcome,
            "mode": case.question.mode,
        }
        generic = judgement.generic
        if generic is None:
            return EvaluationResult(
                **ids, verdict=None, evaluation_status=EvaluationStatus.RECORDED_ONLY
            )

        notes = [*judgement.notes, *generic.notes]
        fields: dict[str, Any] = {}
        findings: list[Finding] = []
        verdict: Verdict | None = Verdict.INCONCLUSIVE
        draft = generic.draft
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
            verdict = generic_verdict(checks, findings)

        # denied: only capability_consistency is judged, so the same rule yields the verdict
        # "was the denial justified?" (the other checks are not_applicable).
        opinions: list[Verdict | None] = [draft.overall_verdict if draft else None]
        assert verdict is not None
        if judgement.plugins:
            grounded: dict[str, CheckStatus] = {}
            rationale = dict(fields.get("check_rationale", {}))
            analyses: dict[str, Any] = {}
            conflicts: list[str] = []
            for name, result in judgement.plugins:
                verdict = worse(verdict, result.verdict)
                grounded.update(result.checks)
                rationale.update(result.check_rationale)
                if result.analysis is not None:
                    analyses[name] = result.analysis
                conflicts.extend(result.cheat_sheet_conflicts)
                notes = [*notes, *result.notes]
                opinions.append(result.llm_verdict)
                findings = [
                    *findings,
                    *(
                        Finding(finding_id=self._ids.next(IdKind.FINDING), **f.model_dump())
                        for f in result.findings
                    ),
                ]
            fields.update(
                grounded_checks=grounded,
                plugins_applied=[name for name, _ in judgement.plugins],
                plugin_analysis=analyses,
                cheat_sheet_conflicts=conflicts,
                check_rationale=rationale,
            )
        return EvaluationResult(
            **ids,
            verdict=verdict,
            evaluation_status=EvaluationStatus.EVALUATED,
            findings=findings,
            bug_candidates=self._bug_candidates(verdict, findings, case, product),
            llm_verdict=combine_opinions(opinions),
            notes=notes,
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


def _check_links(case: EvaluationCase) -> None:
    question, scenario, snapshot, execution = (
        case.question,
        case.scenario,
        case.snapshot,
        case.execution,
    )
    problems = []
    if execution.question_id != question.question_id:
        problems.append(f"execution is for {execution.question_id}, not {question.question_id}")
    if question.scenario_id != scenario.id:
        problems.append(f"question belongs to {question.scenario_id}, not {scenario.id}")
    snapshot_id = snapshot.snapshot_id if snapshot else None
    if question.reference_snapshot_id != snapshot_id:
        problems.append(
            f"question uses snapshot {question.reference_snapshot_id}, not {snapshot_id}"
        )
    if problems:
        raise ArtifactIntegrityError("Mismatched evaluation inputs: " + "; ".join(problems))
