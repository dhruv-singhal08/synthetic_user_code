"""Node 5: evaluate product executions.

Routing by outcome:
- sql_generated  -> deterministic SQL rules (nodes/sql_rules.py), then Stage B (LLM review)
                    unless the rules already decided (parse failure / non-SELECT).
                    Full Stage A static analysis is not implemented (D085): sql_analysis=None.
- clarification  -> ClarificationEvaluator.
- denied / error / timeout -> recorded_only, verdict None, no findings.

Deterministic rules override the LLM; the LLM's own verdict is kept in llm_verdict.
LLM calls run concurrently; ids (EVAL/F/BUG) are assigned afterwards in input order.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import ArtifactIntegrityError, LLMOutputValidationError
from synthetic_user.core.identifiers import IdKind, SequentialIdGenerator
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.evaluation import (
    BugCandidate,
    CheckStatus,
    EvaluationChecks,
    EvaluationResult,
    EvaluationStatus,
    EvidenceStatus,
    Finding,
    FindingSource,
    Severity,
    SqlEvaluationDraft,
    Verdict,
)
from synthetic_user.models.execution import (
    ClarificationResult,
    ExecutionResult,
    SqlGeneratedResult,
)
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import ReferenceCatalog, ReferenceSnapshot
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.clarification_evaluator import (
    ClarificationEvaluator,
    ClarificationOutcome,
)
from synthetic_user.nodes.sql_rules import RuleFinding, SqlRuleResult, apply_sql_rules
from synthetic_user.prompts.sql_evaluation import build_sql_evaluation_prompt

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

_SEVERITY_RANK = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
_GAP_STOPWORDS = frozenset(
    {"when", "with", "from", "that", "this", "into", "over", "than", "then", "they", "them"}
    | {"have", "been", "being", "were", "will", "would", "should", "could", "does", "done"}
    | {"link", "data", "rows", "value", "values", "product", "query", "queries", "user"}
)


@dataclass(frozen=True)
class EvaluationCase:
    question: QuestionCase
    scenario: Scenario
    snapshot: ReferenceSnapshot
    execution: ExecutionResult


@dataclass
class _Judgement:
    case: EvaluationCase
    rules: SqlRuleResult | None = None
    draft: SqlEvaluationDraft | None = None
    clarification: ClarificationOutcome | None = None
    notes: list[str] = field(default_factory=list)


# --- pure helpers -------------------------------------------------------------------------


def check_sql_evaluation(draft: SqlEvaluationDraft, sql: str) -> list[str]:
    """Problems with an LLM SQL evaluation (empty list = acceptable)."""
    errors: list[str] = []
    for i, finding in enumerate(draft.findings, start=1):
        if finding.sql_evidence and not _in_sql(finding.sql_evidence, sql):
            errors.append(f"finding {i}: sql_evidence {finding.sql_evidence!r} is not in the SQL")
    failed = [name for name, status in draft.checks if status is CheckStatus.FAIL]
    for name in failed:
        if not any(
            f.category == name and f.sql_evidence and _in_sql(f.sql_evidence, sql)
            for f in draft.findings
        ):
            errors.append(
                f"check {name} is fail but no finding has category {name!r} with sql_evidence "
                "copied from the SQL"
            )
    if failed and draft.verdict is not Verdict.FAIL:
        errors.append(f"checks {failed} failed, so verdict must be fail")
    if draft.verdict is Verdict.FAIL and not failed:
        errors.append("verdict fail requires at least one failed check")
    return errors


def final_verdict(checks: EvaluationChecks, findings: Sequence[Finding]) -> Verdict:
    """Any failed check backed by a supported finding -> fail; else any failed (unsupported)
    or inconclusive check -> inconclusive; else pass."""
    supported = {f.category for f in findings if f.evidence_status is EvidenceStatus.SUPPORTED}
    statuses = dict(checks)
    failed = [name for name, status in statuses.items() if status is CheckStatus.FAIL]
    if any(name in supported for name in failed):
        return Verdict.FAIL
    if failed or CheckStatus.INCONCLUSIVE in statuses.values():
        return Verdict.INCONCLUSIVE
    return Verdict.PASS


def matches_known_gap(text: str, known_gaps: Sequence[str]) -> bool:
    """True if at least two keywords of a known gap (or its only keyword) occur in text."""
    words = _keywords(text)
    for gap in known_gaps:
        gap_words = _keywords(gap)
        if gap_words and len(gap_words & words) >= min(2, len(gap_words)):
            return True
    return False


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) >= 4} - _GAP_STOPWORDS


def _all_checks(status: CheckStatus, **overrides: CheckStatus) -> EvaluationChecks:
    values = dict.fromkeys(EvaluationChecks.model_fields, status)
    values.update(overrides)
    return EvaluationChecks.model_validate(values)


def _in_sql(fragment: str, sql: str) -> bool:
    return _normalise(fragment) in _normalise(sql)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


# --- evaluator ----------------------------------------------------------------------------


class SqlEvaluator:
    def __init__(
        self,
        llm: LLMClient,
        settings: AppConfig,
        logger: Logger,
        *,
        catalog: ReferenceCatalog,
        id_generator: SequentialIdGenerator,
        clarification_evaluator: ClarificationEvaluator | None = None,
    ) -> None:
        self._llm = llm
        self._settings = settings
        self._log = logger
        self._catalog = catalog
        self._ids = id_generator
        self._clarification = clarification_evaluator or ClarificationEvaluator(
            llm, settings, logger
        )

    async def evaluate(
        self,
        question: QuestionCase,
        scenario: Scenario,
        snapshot: ReferenceSnapshot,
        product: ProductDescription,
        execution: ExecutionResult,
    ) -> EvaluationResult:
        case = EvaluationCase(question, scenario, snapshot, execution)
        return (await self.evaluate_many([case], product))[0]

    async def evaluate_many(
        self, cases: Sequence[EvaluationCase], product: ProductDescription
    ) -> list[EvaluationResult]:
        for case in cases:
            _check_links(case)
        semaphore = asyncio.Semaphore(self._settings.llm.max_concurrency)
        judgements = await asyncio.gather(*(self._judge(c, product, semaphore) for c in cases))
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
        self, case: EvaluationCase, product: ProductDescription, semaphore: asyncio.Semaphore
    ) -> _Judgement:
        execution = case.execution
        if isinstance(execution, SqlGeneratedResult):
            rules = apply_sql_rules(execution.sql, self._settings.sql.dialect, self._catalog)
            judgement = _Judgement(case, rules=rules)
            if rules.blocks_llm:
                return judgement
            system, user = build_sql_evaluation_prompt(
                case.question, case.scenario, case.snapshot, product, execution.sql, None
            )
            sql = execution.sql
            async with semaphore:
                try:
                    judgement.draft = await generate_structured(
                        self._llm,
                        system,
                        user,
                        SqlEvaluationDraft,
                        max_retries=self._settings.llm.max_validation_retries,
                        post_validate=lambda d: check_sql_evaluation(d, sql),
                    )
                except LLMOutputValidationError as exc:
                    self._log.warning("%s: SQL evaluation failed", case.question.question_id)
                    judgement.notes.append(
                        f"SQL evaluation failed after {exc.attempts} attempt(s): "
                        + "; ".join(exc.errors)
                    )
            return judgement
        if isinstance(execution, ClarificationResult):
            async with semaphore:
                outcome = await self._clarification.evaluate(
                    case.question, case.scenario, case.snapshot, product, execution.message
                )
            return _Judgement(case, clarification=outcome, notes=list(outcome.notes))
        return _Judgement(case)

    # --- phase 2: assembly (sequential, assigns ids) --------------------------------------

    def _assemble(self, judgement: _Judgement, product: ProductDescription) -> EvaluationResult:
        case = judgement.case
        ids: dict[str, Any] = {
            "evaluation_id": self._ids.next(IdKind.EVALUATION),
            "question_id": case.question.question_id,
            "scenario_id": case.scenario.id,
            "execution_id": case.execution.execution_id,
            "product_outcome": case.execution.outcome,
        }
        conflicts = [
            f"cheat sheet references {ref!r}, which is not in the catalog"
            for ref in case.snapshot.missing_references
        ]
        if isinstance(case.execution, SqlGeneratedResult) and judgement.rules is not None:
            verdict, checks, findings, notes = self._sql_outcome(
                judgement, case.execution.sql, product
            )
            conflicts = [
                *(judgement.draft.cheat_sheet_conflicts if judgement.draft else []),
                *conflicts,
            ]
            llm_verdict = judgement.draft.verdict if judgement.draft else None
            clarification = None
        elif judgement.clarification is not None:
            verdict = judgement.clarification.verdict
            checks, llm_verdict = None, None
            clarification = judgement.clarification.evaluation
            findings = self._clarification_findings(judgement.clarification, case, product)
            notes = judgement.notes
        else:
            return EvaluationResult(
                **ids, verdict=None, evaluation_status=EvaluationStatus.RECORDED_ONLY
            )

        bugs = self._bug_candidates(verdict, findings, case)
        return EvaluationResult(
            **ids,
            verdict=verdict,
            evaluation_status=EvaluationStatus.EVALUATED,
            checks=checks,
            findings=findings,
            clarification_evaluation=clarification,
            bug_candidates=bugs,
            sql_analysis=None,
            llm_verdict=llm_verdict,
            cheat_sheet_conflicts=conflicts,
            notes=notes,
        )

    def _sql_outcome(
        self, judgement: _Judgement, sql: str, product: ProductDescription
    ) -> tuple[Verdict, EvaluationChecks, list[Finding], list[str]]:
        rules = judgement.rules
        assert rules is not None  # guaranteed by caller
        notes = list(judgement.notes)

        if rules.parse_error is not None:
            finding = self._deterministic(
                RuleFinding(
                    category="syntax_error",
                    description=f"SQL does not parse: {rules.parse_error}",
                    expected_behavior="a single valid SELECT statement",
                    observed_behavior="unparseable SQL",
                    sql_evidence=sql.strip()[:500],
                    severity=Severity.HIGH,
                ),
                product,
            )
            checks = _all_checks(CheckStatus.INCONCLUSIVE, sql_syntax=CheckStatus.FAIL)
            return Verdict.FAIL, checks, [finding], notes
        if rules.non_select is not None:
            finding = self._deterministic(
                RuleFinding(
                    category="non_select_statement",
                    description=f"product generated a {rules.non_select} statement",
                    expected_behavior="a read-only SELECT statement",
                    observed_behavior=f"{rules.non_select} statement",
                    sql_evidence=sql.strip()[:500],
                    severity=Severity.CRITICAL,
                ),
                product,
            )
            checks = _all_checks(
                CheckStatus.NOT_APPLICABLE,
                intent_alignment=CheckStatus.FAIL,
                sql_syntax=CheckStatus.PASS,
            )
            return Verdict.FAIL, checks, [finding], notes

        draft = judgement.draft
        checks = draft.checks if draft else _all_checks(CheckStatus.INCONCLUSIVE)
        findings = [
            Finding(
                finding_id=self._ids.next(IdKind.FINDING),
                source=FindingSource.LLM,
                **{
                    **f.model_dump(),
                    "matches_known_gap": f.matches_known_gap
                    or matches_known_gap(_finding_text(f.model_dump()), product.known_gaps),
                },
            )
            for f in (draft.findings if draft else [])
        ]
        overrides: dict[str, CheckStatus] = {}
        for rule in rules.findings:
            findings.append(self._deterministic(rule, product))
            overrides[rule.category] = CheckStatus.FAIL
        if overrides:
            checks = checks.model_copy(update=overrides)

        verdict = final_verdict(checks, findings)
        if rules.unverifiable_tables and verdict is Verdict.PASS:
            verdict = Verdict.INCONCLUSIVE
            notes.append(
                f"tables not in catalog {rules.unverifiable_tables}: cannot verify, "
                "so pass downgraded to inconclusive"
            )
        return verdict, checks, findings, notes

    def _deterministic(self, rule: RuleFinding, product: ProductDescription) -> Finding:
        text = _finding_text(rule.__dict__)
        return Finding(
            finding_id=self._ids.next(IdKind.FINDING),
            category=rule.category,
            description=rule.description,
            expected_behavior=rule.expected_behavior,
            observed_behavior=rule.observed_behavior,
            sql_evidence=rule.sql_evidence,
            severity=rule.severity,
            confidence=None,
            evidence_status=EvidenceStatus.SUPPORTED,
            source=FindingSource.DETERMINISTIC,
            matches_known_gap=matches_known_gap(text, product.known_gaps),
        )

    def _clarification_findings(
        self, outcome: ClarificationOutcome, case: EvaluationCase, product: ProductDescription
    ) -> list[Finding]:
        if outcome.verdict is not Verdict.FAIL or outcome.evaluation is None:
            return []
        assert isinstance(case.execution, ClarificationResult)
        description = outcome.evaluation.reason
        return [
            Finding(
                finding_id=self._ids.next(IdKind.FINDING),
                category="clarification",
                description=description,
                expected_behavior="answer directly, or ask only for information that is "
                "missing and needed",
                observed_behavior=f"product asked: {case.execution.message}",
                severity=Severity.MEDIUM,
                evidence_status=EvidenceStatus.SUPPORTED,
                source=FindingSource.LLM,
                matches_known_gap=matches_known_gap(description, product.known_gaps),
            )
        ]

    def _bug_candidates(
        self, verdict: Verdict, findings: Sequence[Finding], case: EvaluationCase
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
                question_id=case.question.question_id,
                scenario_id=case.scenario.id,
                execution_id=case.execution.execution_id,
                finding_ids=[f.finding_id for f in qualifying],
            )
        ]


def _finding_text(fields: dict[str, object]) -> str:
    keys = ("category", "description", "expected_behavior", "observed_behavior", "sql_evidence")
    return " ".join(str(fields.get(k) or "") for k in keys)


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
    if question.reference_snapshot_id != snapshot.snapshot_id:
        problems.append(
            f"question uses snapshot {question.reference_snapshot_id}, not {snapshot.snapshot_id}"
        )
    if problems:
        raise ArtifactIntegrityError("Mismatched evaluation inputs: " + "; ".join(problems))
