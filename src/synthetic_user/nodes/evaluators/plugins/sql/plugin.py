"""SQL plugin: grounded evaluation of a "sql" artifact.

applies(): answered + artifact kind "sql" + grounded. Stage A (analysis.py, SQLGlot AST)
feeds Stage B (LLM, prompts/sql_evaluation.py). Deterministic overrides:
- parse failure (incl. empty / several statements) -> sql_syntax fail, verdict fail, no LLM call;
- non-SELECT -> critical intent_alignment fail, verdict fail, no LLM call;
- unknown column on an index table -> column_selection fail (cannot pass);
- join contradicting a verified relationship -> join_correctness fail;
- tables not in the reference index -> a pass is downgraded to inconclusive.
The LLM's own verdict is kept as llm_verdict. SQL is never executed.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import LLMOutputValidationError
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.evaluation import (
    CheckStatus,
    EvidenceStatus,
    FindingSource,
    Severity,
    Verdict,
)
from synthetic_user.models.execution import AnsweredResponse, ProductResponse
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluators.plugins.base import (
    GroundedContext,
    PluginFinding,
    PluginResult,
    checks_verdict,
    is_grounded_answer,
    matches_known_gap,
)
from synthetic_user.nodes.evaluators.plugins.sql.analysis import (
    RuleFinding,
    StageAResult,
    analyze_sql,
)
from synthetic_user.nodes.evaluators.plugins.sql.models import SqlChecks, SqlEvaluationDraft
from synthetic_user.prompts.sql_evaluation import build_sql_evaluation_prompt

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

SQL_ARTIFACT_KIND = "sql"


def sql_of(response: ProductResponse) -> str | None:
    """The SQL artifact of an answered response, if any."""
    if not isinstance(response, AnsweredResponse):
        return None
    artifact = response.first_artifact(SQL_ARTIFACT_KIND)
    if artifact is None or not isinstance(artifact.content, str):
        return None
    return artifact.content


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


def final_verdict(checks: SqlChecks, findings: Sequence[PluginFinding]) -> Verdict:
    return checks_verdict(dict(checks), findings)


def all_checks(status: CheckStatus, **overrides: CheckStatus) -> SqlChecks:
    values = dict.fromkeys(SqlChecks.model_fields, status)
    values.update(overrides)
    return SqlChecks.model_validate(values)


def _in_sql(fragment: str, sql: str) -> bool:
    return _normalise(fragment) in _normalise(sql)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


class SqlPlugin:
    name = "sql"

    def __init__(self, llm: LLMClient, settings: AppConfig, logger: Logger) -> None:
        self._llm = llm
        self._settings = settings
        self._log = logger

    def applies(
        self, response: ProductResponse, question: QuestionCase, scenario: Scenario
    ) -> bool:
        return is_grounded_answer(response, question, scenario) and sql_of(response) is not None

    async def evaluate(self, ctx: GroundedContext) -> PluginResult:
        sql = sql_of(ctx.response)
        assert sql is not None  # guaranteed by applies()
        stage_a = analyze_sql(sql, self._settings.sql.dialect, ctx.index)
        draft: SqlEvaluationDraft | None = None
        notes: list[str] = []
        if not stage_a.blocks_llm:
            system, user = build_sql_evaluation_prompt(
                ctx.question, ctx.scenario, ctx.snapshot, ctx.product, sql, stage_a.analysis
            )
            try:
                draft = await generate_structured(
                    self._llm,
                    system,
                    user,
                    SqlEvaluationDraft,
                    max_retries=self._settings.llm.max_validation_retries,
                    post_validate=lambda d: check_sql_evaluation(d, sql),
                )
            except LLMOutputValidationError as exc:
                self._log.warning("%s: SQL evaluation failed", ctx.question.question_id)
                notes.append(
                    f"SQL evaluation failed after {exc.attempts} attempt(s): "
                    + "; ".join(exc.errors)
                )
        return self._result(stage_a, draft, notes, sql, ctx)

    def _result(
        self,
        stage_a: StageAResult,
        draft: SqlEvaluationDraft | None,
        notes: list[str],
        sql: str,
        ctx: GroundedContext,
    ) -> PluginResult:
        known_gaps = ctx.product.known_gaps
        analysis = stage_a.analysis.model_dump(mode="json")
        conflicts = [
            *(draft.cheat_sheet_conflicts if draft else []),
            *(
                f"cheat sheet references {ref!r}, which is not in the reference index"
                for ref in ctx.snapshot.missing_references
            ),
        ]
        llm_verdict = draft.verdict if draft else None

        def deterministic(rule: RuleFinding) -> PluginFinding:
            text = " ".join(
                (rule.category, rule.description, rule.observed_behavior, rule.sql_evidence)
            )
            return PluginFinding(
                category=rule.category,
                description=rule.description,
                expected_behavior=rule.expected_behavior,
                observed_behavior=rule.observed_behavior,
                evidence=rule.sql_evidence,
                severity=rule.severity,
                evidence_status=EvidenceStatus.SUPPORTED,
                source=FindingSource.DETERMINISTIC,
                matches_known_gap=matches_known_gap(text, known_gaps),
            )

        def result(
            verdict: Verdict, checks: SqlChecks, findings: list[PluginFinding]
        ) -> PluginResult:
            return PluginResult(
                checks=dict(checks),
                findings=findings,
                analysis=analysis,
                verdict=verdict,
                notes=notes,
                cheat_sheet_conflicts=conflicts,
                llm_verdict=llm_verdict,
            )

        if stage_a.parse_error is not None:
            finding = deterministic(
                RuleFinding(
                    category="syntax_error",
                    description=f"SQL does not parse: {stage_a.parse_error}",
                    expected_behavior="a single valid SELECT statement",
                    observed_behavior="unparseable SQL",
                    sql_evidence=sql.strip()[:500],
                    severity=Severity.HIGH,
                )
            )
            checks = all_checks(CheckStatus.INCONCLUSIVE, sql_syntax=CheckStatus.FAIL)
            return result(Verdict.FAIL, checks, [finding])
        if stage_a.non_select is not None:
            finding = deterministic(
                RuleFinding(
                    category="non_select_statement",
                    description=f"product generated a {stage_a.non_select} statement",
                    expected_behavior="a read-only SELECT statement",
                    observed_behavior=f"{stage_a.non_select} statement",
                    sql_evidence=sql.strip()[:500],
                    severity=Severity.CRITICAL,
                )
            )
            checks = all_checks(
                CheckStatus.NOT_APPLICABLE,
                intent_alignment=CheckStatus.FAIL,
                sql_syntax=CheckStatus.PASS,
            )
            return result(Verdict.FAIL, checks, [finding])

        checks = draft.checks if draft else all_checks(CheckStatus.INCONCLUSIVE)
        findings = [
            PluginFinding(
                category=f.category,
                description=f.description,
                expected_behavior=f.expected_behavior,
                observed_behavior=f.observed_behavior,
                evidence=f.sql_evidence,
                severity=f.severity,
                confidence=f.confidence,
                evidence_status=f.evidence_status,
                source=FindingSource.LLM,
                matches_known_gap=f.matches_known_gap
                or matches_known_gap(
                    " ".join(
                        (f.category, f.description, f.observed_behavior, f.sql_evidence or "")
                    ),
                    known_gaps,
                ),
            )
            for f in (draft.findings if draft else [])
        ]
        overrides: dict[str, CheckStatus] = {}
        for rule in stage_a.findings:
            findings.append(deterministic(rule))
            overrides[rule.category] = CheckStatus.FAIL
        if overrides:
            checks = checks.model_copy(update=overrides)

        verdict = final_verdict(checks, findings)
        if stage_a.unverifiable_tables and verdict is Verdict.PASS:
            verdict = Verdict.INCONCLUSIVE
            notes.append(
                f"tables not in the reference index {stage_a.unverifiable_tables}: cannot "
                "verify, so pass downgraded to inconclusive"
            )
        return result(verdict, checks, findings)
