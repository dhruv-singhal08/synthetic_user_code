"""reference_coverage: the grounded check used when no artifact plugin applies to a grounded
answer (e.g. a chatbot with a cheat sheet). One LLM call: does the answer cover the
cheat-sheet entities / metrics? No SQL logic. Not listed in evaluation.plugins - the
evaluator uses it as the grounded fallback.
"""

from __future__ import annotations

import logging
import re

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import LLMOutputValidationError
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.evaluation import (
    CheckStatus,
    FindingSource,
    ReferenceCoverageDraft,
    Verdict,
)
from synthetic_user.models.execution import ProductResponse
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
from synthetic_user.prompts.reference_coverage import (
    build_reference_coverage_prompt,
    coverage_items,
)
from synthetic_user.prompts.response_evaluation import response_text

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

CHECK = "reference_coverage"


def check_coverage(draft: ReferenceCoverageDraft, items: list[str], answer: str) -> list[str]:
    errors: list[str] = []
    listed = [*draft.covered, *draft.missing]
    unknown = [i for i in listed if i not in items]
    absent = [i for i in items if i not in listed]
    twice = sorted({i for i in listed if listed.count(i) > 1})
    if unknown or absent or twice:
        errors.append(
            f"covered + missing must list each reference item once: unknown {unknown}, "
            f"not listed {absent}, repeated {twice}"
        )
    judgement = draft.reference_coverage
    if judgement.status is CheckStatus.FAIL and not any(
        f.category == CHECK for f in draft.findings
    ):
        errors.append("reference_coverage is fail but no finding has category 'reference_coverage'")
    quotes = [judgement.evidence, *(f.evidence for f in draft.findings)]
    for quote in quotes:
        if quote and _norm(quote) not in _norm(answer):
            errors.append(f"evidence {quote!r} is not in the answer")
    for finding in draft.findings:
        if finding.category != CHECK:
            errors.append(f"finding category must be {CHECK!r}, not {finding.category!r}")
    return errors


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


class ReferenceCoveragePlugin:
    name = CHECK

    def __init__(self, llm: LLMClient, settings: AppConfig, logger: Logger) -> None:
        self._llm = llm
        self._settings = settings
        self._log = logger

    def applies(
        self, response: ProductResponse, question: QuestionCase, scenario: Scenario
    ) -> bool:
        return is_grounded_answer(response, question, scenario)

    async def evaluate(self, ctx: GroundedContext) -> PluginResult:
        sheet = ctx.scenario.reference_cheat_sheet
        assert sheet is not None  # guaranteed by applies()
        items = coverage_items(sheet)
        answer = response_text(ctx.response)
        system, user = build_reference_coverage_prompt(ctx.question, sheet, ctx.response)
        try:
            draft = await generate_structured(
                self._llm,
                system,
                user,
                ReferenceCoverageDraft,
                max_retries=self._settings.llm.max_validation_retries,
                post_validate=lambda d: check_coverage(d, items, answer),
            )
        except LLMOutputValidationError as exc:
            self._log.warning("%s: reference coverage failed", ctx.question.question_id)
            return PluginResult(
                checks={CHECK: CheckStatus.INCONCLUSIVE},
                findings=[],
                analysis=None,
                verdict=Verdict.INCONCLUSIVE,
                notes=[
                    f"reference coverage failed after {exc.attempts} attempt(s): "
                    + "; ".join(exc.errors)
                ],
            )
        findings = [
            PluginFinding(
                **f.model_dump(exclude={"matches_known_gap"}),
                source=FindingSource.LLM,
                matches_known_gap=f.matches_known_gap
                or matches_known_gap(
                    f"{f.description} {f.observed_behavior}", ctx.product.known_gaps
                ),
            )
            for f in draft.findings
        ]
        checks = {CHECK: draft.reference_coverage.status}
        return PluginResult(
            checks=checks,
            findings=findings,
            analysis={"covered": list(draft.covered), "missing": list(draft.missing)},
            verdict=checks_verdict(checks, findings),
            check_rationale={CHECK: draft.reference_coverage.rationale},
        )
