"""Generic evaluation (all products, both modes): one LLM judgement of a product response.

GenericEvaluator returns the validated LLM draft (no ids); nodes/evaluator.py turns it into
GenericChecks, check rationale, criterion results and findings (source llm). Drafts are
checked before they are accepted (bounded retry via llm/structured.py):
- every failed check has a finding of that check's category; categories are generic checks;
- evidence quotes occur in the response (whitespace/case-insensitive);
- answer: one criterion result per success criterion, goal_completeness consistent with them,
  clarification_handling not_applicable;
- clarification: clarification detail present and consistent with clarification_handling,
  goal_completeness not_applicable;
- denial: only capability_consistency judged, everything else not_applicable.
After the last retry the evaluation is inconclusive with the errors as notes (never guessed).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import LLMOutputValidationError
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.evaluation import (
    CheckStatus,
    ClarificationEvaluation,
    ClarificationNecessity,
    ClarificationRelevance,
    CriterionStatus,
    EvidenceStatus,
    Finding,
    GenericChecks,
    GenericEvaluationDraft,
    Verdict,
)
from synthetic_user.models.execution import ProductResponse
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.scenario import Scenario
from synthetic_user.prompts.response_evaluation import (
    EvaluationFocus,
    build_response_evaluation_prompt,
    response_text,
)

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

CHECK_NAMES = tuple(GenericChecks.model_fields)
NA = CheckStatus.NOT_APPLICABLE


@dataclass
class GenericJudgement:
    focus: EvaluationFocus
    draft: GenericEvaluationDraft | None = None
    notes: list[str] = field(default_factory=list)


# --- pure helpers ---------------------------------------------------------------------------


def expected_goal_status(statuses: Sequence[CriterionStatus]) -> CheckStatus:
    if any(s in (CriterionStatus.NOT_MET, CriterionStatus.PARTLY) for s in statuses):
        return CheckStatus.FAIL
    if not statuses or any(s is CriterionStatus.UNVERIFIABLE for s in statuses):
        return CheckStatus.INCONCLUSIVE
    return CheckStatus.PASS


def expected_clarification_status(detail: ClarificationEvaluation) -> CheckStatus:
    if (
        detail.necessity is ClarificationNecessity.UNNECESSARY
        or detail.relevance is ClarificationRelevance.IRRELEVANT
        or detail.asks_for_given_info
    ):
        return CheckStatus.FAIL
    if detail.relevance is ClarificationRelevance.RELEVANT and detail.necessity in (
        ClarificationNecessity.NECESSARY,
        ClarificationNecessity.REASONABLE,
    ):
        return CheckStatus.PASS
    return CheckStatus.INCONCLUSIVE


def check_generic_evaluation(
    draft: GenericEvaluationDraft,
    focus: EvaluationFocus,
    criteria: Sequence[str],
    response: str,
) -> list[str]:
    """Problems with a generic evaluation draft (empty list = acceptable)."""
    errors: list[str] = []
    judgements = draft.judgements()
    for i, finding in enumerate(draft.findings, start=1):
        if finding.category not in CHECK_NAMES:
            errors.append(f"finding {i}: category {finding.category!r} is not one of {CHECK_NAMES}")
        if finding.evidence and not _quoted(finding.evidence, response):
            errors.append(f"finding {i}: evidence {finding.evidence!r} is not in the response")
    for result in draft.criteria:
        if result.evidence and not _quoted(result.evidence, response):
            errors.append(f"criterion {result.criterion!r}: evidence is not in the response")
    for name, judgement in judgements.items():
        if judgement.evidence and not _quoted(judgement.evidence, response):
            errors.append(f"{name}: evidence {judgement.evidence!r} is not in the response")
        if judgement.status is CheckStatus.FAIL and not any(
            f.category == name for f in draft.findings
        ):
            errors.append(f"check {name} is fail but no finding has category {name!r}")

    if focus is EvaluationFocus.DENIAL:
        for name, judgement in judgements.items():
            if name != "capability_consistency" and judgement.status is not NA:
                errors.append(f"denial: {name} must be not_applicable (only capability judged)")
        if judgements["capability_consistency"].status is NA:
            errors.append("denial: capability_consistency must be judged")
        if draft.criteria or draft.clarification is not None:
            errors.append("denial: criteria must be empty and clarification null")
        return errors

    if focus is EvaluationFocus.ANSWER:
        if draft.clarification_handling.status is not NA or draft.clarification is not None:
            errors.append(
                "answer: clarification_handling must be not_applicable, clarification null"
            )
        wanted = {_normalise(c): c for c in criteria}
        got = [_normalise(r.criterion) for r in draft.criteria]
        missing = [c for key, c in wanted.items() if key not in got]
        extra = [r.criterion for r in draft.criteria if _normalise(r.criterion) not in wanted]
        repeated = sorted({g for g in got if got.count(g) > 1})
        if missing or extra or repeated:
            errors.append(
                f"criteria must list each success criterion once: missing {missing}, "
                f"unknown {extra}, repeated {repeated}"
            )
        expected = expected_goal_status([r.status for r in draft.criteria])
        if not (missing or extra or repeated) and draft.goal_completeness.status is not expected:
            errors.append(
                f"goal_completeness must be {expected.value} given the criteria results, "
                f"not {draft.goal_completeness.status.value}"
            )
        return errors

    # clarification
    if draft.goal_completeness.status is not NA or draft.criteria:
        errors.append("clarification: goal_completeness not_applicable and criteria empty")
    if draft.clarification is None:
        errors.append("clarification: the clarification detail is required")
    else:
        expected = expected_clarification_status(draft.clarification)
        if draft.clarification_handling.status is not expected:
            errors.append(
                f"clarification_handling must be {expected.value} given the clarification "
                f"detail, not {draft.clarification_handling.status.value}"
            )
    return errors


def generic_verdict(checks: GenericChecks, findings: Sequence[Finding]) -> Verdict:
    """A failed check backed by a supported finding -> fail; else any failed (unsupported) or
    inconclusive check -> inconclusive; else pass. not_applicable checks are ignored."""
    supported = {f.category for f in findings if f.evidence_status is EvidenceStatus.SUPPORTED}
    statuses = dict(checks)
    failed = [name for name, status in statuses.items() if status is CheckStatus.FAIL]
    if any(name in supported for name in failed):
        return Verdict.FAIL
    if failed or CheckStatus.INCONCLUSIVE in statuses.values():
        return Verdict.INCONCLUSIVE
    return Verdict.PASS


def _quoted(fragment: str, text: str) -> bool:
    return _normalise(fragment) in _normalise(text)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


# --- evaluator --------------------------------------------------------------------------------


class GenericEvaluator:
    def __init__(self, llm: LLMClient, settings: AppConfig, logger: Logger) -> None:
        self._llm = llm
        self._settings = settings
        self._log = logger

    async def judge(
        self,
        persona: Persona,
        product: ProductDescription,
        scenario: Scenario,
        question: QuestionCase,
        response: ProductResponse,
        focus: EvaluationFocus,
    ) -> GenericJudgement:
        criteria = question.expectations.success_criteria or scenario.success_criteria
        quotable = response_text(response)
        system, user = build_response_evaluation_prompt(
            persona, product, scenario, question, response, focus
        )
        judgement = GenericJudgement(focus)
        try:
            judgement.draft = await generate_structured(
                self._llm,
                system,
                user,
                GenericEvaluationDraft,
                max_retries=self._settings.llm.max_validation_retries,
                post_validate=lambda d: check_generic_evaluation(d, focus, criteria, quotable),
            )
        except LLMOutputValidationError as exc:
            self._log.warning("%s: generic evaluation failed", question.question_id)
            judgement.notes.append(
                f"generic evaluation failed after {exc.attempts} attempt(s): "
                + "; ".join(exc.errors)
            )
        return judgement
