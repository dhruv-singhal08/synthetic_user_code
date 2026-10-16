"""Judge a clarification request returned by the product (relevance, necessity, reason).

The verdict is computed by code from the LLM's judgement:
fail if unnecessary or asks for info already given; pass if relevant and
necessary/reasonable; otherwise inconclusive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import LLMOutputValidationError
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.evaluation import (
    ClarificationEvaluation,
    ClarificationNecessity,
    ClarificationRelevance,
    Verdict,
)
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import ReferenceSnapshot
from synthetic_user.models.scenario import Scenario
from synthetic_user.prompts.clarification import build_clarification_prompt

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]


@dataclass(frozen=True)
class ClarificationOutcome:
    evaluation: ClarificationEvaluation | None
    verdict: Verdict
    notes: list[str] = field(default_factory=list)


def clarification_verdict(evaluation: ClarificationEvaluation) -> Verdict:
    if evaluation.necessity is ClarificationNecessity.UNNECESSARY or evaluation.asks_for_given_info:
        return Verdict.FAIL
    if evaluation.relevance is ClarificationRelevance.RELEVANT and evaluation.necessity in (
        ClarificationNecessity.NECESSARY,
        ClarificationNecessity.REASONABLE,
    ):
        return Verdict.PASS
    return Verdict.INCONCLUSIVE


class ClarificationEvaluator:
    def __init__(self, llm: LLMClient, settings: AppConfig, logger: Logger) -> None:
        self._llm = llm
        self._settings = settings
        self._log = logger

    async def evaluate(
        self,
        question: QuestionCase,
        scenario: Scenario,
        snapshot: ReferenceSnapshot,
        product: ProductDescription,
        clarification_message: str,
    ) -> ClarificationOutcome:
        system, user = build_clarification_prompt(
            question, scenario, snapshot, product, clarification_message
        )
        try:
            evaluation = await generate_structured(
                self._llm,
                system,
                user,
                ClarificationEvaluation,
                max_retries=self._settings.llm.max_validation_retries,
            )
        except LLMOutputValidationError as exc:
            self._log.warning("%s: clarification evaluation failed", question.question_id)
            return ClarificationOutcome(
                evaluation=None,
                verdict=Verdict.INCONCLUSIVE,
                notes=[
                    f"clarification evaluation failed after {exc.attempts} attempt(s): "
                    + "; ".join(exc.errors)
                ],
            )
        return ClarificationOutcome(
            evaluation=evaluation, verdict=clarification_verdict(evaluation)
        )
