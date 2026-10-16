"""Node 4: submit each question to the product and record the outcome.

The product's SQL is recorded, never executed. Calls run concurrently under
``llm.max_concurrency``; ids (EXE_) are assigned afterwards in question order.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import ProductClientError
from synthetic_user.core.identifiers import IdKind, SequentialIdGenerator
from synthetic_user.models.execution import (
    EXECUTION_RESULT_ADAPTER,
    ExecutionResult,
    ProductOutcome,
)
from synthetic_user.models.question import QuestionCase
from synthetic_user.product.base import ProductClient

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]


class ProductExecutor:
    def __init__(
        self,
        product: ProductClient,
        settings: AppConfig,
        logger: Logger,
        *,
        id_generator: SequentialIdGenerator,
    ) -> None:
        self._product = product
        self._settings = settings
        self._log = logger
        self._ids = id_generator

    async def execute(self, questions: Sequence[QuestionCase]) -> list[ExecutionResult]:
        semaphore = asyncio.Semaphore(self._settings.llm.max_concurrency)
        raw = await asyncio.gather(*(self._ask(q, semaphore) for q in questions))
        results = [
            EXECUTION_RESULT_ADAPTER.validate_python(
                {"execution_id": self._ids.next(IdKind.EXECUTION), **payload}
            )
            for payload in raw
        ]
        self._log.info(
            "Executed %d question(s): %s",
            len(results),
            ", ".join(f"{o}={sum(1 for r in results if r.outcome is o)}" for o in ProductOutcome),
        )
        return results

    async def _ask(self, question: QuestionCase, semaphore: asyncio.Semaphore) -> dict[str, Any]:
        timeout = self._settings.product.timeout_seconds
        base: dict[str, Any] = {
            "question_id": question.question_id,
            "product_id": self._product.product_id,
            "submitted_question": question.utterance,
            "started_at": datetime.now(UTC),
        }
        async with semaphore:
            start = time.perf_counter()
            try:
                response = await asyncio.wait_for(self._product.ask(question.utterance), timeout)
            except TimeoutError:
                return {**base, "outcome": ProductOutcome.TIMEOUT, "timeout_seconds": timeout}
            except ProductClientError as exc:
                return {
                    **base,
                    "outcome": ProductOutcome.ERROR,
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:1000],
                }
            latency_ms = (time.perf_counter() - start) * 1000
        payload = response.model_dump(exclude={"product_version"}, exclude_none=True)
        return {
            **base,
            **payload,
            "metadata": {"product_version": response.product_version, "latency_ms": latency_ms},
        }
