"""Node 4: submit each question to the product through a ProductAdapter and record the outcome.

Depends only on the ProductAdapter protocol. Per question:
- ``asyncio.wait_for(adapter.send(req), product.timeout_seconds)``; timeout -> TimeoutResponse;
- ProductAdapterError -> ErrorResponse; any other exception at this boundary -> ErrorResponse
  (error_type = class name), logged with traceback (one documented broad catch, no bare except);
- output that fails ProductResponse validation -> ErrorResponse "invalid_adapter_response"
  (an adapter defect, recorded only - never judged as a wrong answer);
- latency: the adapter's latency_ms if it gives one, else measured with perf_counter;
- raw_response (if the adapter supplies one) is handed to ``raw_writer`` and cleared from the
  stored response; the record keeps the returned path.
Responses (and any SQL artifact) are recorded, never executed. ``execute_many`` runs calls
under ``llm.max_concurrency``, keeps question order (EXE_ ids assigned afterwards) and always
calls ``adapter.aclose()``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, JsonValue, ValidationError

from synthetic_user.adapters.base import ProductAdapter, ProductRequest
from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import ProductAdapterError, format_validation_error
from synthetic_user.core.identifiers import IdKind, SequentialIdGenerator
from synthetic_user.models.base import StrictModel
from synthetic_user.models.execution import (
    ErrorResponse,
    ExecutionRecord,
    ProductOutcome,
    ProductResponse,
    TimeoutResponse,
)
from synthetic_user.models.question import QuestionCase

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

INVALID_ADAPTER_RESPONSE = "invalid_adapter_response"
# (execution_id, raw JSON) -> path recorded as ExecutionRecord.raw_response_path
RawWriter = Callable[[str, JsonValue], str]


class _ResponseBox(StrictModel):
    response: ProductResponse


class ProductExecutor:
    def __init__(
        self,
        adapter: ProductAdapter,
        settings: AppConfig,
        logger: Logger,
        *,
        id_generator: SequentialIdGenerator | None = None,
    ) -> None:
        self._adapter = adapter
        self._settings = settings
        self._log = logger
        self._ids = id_generator or SequentialIdGenerator()

    async def execute_many(
        self,
        questions: Sequence[QuestionCase],
        *,
        product_id: str,
        raw_writer: RawWriter | None = None,
    ) -> list[ExecutionRecord]:
        """One ExecutionRecord per question, in question order. ``product_id`` comes from the
        product description. The adapter is always closed, even if something fails."""
        semaphore = asyncio.Semaphore(self._settings.llm.max_concurrency)
        try:
            raw = await asyncio.gather(*(self._ask(q, product_id, semaphore) for q in questions))
        finally:
            await self._adapter.aclose()
        records: list[ExecutionRecord] = []
        for payload, raw_body in raw:
            execution_id = self._ids.next(IdKind.EXECUTION)
            raw_path = (
                raw_writer(execution_id, raw_body)
                if raw_writer is not None and raw_body is not None
                else None
            )
            records.append(
                ExecutionRecord.model_validate(
                    {"execution_id": execution_id, "raw_response_path": raw_path, **payload}
                )
            )
        self._log.info(
            "Executed %d question(s) via %s: %s",
            len(records),
            self._adapter.name,
            ", ".join(f"{o}={sum(1 for r in records if r.outcome is o)}" for o in ProductOutcome),
        )
        return records

    async def _ask(
        self, question: QuestionCase, product_id: str, semaphore: asyncio.Semaphore
    ) -> tuple[dict[str, Any], JsonValue]:
        timeout = self._settings.product.timeout_seconds
        base: dict[str, Any] = {
            "question_id": question.question_id,
            "product_id": product_id,
            "adapter": self._adapter.name,
            "submitted_question": question.utterance,
            "started_at": datetime.now(UTC),
        }
        request = ProductRequest(
            question_id=question.question_id,
            utterance=question.utterance,
            context={"scenario_id": question.scenario_id, "persona_id": question.persona_id},
        )
        response: ProductResponse
        measured: float | None = None
        async with semaphore:
            start = time.perf_counter()
            try:
                output = await asyncio.wait_for(self._adapter.send(request), timeout)
                measured = (time.perf_counter() - start) * 1000
                response = self._validated(output, question)
            except TimeoutError:
                response = TimeoutResponse(timeout_seconds=timeout)
            except ProductAdapterError as exc:
                response = ErrorResponse(error_type=type(exc).__name__, message=str(exc)[:1000])
            except Exception as exc:  # adapter boundary: one adapter bug must not sink the run
                self._log.exception(
                    "%s: adapter %s raised %s", question.question_id, self._adapter.name, exc
                )
                response = ErrorResponse(error_type=type(exc).__name__, message=str(exc)[:1000])
        latency = response.latency_ms if response.latency_ms is not None else measured
        raw_body = response.raw_response
        if raw_body is not None:
            response = response.model_copy(update={"raw_response": None})
        return {**base, "latency_ms": latency, "response": response}, raw_body

    def _validated(self, output: object, question: QuestionCase) -> ProductResponse:
        """Re-validate whatever the adapter returned (models are dumped first, so objects
        built without validation are caught too)."""
        data = output.model_dump(mode="json") if isinstance(output, BaseModel) else output
        try:
            return _ResponseBox.model_validate({"response": data}).response
        except ValidationError as exc:
            problems = format_validation_error(exc)
            self._log.warning(
                "%s: adapter %s returned an invalid response: %s",
                question.question_id,
                self._adapter.name,
                problems,
            )
            return ErrorResponse(error_type=INVALID_ADAPTER_RESPONSE, message=problems[:1000])
