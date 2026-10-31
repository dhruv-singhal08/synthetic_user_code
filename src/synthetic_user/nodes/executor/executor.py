"""Node 4: submit each question to the product through a ProductAdapter and record the outcome.

Depends on the ProductAdapter protocol and, when clarification policy is ``ask``, the configured
LLM for persona replies. Per question:
- one stable session id is reused for every product turn;
- a clarification is answered as the persona and sent back until the product reaches a terminal
  answer / denial / error or the overall ``product.timeout_seconds`` expires;
- ProductAdapterError -> ErrorResponse; any other exception at this boundary -> ErrorResponse
  (error_type = class name), logged with traceback (one documented broad catch, no bare except);
- output that fails ProductResponse validation -> ErrorResponse "invalid_adapter_response"
  (an adapter defect, recorded only - never judged as a wrong answer);
- latency: the adapter's latency_ms if it gives one, else measured with perf_counter;
- raw responses (if supplied) are handed to ``raw_writer`` and cleared from the stored response;
  the record keeps their returned path and a user-visible conversation transcript.
Responses (and any SQL artifact) are recorded, never executed. ``execute_many`` runs calls
under ``product.max_concurrency``, keeps question order (EXE_ ids assigned afterwards) and always
calls ``adapter.aclose()``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, JsonValue, ValidationError

from synthetic_user.adapters.base import ProductAdapter, ProductRequest
from synthetic_user.core.config import AppConfig, ClarificationPolicy
from synthetic_user.core.exceptions import ProductAdapterError, format_validation_error
from synthetic_user.core.identifiers import IdKind, SequentialIdGenerator
from synthetic_user.core.trace import record_product_call, trace_redactions, trace_subject
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.base import StrictModel
from synthetic_user.models.execution import (
    AnsweredResponse,
    ClarificationReplyDraft,
    ClarificationResponse,
    ConversationRole,
    ConversationTurn,
    ConversationTurnKind,
    DeniedResponse,
    ErrorResponse,
    ExecutionRecord,
    ProductOutcome,
    ProductResponse,
    TimeoutResponse,
)
from synthetic_user.models.persona import Persona
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import ReferencePack
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.executor.prompt import (
    SINGLE_TURN_ASSUMPTION_INSTRUCTION,
    build_clarification_reply_prompt,
)

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

INVALID_ADAPTER_RESPONSE = "invalid_adapter_response"
# Longest error message stored in an ErrorResponse.
MAX_ERROR_MESSAGE_CHARS = 1_000
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
        llm: LLMClient | None = None,
        id_generator: SequentialIdGenerator | None = None,
    ) -> None:
        self._adapter = adapter
        self._settings = settings
        self._log = logger
        self._llm = llm
        self._ids = id_generator or SequentialIdGenerator()

    async def execute_many(
        self,
        questions: Sequence[QuestionCase],
        *,
        product_id: str,
        persona: Persona | None = None,
        scenarios: Mapping[str, Scenario] | None = None,
        packs: Mapping[str, ReferencePack] | None = None,
        raw_writer: RawWriter | None = None,
    ) -> list[ExecutionRecord]:
        """One ExecutionRecord per question, in question order. ``product_id`` comes from the
        product description. The adapter is always closed, even if something fails."""
        semaphore = asyncio.Semaphore(self._settings.product.max_concurrency)
        try:
            raw = await asyncio.gather(
                *(
                    self._ask(
                        q,
                        product_id,
                        semaphore,
                        persona=persona,
                        scenario=(scenarios or {}).get(q.scenario_id),
                        pack=(
                            (packs or {}).get(q.reference_pack_id)
                            if q.reference_pack_id is not None
                            else None
                        ),
                    )
                    for q in questions
                )
            )
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
        self,
        question: QuestionCase,
        product_id: str,
        semaphore: asyncio.Semaphore,
        *,
        persona: Persona | None,
        scenario: Scenario | None,
        pack: ReferencePack | None,
    ) -> tuple[dict[str, Any], JsonValue]:
        timeout = self._settings.product.timeout_seconds
        session_id = str(uuid.uuid4())
        base: dict[str, Any] = {
            "question_id": question.question_id,
            "product_id": product_id,
            "adapter": self._adapter.name,
            "submitted_question": question.utterance,
            "session_id": session_id,
            "started_at": datetime.now(UTC),
        }
        delivered_utterance = question.utterance
        if self._settings.product.clarification_policy is ClarificationPolicy.ASSUME:
            delivered_utterance = f"{question.utterance}\n\n{SINGLE_TURN_ASSUMPTION_INSTRUCTION}"
        response: ProductResponse = TimeoutResponse(timeout_seconds=timeout)
        turns = [
            ConversationTurn(
                role=ConversationRole.USER,
                kind=ConversationTurnKind.QUESTION,
                message=question.utterance,
            )
        ]
        raw_bodies: list[JsonValue] = []
        completed_calls = 0
        single_call_latency: float | None = None
        redactions = (
            (
                (delivered_utterance, question.utterance),
                (f"\\n\\n{SINGLE_TURN_ASSUMPTION_INSTRUCTION}", ""),
                (f"\n\n{SINGLE_TURN_ASSUMPTION_INSTRUCTION}", ""),
                (SINGLE_TURN_ASSUMPTION_INSTRUCTION, ""),
            )
            if delivered_utterance != question.utterance
            else ()
        )
        async with semaphore:
            conversation_start = time.perf_counter()
            next_utterance = delivered_utterance
            with trace_subject(question.question_id), trace_redactions(*redactions):
                while True:
                    remaining = timeout - (time.perf_counter() - conversation_start)
                    if remaining <= 0:
                        response = TimeoutResponse(timeout_seconds=timeout)
                        turns.append(_response_turn(response, None))
                        break
                    request = ProductRequest(
                        question_id=question.question_id,
                        utterance=next_utterance,
                        canonical_utterance=(
                            question.utterance
                            if next_utterance == delivered_utterance
                            and delivered_utterance != question.utterance
                            else None
                        ),
                        session_id=session_id,
                        context={
                            "scenario_id": question.scenario_id,
                            "persona_id": question.persona_id,
                        },
                    )
                    call_start = time.perf_counter()
                    call_latency: float | None = None
                    try:
                        output = await asyncio.wait_for(self._adapter.send(request), remaining)
                        measured = (time.perf_counter() - call_start) * 1000
                        completed_calls += 1
                        response = _sanitize_response(
                            self._validated(output, question), question.utterance
                        )
                        call_latency = (
                            response.latency_ms if response.latency_ms is not None else measured
                        )
                        single_call_latency = call_latency
                    except TimeoutError:
                        response = TimeoutResponse(timeout_seconds=timeout)
                    except ProductAdapterError as exc:
                        message = _remove_internal_instruction(str(exc), question.utterance)
                        response = ErrorResponse(
                            error_type=type(exc).__name__, message=message[:MAX_ERROR_MESSAGE_CHARS]
                        )
                    # Adapter boundary: one adapter bug must not sink the run.
                    except Exception as exc:
                        self._log.exception(
                            "%s: adapter %s raised %s",
                            question.question_id,
                            self._adapter.name,
                            exc,
                        )
                        message = _remove_internal_instruction(str(exc), question.utterance)
                        response = ErrorResponse(
                            error_type=type(exc).__name__, message=message[:MAX_ERROR_MESSAGE_CHARS]
                        )

                    raw_body = response.raw_response
                    if raw_body is not None:
                        raw_bodies.append(raw_body)
                        response = response.model_copy(update={"raw_response": None})
                    turns.append(_response_turn(response, call_latency))

                    can_continue = (
                        isinstance(response, ClarificationResponse)
                        and self._settings.product.clarification_policy is ClarificationPolicy.ASK
                        and self._llm is not None
                        and persona is not None
                        and scenario is not None
                    )
                    if not can_continue:
                        break
                    assert persona is not None and scenario is not None
                    remaining = timeout - (time.perf_counter() - conversation_start)
                    if remaining <= 0:
                        response = TimeoutResponse(timeout_seconds=timeout)
                        turns.append(_response_turn(response, None))
                        break
                    try:
                        reply = await asyncio.wait_for(
                            self._clarification_reply(persona, scenario, question, turns, pack),
                            remaining,
                        )
                    except TimeoutError:
                        response = TimeoutResponse(timeout_seconds=timeout)
                        turns.append(_response_turn(response, None))
                        break
                    except Exception as exc:
                        self._log.exception(
                            "%s: could not answer product clarification", question.question_id
                        )
                        response = ErrorResponse(
                            error_type="ClarificationReplyError",
                            message=str(exc)[:MAX_ERROR_MESSAGE_CHARS],
                        )
                        turns.append(_response_turn(response, None))
                        break
                    turns.append(
                        ConversationTurn(
                            role=ConversationRole.USER,
                            kind=ConversationTurnKind.CLARIFICATION_ANSWER,
                            message=reply,
                        )
                    )
                    next_utterance = reply

        elapsed_ms = (time.perf_counter() - conversation_start) * 1000
        latency = (
            single_call_latency
            if completed_calls == 1
            else (elapsed_ms if completed_calls else None)
        )
        with trace_subject(question.question_id):
            record_product_call(
                adapter=self._adapter.name,
                question=question.utterance,
                outcome=response.outcome.value,
                answer=response.model_dump(mode="json", exclude={"raw_response"}),
                latency_ms=latency,
            )
        raw_payload: JsonValue = None
        if len(raw_bodies) == 1:
            raw_payload = raw_bodies[0]
        elif raw_bodies:
            raw_payload = raw_bodies
        return {
            **base,
            "latency_ms": latency,
            "response": response,
            "turns": turns,
        }, raw_payload

    async def _clarification_reply(
        self,
        persona: Persona,
        scenario: Scenario,
        question: QuestionCase,
        turns: list[ConversationTurn],
        pack: ReferencePack | None,
    ) -> str:
        assert self._llm is not None
        system, user = build_clarification_reply_prompt(persona, scenario, question, turns, pack)
        draft = await generate_structured(
            self._llm,
            system,
            user,
            ClarificationReplyDraft,
            self._settings.llm.max_validation_retries,
        )
        return draft.reply

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
            return ErrorResponse(
                error_type=INVALID_ADAPTER_RESPONSE, message=problems[:MAX_ERROR_MESSAGE_CHARS]
            )


def _response_turn(response: ProductResponse, latency_ms: float | None) -> ConversationTurn:
    if isinstance(response, ClarificationResponse):
        kind, message = ConversationTurnKind.CLARIFICATION, response.message
    elif isinstance(response, AnsweredResponse):
        kinds = ", ".join(a.kind for a in response.artifacts)
        message = response.text or f"Returned artifact(s): {kinds}"
        kind = ConversationTurnKind.ANSWER
    elif isinstance(response, DeniedResponse):
        kind, message = ConversationTurnKind.DENIAL, response.message
    elif isinstance(response, ErrorResponse):
        kind = ConversationTurnKind.ERROR
        message = response.message or response.error_type
    else:
        kind = ConversationTurnKind.TIMEOUT
        message = "The product conversation timed out."
    return ConversationTurn(
        role=ConversationRole.PRODUCT,
        kind=kind,
        message=message,
        latency_ms=latency_ms,
    )


def _remove_internal_instruction(text: str, canonical_question: str) -> str:
    delivered = f"{canonical_question}\n\n{SINGLE_TURN_ASSUMPTION_INSTRUCTION}"
    return text.replace(delivered, canonical_question).replace(
        SINGLE_TURN_ASSUMPTION_INSTRUCTION, ""
    )


def _sanitize_response(response: ProductResponse, canonical_question: str) -> ProductResponse:
    """Prevent an echoed transport instruction from entering records, raw files, or reports."""

    def clean(value: Any) -> Any:
        if isinstance(value, str):
            return _remove_internal_instruction(value, canonical_question)
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return _ResponseBox.model_validate(
        {"response": clean(response.model_dump(mode="json"))}
    ).response
