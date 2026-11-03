"""Product execution: a generic ProductResponse union on ``outcome`` wrapped in an ExecutionRecord.

Product-specific payloads (SQL, tables, JSON, ...) travel as ResponseArtifacts of an answer.
Metric-driven telemetry is standardized here: an ordered tool-call trace, a normalized
retrieved-context artifact (ranked chunks + citation mappings, from the response or a separate
context API) and correlation ids. ``None`` / ``not_requested`` always means "not exposed",
which is different from "exposed but empty".
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, JsonValue, model_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel

# Adapter name given to records migrated from the pre-v2.1 format (adapter was not recorded).
LEGACY_ADAPTER = "legacy"


class ProductOutcome(StrEnum):
    ANSWERED = "answered"
    CLARIFICATION = "clarification"
    DENIED = "denied"
    ERROR = "error"
    TIMEOUT = "timeout"


class ResponseClassification(StrEnum):
    """Semantic kind of the terminal product response (separate from the transport outcome)."""

    ANSWERED = "answered"
    UNABLE_TO_ANSWER = "unable_to_answer"
    CLARIFICATION = "clarification"
    # No reliable typed field and no judge classified it: never assumed to be "answered".
    UNKNOWN = "unknown"


class ContextAvailability(StrEnum):
    AVAILABLE = "available"  # fetched / read; may hold zero chunks (= nothing retrieved)
    UNAVAILABLE = "unavailable"  # configured, but the response / API exposed no context
    FAILED = "failed"  # the context API call or the mapping failed (error kept)
    NOT_REQUESTED = "not_requested"  # no context source configured for this product


class ContextSourceMode(StrEnum):
    RESPONSE_FIELD = "response_field"
    CONTEXT_API = "context_api"
    NONE = "none"


class RetrievedChunk(StrictModel):
    chunk_id: NonEmptyStr
    # Explicit retrieval rank (1 = best). None when the product does not expose a rank.
    rank: int | None = Field(default=None, ge=1)
    text: str
    source_id: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class CitationRef(StrictModel):
    """A citation exposed in the answer, resolved (or not) to one retrieved chunk."""

    citation_id: str | None = None
    chunk_id: str | None = None
    source_id: str | None = None
    claim: str | None = None  # the cited answer text, when the product exposes it
    resolved: bool = False  # set by the executor: chunk_id names a retrieved chunk


class RetrievedContext(StrictModel):
    availability: ContextAvailability = ContextAvailability.NOT_REQUESTED
    source_mode: ContextSourceMode = ContextSourceMode.NONE
    retrieved_at: datetime | None = None
    # Request / session / trace id tying the context to the same execution.
    correlation_id: str | None = None
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    citations: list[CitationRef] = Field(default_factory=list)
    rank_exposed: bool = False  # chunk ranks are explicit (or list order is declared as rank)
    citations_exposed: bool = False  # the adapter maps citations for this product
    truncated: bool = False  # chunks / text were cut to the configured bounds
    error: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        problems: list[str] = []
        if self.availability in (ContextAvailability.FAILED, ContextAvailability.UNAVAILABLE):
            if not self.error:
                problems.append(f"{self.availability.value} context needs an error reason")
        elif self.error is not None:
            problems.append("only failed / unavailable context carries an error")
        if self.availability is not ContextAvailability.AVAILABLE and (
            self.chunks or self.citations
        ):
            problems.append("only available context carries chunks or citations")
        ids = [chunk.chunk_id for chunk in self.chunks]
        if len(set(ids)) != len(ids):
            problems.append("chunk ids must be unique")
        if self.rank_exposed and any(chunk.rank is None for chunk in self.chunks):
            problems.append("rank_exposed requires a rank on every chunk")
        if problems:
            raise ValueError("; ".join(problems))
        return self


class ToolCall(StrictModel):
    """One observable tool / function call, in execution order."""

    order: int = Field(ge=1)
    name: NonEmptyStr
    operation: str | None = None
    arguments: JsonValue = None
    result: JsonValue = None
    error: str | None = None


class ConversationRole(StrEnum):
    USER = "user"
    PRODUCT = "product"


class ConversationTurnKind(StrEnum):
    QUESTION = "question"
    CLARIFICATION = "clarification"
    CLARIFICATION_ANSWER = "clarification_answer"
    ANSWER = "answer"
    DENIAL = "denial"
    ERROR = "error"
    TIMEOUT = "timeout"


class ConversationTurn(StrictModel):
    """One visible turn in the product conversation; transport details stay in raw responses."""

    role: ConversationRole
    kind: ConversationTurnKind
    message: NonEmptyStr
    # Choices the product offered with a clarification (e.g. "Top 10 routes"); [] otherwise.
    options: list[NonEmptyStr] = Field(default_factory=list)
    latency_ms: float | None = Field(default=None, ge=0)


class ClarificationReplyDraft(StrictModel):
    """Persona-authored answer to one product clarification."""

    reply: NonEmptyStr


class ResponseArtifact(StrictModel):
    kind: NonEmptyStr = Field(description='e.g. "sql", "table", "text", "json"')
    content: str | dict[str, Any] | list[Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class _ResponseBase(StrictModel):
    product_version: str | None = None
    # Optional, set by an adapter that knows them. The executor measures latency itself when
    # latency_ms is None, and moves raw_response (body as received, never headers) to
    # raw/<execution_id>.json, clearing it here.
    latency_ms: float | None = Field(default=None, ge=0)
    raw_response: JsonValue = None
    # The product's own response type exactly as returned (e.g. "sql", "unsupported",
    # "blocked", "clarification") when the adapter maps one (D242); None = not exposed.
    product_response_type: str | None = None


class AnsweredResponse(_ResponseBase):
    outcome: Literal[ProductOutcome.ANSWERED] = ProductOutcome.ANSWERED
    text: str | None = None
    artifacts: list[ResponseArtifact] = Field(default_factory=list)
    # Typed response kind exposed by the adapter (answered / unable_to_answer); None = not typed.
    response_classification: ResponseClassification | None = None
    # Ordered tool calls when the adapter maps a trace; None = no trace exposed.
    tool_trace: list[ToolCall] | None = None
    # Context mapped from the response itself; the executor moves it to ExecutionRecord.context.
    retrieved_context: RetrievedContext | None = None
    correlation_id: str | None = None

    @model_validator(mode="after")
    def _typed_kind(self) -> Self:
        if self.response_classification not in (
            None,
            ResponseClassification.ANSWERED,
            ResponseClassification.UNABLE_TO_ANSWER,
        ):
            raise ValueError("an answered payload is typed answered or unable_to_answer")
        return self

    @model_validator(mode="after")
    def _not_empty(self) -> Self:
        if not (self.text and self.text.strip()) and not self.artifacts:
            raise ValueError("an answered response needs text or at least one artifact")
        return self

    def first_artifact(self, kind: str) -> ResponseArtifact | None:
        return next((a for a in self.artifacts if a.kind == kind), None)


class ClarificationResponse(_ResponseBase):
    outcome: Literal[ProductOutcome.CLARIFICATION] = ProductOutcome.CLARIFICATION
    message: NonEmptyStr
    # Choices the product offered (response_mapping.clarification_options); [] when none.
    options: list[NonEmptyStr] = Field(default_factory=list)


class DeniedResponse(_ResponseBase):
    outcome: Literal[ProductOutcome.DENIED] = ProductOutcome.DENIED
    message: NonEmptyStr


class ErrorResponse(_ResponseBase):
    outcome: Literal[ProductOutcome.ERROR] = ProductOutcome.ERROR
    error_type: NonEmptyStr
    message: str | None = None


class TimeoutResponse(_ResponseBase):
    outcome: Literal[ProductOutcome.TIMEOUT] = ProductOutcome.TIMEOUT
    timeout_seconds: float | None = Field(default=None, gt=0)


ProductResponse = Annotated[
    AnsweredResponse | ClarificationResponse | DeniedResponse | ErrorResponse | TimeoutResponse,
    Field(discriminator="outcome"),
]


class ExecutionRecord(StrictModel):
    execution_id: NonEmptyStr = Field(pattern=r"^EXE_\d{3,}$")
    question_id: NonEmptyStr = Field(pattern=r"^Q_\d{3,}$")
    product_id: NonEmptyStr
    adapter: NonEmptyStr
    submitted_question: NonEmptyStr
    session_id: str | None = None
    started_at: datetime
    latency_ms: float | None = Field(default=None, ge=0)
    response: ProductResponse
    turns: list[ConversationTurn] = Field(default_factory=list)
    raw_response_path: str | None = None
    # The normalized retrieved context of the terminal answer (response field or context API).
    context: RetrievedContext = Field(default_factory=RetrievedContext)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy(cls, data: Any) -> Any:
        if isinstance(data, dict) and "outcome" in data and "response" not in data:
            return migrate_legacy_execution(data)
        return data

    @property
    def outcome(self) -> ProductOutcome:
        return self.response.outcome


def migrate_legacy_execution(old: dict[str, Any]) -> dict[str, Any]:
    """Pre-v2.1 flat execution (outcome + sql/message/... + metadata) -> ExecutionRecord shape.

    ``sql_generated`` becomes ``answered`` with one artifact of kind "sql". Unknown keys are
    kept so validation still rejects them (never silently dropped).
    """
    data = dict(old)
    metadata = data.pop("metadata", None) or {}
    outcome = data.pop("outcome")
    response: dict[str, Any] = {"product_version": metadata.pop("product_version", None)}
    if outcome == "sql_generated":
        response["outcome"] = ProductOutcome.ANSWERED.value
        response["artifacts"] = [{"kind": "sql", "content": data.pop("sql", None)}]
    else:
        response["outcome"] = outcome
        for key in ("message", "error_type", "timeout_seconds"):
            if key in data:
                response[key] = data.pop(key)
    record: dict[str, Any] = {
        "adapter": LEGACY_ADAPTER,
        "latency_ms": metadata.pop("latency_ms", None),
        "response": response,
        **data,
    }
    if metadata:
        record["metadata"] = metadata
    return record
