"""Product execution: a generic ProductResponse union on ``outcome`` wrapped in an ExecutionRecord.

Product-specific payloads (SQL, tables, JSON, ...) travel as ResponseArtifacts of an answer.
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


class AnsweredResponse(_ResponseBase):
    outcome: Literal[ProductOutcome.ANSWERED] = ProductOutcome.ANSWERED
    text: str | None = None
    artifacts: list[ResponseArtifact] = Field(default_factory=list)

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
    started_at: datetime
    latency_ms: float | None = Field(default=None, ge=0)
    response: ProductResponse
    raw_response_path: str | None = None

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
