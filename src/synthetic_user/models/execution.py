"""Product execution results: a discriminated union on ``outcome``."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from synthetic_user.models.base import NonEmptyStr, StrictModel


class ProductOutcome(StrEnum):
    SQL_GENERATED = "sql_generated"
    CLARIFICATION = "clarification"
    DENIED = "denied"
    ERROR = "error"
    TIMEOUT = "timeout"


class ExecutionMetadata(StrictModel):
    product_version: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)


class _ExecutionBase(StrictModel):
    execution_id: NonEmptyStr = Field(pattern=r"^EXE_\d{3,}$")
    question_id: NonEmptyStr = Field(pattern=r"^Q_\d{3,}$")
    product_id: NonEmptyStr
    submitted_question: NonEmptyStr
    started_at: datetime
    metadata: ExecutionMetadata = Field(default_factory=ExecutionMetadata)


class SqlGeneratedResult(_ExecutionBase):
    outcome: Literal[ProductOutcome.SQL_GENERATED] = ProductOutcome.SQL_GENERATED
    sql: NonEmptyStr


class ClarificationResult(_ExecutionBase):
    outcome: Literal[ProductOutcome.CLARIFICATION] = ProductOutcome.CLARIFICATION
    message: NonEmptyStr


class DeniedResult(_ExecutionBase):
    outcome: Literal[ProductOutcome.DENIED] = ProductOutcome.DENIED
    message: NonEmptyStr


class ErrorResult(_ExecutionBase):
    outcome: Literal[ProductOutcome.ERROR] = ProductOutcome.ERROR
    error_type: NonEmptyStr
    message: str | None = None


class TimeoutResult(_ExecutionBase):
    outcome: Literal[ProductOutcome.TIMEOUT] = ProductOutcome.TIMEOUT
    timeout_seconds: float | None = Field(default=None, gt=0)


ExecutionResult = Annotated[
    SqlGeneratedResult | ClarificationResult | DeniedResult | ErrorResult | TimeoutResult,
    Field(discriminator="outcome"),
]

EXECUTION_RESULT_ADAPTER: TypeAdapter[ExecutionResult] = TypeAdapter(ExecutionResult)
