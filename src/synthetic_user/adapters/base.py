"""ProductClient Protocol: the NL2SQL product under test, as Node 4 sees it."""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from pydantic import model_validator

from synthetic_user.core.exceptions import ProductClientError
from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.execution import ProductOutcome


class ProductResponse(StrictModel):
    """What the product returned for one question (no ids or timings; Node 4 adds those)."""

    outcome: ProductOutcome
    sql: str | None = None
    message: str | None = None
    error_type: str | None = None
    product_version: str | None = None

    @model_validator(mode="after")
    def _payload_matches_outcome(self) -> Self:
        if self.outcome is ProductOutcome.SQL_GENERATED and not self.sql:
            raise ValueError("sql is required for outcome sql_generated")
        if (
            self.outcome in (ProductOutcome.CLARIFICATION, ProductOutcome.DENIED)
            and not self.message
        ):
            raise ValueError(f"message is required for outcome {self.outcome}")
        if self.outcome is ProductOutcome.ERROR and not self.error_type:
            raise ValueError("error_type is required for outcome error")
        return self


@runtime_checkable
class ProductClient(Protocol):
    @property
    def product_id(self) -> NonEmptyStr: ...

    async def ask(self, utterance: str) -> ProductResponse: ...


class UnavailableProductClient:
    """Placeholder for steps that never call the product (raises if asked)."""

    @property
    def product_id(self) -> str:
        return "not-configured"

    async def ask(self, utterance: str) -> ProductResponse:
        raise ProductClientError("this step does not call the product")
