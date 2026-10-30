"""ProductAdapter Protocol: the only way the pipeline talks to a product under test.

Each product = one adapter, selected by config (adapters/registry.py). Adapters return the
generic ProductResponse; all product-specific parsing stays inside the adapter.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import Field

from synthetic_user.core.exceptions import ProductAdapterError
from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.execution import ProductResponse

__all__ = ["ProductAdapter", "ProductRequest", "ProductResponse", "UnavailableAdapter"]


class ProductRequest(StrictModel):
    question_id: NonEmptyStr
    # ``utterance`` is delivered to the product. ``canonical_utterance`` is the generated
    # question retained for fixture matching and safe tracing when a transport-only instruction
    # has been appended. It is excluded from serialization so adapters cannot send it as a
    # second request field accidentally.
    utterance: NonEmptyStr
    canonical_utterance: NonEmptyStr | None = Field(default=None, exclude=True, repr=False)
    session_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    @property
    def report_utterance(self) -> str:
        """Question text that may be persisted or shown in reports and traces."""
        return self.canonical_utterance or self.utterance


@runtime_checkable
class ProductAdapter(Protocol):
    name: str

    async def send(self, req: ProductRequest) -> ProductResponse: ...

    async def aclose(self) -> None: ...


class UnavailableAdapter:
    """Placeholder for steps that never call the product (raises if asked)."""

    name = "unavailable"

    async def send(self, req: ProductRequest) -> ProductResponse:
        raise ProductAdapterError("this step does not call the product")

    async def aclose(self) -> None:
        return None
