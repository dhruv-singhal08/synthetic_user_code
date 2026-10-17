"""Product contract: how to talk to a product, requested from its URL (web UI, D113).

The real HTTP connection is NOT implemented yet. ``request_contract`` returns a MOCK contract
(``mock=True``) describing the API the real product is expected to expose; questions are then
answered by MockLLMAdapter.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import Field

from synthetic_user.core.exceptions import ProductAdapterError
from synthetic_user.models.base import NonEmptyStr, StrictModel


class ContractEndpoint(StrictModel):
    method: Literal["GET", "POST"]
    path: NonEmptyStr
    request_fields: dict[str, str] = Field(default_factory=dict)
    response_fields: dict[str, str] = Field(default_factory=dict)


class ProductContract(StrictModel):
    contract_version: NonEmptyStr
    product_id: NonEmptyStr
    product_url: NonEmptyStr
    sql_dialect: NonEmptyStr
    ask: ContractEndpoint
    outcomes: list[NonEmptyStr]
    mock: bool


def request_contract(url: str, product_id: str, sql_dialect: str) -> ProductContract:
    """MOCK: validate the URL and return the contract the product is expected to publish."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ProductAdapterError(f"Product URL must be an http(s) URL, got {url!r}")
    return ProductContract(
        contract_version="mock-1",
        product_id=product_id,
        product_url=url.strip(),
        sql_dialect=sql_dialect,
        ask=ContractEndpoint(
            method="POST",
            path="/v1/ask",
            request_fields={"question": "string - the user's natural-language question"},
            response_fields={
                "outcome": "answered | clarification | denied | error",
                "text": "string - optional answer text when outcome is answered",
                "artifacts": 'list of {kind, content, metadata} - e.g. kind "sql" holds the '
                "generated SQL",
                "message": "string - clarification question or denial reason",
            },
        ),
        outcomes=["answered", "clarification", "denied", "error", "timeout"],
        mock=True,
    )
