"""MockLLMAdapter: an LLM plays an NL2SQL product behind a (mock) product contract.

Used by the web UI when a contract was requested from a product URL (D113). Real products use
the http_json or a python adapter. The SQL it returns is recorded, never executed.
"""

from __future__ import annotations

import re

from synthetic_user.adapters.base import ProductRequest
from synthetic_user.adapters.contract import ProductContract
from synthetic_user.core.exceptions import LLMProviderError, ProductAdapterError
from synthetic_user.llm.base import LLMClient
from synthetic_user.models.execution import (
    AnsweredResponse,
    ClarificationResponse,
    DeniedResponse,
    ErrorResponse,
    ProductResponse,
    ResponseArtifact,
)
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import ReferenceIndex
from synthetic_user.prompts.mock_product import build_mock_product_prompt

_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_mock_reply(text: str) -> ProductResponse:
    """Map the mock product's text reply to a ProductResponse."""
    body = text.strip()
    upper = body.upper()
    if upper.startswith("CLARIFY:"):
        return ClarificationResponse(message=body.split(":", 1)[1].strip())
    if upper.startswith("DENIED:"):
        return DeniedResponse(message=body.split(":", 1)[1].strip())
    match = _SQL_BLOCK.search(body)
    sql = (match.group(1) if match else body).strip()
    if not sql:
        return ErrorResponse(error_type="EmptyReply", message="mock product replied with nothing")
    return AnsweredResponse(artifacts=[ResponseArtifact(kind="sql", content=sql)])


class MockLLMAdapter:
    name = "mock"

    def __init__(
        self,
        llm: LLMClient,
        contract: ProductContract,
        index: ReferenceIndex,
        product: ProductDescription,
    ) -> None:
        self._llm = llm
        self._contract = contract
        self._index = index
        self._product = product

    async def send(self, req: ProductRequest) -> ProductResponse:
        system, user = build_mock_product_prompt(
            req.utterance, self._index, self._product, self._contract.sql_dialect
        )
        try:
            text = await self._llm.generate_text(system, user)
        except LLMProviderError as exc:
            raise ProductAdapterError(f"mock product failed: {exc}") from exc
        response = parse_mock_reply(text)
        return response.model_copy(update={"product_version": self._contract.contract_version})

    async def aclose(self) -> None:
        return None
