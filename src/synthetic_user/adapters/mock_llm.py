"""MockLLMAdapter: an LLM plays the product under test behind a (mock) product contract.

Used by the web UI when a contract was requested from a product URL (D113). Real products use
the http_json or a python adapter. Fenced blocks in its reply become artifacts of that kind
(e.g. ```sql -> kind "sql"); they are recorded, never executed.
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
from synthetic_user.models.reference import Reference
from synthetic_user.prompts.mock_product import build_mock_product_prompt

_FENCED = re.compile(r"```([A-Za-z0-9_-]*)[ \t]*\n?(.*?)```", re.DOTALL)


def parse_mock_reply(text: str) -> ProductResponse:
    """Map the mock product's text reply to a ProductResponse."""
    body = text.strip()
    upper = body.upper()
    if upper.startswith("CLARIFY:"):
        return ClarificationResponse(message=body.split(":", 1)[1].strip())
    if upper.startswith("DENIED:"):
        return DeniedResponse(message=body.split(":", 1)[1].strip())
    artifacts = [
        ResponseArtifact(kind=(kind or "text").lower(), content=content.strip())
        for kind, content in _FENCED.findall(body)
        if content.strip()
    ]
    remaining = _FENCED.sub("", body).strip()
    if not remaining and not artifacts:
        return ErrorResponse(error_type="EmptyReply", message="mock product replied with nothing")
    return AnsweredResponse(text=remaining or None, artifacts=artifacts)


class MockLLMAdapter:
    name = "mock"

    def __init__(
        self,
        llm: LLMClient,
        contract: ProductContract,
        product: ProductDescription,
        reference: Reference | None,
    ) -> None:
        self._llm = llm
        self._contract = contract
        self._product = product
        self._reference = reference

    async def send(self, req: ProductRequest) -> ProductResponse:
        system, user = build_mock_product_prompt(req.utterance, self._product, self._reference)
        try:
            text = await self._llm.generate_text(system, user)
        except LLMProviderError as exc:
            raise ProductAdapterError(f"mock product failed: {exc}") from exc
        response = parse_mock_reply(text)
        return response.model_copy(update={"product_version": self._contract.contract_version})

    async def aclose(self) -> None:
        return None
