"""MockProductClient: an LLM plays the NL2SQL product behind a (mock) product contract.

Replace with a real HTTP client implementing the same ProductClient Protocol once the
product connection exists. The SQL it returns is recorded, never executed.
"""

from __future__ import annotations

import re

from synthetic_user.core.exceptions import LLMProviderError, ProductClientError
from synthetic_user.llm.base import LLMClient
from synthetic_user.models.execution import ProductOutcome
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import ReferenceCatalog
from synthetic_user.product.base import ProductResponse
from synthetic_user.product.contract import ProductContract
from synthetic_user.prompts.mock_product import build_mock_product_prompt

_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_mock_reply(text: str) -> ProductResponse:
    """Map the mock product's text reply to a ProductResponse."""
    body = text.strip()
    upper = body.upper()
    if upper.startswith("CLARIFY:"):
        return ProductResponse(
            outcome=ProductOutcome.CLARIFICATION, message=body.split(":", 1)[1].strip()
        )
    if upper.startswith("DENIED:"):
        return ProductResponse(outcome=ProductOutcome.DENIED, message=body.split(":", 1)[1].strip())
    match = _SQL_BLOCK.search(body)
    sql = (match.group(1) if match else body).strip()
    if not sql:
        return ProductResponse(
            outcome=ProductOutcome.ERROR,
            error_type="EmptyReply",
            message="mock product replied with nothing",
        )
    return ProductResponse(outcome=ProductOutcome.SQL_GENERATED, sql=sql)


class MockProductClient:
    def __init__(
        self,
        llm: LLMClient,
        contract: ProductContract,
        catalog: ReferenceCatalog,
        product: ProductDescription,
    ) -> None:
        self._llm = llm
        self._contract = contract
        self._catalog = catalog
        self._product = product

    @property
    def product_id(self) -> str:
        return self._contract.product_id

    async def ask(self, utterance: str) -> ProductResponse:
        system, user = build_mock_product_prompt(
            utterance, self._catalog, self._product, self._contract.sql_dialect
        )
        try:
            text = await self._llm.generate_text(system, user)
        except LLMProviderError as exc:
            raise ProductClientError(f"mock product failed: {exc}") from exc
        response = parse_mock_reply(text)
        return response.model_copy(update={"product_version": self._contract.contract_version})
