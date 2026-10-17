"""Template: a custom product adapter, plugged in by config without editing synthetic_user.

Use this when your product is not plain JSON over HTTP (custom SDK, websockets, signed
requests, multi-step auth, ...). Copy this file into your own package, fill in ``send``, then
point the config at it:

    # config/products/my_product.adapter.yaml
    type: python
    options:
      class: "my_package.my_product_adapter:MyProductAdapter"   # "module.path:ClassName"
      init:                                                    # passed as keyword arguments
        base_url: https://my-product.example/api
        token_env: MY_PRODUCT_TOKEN                            # read the secret from env

The module must be importable (installed, or on PYTHONPATH). The class must provide:
  - ``name``: short adapter name, recorded on every ExecutionRecord;
  - ``async send(req: ProductRequest) -> ProductResponse``;
  - ``async aclose() -> None``: release connections; called once after the run.

Rules:
  - Return ONLY the generic responses from synthetic_user.models.execution (AnsweredResponse,
    ClarificationResponse, DeniedResponse, ErrorResponse, TimeoutResponse). Keep every
    product-specific detail (payload shape, status codes, field names) inside this class.
  - Put product payloads you want evaluated in ``artifacts`` (e.g. kind "sql", "table",
    "json"); free text goes in ``text``.
  - For connection/transport failures raise ProductAdapterError (recorded as outcome "error");
    the executor applies product.timeout_seconds around ``send``.
  - Never log or return secrets (tokens, keys, cookies).
"""

from __future__ import annotations

import os

from synthetic_user.adapters.base import ProductRequest
from synthetic_user.core.exceptions import ConfigError
from synthetic_user.models.execution import (
    AnsweredResponse,
    ClarificationResponse,
    ProductResponse,
    ResponseArtifact,
)


class MyProductAdapter:
    name = "my_product"

    def __init__(self, base_url: str, token_env: str) -> None:
        token = os.environ.get(token_env, "")
        if not token:
            # Name the variable, never print its value.
            raise ConfigError(f"environment variable {token_env} is not set")
        self._base_url = base_url
        self._token = token
        # e.g. self._client = MyProductSdk(base_url, token)

    async def send(self, req: ProductRequest) -> ProductResponse:
        # reply = await self._client.ask(req.utterance, session=req.session_id)
        reply = {"kind": "answer", "answer": f"(template) you asked: {req.utterance}", "sql": None}

        # Map YOUR product's reply to the generic response here, and only here.
        if reply["kind"] == "clarify":
            return ClarificationResponse(message=str(reply["answer"]))
        artifacts = (
            [ResponseArtifact(kind="sql", content=str(reply["sql"]))] if reply["sql"] else []
        )
        return AnsweredResponse(text=str(reply["answer"]), artifacts=artifacts)

    async def aclose(self) -> None:
        # await self._client.close()
        return None
