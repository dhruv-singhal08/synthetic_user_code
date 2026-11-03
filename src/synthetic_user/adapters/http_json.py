"""HttpJsonAdapter: talk to any JSON-over-HTTP product from config alone (no product code).

Options (``product_adapter.options``)::

    url: https://product.example/v1/ask
    method: POST                      # GET sends no body
    headers: {Authorization: "Bearer ${PRODUCT_TOKEN}"}   # ${ENV_VAR} filled from env
    timeout: 30
    tls_ca_bundle: amexcerts          # optional: amexcerts, a PEM path, or ${ENV_VAR}
    ssl_verify: true                  # false = no certificate check (test servers only)
    request_template: {question: "{utterance}", session: "{session_id}", trace: "{uuid}"}
    response_mapping:
      text: data.answer               # dotted paths; list items by index, e.g. data.items.0
      artifacts: [{kind: sql, path: data.sql}]
      clarification_when: "data.type == clarification"
      clarification_message: data.message     # default: the text path
      denied_when: "data.status == unsupported"
      denied_message: data.message            # default: the text path
      product_version: meta.version
      unable_to_answer_when: "data.kind == cannot_answer"   # typed inability (optional)
      correlation_id: meta.request_id         # the product's request / trace id (optional)
      tool_trace: {path: data.tool_calls, name: name, arguments: input, result: output}
      context:   {path: data.chunks, chunk_id: id, text: text, source_id: doc, rank: rank}
      citations: {path: data.citations, chunk_id: chunk, claim: text}
    context_api:                      # OR a separate context endpoint (not both)
      url: https://product.example/v1/context/{correlation_id}
      context: {path: chunks, chunk_id: id, text: text, rank: rank}

Conditions are ``path == value`` only (no eval, no JSONPath). Non-string JSON values compare
by their JSON text (true, 3, null). HTTP >= 400 and non-JSON bodies become ErrorResponse with
the status code. Transport = stdlib urllib in a worker thread (D136); tests inject their own.
Resolved header values are never logged and are scrubbed from every error message.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Self

from pydantic import ValidationError

from synthetic_user.adapters.base import ContextRequest, ProductRequest
from synthetic_user.adapters.http_common import (
    ERROR_SNIPPET_CHARS,
    HttpOptionsBase,
    HttpRequest,
    HttpResult,
    ResponseMapper,
    correlation_for,
    fetch_context,
    host,
    lookup,
    redact_json,
    render_template,
    resolve_headers,
    scrub_text,
    ssl_context,
    tls_settings,
)
from synthetic_user.core.exceptions import ConfigError, ProductAdapterError, format_validation_error
from synthetic_user.core.trace import record_product_http
from synthetic_user.models.execution import (
    ErrorResponse,
    ProductResponse,
    RetrievedContext,
    TimeoutResponse,
)

_log = logging.getLogger(__name__)

__all__ = [
    "HttpJsonAdapter",
    "HttpJsonOptions",
    "HttpRequest",
    "HttpResult",
    "Transport",
    "lookup",
    "render_template",
    "urllib_transport",
]

# --- options ---------------------------------------------------------------------------------


class HttpJsonOptions(HttpOptionsBase):
    pass


# --- transport ------------------------------------------------------------------------------


Transport = Callable[[HttpRequest], Awaitable[HttpResult]]


async def urllib_transport(request: HttpRequest) -> HttpResult:
    """Blocking urllib call run in a worker thread (no extra dependency, D136)."""
    return await asyncio.to_thread(_urllib_send, request)


def _urllib_send(request: HttpRequest) -> HttpResult:
    # The scheme is validated as http(s) in HttpJsonOptions.
    prepared = urllib.request.Request(
        request.url, data=request.body, headers=dict(request.headers), method=request.method
    )
    try:
        with urllib.request.urlopen(
            prepared, timeout=request.timeout, context=ssl_context(request)
        ) as response:
            return HttpResult(response.status, response.read())
    except urllib.error.HTTPError as exc:
        return HttpResult(exc.code, exc.read())
    except TimeoutError:
        raise  # the contract timeout: a timeout outcome, not a product error
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):  # connect timeout, wrapped by urllib
            raise TimeoutError(str(exc.reason)) from exc
        raise ProductAdapterError(
            f"request to {host(request.url)} failed: {type(exc).__name__}: {exc.reason}"
        ) from exc
    except OSError as exc:
        reason = getattr(exc, "reason", exc)
        raise ProductAdapterError(
            f"request to {host(request.url)} failed: {type(exc).__name__}: {reason}"
        ) from exc


# --- adapter ----------------------------------------------------------------------------------


class HttpJsonAdapter:
    name = "http_json"

    def __init__(
        self,
        options: HttpJsonOptions,
        *,
        env: Mapping[str, str],
        ssl_verify: bool = True,
        transport: Transport = urllib_transport,
    ) -> None:
        self._options = options
        self._env = env
        self._ssl_verify, self._tls_ca_bundle = tls_settings(options, env, ssl_verify=ssl_verify)
        self._transport = transport
        self._mapper = ResponseMapper(options.response_mapping)

    @classmethod
    def from_options(
        cls,
        options: Mapping[str, Any],
        *,
        env: Mapping[str, str],
        ssl_verify: bool = True,
        transport: Transport = urllib_transport,
    ) -> Self:
        try:
            parsed = HttpJsonOptions.model_validate(options)
        except ValidationError as exc:
            raise ConfigError(
                f"Invalid product_adapter options for http_json: {format_validation_error(exc)}"
            ) from exc
        return cls(parsed, env=env, ssl_verify=ssl_verify, transport=transport)

    def __repr__(self) -> str:
        count = len(self._options.headers)
        return f"HttpJsonAdapter(url={self._options.url!r}, headers=<{count} hidden>)"

    async def send(self, req: ProductRequest) -> ProductResponse:
        options = self._options
        resolved = resolve_headers(options.headers, self._env)
        request_uuid = str(uuid.uuid4())
        rendered = render_template(options.request_template, req, request_uuid=request_uuid)
        body = None
        if options.method != "GET":
            body = json.dumps(rendered).encode("utf-8")
        request = HttpRequest(
            method=options.method,
            url=options.url,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **resolved.values,
            },
            body=body,
            timeout=options.timeout,
            tls_ca_bundle=self._tls_ca_bundle,
            ssl_verify=self._ssl_verify,
        )
        trace_request = {
            "method": options.method,
            "url": options.url,
            "header_names": sorted(request.headers),  # never the values
            "body": rendered if body is not None else None,
        }
        try:
            result = await self._transport(request)
        except TimeoutError:
            message = f"HTTP request timed out after {options.timeout:g} seconds"
            record_product_http(request=trace_request, response=None, error=message)
            return TimeoutResponse(timeout_seconds=options.timeout)
        except ProductAdapterError as exc:
            message = scrub_text(str(exc), resolved.secrets)
            record_product_http(request=trace_request, response=None, error=message)
            raise ProductAdapterError(message) from None
        _log.debug("http_json %s %s -> HTTP %d", options.method, host(options.url), result.status)
        record_product_http(
            request=trace_request,
            response={
                "status": result.status,
                "body": scrub_text(result.body.decode("utf-8", errors="replace"), resolved.secrets),
            },
            error=None,
        )
        response = self._to_response(result, resolved.secrets)
        return correlation_for(response, options, req, request_uuid)

    async def fetch_context(self, req: ContextRequest) -> RetrievedContext:
        options = self._options.context_api
        if options is None:
            return RetrievedContext()  # not_requested: no separate context source configured
        return await fetch_context(
            options,
            {
                "correlation_id": req.correlation_id,
                "session_id": req.session_id,
                "question_id": req.question_id,
                "utterance": req.utterance,
                "uuid": req.correlation_id,
            },
            self._env,
            self._transport,
            tls_ca_bundle=self._tls_ca_bundle,
            ssl_verify=self._ssl_verify,
        )

    async def aclose(self) -> None:
        return None

    def _to_response(self, result: HttpResult, secrets: tuple[str, ...]) -> ProductResponse:
        if result.status >= 400:
            snippet = result.body[:ERROR_SNIPPET_CHARS].decode("utf-8", errors="replace")
            return ErrorResponse(
                error_type="HttpError",
                message=scrub_text(f"HTTP {result.status}: {snippet}", secrets),
            )
        try:
            document = json.loads(result.body)
        except ValueError:
            return ErrorResponse(
                error_type="NonJsonResponse", message=f"HTTP {result.status}: body is not JSON"
            )
        # Map and retain only the scrubbed document. A product that echoes a credential must
        # not be able to put it into either the answer record or raw/<execution_id>.json.
        safe_document = redact_json(document, secrets)
        response = self._mapper.map(safe_document, result.status)
        return response.model_copy(update={"raw_response": safe_document})
