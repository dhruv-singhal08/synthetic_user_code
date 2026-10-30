"""HTTP adapter for products that return Server-Sent Events (SSE)."""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Self

import aiohttp
from pydantic import ValidationError

from synthetic_user.adapters.base import ProductRequest
from synthetic_user.adapters.http_common import (
    HttpOptionsBase,
    HttpRequest,
    ResponseMapper,
    host,
    lookup,
    redact_json,
    render_template,
    resolve_ca_bundle,
    resolve_headers,
    scrub_text,
)
from synthetic_user.core.exceptions import ConfigError, ProductAdapterError, format_validation_error
from synthetic_user.core.trace import record_product_http
from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.execution import ErrorResponse, ProductResponse, TimeoutResponse

_log = logging.getLogger(__name__)


class SseStreamOptions(StrictModel):
    data_prefix: NonEmptyStr = "data: "
    done_marker: NonEmptyStr = "[DONE]"
    event_field: NonEmptyStr = "event"


class HttpSseOptions(HttpOptionsBase):
    stream: SseStreamOptions


@dataclass(frozen=True)
class SseHttpResult:
    status: int
    lines: list[str]


SseTransport = Callable[[HttpRequest], Awaitable[SseHttpResult]]


async def aiohttp_sse_transport(request: HttpRequest) -> SseHttpResult:
    """Open an SSE response and consume it line by line under one total timeout."""

    timeout = aiohttp.ClientTimeout(total=request.timeout)
    context = (
        ssl.create_default_context(cafile=request.tls_ca_bundle)
        if request.tls_ca_bundle is not None
        else None
    )
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.request(
                request.method,
                request.url,
                data=request.body,
                headers=dict(request.headers),
                ssl=context if context is not None else True,
            ) as response,
        ):
            lines: list[str] = []
            while True:
                raw = await response.content.readline()
                if not raw:
                    break
                lines.append(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
            return SseHttpResult(response.status, lines)
    except TimeoutError:
        raise
    except (aiohttp.ClientError, OSError) as exc:
        raise ProductAdapterError(
            f"stream request to {host(request.url)} failed: {type(exc).__name__}: {exc}"
        ) from exc


class HttpSseAdapter:
    name = "http_sse"

    def __init__(
        self,
        options: HttpSseOptions,
        *,
        env: Mapping[str, str],
        transport: SseTransport = aiohttp_sse_transport,
    ) -> None:
        self._options = options
        self._env = env
        self._tls_ca_bundle = resolve_ca_bundle(options.tls_ca_bundle, env)
        self._transport = transport
        self._mapper = ResponseMapper(options.response_mapping)

    @classmethod
    def from_options(
        cls,
        options: Mapping[str, Any],
        *,
        env: Mapping[str, str],
        transport: SseTransport = aiohttp_sse_transport,
    ) -> Self:
        try:
            parsed = HttpSseOptions.model_validate(options)
        except ValidationError as exc:
            raise ConfigError(
                f"Invalid product_adapter options for http_sse: {format_validation_error(exc)}"
            ) from exc
        return cls(parsed, env=env, transport=transport)

    def __repr__(self) -> str:
        count = len(self._options.headers)
        return f"HttpSseAdapter(url={self._options.url!r}, headers=<{count} hidden>)"

    async def send(self, req: ProductRequest) -> ProductResponse:
        options = self._options
        resolved = resolve_headers(options.headers, self._env)
        rendered = render_template(options.request_template, req)
        body = None if options.method == "GET" else json.dumps(rendered).encode("utf-8")
        request = HttpRequest(
            method=options.method,
            url=options.url,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                **resolved.values,
            },
            body=body,
            timeout=options.timeout,
            tls_ca_bundle=self._tls_ca_bundle,
        )
        trace_request = {
            "method": options.method,
            "url": options.url,
            "header_names": sorted(request.headers),
            "body": rendered if body is not None else None,
        }
        try:
            result = await asyncio.wait_for(self._transport(request), options.timeout)
        except TimeoutError:
            message = f"SSE stream timed out after {options.timeout:g} seconds"
            record_product_http(request=trace_request, response=None, error=message)
            return TimeoutResponse(timeout_seconds=options.timeout)
        except ProductAdapterError as exc:
            message = scrub_text(str(exc), resolved.secrets)
            record_product_http(request=trace_request, response=None, error=message)
            raise ProductAdapterError(message) from None

        _log.debug("http_sse %s %s -> HTTP %d", options.method, host(options.url), result.status)
        if result.status in (401, 403):
            message = "auth failed - token missing or expired"
            record_product_http(
                request=trace_request,
                response={"status": result.status, "body": message},
                error=None,
            )
            return ErrorResponse(error_type="AuthenticationError", message=message)
        if result.status >= 400:
            snippet = "\n".join(result.lines)[:200]
            message = scrub_text(f"HTTP {result.status}: {snippet}", resolved.secrets)
            record_product_http(
                request=trace_request,
                response={"status": result.status, "body": message},
                error=None,
            )
            return ErrorResponse(error_type="HttpError", message=message)

        events = parse_sse_events(result.lines, options.stream)
        # Map from the scrubbed events as well as storing them. This prevents a product that
        # echoes a credential from putting it into an answer, artifact, raw file, or run log.
        redacted_events = redact_json(events, resolved.secrets)
        assert isinstance(redacted_events, list)  # redact_json preserves container shape
        document = aggregate_events(redacted_events, options.stream.event_field)
        record_product_http(
            request=trace_request,
            response={
                "status": result.status,
                "body": json.dumps(redacted_events, ensure_ascii=False),
            },
            error=None,
        )
        response = self._mapper.map(
            document,
            result.status,
            condition_documents=redacted_events,
            require_mapped_text=True,
        )
        return response.model_copy(update={"raw_response": redacted_events})

    async def aclose(self) -> None:
        return None


def parse_sse_events(lines: Iterable[str], options: SseStreamOptions) -> list[dict[str, Any]]:
    """Parse SSE frames into ordered JSON event objects, stopping at the done marker."""

    events: list[dict[str, Any]] = []
    for payload in _data_payloads(lines, options.data_prefix):
        if payload == options.done_marker:
            break
        try:
            chunk = json.loads(payload)
        except ValueError:
            _log.warning("Skipping non-JSON SSE data payload")
            continue
        if not isinstance(chunk, dict):
            _log.warning("Skipping SSE JSON payload that is not an object")
            continue
        events.append(chunk)
    return events


def aggregate_events(events: list[dict[str, Any]], event_field: str) -> dict[str, Any]:
    """Keep the last event per name plus the full ordered event list under ``_events``."""

    document: dict[str, Any] = {"_events": events}
    for event in events:
        name = lookup(event, event_field)
        if isinstance(name, str) and name and name != "_events":
            document[name] = event
    return document


def _data_payloads(lines: Iterable[str], data_prefix: str) -> Iterable[str]:
    data_lines: list[str] = []

    def flush() -> str | None:
        if not data_lines:
            return None
        payload = "\n".join(data_lines)
        data_lines.clear()
        return payload

    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            payload = flush()
            if payload is not None:
                yield payload
            continue
        if line.startswith(data_prefix):
            data_lines.append(line[len(data_prefix) :])
    payload = flush()
    if payload is not None:
        yield payload
