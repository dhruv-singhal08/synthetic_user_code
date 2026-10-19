"""HttpJsonAdapter: talk to any JSON-over-HTTP product from config alone (no product code).

Options (``product_adapter.options``)::

    url: https://product.example/v1/ask
    method: POST                      # GET sends no body
    headers: {Authorization: "Bearer ${PRODUCT_TOKEN}"}   # ${ENV_VAR} filled from env
    timeout: 30
    request_template: {question: "{utterance}", session: "{session_id}", trace: "{uuid}"}
    response_mapping:
      text: data.answer               # dotted paths; list items by index, e.g. data.items.0
      artifacts: [{kind: sql, path: data.sql}]
      clarification_when: "data.type == clarification"
      clarification_message: data.message     # default: the text path
      denied_when: "data.status == unsupported"
      denied_message: data.message            # default: the text path
      product_version: meta.version

Conditions are ``path == value`` only (no eval, no JSONPath). Non-string JSON values compare
by their JSON text (true, 3, null). HTTP >= 400 and non-JSON bodies become ErrorResponse with
the status code. Transport = stdlib urllib in a worker thread (D136); tests inject their own.
Resolved header values are never logged and are scrubbed from every error message.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.error
import urllib.request
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, ValidationError, field_validator, model_validator

from synthetic_user.adapters.base import ProductRequest
from synthetic_user.core.exceptions import ConfigError, ProductAdapterError, format_validation_error
from synthetic_user.core.trace import record_product_http
from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.execution import (
    AnsweredResponse,
    ClarificationResponse,
    DeniedResponse,
    ErrorResponse,
    ProductResponse,
    ResponseArtifact,
)

_log = logging.getLogger(__name__)

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_PLACEHOLDERS = ("utterance", "session_id", "uuid")
_MISSING = object()


# --- options ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    path: str
    expected: str

    @classmethod
    def parse(cls, text: str) -> Condition:
        if text.count("==") != 1:
            raise ValueError(f"condition must be 'path == value', got {text!r}")
        path, expected = (part.strip() for part in text.split("=="))
        if not path or not expected:
            raise ValueError(f"condition must be 'path == value', got {text!r}")
        if len(expected) >= 2 and expected[0] == expected[-1] and expected[0] in "'\"":
            expected = expected[1:-1]
        return cls(path, expected)

    def matches(self, document: Any) -> bool:
        found = lookup(document, self.path)
        return found is not _MISSING and _as_text(found) == self.expected


class ArtifactMapping(StrictModel):
    kind: NonEmptyStr
    path: NonEmptyStr


class ResponseMapping(StrictModel):
    text: NonEmptyStr | None = None
    artifacts: list[ArtifactMapping] = Field(default_factory=list)
    clarification_when: NonEmptyStr | None = None
    clarification_message: NonEmptyStr | None = None
    denied_when: NonEmptyStr | None = None
    denied_message: NonEmptyStr | None = None
    product_version: NonEmptyStr | None = None

    @field_validator("clarification_when", "denied_when")
    @classmethod
    def _valid_condition(cls, value: str | None) -> str | None:
        if value is not None:
            Condition.parse(value)
        return value

    @model_validator(mode="after")
    def _maps_an_answer(self) -> Self:
        if self.text is None and not self.artifacts:
            raise ValueError("response_mapping needs a text path or at least one artifact")
        return self


class HttpJsonOptions(StrictModel):
    url: NonEmptyStr
    method: Literal["GET", "POST", "PUT", "PATCH"] = "POST"
    headers: dict[NonEmptyStr, str] = Field(default_factory=dict)
    timeout: float = Field(default=30.0, gt=0)
    request_template: dict[str, Any] | list[Any] = Field(
        default_factory=lambda: {"utterance": "{utterance}"}
    )
    response_mapping: ResponseMapping

    @field_validator("url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError("url must be an http(s) URL")
        return value


# --- transport ------------------------------------------------------------------------------


@dataclass(frozen=True)
class HttpRequest:
    method: str
    url: str
    headers: Mapping[str, str] = field(repr=False)
    body: bytes | None
    timeout: float


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: bytes


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
        with urllib.request.urlopen(prepared, timeout=request.timeout) as response:
            return HttpResult(response.status, response.read())
    except urllib.error.HTTPError as exc:
        return HttpResult(exc.code, exc.read())
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise ProductAdapterError(
            f"request to {_host(request.url)} failed: {type(exc).__name__}: {reason}"
        ) from exc


# --- adapter ----------------------------------------------------------------------------------


class HttpJsonAdapter:
    name = "http_json"

    def __init__(
        self,
        options: HttpJsonOptions,
        *,
        env: Mapping[str, str],
        transport: Transport = urllib_transport,
    ) -> None:
        self._options = options
        self._headers = resolve_env_refs(options.headers, env)
        # Env values used in headers: scrubbed from any error text (longest first).
        self._secrets = sorted(
            {env[n] for template in options.headers.values() for n in _ENV_REF.findall(template)},
            key=len,
            reverse=True,
        )
        self._transport = transport
        mapping = options.response_mapping
        self._clarification = (
            Condition.parse(mapping.clarification_when) if mapping.clarification_when else None
        )
        self._denied = Condition.parse(mapping.denied_when) if mapping.denied_when else None

    @classmethod
    def from_options(
        cls,
        options: Mapping[str, Any],
        *,
        env: Mapping[str, str],
        transport: Transport = urllib_transport,
    ) -> Self:
        try:
            parsed = HttpJsonOptions.model_validate(options)
        except ValidationError as exc:
            raise ConfigError(
                f"Invalid product_adapter options for http_json: {format_validation_error(exc)}"
            ) from exc
        return cls(parsed, env=env, transport=transport)

    def __repr__(self) -> str:
        return f"HttpJsonAdapter(url={self._options.url!r}, headers=<{len(self._headers)} hidden>)"

    async def send(self, req: ProductRequest) -> ProductResponse:
        options = self._options
        body = None
        if options.method != "GET":
            body = json.dumps(render_template(options.request_template, req)).encode("utf-8")
        request = HttpRequest(
            method=options.method,
            url=options.url,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **self._headers,
            },
            body=body,
            timeout=options.timeout,
        )
        trace_request = {
            "method": options.method,
            "url": options.url,
            "header_names": sorted(request.headers),  # never the values
            "body": render_template(options.request_template, req) if body is not None else None,
        }
        try:
            result = await self._transport(request)
        except ProductAdapterError as exc:
            message = self._scrub(str(exc))
            record_product_http(request=trace_request, response=None, error=message)
            raise ProductAdapterError(message) from None
        _log.debug("http_json %s %s -> HTTP %d", options.method, _host(options.url), result.status)
        record_product_http(
            request=trace_request,
            response={
                "status": result.status,
                "body": self._scrub(result.body.decode("utf-8", errors="replace")),
            },
            error=None,
        )
        return self._to_response(result)

    async def aclose(self) -> None:
        return None

    def _to_response(self, result: HttpResult) -> ProductResponse:
        if result.status >= 400:
            snippet = result.body[:200].decode("utf-8", errors="replace")
            return ErrorResponse(
                error_type="HttpError", message=self._scrub(f"HTTP {result.status}: {snippet}")
            )
        try:
            document = json.loads(result.body)
        except ValueError:
            return ErrorResponse(
                error_type="NonJsonResponse", message=f"HTTP {result.status}: body is not JSON"
            )
        # The JSON body (never headers) is kept for raw/<execution_id>.json.
        return self._map(document, result.status).model_copy(update={"raw_response": document})

    def _map(self, document: Any, status: int) -> ProductResponse:
        mapping = self._options.response_mapping
        version = _text_at(document, mapping.product_version)
        if self._clarification and self._clarification.matches(document):
            message = _text_at(document, mapping.clarification_message or mapping.text)
            if message:
                return ClarificationResponse(message=message, product_version=version)
            return _mapping_error(status, "clarification matched but no message was found")
        if self._denied and self._denied.matches(document):
            message = _text_at(document, mapping.denied_message or mapping.text)
            if message:
                return DeniedResponse(message=message, product_version=version)
            return _mapping_error(status, "denied matched but no message was found")
        text = _text_at(document, mapping.text)
        artifacts = [
            ResponseArtifact(kind=a.kind, content=_artifact_content(found))
            for a in mapping.artifacts
            if (found := lookup(document, a.path)) not in (_MISSING, None, "")
        ]
        if not text and not artifacts:
            return _mapping_error(status, "no text or artifact at the mapped paths")
        return AnsweredResponse(text=text, artifacts=artifacts, product_version=version)

    def _scrub(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "***")
        return text


# --- helpers ----------------------------------------------------------------------------------


def resolve_env_refs(headers: Mapping[str, str], env: Mapping[str, str]) -> dict[str, str]:
    """Fill ``${NAME}`` in header values from env. A missing variable is a ConfigError naming
    the variable (never a value)."""
    resolved: dict[str, str] = {}
    for header, template in headers.items():
        missing = [n for n in _ENV_REF.findall(template) if not env.get(n)]
        if missing:
            raise ConfigError(
                f"product_adapter header {header!r} needs environment variable(s) {missing}"
            )
        resolved[header] = _ENV_REF.sub(lambda m: env[m.group(1)], template)
    return resolved


def render_template(template: Any, req: ProductRequest) -> Any:
    """Replace {utterance} / {session_id} / {uuid} in string values. {uuid} is one fresh
    UUID4 per request (a new product session per question). A value that is exactly one
    placeholder becomes the raw value (so a missing session_id is JSON null)."""
    values = {"utterance": req.utterance, "session_id": req.session_id, "uuid": str(uuid.uuid4())}
    return _render(template, values)


def _render(template: Any, values: Mapping[str, str | None]) -> Any:
    if isinstance(template, dict):
        return {key: _render(value, values) for key, value in template.items()}
    if isinstance(template, list):
        return [_render(value, values) for value in template]
    if isinstance(template, str):
        for name in _PLACEHOLDERS:
            if template == "{" + name + "}":
                return values[name]
        for name in _PLACEHOLDERS:
            template = template.replace("{" + name + "}", values[name] or "")
    return template


def lookup(document: Any, path: str) -> Any:
    """Follow a dotted path through dicts (keys) and lists (integer indexes)."""
    current = document
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return _MISSING
    return current


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def _text_at(document: Any, path: str | None) -> str | None:
    if path is None:
        return None
    found = lookup(document, path)
    if found is _MISSING or found is None:
        return None
    text = _as_text(found).strip()
    return text or None


def _artifact_content(found: Any) -> str | dict[str, Any] | list[Any]:
    return found if isinstance(found, str | dict | list) else json.dumps(found)


def _mapping_error(status: int, problem: str) -> ErrorResponse:
    return ErrorResponse(error_type="MappingError", message=f"HTTP {status}: {problem}")


def _host(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.hostname or ''}"
