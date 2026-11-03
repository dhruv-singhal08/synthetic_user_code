"""Shared configuration, mapping, templating, and redaction for HTTP product adapters.

Besides the answer (text / artifacts / clarification / denial / error), a response mapping can
expose metric telemetry, all by dotted paths (no eval, no JSONPath):

- ``context``: retrieved chunks IN the product response (chunk id, text, source, rank);
- ``citations``: citation -> chunk mappings exposed with the answer;
- ``tool_trace``: the ordered tool / function calls (name, operation, arguments, result, error);
- ``unable_to_answer_when``: a typed "cannot answer" response kind inside an answered payload;
- ``correlation_id``: the product's request / trace id for this execution.

Alternatively ``context_api`` fetches the context of the SAME execution from a separate
endpoint (correlated by request / session / trace id). The two context sources are mutually
exclusive. Mappings are validated when the contract is loaded; content is validated per
response at runtime, and a malformed value yields an explicit failed / unavailable context
instead of an error for the whole answer.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
import ssl
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import quote, urlsplit

from pydantic import Field, field_validator, model_validator

from synthetic_user.adapters.base import ProductRequest
from synthetic_user.core.exceptions import ConfigError, ProductAdapterError
from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.execution import (
    AnsweredResponse,
    CitationRef,
    ClarificationResponse,
    ContextAvailability,
    ContextSourceMode,
    DeniedResponse,
    ErrorResponse,
    ProductResponse,
    ResponseArtifact,
    ResponseClassification,
    RetrievedChunk,
    RetrievedContext,
    ToolCall,
)

_log = logging.getLogger(__name__)
# Characters of an HTTP error body quoted in the ErrorResponse message.
ERROR_SNIPPET_CHARS = 200
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_PLACEHOLDERS = ("utterance", "session_id", "uuid", "correlation_id", "question_id")
# Characters of a context-API error body quoted in the failed-context reason.
CONTEXT_ERROR_SNIPPET_CHARS = 200
_MISSING = object()
_SENSITIVE_HEADER_NAMES = frozenset(
    {"authorization", "proxy-authorization", "x-api-key", "api-key"}
)


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


class ContextMapping(StrictModel):
    """Retrieved chunks: ``path`` names a list; the other paths are relative to each item."""

    path: NonEmptyStr
    text: NonEmptyStr
    # None -> positional ids "chunk-1", "chunk-2", ... (citations then cannot resolve by id).
    chunk_id: NonEmptyStr | None = None
    source_id: NonEmptyStr | None = None
    rank: NonEmptyStr | None = None
    # true = the product documents that list order IS the retrieval rank (never assumed).
    order_is_rank: bool = False
    metadata: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_rank_source(self) -> Self:
        if self.rank is not None and self.order_is_rank:
            raise ValueError("context: give a rank path OR order_is_rank, not both")
        return self


class CitationMapping(StrictModel):
    """Citations exposed with the answer: ``path`` names a list; others are item paths."""

    path: NonEmptyStr
    chunk_id: NonEmptyStr | None = None
    source_id: NonEmptyStr | None = None
    citation_id: NonEmptyStr | None = None
    claim: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _resolvable(self) -> Self:
        if self.chunk_id is None and self.source_id is None:
            raise ValueError("citations: map chunk_id or source_id so a citation can resolve")
        return self


class ToolTraceMapping(StrictModel):
    """Tool calls: ``path`` names a list (in call order); others are item paths."""

    path: NonEmptyStr
    name: NonEmptyStr
    operation: NonEmptyStr | None = None
    arguments: NonEmptyStr | None = None
    result: NonEmptyStr | None = None
    error: NonEmptyStr | None = None


class ResponseMapping(StrictModel):
    text: NonEmptyStr | None = None
    artifacts: list[ArtifactMapping] = Field(default_factory=list)
    clarification_when: NonEmptyStr | None = None
    clarification_message: NonEmptyStr | None = None
    denied_when: NonEmptyStr | None = None
    denied_message: NonEmptyStr | None = None
    error_when: NonEmptyStr | None = None
    error_message: NonEmptyStr | None = None
    product_version: NonEmptyStr | None = None
    # Typed "I cannot answer" inside an answered payload (product_outcome stays answered).
    unable_to_answer_when: NonEmptyStr | None = None
    correlation_id: NonEmptyStr | None = None
    context: ContextMapping | None = None
    citations: CitationMapping | None = None
    tool_trace: ToolTraceMapping | None = None

    @field_validator("clarification_when", "denied_when", "error_when", "unable_to_answer_when")
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


class ContextApiOptions(StrictModel):
    """A separate endpoint returning the context used for the SAME execution.

    ``url`` and ``request_template`` may use {correlation_id} (the product's request / trace id,
    else the request {uuid}, else the session id), {session_id}, {question_id} and {utterance}.
    """

    url: NonEmptyStr
    method: Literal["GET", "POST"] = "GET"
    headers: dict[NonEmptyStr, str] = Field(default_factory=dict)
    timeout: float = Field(default=30.0, gt=0)
    request_template: dict[str, Any] | list[Any] | None = None
    context: ContextMapping
    citations: CitationMapping | None = None

    @field_validator("url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        parts = urlsplit(value.replace("{", "").replace("}", ""))
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError("context_api.url must be an http(s) URL")
        return value


class HttpOptionsBase(StrictModel):
    url: NonEmptyStr
    method: Literal["GET", "POST", "PUT", "PATCH"] = "POST"
    headers: dict[NonEmptyStr, str] = Field(default_factory=dict)
    timeout: float = Field(default=30.0, gt=0)
    tls_ca_bundle: NonEmptyStr | None = None
    # false = do not check the product's TLS certificate or host name (a test server with a
    # self-signed certificate only). product.ssl_verify in the app config can also turn it off.
    ssl_verify: bool = True
    request_template: dict[str, Any] | list[Any] = Field(
        default_factory=lambda: {"utterance": "{utterance}"}
    )
    response_mapping: ResponseMapping
    context_api: ContextApiOptions | None = None

    @field_validator("url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError("url must be an http(s) URL")
        return value

    @model_validator(mode="after")
    def _one_context_source(self) -> Self:
        if self.context_api is not None and self.response_mapping.context is not None:
            raise ValueError(
                "choose ONE context source: response_mapping.context (context in the product "
                "response) or context_api (separate context endpoint)"
            )
        return self

    @property
    def context_source(self) -> ContextSourceMode:
        if self.context_api is not None:
            return ContextSourceMode.CONTEXT_API
        if self.response_mapping.context is not None:
            return ContextSourceMode.RESPONSE_FIELD
        return ContextSourceMode.NONE


@dataclass(frozen=True)
class ResolvedHeaders:
    values: dict[str, str]
    secrets: tuple[str, ...]


@dataclass(frozen=True)
class HttpRequest:
    method: str
    url: str
    headers: Mapping[str, str] = field(repr=False)
    body: bytes | None
    timeout: float
    tls_ca_bundle: str | None = None
    ssl_verify: bool = True


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: bytes


def resolve_headers(templates: Mapping[str, str], env: Mapping[str, str]) -> ResolvedHeaders:
    """Resolve header environment references at request time and return scrub values."""

    values: dict[str, str] = {}
    secrets: set[str] = set()
    for header, template in templates.items():
        names = _ENV_REF.findall(template)
        missing = [name for name in names if not env.get(name)]
        if missing:
            joined = ", ".join(missing)
            raise ConfigError(
                f"product_adapter header {header!r} needs non-empty environment variable(s): "
                f"{joined}"
            )
        values[header] = _ENV_REF.sub(lambda match: env[match.group(1)], template)
        secrets.update(env[name] for name in names)
        if header.casefold() in _SENSITIVE_HEADER_NAMES and values[header]:
            secrets.add(values[header])
    return ResolvedHeaders(values, tuple(sorted(secrets, key=len, reverse=True)))


def resolve_ca_bundle(template: str | None, env: Mapping[str, str]) -> str | None:
    """Resolve a configured TLS trust bundle (verification stays on)."""

    if template is None:
        return None
    missing = [name for name in _ENV_REF.findall(template) if not env.get(name)]
    if missing:
        raise ConfigError(f"product_adapter tls_ca_bundle needs environment variable(s) {missing}")
    value = _ENV_REF.sub(lambda match: env[match.group(1)], template)
    if value.casefold() == "amexcerts":
        try:
            package = importlib.import_module("amexcerts")
            value = str(package.certificate_path())
        except (ImportError, AttributeError) as exc:
            raise ConfigError(
                "product_adapter tls_ca_bundle is 'amexcerts', but the amexcerts package "
                "is not installed"
            ) from exc
    path = Path(value).expanduser()
    if not path.is_file():
        raise ConfigError(f"product_adapter TLS CA bundle not found: {path}")
    return str(path)


def tls_settings(
    options: HttpOptionsBase, env: Mapping[str, str], *, ssl_verify: bool
) -> tuple[bool, str | None]:
    """(verify, CA bundle path) for an HTTP adapter.

    Verification is on unless the contract (``ssl_verify: false``) or the app config
    (``product.ssl_verify: false``) turns it off; the CA bundle is only resolved when it is on.
    """
    verify = options.ssl_verify and ssl_verify
    if not verify:
        _log.warning(
            "TLS certificate verification is OFF for %s (ssl_verify: false)", host(options.url)
        )
        return False, None
    return True, resolve_ca_bundle(options.tls_ca_bundle, env)


def ssl_context(request: HttpRequest) -> ssl.SSLContext | None:
    """The TLS context for a request; None = the default (verify against the system CAs)."""
    if not request.ssl_verify:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    if request.tls_ca_bundle is not None:
        return ssl.create_default_context(cafile=request.tls_ca_bundle)
    return None


def render_template(template: Any, req: ProductRequest, *, request_uuid: str | None = None) -> Any:
    """Replace supported placeholders, sharing one UUID across the request (fresh by default;
    adapters pass their own so the same id can correlate a later context-API call)."""

    values = {
        "utterance": req.utterance,
        "session_id": req.session_id,
        "uuid": request_uuid or str(uuid.uuid4()),
        "correlation_id": None,
        "question_id": req.question_id,
    }
    return _render(template, values)


def uses_placeholder(template: Any, name: str) -> bool:
    return "{" + name + "}" in json.dumps(template)


def _render(template: Any, values: Mapping[str, str | None]) -> Any:
    if isinstance(template, dict):
        return {key: _render(value, values) for key, value in template.items()}
    if isinstance(template, list):
        return [_render(value, values) for value in template]
    if isinstance(template, str):
        for name in _PLACEHOLDERS:
            if template == "{" + name + "}":
                return values.get(name)
        for name in _PLACEHOLDERS:
            template = template.replace("{" + name + "}", values.get(name) or "")
    return template


def lookup(document: Any, path: str) -> Any:
    """Follow a dotted path through dicts and integer-indexed lists."""

    current = document
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return _MISSING
    return current


class ResponseMapper:
    """Map a JSON document into the generic ProductResponse union."""

    def __init__(self, mapping: ResponseMapping) -> None:
        self.mapping = mapping
        self._error = Condition.parse(mapping.error_when) if mapping.error_when else None
        self._clarification = (
            Condition.parse(mapping.clarification_when) if mapping.clarification_when else None
        )
        self._denied = Condition.parse(mapping.denied_when) if mapping.denied_when else None
        self._unable = (
            Condition.parse(mapping.unable_to_answer_when)
            if mapping.unable_to_answer_when
            else None
        )

    def map(
        self,
        document: Any,
        status: int,
        *,
        condition_documents: Sequence[Any] = (),
        require_mapped_text: bool = False,
    ) -> ProductResponse:
        mapping = self.mapping
        version = _text_at(document, mapping.product_version)
        if matched := self._matched(self._error, document, condition_documents):
            message = _message_at(matched, document, mapping.error_message)
            return ErrorResponse(
                error_type="ProductError",
                message=message or f"HTTP {status}: product reported an error",
                product_version=version,
            )
        if matched := self._matched(self._clarification, document, condition_documents):
            message = _message_at(matched, document, mapping.clarification_message or mapping.text)
            if message:
                return ClarificationResponse(message=message, product_version=version)
            return mapping_error(status, "clarification matched but no message was found")
        if matched := self._matched(self._denied, document, condition_documents):
            message = _message_at(matched, document, mapping.denied_message or mapping.text)
            if message:
                return DeniedResponse(message=message, product_version=version)
            return mapping_error(status, "denied matched but no message was found")
        text = _text_at(document, mapping.text)
        artifacts = [
            ResponseArtifact(kind=item.kind, content=_artifact_content(found))
            for item in mapping.artifacts
            if (found := lookup(document, item.path)) not in (_MISSING, None, "")
        ]
        if require_mapped_text and mapping.text is not None and not text:
            return mapping_error(status, "no answer event received")
        if not text and not artifacts:
            return mapping_error(status, "no text or artifact at the mapped paths")
        classification = None
        if self._unable is not None:
            classification = (
                ResponseClassification.UNABLE_TO_ANSWER
                if self._unable.matches(document)
                else ResponseClassification.ANSWERED
            )
        context = None
        if mapping.context is not None:
            context = map_context(
                document, mapping.context, mapping.citations, ContextSourceMode.RESPONSE_FIELD
            )
        return AnsweredResponse(
            text=text,
            artifacts=artifacts,
            product_version=version,
            response_classification=classification,
            tool_trace=map_tool_trace(document, mapping.tool_trace),
            retrieved_context=context,
            correlation_id=_text_at(document, mapping.correlation_id),
        )

    @staticmethod
    def _matched(
        condition: Condition | None, document: Any, alternatives: Sequence[Any]
    ) -> Any | None:
        if condition is None:
            return None
        if condition.matches(document):
            return document
        return next((item for item in reversed(alternatives) if condition.matches(item)), None)


def map_tool_trace(document: Any, mapping: ToolTraceMapping | None) -> list[ToolCall] | None:
    """Ordered tool calls; None when no trace is mapped or the response has none at the path."""
    if mapping is None:
        return None
    found = lookup(document, mapping.path)
    if found is _MISSING or found is None or not isinstance(found, list):
        return None
    calls: list[ToolCall] = []
    for order, item in enumerate(found, start=1):
        name = _text_at(item, mapping.name)
        if name is None:
            continue  # an item without a tool name is not an observable call
        calls.append(
            ToolCall(
                order=order,
                name=name,
                operation=_text_at(item, mapping.operation),
                arguments=_json_at(item, mapping.arguments),
                result=_json_at(item, mapping.result),
                error=_text_at(item, mapping.error),
            )
        )
    return calls


def map_context(
    document: Any,
    mapping: ContextMapping,
    citations: CitationMapping | None,
    source_mode: ContextSourceMode,
    *,
    correlation_id: str | None = None,
) -> RetrievedContext:
    """Normalize retrieved chunks (+ citations) found in ``document``.

    Missing path -> unavailable; a value of the wrong shape -> failed with the reason; an empty
    list -> available with zero chunks (nothing was retrieved).
    """
    now = datetime.now(UTC)
    base: dict[str, Any] = {
        "source_mode": source_mode,
        "retrieved_at": now,
        "correlation_id": correlation_id,
        "citations_exposed": citations is not None,
    }
    found = lookup(document, mapping.path)
    if found is _MISSING or found is None:
        return RetrievedContext(
            availability=ContextAvailability.UNAVAILABLE,
            error=f"no retrieved context at '{mapping.path}'",
            **base,
        )
    if not isinstance(found, list):
        return RetrievedContext(
            availability=ContextAvailability.FAILED,
            error=f"retrieved context at '{mapping.path}' is not a list",
            **base,
        )
    chunks: list[RetrievedChunk] = []
    problems: list[str] = []
    for position, item in enumerate(found, start=1):
        text = _text_at(item, mapping.text)
        if text is None:
            problems.append(f"chunk {position}: no text at '{mapping.text}'")
            continue
        rank: int | None = position if mapping.order_is_rank else None
        if mapping.rank is not None:
            raw_rank = lookup(item, mapping.rank)
            if isinstance(raw_rank, int) and not isinstance(raw_rank, bool) and raw_rank >= 1:
                rank = raw_rank
            elif isinstance(raw_rank, str) and raw_rank.strip().isdigit() and int(raw_rank) >= 1:
                rank = int(raw_rank)
            else:
                problems.append(f"chunk {position}: rank at '{mapping.rank}' is not a positive int")
                continue
        chunk_id = _text_at(item, mapping.chunk_id) if mapping.chunk_id else f"chunk-{position}"
        if chunk_id is None:
            problems.append(f"chunk {position}: no id at '{mapping.chunk_id}'")
            continue
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                rank=rank,
                text=text,
                source_id=_text_at(item, mapping.source_id),
                metadata={
                    path: value
                    for path in mapping.metadata
                    if (value := _json_at(item, path)) is not None
                },
            )
        )
    ids = [chunk.chunk_id for chunk in chunks]
    if len(set(ids)) != len(ids):
        problems.append("duplicate chunk ids")
    ranks = [chunk.rank for chunk in chunks if chunk.rank is not None]
    if len(set(ranks)) != len(ranks):
        problems.append("duplicate chunk ranks")
    if problems:
        return RetrievedContext(
            availability=ContextAvailability.FAILED,
            error="invalid retrieved context: " + "; ".join(problems[:5]),
            **base,
        )
    return RetrievedContext(
        availability=ContextAvailability.AVAILABLE,
        chunks=sorted(chunks, key=lambda c: c.rank or 0) if ranks else chunks,
        citations=_map_citations(document, citations),
        rank_exposed=bool(chunks) and len(ranks) == len(chunks),
        **base,
    )


def _map_citations(document: Any, mapping: CitationMapping | None) -> list[CitationRef]:
    if mapping is None:
        return []
    found = lookup(document, mapping.path)
    if not isinstance(found, list):
        return []  # the mapping exists: no list at the path means no citation in this answer
    return [
        CitationRef(
            citation_id=_text_at(item, mapping.citation_id),
            chunk_id=_text_at(item, mapping.chunk_id),
            source_id=_text_at(item, mapping.source_id),
            claim=_text_at(item, mapping.claim),
        )
        for item in found
    ]


JsonTransport = Callable[[HttpRequest], Awaitable[HttpResult]]


async def fetch_context(
    options: ContextApiOptions,
    values: Mapping[str, str | None],
    env: Mapping[str, str],
    transport: JsonTransport,
    *,
    tls_ca_bundle: str | None,
    ssl_verify: bool,
) -> RetrievedContext:
    """Call the context API for one execution and normalize its reply (never raises for a
    product-side problem: failures become a failed context with the reason)."""
    correlation = values.get("correlation_id")
    base: dict[str, Any] = {
        "source_mode": ContextSourceMode.CONTEXT_API,
        "retrieved_at": datetime.now(UTC),
        "correlation_id": correlation,
        "citations_exposed": options.citations is not None,
    }
    resolved = resolve_headers(options.headers, env)
    url = options.url
    for name in _PLACEHOLDERS:
        url = url.replace("{" + name + "}", quote(values.get(name) or "", safe=""))
    rendered = (
        _render(options.request_template, values) if options.request_template is not None else None
    )
    body = json.dumps(rendered).encode("utf-8") if options.method == "POST" else None
    request = HttpRequest(
        method=options.method,
        url=url,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            **resolved.values,
        },
        body=body,
        timeout=options.timeout,
        tls_ca_bundle=tls_ca_bundle,
        ssl_verify=ssl_verify,
    )
    try:
        result = await transport(request)
    except TimeoutError:
        return RetrievedContext(
            availability=ContextAvailability.FAILED,
            error=f"context API timed out after {options.timeout:g} seconds",
            **base,
        )
    except ProductAdapterError as exc:
        return RetrievedContext(
            availability=ContextAvailability.FAILED,
            error=scrub_text(f"context API request failed: {exc}", resolved.secrets),
            **base,
        )
    if result.status >= 400:
        snippet = result.body[:CONTEXT_ERROR_SNIPPET_CHARS].decode("utf-8", errors="replace")
        return RetrievedContext(
            availability=ContextAvailability.FAILED,
            error=scrub_text(f"context API HTTP {result.status}: {snippet}", resolved.secrets),
            **base,
        )
    try:
        document = redact_json(json.loads(result.body), resolved.secrets)
    except ValueError:
        return RetrievedContext(
            availability=ContextAvailability.FAILED,
            error=f"context API HTTP {result.status}: body is not JSON",
            **base,
        )
    return map_context(
        document,
        options.context,
        options.citations,
        ContextSourceMode.CONTEXT_API,
        correlation_id=correlation,
    )


def correlation_for(
    response: ProductResponse, options: HttpOptionsBase, req: ProductRequest, request_uuid: str
) -> ProductResponse:
    """Set the answer's correlation id: the product's own id, else the request {uuid} when the
    template sends it, else the session id. Used to fetch context for the same execution."""
    if not isinstance(response, AnsweredResponse) or response.correlation_id:
        return response
    correlation: str | None = (
        request_uuid if uses_placeholder(options.request_template, "uuid") else req.session_id
    )
    return response.model_copy(update={"correlation_id": correlation})


def scrub_text(text: str, secrets: Sequence[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


def redact_json(value: Any, secrets: Sequence[str]) -> Any:
    """Recursively remove secrets and any Authorization-like fields from stored evidence."""

    if isinstance(value, dict):
        return {
            key: (
                "***" if key.casefold() in _SENSITIVE_HEADER_NAMES else redact_json(item, secrets)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_json(item, secrets) for item in value]
    if isinstance(value, str):
        return scrub_text(value, secrets)
    return value


def mapping_error(status: int, problem: str) -> ErrorResponse:
    return ErrorResponse(error_type="MappingError", message=f"HTTP {status}: {problem}")


def host(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.hostname or ''}"


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


def _json_at(document: Any, path: str | None) -> Any:
    if path is None:
        return None
    found = lookup(document, path)
    return None if found is _MISSING else found


def _message_at(matched: Any, document: Any, path: str | None) -> str | None:
    fallback = _text_at(document, path) if matched is not document else None
    return _text_at(matched, path) or fallback


def _artifact_content(found: Any) -> str | dict[str, Any] | list[Any]:
    return found if isinstance(found, str | dict | list) else json.dumps(found)
