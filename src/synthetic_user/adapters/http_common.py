"""Shared configuration, mapping, templating, and redaction for HTTP product adapters."""

from __future__ import annotations

import importlib
import json
import logging
import re
import ssl
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from synthetic_user.adapters.base import ProductRequest
from synthetic_user.core.exceptions import ConfigError
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
# Characters of an HTTP error body quoted in the ErrorResponse message.
ERROR_SNIPPET_CHARS = 200
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_PLACEHOLDERS = ("utterance", "session_id", "uuid")
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

    @field_validator("clarification_when", "denied_when", "error_when")
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

    @field_validator("url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError("url must be an http(s) URL")
        return value


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


def render_template(template: Any, req: ProductRequest) -> Any:
    """Replace supported placeholders, sharing one fresh UUID across the request."""

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
        return AnsweredResponse(text=text, artifacts=artifacts, product_version=version)

    @staticmethod
    def _matched(
        condition: Condition | None, document: Any, alternatives: Sequence[Any]
    ) -> Any | None:
        if condition is None:
            return None
        if condition.matches(document):
            return document
        return next((item for item in reversed(alternatives) if condition.matches(item)), None)


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


def _message_at(matched: Any, document: Any, path: str | None) -> str | None:
    fallback = _text_at(document, path) if matched is not document else None
    return _text_at(matched, path) or fallback


def _artifact_content(found: Any) -> str | dict[str, Any] | list[Any]:
    return found if isinstance(found, str | dict | list) else json.dumps(found)
