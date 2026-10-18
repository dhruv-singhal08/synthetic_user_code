from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from synthetic_user.adapters.base import ProductAdapter, ProductRequest
from synthetic_user.adapters.fixture import FixtureAdapter
from synthetic_user.adapters.http_json import (
    HttpJsonAdapter,
    HttpRequest,
    HttpResult,
    lookup,
    render_template,
)
from synthetic_user.adapters.registry import create_adapter
from synthetic_user.core.config import AppConfig, ProductAdapterSettings, load_config
from synthetic_user.core.exceptions import ConfigError, ProductAdapterError
from synthetic_user.models.execution import (
    AnsweredResponse,
    ClarificationResponse,
    DeniedResponse,
    ErrorResponse,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SECRET = "s3cr3t-token-value"
ENV = {"PRODUCT_API_TOKEN": SECRET}

HTTP_OPTIONS: dict[str, Any] = {
    "url": "https://product.example/v1/ask",
    "headers": {"Authorization": "Bearer ${PRODUCT_API_TOKEN}", "X-Team": "finance"},
    "request_template": {"question": "{utterance}", "session": "{session_id}", "v": 2},
    "response_mapping": {
        "text": "data.answer",
        "artifacts": [{"kind": "sql", "path": "data.sql"}, {"kind": "table", "path": "data.rows"}],
        "clarification_when": "data.type == clarification",
        "denied_when": "data.status == 'unsupported'",
        "product_version": "meta.version",
    },
}


def _settings(app_config: AppConfig, type_: str, options: dict[str, Any]) -> AppConfig:
    return app_config.model_copy(
        update={
            "product_adapter": ProductAdapterSettings.model_validate(
                {"type": type_, "options": options}
            )
        }
    )


class StubTransport:
    """Local stand-in for the network: records requests, replies with a canned result."""

    def __init__(self, status: int = 200, body: Any = None, raw: bytes | None = None) -> None:
        self.requests: list[HttpRequest] = []
        self._result = HttpResult(status, raw if raw is not None else json.dumps(body).encode())

    async def __call__(self, request: HttpRequest) -> HttpResult:
        self.requests.append(request)
        return self._result


def _http(transport: StubTransport, **overrides: Any) -> HttpJsonAdapter:
    return HttpJsonAdapter.from_options({**HTTP_OPTIONS, **overrides}, env=ENV, transport=transport)


def _req(utterance: str = "Who spent most?", session_id: str | None = None) -> ProductRequest:
    return ProductRequest(question_id="Q_001", utterance=utterance, session_id=session_id)


# --- registry -----------------------------------------------------------------------------------


def test_registry_selects_fixture_from_repo_config(app_config: AppConfig) -> None:
    assert app_config.product_adapter.file is None  # inline in config/app.yaml
    adapter = create_adapter(app_config)
    assert isinstance(adapter, FixtureAdapter) and isinstance(adapter, ProductAdapter)


def test_registry_selects_http_json(app_config: AppConfig) -> None:
    adapter = create_adapter(_settings(app_config, "http_json", HTTP_OPTIONS), env=ENV)
    assert isinstance(adapter, HttpJsonAdapter) and adapter.name == "http_json"


def test_repo_fixture_adapter_file_matches_inline_config(app_config: AppConfig) -> None:
    raw = yaml.safe_load(
        (REPO_ROOT / "config/products/sample_nl2sql.adapter.yaml").read_text("utf-8")
    )
    assert raw == app_config.product_adapter.model_dump(mode="json", exclude={"file"})


def test_repo_http_json_example_is_valid(app_config: AppConfig) -> None:
    raw = yaml.safe_load(
        (REPO_ROOT / "config/products/sample_policy_chatbot.adapter.yaml").read_text("utf-8")
    )
    adapter = create_adapter(_settings(app_config, raw["type"], raw["options"]), env=ENV)
    assert isinstance(adapter, HttpJsonAdapter)


@pytest.mark.parametrize(
    ("type_", "options", "match"),
    [
        ("fixture", {}, "options for fixture"),
        ("fixture", {"path": "x.json", "extra": 1}, "options for fixture"),
        ("http_json", {"url": "ftp://x", "response_mapping": {"text": "a"}}, "http\\(s\\) URL"),
        ("http_json", {"url": "https://x", "response_mapping": {}}, "text path or at least one"),
        (
            "http_json",
            {"url": "https://x", "response_mapping": {"text": "a", "denied_when": "a = b"}},
            "path == value",
        ),
        ("python", {"class": "no_colon_here"}, "options for python"),
        ("python", {"class": "definitely_missing_module_xyz:Thing"}, "cannot import"),
    ],
)
def test_registry_bad_options_are_config_errors(
    app_config: AppConfig, type_: str, options: dict[str, Any], match: str
) -> None:
    with pytest.raises(ConfigError, match=match):
        create_adapter(_settings(app_config, type_, options), env=ENV)


def test_unknown_adapter_type_rejected_at_config_load(tmp_path: Path) -> None:
    data = yaml.safe_load((REPO_ROOT / "config" / "app.yaml").read_text(encoding="utf-8"))
    data["product_adapter"] = {"type": "grpc", "options": {}}
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="product_adapter.type"):
        load_config(path, env={})


CUSTOM_MODULE = textwrap.dedent(
    """
    from synthetic_user.models.execution import AnsweredResponse

    class EchoAdapter:
        name = "echo"

        def __init__(self, prefix: str) -> None:
            self.prefix = prefix

        async def send(self, req):
            return AnsweredResponse(text=self.prefix + req.utterance)

        async def aclose(self) -> None:
            return None

    class NotAnAdapter:
        def __init__(self) -> None:
            pass

    NOT_A_CLASS = 3
    """
)


@pytest.fixture
def custom_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    package = tmp_path / "team_adapters_c03"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "custom.py").write_text(CUSTOM_MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return "team_adapters_c03.custom"


async def test_registry_python_import_path(app_config: AppConfig, custom_module: str) -> None:
    settings = _settings(
        app_config, "python", {"class": f"{custom_module}:EchoAdapter", "init": {"prefix": "> "}}
    )
    adapter = create_adapter(settings)
    assert adapter.name == "echo"
    response = await adapter.send(_req("hello"))
    assert isinstance(response, AnsweredResponse) and response.text == "> hello"


@pytest.mark.parametrize(
    ("target", "init", "match"),
    [
        ("MissingClass", {}, "is not a class"),
        ("NOT_A_CLASS", {}, "is not a class"),
        ("NotAnAdapter", {}, "does not implement ProductAdapter"),
        ("EchoAdapter", {"wrong": 1}, "rejected options.init"),
    ],
)
def test_registry_python_bad_class(
    app_config: AppConfig, custom_module: str, target: str, init: dict[str, Any], match: str
) -> None:
    settings = _settings(app_config, "python", {"class": f"{custom_module}:{target}", "init": init})
    with pytest.raises(ConfigError, match=match):
        create_adapter(settings)


# --- http_json mapping (local stub transport, no network) -----------------------------------------


async def test_request_built_from_template_with_env_headers() -> None:
    transport = StubTransport(body={"data": {"answer": "Alice"}})
    await _http(transport).send(_req("Top spenders?", session_id="S1"))
    sent = transport.requests[0]
    assert (sent.method, sent.url, sent.timeout) == ("POST", HTTP_OPTIONS["url"], 30.0)
    assert sent.headers["Authorization"] == f"Bearer {SECRET}"
    assert sent.headers["X-Team"] == "finance"
    assert sent.body is not None
    assert json.loads(sent.body) == {"question": "Top spenders?", "session": "S1", "v": 2}


def test_template_placeholders() -> None:
    template = {"q": "{utterance}", "s": "{session_id}", "t": "ask: {utterance} ({session_id})"}
    assert render_template(template, _req("hi")) == {"q": "hi", "s": None, "t": "ask: hi ()"}
    assert render_template(["{utterance}", 1], _req("x")) == ["x", 1]


def test_uuid_placeholder_is_fresh_per_request_and_shared_within_one() -> None:
    template = {"session_id": "{uuid}", "trace": "t-{uuid}", "history": []}
    first = render_template(template, _req("hi"))
    second = render_template(template, _req("hi"))
    assert first["trace"] == f"t-{first['session_id']}" and first["history"] == []
    assert len(first["session_id"]) == 36 and first["session_id"] != second["session_id"]


async def test_get_sends_no_body() -> None:
    transport = StubTransport(body={"data": {"answer": "ok"}})
    await _http(transport, method="GET").send(_req())
    assert transport.requests[0].body is None


async def test_answer_maps_text_artifacts_and_version() -> None:
    body = {
        "data": {"answer": "Alice spent most", "sql": "SELECT 1", "rows": [{"n": "Alice"}]},
        "meta": {"version": 7},
    }
    response = await _http(StubTransport(body=body)).send(_req())
    assert isinstance(response, AnsweredResponse)
    assert response.text == "Alice spent most" and response.product_version == "7"
    assert [(a.kind, a.content) for a in response.artifacts] == [
        ("sql", "SELECT 1"),
        ("table", [{"n": "Alice"}]),
    ]


async def test_missing_artifact_path_is_skipped_not_invented() -> None:
    response = await _http(StubTransport(body={"data": {"answer": "text only"}})).send(_req())
    assert isinstance(response, AnsweredResponse) and response.artifacts == []


async def test_clarification_and_denied_rules() -> None:
    clarify = {"data": {"type": "clarification", "answer": "Which currency?"}}
    response = await _http(StubTransport(body=clarify)).send(_req())
    assert isinstance(response, ClarificationResponse) and response.message == "Which currency?"
    denied = {"data": {"status": "unsupported", "answer": "Forecasts are not supported."}}
    response = await _http(StubTransport(body=denied)).send(_req())
    assert isinstance(response, DeniedResponse)
    assert response.message == "Forecasts are not supported."


async def test_equality_uses_json_text_for_non_strings() -> None:
    mapping = {**HTTP_OPTIONS["response_mapping"], "denied_when": "data.ok == false"}
    adapter = _http(
        StubTransport(body={"data": {"ok": False, "answer": "no"}}), response_mapping=mapping
    )
    assert isinstance(await adapter.send(_req()), DeniedResponse)


@pytest.mark.parametrize(
    ("status", "raw", "error_type", "fragment"),
    [
        (503, b"Service Unavailable", "HttpError", "HTTP 503"),
        (404, b'{"error": "nope"}', "HttpError", "HTTP 404"),
        (200, b"<html>oops</html>", "NonJsonResponse", "HTTP 200"),
        (200, b'{"data": {}}', "MappingError", "no text or artifact"),
        (200, b'{"data": {"type": "clarification"}}', "MappingError", "no message"),
    ],
)
async def test_http_errors_and_bad_bodies_become_error_responses(
    status: int, raw: bytes, error_type: str, fragment: str
) -> None:
    response = await _http(StubTransport(status=status, raw=raw)).send(_req())
    assert isinstance(response, ErrorResponse)
    assert response.error_type == error_type
    assert response.message is not None and fragment in response.message


def test_lookup_dotted_paths() -> None:
    doc = {"a": {"b": [{"c": 1}, {"c": None}]}}
    assert lookup(doc, "a.b.0.c") == 1
    assert lookup(doc, "a.b.1.c") is None
    missing = lookup(doc, "a.x")
    assert missing is not None and lookup(doc, "a.b.5.c") is missing


# --- env vars and secrets ---------------------------------------------------------------------


def test_missing_env_var_names_variable_only() -> None:
    with pytest.raises(ConfigError, match="PRODUCT_API_TOKEN") as info:
        HttpJsonAdapter.from_options(HTTP_OPTIONS, env={}, transport=StubTransport())
    assert SECRET not in str(info.value)


async def test_secrets_never_logged_or_returned(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    echo = StubTransport(status=401, raw=f"bad token Bearer {SECRET}".encode())
    adapter = _http(echo)
    response = await adapter.send(_req())
    assert isinstance(response, ErrorResponse) and response.message is not None
    assert SECRET not in response.message and "***" in response.message
    assert SECRET not in repr(adapter) and SECRET not in repr(echo.requests[0])

    async def failing(request: HttpRequest) -> HttpResult:
        raise ProductAdapterError(f"connection reset after sending {SECRET}")

    broken = HttpJsonAdapter.from_options(HTTP_OPTIONS, env=ENV, transport=failing)
    with pytest.raises(ProductAdapterError) as info:
        await broken.send(_req())
    assert SECRET not in str(info.value)

    ok = _http(StubTransport(body={"data": {"answer": "fine"}}))
    await ok.send(_req())
    assert SECRET not in caplog.text
    assert "http_json POST https://product.example -> HTTP 200" in caplog.text


def test_manifest_config_keeps_env_templates_and_redacts_literals(app_config: AppConfig) -> None:
    options = {
        **HTTP_OPTIONS,
        "headers": {"Authorization": "Bearer ${PRODUCT_API_TOKEN}", "X-Key": "literal-key"},
    }
    effective = _settings(app_config, "http_json", options).effective_config()
    headers = effective["product_adapter"]["options"]["headers"]
    assert headers == {"Authorization": "Bearer ${PRODUCT_API_TOKEN}", "X-Key": "***"}


def test_adapter_file_loaded_and_file_key_reserved(tmp_path: Path) -> None:
    adapter_file = tmp_path / "p.adapter.yaml"
    adapter_file.write_text(yaml.safe_dump({"type": "http_json", "options": HTTP_OPTIONS}), "utf-8")
    data = yaml.safe_load((REPO_ROOT / "config" / "app.yaml").read_text(encoding="utf-8"))
    data["product_adapter"] = str(adapter_file)
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = load_config(path, env={})
    assert cfg.product_adapter.type == "http_json" and cfg.product_adapter.file == adapter_file
    adapter_file.write_text(yaml.safe_dump({"type": "fixture", "file": "x"}), "utf-8")
    with pytest.raises(ConfigError, match="'file' is set by the loader"):
        load_config(path, env={})
