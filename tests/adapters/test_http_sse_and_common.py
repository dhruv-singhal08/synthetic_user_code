"""Server-Sent Events adapter and the HTTP helpers shared with http_json."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from synthetic_user.adapters.base import ProductRequest
from synthetic_user.adapters.http_common import (
    HttpRequest,
    redact_json,
    render_template,
    resolve_ca_bundle,
    resolve_headers,
    scrub_text,
)
from synthetic_user.adapters.http_sse import (
    HttpSseAdapter,
    SseHttpResult,
    SseStreamOptions,
    aggregate_events,
    parse_sse_events,
)
from synthetic_user.adapters.registry import create_adapter
from synthetic_user.core.config import AppConfig, ProductAdapterSettings
from synthetic_user.core.exceptions import ConfigError
from synthetic_user.models.execution import (
    AnsweredResponse,
    ClarificationResponse,
    ErrorResponse,
    TimeoutResponse,
)

TOKEN = "sse-secret-token-42"
OPTIONS: dict[str, Any] = {
    "url": "https://agent.example.com/stream",
    "headers": {"Authorization": "Bearer ${AGENT_TOKEN}"},
    "timeout": 5,
    "request_template": {"query": "{utterance}", "session": "{session_id}"},
    "stream": {},
    "response_mapping": {
        "text": "final.text",
        "artifacts": [{"kind": "tool_calls", "path": "tool.calls"}],
        "clarification_when": "event == clarify",
        "clarification_message": "message",
    },
}


def _frames(*events: dict[str, Any], done: bool = True) -> list[str]:
    lines: list[str] = []
    for event in events:
        lines += [f"data: {json.dumps(event)}", ""]
    if done:
        lines += ["data: [DONE]", ""]
    return lines


class StubSse:
    def __init__(self, result: SseHttpResult | BaseException, delay: float = 0) -> None:
        self.result, self.delay = result, delay
        self.sent: list[HttpRequest] = []

    async def __call__(self, request: HttpRequest) -> SseHttpResult:
        self.sent.append(request)
        await asyncio.sleep(self.delay)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _adapter(transport: StubSse, **overrides: Any) -> HttpSseAdapter:
    return HttpSseAdapter.from_options(
        {**OPTIONS, **overrides}, env={"AGENT_TOKEN": TOKEN}, transport=transport
    )


def _req(utterance: str = "Top complaint themes?") -> ProductRequest:
    return ProductRequest(question_id="Q_001", utterance=utterance, session_id="S-1")


# --- SSE parsing ---------------------------------------------------------------------------


def test_events_parsed_in_order_until_done_skipping_bad_payloads() -> None:
    lines = [
        'data: {"event": "tool", "n": 1}',
        "",
        "data: not json",
        "",
        "data: [1, 2]",
        "",
        'data: {"event": "final",',
        'data:  "text": "multi-line"}',
        "",
        "data: [DONE]",
        "",
        'data: {"event": "after-done"}',
    ]
    events = parse_sse_events(lines, SseStreamOptions())
    assert events == [{"event": "tool", "n": 1}, {"event": "final", "text": "multi-line"}]


def test_aggregate_keeps_last_event_per_name_and_full_list() -> None:
    events = [{"event": "tool", "n": 1}, {"event": "tool", "n": 2}, {"x": 1}]
    document = aggregate_events(events, "event")
    assert document["tool"] == {"event": "tool", "n": 2}
    assert document["_events"] == events


# --- adapter -------------------------------------------------------------------------------


async def test_answer_and_tool_calls_mapped_from_stream() -> None:
    transport = StubSse(
        SseHttpResult(
            200,
            _frames(
                {"event": "tool", "calls": [{"name": "aggregate_feedback"}]},
                {"event": "final", "text": "Delays are the top theme."},
            ),
        )
    )
    response = await _adapter(transport).send(_req())
    assert isinstance(response, AnsweredResponse)
    assert response.text == "Delays are the top theme."
    assert [a.kind for a in response.artifacts] == ["tool_calls"]
    sent = transport.sent[0]
    assert sent.headers["Accept"] == "text/event-stream"
    assert sent.headers["Authorization"] == f"Bearer {TOKEN}"
    assert json.loads(sent.body or b"{}") == {"query": "Top complaint themes?", "session": "S-1"}


async def test_clarification_found_in_an_earlier_event() -> None:
    transport = StubSse(
        SseHttpResult(200, _frames({"event": "clarify", "message": "Which period?"}))
    )
    response = await _adapter(transport).send(_req())
    assert isinstance(response, ClarificationResponse) and response.message == "Which period?"


async def test_stream_without_the_mapped_answer_is_a_mapping_error() -> None:
    transport = StubSse(SseHttpResult(200, _frames({"event": "tool", "calls": [{"n": 1}]})))
    response = await _adapter(transport).send(_req())
    assert isinstance(response, ErrorResponse) and "no answer event" in response.message


async def test_auth_failure_and_http_error() -> None:
    expired = await _adapter(StubSse(SseHttpResult(401, []))).send(_req())
    assert isinstance(expired, ErrorResponse) and expired.error_type == "AuthenticationError"
    broken = await _adapter(StubSse(SseHttpResult(500, [f"boom {TOKEN}"]))).send(_req())
    assert isinstance(broken, ErrorResponse) and broken.error_type == "HttpError"
    assert TOKEN not in broken.message


async def test_slow_stream_is_a_timeout_response() -> None:
    transport = StubSse(SseHttpResult(200, _frames()), delay=1)
    response = await _adapter(transport, timeout=0.05).send(_req())
    assert isinstance(response, TimeoutResponse) and response.timeout_seconds == 0.05


async def test_echoed_credentials_never_reach_answer_or_raw_events() -> None:
    transport = StubSse(
        SseHttpResult(200, _frames({"event": "final", "text": f"token is {TOKEN}"}))
    )
    response = await _adapter(transport).send(_req())
    assert isinstance(response, AnsweredResponse) and response.text == "token is ***"
    assert TOKEN not in json.dumps(response.raw_response)


def test_registry_builds_the_sse_adapter(app_config: AppConfig) -> None:
    settings = app_config.model_copy(
        update={"product_adapter": ProductAdapterSettings(type="http_sse", options=OPTIONS)}
    )
    adapter = create_adapter(
        settings, env={"AGENT_TOKEN": TOKEN}, sse_transport=StubSse(SseHttpResult(200, []))
    )
    assert isinstance(adapter, HttpSseAdapter) and TOKEN not in repr(adapter)


def test_sse_options_require_a_stream_block() -> None:
    options = {k: v for k, v in OPTIONS.items() if k != "stream"}
    with pytest.raises(ConfigError, match="stream"):
        HttpSseAdapter.from_options(options, env={}, transport=StubSse(SseHttpResult(200, [])))


# --- shared helpers ------------------------------------------------------------------------


def test_headers_resolved_from_env_with_secrets_collected() -> None:
    resolved = resolve_headers(
        {"Authorization": "Bearer ${T}", "X-Api-Key": "literal-key", "Accept": "*/*"},
        {"T": "abc"},
    )
    assert resolved.values["Authorization"] == "Bearer abc"
    # env values AND sensitive header values are scrubbed (longest first)
    assert set(resolved.secrets) == {"Bearer abc", "literal-key", "abc"}
    assert [len(s) for s in resolved.secrets] == sorted(map(len, resolved.secrets), reverse=True)
    with pytest.raises(ConfigError, match="MISSING") as info:
        resolve_headers({"Authorization": "${MISSING}"}, {})
    assert "Authorization" in str(info.value)


def test_token_written_in_the_contract_needs_no_env_and_is_masked() -> None:
    resolved = resolve_headers(
        {
            "Authorization": "Bearer literal-blue-token",
            "X-Auth-Token": "second-literal",
            "Accept": "application/json",
        },
        {},
    )
    assert resolved.values["Authorization"] == "Bearer literal-blue-token"
    assert set(resolved.secrets) == {
        "Bearer literal-blue-token",
        "literal-blue-token",  # the bare credential is masked too
        "second-literal",
    }
    echoed = "auth=Bearer literal-blue-token token=literal-blue-token x=second-literal json"
    assert scrub_text(echoed, resolved.secrets) == "auth=*** token=*** x=*** json"


def test_ca_bundle_resolution_keeps_verification_on(tmp_path: Path) -> None:
    assert resolve_ca_bundle(None, {}) is None
    bundle = tmp_path / "ca.pem"
    bundle.write_text("cert", encoding="utf-8")
    assert resolve_ca_bundle("${CA}", {"CA": str(bundle)}) == str(bundle)
    with pytest.raises(ConfigError, match="not found"):
        resolve_ca_bundle(str(tmp_path / "missing.pem"), {})
    with pytest.raises(ConfigError, match="CA"):
        resolve_ca_bundle("${CA}", {})


def test_amexcerts_bundle_needs_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_amexcerts(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "amexcerts":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_amexcerts)
    monkeypatch.delitem(__import__("sys").modules, "amexcerts", raising=False)
    with pytest.raises(ConfigError, match="amexcerts package is not installed"):
        resolve_ca_bundle("amexcerts", {})


def test_redact_json_masks_auth_fields_and_secret_values() -> None:
    document = {"Authorization": "Bearer x", "nested": [{"note": "key=s3cret"}], "n": 1}
    assert redact_json(document, ("s3cret",)) == {
        "Authorization": "***",
        "nested": [{"note": "key=***"}],
        "n": 1,
    }


def test_template_shares_one_uuid_and_keeps_raw_placeholders() -> None:
    rendered = render_template(
        {"session": "{uuid}", "trace": "t-{uuid}", "sid": "{session_id}"},
        ProductRequest(question_id="Q_001", utterance="hi"),
    )
    assert rendered["trace"] == f"t-{rendered['session']}" and rendered["sid"] is None
