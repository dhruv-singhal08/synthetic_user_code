"""Product login of the HTTP adapters (D263): token from a sign-in cookie, renewed in a run."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs

import pytest

from synthetic_user.adapters.auth import (
    FormLoginAuth,
    LoginRequest,
    LoginResult,
    TokenProvider,
    jwt_expiry,
    urllib_login,
)
from synthetic_user.adapters.base import ProductRequest
from synthetic_user.adapters.http_common import HttpRequest, HttpResult
from synthetic_user.adapters.http_json import HttpJsonAdapter
from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import ConfigError, ProductAuthError
from synthetic_user.models.question import QuestionCase
from synthetic_user.nodes.executor.executor import NOT_SENT, ProductExecutor

LOGIN_URL = "https://login.example.com/ssoi/signin"
CREDENTIALS = {"BLUE_USER": "tester", "BLUE_PASSWORD": "pa55-secret"}
OPTIONS: dict[str, Any] = {
    "url": "https://product.example.com/generate/sql/",
    "auth": {"url": LOGIN_URL},
    "headers": {"Authorization": "Bearer ${BLUE_TOKEN}"},
    "request_template": {"input": "{utterance}"},
    "response_mapping": {"text": "message", "artifacts": [{"kind": "sql", "path": "result"}]},
}


def jwt(exp: float) -> str:
    """An unsigned JWT-shaped token with an ``exp`` claim (only the expiry is read)."""
    claims = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"eyJhbGciOiJIUzI1NiJ9.{claims}.sig"


class Login:
    """A stub sign-in: hands out token-1, token-2, ... (or no cookie when ``reject``)."""

    def __init__(self, reject: bool = False) -> None:
        self.calls: list[LoginRequest] = []
        self.reject = reject

    async def __call__(self, request: LoginRequest) -> LoginResult:
        self.calls.append(request)
        await asyncio.sleep(0)
        if self.reject:
            return LoginResult(401, {})
        return LoginResult(200, {"bluetoken": f"token-{len(self.calls)}"})


class Product:
    """A stub product: 401 for tokens in ``rejected``, else an SQL answer echoing nothing."""

    def __init__(self, rejected: set[str] | None = None) -> None:
        self.rejected = rejected or set()
        self.tokens: list[str] = []

    async def __call__(self, request: HttpRequest) -> HttpResult:
        token = request.headers["Authorization"].removeprefix("Bearer ")
        self.tokens.append(token)
        await asyncio.sleep(0)
        if token in self.rejected:
            return HttpResult(401, b'{"error": "token expired"}')
        body = {"type": "sql", "result": "SELECT 1", "message": f"ok ({token})"}
        return HttpResult(200, json.dumps(body).encode())


def adapter(product: Product, login: Login, env: dict[str, str] | None = None) -> HttpJsonAdapter:
    return HttpJsonAdapter.from_options(
        OPTIONS,
        env=CREDENTIALS if env is None else env,
        transport=product,
        login_transport=login,
    )


def ask(question_id: str = "Q_001") -> ProductRequest:
    return ProductRequest(question_id=question_id, utterance="Top spenders?")


# --- the token ------------------------------------------------------------------------------


def test_jwt_expiry_is_read_without_verification() -> None:
    assert jwt_expiry(jwt(1_800_000_000)) == 1_800_000_000
    assert jwt_expiry("not-a-jwt") is None
    assert jwt_expiry("a.%%%.c") is None


def test_missing_credentials_fail_before_any_call() -> None:
    with pytest.raises(ConfigError) as caught:
        adapter(Product(), Login(), env={"BLUE_USER": "tester"})
    assert "BLUE_USER and BLUE_PASSWORD" in str(caught.value)


async def test_one_login_for_concurrent_calls_and_the_token_fills_the_header() -> None:
    product, login = Product(), Login()
    product_adapter = adapter(product, login)
    responses = await asyncio.gather(*(product_adapter.send(ask(f"Q_{i}")) for i in range(5)))
    assert len(login.calls) == 1
    assert dict(login.calls[0].form) == {"userid": "tester", "pwd": "pa55-secret"}
    assert login.calls[0].url == LOGIN_URL
    assert product.tokens == ["token-1"] * 5
    # The token is a secret like any header value: scrubbed from what is stored.
    assert all("token-1" not in r.model_dump_json() for r in responses)


async def test_the_token_is_renewed_before_it_expires() -> None:
    now = [1_000_000.0]
    tokens = iter([jwt(now[0] + 3600), jwt(now[0] + 7200)])

    async def login(request: LoginRequest) -> LoginResult:
        return LoginResult(200, {"bluetoken": next(tokens)})

    provider = TokenProvider(
        FormLoginAuth(url=LOGIN_URL),
        CREDENTIALS,
        ssl_verify=True,
        tls_ca_bundle=None,
        transport=login,
        clock=lambda: now[0],
    )
    first = await provider.token()
    now[0] += 3000  # 10 minutes left: still used
    assert await provider.token() == first
    now[0] += 400  # inside the 5-minute margin: renewed
    assert await provider.token() != first
    assert provider.expires_at == 1_000_000.0 + 7200


async def test_a_rejected_token_is_renewed_once_and_the_call_retried() -> None:
    product, login = Product(rejected={"token-1"}), Login()
    response = await adapter(product, login).send(ask())
    assert response.outcome == "answered"
    assert len(login.calls) == 2 and product.tokens == ["token-1", "token-2"]


async def test_a_fresh_token_rejected_too_stops_without_more_logins() -> None:
    product, login = Product(rejected={"token-1", "token-2"}), Login()
    product_adapter = adapter(product, login)
    with pytest.raises(ProductAuthError, match="rejected a freshly issued login token"):
        await product_adapter.send(ask())
    with pytest.raises(ProductAuthError):
        await product_adapter.send(ask("Q_002"))
    assert len(login.calls) == 2  # no third login


async def test_a_failed_login_is_final_and_never_shows_the_password() -> None:
    login = Login(reject=True)
    product_adapter = adapter(Product(), login)
    with pytest.raises(ProductAuthError) as caught:
        await product_adapter.send(ask())
    assert "check the user ID and password" in str(caught.value)
    assert "pa55-secret" not in str(caught.value) and "pa55-secret" not in repr(login.calls[0])
    with pytest.raises(ProductAuthError):
        await product_adapter.send(ask("Q_002"))
    assert len(login.calls) == 1  # a wrong password is never retried (account lockout)


# --- the executor ---------------------------------------------------------------------------


async def test_after_a_failed_login_the_remaining_questions_are_not_sent(
    app_config: AppConfig, sample_scenarios: Any, sample_packs: Any
) -> None:
    from nodes.test_evaluator import _question

    question = _question(sample_scenarios[0], sample_packs[0])
    questions = [
        QuestionCase.model_validate(
            {**question.model_dump(mode="json", by_alias=True), "question_id": f"Q_00{i}"}
        )
        for i in range(1, 4)
    ]
    product, login = Product(), Login(reject=True)
    records = await ProductExecutor(
        adapter(product, login), app_config, logging.getLogger("t")
    ).execute_many(questions, product_id="p")
    assert [r.outcome for r in records] == ["error"] * 3
    assert records[0].response.error_type == "ProductAuthError"  # type: ignore[union-attr]
    assert [r.response.error_type for r in records[1:]] == ["NotSent", "NotSent"]  # type: ignore[union-attr]
    assert records[1].response.message.startswith(NOT_SENT.split("{", 1)[0])  # type: ignore[union-attr]
    assert len(login.calls) == 1 and product.tokens == []


# --- the real sign-in call (local server: form POST, redirect, cookie) -----------------------


@pytest.fixture
def signin_server() -> Iterator[str]:
    """POST /signin with the right form -> 302 + Set-Cookie bluetoken -> /home (200)."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return None

        def do_POST(self) -> None:
            form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
            ok = form == {"userid": ["tester"], "pwd": ["pa55-secret"]}
            assert self.headers["Content-Type"] == "application/x-www-form-urlencoded"
            self.send_response(302 if ok else 401)
            if ok:
                self.send_header("Set-Cookie", "bluetoken=REAL-TOKEN; Path=/; HttpOnly")
                self.send_header("Location", "/home")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/signin"
    server.shutdown()
    thread.join()


async def test_urllib_login_reads_the_cookie_across_the_redirect(signin_server: str) -> None:
    good = await urllib_login(
        LoginRequest(url=signin_server, form={"userid": "tester", "pwd": "pa55-secret"}, timeout=5)
    )
    assert good.cookies.get("bluetoken") == "REAL-TOKEN"
    bad = await urllib_login(
        LoginRequest(url=signin_server, form={"userid": "tester", "pwd": "wrong"}, timeout=5)
    )
    assert bad.status == 401 and "bluetoken" not in bad.cookies
