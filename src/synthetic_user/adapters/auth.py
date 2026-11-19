"""Product login for the HTTP adapters (D263): a token that renews itself during a run.

A contract may add ``options.auth`` (``type: form_login``): the adapter signs in by POSTing a
form (user id + password) to the login URL, takes the token from a response cookie and uses it
for ``${<token_variable>}`` in the headers instead of the environment variable. The token is
renewed ``refresh_before_seconds`` before it expires (the ``exp`` claim when the token is a JWT,
else after ``fallback_lifetime_seconds``); a product 401 / 403 renews it once more.

Credentials never come from the contract (it is stored with the run and tracked by git): the
user id, password and an optional login-URL override are read from the environment variables the
section names (CLI), or from the values the UI keeps in server memory only. They are never
written to a file, log, trace or result.

A failed login is final for the adapter (no further attempts): retrying a wrong password, once
per question in flight, could lock the account. The executor then stops sending questions.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import http.cookiejar
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from synthetic_user.adapters.http_common import HttpRequest, host, ssl_context
from synthetic_user.core.exceptions import ConfigError, ProductAuthError
from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.execution import ProductResponse

_log = logging.getLogger(__name__)
_ENV_NAME = r"^[A-Za-z_][A-Za-z0-9_]*$"
# Product statuses that mean "the token is not accepted": renew once, then stop.
REJECTED_STATUSES = frozenset({401, 403})
MISSING_CREDENTIALS = (
    "Product login needs a user ID and password: log in under Product connection (UI) or set "
    "the environment variables {user_env} and {password_env} (CLI)"
)


class FormLoginAuth(StrictModel):
    """``options.auth`` of an HTTP contract: sign in with a form, token from a cookie."""

    type: Literal["form_login"] = "form_login"
    url: NonEmptyStr  # the login endpoint; the env var ``url_env`` (or the UI) may override it
    user_field: NonEmptyStr = "userid"
    password_field: NonEmptyStr = "pwd"
    token_cookie: NonEmptyStr = "bluetoken"
    token_variable: NonEmptyStr = Field(default="BLUE_TOKEN", pattern=_ENV_NAME)
    user_env: NonEmptyStr = Field(default="BLUE_USER", pattern=_ENV_NAME)
    password_env: NonEmptyStr = Field(default="BLUE_PASSWORD", pattern=_ENV_NAME)
    url_env: NonEmptyStr = Field(default="BLUE_LOGIN_URL", pattern=_ENV_NAME)
    refresh_before_seconds: int = Field(default=300, ge=0)
    fallback_lifetime_seconds: int = Field(default=3000, ge=60)
    timeout: float = Field(default=30.0, gt=0)

    @field_validator("url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        _check_url(value)
        return value


@dataclass(frozen=True)
class LoginRequest:
    url: str
    form: Mapping[str, str] = field(repr=False)  # holds the password: never printed
    timeout: float
    ssl_verify: bool = True
    tls_ca_bundle: str | None = None


@dataclass(frozen=True)
class LoginResult:
    status: int
    cookies: Mapping[str, str] = field(repr=False)


LoginTransport = Callable[[LoginRequest], Awaitable[LoginResult]]


async def urllib_login(request: LoginRequest) -> LoginResult:
    """The form POST in a worker thread; cookies of every redirect hop are collected."""
    return await asyncio.to_thread(_urllib_login, request)


def _urllib_login(request: LoginRequest) -> LoginResult:
    jar = http.cookiejar.CookieJar()
    tls = ssl_context(
        HttpRequest(
            method="POST",
            url=request.url,
            headers={},
            body=None,
            timeout=request.timeout,
            tls_ca_bundle=request.tls_ca_bundle,
            ssl_verify=request.ssl_verify,
        )
    )
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar), urllib.request.HTTPSHandler(context=tls)
    )
    prepared = urllib.request.Request(
        request.url,
        data=urllib.parse.urlencode(dict(request.form)).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    target = host(request.url)
    try:
        with opener.open(prepared, timeout=request.timeout) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except TimeoutError as exc:
        raise ProductAuthError(f"login to {target} timed out") from exc
    except urllib.error.URLError as exc:
        raise ProductAuthError(f"login to {target} failed: {exc.reason}") from exc
    except OSError as exc:
        raise ProductAuthError(f"login to {target} failed: {type(exc).__name__}") from exc
    return LoginResult(status, {cookie.name: cookie.value or "" for cookie in jar})


class TokenProvider:
    """One product token for one adapter (shared by its concurrent calls)."""

    def __init__(
        self,
        auth: FormLoginAuth,
        env: Mapping[str, str],
        *,
        ssl_verify: bool,
        tls_ca_bundle: str | None,
        transport: LoginTransport = urllib_login,
        clock: Callable[[], float] = time.time,
    ) -> None:
        user, password = env.get(auth.user_env, ""), env.get(auth.password_env, "")
        if not user or not password:
            raise ConfigError(
                MISSING_CREDENTIALS.format(user_env=auth.user_env, password_env=auth.password_env)
            )
        self._auth = auth
        self._user = user
        self._password = password
        self.url = env.get(auth.url_env, "").strip() or auth.url
        try:
            _check_url(self.url)
        except ValueError as exc:
            raise ConfigError(f"product login URL {self.url!r}: {exc}") from None
        self._ssl_verify = ssl_verify
        self._tls_ca_bundle = tls_ca_bundle
        self._transport = transport
        self._clock = clock
        self._lock = asyncio.Lock()
        self._token: str | None = None
        self._expires_at = 0.0
        self._failure: str | None = None

    @property
    def user(self) -> str:
        return self._user

    @property
    def variable(self) -> str:
        return self._auth.token_variable

    @property
    def expires_at(self) -> float:
        return self._expires_at

    async def token(self) -> str:
        """The current token, signing in first when there is none or it is about to expire."""
        async with self._lock:
            self._raise_if_failed()
            if self._token is None or self._clock() >= self._expires_at - (
                self._auth.refresh_before_seconds
            ):
                await self._login()
            assert self._token is not None
            return self._token

    async def renew(self, rejected: str) -> str:
        """After a product 401 / 403 with ``rejected``: sign in again, unless another call
        already did (only one new login for all the calls in flight)."""
        async with self._lock:
            self._raise_if_failed()
            if self._token == rejected:
                await self._login()
            assert self._token is not None
            return self._token

    def fail(self, message: str) -> None:
        """The product rejected even a fresh token: stop signing in for this adapter."""
        self._failure = message

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise ProductAuthError(self._failure)

    async def _login(self) -> None:
        request = LoginRequest(
            url=self.url,
            form={self._auth.user_field: self._user, self._auth.password_field: self._password},
            timeout=self._auth.timeout,
            ssl_verify=self._ssl_verify,
            tls_ca_bundle=self._tls_ca_bundle,
        )
        try:
            result = await self._transport(request)
        except ProductAuthError as exc:
            self._failure = str(exc)
            raise
        token = result.cookies.get(self._auth.token_cookie, "")
        if not token:
            self._failure = (
                f"login to {host(self.url)} as {self._user} failed: HTTP {result.status}, no "
                f"{self._auth.token_cookie!r} cookie in the reply (check the user ID and password)"
            )
            raise ProductAuthError(self._failure)
        now = self._clock()
        expiry = jwt_expiry(token)
        fallback = now + self._auth.fallback_lifetime_seconds
        self._expires_at = expiry if expiry is not None and expiry > now else fallback
        self._token = token
        _log.info(
            "Product login to %s as %s: token valid until %s",
            host(self.url),
            self._user,
            datetime.fromtimestamp(self._expires_at).strftime("%H:%M:%S"),
        )


SendOnce = Callable[[Mapping[str, str]], Awaitable[tuple[ProductResponse, int | None]]]


async def send_with_auth(
    auth: TokenProvider | None, env: Mapping[str, str], send_once: SendOnce
) -> ProductResponse:
    """One product call; with a login, ``${token_variable}`` comes from the token and a
    401 / 403 renews it and retries once. Without a login: exactly one call with ``env``."""
    if auth is None:
        response, _status = await send_once(env)
        return response
    token = await auth.token()
    response, status = await send_once({**env, auth.variable: token})
    if status not in REJECTED_STATUSES:
        return response
    renewed = await auth.renew(token)
    response, status = await send_once({**env, auth.variable: renewed})
    if status in REJECTED_STATUSES:
        message = f"the product rejected a freshly issued login token (HTTP {status})"
        auth.fail(message)
        raise ProductAuthError(message)
    return response


async def env_with_token(auth: TokenProvider | None, env: Mapping[str, str]) -> Mapping[str, str]:
    """``env`` plus the current token (e.g. for a context API that needs it too)."""
    return env if auth is None else {**env, auth.variable: await auth.token()}


def jwt_expiry(token: str) -> float | None:
    """The ``exp`` claim of a JWT (read, not verified: it only schedules the renewal)."""
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, binascii.Error):
        return None
    exp = claims.get("exp") if isinstance(claims, dict) else None
    return float(exp) if isinstance(exp, int | float) and not isinstance(exp, bool) else None


def _check_url(value: str) -> None:
    parts = urllib.parse.urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("auth url must be an http(s) URL")
