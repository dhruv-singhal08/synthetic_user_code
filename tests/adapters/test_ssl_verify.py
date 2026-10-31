"""The SSL on / off flag: ``product.ssl_verify`` (app config) and ``ssl_verify`` (contract)."""

from __future__ import annotations

import http.server
import json
import shutil
import ssl
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from synthetic_user.adapters.base import ProductRequest
from synthetic_user.adapters.http_common import ssl_context, tls_settings
from synthetic_user.adapters.http_json import HttpJsonOptions, HttpRequest, HttpResult, _urllib_send
from synthetic_user.adapters.registry import create_adapter
from synthetic_user.core.config import AppConfig, ProductAdapterSettings, ProductSettings
from synthetic_user.core.exceptions import ConfigError, ProductAdapterError

OPTIONS: dict[str, Any] = {
    "url": "https://product.example/v1/ask",
    "tls_ca_bundle": "${MISSING_CA}",  # resolving it would fail: proves it is skipped when off
    "response_mapping": {"text": "answer"},
}


def _request(*, verify: bool, url: str = "https://product.example/") -> HttpRequest:
    return HttpRequest(method="GET", url=url, headers={}, body=None, timeout=5, ssl_verify=verify)


def test_ssl_context_default_verify_and_off() -> None:
    assert ssl_context(_request(verify=True)) is None  # urllib / aiohttp default: verify
    off = ssl_context(_request(verify=False))
    assert off is not None
    assert off.verify_mode is ssl.CERT_NONE and off.check_hostname is False


def test_verification_is_on_unless_contract_or_app_turns_it_off() -> None:
    options = HttpJsonOptions.model_validate(OPTIONS)
    with pytest.raises(ConfigError, match="MISSING_CA"):  # on: the bundle is resolved
        tls_settings(options, {}, ssl_verify=True)
    assert tls_settings(options, {}, ssl_verify=False) == (False, None)  # app config off
    contract_off = HttpJsonOptions.model_validate({**OPTIONS, "ssl_verify": False})
    assert tls_settings(contract_off, {}, ssl_verify=True) == (False, None)  # contract off
    assert HttpJsonOptions.model_validate({**OPTIONS, "tls_ca_bundle": None}).ssl_verify is True


class _Stub:
    def __init__(self) -> None:
        self.requests: list[HttpRequest] = []

    async def __call__(self, request: HttpRequest) -> HttpResult:
        self.requests.append(request)
        return HttpResult(200, json.dumps({"answer": "ok"}).encode())


async def test_app_config_flag_reaches_the_request(app_config: AppConfig) -> None:
    settings = app_config.model_copy(
        update={
            "product": ProductSettings(ssl_verify=False),
            "product_adapter": ProductAdapterSettings.model_validate(
                {"type": "http_json", "options": OPTIONS}
            ),
        }
    )
    stub = _Stub()
    adapter = create_adapter(settings, env={}, transport=stub)
    await adapter.send(ProductRequest(question_id="Q_001", utterance="hi"))
    assert stub.requests[0].ssl_verify is False and stub.requests[0].tls_ca_bundle is None
    assert settings.effective_config()["product"]["ssl_verify"] is False  # in the run manifest


@pytest.fixture
def self_signed_server(tmp_path: Path) -> Iterator[str]:
    """A local https server with a self-signed certificate (needs the openssl CLI)."""
    if shutil.which("openssl") is None:
        pytest.skip("openssl CLI not available")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            *("openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1"),
            *("-subj", "/CN=localhost", "-keyout", str(key), "-out", str(cert)),
        ],
        check=True,
        capture_output=True,
    )

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"answer": "ok"}')

        def log_message(self, *args: Any) -> None:
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()


def test_self_signed_certificate_is_refused_unless_verification_is_off(
    self_signed_server: str,
) -> None:
    with pytest.raises(ProductAdapterError, match="CERTIFICATE_VERIFY_FAILED"):
        _urllib_send(_request(verify=True, url=self_signed_server))
    result = _urllib_send(_request(verify=False, url=self_signed_server))
    assert result.status == 200 and json.loads(result.body) == {"answer": "ok"}
