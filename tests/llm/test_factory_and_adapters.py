from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from synthetic_user.core.config import AppConfig, LLMProvider, load_config
from synthetic_user.core.exceptions import ConfigError, LLMProviderError
from synthetic_user.llm import factory
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.factory import create_llm_client
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.llm.gemini_adapter import GeminiClient
from synthetic_user.llm.safechain_adapter import SafeChainClient

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "app.yaml"


def _config(provider: str, env: dict[str, str] | None = None) -> AppConfig:
    base = load_config(REPO_CONFIG, env=env or {})
    llm = base.llm.model_copy(update={"provider": LLMProvider(provider), "model": "test-model"})
    return base.model_copy(update={"llm": llm})


def _packages_present(monkeypatch: pytest.MonkeyPatch, present: set[str]) -> None:
    def find_spec(name: str) -> object | None:
        return object() if name in present else None

    monkeypatch.setattr(factory.importlib.util, "find_spec", find_spec)


# --- factory -----------------------------------------------------------------------------


def test_factory_fake() -> None:
    client = create_llm_client(_config("fake"))
    assert isinstance(client, FakeLLMClient)
    assert client.model_name == "test-model"


def test_factory_gemini_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    _packages_present(monkeypatch, set())
    cfg = _config("gemini", env={"GOOGLE_API_KEY": "k"})
    with pytest.raises(ConfigError, match="poetry install --with gemini"):
        create_llm_client(cfg)


def test_factory_gemini_builds_client_without_importing_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _packages_present(monkeypatch, {"langchain_google_genai"})
    client = create_llm_client(_config("gemini", env={"GOOGLE_API_KEY": "k"}))
    assert isinstance(client, GeminiClient)
    assert isinstance(client, LLMClient)
    assert (client.provider_name, client.model_name) == ("gemini", "test-model")


def test_factory_gemini_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _packages_present(monkeypatch, {"langchain_google_genai"})
    with pytest.raises(ConfigError, match="GOOGLE_API_KEY"):
        create_llm_client(_config("gemini"))


def test_factory_safechain_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    _packages_present(monkeypatch, set())
    with pytest.raises(ConfigError, match=r"safechain==1\.5\.0"):
        create_llm_client(_config("safechain"))


def _safechain_config(tmp_path: Path, *, create_file: bool = True) -> AppConfig:
    path = tmp_path / "safechain.yaml"
    if create_file:
        path.write_text("models: {}\n", encoding="utf-8")
    cfg = _config("fake")
    llm = cfg.llm.model_copy(
        update={"provider": LLMProvider.SAFECHAIN, "model": "1", "safechain_config_path": path}
    )
    return cfg.model_copy(update={"llm": llm})


def test_factory_safechain_builds_client_without_importing_sdk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _packages_present(monkeypatch, {"safechain"})
    client = create_llm_client(_safechain_config(tmp_path))
    assert isinstance(client, SafeChainClient)
    assert isinstance(client, LLMClient)
    assert (client.provider_name, client.model_name) == ("safechain", "1")


def test_factory_safechain_missing_config_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _packages_present(monkeypatch, {"safechain"})
    with pytest.raises(ConfigError, match="SAFECHAIN_CONFIG_PATH"):
        create_llm_client(_safechain_config(tmp_path, create_file=False))


def test_factory_unknown_provider() -> None:
    cfg = _config("fake")
    bogus_llm = cfg.llm.model_construct(**{**cfg.llm.model_dump(), "provider": "openai"})
    with pytest.raises(ConfigError, match=r"Unknown llm.provider"):
        create_llm_client(cfg.model_copy(update={"llm": bogus_llm}))


# --- safechain adapter (offline, fake SDK) -----------------------------------------------


class _FakeSafeChain:
    """Stands in for safechain.lcel: amodel(id) -> chat runnable, LCELRetry(model) -> model."""

    def __init__(self) -> None:
        self.reply: object = ""
        self.model_ids: list[str] = []
        self.prompts: list[list[Any]] = []
        self.config_path_seen: str | None = None

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        def chat(prompt_value: Any) -> AIMessage:
            self.prompts.append(prompt_value.to_messages())
            if isinstance(self.reply, Exception):
                raise self.reply
            return AIMessage(content=str(self.reply))

        async def amodel(model_id: str) -> Any:
            self.config_path_seen = os.environ.get("CONFIG_PATH")
            self.model_ids.append(model_id)
            return RunnableLambda(chat)

        def lcel_retry(model: Any, log_func: Any = None) -> Any:
            return model

        modules = {
            "safechain": types.ModuleType("safechain"),
            "safechain.lcel": types.ModuleType("safechain.lcel"),
            "safechain.lcel.model": types.ModuleType("safechain.lcel.model"),
            "safechain.lcel.retry": types.ModuleType("safechain.lcel.retry"),
        }
        modules["safechain.lcel.model"].amodel = amodel  # type: ignore[attr-defined]
        modules["safechain.lcel.retry"].LCELRetry = lcel_retry  # type: ignore[attr-defined]
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def fake_safechain(monkeypatch: pytest.MonkeyPatch) -> _FakeSafeChain:
    monkeypatch.delenv("CONFIG_PATH", raising=False)
    sdk = _FakeSafeChain()
    sdk.install(monkeypatch)
    return sdk


@pytest.fixture
def safechain_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "safechain.yaml"
    path.write_text("models: {}\n", encoding="utf-8")
    return path


async def test_safechain_json_call(fake_safechain: _FakeSafeChain, safechain_yaml: Path) -> None:
    fake_safechain.reply = '```json\n{"status": "connected"}\n```'
    client = SafeChainClient(model="3", config_path=safechain_yaml, timeout_seconds=5)
    # Braces in prompts must reach the model verbatim, not be treated as template variables.
    assert await client.generate_json("sys {x}", 'user {"a": 1}') == {"status": "connected"}
    assert fake_safechain.model_ids == ["3"]
    assert fake_safechain.config_path_seen == str(safechain_yaml.resolve())
    system, human = fake_safechain.prompts[0]
    assert (system.type, system.content) == ("system", "sys {x}")
    assert (human.type, human.content) == ("human", 'user {"a": 1}')


async def test_safechain_sdk_errors_are_wrapped(
    fake_safechain: _FakeSafeChain, safechain_yaml: Path
) -> None:
    fake_safechain.reply = RuntimeError("401 from idaas")
    client = SafeChainClient(model="1", config_path=safechain_yaml, timeout_seconds=5)
    with pytest.raises(LLMProviderError, match=r"model id 1\): RuntimeError: 401 from idaas"):
        await client.generate_text("s", "u")


async def test_safechain_missing_config_file(
    fake_safechain: _FakeSafeChain, tmp_path: Path
) -> None:
    client = SafeChainClient(model="1", config_path=tmp_path / "nope.yaml", timeout_seconds=5)
    with pytest.raises(LLMProviderError, match="config file not found"):
        await client.generate_text("s", "u")


async def test_safechain_without_package_raises_provider_error(
    monkeypatch: pytest.MonkeyPatch, safechain_yaml: Path
) -> None:
    monkeypatch.setitem(sys.modules, "safechain", None)
    monkeypatch.delenv("CONFIG_PATH", raising=False)
    client = SafeChainClient(model="1", config_path=safechain_yaml, timeout_seconds=5)
    with pytest.raises(LLMProviderError, match=r"safechain==1\.5\.0"):
        await client.generate_text("s", "u")


# --- gemini adapter (offline) ------------------------------------------------------------


def test_gemini_adapter_imports_without_gemini_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "langchain_google_genai", None)
    module = importlib.reload(importlib.import_module("synthetic_user.llm.gemini_adapter"))
    client = module.GeminiClient(model="m", api_key=SecretStr("k"), timeout_seconds=1)
    assert client.provider_name == "gemini"


async def test_gemini_call_without_package_raises_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "langchain_google_genai", None)
    client = GeminiClient(model="m", api_key=SecretStr("k"), timeout_seconds=1)
    with pytest.raises(LLMProviderError, match="poetry install --with gemini"):
        await client.generate_text("s", "u")


class _FakeChat:
    instances: list[_FakeChat] = []  # noqa: RUF012 - test double

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.reply: object = ""
        _FakeChat.instances.append(self)

    async def ainvoke(self, messages: list[Any]) -> Any:
        if isinstance(self.reply, Exception):
            raise self.reply
        return types.SimpleNamespace(content=self.reply)


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> type[_FakeChat]:
    _FakeChat.instances = []
    module = types.ModuleType("langchain_google_genai")
    module.ChatGoogleGenerativeAI = _FakeChat  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_google_genai", module)
    return _FakeChat


async def test_gemini_passes_config_and_secret(fake_sdk: type[_FakeChat]) -> None:
    client = GeminiClient(
        model="gem-x", api_key=SecretStr("sk-1"), timeout_seconds=12, temperature=0.2
    )
    client._get_chat().reply = '```json\n{"ok": true}\n```'
    assert await client.generate_json("s", "u") == {"ok": True}
    kwargs = fake_sdk.instances[0].kwargs
    assert kwargs["model"] == "gem-x"
    assert kwargs["timeout"] == 12
    assert kwargs["temperature"] == 0.2
    assert isinstance(kwargs["google_api_key"], SecretStr)


async def test_gemini_omits_temperature_when_unset(fake_sdk: type[_FakeChat]) -> None:
    GeminiClient(model="m", api_key=SecretStr("k"), timeout_seconds=1)._get_chat()
    assert "temperature" not in fake_sdk.instances[0].kwargs


async def test_gemini_list_content_is_joined(fake_sdk: type[_FakeChat]) -> None:
    client = GeminiClient(model="m", api_key=SecretStr("k"), timeout_seconds=1)
    client._get_chat().reply = [{"type": "text", "text": "hel"}, "lo"]
    assert await client.generate_text("s", "u") == "hello"


async def test_gemini_sdk_errors_are_wrapped(fake_sdk: type[_FakeChat]) -> None:
    client = GeminiClient(model="m", api_key=SecretStr("k"), timeout_seconds=1)
    client._get_chat().reply = TimeoutError("slow")
    with pytest.raises(LLMProviderError, match="TimeoutError"):
        await client.generate_text("s", "u")


# --- live --------------------------------------------------------------------------------


@pytest.mark.live
async def test_live_gemini_generate_json() -> None:
    from dotenv import load_dotenv

    load_dotenv(override=False)
    env = dict(os.environ)
    if not (env.get("GOOGLE_API_KEY") or env.get("GEMINI_API_KEY")) or not env.get("GEMINI_MODEL"):
        pytest.skip("GOOGLE_API_KEY/GEMINI_API_KEY and GEMINI_MODEL must be set for the live test")
    cfg = load_config(REPO_CONFIG, env=env)
    llm = cfg.llm.model_copy(update={"provider": LLMProvider.GEMINI, "model": env["GEMINI_MODEL"]})
    client = create_llm_client(cfg.model_copy(update={"llm": llm}))
    result = await client.generate_json(
        "Reply with only a JSON object.", 'Return {"status": "ok"} exactly.'
    )
    assert result.get("status") == "ok"


@pytest.mark.live
async def test_live_safechain_generate_json() -> None:
    from dotenv import load_dotenv

    load_dotenv(override=False)
    env = dict(os.environ)
    if importlib.util.find_spec("safechain") is None or not env.get("SAFECHAIN_CONFIG_PATH"):
        pytest.skip("safechain package and SAFECHAIN_CONFIG_PATH are needed for the live test")
    cfg = load_config(REPO_CONFIG, env={**env, "LLM_PROVIDER": "safechain"})
    client = create_llm_client(cfg)
    result = await client.generate_json(
        "Respond with only a JSON object containing the key 'status'.",
        "Set status to 'connected'.",
    )
    assert result.get("status") == "connected"
