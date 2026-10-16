"""Gemini via langchain-google-genai. The SDK is imported lazily on first call."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr

from synthetic_user.core.exceptions import LLMProviderError
from synthetic_user.llm.json_utils import extract_json_object

INSTALL_HINT = "poetry install --with gemini"


class GeminiClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: SecretStr,
        timeout_seconds: float,
        temperature: float | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._temperature = temperature
        self._chat: Any = None

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate_json(self, system: str, user: str) -> dict[str, Any]:
        return extract_json_object(await self.generate_text(system, user))

    async def generate_text(self, system: str, user: str) -> str:
        chat = self._get_chat()
        try:
            response = await chat.ainvoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
        except Exception as exc:  # SDK raises many unrelated types; normalize them all.
            raise LLMProviderError(
                f"Gemini call failed ({self._model}): {type(exc).__name__}: {exc}"
            ) from exc
        return _content_to_text(getattr(response, "content", None))

    def _get_chat(self) -> Any:
        if self._chat is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:
                raise LLMProviderError(
                    f"langchain-google-genai is not installed; run `{INSTALL_HINT}`"
                ) from exc
            kwargs: dict[str, Any] = {
                "model": self._model,
                "google_api_key": self._api_key,
                "timeout": self._timeout_seconds,
            }
            if self._temperature is not None:
                kwargs["temperature"] = self._temperature
            self._chat = ChatGoogleGenerativeAI(**kwargs)
        return self._chat


def _content_to_text(content: object) -> str:
    """Gemini returns either a string or a list of content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    raise LLMProviderError(f"Gemini returned unsupported content type: {type(content).__name__}")
