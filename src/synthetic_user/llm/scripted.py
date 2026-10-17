"""ScriptedLLMClient: offline LLM driven by a JSON rule script (provider "fake" + script).

Script format::

    {"rules": [{"contains": ["You design realistic"], "response": {...}}, ...]}

For every call, the first rule whose substrings ALL occur in ``system + user`` answers.
Rules are reusable (not consumed). ``response`` is a JSON object (for generate_json) or a
string. No rule -> LLMProviderError naming the call.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from synthetic_user.core.exceptions import (
    InputAssetError,
    LLMProviderError,
    format_validation_error,
)
from synthetic_user.llm.fake import FakeCall
from synthetic_user.llm.json_utils import extract_json_object
from synthetic_user.models.base import NonEmptyStr, StrictModel


class ScriptRule(StrictModel):
    contains: list[NonEmptyStr] = Field(min_length=1)
    response: dict[str, Any] | str


class LLMScript(StrictModel):
    rules: list[ScriptRule] = Field(min_length=1)


class ScriptedLLMClient:
    def __init__(self, script: LLMScript, *, model_name: str = "fake-scripted") -> None:
        self._script = script
        self._model_name = model_name
        self.calls: list[FakeCall] = []

    @classmethod
    def from_file(cls, path: Path, *, model_name: str = "fake-scripted") -> ScriptedLLMClient:
        if not path.is_file():
            raise InputAssetError(f"Fake LLM script not found: {path}")
        try:
            script = LLMScript.model_validate_json(path.read_bytes())
        except ValidationError as exc:
            raise InputAssetError(
                f"Fake LLM script {path} is invalid: {format_validation_error(exc)}"
            ) from exc
        return cls(script, model_name=model_name)

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def generate_json(self, system: str, user: str) -> dict[str, Any]:
        response = self._respond("generate_json", system, user)
        if isinstance(response, str):
            return extract_json_object(response)
        return copy.deepcopy(response)

    async def generate_text(self, system: str, user: str) -> str:
        response = self._respond("generate_text", system, user)
        return response if isinstance(response, str) else json.dumps(response)

    def _respond(self, method: str, system: str, user: str) -> dict[str, Any] | str:
        self.calls.append(FakeCall(method, system, user))
        prompt = f"{system}\n{user}"
        for rule in self._script.rules:
            if all(part in prompt for part in rule.contains):
                return rule.response
        raise LLMProviderError(
            f"Fake LLM script has no rule for call #{len(self.calls)} ({method}); "
            f"system prompt starts: {system[:80]!r}"
        )
