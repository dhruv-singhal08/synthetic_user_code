"""LLMClient Protocol. Nodes depend only on this, never on a provider SDK."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    async def generate_json(self, system: str, user: str) -> dict[str, Any]: ...

    async def generate_text(self, system: str, user: str) -> str: ...
