"""Run manifest: what a run used and which artifacts it produced."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Self

from pydantic import Field, model_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel

_SECRET_KEY_MARKERS = ("secret", "api_key", "password")


class RunManifest(StrictModel):
    run_id: NonEmptyStr = Field(pattern=r"^RUN_\d{8}_\d{6}_[0-9a-f]{4}$")
    created_at: datetime
    persona_id: NonEmptyStr
    product_id: NonEmptyStr
    catalog_id: NonEmptyStr
    catalog_version: NonEmptyStr
    reference_index_id: NonEmptyStr
    llm_provider: NonEmptyStr
    llm_model: NonEmptyStr
    pipeline_version: NonEmptyStr
    effective_config: dict[str, Any]
    artifacts: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _no_secrets_in_config(self) -> Self:
        leaked = _secret_keys(self.effective_config)
        if leaked:
            raise ValueError(f"effective_config must not contain secrets: {leaked}")
        return self


def _secret_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}{key}"
            if any(marker in str(key).lower() for marker in _SECRET_KEY_MARKERS):
                found.append(path)
            found.extend(_secret_keys(child, f"{path}."))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            found.extend(_secret_keys(child, f"{prefix}{i}."))
    return found
