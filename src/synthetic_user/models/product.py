"""Description of the conversational product under test (any kind: NL2SQL, chatbot, ...).

Product description schema v2 (D205), agreed with the user. Each field is used by the
prompts: scenarios (all fields), questions (scope + data sources), evaluation (scope, known
limitations, expected outputs) and the mock product.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import Field, model_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel

# Keys of the previous schema -> their new name (refused with a clear message, never mapped).
RENAMED_KEYS = {
    "not_supported_yet": "not_supported",
    "known_gaps": "known_limitations",
    "response_artifacts_expected": "expected_outputs",
}
REMOVED_KEYS = {"product_type": "describe the product in description / expected_outputs"}


class DataSource(StrictModel):
    name: NonEmptyStr
    description: NonEmptyStr


class ProductDescription(StrictModel):
    product_id: NonEmptyStr
    product_version: NonEmptyStr
    name: NonEmptyStr
    description: NonEmptyStr
    business_problem: NonEmptyStr | None = None  # why the product exists
    data_sources: list[DataSource] = Field(default_factory=list)  # what it answers from
    supported: list[NonEmptyStr] = Field(default_factory=list)
    not_supported: list[NonEmptyStr] = Field(default_factory=list)
    known_limitations: list[NonEmptyStr] = Field(default_factory=list)
    user_visible_capabilities: list[NonEmptyStr] = Field(default_factory=list)
    # What real users ask: inspires scenarios; generated questions must not copy them.
    typical_user_questions: list[NonEmptyStr] = Field(default_factory=list)
    # What an answer consists of, e.g. "SQL query", "Executive summaries".
    expected_outputs: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _old_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            problems = [
                f"'{old}' is now '{new}'" for old, new in RENAMED_KEYS.items() if old in data
            ] + [
                f"'{old}' was removed: {hint}" for old, hint in REMOVED_KEYS.items() if old in data
            ]
            if problems:
                raise ValueError(
                    "product description uses the previous schema: " + "; ".join(problems)
                )
        return data

    @model_validator(mode="after")
    def _supported_and_unsupported_disjoint(self) -> Self:
        overlap = sorted(set(self.supported) & set(self.not_supported))
        if overlap:
            raise ValueError(f"capabilities both supported and not_supported: {overlap}")
        return self
