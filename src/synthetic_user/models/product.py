"""Description of the NL2SQL product under test."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel


class ProductDescription(StrictModel):
    product_id: NonEmptyStr
    product_version: NonEmptyStr
    spec_version: NonEmptyStr
    name: NonEmptyStr
    description: NonEmptyStr
    supported: list[NonEmptyStr] = Field(default_factory=list)
    not_supported_yet: list[NonEmptyStr] = Field(default_factory=list)
    known_gaps: list[NonEmptyStr] = Field(default_factory=list)
    user_visible_capabilities: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _supported_and_unsupported_disjoint(self) -> Self:
        overlap = sorted(set(self.supported) & set(self.not_supported_yet))
        if overlap:
            raise ValueError(f"capabilities both supported and not_supported_yet: {overlap}")
        return self
