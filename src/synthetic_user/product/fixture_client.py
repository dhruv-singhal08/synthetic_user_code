"""Fixture product client: answers from a JSON file of keyword rules (offline, deterministic).

File format::

    {
      "product_id": "sample_nl2sql",
      "product_version": "fixture-1",
      "rules": [{"contains": ["top", "spenders"], "response": {"outcome": "sql_generated", ...}}],
      "default": {"outcome": "denied", "message": "..."}
    }

The first rule whose words all appear in the question (case-insensitive) wins; otherwise
``default``. Used for tests, offline runs and manual SQL capture.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, ValidationError

from synthetic_user.core.exceptions import InputAssetError, format_validation_error
from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.product.base import ProductResponse


class FixtureRule(StrictModel):
    contains: list[NonEmptyStr] = Field(min_length=1)
    response: ProductResponse


class ProductFixture(StrictModel):
    product_id: NonEmptyStr
    product_version: str | None = None
    rules: list[FixtureRule] = Field(default_factory=list)
    default: ProductResponse


class FixtureProductClient:
    def __init__(self, fixture: ProductFixture) -> None:
        self._fixture = fixture
        self.asked: list[str] = []

    @classmethod
    def from_file(cls, path: Path) -> FixtureProductClient:
        if not path.is_file():
            raise InputAssetError(f"Product fixture not found: {path}")
        try:
            return cls(ProductFixture.model_validate_json(path.read_bytes()))
        except ValidationError as exc:
            raise InputAssetError(
                f"Product fixture {path} is invalid: {format_validation_error(exc)}"
            ) from exc

    @property
    def product_id(self) -> str:
        return self._fixture.product_id

    async def ask(self, utterance: str) -> ProductResponse:
        self.asked.append(utterance)
        text = utterance.casefold()
        response = next(
            (
                rule.response
                for rule in self._fixture.rules
                if all(word.casefold() in text for word in rule.contains)
            ),
            self._fixture.default,
        )
        if response.product_version is None and self._fixture.product_version:
            response = response.model_copy(
                update={"product_version": self._fixture.product_version}
            )
        return response
