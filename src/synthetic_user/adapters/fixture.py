"""FixtureAdapter: answers from a JSON file (offline, deterministic).

File format::

    {
      "product_id": "sample_nl2sql",
      "product_version": "fixture-1",
      "by_question_id": {"Q_001": {"outcome": "answered", "text": "..."}},
      "by_utterance": {"who spent the most": {"outcome": "clarification", "message": "..."}},
      "rules": [{"contains": ["top", "spenders"], "response": {"outcome": "answered", ...}}],
      "answers": [{"question_id": "Q_001", "utterance": "...", "outcome": "answered",
                   "text": "...", "artifacts": []}],
      "default": {"outcome": "denied", "message": "..."}
    }

``answers`` is the format written by ``export-fixture-template`` so testers can paste real
answers for any product: each entry is indexed by its question_id AND its utterance (so it
still matches when a later run renumbers the questions). An entry that was not filled in is
an error naming its question_id, never skipped silently.

Lookup order: question_id -> normalized utterance (case, punctuation and spacing ignored) ->
keyword rules (first rule whose words all appear) -> default. Any entry may be
``{"simulate_timeout": true}``: the call then waits on an Event that only ``aclose`` sets, so
the executor's timeout fires. Used for tests, offline runs and manual answer capture.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, ValidationError, field_validator, model_validator

from synthetic_user.adapters.base import ProductRequest
from synthetic_user.core.exceptions import InputAssetError, format_validation_error
from synthetic_user.models.base import NonEmptyStr, StrictModel, find_duplicates
from synthetic_user.models.execution import (
    ErrorResponse,
    ProductOutcome,
    ProductResponse,
    ResponseArtifact,
)

_NON_WORD = re.compile(r"[^0-9a-z]+")


def normalize_utterance(text: str) -> str:
    return _NON_WORD.sub(" ", text.casefold()).strip()


class SimulatedTimeout(StrictModel):
    simulate_timeout: Literal[True]


FixtureEntry = ProductResponse | SimulatedTimeout


class FixtureAnswer(StrictModel):
    """One pasted answer (export-fixture-template format)."""

    question_id: NonEmptyStr
    utterance: NonEmptyStr
    outcome: ProductOutcome
    text: str | None = None
    artifacts: list[ResponseArtifact] = Field(default_factory=list)
    message: str | None = None
    error_type: str | None = None
    timeout_seconds: float | None = None
    product_version: str | None = None

    def to_response(self) -> ProductResponse:
        fields = self.model_dump(exclude={"question_id", "utterance"}, exclude_none=True)
        if self.outcome is not ProductOutcome.ANSWERED:
            fields.pop("artifacts", None)
        try:
            return _ResponseBox.model_validate({"response": fields}).response
        except ValidationError as exc:
            raise ValueError(
                f"answer for {self.question_id} is not filled in correctly: "
                f"{format_validation_error(exc)}"
            ) from exc


class _ResponseBox(StrictModel):
    response: ProductResponse


class FixtureRule(StrictModel):
    contains: list[NonEmptyStr] = Field(min_length=1)
    response: FixtureEntry


class ProductFixture(StrictModel):
    product_id: NonEmptyStr
    product_version: str | None = None
    by_question_id: dict[NonEmptyStr, FixtureEntry] = Field(default_factory=dict)
    by_utterance: dict[NonEmptyStr, FixtureEntry] = Field(default_factory=dict)
    rules: list[FixtureRule] = Field(default_factory=list)
    answers: list[FixtureAnswer] = Field(default_factory=list)
    default: FixtureEntry

    @field_validator("by_utterance")
    @classmethod
    def _normalized_keys_unique(cls, value: dict[str, FixtureEntry]) -> dict[str, FixtureEntry]:
        clashes = find_duplicates([normalize_utterance(k) for k in value])
        if clashes:
            raise ValueError(f"utterances identical after normalization: {clashes}")
        return {normalize_utterance(k): v for k, v in value.items()}

    @model_validator(mode="after")
    def _index_answers(self) -> Self:
        """Fold ``answers`` into by_question_id / by_utterance; clashes are errors."""
        for answer in self.answers:
            response = answer.to_response()
            key = normalize_utterance(answer.utterance)
            if answer.question_id in self.by_question_id:
                raise ValueError(f"{answer.question_id} is answered twice")
            if key in self.by_utterance:
                raise ValueError(f"utterance of {answer.question_id} is answered twice")
            self.by_question_id[answer.question_id] = response
            self.by_utterance[key] = response
        return self


class FixtureAdapter:
    name = "fixture"

    def __init__(self, fixture: ProductFixture) -> None:
        self._fixture = fixture
        self._closed = asyncio.Event()
        self.asked: list[ProductRequest] = []

    @classmethod
    def from_file(cls, path: Path) -> Self:
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

    async def send(self, req: ProductRequest) -> ProductResponse:
        self.asked.append(req)
        entry = self._lookup(req)
        if isinstance(entry, SimulatedTimeout):
            await self._closed.wait()
            return ErrorResponse(error_type="AdapterClosed", message="simulated timeout ended")
        if entry.product_version is None and self._fixture.product_version:
            return entry.model_copy(update={"product_version": self._fixture.product_version})
        return entry

    async def aclose(self) -> None:
        self._closed.set()

    def _lookup(self, req: ProductRequest) -> FixtureEntry:
        fixture = self._fixture
        if req.question_id in fixture.by_question_id:
            return fixture.by_question_id[req.question_id]
        normalized = normalize_utterance(req.utterance)
        if normalized in fixture.by_utterance:
            return fixture.by_utterance[normalized]
        text = req.utterance.casefold()
        for rule in fixture.rules:
            if all(word.casefold() in text for word in rule.contains):
                return rule.response
        return fixture.default


def fixture_template(
    product_id: str, questions: list[tuple[str, str]], product_version: str | None = None
) -> dict[str, Any]:
    """An ``answers`` fixture to fill in by hand: one entry per (question_id, utterance).

    Every entry starts as an empty "answered" (text null, no artifacts), which does not load
    until a real answer is pasted - so an unfilled entry can never pass as a product answer.
    """
    return {
        "product_id": product_id,
        "product_version": product_version,
        "answers": [
            {
                "question_id": question_id,
                "utterance": utterance,
                "outcome": ProductOutcome.ANSWERED.value,
                "text": None,
                "artifacts": [],
            }
            for question_id, utterance in questions
        ],
        "default": {
            "outcome": ProductOutcome.ERROR.value,
            "error_type": "NoFixtureMatch",
            "message": "No answer pasted for this question.",
        },
    }
