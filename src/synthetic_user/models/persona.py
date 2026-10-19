"""Persona: who the synthetic user is, how they talk and what they ask (persona schema v2, D200).

Every field below is read by the scenario and/or question prompt (prompts/persona.py says
how), so changing a value changes what is generated. ``id`` and ``version`` are metadata only.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Self

from pydantic import ConfigDict, Field, model_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel


class ExpertiseLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Expertise areas that, at level high, let questions use internal names / identifiers.
TECHNICAL_AREAS = ("sql", "technical_knowledge")
_AREA_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class PersonaIdentity(StrictModel):
    name: NonEmptyStr
    role: NonEmptyStr
    description: NonEmptyStr | None = None


class PersonaExpertise(StrictModel):
    """``domain`` (required) plus any number of named areas (e.g. product_management,
    data_analysis, sql), each low | medium | high, and free-text ``notes``."""

    # The only model with open keys: extra keys ARE the expertise areas, each validated as
    # an ExpertiseLevel; any other value is rejected.
    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, ExpertiseLevel] = Field(init=False)

    domain: ExpertiseLevel
    notes: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _area_names(self) -> Self:
        bad = [k for k in self.model_extra or {} if not _AREA_NAME.fullmatch(k)]
        if bad:
            raise ValueError(f"expertise area names must be snake_case: {bad}")
        return self

    def levels(self) -> dict[str, ExpertiseLevel]:
        """Every expertise area with its level, ``domain`` first."""
        return {"domain": self.domain, **(self.model_extra or {})}


class PersonaBehavior(StrictModel):
    style: NonEmptyStr
    traits: list[NonEmptyStr] = Field(default_factory=list)


class ComplexityPreference(StrEnum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    MIXED = "mixed"


class AmbiguityPreference(StrEnum):
    NONE = "none"
    OCCASIONAL = "occasional"
    FREQUENT = "frequent"


class QuestionPreferences(StrictModel):
    """How this persona's questions are shaped. A field left out falls back to the default
    noted here (ambiguity: generation.question config)."""

    complexity: ComplexityPreference = ComplexityPreference.MIXED
    follow_up_questions: bool = False
    ambiguity: AmbiguityPreference | None = None


class Persona(StrictModel):
    id: NonEmptyStr
    version: NonEmptyStr
    identity: PersonaIdentity
    expertise: PersonaExpertise
    behavior: PersonaBehavior
    typical_tasks: list[NonEmptyStr] = Field(min_length=1)
    primary_motivation: NonEmptyStr
    decision_inputs: list[NonEmptyStr] = Field(default_factory=list)
    question_preferences: QuestionPreferences = Field(default_factory=QuestionPreferences)
    # Style examples for questions (+ the run's pasted example questions); never copied.
    example_utterances: list[NonEmptyStr] = Field(default_factory=list)

    @property
    def is_technical(self) -> bool:
        levels = self.expertise.levels()
        return any(levels.get(area) is ExpertiseLevel.HIGH for area in TECHNICAL_AREAS)
