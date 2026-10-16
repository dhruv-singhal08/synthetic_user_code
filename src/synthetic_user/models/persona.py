"""Persona: who the synthetic user is and how they talk."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from synthetic_user.models.base import NonEmptyStr, StrictModel


class ExpertiseLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PersonaIdentity(StrictModel):
    name: NonEmptyStr
    role: NonEmptyStr
    description: str | None = None


class PersonaExpertise(StrictModel):
    domain: ExpertiseLevel
    sql: ExpertiseLevel
    notes: str | None = None


class PersonaBehavior(StrictModel):
    style: NonEmptyStr
    traits: list[NonEmptyStr] = Field(default_factory=list)


class Persona(StrictModel):
    id: NonEmptyStr
    version: NonEmptyStr
    identity: PersonaIdentity
    expertise: PersonaExpertise
    behavior: PersonaBehavior
    typical_tasks: list[NonEmptyStr] = Field(min_length=1)
    primary_motivation: NonEmptyStr
    example_utterances: list[NonEmptyStr] = Field(default_factory=list)
