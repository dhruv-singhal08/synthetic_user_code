"""Typed exception hierarchy. All synthetic_user errors derive from SyntheticUserError."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import ValidationError


class SyntheticUserError(Exception):
    """Base class for all synthetic_user errors."""


class ConfigError(SyntheticUserError):
    """Configuration is missing, unreadable, or invalid."""


class InputAssetError(SyntheticUserError):
    """An input asset (catalog, fixture, reference file) is missing or malformed."""


class LLMProviderError(SyntheticUserError):
    """The LLM provider call failed (transport, auth, timeout, provider-side error)."""


class LLMOutputValidationError(SyntheticUserError):
    """LLM output failed schema validation after all bounded retries."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        errors: Sequence[str],
        raw_output: str | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.errors: tuple[str, ...] = tuple(errors)
        # The text that could not be parsed (for the run trace; never shown in the message).
        self.raw_output = raw_output

    def __str__(self) -> str:
        return f"{self.args[0]} (attempts={self.attempts}, errors={len(self.errors)})"


class ReferenceIntegrityError(SyntheticUserError):
    """A reference item contradicts the reference index (unknown table, column, relationship)."""


class ArtifactNotFoundError(SyntheticUserError):
    """A required run artifact does not exist."""


class ArtifactIntegrityError(SyntheticUserError):
    """A run artifact exists but is corrupt or inconsistent."""


class ProductAdapterError(SyntheticUserError):
    """A product adapter failed to produce a response (transport, connection, bad setup)."""


class UnsupportedModeError(SyntheticUserError):
    """The requested step does not support the current generation mode yet."""


class SQLAnalysisError(SyntheticUserError):
    """Static analysis of generated SQL failed."""


def format_validation_error(exc: ValidationError) -> str:
    """One line per error as ``loc: msg``. Input values are omitted so secrets never leak."""
    return "; ".join(
        f"{'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors(include_input=False)
    )
