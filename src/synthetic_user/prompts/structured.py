"""Prompt fragments used by llm/structured.py."""

from __future__ import annotations

RETRY_FEEDBACK = "\n\nprevious output invalid: {errors}"


def with_retry_feedback(user: str, errors: list[str]) -> str:
    return user + RETRY_FEEDBACK.format(errors="; ".join(errors))
