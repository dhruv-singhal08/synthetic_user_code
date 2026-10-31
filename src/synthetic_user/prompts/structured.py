"""Prompt fragments used by llm/structured.py (text in ``common.yaml``)."""

from __future__ import annotations

from synthetic_user.prompts.templates import prompts_beside

PROMPTS = prompts_beside(__file__, "common.yaml")


def with_retry_feedback(user: str, errors: list[str]) -> str:
    return user + PROMPTS.text("retry_feedback", errors="; ".join(errors))
