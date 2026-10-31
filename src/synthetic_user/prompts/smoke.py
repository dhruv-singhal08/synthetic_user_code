"""Prompts for the `smoke-llm` CLI command (text in ``common.yaml``)."""

from __future__ import annotations

from synthetic_user.prompts.templates import prompts_beside

PROMPTS = prompts_beside(__file__, "common.yaml")
SMOKE_SYSTEM = PROMPTS.text("smoke_system")
SMOKE_USER = PROMPTS.text("smoke_user")
