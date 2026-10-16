"""Prompts for the `smoke-llm` CLI command."""

from __future__ import annotations

SMOKE_SYSTEM = "You are a connectivity check. Reply with a single JSON object and nothing else."
SMOKE_USER = 'Return exactly this JSON object: {"status": "ok"}'
