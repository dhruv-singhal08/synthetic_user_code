# CLAUDE.md

## PROJECT

synthetic-user. Synthetic User that tests an NL2SQL product.

- Spec: phase instruction images + this file.
- Log: `docs/PROGRESS.md`.
- Decisions: `docs/DECISIONS.md` (one line per non-obvious choice).

Pipeline (single-turn):

```
Catalog -> [1] Reference Index -> [2] Scenario + Cheat Sheet -> [3] Question
        -> [4] Product Executor -> [5] SQL Evaluator -> Bug Candidates
```

One orchestrator. Nodes are classes, not agents.

## RULES

- Python 3.11, Poetry, `src/` layout, package `synthetic_user`.
- Pydantic v2 everywhere, `extra="forbid"`, `StrEnum` enums.
- Nodes depend only on `LLMClient` and `ProductClient` Protocols.
- Provider SDKs (gemini, safechain) imported lazily, only in `llm/*_adapter.py`.
- Prompts only in `src/synthetic_user/prompts/*.py`.
- All LLM output validated via `llm/structured.py` (bounded retry, typed error).
- Never invent tables, columns, relationships, capabilities, verdicts.
  LLM can never mark a relationship "verified" unless catalog says so.
- Invalid items are rejected AND recorded with reason. Never silently fixed.
- No bare except, no `time.sleep`, no globals, no hardcoded paths/keys.
- Never execute generated SQL. Never log secrets.
- Tests run offline: `FakeLLMClient` + fixture product client.
  Real-provider tests marked `@pytest.mark.live`, skipped by default.
- No LangGraph, vector DB, queues in MVP 1. Web server: only the local FastAPI UI
  (`synthetic-user serve`), added at the user's request (D111).

## QUALITY GATE (before every report)

```
poetry run ruff check . && poetry run ruff format --check . && poetry run mypy src && poetry run pytest -q
```

## PHASE REPORT

Format: files changed / commands + results / verified / NOT verified (external) / decisions / next prerequisites.

Update `docs/PROGRESS.md` at the end of each phase.
