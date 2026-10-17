# CLAUDE.md

## PROJECT

synthetic-user. Synthetic User for automated testing of conversational products
(NL2SQL, chatbot, search, summariser, ...).

- Spec: instruction images + this file (no SPEC.md).
- Log: `docs/PROGRESS.md`.
- Decisions: `docs/DECISIONS.md` (one line per non-obvious choice).

Pipeline (single-turn):

```
Persona + Product [+ pre-built Reference Index]
  -> [1] Scenarios (+ Cheat Sheet only when grounded) -> [2] Questions
  -> [3] Product Executor (ProductAdapter) -> [4] Evaluator (generic criteria
     + grounded checks + plugins, e.g. SQL) -> Bug Candidates
```

One orchestrator. Nodes are classes, not agents.

## HARD RULES (v2.1)

1. PRODUCT AGNOSTIC. The tool tests ANY conversational product, not only NL2SQL.
   - No NL2SQL / SQL wording in generic models, prompts or nodes.
   - SQL is an optional PLUGIN, used only when a product response contains a
     SQL artifact. All SQL code (parsing, rules, SQL evaluation prompt, SQL
     models) lives in the plugin, never in the generic path.
2. REFERENCE INDEX OPTIONAL. Mode is automatic: index given -> `grounded`,
   else `ungrounded`.
   - `grounded`: persona + product + reference index -> scenarios WITH cheat
     sheet -> questions -> evaluation uses cheat sheet + reference snapshot +
     plugins.
   - `ungrounded`: persona + product only -> scenarios WITHOUT cheat sheet ->
     questions -> evaluation only on generic criteria (relevance, goal
     completeness, etc.). No cheat sheet, no SQL correctness verdict.
   - The index is a PRE-BUILT file supplied by the user; the tool never
     generates one (no LLM indexing).
   - The mode is recorded in every scenario, question, evaluation and the run
     manifest.
3. CLEAN PRODUCT ADAPTER LAYER. The executor talks only to a `ProductAdapter`
   Protocol. Each product = one adapter, selected by config. Adding a new
   product must NOT touch nodes, prompts, models or pipeline.

## RULES

- Python 3.11, Poetry, `src/` layout, package `synthetic_user`.
- Pydantic v2 everywhere, `extra="forbid"`, `StrEnum` enums.
- Nodes depend only on `LLMClient` and `ProductAdapter` (`adapters/base.py`) Protocols.
- Provider SDKs (gemini, safechain) imported lazily, only in `llm/*_adapter.py`.
- Prompts only in `src/synthetic_user/prompts/*.py`.
- All LLM output validated via `llm/structured.py` (bounded retry, typed error).
- Never invent tables, columns, relationships, capabilities, verdicts.
  LLM can never mark a relationship "verified" unless the reference says so.
- Invalid items are rejected AND recorded with reason. Never silently fixed.
- No bare except, no `time.sleep`, no globals, no hardcoded paths/keys.
- Never execute generated SQL. Never log secrets.
- Tests run offline: `FakeLLMClient` + fixture product adapter.
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
