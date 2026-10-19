# CLAUDE.md

## PROJECT

synthetic-user. Synthetic User for automated testing of conversational products
(NL2SQL, chatbot, search, summariser, ...).

- Spec: instruction images + this file (no SPEC.md).
- Log: `docs/PROGRESS.md`.
- Decisions: `docs/DECISIONS.md` (one line per non-obvious choice).

Pipeline (single-turn, design v3: `docs/DESIGN_V3_REFERENCE_AND_EVALUATION.md`):

```
Persona + Product [+ Reference (data_catalog | document_collection) + optional instructions]
  -> [1] Scenarios (+ Reference Pack only when grounded) -> [2] Questions
  -> [3] Product Executor (ProductAdapter) -> [4] Evaluator (one generic judge, 5 metrics;
     correctness / groundedness against the cheat sheet when grounded) -> Bug Candidates
```

One orchestrator. Nodes are classes, not agents.

## HARD RULES (v2.1)

1. PRODUCT AGNOSTIC. The tool tests ANY conversational product, not only NL2SQL.
   - No NL2SQL / SQL wording in generic models, prompts or nodes.
   - Artifacts (e.g. SQL) are judged by the generic evaluator against the cheat sheet
     (Reference Pack); there is no product-specific evaluation code. The artifact
     plugin hook (`nodes/evaluators/plugins/`) is kept but DISCONNECTED.
2. REFERENCE OPTIONAL. Mode is automatic: reference given -> `grounded`, else
   `ungrounded`.
   - The reference has ONE of two fixed formats (design v4, D203), YAML or JSON:
     `data_catalog` (tables with description / rules / columns, relationships with a
     user-set `verified`, catalog rules R1..) or `document_collection` (documents: name +
     free-text summary). Anything else is refused. Optional usage instructions (.md/.txt
     upload or typed in the UI); without them a default per format is used. The tool never
     generates or rewrites a reference (`convert-reference` only maps a REFIDX index).
   - `grounded`: scenarios WITH a cheat sheet (Reference Pack): the LLM PICKS tables /
     columns / joins / rule ids, or documents, by exact name; code checks every pick exists
     (unknown picks rejected AND recorded) and copies the user's own text into the pack.
   - Question generation and evaluation see the cheat sheet, never the whole reference.
   - `ungrounded`: persona + product only -> no pack -> generic criteria only.
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
