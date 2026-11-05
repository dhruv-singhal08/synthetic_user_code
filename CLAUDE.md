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
  + run metric selection (11 built-in metrics + custom metrics; no evaluation profiles)
  -> [0] frozen metric config (compiler call only for custom metrics)
  -> [1] Scenarios (one call; + Reference Pack only when grounded)
  -> [2] Questions (one call per scenario; each with an Evaluation Context: success criteria
     C1.., Reference Spec = the scenario's pack, Metric Mapping; no expected_behavior, D245)
  -> [3] Product Executor (ProductAdapter; + tool trace / retrieved context)
  -> [4] Evaluator (deterministic applicability; SQL: parse + catalog reconciliation + exact
     lookups; General Judge + Retrieval/Grounding Judge, at most one call each - the General
     Judge at most two when it requests one batch of lookups, D231; check-level decisions with
     evidence / guidance ids, coverage and limitations, D246-D249; applicable rules only and
     per-metric validation, D251; severity verdict policy)
     -> Bug Candidates
```

One orchestrator. Nodes are classes, not agents. One package per node
(`nodes/{scenario,question,executor,evaluator}/`): its code, the code that assembles its
prompts (`*prompt.py`) and the prompt text (`*prompt.yaml`). Metric definitions (rubrics)
are data in `metrics/catalog.yaml`. Layout: README "Code layout". Decisions D211-D224;
SQL evaluation with exact catalog lookups D229-D234.

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
     free-text summary). `reference_data` (the user's mapping-style catalog, D229) is mapped
     one-to-one onto `data_catalog`. Anything else is refused. Optional usage instructions (.md/.txt
     upload or typed in the UI); without them a default per format is used. The tool never
     generates or rewrites a reference (`convert-reference` only maps a REFIDX index).
   - `grounded`: scenarios WITH a cheat sheet (Reference Pack): the LLM PICKS tables /
     columns / joins / rule ids, or documents, by exact name; code checks every pick exists
     (unknown picks rejected AND recorded) and copies the user's own text into the pack.
   - Question generation and evaluation see the cheat sheet, never the whole reference.
     Exception (D230, user-approved): SQL evaluation also gets EXACT catalog lookups of the
     identifiers / rules the SQL actually uses (`references/catalog_index.py`); the cheat
     sheet is expected evidence, not an allow-list.
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
- Prompt TEXT only in YAML prompt files: `nodes/<node>/*prompt.yaml`,
  `adapters/mock_llm_prompt.yaml`, shared pieces in `prompts/*.yaml`. Python only picks
  entries and fills `${name}` placeholders (`prompts/templates.py`); no prompt strings in code.
  A refactor must keep every rendered prompt byte-identical.
- Tunables and switches (limits, counts, timeouts, `product.ssl_verify`) live in
  `config/app*.yaml`; fixed limits tied to a model's schema are named constants, not literals.
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
