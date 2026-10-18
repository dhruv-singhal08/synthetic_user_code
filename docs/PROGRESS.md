# Progress

## Phase 0 - Setup
- [x] CLAUDE.md created
- [x] docs/SPEC.md, docs/PROGRESS.md, docs/DECISIONS.md created
- [x] docs/SPEC.md: no spec document; instruction cards are the spec (D057)

## P0 - Inspect + Plan (no code) - BLOCKED (partial)

### Repo inspection (2026-10-13)
- Contents: `CLAUDE.md`, `docs/SPEC.md` (placeholder), `docs/PROGRESS.md`, `docs/DECISIONS.md`.
- Not a git repository. No `pyproject.toml`, no `src/`, no tests.
- Code worth keeping: none (no code exists).
- Conflicts with SPEC: none detectable (SPEC not yet transcribed).

### Structure
- SPEC section 4 structure: NOT confirmed (SPEC missing).
- Additions recorded in DECISIONS.md (D001-D009):
  - `llm/fake.py`, `llm/structured.py`, `llm/json_utils.py`
  - `references/integrity.py`, `references/snapshot.py`
  - `nodes/clarification_evaluator.py`, `core/loaders.py`, `models/run.py`
  - `data/fixtures/product_responses/sample_responses.json`

### Pinned versions
- `aiohttp==3.13.1`
- `safechain==1.5.0` (NOT 2.x)
- Others: not yet provided.

### Phase plan P1A..P9
- Not written: files / tests / done criteria per phase must come from SPEC, which is not yet transcribed.

### Blocking missing info
1. `docs/SPEC.md` content (sections 0-21), especially section 4 (structure) and the phase definitions.
2. Remaining pinned versions ("others: I will type them below this image" - not received).

### Deferred (not blockers)
- SafeChain source.
- Real product API.

## P1A - Poetry, config, core - DONE (2026-10-13)

### Files
- `pyproject.toml`, `poetry.toml`, `poetry.lock`, `.gitignore`, `.env.example`, `.vscode/settings.json`, `.vscode/launch.json`
- `config/app.yaml`, `artifacts/.gitkeep`
- `src/synthetic_user/__init__.py`, `cli.py`
- `src/synthetic_user/{core,llm,models,nodes,prompts,references}/__init__.py`
- `src/synthetic_user/core/{config,exceptions,logging,identifiers}.py`
- `tests/core/{test_config,test_identifiers,test_exceptions_and_logging}.py`

### Quality gate
- ruff check: pass; ruff format --check: pass; mypy src (strict): pass (12 files); pytest -q: 46 passed.
- CLI smoke: `synthetic-user run` loads config, logs with run id, exits 1 (pipeline not implemented); missing `--config` file exits 2.

### NOT verified
- Python 3.11 interpreter (only 3.12 installed locally).
- `gemini` dependency group not installed/tested; safechain not installed.
- Package tree vs SPEC section 4 (SPEC still not transcribed).

## P1B - Models + sample data - DONE (2026-10-13)

### Files
- `src/synthetic_user/models/{base,persona,product,reference,scenario,question,execution,evaluation,run}.py`
- `src/synthetic_user/core/loaders.py` (load_persona / load_product / load_catalog -> InputAssetError with details)
- `core/config.py` (GEMINI_MODEL override, inputs persona/product paths, effective_config), `core/exceptions.py` (format_validation_error)
- `data/personas/finance_controller.yaml`, `data/products/sample_product.yaml`, `data/references/sample_catalog.yaml`
- `.env.example` (+GEMINI_MODEL), `.env` (blank placeholders, git-ignored), `config/app.yaml` (inputs)
- `tests/core/test_loaders.py`, `tests/models/test_execution_models.py`, `tests/models/test_domain_models.py`

### Quality gate
- ruff check: pass; ruff format --check: pass; mypy src (strict, 22 files): pass; pytest -q: 104 passed.

### NOT verified
- Field-for-field match with SPEC sections 5-11 (SPEC not transcribed); provisional fields listed in D029/D040.
- Persona/product YAML content vs SPEC 5.
- Real Gemini key/model in `.env` (still blank).

## P1C - LLM abstraction - DONE (2026-10-13)

### Files
- `src/synthetic_user/llm/{base,json_utils,structured,fake,gemini_adapter,safechain_adapter,factory}.py`
- `src/synthetic_user/prompts/structured.py`
- `pyproject.toml` (mypy override for optional SDKs)
- `tests/llm/{test_json_utils,test_structured,test_fake,test_factory_and_adapters}.py`

### Quality gate
- ruff check: pass; ruff format --check: pass; mypy src (30 files): pass; pytest -q: 148 passed, 1 deselected (live). Also clean with `-W error`.
- `pytest -m live`: 1 skipped (GOOGLE_API_KEY / GEMINI_MODEL blank in .env).

### NOT verified
- Real Gemini call (langchain-google-genai not installed; key not set). Adapter tested against an in-memory fake SDK module.
- SafeChain (stub only; source deferred).
- LLMClient signature vs SPEC section 12 (SPEC not transcribed).

## P1D - Storage + CLI + smoke - DONE (2026-10-13)

### Files
- `src/synthetic_user/storage/{__init__,json_store}.py`
- `src/synthetic_user/cli.py` (rewritten), `src/synthetic_user/prompts/smoke.py`
- `core/config.py` + `llm/factory.py` (GEMINI_API_KEY alias), `.env.example`
- `README.md`
- `tests/storage/test_json_store.py`, `tests/cli/test_cli.py`; updated `tests/core/test_config.py`, `tests/llm/test_factory_and_adapters.py`

### Quality gate
- ruff check / format --check: pass; mypy src (33 files): pass; pytest -q: 184 passed, 1 deselected.
- Live: `pytest -m live` 1 passed. `smoke-llm` with provider gemini: `provider=gemini model=gemini-3.5-flash-lite status=ok` (exit 0).

### NOT verified
- SafeChain (stub). Python 3.11 interpreter (3.12 used). SPEC field-for-field (not transcribed).

## PHASE 1 - DONE (P1A-P1D); ruff + mypy + pytest all pass.

## P2 - Node 1 reference index - DONE (2026-10-13)

### Files
- `src/synthetic_user/references/{integrity,reuse}.py`, `src/synthetic_user/prompts/reference_index.py`, `src/synthetic_user/nodes/reference_index.py`
- `models/reference.py` (+ReferenceIndexDraft), `core/config.py` + `config/app.yaml` (+reference_index.max_tables_per_call), `core/identifiers.py` (+new_index_id), `cli.py` (build-index)
- `docs/SPEC.md` (cards are the spec)
- `tests/conftest.py`, `tests/fixtures/sample_reference_index.json`, `tests/references/{test_integrity,test_reuse}.py`, `tests/nodes/test_reference_index_node.py`, `tests/cli/test_cli_build_index.py`

### Quality gate
- ruff check / format --check: pass; mypy src (37 files): pass; pytest -q -W error: 221 passed, 1 deselected.
- Real Gemini `build-index` on the sample catalog: 1st attempt failed cleanly (Gemini 503/504 overload, exit 1); 2nd attempt OK - 4 tables, verified + unknown relationships preserved, 2 uncertainties, ~90 s.

## P3 - Node 2 scenario + cheat sheet - DONE (2026-10-13)

### Files
- `src/synthetic_user/nodes/scenario_generator.py`, `src/synthetic_user/prompts/scenario.py`, `src/synthetic_user/references/cheat_sheet.py`, `src/synthetic_user/storage/runs.py`
- `models/scenario.py` (ScenarioDraft, ScenarioBatchDraft, ScenarioGenerationResult, parent ids), `core/config.py` + `config/app.yaml` (max_topup_rounds), `__init__.py` (__version__), `cli.py` (generate-scenarios)
- `docs/SPEC.md`, `CLAUDE.md` (clarification card)
- `tests/conftest.py` (persona/product/scenario drafts), `tests/nodes/test_scenario_generator.py`, `tests/cli/test_cli_generate_scenarios.py`, `tests/models/test_domain_models.py`

### Quality gate
- ruff check / format --check: pass; mypy src (41 files): pass; pytest -q -W error: 245 passed, 1 deselected.
- Real Gemini: build-index + generate-scenarios --count 3 -> 3 accepted, 0 rejected (ranking/supported single-table; aggregation/supported 3-table with verified + unknown joins; comparison/unsupported multi-currency).

## P4 - Snapshot + Node 3 questions - DONE (2026-10-13)

### Files
- `src/synthetic_user/references/{snapshot,schema_leak}.py`, `src/synthetic_user/prompts/question.py`, `src/synthetic_user/nodes/question_generator.py`
- `models/question.py` (QuestionStyle, QuestionDraft, QuestionBatchDraft, RejectedQuestion, QuestionGenerationResult), `cli.py` (generate-questions)
- `tests/conftest.py` (sample_scenarios), `tests/references/test_snapshot_and_leaks.py`, `tests/nodes/test_question_generator.py`, `tests/cli/test_cli_generate_questions.py`

### Quality gate
- ruff check / format --check: pass; mypy src (45 files): pass; pytest -q -W error: 273 passed, 1 deselected.
- Real Gemini end-to-end (build-index -> generate-scenarios --count 2 -> generate-questions --count 2): 4 questions, 0 rejected, 0 leak warnings.

## P6B - Node 5 evaluator - DONE (2026-10-13)
(P6A SQL analysis skipped by the user - D085. P5 product executor card not received.)

### Files
- `src/synthetic_user/nodes/{sql_evaluator,clarification_evaluator,sql_rules}.py`
- `src/synthetic_user/prompts/{sql_evaluation,clarification,evaluation_context}.py`
- `src/synthetic_user/benchmark.py`; `cli.py` (evaluate, benchmark-evaluator)
- `models/evaluation.py` (FindingDraft, SqlEvaluationDraft, notes, reasonable necessity, asks_for_given_info)
- `tests/fixtures/evaluation/*.json` (20 cases), `tests/fixtures/benchmark/{labeled_cases.jsonl,scenarios.json}`
- `tests/nodes/test_sql_evaluator.py`, `tests/test_benchmark.py`, `tests/cli/test_cli_evaluate.py`

### Quality gate
- ruff check / format --check: pass; mypy src (52 files): pass; pytest -q -W error: 312 passed, 1 deselected.
- Live `benchmark-evaluator` (Gemini): agreement 0.87 (13/15). DIFF B09 (label inconclusive, got fail: unknown table) and B11 (label fail, got pass: missing monthly grouping).

## Notebook - pipeline walkthrough (2026-10-13)
- `notebooks/pipeline_walkthrough.ipynb`: per-node inputs, prompts, raw LLM output, code decisions, outputs; saved executed in `fake` mode (offline). Set PROVIDER = "gemini" and Run All for a real run.
- Real Gemini run reached Nodes 1-3 (index, 2 scenarios, 4 questions) but the Node 4 stand-in cell exceeded 900 s during a Gemini 503/overload period - NOT verified end-to-end on Gemini.
- Gate after adding notebook tooling: ruff/format/mypy pass; pytest 312 passed.
- Notebook re-run on Gemini (2026-10-14, RUN_20261014_102436_05b4): all 28 cells OK end-to-end; outputs saved in the notebook (PROVIDER now "gemini").

## Sanity check (2026-10-14)
- Rules: no bare except, no sleep, no globals, no keys/URLs/model names in src, no SQL execution, secrets never unwrapped/logged, provider SDKs only in llm/*_adapter.py, all LLM calls via llm/structured.py, nodes import no adapters/storage, all models extra="forbid", all enums StrEnum.
- Hardcodes removed (D103): benchmark paths -> config; benchmark product_id/date -> configured product + real time. Only path literal left: DEFAULT_CONFIG_PATH (card 05).
- Config paths: all exist except product.fixture_path (data/fixtures/product_responses/sample_responses.json, D009) - belongs to the product-executor card (13), not received.
- Plan gaps: card 13 (Node 4 product executor, ProductClient Protocol, fixture product client) not received; card 14 skipped by user (D085); no orchestrator yet (`execute`, `run` are placeholders, exit 2).
- Gate: ruff/format/mypy pass; pytest 312 passed, 1 deselected (live).

## P7 - Pipeline + full CLI + README - DONE (2026-10-14)

### Files
- `src/synthetic_user/pipeline.py` (SyntheticUserPipeline, RunSummary, Step, validate_run)
- `src/synthetic_user/product/{__init__,base,fixture_client,factory}.py`, `src/synthetic_user/nodes/product_executor.py` (minimal card-13 stand-ins, D104)
- `src/synthetic_user/llm/scripted.py`, `llm/factory.py` (fake_script_path), `storage/json_store.py` (exists/delete), `core/config.py` (llm.fake_script_path)
- `src/synthetic_user/cli.py` rewritten: all commands via the pipeline; new `execute`, `run`; per-command --config/--log-level
- `config/app.fake.yaml`, `config/app.yaml` (fake_script_path), `tests/fixtures/fake_llm_script.json`, `data/fixtures/product_responses/sample_responses.json`
- `README.md` (full rewrite)
- `tests/integration/test_end_to_end.py`, `tests/product/test_product.py`, `tests/cli/test_cli.py` (placeholders removed)

### Quality gate
- ruff check / format --check: pass; mypy src (59 files): pass; pytest -q -W error: 333 passed, 1 deselected.
- Offline: `synthetic-user run --config config/app.fake.yaml` -> 3 scenarios, 3 questions, outcomes sql/clarification/denied = 1/1/1, verdicts pass 1 / fail 1 / recorded_only 1, 1 bug candidate. Re-run reuses all steps; --from-step / --force-step cascade as designed.
- Live (Gemini): smoke-llm OK (4.5 s). 1-scenario live run completed (exit 0): 1 scenario, 1 question, product fixture had no matching rule -> error/recorded_only. Live, quality unverified.

## Web UI (2026-10-14) - user request
- Files: `src/synthetic_user/web/{__init__,server,workspace,results_csv}.py`, `web/static/{index.html,app.js,style.css}`, `product/{contract,mock_client}.py`, `product/base.py` (UnavailableProductClient), `prompts/mock_product.py`, `core/loaders.py` (parse_asset), `storage/json_store.py` (write_text), `cli.py` (`serve`), fake LLM script (mock-product rules), README, CLAUDE.md, tests `tests/web/*`.
- Gate: ruff/format/mypy (66 files) pass; pytest 356 passed, 1 deselected.
- Verified: offline server (`serve --config config/app.fake.yaml`) driven through the HTTP API (create run with persona query, mock contract, all 5 steps, CSV download) and rendered in headless Edge (screenshot reviewed; fixed a grid-overflow layout bug and a JS newline bug found that way).
- NOT verified: UI with real Gemini; real product connection (mock only); manual click-through by a person.

## SafeChain adapter (2026-10-16) - user request
- Files: `llm/safechain_adapter.py` (stub -> real), `llm/factory.py`, `core/config.py` (LLM_PROVIDER, SAFECHAIN_MODEL_ID, SAFECHAIN_CONFIG_PATH, llm.safechain_config_path), `config/app.safechain.yaml` (new), `config/app.yaml`, `.env.example`, README, tests `tests/llm/test_factory_and_adapters.py`, `tests/core/test_config.py`. D117-D118.
- Gate: ruff/format/mypy (66 files) pass; pytest -W error 365 passed, 2 deselected (live gemini + live safechain).
- Verified: offline against a fake safechain SDK (prompt braces passed verbatim, CONFIG_PATH handed to SDK, errors wrapped); CLI with LLM_PROVIDER=safechain gives clear config/package errors; offline `run` unchanged.
- NOT verified: real SafeChain call (Amex network only); langchain-core version compatibility of safechain 1.5.0 with the lock (1.6.10).

## Dependency sync with Amex toml (2026-10-16) - user request
- Files: `pyproject.toml` (Amex pins + Artifactory source, D119), `poetry.lock` (removed; run `poetry lock` on the Amex laptop), `tests/conftest.py` (pytest-asyncio 0.24 loop cleanup), `llm/safechain_adapter.py` (install hint), README (Install, SafeChain setup), local `.venv` re-pinned (langchain-core 0.3.81, langchain-google-genai 2.1.12, ruff 0.9.10, mypy 1.20.2, pytest 8.4.2, pytest-asyncio 0.24.0, aiohttp 3.13.1).
- Gate (local venv, Amex pins): ruff/format pass; mypy (66 files) pass; pytest -W error 365 passed, 2 deselected; `pip check` clean. Same in a fresh scratch venv locked from PyPI (minus safechain).
- Live: Gemini smoke-llm OK on the downgraded stack. Offline `run` OK.
- NOT verified: `poetry lock` against Artifactory (Amex only); whether Artifactory carries fastapi 0.143 / uvicorn 0.54 / python-multipart 0.0.32 (may 403 if quarantined); real SafeChain call.

## Amex runbook (2026-10-16) - user request
- `docs/RUN_ON_AMEX.md`: copy-paste PowerShell for Python check, `poetry lock/install`, .env + SafeChain YAML, verify (pytest, smoke-llm, offline run), start the UI (`serve --config config/app.safechain.yaml`), what to send back on failure. Linked from README.
- Fix: CLI/integration tests now also clear LLM_PROVIDER / SAFECHAIN_MODEL_ID / SAFECHAIN_CONFIG_PATH (10 tests failed when those were set in the shell).
- Verified with a fake safechain package (mimics CONFIG_PATH YAML, model ids, CIBIS env): smoke-llm OK; CLI `run` incl. index exit 0; web API all 5 steps 200; `serve --config config/app.safechain.yaml` serves the UI (HTTP 200). Gate: ruff/format/mypy pass; pytest -W error 365 passed with and without SafeChain env vars.

## Prebuilt reference index (2026-10-16) - user request
- Files: `references/prebuilt_index.py` (new), `models/reference.py` (business_rules, calculated_fields, sensitive_columns, pii, global_rules), `core/loaders.py` (load_reference / parse_reference), `references/reuse.py` + `pipeline.py` + `cli.py` + `web/server.py` + `web/workspace.py` (prebuilt index replaces Node 1), `references/integrity.py` (check_time_types), `prompts/scenario.py` (catalog rules block), `prompts/evaluation_context.py`, `web/static/index.html` (.json upload), `data/references/reference_index_sample_v2.json`, `config/app.safechain.yaml` (catalog_path -> sample JSON), README, RUN_ON_AMEX, tests `tests/references/test_prebuilt_index.py` (19). D120-D121.
- Gate: ruff/format/mypy pass; pytest -W error passes (see below).
- Live (Gemini, scratch config with the sample JSON): `run` exit 0 - index from file (no LLM), 2 scenarios using business rules (USD default, gl_rt conversion), 2 questions, 0 rejected; execute = error/recorded_only because the CLI fixture product has no rules for these questions (expected).
- NOT verified: the full 11-table Amex file (may contain fields not in the sample -> strict validation will name them); SafeChain on this data.

## C02 Refactor models, product-agnostic (2026-10-17) - v2.1 cards
- Files: `models/{product,run,scenario,question,execution,evaluation,reference}.py`; cascade in `nodes/{scenario_generator,question_generator,product_executor,sql_evaluator}.py`, `product/{base,fixture_client,mock_client,contract}.py`, `prompts/{scenario,question,evaluation_context}.py`, `references/snapshot.py` (snapshots_for), `pipeline.py`, `cli.py`, `benchmark.py`, `web/{results_csv,workspace}.py`, `web/static/app.js`; data `data/products/{sample_product,sample_chatbot}.yaml`, `data/fixtures/product_responses/sample_responses.json`; fixtures `tests/fixtures/{fake_llm_script.json,benchmark/scenarios.json,evaluation/*.json}`; tests across models/nodes/product/storage/web/cli/integration. D126-D134.
- Gate: ruff check pass; ruff format --check pass (97 files); mypy (67 files) pass; pytest 417 passed, 2 deselected.
- Verified: offline `run --config config/app.fake.yaml` exit 0 (answered=1 clarification=1 denied=1; pass=1 fail=1 recorded_only=1; 1 bug); artifacts carry mode=grounded, success_criteria, adapter=fixture, answered + sql artifact, grounded_checks + plugins_applied ["sql"].
- NOT verified: live Gemini/SafeChain with the new success_criteria field; ungrounded runs (models accept them, nodes refuse until later cards); notebook; old run folders (do not load, D134).

## C03 Product adapter layer (2026-10-17) - v2.1 cards
- Files: `src/synthetic_user/adapters/{__init__,base,registry,fixture,http_json,mock_llm,contract}.py` (new package; `product/` removed: base/fixture/mock/contract/factory moved in), `core/config.py` (product_adapter section, adapter-file loading, manifest header redaction), `core/exceptions.py` (ProductAdapterError), `nodes/product_executor.py`, `pipeline.py` (aclose), `cli.py`, `web/{server,workspace}.py`, `web/static/app.js`, `config/app*.yaml`, `config/products/{sample_nl2sql,sample_policy_chatbot}.adapter.yaml`, `examples/adapters/my_product_adapter.py`, README, CLAUDE.md; tests `tests/adapters/{test_fixture_and_executor,test_registry_and_http_json}.py` (tests/product removed), `tests/core/test_config.py`, integration/web tests. D135-D141.
- Gate: ruff check pass; ruff format --check pass (100 files); mypy src pass; pytest 456 passed, 2 deselected.
- Verified: offline `run --config config/app.fake.yaml` exit 0 (answered=1 clarification=1 denied=1); HttpJsonAdapter with the REAL urllib transport against a local http.server: env header substituted, answer + sql artifact, clarification rule, HTTP 500 -> ErrorResponse, refused connection -> ProductAdapterError.
- NOT verified: any real product endpoint; urllib through the Amex proxy/TLS setup; a third-party python adapter beyond the test module.

## C04 Reference index loader, no generation (2026-10-17) - v2.1 cards
- Files: `models/reference.py` (ReferenceIndex = real file structure + lookups; catalog models removed), `references/{loader,integrity}.py` (new loader; integrity rewritten as warnings), `references/{cheat_sheet,snapshot,schema_leak}.py`, `nodes/{sql_rules,sql_evaluator,question_generator}.py`, `adapters/mock_llm.py`, `prompts/mock_product.py`, `pipeline.py` (index = load + validate step, RunInputs.reference_index, mode), `cli.py` (validate-index, --reference-index; build-index removed), `core/{config,loaders,exceptions}.py`, `storage/json_store.py`, `benchmark.py`, `web/{server,workspace}.py`, `web/static/{app.js,index.html}`, `config/app*.yaml`, `data/references/sample_card_spend_index.json` (new; sample_catalog.yaml removed), README, RUN_ON_AMEX; deleted the Node 1 node/prompt, reuse, prebuilt_index and their tests; tests `tests/references/test_loader_and_integrity.py` (new) + updates. D142-D149.
- Verified: `validate-index` on both samples (0 warnings, exit 0), on a dangling relationship (1 WARNING, exit 0), on an unknown field (exit 1, names `tables.0.colour`); offline `run --config config/app.fake.yaml` grounded, exit 0 (answered=1 clarification=1 denied=1).
- NOT verified: the user's real `data/references/reference_index.json` (not in the repo; strict loading will name any field the sample did not show).

## C05 Node 2 scenarios, two modes (2026-10-17) - v2.1 cards
- Files: `prompts/scenario.py` (shared core + grounded / ungrounded builders), `nodes/scenario_generator.py` (index | None, mode checks, ungrounded signature), `models/scenario.py` (generic TaskType values), `cli.py` (generate-scenarios --reference-index, prints mode), `tests/fixtures/fake_llm_script.json` (ungrounded rule), `tests/conftest.py` (sample_chatbot), `tests/nodes/test_scenario_generator.py`, `tests/cli/test_cli_generate_scenarios.py`. D150-D153.
- Gate (after C04 + C05): ruff check pass; ruff format --check pass; mypy src pass; pytest 417 passed, 2 deselected.
- Verified offline: ungrounded `generate-scenarios --product data/products/sample_chatbot.yaml` (no index) -> "mode ungrounded" logged, 2 accepted, manifest mode ungrounded / reference_index_id null; an ungrounded `run` stops at questions with the "not implemented yet" message.
- Verified LIVE (Gemini, scratch artifacts): ungrounded chatbot -> 3 accepted, business-level criteria, no table names; grounded with the Amex sample index -> 2 accepted, criteria use the index's rules (USD default, cost-per-mile calculated field), the cross-table join copied as verified.
- NOT verified: SafeChain; ungrounded questions/evaluation (later cards).

## C06 Node 3 questions (2026-10-17) - v2.1 cards
- Files: `prompts/question.py` (product-agnostic; grounded adds cheat sheet + snapshot), `nodes/question_generator.py` (index | None, mode per scenario, snapshot + leak check grounded only), `pipeline.py` (snapshots.json grounded only, optional artifact, typed error for ungrounded evaluation), `cli.py` (generate-questions prints mode + snapshot count), `tests/fixtures/fake_llm_script.json` (ungrounded question rules), tests `tests/nodes/test_question_generator.py`, `tests/integration/test_end_to_end.py`. D154-D157.
- Gate: ruff check pass; ruff format --check pass (93 files); mypy src pass (65 files); pytest 422 passed, 2 deselected.
- Verified offline: ungrounded chatbot `generate-scenarios` + `generate-questions` -> 2 questions, snapshots=0, no snapshots.json in the run folder; grounded offline `run` unchanged.
- Verified LIVE (Gemini): ungrounded chatbot -> 4 plain-language questions (no snapshot), unsupported scenario questions expect "not supported" behaviour; grounded (Amex sample index) -> 4 business-language questions, snapshots=2, 0 leak warnings.
- NOT verified: SafeChain; evaluation of ungrounded questions (C07+).

## C07 Node 4 executor (2026-10-17) - v2.1 cards
- Files: `nodes/product_executor.py` (rewritten: execute_many, boundary handling, response re-validation, latency, raw writer, aclose always), `models/execution.py` (latency_ms / raw_response on responses), `adapters/fixture.py` (answers format, fixture_template), `adapters/http_json.py` (supplies raw JSON body), `core/config.py` (load_adapter_file), `pipeline.py` (raw/<execution_id>.json), `cli.py` (execute --adapter-config, export-fixture-template), `data/fixtures/product_responses/{sample_responses,sample_chatbot_responses}.json`, `config/products/sample_policy_chatbot_fixture.adapter.yaml`; tests `tests/nodes/test_product_executor.py` (new), `tests/adapters/test_fixture_and_executor.py`, `tests/integration/test_end_to_end.py`, `tests/cli/test_cli.py`. D158-D164.
- Gate: ruff check pass; ruff format --check pass (94 files); mypy src pass (65 files); pytest 437 passed, 2 deselected.
- Verified: offline grounded run; then `execute --adapter-config` with an http_json adapter against a local http.server -> answered/clarification/denied, raw/EXE_001..003.json written, raw_response_path recorded, the env-var token appears in neither CLI output nor any written file, no headers saved. CLI flow export-fixture-template -> fill -> execute --adapter-config covered by an integration test.
- NOT verified: a real product endpoint; SafeChain (not used by this node).

## C08 + C09 Evaluator: generic (all products) + grounded plugins (2026-10-18) - v2.1 cards
- Files: `prompts/response_evaluation.py` (new), `prompts/reference_coverage.py` (new), `prompts/sql_evaluation.py` (Stage A input, equivalent-SQL wording), `nodes/evaluator.py` (new router: generic + plugins, ids, bugs), `nodes/evaluators/generic.py` (new), `nodes/evaluators/plugins/{base,registry,reference_coverage}.py` (new), `nodes/evaluators/plugins/sql/{models,analysis,plugin}.py` (new; replaces `nodes/sql_rules.py`, `nodes/sql_evaluator.py`), removed `nodes/clarification_evaluator.py`, `prompts/clarification.py`; `models/evaluation.py` (GenericEvaluationDraft, CheckJudgement, CriterionResult, ReferenceCoverageDraft, criteria_results, plugin_analysis, Finding.evidence; SQL models moved out), `core/config.py` (evaluation.plugins), `config/app*.yaml`, `pipeline.py`, `cli.py`, `benchmark.py`, `tests/fixtures/evaluation/*.json` (generic judgement first), `tests/fixtures/fake_llm_script.json` (generic rules); tests `tests/nodes/test_generic_evaluator.py`, `tests/nodes/test_plugins.py` (new) + updates. D165-D173.
- Gate: ruff check pass; ruff format --check pass (105 files); mypy src pass (74 files); pytest 458 passed, 2 deselected.
- Verified offline: `run --config config/app.fake.yaml` -> answered fail (generic pass + SQL plugin join_correctness), clarification pass (generic clarification focus), denied recorded_only with capability check.
- Verified LIVE (Gemini): grounded Amex-sample run with pasted answers -> correct SQL: SQL checks all pass; AVG-instead-of-SUM SQL: 5 SQL checks fail + goal_completeness partly/not_met; text-only fabricated EUR answer: reference_coverage + safety_and_honesty + capability fail; justified denial: capability pass, no verdict. Ungrounded chatbot run -> correct-looking answer: criteria unverifiable -> inconclusive (no ground truth); vague answer: goal fail; clarification asking for given info: clarification_handling fail.
- Observation: on the correct SQL answer the live judge failed persona_fit, which fails the whole evaluation (D168) - tuning question for the next card.
- NOT verified: SafeChain; the full 11-table Amex index.

## C10 Evaluator verdict + bugs (2026-10-18) - v2.1 cards (gate covers C08-C10)
- Files: `nodes/evaluator.py` (combine / combine_opinions, denied verdict from capability_consistency, llm_verdict, product_id on bugs), `models/evaluation.py` (BugCandidate.product_id, GenericEvaluationDraft.overall_verdict, CriterionResult.evidence), `prompts/response_evaluation.py` (overall_verdict, response_artifacts_expected rule), `nodes/evaluators/generic.py` (criterion evidence check), `benchmark.py` (v2: product_id, answered/clarification/denied, per-product evaluation, rows with mode + product), `core/config.py` (inputs.benchmark_product_paths), `cli.py` (benchmark per-group agreement), `config/app*.yaml`, `tests/fixtures/benchmark/{labeled_cases.jsonl,scenarios.json}` (16 cases, both modes, both product types), `tests/fixtures/evaluation/*.json`, `tests/fixtures/fake_llm_script.json`; tests `tests/nodes/test_generic_evaluator.py` (combination + bug rules), `tests/test_benchmark.py`, integration / web / cli / domain-model updates. D174-D179.
- Gate (C08-C10 together): ruff check pass; ruff format --check pass (105 files); mypy src pass (74 files); pytest 463 passed, 2 deselected.
- Verified LIVE: `benchmark-evaluator --config config/app.gemini.yaml` -> agreement 0.88 (14/16): grounded sample_nl2sql 9/11, ungrounded sample_policy_chatbot 5/5. Disagreements: B01 (correct SQL; generic persona_fit fails "raw SQL to a low-SQL finance controller") and B09 (label inconclusive for an unverifiable table; the SQL also sums mixed currencies, judged fail). Before the D179 prompt fix B01 also failed goal_completeness.
- Verified offline: `run --config config/app.fake.yaml` -> pass=2 (clarification, justified denial) fail=1 (wrong join) bug_candidates=1.
- NOT verified: human review of the benchmark labels; SafeChain.

## UI v2 (2026-10-18) - user request
- Files: `web/static/{index.html,app.js,style.css}`, `web/static/product_contract_template.yaml` (new), `web/server.py` (POST/DELETE /connection, /contract-template, http_json execute, step-order guard, explicit step branches), `web/workspace.py` (ProductConnection, parse_contract, connect), `tests/web/test_server.py`, README (Web UI), D180-D182.
- UI: formal navy/blue theme; inputs (persona sample/upload + queries; product upload only; reference index toggle + upload only); product connection (URL + uploaded contract, or mock product); 4-step pipeline with progress; results (KPIs, distribution, filters, search, CSV table, detail drawer, CSV/JSON downloads); the side rail shows the current run's inputs and product.
- Gate: ruff check pass; ruff format --check pass (105 files); mypy src pass (74 files); pytest 468 passed, 2 deselected.
- Verified: offline server (`serve` with the fake config) + headless Edge screenshots: fresh page, a reopened grounded run with all 4 steps done (3 rows, 1 fail + bug, CSV link), and a connected contract (endpoint, header names only). Execute through an uploaded contract tested with an injected transport (token sent in the header, absent from every API response).
- NOT verified: a real product endpoint through the UI; click-through by a person (drag and drop, file pickers); browsers other than Edge.
