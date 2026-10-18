# synthetic-user

A **synthetic user** for automated testing of conversational products (NL2SQL, chatbot,
search, summariser, ...). It invents realistic scenarios for a persona, asks the product
questions in that persona's voice, and judges what the product returns, producing **bug
candidates** for a human to review. With your **pre-built reference index** (a JSON file;
this tool never generates one) runs are **grounded**: scenarios carry a cheat sheet and SQL
answers are checked against the index. Without one, runs are **ungrounded** (no cheat sheet).

> **Status: MVP 1, not production-ready.** See [Limitations](#limitations).

## Pipeline

```
Persona + Product [+ pre-built Reference Index] ──> [index check] ──> [2] Scenarios
        (+ Cheat Sheet when grounded) ──> [3] Questions ──> [4] Product Executor
        ──> [5] Evaluator ──> Bug Candidates
```

| Node | What it does | LLM? |
|---|---|---|
| Index check | Loads and validates your pre-built reference index (warnings for duplicates / dangling relationships, never fixed). None given -> ungrounded. | no |
| 2 Scenarios | Business goal + 2-5 success criteria for tasks this persona would do with this product. Grounded: also a plausible analysis path ("cheat sheet") using only index tables/columns. Ungrounded: no cheat sheet, nothing invented. | yes |
| 3 Questions | Natural-language questions in the persona's voice for any product type; near-duplicate checks. Grounded: a deterministic snapshot of the index per scenario + schema-leak check. Ungrounded: no snapshot. | yes |
| 4 Product executor | Sends each question through the configured product adapter and records answered (text + artifacts, e.g. SQL) / clarification / denied / error / timeout. SQL is **never executed**. | no |
| 5 Evaluator | Generic checks for every product (relevance, goal completeness per success criterion, persona fit, capability consistency, clarification handling, safety & honesty; evidence quoted from the response). Grounded only: artifact plugins (`evaluation.plugins`, e.g. `sql`: SQLGlot static analysis + LLM review, deterministic overrides) or, if none applies, a reference-coverage check. Denied = verdict from capability consistency only (was the denial justified?); error/timeout recorded only. The judges' own opinions are stored as `llm_verdict`, never decisive. Failed evaluations with supported evidence become bug candidates. | yes |

One orchestrator (`src/synthetic_user/pipeline.py`) runs the nodes; nodes are plain classes.

## Install

Requires Python 3.11 or 3.12 and [Poetry](https://python-poetry.org/).

`pyproject.toml` matches the Amex SafeChain environment: `safechain==1.5.0` (not 2.x),
`langchain-core==0.3.81`, `aiohttp==3.13.1`, packages from Amex Artifactory (primary source).
Do not bump these without re-testing SafeChain there.

**Amex laptop** (Artifactory reachable):

```bash
poetry lock                    # first time / after pyproject changes (no lock is committed)
poetry install                 # core (incl. safechain) + dev tools
poetry install --with gemini   # + Gemini provider (langchain-google-genai 2.x, langchain-core 0.3)
```

**Outside Amex** safechain and Artifactory cannot be resolved, so `poetry lock/install` fail.
Use an existing venv with the same pins (everything except safechain is on public PyPI):
`langchain-core==0.3.81 langchain-google-genai==2.1.12 aiohttp==3.13.1 ruff==0.9.10
mypy==1.20.2 pytest==8.4.2 pytest-asyncio==0.24.0`. Gemini and the fake provider work there.

| Group | Contents | When |
|---|---|---|
| main | safechain, langchain-core, pydantic, sqlglot, typer, fastapi, ... | always |
| `gemini` (optional) | `langchain-google-genai` ^2.0 | `llm.provider: gemini` |
| `dev` (default) | pytest ^8, pytest-asyncio ^0.24, ruff ^0.9, mypy ^1.13, types-PyYAML, ipykernel, nbclient, nbformat, httpx | development, notebook |

If Poetry is not on your PATH, use `python -m poetry ...`.

## Configuration

Config file order: `--config` > `CONFIG_PATH` env var > `./config/app.yaml`. Relative paths
resolve against the current working directory, so run commands from the repo root.
Unknown keys are rejected.

### `.env`

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `GOOGLE_API_KEY` | Gemini API key. `GEMINI_API_KEY` is accepted as an alias (`GOOGLE_API_KEY` wins). Never commit `.env`. |
| `LLM_PROVIDER` | `gemini` \| `safechain` \| `fake`; overrides `llm.provider`. |
| `GEMINI_MODEL` | Gemini model name; overrides `llm.model` when the provider is gemini. |
| `SAFECHAIN_MODEL_ID` | SafeChain `models` key (e.g. `1`); overrides `llm.model` when the provider is safechain. |
| `SAFECHAIN_CONFIG_PATH` | SafeChain's own YAML; overrides `llm.safechain_config_path`. |
| `CIBIS_CONSUMER_INTEGRATION_ID`, `CIBIS_CONSUMER_SECRET` | SafeChain IDaaS credentials (read by SafeChain). |
| `CONFIG_PATH` | Config file when `--config` is not given. |
| `LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR` (`--log-level` overrides). |

Secrets are read **only** from the environment; a `secrets:` key in YAML is an error.

### Every config key (`config/app.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `llm.provider` | `fake` | `gemini` \| `safechain` \| `fake` |
| `llm.model` | `fake` | Provider model name (`GEMINI_MODEL` / `SAFECHAIN_MODEL_ID` override) |
| `llm.safechain_config_path` | `null` | SafeChain YAML; required for safechain (`SAFECHAIN_CONFIG_PATH` overrides) |
| `llm.temperature` | `null` | `null` = provider default, else 0.0–2.0 |
| `llm.timeout_seconds` | `60` | Per LLM call |
| `llm.max_validation_retries` | `2` | Retries when LLM output fails JSON/schema/integrity checks |
| `llm.max_concurrency` | `4` | Max concurrent LLM / product calls |
| `llm.fake_script_path` | `null` | `fake` only: JSON rule script for offline runs |
| `product_adapter.type` | `fixture` | `fixture` \| `http_json` \| `python` (see "Product adapters") |
| `product_adapter.options` | `{path: ./data/fixtures/product_responses/sample_responses.json}` | Per type; or set `product_adapter: <path to .adapter.yaml>` |
| `product.timeout_seconds` | `30` | Per product call; exceeded -> `timeout` outcome |
| `sql.dialect` | `bigquery` | sqlglot dialect used to parse product SQL |
| `evaluation.plugins` | `[sql]` | Artifact plugins for grounded evaluation (unknown names are rejected) |
| `generation.scenario.scenario_count` | `10` | Scenarios requested per run |
| `generation.scenario.allow_multi_table` | `true` | Allow scenarios needing joins |
| `generation.scenario.include_unsupported_scenarios` | `true` | Keep scenarios the product should refuse |
| `generation.scenario.diversity_preference` | `balanced` | Free-text hint to the scenario generator |
| `generation.scenario.max_topup_rounds` | `1` | Extra calls if too few scenarios are accepted (never padded) |
| `generation.question.question_count` | `1` | Questions per scenario |
| `generation.question.ambiguity_level` | `low` | `low` \| `medium` \| `high` mix of precise/vague/investigative |
| `generation.question.language` | `en` | Question language |
| `generation.question.allow_vague_questions` | `false` | Allow `style: vague` questions |
| `generation.question.reject_schema_leaks` | `true` | Retry/reject questions naming tables/columns (non-technical personas) |
| `paths.artifacts_dir` | `./artifacts` | Where runs are written |
| `inputs.persona_path` | `./data/personas/finance_controller.yaml` | Persona |
| `inputs.product_path` | `./data/products/sample_product.yaml` | Product description (supported / not_supported_yet / known_gaps) |
| `inputs.reference_index_file` | `./data/references/sample_card_spend_index.json` | Your pre-built reference index; omit -> ungrounded |
| `inputs.benchmark_cases_path` | `./tests/fixtures/benchmark/labeled_cases.jsonl` | `benchmark-evaluator` cases |
| `inputs.benchmark_scenarios_path` | `./tests/fixtures/benchmark/scenarios.json` | `benchmark-evaluator` scenarios |

`config/app.fake.yaml` is the same config wired for **offline** use: scripted fake LLM
(`tests/fixtures/fake_llm_script.json`), fixture product, 3 scenarios, artifacts in
`./artifacts/offline`.

## Commands

Global options `--config PATH`, `--log-level LEVEL` (also accepted after the command name). Exit code `0` = completed
(fail verdicts are results, not errors); `1` = error.

| Command | Options (defaults) | Does |
|---|---|---|
| `smoke-llm` | – | One tiny JSON request; prints provider, model, latency |
| `validate-index` | `--reference-index PATH` (required) | Load + check a pre-built index: tables, columns, relationships, warnings; exit 1 on load errors |
| `generate-scenarios` | `--run-id` (new), `--persona`/`--product` (inputs), `--reference-index` (`inputs.reference_index_file`; none -> ungrounded), `--count` (`scenario_count`) | Node 2 (grounded or ungrounded); always regenerates |
| `generate-questions` | `--run-id` (required), `--count` (`question_count`) | Node 3 (+ snapshots when grounded); always regenerates |
| `execute` | `--run-id` (required), `--adapter-config FILE` (config `product_adapter`) | Node 4 via the product adapter; always regenerates |
| `export-fixture-template` | `--run-id` (required), `--out` (`runs/<run_id>/fixture_template.json`) | Template (question_id, utterance, outcome, text, artifacts[]) to paste real answers into, for any product |
| `evaluate` | `--run-id`, **or** `--questions --executions --scenarios` (+ `--out-dir`, default: executions file's folder) | Node 5 |
| `run` | `--run-id` (new), `--reference-index`, `--from-step STEP`, `--force-step STEP` (repeatable), `--scenario-count`, `--question-count` | Full pipeline with reuse |
| `benchmark-evaluator` | `--cases`, `--scenarios` (from `inputs.benchmark_*`; products from `inputs.benchmark_product_paths`) | **Live only**: evaluator agreement with labeled cases, per mode/product and overall |
| `serve` | `--host` (`127.0.0.1`), `--port` (`8000`) | Web UI + API (see [Web UI](#web-ui)) |

Steps: `index`, `scenarios`, `questions`, `execute`, `evaluate`.

`run` reuses every existing artifact of the run. A step is regenerated when named by
`--from-step` / `--force-step`, when its files are missing, or when an earlier step was
regenerated (regeneration always cascades downstream; stale downstream files are deleted).
If the stored reference index changes, the run is regenerated from `scenarios`.
`manifest.json` is updated after every step, and cross-references
(question→scenario, execution→question, evaluation→all) are validated on every load.

```bash
synthetic-user run --config config/app.fake.yaml           # offline, ~1 s
synthetic-user run                                        # with your config
synthetic-user run --run-id RUN_... --from-step evaluate    # re-judge only
```

`run` prints scenarios accepted/rejected, questions, outcomes, verdicts, bug candidates
and the run folder.

## Web UI

```bash
synthetic-user serve                                   # http://127.0.0.1:8000
synthetic-user serve --config config/app.fake.yaml     # fully offline demo
```

A local FastAPI server (`src/synthetic_user/web/`) with a single-page UI. It binds to
`127.0.0.1` by default and has no authentication, so do not expose it on a network.
API docs are at `/api/docs`.

1. **Inputs:**
   - **Persona:** the sample persona or your own YAML upload, plus optional **sample queries**
     (one per line) that are added to the persona's example utterances to mimic the user's
     style; generated questions never copy them.
   - **Product description:** upload only (YAML), no sample.
   - **Reference index:** toggle. Off (default) -> ungrounded run. On -> upload the pre-built
     index JSON (no sample) -> grounded run with cheat sheets and grounded checks.
   Each file is validated on upload and previewed.
2. **Product connection:** the **product URL** plus a **contract** (YAML/JSON) that says how
   to call the product and read its reply: `method`, `headers` (`${ENV_VAR}` only for
   secrets), `timeout`, `request_template` (`{utterance}`, `{session_id}`) and
   `response_mapping` (dotted paths for text / artifacts, clarification / denied conditions).
   It is the `http_json` adapter's options (a `*.adapter.yaml` file works too); download a
   template from the UI (`/api/contract-template`). The URL field wins over a `url` in the
   contract. Header values are never shown or returned by the API. **Mock product** (demo
   only) instead routes questions to an LLM stand-in that answers from the reference index.
   Not connected -> the configured `product_adapter` answers.
3. **Pipeline:** generate scenarios (the index check runs first), generate questions, run on
   product, evaluate - one by one, in order, or **Run all**. Re-running a step regenerates
   everything after it.
4. **Results:** KPI tiles, verdict distribution, filters/search and one row per question - the
   same table as `runs/<run_id>/results.csv`, which is rewritten after every step. Click a row
   for every field (scenario, cheat sheet, question, response, checks, findings, bug);
   download the CSV or any step's JSON. Reload-safe: the run id is kept in the URL (`?run=`).

UI runs store their inputs (and an uploaded contract, `product_contract.yaml`) in
`runs/<run_id>/inputs/` and UI state in `ui_state.json`.

## Artifact layout

```
artifacts/
  runs/<run_id>/
    manifest.json             # ids, versions, provider/model, effective config (no secrets), artifacts
    scenarios.json            # accepted Scenario[]
    scenarios_rejected.json   # rejected + duplicates, with reasons
    snapshots.json            # per-scenario reference index snapshot (grounded runs only)
    questions.json            # QuestionCase[]
    questions_rejected.json   # with reasons
    executions.json           # ExecutionRecord[] (response: answered | clarification | denied | error | timeout)
    raw/<execution_id>.json   # raw product response body, when the adapter supplies one (never headers)
    fixture_template.json     # from export-fixture-template
    evaluations.json          # EvaluationResult[] (generic checks + rationale, criteria results, grounded checks, plugin analysis, findings, verdict)
    bug_candidates.json       # BugCandidate[] (status needs_review, full id chain)
  notebook_runs/<run_id>/     # notebooks/pipeline_walkthrough.ipynb output
```

All JSON writes are atomic. Ids: `RUN_…`, `IDX_…`, `SCN_001`, `Q_001`, `EXE_001`,
`EVAL_001`, `F_001`, `BUG_001` - assigned by code, never by the LLM.

## Manual SQL capture flow

Without a reachable product (or before writing its adapter), you supply what the product said:

1. **Fixture answers (repeatable):** edit `data/fixtures/product_responses/sample_responses.json`
   (or a copy set as `product_adapter.options.path`). Lookup order: `by_question_id`
   (`Q_001` -> response), then `by_utterance` (case/punctuation-insensitive), then `rules`
   (`contains`: all words must match, case-insensitive), then `default`. A response is
   `{"outcome": "answered", "text": "...", "artifacts": [{"kind": "sql", "content": "..."}]}`
   (text and/or artifacts), `clarification` + `message`, `denied` +
   `message`, `error` + `error_type`, `timeout`, or `{"simulate_timeout": true}` (waits until
   `product.timeout_seconds` fires).
   Then `synthetic-user execute --run-id RUN_…` and `synthetic-user evaluate --run-id RUN_…`.
   **Template for any product:** `export-fixture-template --run-id RUN_…` writes one entry per
   question (`question_id`, `utterance`, `outcome`, `text`, `artifacts`). Paste the real answers
   (unfilled entries are rejected, never treated as answers), then
   `execute --run-id RUN_… --adapter-config <file>` with a `{type: fixture, options: {path: …}}`
   adapter file. Answers match by question id, or by utterance if a later run renumbers them.
2. **Hand-written executions:** after `generate-questions`, copy the questions into the real
   product, paste each answer into `runs/<run_id>/executions.json` (one `ExecutionRecord`
   per question; see `src/synthetic_user/models/execution.py`), then
   `synthetic-user evaluate --run-id RUN_…`. Or use file mode:
   `evaluate --questions q.json --executions e.json --scenarios s.json --out-dir out/`.
3. **Notebook:** `notebooks/pipeline_walkthrough.ipynb` shows every node's inputs, prompts,
   raw LLM output and decisions, and has `SQL_OVERRIDES` to paste SQL per question.

## Product adapters

The executor talks to products only through a `ProductAdapter` (`src/synthetic_user/adapters/`).
One adapter per product, chosen by `product_adapter` in the app config (inline, or a path to
`config/products/<product_id>.adapter.yaml`). Adding a product never touches nodes, prompts,
models or the pipeline.

| type | Use | Options |
|---|---|---|
| `fixture` | offline / captured answers | `path` |
| `http_json` | any JSON-over-HTTP product, config only | `url`, `method`, `headers` (`${ENV_VAR}` filled from env), `timeout`, `request_template` (`{utterance}`, `{session_id}`), `response_mapping` (dotted paths + `path == value` rules) |
| `python` | anything else (SDK, websockets, custom auth) | `class: "pkg.module:Class"`, `init: {...}` (keyword args) |

Examples: `config/products/sample_nl2sql.adapter.yaml` (fixture),
`config/products/sample_policy_chatbot.adapter.yaml` (http_json, dummy URL). Template for a
custom adapter: `examples/adapters/my_product_adapter.py`. HTTP errors and non-JSON bodies
become `error` outcomes with the status code; header secrets are never logged and are
written to the run manifest only as their `${ENV_VAR}` template (literal values show `***`).

## Reference index (pre-built, never generated)

The reference index is a JSON file you supply (`inputs.reference_index_file` or
`--reference-index`). It is the **only** reference source: tables, columns (types,
descriptions, PII, categorical values), relationships with verification, business rules,
calculated fields, sensitive columns and global rules. Loader: `references/loader.py`.

- Field names are the real file's (sample: `data/references/reference_index_sample_v2.json`,
  4 of 11 tables transcribed from screenshots; strings ending `[TRUNCATED IN SCREENSHOT]` were
  cut off). Provenance keys the tool does not use (`source_files`, `scope`,
  `integrity_report`, `business_domain`, `semantic_roles`, ...) are kept in `extras`.
- Strict: an unknown field or a missing required one is an error naming the file and the field
  path (e.g. `tables.0.columns.2.colour`). Content problems (duplicate tables/columns,
  relationships to unknown tables/columns, scope counts, a failed `integrity_report`) are
  **warnings**: listed by `validate-index` and the index step, never fixed.
- Runs record `<index_id>-<content hash>`, so editing the file regenerates later steps.
- Offline sample (card spend): `data/references/sample_card_spend_index.json`.

## Switching providers

| Provider | Config | Needs |
|---|---|---|
| `fake` | `llm.provider: fake` (+ `llm.fake_script_path` for full offline runs) | nothing |
| `gemini` | `llm.provider: gemini`; model via `llm.model` or `GEMINI_MODEL` | `poetry install --with gemini`, `GOOGLE_API_KEY` |
| `safechain` | `llm.provider: safechain` (`config/app.safechain.yaml`); model = SafeChain `models` key via `llm.model` or `SAFECHAIN_MODEL_ID` | `safechain==1.5.0` (Amex env), SafeChain YAML, `CIBIS_*` credentials |

**Switch without editing YAML:** set `LLM_PROVIDER=gemini|safechain|fake` in `.env`. When it
changes the provider, the matching model variable (`GEMINI_MODEL` / `SAFECHAIN_MODEL_ID`) must be
set too. Check connectivity with `synthetic-user smoke-llm`.

**Amex laptop step-by-step (install, verify, start the UI):** [docs/RUN_ON_AMEX.md](docs/RUN_ON_AMEX.md).

**SafeChain setup (Amex laptop):**

1. `poetry lock && poetry install` (safechain 1.5.0 comes from Artifactory; see [Install](#install)).
2. Copy your SafeChain YAML (`idaas` + `models`, e.g. `config_e2.yaml`) to
   `config/safechain.yaml`, or point `SAFECHAIN_CONFIG_PATH` at it.
3. In `.env`: `CIBIS_CONSUMER_INTEGRATION_ID`, `CIBIS_CONSUMER_SECRET` (names from
   `idaas.id_key` / `idaas.secret_key`), and `SAFECHAIN_MODEL_ID` (e.g. `1` = gpt-5.2,
   `3` = gpt-5-mini) or `LLM_PROVIDER=safechain`.
4. **Do not set `CONFIG_PATH` to the SafeChain YAML** - `CONFIG_PATH` is this app's config.
   The adapter passes `SAFECHAIN_CONFIG_PATH` to SafeChain (as `CONFIG_PATH`) itself.
5. `poetry run synthetic-user smoke-llm --config config/app.safechain.yaml`

Each call builds `prompt | LCELRetry(amodel(id)) | StrOutputParser()`; JSON is parsed and
schema-validated by `llm/structured.py` like every provider. Temperature comes from the
SafeChain YAML `model_kwargs`, not `llm.temperature`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `llm.provider 'gemini' needs package 'langchain_google_genai'` | `poetry install --with gemini` (SafeChain: install `safechain==1.5.0` in the Amex env) |
| `Config file not found` / wrong config used | Run from the repo root, or pass `--config`; check `CONFIG_PATH` in `.env` (real env vars win over `.env`) |
| `GOOGLE_API_KEY (or GEMINI_API_KEY) must be set` | Put the key in `.env` **and save it**; make sure the variable is not set twice (the last line wins) |
| `403` when installing packages | Your package mirror quarantined a version. Use the pinned versions known to work (`aiohttp==3.13.1`, `safechain==1.5.0`) or ask for the package to be released; do not bypass the mirror |
| `Invalid config ...: llm.modle: Extra inputs are not permitted` | Typo or unknown key - every config key is listed above |
| `... output invalid after N attempt(s)` / `ReferenceIntegrityError` | The LLM kept returning invalid or ungrounded output. Retry, raise `llm.max_validation_retries`, or inspect prompts + raw output in the notebook |
| Many SQL `syntax_error` findings on valid SQL | Dialect mismatch: set `sql.dialect` to the product's SQL dialect (sqlglot name; Spanner has no sqlglot dialect - `bigquery` is closest) |
| Gemini `503 UNAVAILABLE` / `504 DEADLINE_EXCEEDED` | Provider overload; re-run (completed steps are reused) or raise `llm.timeout_seconds` |
| `Run artifacts are inconsistent` | Files in `runs/<run_id>/` were edited or mixed; re-run with `--from-step` at the first broken step |
| `do not match run` | Configured persona/product/mode/reference index differ from the run's manifest; use the original config or a new `--run-id` |

## Development

```bash
poetry run ruff check . && poetry run ruff format --check . && poetry run mypy src && poetry run pytest -q
poetry run pytest -m live        # real-provider tests (need GOOGLE_API_KEY + GEMINI_MODEL)
```

Tests run fully offline (scripted fake LLM + fixture product). Rules for contributors are in
`CLAUDE.md`; every non-obvious choice is in `docs/DECISIONS.md`; progress in `docs/PROGRESS.md`.

## Limitations

- **Not production-ready.** MVP 1, single-turn only.
- **Product connection untested against a real product:** `http_json` is verified only against a local test server; the UI's contract flow is still a mock.
- **Web UI is local-only:** no authentication; one user at a time.
- **SafeChain** adapter is tested offline against a fake SDK only; first real call is on the Amex laptop.
- **No full static SQL analysis** (Stage A skipped); only minimal deterministic SQL rules.
- **LLM judgements are not calibrated:** confidence values are uncalibrated, and the
  benchmark labels in `tests/fixtures/benchmark/` were drafted by the assistant and still
  need human review.
- **Verdicts judge SQL structure against metadata, never data:** SQL is never executed, so
  numeric correctness is never claimed.
- Sample persona/product/reference index are illustrative placeholders.
