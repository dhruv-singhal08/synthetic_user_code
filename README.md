# synthetic-user

A **synthetic user** for automated testing of conversational products (NL2SQL, chatbot,
search, summariser, ...). It invents realistic scenarios for a persona, asks the product
questions in that persona's voice, and judges what the product returns, producing **bug
candidates** for a human to review. With a **reference** - a `data_catalog` (tables, columns,
relationships, rules) or a `document_collection` (documents with summaries), plus optional
usage instructions - runs are **grounded**: every scenario gets a **cheat sheet** picked from
the reference and checked by code, and answers are checked against it. Without one, runs are
**ungrounded** (generic checks only). Design: [docs/DESIGN_V3_REFERENCE_AND_EVALUATION.md](docs/DESIGN_V3_REFERENCE_AND_EVALUATION.md).

> **Status: MVP 1, not production-ready.** See [Limitations](#limitations).

## Pipeline

```
Persona + Product [+ Reference + instructions] ──> [reference check] ──> [2] Scenarios
        (+ Reference Pack when grounded) ──> [3] Questions ──> [4] Product Executor
        ──> [5] Evaluator ──> Bug Candidates
```

| Node | What it does | LLM? |
|---|---|---|
| Reference check | Loads your reference (data_catalog or document_collection, YAML/JSON, size-capped) and optional usage instructions; the id changes when either changes. None given -> ungrounded. | no |
| 2 Scenarios | Business goal + success criteria (guidance) for tasks this persona would do with this product. Grounded: also a **cheat sheet** (Reference Pack) - the tables / columns / joins / rule ids, or the documents, a correct answer needs, picked by exact name; code checks every pick and copies the reference's own text in (unknown picks dropped and recorded; none valid -> scenario rejected). Saved in `reference_packs.json`, editable in the UI. | yes |
| 3 Questions | Natural-language questions in the persona's voice, each with its own `expected_behavior` (written, including which readings are acceptable) and 1-4 `success_criteria`. The persona decides how it is phrased. Grounded: the cheat sheet + instructions inform the question; identifier-like names must not leak into it. | yes |
| 4 Product executor | Sends each question through the configured product adapter and records answered (text + artifacts, e.g. SQL) / clarification / denied / error / timeout. Artifacts are **never executed**. | no |
| 5 Evaluator | One generic judge per **answered** response, given only the evaluation contract (question, goal, expected behaviour, success criteria, product, full response with artifacts; grounded: the cheat sheet + usage instructions). It scores **answer relevancy** and **correctness** (0 / 0.5 / 1), rates each success criterion and labels each factual claim; code computes **completeness** and **hallucination rate**, the verdict and bug candidates. Each score comes with its reason (D206). Groundedness is still computed in `evaluations.json` but not reported for now. Clarification / denied / error / timeout are recorded only. Run KPIs (task success, metric means and pass rates, availability, latency, new vs known-gap bugs) go to `kpis.json` / `kpis.csv`. See design doc section 11. | yes |

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
| main | safechain, langchain-core, pydantic, typer, fastapi, ... | always |
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
| `reference.max_chars` | `200000` | Size limit of the reference as sent to prompts (the whole file goes in); larger files are refused |
| `trace.enabled` | `true` | Write `runs/<run_id>/run.log` (every LLM / product call, prompt + output) |
| `generation.scenario.scenario_count` | `10` | Scenarios requested per run |
| `generation.scenario.include_unsupported_scenarios` | `true` | Keep scenarios the product should refuse |
| `generation.scenario.diversity_preference` | `balanced` | Free-text hint to the scenario generator |
| `generation.scenario.max_topup_rounds` | `1` | Extra calls if too few scenarios are accepted (never padded) |
| `generation.question.question_count` | `1` | Questions per scenario |
| `generation.question.ambiguity_level` | `low` | `low` \| `medium` \| `high` mix of precise/vague/investigative |
| `generation.question.language` | `en` | Question language |
| `generation.question.allow_vague_questions` | `false` | Allow `style: vague` questions |
| `generation.question.reject_reference_leaks` | `true` | Retry/reject questions repeating identifier-like reference names, e.g. `card_transactions` (non-technical personas) |
| `paths.artifacts_dir` | `./artifacts` | Where runs are written |
| `inputs.persona_path` | `./data/personas/finance_controller.yaml` | Persona |
| `inputs.product_path` | `./data/products/sample_product.yaml` | Product description (schema v2, see [Product description](#product-description)) |
| `inputs.reference_file` | `./data/references/sample_card_spend_catalog.yaml` | Your reference (data_catalog / document_collection, .yaml / .json); omit -> ungrounded |
| `inputs.reference_guide_file` | `./data/references/sample_card_spend_instructions.md` | Optional usage instructions (.md / .txt); omit -> default for the format |
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
| `check-reference` | `--reference PATH` (required), `--reference-guide PATH` | Load a reference (+ instructions); prints id, kind, counts, size; exit 1 on load errors |
| `convert-reference` | `--input` REFIDX JSON, `--out` YAML | Convert an existing reference index into the `data_catalog` format (validated before writing) |
| `generate-scenarios` | `--run-id` (new), `--persona`/`--product` (inputs), `--reference` / `--reference-guide` (`inputs.reference_file` / `reference_guide_file`; none -> ungrounded), `--count` (`scenario_count`) | Node 2 (+ reference packs when grounded); always regenerates |
| `generate-questions` | `--run-id` (required), `--count` (`question_count`) | Node 3; always regenerates |
| `execute` | `--run-id` (required), `--adapter-config FILE` (config `product_adapter`) | Node 4 via the product adapter; always regenerates |
| `export-fixture-template` | `--run-id` (required), `--out` (`runs/<run_id>/fixture_template.json`) | Template (question_id, utterance, outcome, text, artifacts[]) to paste real answers into, for any product |
| `evaluate` | `--run-id`, **or** `--questions --executions --scenarios` (+ `--reference-packs` for grounded questions, `--out-dir`, default: executions file's folder) | Node 5 |
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

## Product description

A product description YAML (`data/products/`; schema v2, D205; template at
`/api/product-template`, linked in the UI). Example: `data/products/voice_of_customer_agent.yaml`.

| Field | Required | Used by |
|---|---|---|
| `product_id`, `product_version`, `name`, `description` | yes | all prompts |
| `business_problem` | no | scenarios (why the product exists) |
| `data_sources` (name + description) | no | scenarios (goals must fit the data), questions, evaluation, mock product |
| `supported`, `not_supported` | no | scenarios (capability + rationale), questions, evaluation (out-of-scope requests should be declined) |
| `known_limitations` | no | scenarios, questions, evaluation (findings matching them are known gaps, not new bugs) |
| `user_visible_capabilities` | no | scenarios, questions, mock product |
| `typical_user_questions` | no | scenarios (realistic goals and coverage); generated questions that copy one are rejected |
| `expected_outputs` | no | scenarios, evaluation ("an answer is expected to consist of ..."), mock product |

Files with the previous names are refused with the new name: `not_supported_yet` ->
`not_supported`, `known_gaps` -> `known_limitations`, `response_artifacts_expected` ->
`expected_outputs`; `product_type` was removed.

## Persona

A persona YAML (`data/personas/`; schema v2, D200) describes the synthetic user. Every field
is read by the scenario and/or question prompt with an explicit rule
(`prompts/persona.py`), so changing a value changes what is generated:

| Field | Required | Used by | How |
|---|---|---|---|
| `id`, `version` | yes | - | metadata only (never sent to the LLM) |
| `identity` (`name`, `role`, `description`) | name, role | scenarios, questions | who the user is |
| `expertise` | `domain` | scenarios, questions | `domain` + any named areas (`product_management`, `data_analysis`, `sql`, ...), each `low` \| `medium` \| `high`, plus `notes`. High areas -> specialist vocabulary and depth. `sql` or `technical_knowledge` high -> internal names allowed in questions (else plain language, enforced) |
| `behavior.style` | yes | questions | tone and length |
| `behavior.traits` | no | scenarios, questions | shape goals / success criteria; every question reflects at least one |
| `typical_tasks` | at least 1 | scenarios | scenarios are spread over them; each scenario records its `typical_task` (must be one of them, else rejected) |
| `primary_motivation` | yes | scenarios, questions | frames each business goal and the reason behind each question |
| `decision_inputs` | no | scenarios, questions | evidence the persona relies on; goals and questions draw on it |
| `question_preferences.complexity` | no (`mixed`) | scenarios, questions | `simple` \| `medium` \| `complex` \| `mixed`; a fixed value is enforced per question (`question_characteristics.complexity`, retry then reject) |
| `question_preferences.follow_up_questions` | no (`false`) | questions | `true` -> self-contained deeper follow-up questions (why, impact, risk) |
| `question_preferences.ambiguity` | no (config) | questions | `none` (precise only, no vague) \| `occasional` (mix, vague allowed) \| `frequent` (mostly vague / investigative); overrides `generation.question.ambiguity_level` / `allow_vague_questions`, which apply only when it is left out |
| `example_utterances` | no | questions | style examples (with the UI's example questions); never copied (near-copies rejected) |

## Web UI

```bash
synthetic-user serve                                   # http://127.0.0.1:8000
synthetic-user serve --config config/app.fake.yaml     # fully offline demo
```

A local FastAPI server (`src/synthetic_user/web/`) with a single-page UI. It binds to
`127.0.0.1` by default and has no authentication, so do not expose it on a network.
API docs are at `/api/docs`.

The UI (`static/`, D199 / D201) has a workflow sidebar, one page per step (Test setup,
Product connection, Run tests, Test report), recent runs, and a printable test report
(executive summary, the 5 quality metrics, bug candidates, breakdown by task type /
capability, results table). No external assets (system fonts, inline icons).

1. **Inputs:**
   - **Persona:** the sample persona or your own YAML upload, plus optional **sample queries**
     (one per line) that are added to the persona's example utterances to mimic the user's
     style; generated questions never copy them.
   - **Product description:** upload only (YAML), no sample.
   - **Reference:** toggle. Off (default) -> ungrounded run. On -> upload the reference
     (data_catalog or document_collection, YAML / JSON; templates linked) and, optionally,
     usage instructions (Default / Upload / Write; the box shows the default as placeholder)
     -> grounded run with a cheat sheet per scenario.
   Each file is validated on upload and previewed.
2. **Product connection:** the **product URL** plus a **contract** (YAML/JSON) that says how
   to call the product and read its reply: `method`, `headers` (`${ENV_VAR}` only for
   secrets), `timeout`, `request_template` (`{utterance}`, `{uuid}` = fresh per question, `{session_id}`) and
   `response_mapping` (dotted paths for text / artifacts, clarification / denied conditions).
   It is the `http_json` adapter's options (a `*.adapter.yaml` file works too); download a
   template from the UI (`/api/contract-template`). The URL field wins over a `url` in the
   contract. Header values are never shown or returned by the API. **Mock product** (demo
   only) instead routes questions to an LLM stand-in for the described product (using the
   reference when there is one).
   Not connected -> the configured `product_adapter` answers.
3. **Pipeline:** generate scenarios (the reference check runs first), generate questions, run
   on product, evaluate - one by one, in order, or **Run all**. Re-running a step regenerates
   everything after it. After scenarios, open a row to **review or edit its cheat sheet**
   (the picks as JSON; every name is re-checked against the reference); saving makes
   questions and later steps stale.
4. **Results:** KPI scorecard (task success, availability + latency, new bugs, and the 4
   reported quality metrics with pass rate and n), verdict distribution, filters/search and one
   row per question - the same table as `runs/<run_id>/results.csv` (28 columns: each metric
   followed by its reason, then final_verdict + verdict_reason, bug_ids, evaluation_notes; cells with several
   items use "• " bullet lines), rewritten after every step; **Download kpis.csv** for the
   run-level KPIs. Click a row
   for every field (scenario, reference pack, question + expectations, response, checks,
   findings, bug). The table and CSV show only what evaluation uses;
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
    reference_packs.json      # per-scenario Reference Pack (grounded runs only; UI-editable)
    questions.json            # QuestionCase[]
    questions_rejected.json   # with reasons
    executions.json           # ExecutionRecord[] (response: answered | clarification | denied | error | timeout)
    raw/<execution_id>.json   # raw product response body, when the adapter supplies one (never headers)
    fixture_template.json     # from export-fixture-template
    evaluations.json          # EvaluationResult[] (5 metrics, claims, criteria results, findings, verdict)
    bug_candidates.json       # BugCandidate[] (status needs_review, full id chain)
    kpis.json, kpis.csv       # run KPIs for product managers (code-computed)
    run.log                   # every LLM / product call: prompts sent, outputs received (see below)
  notebook_runs/<run_id>/     # notebooks/pipeline_walkthrough.ipynb output
```

**Run log** (`run.log`, D202): one readable text file per run, appended as each call
finishes, so `tail -f artifacts/runs/<run_id>/run.log` follows a live run. Per step a
STARTED / FINISHED banner; per LLM call the step, subject (scenario or question id), what was
asked for + attempt, model and latency, then the full SYSTEM PROMPT, USER PROMPT and OUTPUT
(for unparseable output: the error and the raw text), followed by a VALIDATION line (accepted,
or the errors that caused a retry / rejection). Per product call the question sent and the
answer recorded; for HTTP products also the request body and the reply (header values never
written). The run's own INFO / WARNING lines are interleaved. Re-running a step appends.
Switch off with `trace.enabled: false`.

All JSON writes are atomic. Ids: `RUN_…`, `SCN_001`, `RP_SCN_001`, `Q_001`, `EXE_001`,
`EVAL_001`, `F_001`, `BUG_001` - assigned by code, never by the LLM.

## Manual answer capture flow

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
   `evaluate --questions q.json --executions e.json --scenarios s.json
   --reference-packs reference_packs.json --out-dir out/` (packs needed for grounded runs).
3. **Notebook:** `notebooks/pipeline_walkthrough.ipynb` predates design v2.1/v3 and does not
   run against the current code (to be ported).

## Product adapters

The executor talks to products only through a `ProductAdapter` (`src/synthetic_user/adapters/`).
One adapter per product, chosen by `product_adapter` in the app config (inline, or a path to
`config/products/<product_id>.adapter.yaml`). Adding a product never touches nodes, prompts,
models or the pipeline.

| type | Use | Options |
|---|---|---|
| `fixture` | offline / captured answers | `path` |
| `http_json` | any JSON-over-HTTP product, config only | `url`, `method`, `headers` (`${ENV_VAR}` filled from env), `timeout`, `request_template` (`{utterance}`, `{uuid}` = fresh per question, `{session_id}`), `response_mapping` (dotted paths + `path == value` rules) |
| `python` | anything else (SDK, websockets, custom auth) | `class: "pkg.module:Class"`, `init: {...}` (keyword args) |

Examples: `config/products/sample_nl2sql.adapter.yaml` (fixture),
`config/products/sample_policy_chatbot.adapter.yaml` (http_json, dummy URL). Template for a
custom adapter: `examples/adapters/my_product_adapter.py`. HTTP errors and non-JSON bodies
become `error` outcomes with the status code; header secrets are never logged and are
written to the run manifest only as their `${ENV_VAR}` template (literal values show `***`).

## Reference: two formats, cheat sheet per scenario

A reference (`inputs.reference_file` / `--reference` / the UI) is a YAML or JSON file in ONE
of two formats (D203); anything else is refused with the field that is wrong. Templates:
`/api/reference-template/data_catalog`, `/api/reference-template/document_collection` (also
linked in the UI). Loader: `references/loader.py`.

| | `kind: data_catalog` | `kind: document_collection` |
|---|---|---|
| Content | `tables` (name, description, `rules`, `columns`: name, type, description, values), `relationships` (`from` / `to` as `table.column`, `verified`), catalog `rules` (R1, R2, ... in order) | `documents`: `name` + free-text `summary` |
| Cheat sheet (per scenario) | the tables + columns, joins (listed relationships only) and rule ids the answer must use; table rules come with their table | the documents whose content answers the goal |
| Evaluation | correctness = right tables / columns / joins, rules respected (extras fine; a NOT verified join presented as reliable is at most 0.5); claims checked against the cheat sheet | claims checked against the documents' summaries (not covered = not wrong; contradicted = hallucination); the product does not cite sources, content only |

- **Picks, not quotes:** the scenario step picks items by exact name / id; code checks each
  one exists, copies the reference's own text (descriptions, rules, summaries) into the pack
  and rejects (records) unknown picks; no valid table / document rejects the scenario. Packs
  are in `reference_packs.json` and editable in the UI (the same picks as JSON; all-or-nothing).
- **Only the cheat sheet** goes to question generation and evaluation, never the whole
  reference. The scenario step sees the whole reference (rendered compactly by code).
- **Usage instructions** (optional): `inputs.reference_guide_file` / `--reference-guide`
  (.md/.txt), or uploaded / typed in the UI. Without them a default per format is used (shown
  as the placeholder in the UI). Samples: `data/references/sample_card_spend_instructions.md`,
  `sample_travel_policy_instructions.md`.
- **Size:** the rendered reference is capped by `reference.max_chars` (default 200,000).
- Runs record `<file stem>-<hash of content + instructions>`, so editing either regenerates
  later steps.
- **Existing REFIDX index** (tables / columns / relationships / business rules JSON):
  `synthetic-user convert-reference --input index.json --out catalog.yaml`.
- Samples: `data/references/sample_card_spend_catalog.yaml` (data catalog),
  `data/references/sample_travel_policy_docs.yaml` (document collection).

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
| `... output invalid after N attempt(s)` | The LLM kept returning invalid output. Retry or raise `llm.max_validation_retries`; the reasons are in the notes / rejection files |
| Scenarios rejected with `reference pack names no table / document that is in the reference` | The LLM picked names the reference does not have. Retry; check the names in the reference are the ones users and the product use |
| `Reference ...: 'kind' must be one of data_catalog, document_collection` | The file is not in one of the two formats; download a template from the UI, or convert a REFIDX index with `convert-reference` |
| `Reference ... has N characters; the limit is ...` | Trim the file (or split it per product area) or raise `reference.max_chars` if your LLM's context allows |
| Gemini `503 UNAVAILABLE` / `504 DEADLINE_EXCEEDED` | Provider overload; re-run (completed steps are reused) or raise `llm.timeout_seconds` |
| `Run artifacts are inconsistent` | Files in `runs/<run_id>/` were edited or mixed; re-run with `--from-step` at the first broken step |
| `do not match run` / `was built from reference` | Configured persona/product/mode/reference (file or guide) differ from the run's manifest; use the original inputs or a new `--run-id` |

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
- **No code-level SQL checks** (design v3): SQL and other artifacts are judged only by the LLM
  against the Reference Pack; a broken query or a made-up column is caught only if the judge
  notices it. The artifact plugin hook is kept but disconnected.
- **Only answers are judged:** clarifications and denials are recorded, not scored (a wrongly
  declined supported request is not flagged).
- **LLM judgements are not calibrated:** confidence values are uncalibrated, and the
  benchmark labels in `tests/fixtures/benchmark/` were drafted by the assistant and still
  need human review.
- **Verdicts judge answers against the reference, never data:** artifacts are never executed,
  so numeric correctness is never claimed.
- Sample persona/product/references are illustrative placeholders.
