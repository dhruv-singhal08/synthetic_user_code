# ONE-SHOT REPOSITORY REFACTOR INSTRUCTIONS — SYNTHETIC EVAL AGENT


## 0. Scope, precedence, and non-goals

This file is the authoritative target specification for refactoring metric selection, scenario/question generation, product execution, evaluation, SQL parsing, and reporting. It supersedes older source text that instructs the system to preserve `AGENT` and `NL2SQL` evaluation profiles, to use a Golden Set, or to make one combined question-generation call for all scenarios. **There must be no user-facing or internal evaluation-profile routing after migration. The user chooses metrics directly for the whole run. Categories are UI organization only, not profiles.**

**Preserve:** existing Persona and Product Description semantics; Reference Data authority; grounded/ungrounded *generation mode* (distinct from evaluation profile); scenario-specific SQL reference packs when relevant; supported/unsupported scenario intent; one scenario-generation LLM call for all requested scenarios; one question-generation LLM call per accepted scenario; configurable question-call concurrency default 5; deterministic/Pydantic integrity validation; one general judge and one retrieval/grounding judge when selected; deterministic KPI aggregation and bug/findings traceability.

**Remove from generation:** Golden Set parameters, imports, prompts, validation, duplicate seeding, and `golden_set_criteria`; regex/keyword-based **semantic** rejection and duplicate/conflict detection. Do not replace regex heuristics with equivalent keyword heuristics or LLM-based semantic duplicate/conflict rejection. Preserve exact identifier uniqueness, structural IDs, safe text normalization, security/leak checks, and canonical reference membership validation; these are not semantic duplicate detection. If two custom metrics overlap, honor both as specified. Do not silently merge, reinterpret, or reject them as duplicates.

**Deferred:** final verdict aggregation policy, optional/advisory metric weighting, configurable PASS thresholds. Preserve the current deterministic verdict policy behind a replaceable `VerdictPolicy` interface (or equivalent small function), clearly label it as a provisional compatibility default, and do not let LLM overall verdict override it. Do not invent new business requirements. Do not add new predefined metrics beyond the catalog below.

## 1. Mandatory repository reconnaissance

Before edits, locate and read the actual modules for: run configuration and orchestration; evaluation profile and metric enums; scenario and question models and generators; generator prompt builders/YAML; Reference/ReferencePack; ProductAdapter and adapter YAML parsing; executor and response/artifact models; Agent and NL2SQL evaluators and prompt YAML; evaluator assembly and verdict; SQL no-output detection; trace tags; KPI and CSV; benchmark fixtures and tests. Build a dependency map and search all imports/serializers/tests/CLI/API/UI/config for `EvaluationProfile`, `agent_metrics`, `NL2SQL_METRICS`, `AGENT_REQUIRED_METRICS`, Golden Set references, and fixed expectation fields. Inspect real source before selecting exact paths or making assumptions. The uploaded examples included `agent.py`, `agent_prompt.yaml`, `nl2sql.py`, `nl2sql_prompt.yaml`, `evaluator.py`, `executor.py`, `no_sql.py`, `trace_tags.py`, KPI code, and generation instructions; not every repository file was uploaded. **The repository is the source of truth for exact class signatures.**

Implement a migration for existing persisted run artifacts and configurations where possible. If an old run cannot be converted without guessing, retain a clearly labeled legacy reader or return a descriptive migration error; do not silently reinterpret old scores.

## 2. Final predefined metric catalog — exactly 11

Metric IDs should be stable snake_case. UI categories do not determine which judge or product type can be used.

| Category | Stable ID | Display name | Core comparison |
|---|---|---|---|
| Answer Quality / Task Fulfilment | `relevance` | Relevance | User question vs response: addresses intended request? |
| Answer Quality / Task Fulfilment | `correctness_completeness` | Correctness and Completeness | All requested parts delivered, and delivered content correct to the extent verifiable. Merge old Completion with proposed Answer Correctness; no separate predefined `completion` or `answer_correctness`. |
| Retrieval Quality | `context_precision` | Context Precision | Question vs retrieved chunks: relevance/focus of retrieved evidence. |
| Retrieval Quality | `context_recall` | Context Recall | Retrieved chunks vs automatically derived required information: coverage of expected information, not exhaustive corpus recall. |
| Retrieval Quality | `mrr` | Mean Reciprocal Rank (MRR) | Rank of first relevant retrieved chunk, aggregated across evaluated questions. |
| Grounding & Citations | `faithfulness` | Groundedness / Faithfulness | Material answer claims vs retrieved/tool evidence: are claims supported? One metric, not two. |
| Grounding & Citations | `citation_correctness` | Citation Correctness | Cited answer claim vs its specific cited retrieved chunk: does the cited evidence support the claim? |
| SQL Quality | `semantic_correctness` | Semantic Correctness | SQL implements requested result semantics, without claiming to verify executed values. |
| SQL Quality | `rules_adherence` | Rules Adherence | SQL complies with applicable reference business/table rules and documented joins. |
| SQL Quality | `schema_faithfulness` | Schema Faithfulness | SQL explicit database table/column identifiers conform to allowed reference schema. |
| Agent | `tool_correctness` | Tool Correctness | Correct tool/operation, source, arguments, filters, grouping, time boundaries, based on exposed trace and reference. Supersedes `tool_selection`. |

Custom is an extensible category, **not a predefined metric**. Do not add Precision@K, Recall@K, Instruction adherence, workflow completion, SQL execution, or citation coverage to the initial predefined catalog.

Preserve existing 0 / 0.5 / 1 / null metric scoring where applicable. Keep metric independence: a schema identifier problem belongs to Schema Faithfulness; a business-rule/join violation to Rules Adherence; a missing requested SQL result behavior to Semantic Correctness. Groundedness checks evidence support, not the real-world truth of retrieved sources. Correctness/Completeness must not assert factual accuracy without trusted evidence. Maintain findings with rationale, evidence, what worked, what did not, evidence support status, and trace links.

## 3. Run-level metric selection and registry

At run creation, the user chooses arbitrary built-in metric IDs and any custom metric descriptions, supplied as free text and/or Excel (XLSX; CSV optional if existing ingestion supports it). Selection applies to the entire run, not to a profile and not individually to each question. Per-question applicability can differ. Freeze a versioned, immutable resolved metric configuration snapshot before scenario generation; use exactly that snapshot through generation, execution, evaluation, and reporting. Record compiler version, metric definitions/rubrics, and stable IDs. No automatic metric merge/deduplication/conflict judgment.

Create a central `MetricDefinition`/registry with at least:

- `metric_id`, `display_name`, `category`, `version`, `origin` (`builtin` / `custom`), `description`;
- `question_expectation_schema` (only question-specific values, not duplicated authoritative reference data), `generation_instructions`;
- `required_reference_assets`, `required_execution_evidence`, `applicability_policy`;
- `judge_group` (`general`, `retrieval_grounding`, or `deterministic`), scoring rubric, score direction and scale;
- evidence requirements and an explicit optionality/requiredness definition for each field.

Do not hard-code enum-only models that prevent runtime custom metric IDs. Prefer validated metric ID strings with built-in definitions in a registry. Use Pydantic discriminated models or validated JSON schema for custom expectation values as appropriate; never execute user-supplied Python/SQL/shell as part of metric compilation. Make schema validation deterministic.

### Metric Compiler — once per run when custom metrics exist

If no custom metric input, skip the compiler LLM call and load built-in registry entries. If custom free text or spreadsheet rows exist, parse/normalize the input deterministically, then make one structured LLM compilation call for the run (normal path; no separate planner per question). Supply user text/rows, product contract, Reference Data availability, built-in registry, available question fields, and available adapter/telemetry types. Compiler output per custom metric must include:

- stable unique ID (assigned/normalized by code), exact user intent, `category=custom`, clear operational definition;
- evaluation rubric with explicit score anchors and required evidence;
- whether question-specific expectations are needed, and their structured field names/types/generation instructions;
- whether a specific reference asset or execution telemetry is required;
- judge routing (`general` by default, `retrieval_grounding` if it fundamentally needs retrieved context; `deterministic` only when a supported safe implementation exists);
- applicability and explicit unsupported/uncompilable reason.

Examples: SQL dialect adherence may need generated SQL and an explicit dialect but no generated per-question expectation; workflow adherence may need `expected_workflow`. Do not hallucinate tool catalogs, reference policies, ground truth, SQL dialects, or API endpoints. If a custom metric cannot be compiled, mark it `unavailable` with reason, preserve the original user text, and show it in run results. **Do not silently drop it or block other metrics unless the user requested an all-or-nothing run.** Do not reject a custom metric just because it resembles another metric.

Excel ingestion: read workbook rows/sheets, preserve original row provenance, handle blank/malformed rows with explicit errors, bound file size and content, treat spreadsheet text as untrusted instructions, and pass only relevant extracted requirements to the compiler. Test multiple metrics in one workbook, mixed free text + workbook, and invalid rows. Do not require an Excel plugin for this local repository functionality.

## 4. Scenario Generation (Node 2)

**One LLM call for the entire requested scenario batch**, no top-up rounds, no separate validation LLM call. Preserve configurable scenario count, supported/unsupported coverage settings, generation mode, reference packs for SQL-grounded scenarios, and partial result reporting. Inputs: Persona, Product Description, Reference Data (optional), requested count, existing signatures (if used for structural run history; not regex semantic duplicate rejection), frozen resolved metric config. No Golden Set, no evaluation profile.

Persona responsibilities: role/description, typical tasks, primary motivation, decision inputs, expertise levels/notes, traits, complexity preference determine **what** realistic business tasks to test. Persona style and example utterances belong primarily to Question Generation. Product responsibilities: name/description, business problem, data sources, supported/not supported, known limitations, user-visible capabilities, typical questions (inspiration only), expected outputs. Reference Data is authoritative for operational tools, sources, schemas, rules, joins and feasibility. If supported capability cannot be established, classify `unknown` rather than inventing support. Unsupported scenario must retain unsupported objective; do not turn it into a supported proxy.

Selected metrics may influence scenario **coverage** when organically relevant (retrieval-dependent, rule-dependent, SQL, tool-dependent tasks), but must not force unnecessary tools/retrieval/complexity or produce final utterances. Scenario output should preserve existing business goal, title, task type, capability and rationale, question intent, scenario-level success criteria, reference linkage, and Python-owned IDs/signature. **Remove `evaluation_profile` from scenario model and outputs.** Preserve `mode` (`grounded` / `ungrounded`) as a separate generation property. Scenario-level success criteria are business-level guidance, not automatically copied to question expectations.

## 5. Question Generation (Node 3)

**One independent structured LLM call per accepted scenario**, each producing its configured question count. Run scenario calls concurrently with configurable `max_parallel_scenarios`, default 5, via semaphore/worker pool. Preserve input scenario order in final results, isolate failures per scenario, accept valid items individually, record rejected/partial counts, and never silently regenerate failed items. No separate semantic-validation LLM call; no Golden Set. Strict normal-path generation call budget = `1 + accepted_scenario_count` (excluding metric compiler, execution, clarification and evaluation).

Inputs: Persona, Product Description, one scenario per LLM prompt, Reference Data and applicable ReferencePack, question generation policy, frozen metric config and **only the generation requirements for selected metrics**. Keep existing Persona question style, traits, expertise, ambiguity, follow-up preferences, and product output constraints. Preserve supported/unsupported intent; natural nontechnical questions must not leak internal SQL schema unless explicitly appropriate. Generate the utterance first; derive expectations from the **exact utterance** and authoritative rules, not a broader scenario, an invented factual answer, or a copied example.

### Stable QuestionCase shape

Keep useful non-evaluation metadata (IDs, scenario/persona links, mode, `reference_pack_id`, `intended_goal`, task intent, question characteristics, structured plan, etc.) where already used. Python owns IDs, links, complexity, reference metadata and any deterministic derived fields. Universal evaluation expectations are exactly:

```json
{
  "expectations": {
    "expected_behavior": "What a valid product response should do for this exact utterance",
    "success_criteria": ["Observable, fair requirement"],
    "metric_expectations": {
      "metric_id": {
        "status": "generated",
        "value": {"metric_specific_field": "question-specific expectation"},
        "reason": null
      }
    }
  }
}
```

Do not include fixed top-level `expected_tools`, `expected_journey`, `expected_time_period`, `expected_data_sources`, `expected_content`, `prohibited_behaviors`, `reference_context`, or `golden_set_criteria` in the new universal expectation schema. Migrate consumers carefully. A metric that requires no question-specific expectation does not need an entry under `metric_expectations`; selection comes from run config, not presence of a question field. Do not add empty objects for every selected metric.

For metric-specific expectations, support:

- `generated`: validated required expectation value exists;
- `not_applicable`: this metric does not apply to this question;
- `cannot_generate`: metric may apply but its necessary expectation cannot be established from the question and authoritative assets; include a specific reason and null value.

For `cannot_generate`, do not fabricate expected values, do not score the metric zero, do not silently treat as N/A, and do not reject an otherwise valid question solely for that one metric. If the missing material also makes the universal question/criteria incoherent, reject that question with a clear reason. If a metric needs no generated expectation, its applicability is determined later from the registry and observed evidence.

Examples of conditional fields: `tool_correctness` may need `expected_tools` / `expected_data_sources` (canonical, not mandatory exact sequence); `context_recall` needs `required_information` when derivable; a custom workflow metric may need `expected_workflow`; SQL schema/rules must come from Reference Data, not be regenerated by the LLM. `relevance`, ordinary groundedness and context precision generally need no extra generated field.

### Context Recall operational definition

When selected, automatically derive required information categories/constraints from utterance, expected behavior, success criteria, and authoritative reference. Example: comparison of January and February requires evidence for both periods and the relevant comparison dimensions. The retrieval judge compares returned chunks to these requirements. **This is expectation-based context coverage, not true exhaustive recall over the full corpus.** Never claim corpus-level recall without complete relevant-document labels. If required information cannot be determined reliably, set `cannot_generate` and preserve reason. In reports expose exactly what required-information criteria were used and the limitations of the calculation. Do not generate fictional source documents or facts.

### Validation

Retain Pydantic schema/enums, required fields, counts, canonical tool/source membership, reference-pack integrity, ID/link integrity, exact structural uniqueness, and security/leak checks. Preserve max-word, deliverable/source, follow-up, and cross-source constraints only where they already exist; do not introduce new policy controls. Remove regex-based semantic compound-output, second-request, ranking, currency, condition, or semantic duplicate/conflict rejection. Regex for basic text normalization, parsing and security checks is not prohibited. Semantic fairness is enforced by prompt instructions and offline regression testing, not brittle regex rejection.

## 6. RAG context acquisition: two mutually selectable adapter modes

For any selected `context_precision`, `context_recall`, `mrr`, `faithfulness`, or `citation_correctness`, the product connection configuration must let the user choose where retrieved context comes from:

1. **In product response:** user specifies a response JSON path/mapping in the product adapter YAML. Executor extracts retrieved chunks and citation/source identifiers from that field; no separate context API is required.
2. **Separate context API:** product connection UI/config displays an additional context endpoint connection and authentication/mapping configuration. Fetch context for the **same execution**, preferably using request/session/trace ID, not an unrelated fresh search. Do not assume that rerunning retrieval reproduces the context used by the answer.

Normalize both into one typed context artifact with `availability` (`available`, `unavailable`, `failed`, `not_requested`), retrieval timestamp, source mode, correlation ID, ranked chunks (`chunk_id`, `rank`, `text`, `source_id`, optional metadata), and citation mappings when exposed. Preserve original order; MRR requires rank. Distinguish zero retrieved chunks from missing/unavailable telemetry. Handle API failures/timeouts independently of the product answer; preserve per-question error reasons. Protect credentials and sensitive context, bound sizes, and avoid storing secrets in logs.

For citation correctness, citations must be exposed in the answer and resolvable to the **specific** retrieved chunks; source content is the same retrieved chunk content, not an invented separate corpus. Unsupported citation mapping is a missing-evidence case; missing citations where citations were required may be a genuine failure according to the metric rubric. Do not assume citation presence merely from a `citations` marker.

## 7. Product Execution (Node 4)

Preserve existing adapter request, session reuse, persona clarification handling, timeouts, errors, latency, raw response archival, order and concurrency. Add optional metric-driven telemetry capture without changing the actual user utterance to force behavior. Standardize SQL artifacts, tool-call trace (name, operation, args, result/error, order), retrieved chunks, citations, and correlation IDs. Adapter YAML must support context-in-response path and optional separate context API; validate mapping at configuration time and actual content at runtime. Use explicit evidence availability states and errors. Tool correctness requires observable tool trace and authoritative tool definitions; don't hallucinate hidden calls. Preserve clarification handling and transport outcomes. Apply the response-classification and evaluation-routing contract in Section 7.1; a terminal substantive denial is a response and is eligible for quality evaluation, while errors/timeouts and unresolved clarifications are not.

### 7.1 Product outcome, response classification, and evaluation gating (required)

Add two separately exposed columns to evaluation artifacts/reports: `product_outcome` and `response_classification`. Reuse existing typed executor outcomes rather than ask an LLM to infer network failures. The first field is the execution/transport result (`answered`, `clarification`, `denied`, `error`, `timeout`, using the repository's actual enum names). The second field describes the semantic response kind: `answered`, `unable_to_answer`, `clarification`, or `null` when no product response exists. Store classification evidence/notes when produced by a judge. Do not overwrite the executor outcome with an LLM label.

Routing rules:

| Observed terminal state | `product_outcome` | `response_classification` | Quality judging |
|---|---|---|---|
| Substantive answer | `answered` | `answered` | Run selected applicable judges |
| Product responds with inability/refusal text inside an answered payload | `answered` | `unable_to_answer` | Run selected applicable judges; a justified limitation can be correct |
| Explicit typed denial with substantive message | `denied` | `unable_to_answer` | Run applicable answer-quality judge checks, without fabricating SQL/tool/context evidence |
| Clarification request that can be continued | `clarification` (intermediate turn) | `clarification` | Existing clarification policy/handler continues conversation; no per-turn quality judge |
| Unresolved terminal clarification | `clarification` | `clarification` | Record only, no answer-quality LLM judging |
| Transport/adapter failure | `error` | `null` | Record only, no LLM judging |
| Timeout | `timeout` | `null` | Record only, no LLM judging |

For multi-turn interactions, classify and evaluate the terminal state only; preserve intermediate clarification turns in the transcript. If a clarification is answered and the product later returns a substantive answer, the final outcome is answered and normal evaluation applies. If the product later denies the request, use `denied` and evaluate the denial against product capabilities and expected behavior. A denial must not automatically fail or pass; the question may deliberately test unsupported behavior. A response that says 'I cannot answer' is not an API error. If the adapter already exposes a reliable typed response kind, use it; otherwise the existing General Judge may return `response_classification` for answered payloads as part of its normal call (no extra classifier LLM call). For a run with only retrieval metrics, classify an answered payload from explicit adapter fields when available; otherwise report classification as unknown/unresolved rather than launch an extra judge solely to classify it. If needed, extend the classification enum with `unknown` to avoid a false `answered` assumption.

Compatibility note: The existing evaluator currently marks typed denials `recorded_only`. This section intentionally changes that behavior for substantive denials; update tests, judged denominators, and outcome reporting accordingly. Keep error/timeout/unresolved clarification `recorded_only`. Preserve availability and latency calculations independently of quality verdicts. Never judge a timeout/error merely to generate a metric score.

## 8. SQL parser plugin / component

Introduce a local, pluggable SQL AST parser (SQLGlot or an equivalent maintained parser already used by the repo) for generated SQL. This is a code plugin/interface, not a separate LLM judge or an external product plugin requirement. Capture SQL dialect from explicit user/product/reference configuration when available; do not guess dialect-specific correctness. Extract tables, base columns, alias resolution, CTEs, joins and join predicates, filters, projections, aggregations, grouping, ordering, limits and parse diagnostics; preserve raw SQL and normalized structured representation. Distinguish base database identifiers from CTE names, SQL output aliases and functions. Provide parser output to the SQL metric rubrics in the General Judge along with authoritative ReferencePack and exact question expectations. The parser is supporting evidence, not a substitute for semantic evaluation. Parser failure must be explicit and must not automatically fail all SQL metrics; judge original SQL where possible and mark parser-dependent checks inconclusive if not possible. No SQL execution or factual result verification is implied.

## 9. Two independent evaluation judge groups

After product execution and context capture, determine metric applicability **deterministically** from the frozen metric registry, question-specific expectation status, reference availability, and actual evidence quality. Do not rely solely on matching arbitrary artifact key names. Group applicable LLM metrics into:

- **General Judge:** one LLM call per evaluable terminal response when needed for `relevance`, `correctness_completeness`, `semantic_correctness`, `rules_adherence`, `schema_faithfulness`, `tool_correctness`, and custom metrics routed to general. Use parser output for SQL checks. Ground correctness only in reliable available evidence; if factual correctness cannot be verified, explicitly mark the limitation rather than award full factual correctness. Do not conflate answer correctness with SQL static semantics.
- **Retrieval/Grounding Judge:** one separate LLM call per evaluable terminal response when needed for `context_precision`, `context_recall`, `faithfulness`, `citation_correctness`, relevance judgments for `mrr`, and custom metrics explicitly requiring retrieved context. Use the normalized context artifact, answer/citations and required-information expectations. Keep retrieval evidence separate from general judgment to avoid giant prompts and metric interference.

The two judges may run concurrently. **Do not merge them into one LLM call.** If only one group has selected applicable metrics, call only that group. If no LLM metrics are applicable, make zero judge calls. MRR is not a separate LLM call: the retrieval judge returns per-chunk relevance labels, then Python calculates per-question reciprocal rank (`1 / first_relevant_rank`, else `0`) and run-level mean. If ranked chunks/relevance judgments are unavailable, mark MRR unscorable with reason; never assume arbitrary chunk order is retrieval rank. Do not confuse a single-question reciprocal rank with run-level MRR.

Both judges output per-metric scores/status, rationale, evidence quotes/source IDs, findings, what_worked/what_didnt, criterion results where relevant, and trace links. Preserve one score per selected metric, with a clear state for selected-but-unavailable metrics. Validate judge JSON structurally and validate quoted evidence against actual answer/SQL/chunk/tool artifacts as appropriate. No third planner LLM call per question. Existing validation retries may be configurable, but the normal-path budget is at most **two evaluation LLM calls per answered question** (one per judge group). Do not introduce a separate LLM validation call.

### Metric status taxonomy

Distinguish: `scored`, `not_applicable` (genuinely irrelevant), `cannot_generate` (required question expectation absent), `evidence_unavailable` (applicable but required execution context/trace missing), `inconclusive` (judge cannot reliably decide despite evidence), `evaluation_error` (judge/parse/API failure), and `not_enabled` (not selected). `unavailable` is also allowed at compiled custom metric definition level. Non-scored statuses never silently become score 0 or score 1. Record reason, missing requirement and source of failure. You may represent these with a stable internal enum and public display mapping; do not force a schema change to every downstream consumer without migration.

### Metric-specific scoring boundaries

- **Relevance:** direct responsiveness, not factual correctness.
- **Correctness and Completeness:** all requested components + correctness of delivered components where independently verifiable. If content correctness is unverified, do not assert full verified factual correctness; record uncertainty at metric/criterion level and preserve completed-part observations.
- **Context Precision:** relevance of retrieved evidence to the task, not answer truth.
- **Context Recall:** coverage of generated required-information checklist, with explicit disclosure that this is not full corpus recall.
- **MRR:** rank of first relevant chunk; deterministic aggregation over valid evaluated questions; define denominator and exclusions in report.
- **Faithfulness:** claims supported by retrieved context/tool evidence; not a guarantee the source itself is true.
- **Citation Correctness:** cited claim specifically supported by referenced retrieved chunk; missing required citations vs unavailable citation mapping distinguished.
- **Semantic Correctness:** SQL requested result behavior, no execution-ground-truth claims.
- **Rules Adherence:** applicable rules and documented joins only.
- **Schema Faithfulness:** explicit base table/column names only; no aliases/CTEs/functions.
- **Tool Correctness:** correct tool, operation, args, source, filters and boundaries; do not penalize a correct selection solely because the external tool returned an error.

## 10. Final verdict: deliberately deferred

Retain a small deterministic verdict calculator, not the LLM's advisory overall verdict. For backward compatibility, the existing `metrics_verdict()` semantics may serve as the temporary default: a metric below 1 with a supported finding in that category -> FAIL; otherwise a sub-perfect or inconclusive required metric -> INCONCLUSIVE; otherwise PASS. Preserve special cases such as no SQL where applicable, but use Section 7.1 for outcome routing: errors, timeouts, and unresolved clarifications are recorded-only; substantive denials are eligible for applicable answer-quality evaluation. Do not freeze new threshold/weight/advisory policy or allow `not_applicable`/missing telemetry to count as passing evidence. Design it so aggregation rules can change later without regenerating questions or rerunning judges; retain granular scores, statuses and evidence. If selected metrics have no usable scores, do not report a misleading PASS. Document provisional compatibility behavior.

## 11. Reporting and KPI migration

Retain existing metric means and perfect-score pass rates over scored answers only, N/A/inconclusive excluded, latency p50/p95, availability, bug severities, known-gap tracking, and task/capability breakdowns. Preserve original metric IDs in legacy reports where needed and map `completion -> correctness_completeness` only through an explicit migration, not silent historical score rewriting. `tool_selection -> tool_correctness` is an expanded definition; do not relabel historical tool-selection scores as if they were judged under the new rubric. New reports must list selected metrics, compiler failures, per-metric denominators, scored/N/A/cannot-generate/evidence-unavailable/inconclusive/error counts, category grouping, and reasons. Display how Context Recall required-information criteria were derived, and that its score is expectation-coverage rather than exhaustive recall. Display per-question reciprocal rank and run-level MRR with clear denominator and missing-rank exclusions. Keep category aggregates descriptive; do not invent weighted composite category scores. Keep accuracy and task-success labels accurate; avoid presenting two identical formulas as different measurements without explanation.

Update finding consolidation to work for custom metric IDs without `KeyError` in a fixed fallback map. Preserve stable theme reporting where possible. No regex-based semantic duplicate/conflict rejection of metrics, questions or findings; existing report theme regex classification is not a validation gate but may be migrated to structured issue codes when practical. Preserve bug candidate provenance and evidence status.

## 12. Configuration, Evaluation UI, interfaces, and compatibility

**Authoritative UI ownership:** Add an Evaluation tab in the existing application UI. The user selects all built-in metrics there, supplies optional custom metric definitions as free text or Excel, and sets scenario count and questions per scenario there. These selections apply to the entire run. The UI submits a run-creation payload; the backend resolves and freezes the selected metric definitions before scenario generation. Do not create or expose evaluation profiles. The tab should display the existing question-complexity selection (such as simple, medium, mixed) **only if and as already supported by the repository**; preserve its existing option names, defaults, semantics, and location/behavior where possible. Do not invent new complexity modes.

**Internal-only configuration:** `max_parallel_scenarios` controls the number of concurrent per-scenario question-generation LLM calls. It must be configurable internally, with a default of 5, but must not be exposed as a user-facing Evaluation tab control. Preserve existing LLM/product concurrency controls and unrelated operational settings.

Do not add speculative generation settings. The earlier illustrative settings `max_words`, `allow_follow_ups`, `allow_cross_source_comparison`, `max_deliverables`, `max_sources`, and `max_success_criteria` are not requirements to introduce. Inspect actual repository models/config/prompt logic first: if any already exist, preserve their current names, defaults, validation, and behavior; if absent, do not implement them merely because they appeared in an earlier draft. Do not change the existing question complexity behavior or remove working generation limits. Preserve scenario count and question count as user-selectable run inputs, wiring them to existing config names rather than duplicating settings.

Illustrative UI run-creation payload (adapt names to existing API contracts):

```json
{
  "evaluation": {
    "selected_metrics": ["relevance", "correctness_completeness", "context_precision"],
    "custom_metrics": []
  },
  "generation": {
    "scenario_count": 20,
    "questions_per_scenario": 3
  }
}
```

If the existing UI already exposes a question-complexity preference, retain and submit that existing field without renaming or changing its values. Do not add it to the payload as a new field unless the existing API requires it. The metric list above is illustrative, not a default or a mandate to select all metrics.

Illustrative internal configuration (not exposed in UI):

```yaml
generation:
  question:
    max_parallel_scenarios: 5
```

Reuse the repository's actual configuration layout and naming where possible. Do not create a second competing source of truth for scenario count or question count. Existing server defaults may remain as fallbacks when a UI request omits an optional field, but the user's submitted run selections take precedence.

**Context connection UI:** When any selected metric needs retrieved context (`context_precision`, `context_recall`, `mrr`, `faithfulness`, `citation_correctness`), the product connection flow must show a context-source selector. Choose between a field/path in the product response mapped in adapter YAML, or a separate context API with its connection/authentication and execution-correlation mapping. This connection configuration belongs to the product connection interface, not to a new generation-policy control.

**UI implementation rule:** Inspect the real frontend routes, components, state management, and backend run-creation API. Implement the Evaluation tab and wire it end-to-end if frontend code is present. If no frontend exists in the repository, implement/validate backend request contracts and provide explicit UI integration requirements; do not claim the tab was implemented. Keep the UI compact and do not expose internal concurrency, retry, validation, or arbitrary generation policy settings.

Persist the exact run-level metric configuration, compiled custom definitions, and user-selected generation counts. Schema-version stored artifacts. Ensure evaluation uses the same frozen configuration used for question generation. Handle legacy `evaluation_profile` and `agent_metrics` with explicit migration/errors, never a hidden profile fallback. Do not alter grounded/ungrounded generation mode based on selected metric categories.

## 13. Testing and acceptance criteria

Implement focused unit, contract, integration and regression tests, with fake LLMs/adapters where necessary:

1. No profile-based branching or `EvaluationProfile` required for a new run; arbitrary cross-category metric selections work.
2. Exactly 11 built-in metrics with the agreed IDs/categories; no separate `completion`, `answer_correctness`, `tool_selection` in the new predefined catalog.
3. Compiler skipped for built-in-only run; exactly one compiler call per run containing custom metrics (free text, workbook, mixed); malformed/uncompilable metrics surfaced, not dropped; no duplicate/conflict semantic rejection.
4. Scenario generator one call total; question generator one call per accepted scenario; concurrency defaults to 5 and is configurable; stable order and partial failures.
5. Golden Set absent from generation interfaces, prompts and validation; reference grounding independent of Golden Set; supported/unsupported intent preserved.
6. Universal `expected_behavior` and `success_criteria` always present; only required metric-specific expectations generated; `cannot_generate` recorded and other metrics still run.
7. Context Recall checklist generated from question and reference, transparently reported, with no unsupported claim of full corpus recall.
8. Adapter response-field context and separate context API both normalize to same ranked chunk model; context correlated to original execution; missing vs empty context distinguished.
9. Citation references map to specific retrieved chunks; unsupported citations, missing citations and missing telemetry handled distinctly.
10. SQL AST extraction handles joins, aliases, CTEs, columns, aggregation, filters and parser failure; no SQL execution; correct metric boundaries.
11. General judge and retrieval/grounding judge are separate, at most one normal call each per answered question, concurrent where possible; MRR adds no third call.
12. Per-question reciprocal rank and run-level MRR are correct for first relevant rank 1, rank 3, no relevant chunk, empty ranked results, and unavailable rank; denominator explicit.
13. Relevance vs merged Correctness/Completeness independent; missing factual evidence never silently treated as verified correctness.
14. Tool Correctness uses actual trace and canonical reference; correctly configured tool returning error not automatically judged wrong.
15. Custom metrics requiring per-question expectations and those requiring none both work; compiler-defined retrieval-dependent custom metrics route to retrieval judge.
16. Status distinctions and reasons survive serialization, CSV/report rendering, partial failures and judge validation errors.
17. Two result columns (`product_outcome`, `response_classification`) persist and appear in reporting; answered/inability/typed denial are classified appropriately; existing multi-turn clarification is reused; unresolved clarification, timeout and error never invoke judges; substantive denials can be judged; availability, bugs, severity, known gaps and traceability remain correct.
18. No regex-based semantic acceptance/rejection/duplicate/conflict heuristics remain in generation or custom metric compilation; deterministic structural validation stays.
19. Existing completed-run regressions for unsupported-scenario drift, irrelevant extra criteria, merchant-type aggregation, and reference-pack integrity remain covered.
20. Old persisted runs/configurations have documented compatibility/migration behavior; no historical scores silently relabeled under new semantics.

Run formatting/lint, type checking if configured, focused tests, then full test suite where feasible. If any test cannot run, report the exact missing dependency/credential and which checks were actually executed. Do not claim all tests pass without running them.

## 14. Deliverables and execution protocol

1. Implement all required source changes, not just a proposal.
2. Update model definitions, registry, compiler, generators/prompts, adapter/context config, executor, SQL parser, judges, verdict adapter, KPI/reporting, orchestrator/API/CLI/UI integration as available.
3. Add migrations and regression tests; remove dead profile/Golden Set code paths from active generation/evaluation.
4. Produce a concise file-by-file change log and note any unavoidable compatibility breaks.
5. Report actual lint/type/test commands and results; identify missing integrations precisely.
6. Provide a brief architecture summary with exact run-level LLM call budget: compiler 0 or 1; scenario generation 1; question generation 1 per accepted scenario; evaluation 0–2 per answered question depending on applicable judge groups; no separate MRR or validation LLM calls on the normal path.
7. Keep the final verdict policy clearly labeled provisional and isolated. Do not make unrequested product-policy decisions.

**Success condition:** a new run can select any combination of the 11 built-in metrics and compiled custom metrics, generate the necessary question-specific expectations, collect matching execution evidence including either context acquisition mode, evaluate through the two independent judge groups, and report transparent per-metric outcomes without any evaluation-profile routing or Golden Set dependency.
