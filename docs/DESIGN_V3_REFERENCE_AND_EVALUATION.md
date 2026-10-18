# Design v3: generic reference + standard evaluation fields

Status: IMPLEMENTED (2026-10-18, D186-D193). Evaluation KPIs come next.

## 1. Decisions taken

| # | Decision |
|---|---|
| 1 | The **question** owns success criteria: 1-4 items, refined to the question's wording. |
| 2 | `unacceptable_behaviors` is removed; the important ones become negative criteria ("does not mix currencies"). |
| 3 | Cheat-sheet `operations`, `optional_filters`, `semantic_considerations` are **dropped**. |
| 4 | Generation keeps its rich fields; the **evaluator and the CSV/UI show only the fields evaluation uses**. Full detail stays in the JSON artifacts. |
| 5 | Reference = **any user file** (JSON / TXT / MD) + optional **usage guide** written by the user. The tool never builds a reference with an LLM. |
| 6 | Everything the LLM extracts from the reference carries a **verbatim quote**; code checks it exists in the file, else the item is rejected and the reason recorded. |
| 7 | Cheat sheet + snapshot are replaced by one generated, saved **Reference Pack** per scenario: the authoritative answer key used to validate responses. |
| 8 | `expected_behavior` is a **written description** of what the product should do for this question (not an answer / clarify / decline label). |
| 9 | **No special format**: the NL2SQL catalog JSON is just another reference file, read as text and interpreted through the usage guide. The structured catalog loader and its checks are removed. |
| 10 | **SQL plugin removed** (Stage A SQLGlot analysis, 8 SQL checks, SQL prompt, `sql` config, `sqlglot` dependency). Artifacts such as SQL are judged by the generic evaluator against the Reference Pack. |

## 2. Inputs

### 2.1 Reference file (optional; present -> `grounded`)
- Formats: `.json`, `.txt`, `.md`, all treated as text (JSON is re-indented for prompts).
- Max size `reference.max_chars` = 200,000 characters as sent to prompts (above: clear error).
- Stored as-is in the run; `reference_id` = file name + sha256[:8], recorded in the manifest.

### 2.2 Usage guide (optional, `.md` / `.txt`, written by the user)
Free text; the UI offers this template (all headings optional):

```
## What this reference is      e.g. "data catalog of VoC tables"
## How to read it              e.g. "each heading is a table, the paragraph is its description"
## What is authoritative       e.g. "descriptions are exact; examples are illustrative"
## What counts as core         e.g. "the tables and columns a correct answer must use"
## Evaluation guidance         e.g. "answers may use extra tables; currency must be stated"
## Ignore                      e.g. "the changelog section"
```
Without a guide the default is: "descriptive reference, not exhaustive; use only what it states".
The guide is passed to the scenario, question and evaluation prompts.

## 3. Reference Pack (replaces cheat sheet + snapshot)

Stored together in `reference_packs.json` (one entry per grounded scenario, pack_id
`RP_<scenario_id>`), written by the scenarios step.

| Field | Type | Rule |
|---|---|---|
| `pack_id` | `RP_SCN_001` | code |
| `scenario_id`, `reference_id` | str | code |
| `core_refs` | list, >= 1 | what a correct answer MUST rely on |
| `core_refs[].name` | str | e.g. "complaint_tickets", "card_transactions.amount" |
| `core_refs[].quote` | str | verbatim from the reference; code-checked |
| `core_refs[].why` | str | why the goal needs it |
| `rules` | list | constraints the answer must respect (e.g. default currency) |
| `rules[].rule` / `rules[].quote` | str | quote code-checked |
| `rejected_items` | list | proposed items whose quote was not found, with the reason |
| `edited` | bool | true once a person changed the pack in the UI |

Validation: quote not found in the reference (whitespace / case-insensitive) -> item rejected
with reason; no valid core_ref left -> scenario rejected (recorded).

Extra items, filters or facts in an answer are never wrong by themselves (D185).

## 4. Scenario (generation; mostly unchanged)

Kept: `business_goal`, `title`, `task_type`, `capability`, `capability_rationale`,
`success_criteria` (guidance for question generation only), `question_guidance`.
Changed: `reference_cheat_sheet` -> `reference_pack_id`.

## 5. Question

| Field | Change |
|---|---|
| `utterance`, `intended_goal`, `intent`, `question_characteristics` | unchanged (generation metadata) |
| `expectations.expected_behavior` | **new, required**: 1-3 sentences describing what the product should do for this exact question, e.g. "Return the top 10 employees by total card spend for the last 3 months, keeping currencies separate. Asking which period is meant is reasonable; refusing is not." |
| `expectations.success_criteria` | **required, 1-4**, owned by the question; may include negative items |
| `expectations.acceptable_variations` | **new** = old `acceptable_behaviors` + `interpretation_notes` merged; things NOT to penalise |
| `expectations.clarification` | **removed** (expressed in expected_behavior) |
| `expectations.unacceptable_behaviors` | **removed** (folded into criteria) |
| `interpretation_notes` | **removed** (merged above) |

## 6. Evaluation contract (exactly what the evaluator sees)

> The checks listed at the end of this section were replaced by the 5 KPI metrics in
> section 11 (the evaluator no longer sees the persona; non-answers are recorded only).

Generic (all products, both modes):
1. `question` (utterance)
2. `goal` (scenario.business_goal)
3. `expected_behavior`
4. `success_criteria` (the ONLY input that can fail goal_completeness)
5. `acceptable_variations` (can only relax the judgement)
6. `product`: supported, not_supported_yet, known_gaps, response_artifacts_expected
7. `persona`: expertise level only
8. `response`: text + every artifact (e.g. SQL) in full, up to a size cap

Grounded adds:
9. `reference_pack`: core_refs + rules (authoritative)
10. `guide` (the usage guide text, if given)

Not shown to the evaluator any more: title, task_type, capability_rationale, intent,
intended_goal, scenario success_criteria, question_guidance.

Checks (one LLM call per response):
- relevance, goal_completeness, persona_fit, capability_consistency, clarification_handling,
  safety_and_honesty (unchanged), now judged against `expected_behavior`.
- Grounded only: `reference_alignment` = the answer (incl. its artifacts) relies on the
  core_refs, or a stated equivalent, and does not contradict the rules. Replaces
  `reference_coverage` and the SQL checks.

## 7. Results CSV / UI (one row per question)

| Group | Columns |
|---|---|
| Scenario | scenario_id, mode, scenario_title, business_goal |
| Question | question_id, question, expected_behavior, success_criteria, acceptable_variations |
| Reference | core_refs (names), reference_rules |
| Product | product_outcome, answer_text, artifacts (kind: content), product_message, latency_ms |
| Evaluation | final_verdict, criteria_results, failed_checks, inconclusive_checks, findings, bug_ids, evaluation_notes |

Removed from the CSV (still in JSON artifacts): task_type, capability, capability_rationale,
scenario success_criteria, all 8 `cheat_*`, question_style, ambiguity_level,
clarification_expected, acceptable/unacceptable_behaviors, interpretation_notes,
schema_leak_warnings, adapter, artifact_kinds, generated_sql (now in artifacts), llm_verdict,
evaluation_status, plugins_applied, bug_titles.

## 8. Rule and compatibility changes

- CLAUDE.md hard rule 1: drop the SQL plugin sentence; "artifacts (e.g. SQL) are judged by the
  generic evaluator; no product-specific evaluation code".
- CLAUDE.md hard rule 2: "Reference optional: any user-supplied file (JSON/TXT/MD) plus an
  optional usage guide; never generated by the tool. Grounded items must quote it verbatim."
- Removed: SQL plugin, plugin registry, `evaluation.plugins` and `sql.dialect` config,
  `sqlglot`, the structured reference loader / integrity checks. The leak check now uses the
  pack's identifier-like core_ref names (single tokens with `_` or `.`, e.g.
  `card_transactions`); plain-language names are never leaks.
- Run folders from before v3 are not loaded (as D134); old files stay readable as files.
- The UI gets a "Reference pack" review step: packs are shown after scenarios and can be
  edited before questions are generated; edits are re-validated (quotes must still exist).

## 9. Trade-off accepted

Without the SQL plugin, nothing parses SQL: syntax errors, non-SELECT statements and unknown
columns are caught only by the LLM judge (safety_and_honesty / reference_alignment), not by
code. In exchange, every product and every reference format goes through one path.

## 10. Open choices (defaults applied unless you say otherwise)

| Choice | Default |
|---|---|
| Guide format | free text with the template above (not parsed) |
| Quote check | strict: reject |
| Max reference size | 200 KB |
| Reference pack editable in the UI before questions | yes |
| Plugin hook | kept but disconnected (`nodes/evaluators/plugins/`: protocol + empty registry; the evaluator does not call it) |

## 11. Evaluation KPIs (agreed 2026-10-18, implemented: D194-D198)

Audience: the product manager of the product under test.

### 11.1 Quality metrics (per answer 0..1; per run: mean, pass rate, n)

| Metric | Measures | Applies to | Scored by |
|---|---|---|---|
| Answer relevancy | does the answer address what was asked | every answer | judge: 0 / 0.5 / 1 |
| Correctness | key facts / query right according to the Reference Pack (extras never wrong by themselves) | grounded answers (N/A otherwise) | judge: 0 / 0.5 / 1 |
| Groundedness | supported claims / all claims | grounded answers (N/A otherwise) | code, from the judge's claim labels |
| Hallucination rate | contradicted (grounded) or fabricated (ungrounded) claims / all claims; lower is better | every answer | code, from claim labels |
| Completeness | success criteria met (met 1, partly 0.5, not met 0; unverifiable left out) | every answer | code, from criteria results |

Claims: the judge splits the answer into short fragments copied exactly from it (for a query:
its joins, filters, aggregations; framing sentences are not claims) and labels each
supported / contradicted / not_in_reference (grounded) or unverified / fabricated
(ungrounded). No claims -> groundedness and hallucination rate N/A.

### 11.2 Per question
- Final verdict (code): a metric FAILS when relevancy, correctness or completeness is 0, or
  hallucination rate > 0 (groundedness alone never fails); a failing metric backed by a
  supported finding of that category -> fail; else any failing or inconclusive metric ->
  inconclusive; else pass. Only answered responses are judged: clarification / denied / error
  / timeout are recorded only (persona fit, capability consistency and clarification handling
  were removed at the user's request).
- Also kept: criteria results, claims, findings (category = metric), bug candidates,
  llm_verdict (recorded, not decisive).

### 11.3 Operational KPIs (code only)
Task success rate (pass / judged), availability (1 - (error + timeout) / asked), latency
p50 / p95, new bugs (by severity) vs known-gap bugs (every finding matches a known gap),
inconclusive rate. Breakdowns by task type and capability in kpis.json.

### 11.4 Outputs
- `kpis.json` (RunKpis) and `kpis.csv` (one row per KPI: value, n, excluded, note), written by
  the evaluate step; UI scorecard (task success, availability + latency, new bugs, the 5
  metric means with pass rate and n); CLI prints the KPI lines after a run.
- `results.csv`: 27 columns (run 1, scenario 4, question 5, reference 2, product 5, metrics 5,
  evaluation 5); a cell with several items lists them as "• item" lines.

Not included (can be added later): product-specific custom metrics, targets / colours,
run-to-run comparison.
