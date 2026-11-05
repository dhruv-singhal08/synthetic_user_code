# FINAL IMPLEMENTATION PROMPT — SYNTHETIC EVALUATION AGENT

> Transcribed from 18 photos of the instruction note (5 November 2026, 7:42 AM).
> Text in `[brackets]` was cut off at the right edge of the photo and has been completed
> from context. **Gap:** the photos skip from the end of section 22 (Part 13) to Part 15;
> any further Part 13 sections (e.g. 23) and all of Part 14 were not captured.

## Role and objective

You are a senior Python architect specializing in LLM-based synthetic evaluation, structured
generation, NL2SQL evaluation, agent evaluation, SQLGlot, and evidence-based scoring.

Work directly in my existing Synthetic Evaluation Agent repository.

Your task is to inspect, implement, test, and verify the following architecture. This is not a
design-only exercise.

## Primary objective

For every evaluation metric score, the system must explain:

1. What was evaluated.
2. What was compared against what.
3. Which Success Criteria were evaluated.
4. Which Reference Spec tables, columns, joins, and rules were used.
5. Which Evaluation Guidance instructions influenced the decision.
6. Which actual product response, SQL expression, or tool-call evidence was inspected.
7. Which requirements passed, failed, were not applicable, or could not be verified.
8. Why the final score and overall verdict were assigned.

Both run.log and the final evaluation report must explicitly convey this information.

## Core principle

Metric → Mapped Criteria + Reference Requirements + Applied Guidance → Actual Evidence → Check
Decisions → Score.

---

## PART 1 — ARCHITECTURE

### 1. Preserve the existing pipeline

Implement the following architecture:

Scenario Generation → Reference Pack → Question Generation → Evaluation Context → Product
Execution → General Judge → Evaluation Report

Do not introduce additional pipeline nodes.

Preserve:

- Existing Scenario Generation.
- Existing Question Generation.
- Existing Reference Pack construction and enrichment.
- Existing Product Executor.
- Existing General Judge.
- Existing SQLGlot parsing.
- Existing deterministic catalog reconciliation.
- Existing bounded evidence-gathering mechanism.
- Existing metric scoring and verdict policies wherever possible.
- Existing Agent and NL2SQL evaluation profiles.

Do not introduce:

- A new Reference Resolver.
- A new Python catalog-validation stage.
- A separate SQL evaluator.
- An unrestricted evaluation-agent loop.
- Unnecessary LLM calls.
- Unrelated architectural refactoring.

The goal is to improve the existing pipeline, not replace it.

### 2. Audit before implementation

Inspect the actual repository and identify:

- Scenario generation code and prompts.
- Question generation code and prompts.
- Persona and Product Description models.
- Reference Data and Reference Pack models.
- Golden Set usage.
- Product Executor.
- General Judge implementation.
- General Judge prompt builders and YAML templates.
- Evaluation Guidance sources.
- SQLGlot parsing.
- Catalog reconciliation.
- Optional evidence lookup functions.
- Metric configuration and scoring.
- Final verdict aggregation.
- Logging and tracing.
- Report generation.
- Existing tests.

Trace the complete data flow.

Identify which generated fields are actually consumed and which are redundant.

Do not assume a function, field, or behavior exists without inspecting the code.

Establish a baseline by running relevant existing tests before modifications.

---

## PART 2 — SCENARIO GENERATION

### 3. Scenario Node responsibilities

Scenario Generation determines what business objective should be tested.

Inputs include:

- Persona.
- Product Description.
- Reference Data.
- Golden Set.
- Evaluation profile.
- Generation configuration.

Outputs must preserve:

- Scenario ID.
- Business goal.
- Task type.
- Capability classification.
- Question intent.
- Relevant scenario metadata.
- Selected Reference Pack.

#### Improve the Scenario prompt

The Scenario LLM must:

1. Generate realistic business goals aligned with the persona.
2. Respect product capabilities and limitations.
3. Use Reference Data as the authoritative technical scope.
4. Use Golden Set examples for quality and coverage guidance, not as mandatory templates.
5. Preserve supported and unsupported scenario intent.
6. Select the minimal but sufficient Reference Pack for the scenario.
7. Avoid unrelated tables, columns, joins, and rules.
8. Avoid inventing reference identifiers.

#### Reference Pack selection

The Scenario Node must select, where applicable:

- Required tables.
- Required columns.
- Relevant joins and relationships.
- Table-level rules.
- Catalog-level rules.
- Technical constraints.
- Prohibited behaviors.
- Reasons for selecting reference items.

Selection must account for:

- Requested measures.
- Aggregations.
- Grouping dimensions.
- Filters.
- Time periods.
- Comparisons.
- Required calculations.
- Applicable business rules.

Preserve existing Python Reference Pack enrichment and construction.

Do not add a separate reference-selection or catalog-validation stage.

**Important:** The Reference Pack describes the expected authoritative evidence scope. It must
not automatic[ally fail] an alternative correct implementation that is supported by the full
catalog.

---

## PART 3 — QUESTION GENERATION

### 4. Question Node responsibilities

The Question Node receives:

- Persona.
- Product Description.
- Approved Scenario.
- Scenario Reference Pack.
- Relevant Golden Set examples.
- Question generation settings.
- User-selected evaluation metrics.

It must generate:

1. The actual user utterance.
2. The structured question plan, where existing functionality requires it.
3. The complete Evaluation Context.

#### Question generation sequence

The Question LLM must:

1. Generate the user utterance first.
2. Keep it aligned with the scenario's business goal.
3. Keep it within the selected Reference Pack's supported scope.
4. Preserve persona style, expertise, and realistic wording.
5. Preserve unsupported requests when testing product limitations.
6. Derive Success Criteria from the exact utterance.
7. Include relevant reference-grounded requirements.
8. Map the selected metrics to applicable evaluation requirements.

Do not allow the generated question to silently introduce unsupported metrics, dimensions,
filters, calcul[ations, or] operations.

If the model cannot generate a valid question within the scenario's reference scope, return a
clear gener[ation failure] rather than fabricating requirements.

#### Remove expected_behavior

Remove `expected_behavior` as an independent field in the new Evaluation Context.

Its necessary meaning must be expressed through Success Criteria.

Update:

- Generation prompts.
- Pydantic models.
- Question serialization.
- Evaluation prompt builders.
- Evaluation inputs.
- Reports.
- Tests.

Handle existing stored records through an explicit backward-compatibility or migration strategy.

Do not silently discard meaningful legacy expectations.

#### Remove unnecessary generated metadata

Audit fields such as:

- `metric_expectations`
- `question_characteristics`
- `intent.task_type`
- `plan.deliverables`
- `plan.filters`
- `plan.is_follow_up`

Keep fields that are actually needed for execution, validation, coverage, reporting, or
evaluation.

Derive deterministic metadata in Python where feasible.

Do not remove existing fields merely because the General Judge does not consume them directly.

---

## PART 4 — EVALUATION CONTEXT

### 5. Central Evaluation Context contract

Each generated question must contain:

```json
{
  "evaluation_context": {
    "success_criteria": [],
    "reference_spec": {},
    "metric_mapping": {}
  }
}
```

These three components together define the question-specific evaluation contract.

Evaluation Guidance remains separate because it belongs to evaluator configuration.

### 6. Success Criteria

Success Criteria define what the product must accomplish.

Each criterion must be:

- Atomic.
- Observable.
- Independently testable.
- Question-specific.
- Grounded in the utterance, product contract, or authoritative reference.
- Assigned a stable ID such as C1, C2, C3.

Examples:

- Calculate the requested measure.
- Group by the requested dimension.
- Apply the correct period.
- Calculate the requested comparison.
- Return the requested output.
- State required limitations.
- Refuse appropriately when a request is unsupported.

Do not artificially limit criteria to two when additional requirements are necessary.

Do not duplicate the same requirement across multiple criteria.

Do not introduce unrelated requirements.

For unsupported scenarios, criteria must describe the correct limitation, refusal, or supported
[alternative].

#### Example

```json
{
  "success_criteria": [
    {
      "id": "C1",
      "requirement": "Calculate total delinquent balance by BCA."
    },
    {
      "id": "C2",
      "requirement": "Use the last 12 complete billing months."
    },
    {
      "id": "C3",
      "requirement": "Compare against the corresponding previous-year period."
    },
    {
      "id": "C4",
      "requirement": "Calculate year-over-year changes."
    }
  ]
}
```

Prefer assigning or normalizing stable IDs in Python rather than relying on arbitrary
LLM-generat[ed IDs].

### 7. Reference Spec

The Reference Spec is the technical and business reference component of the Evaluation
Contex[t].

It must reuse the Reference Pack selected during Scenario Generation.

Include relevant existing reference information:

- Tables.
- Columns.
- Joins.
- Relationships.
- Table rules.
- Catalog rules.
- Restrictions.
- Prohibited behaviors.
- Source descriptions.
- Available reference identifiers and versions.

Do not ask the Question LLM to reselect or independently modify the Reference Pack.

Reuse the existing Reference Pack and preserve its identity and provenance.

Avoid copying large reference documents unnecessarily. Use existing pack objects, identifiers, or
references where possible.

The Reference Spec should not become another free-text Success Criteria list.

#### Important distinction

Success Criteria define required business outcomes.

Reference Spec defines authoritative technical constraints.

The evaluator must use both according to Metric Mapping.

### 8. Metric Mapping

Metric Mapping defines which requirements each selected metric must evaluate.

Only map metrics enabled by the user's evaluation configuration.

Preserve existing metric names and identifiers.

For each metric, specify:

- Applicable Success Criteria IDs.
- Applicable Reference Spec requirements or rule IDs.
- Comparison basis.
- Required evidence type.

Example:

```json
{
  "metric_mapping": {
    "relevance": {
      "criteria": ["C1", "C2", "C3", "C4"],
      "reference_requirements": [],
      "comparison_basis": "question_vs_response"
    },
    "correctness_completeness": {
      "criteria": ["C1", "C2", "C3", "C4"],
      "reference_requirements": [],
      "comparison_basis": "criteria_vs_response"
    },
    "semantic_correctness": {
      "criteria": ["C1", "C2", "C3", "C4"],
      "reference_requirements": [],
      "comparison_basis": "criteria_vs_sql"
    },
    "rules_adherence": {
      "criteria": [],
      "reference_requirements": [
        "TABLE-RULE-01.02",
        "TABLE-RULE-01.07"
      ],
      "comparison_basis": "rules_vs_sql"
    },
    "schema_faithfulness": {
      "criteria": [],
      "reference_requirements": ["reference_spec.schema"],
      "comparison_basis": "schema_vs_sql"
    }
  }
}
```

The IDs are illustrative. Use actual IDs from the repository.

#### Mapping rules

The Question LLM may propose Metric Mapping as part of its structured output.

Python should perform lightweight structural normalization and integrity checks.

Verify:

- Every mapped criterion ID exists.
- Every mapped reference requirement exists in the supplied Reference Spec.
- Every selected metric has a mapping.
- No unselected metric is evaluated.
- No mapped requirement is silently omitted.
- No criterion relevant to an enabled metric is silently excluded.

Do not introduce new catalog-based validation or reference resolution.

Preserve existing metric definitions and scoring behavior.

---

## PART 5 — EVALUATION GUIDANCE

### 9. Evaluation Guidance is a separate authoritative input

Inspect the existing Evaluation Guidance from:

- General Judge system prompts.
- General Judge user prompts.
- YAML templates.
- Metric definitions.
- Scoring policies.
- Reference interpretation instructions.
- Evaluation configuration.
- Exceptions and special-case rules.

Evaluation Guidance may define:

- Metric interpretation.
- Scoring rules.
- Reference interpretation.
- Applicability rules.
- Exceptions.
- Evidence sufficiency.
- Verification limitations.

Do not merge Evaluation Guidance into generated Success Criteria.

Do not ask Question Generation to reproduce evaluator-specific guidance.

#### Guidance traceability

When guidance materially influences a metric decision, record:

- Guidance ID.
- Guidance source.
- Applicable metric.
- Exact instruction or resolvable source reference.
- Guidance type.
- Applicability reason.
- How it influenced the decision.

Guidance types may include:

- `evaluation_requirement`
- `interpretation_rule`
- `exception`
- `scoring_policy`
- `evidence_policy`

Use actual source-backed guidance identifiers.

Do not invent guidance requirements.

Do not treat every generic prompt sentence as a separate scoring rule.

#### Guidance and Metric Mapping

Metric Mapping defines the question-specific evaluation scope.

Evaluation Guidance defines how that scope is interpreted and scored.

If guidance requires an additional mandatory check or exception, the evaluator must
exp[licitly record it as a] derived check.

Do not silently apply hidden requirements.

---

## PART 6 — PRODUCT EXECUTION

### 10. Preserve Product Executor

Preserve existing product execution behavior.

Continue recording:

- Question ID.
- Scenario ID.
- Submitted utterance.
- Actual delivered request.
- Product response.
- SQL artifacts.
- Tool-call traces.
- Conversation turns.
- Clarification behavior.
- Errors and timeouts.
- Latency and execution metadata.

Keep the Evaluation Context associated with its corresponding question and execution.

Do not expose internal execution instructions as user-facing requirements.

Do not treat adapter errors as incorrect product answers.

Do not unnecessarily change the adapter interface.

---

## PART 7 — SQLGLOT AND DETERMINISTIC EVIDENCE

### 11. Preserve SQLGlot

For NL2SQL:

SQL Artifact → SQLGlot Parsing → Structured SQL Evidence → General Judge

Preserve the original SQL and the parsed evidence.

Use existing SQLGlot functionality to extract, where supported:

- Tables.
- Columns.
- Joins.
- Filters.
- Aggregations.
- Grouping.
- Ordering.
- Limits.
- SQL expressions.

SQLGlot must not:

- Generate Success Criteria.
- Generate Metric Mapping.
- Select Reference Packs.
- Assign metric scores.
- Claim database execution correctness.

If parsing fails, record the failure and preserve existing evaluation error handling.

#### Preserve catalog reconciliation

Keep the existing:

```
SQL → SQL Parser → Catalog Reconciliation → General Judge
```

Preserve identifier classifications:

- `valid_in_pack`
- `valid_extra`
- `invalid`
- `unresolved`

Do not automatically fail valid catalog identifiers outside the selected Reference Spec.

Do not treat CTE aliases, SQL aliases, or function names as invalid base identifiers.

Do not introduce a new catalog-validation layer.

---

## PART 8 — GENERAL JUDGE

### 12. Preserve the existing General Judge

The Judge receives:

- Original question.
- Evaluation Context.
- Product response.
- Original SQL where applicable.
- SQLGlot analysis.
- Catalog reconciliation evidence.
- Agent tool-call trace where applicable.
- Evaluation Guidance.
- Existing available reference evidence.

The Judge must evaluate only the selected metrics.

#### Evaluation procedure

For each metric:

1. Read its Metric Mapping.
2. Identify assigned Success Criteria.
3. Identify assigned Reference Spec requirements.
4. Identify applicable Evaluation Guidance.
5. Inspect actual response, SQL, or tool-trace evidence.
6. Evaluate every mapped requirement.
7. Record check-level decisions.
8. Apply existing scoring policy.
9. Return an evidence-backed metric score.

Allowed check statuses:

- `passed`
- `failed`
- `not_applicable`
- `unresolved`

The Judge must not invent evidence or requirements.

It must not silently omit mapped requirements.

### 13. Preserve SQL Semantic Correctness

Continue evaluating SQL semantics through static SQL analysis and LLM reasoning.

Compare:

- Original question.
- Mapped Success Criteria.
- Original SQL.
- SQLGlot evidence.
- Available reference evidence.
- Applicable guidance.

Evaluate relevant:

- Measures.
- Aggregations.
- Grouping.
- Filters.
- Time periods.
- Comparisons.
- Ranking.
- Calculations.
- Join semantics.

Record exact supporting SQL expressions.

Do not claim database execution or result verification.

### 14. Preserve Schema Faithfulness

Continue using deterministic catalog reconciliation and Judge interpretation.

Record:

- Parsed identifiers.
- Catalog reconciliation results.
- Expected Reference Spec.
- Valid identifiers.
- Additional valid identifiers.
- Invalid identifiers.
- Unresolved identifiers.
- Applied alias/equivalence guidance.
- Evidence supporting the final judgment.

The full catalog remains the authority for identifier validity.

### 15. Preserve Rules Adherence

Evaluate applicable rules from:

- Reference Spec.
- Existing catalog evidence.
- Optional evidence lookups.
- Applicable Evaluation Guidance.

For each rule, record:

- Rule ID.
- Rule source.
- Applicability.
- Expected constraint.
- Observed evidence.
- Compliance decision.
- Reason.

Report:

- Rules in evaluation scope.
- Applicable rules.
- Not-applicable rules.
- Passed rules.
- Failed rules.
- Unresolved rules.

Do not calculate adherence using all retrieved rules as the denominator.

Do not treat unresolved applicable rules as passed.

### 16. Preserve Agent tool-calling evaluation

Use:

- Tool definitions.
- Tool-call trace.
- Tool names.
- Tool arguments.
- Available results.
- Success Criteria.
- Reference Spec.
- Evaluation Guidance.

Evaluate:

- Tool selection.
- Argument suitability.
- Required tool usage.
- Missing required calls.
- Unsupported or unnecessary calls when demonstrable.
- Failed calls and recovery where visible.

Record the actual tool calls and evidence inspected.

Do not claim complete trajectory correctness if the available trace is incomplete.

Do not add a separate trajectory evaluator.

---

## PART 9 — BOUNDED EVIDENCE GATHERING

### 17. Preserve optional Judge-requested lookups

Preserve the existing bounded evidence-gathering mechanism.

The Judge may request one additional batch of exact reference lookups when evidence is
insufficie[nt].

Keep existing lookup operations:

- `lookup_schema`
- `lookup_rules`
- `lookup_relationships`
- `lookup_column_alternatives`

Preserve existing lookup limits and error handling.

Do not introduce an unrestricted agent loop.

#### Lookup traceability

For each lookup, record:

- Lookup round.
- Operation.
- Arguments.
- Reason for lookup.
- Result status.
- Returned evidence IDs.
- Whether the evidence influenced the final judgment.

If no optional lookup occurs, explicitly record:

```
additional_lookup_performed = false
```

Distinguish:

- Initially supplied Reference Spec.
- Existing deterministic catalog reconciliation.
- Precomputed catalog lookups.
- Optional Judge-requested lookups.

Do not claim exhaustive evidence retrieval.

---

## PART 10 — EVALUATION OUTPUT

### 18. Metric-level output contract

Every evaluated metric must return:

- Metric ID.
- Score.
- Comparison basis.
- Assigned Success Criteria IDs.
- Assigned Reference Spec requirement IDs.
- Applied Evaluation Guidance IDs.
- Check-level decisions.
- Observed evidence IDs.
- Coverage.
- Verification method.
- Rationale.

Illustrative output:

```json
{
  "metric": "semantic_correctness",
  "score": 1.0,
  "comparison_basis": "criteria_vs_sql",
  "checks": [
    {
      "check_id": "CHECK:001",
      "requirement_source": "success_criteria",
      "requirement_id": "C1",
      "status": "passed",
      "observed_evidence_ids": ["SQL:001"],
      "reason": "The SQL aggregates the required measure."
    },
    {
      "check_id": "CHECK:002",
      "requirement_source": "success_criteria",
      "requirement_id": "C2",
      "status": "passed",
      "observed_evidence_ids": ["SQL:002"],
      "reason": "The SQL uses the requested grouping."
    }
  ],
  "coverage": {
    "assigned": 2,
    "evaluated": 2,
    "unresolved": 0
  },
  "verification_method": "static_sql_analysis",
  "rationale": "All assigned semantic requirements passed."
}
```

Adapt to existing Pydantic models and scoring conventions.

#### Evidence integrity

Ensure:

- Every evidence ID resolves.
- Every mapped requirement has a decision.
- Every applied guidance ID resolves.
- Coverage counts are correct.
- No unsupported evidence is invented.
- Unresolved requirements are visible.
- Scores follow existing policy.
- Full verified credit does not conceal failed or unresolved mandatory checks.

### 19. Final verdict

Preserve the existing deterministic verdict policy:

- PASS.
- FAIL.
- INCONCLUSIVE.

Record:

- Individual metric scores.
- Overall verdict.
- Applied scoring policy.
- Failed requirements.
- Unresolved evidence.
- Applied exceptions.
- Verification limitations.

Do not change aggregate scoring behavior without first examining the existing implementation.

---

## PART 11 — RUN.LOG TRACEABILITY

### 20. Improve existing run.log

The run log must make the evaluation lifecycle reconstructable.

For every question, log:

#### A. Evaluation input

- Run ID.
- Question ID.
- Scenario ID.
- Selected metrics.
- Success Criteria.
- Reference Spec summary.
- Metric Mapping.
- Relevant Evaluation Guidance IDs.

#### B. Evidence preparation

For NL2SQL:

- SQL artifact ID.
- SQLGlot parse status.
- Parsed SQL structure.
- Catalog reconciliation.
- Invalid/unresolved identifiers.
- Parser warnings.

For Agent:

- Tool calls.
- Tool names and arguments.
- Available results.
- Missing trace evidence.

#### C. General Judge invocation

- Judge invocation ID.
- Question ID.
- Assigned requirements per metric.
- Guidance supplied.
- Evidence supplied.
- Prompt/template version or fingerprint.
- Attempt number.

#### D. Optional evidence lookups

- Lookup requested or not requested.
- Lookup operation and arguments.
- Returned evidence.
- Lookup result status.
- Evidence used in final judgment.

#### E. Metric decisions

- Metric.
- Score.
- Comparison basis.
- Criteria evaluated.
- Rules evaluated.
- Guidance applied.
- Check-level results.
- Evidence IDs.
- Coverage.
- Rationale.

#### F. Final verdict

- Metric scores.
- Overall verdict.
- Failed requirements.
- Unresolved evidence.
- Applied exceptions.
- Verification limitations.

Preserve existing tracing conventions.

Do not duplicate large raw payloads unnecessarily.

Do not log secrets, credentials, sensitive raw data, or internal authentication information.

---

## PART 12 — FINAL REPORT

### 21. Make the report explain every score

The final report must communicate the same evaluation decisions as run.log.

#### Question-level summary

Show:

- Question ID.
- Scenario ID.
- User utterance.
- Product outcome.
- Selected metrics.
- Success Criteria.
- Reference Spec summary.
- Overall verdict.

#### Metric-level details

For every metric, show:

- Metric name.
- Score.
- Comparison basis.
- Success Criteria evaluated.
- Reference requirements evaluated.
- Evaluation Guidance applied.
- Check-level pass/fail/not-applicable/unresolved decisions.
- Observed evidence.
- Explanation.
- Coverage.
- Verification method.

#### Rules Adherence details

Show:

- Rules available in evaluation scope.
- Applicable rules.
- Not-applicable rules.
- Passed rules.
- Failed rules.
- Unresolved rules.
- Rule source.
- Supporting evidence.

#### Evidence provenance

Distinguish:

1. Generated Success Criteria.
2. Scenario-selected Reference Spec.
3. Evaluation Guidance.
4. Original product response.
5. SQLGlot evidence.
6. Catalog reconciliation.
7. Optional evidence lookups.
8. Agent tool-call evidence.

#### Limitations

Explicitly report:

- Static SQL analysis without database execution.
- Incomplete reference evidence.
- Parsing failures.
- Missing tool traces.
- Unresolved rule applicability.
- Unavailable lookup results.

Keep the report concise at the summary level and detailed at the metric/check level.

---

## PART 13 — VALIDATION AND TESTING

### 22. Structural integrity checks

Use lightweight Python checks for Evaluation Context integrity.

Verify:

- Unique Success Criteria IDs.
- Valid metric identifiers.
- Mappings refer to existing criteria.
- Mappings refer to available Reference Spec sections.
- Every configured metric has a mapping.
- Every mapped requirement has an evaluation decision.
- Evidence IDs resolve.
- Guidance IDs resolve.
- Coverage counts match decisions.
- Scores comply with existing scoring policy.

These are contract-integrity checks, not new catalog validation.

Do not add an extra LLM call for these checks.

> **[Not captured in the photos: the rest of Part 13 (section 23 onward) and all of Part 14.]**

---

## PART 15 — NON-NEGOTIABLE CONSTRAINTS

- No new Reference Resolver.
- No new Python catalog-validation stage.
- No additional evaluation-agent loop.
- No unnecessary LLM calls.
- No separate `expected_behavior` field in the new contract.
- No duplicate Reference Pack selection in Question Generation.
- No replacement of SQLGlot.
- No replacement of existing catalog reconciliation.
- No replacement of the General Judge.
- No removal of existing optional evidence lookups.
- No fabricated reference facts.
- No fabricated evidence IDs.
- No silent omission of mapped requirements.
- No hidden Evaluation Guidance affecting scores without traceability.
- No unsupported claims of database execution.
- Preserve Agent and NL2SQL functionality.
- Preserve existing scoring and verdict behavior where defensible.
