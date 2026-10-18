# Usage guide: sample card spend data catalog

## What this reference is
A data catalog (JSON) of the corporate card spend tables an NL2SQL product queries.

## How to read it
- Each entry in `tables` is a table: `name`, `business_purpose`, `grain`, `columns` (with
  descriptions and types), `metrics`, `dimensions` and `time_columns`.
- `relationships` lists joins between tables; `verification` says whether a join is confirmed
  ("verified") or not ("unknown").
- `global_semantic_rules` and `uncertainties` apply to every question.

## What is authoritative
- Table and column names, descriptions, metric definitions and relationship verification.
- `global_semantic_rules` are hard rules (e.g. never sum amounts across currencies).

## What counts as core
The tables and columns a correct query must use: the metric columns, the dimensions the
question groups or filters by, and the join keys between them. Name them as `table` or
`table.column`.

## Evaluation guidance
- The product answers with SQL. Judge whether the SQL answers the question with the core
  tables and columns; extra columns, filters, joins or ordering are fine unless they change
  the answer.
- A join marked "unknown" must not be presented as reliable.
- Amounts in different currencies must stay separate.

## Ignore
`index_id`, `generation_mode`, `llm_provider`, `source_files`, `integrity_report` (bookkeeping).
