# Usage instructions: sample card spend catalog

- The product answers with SQL over these tables. Judge whether the SQL answers the question
  with the right tables and columns; extra columns, filters, joins or ordering are fine unless
  they change the answer.
- Spend means the sum of `card_transactions.amount`, kept per `currency_code` (never summed
  across currencies).
- `merchants` has no relationship to `card_transactions`: merchant names cannot be joined in.
- The employee-to-department relationship is NOT verified: department totals must be presented
  as uncertain.
