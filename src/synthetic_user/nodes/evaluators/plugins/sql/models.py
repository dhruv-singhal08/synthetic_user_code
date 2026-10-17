"""Models of the SQL plugin: its checks, the LLM (Stage B) draft and the static (Stage A)
analysis. Kept here so the generic evaluation models stay product-agnostic."""

from __future__ import annotations

from pydantic import Field

from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.evaluation import CheckStatus, EvidenceStatus, Severity, Verdict


class SqlChecks(StrictModel):
    """Grounded checks of the SQL plugin (stored in EvaluationResult.grounded_checks)."""

    intent_alignment: CheckStatus
    table_selection: CheckStatus
    column_selection: CheckStatus
    join_correctness: CheckStatus
    filter_correctness: CheckStatus
    aggregation_correctness: CheckStatus
    sql_syntax: CheckStatus
    semantic_considerations: CheckStatus


class FindingDraft(StrictModel):
    category: NonEmptyStr = Field(
        description="the check this finding explains, e.g. join_correctness"
    )
    description: NonEmptyStr
    expected_behavior: NonEmptyStr
    observed_behavior: NonEmptyStr
    sql_evidence: str | None = Field(
        default=None, description="exact fragment copied from the SQL under evaluation"
    )
    severity: Severity
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="uncalibrated; null if not estimated"
    )
    evidence_status: EvidenceStatus
    matches_known_gap: bool = False


class SqlEvaluationDraft(StrictModel):
    checks: SqlChecks
    findings: list[FindingDraft] = Field(default_factory=list)
    cheat_sheet_conflicts: list[NonEmptyStr] = Field(default_factory=list)
    verdict: Verdict


class JoinInfo(StrictModel):
    kind: str  # e.g. "JOIN", "LEFT JOIN"
    table: NonEmptyStr  # joined table (as written)
    condition: str | None = None
    equality_pairs: list[tuple[str, str]] = Field(default_factory=list)


class SqlAnalysis(StrictModel):
    """Stage A: what the SQLGlot AST shows (no regex, SQL never executed)."""

    dialect: NonEmptyStr
    parse_ok: bool
    parse_errors: list[NonEmptyStr] = Field(default_factory=list)
    statement_type: str | None = None
    tables: list[NonEmptyStr] = Field(default_factory=list)
    ctes: list[NonEmptyStr] = Field(default_factory=list)
    aliases: dict[str, str] = Field(default_factory=dict)
    columns: list[NonEmptyStr] = Field(default_factory=list)
    joins: list[JoinInfo] = Field(default_factory=list)
    filters: list[NonEmptyStr] = Field(default_factory=list)
    date_expressions: list[NonEmptyStr] = Field(default_factory=list)
    aggregations: list[NonEmptyStr] = Field(default_factory=list)
    group_by: list[NonEmptyStr] = Field(default_factory=list)
    order_by: list[NonEmptyStr] = Field(default_factory=list)
    limit: str | None = None
    # identifier issues: unknown = column missing on an index table; unverifiable = table not
    # in the reference index at all
    unknown_columns: list[NonEmptyStr] = Field(default_factory=list)
    unverifiable_tables: list[NonEmptyStr] = Field(default_factory=list)
    # join checks vs index relationships (contradictions are also findings)
    join_issues: list[NonEmptyStr] = Field(default_factory=list)
