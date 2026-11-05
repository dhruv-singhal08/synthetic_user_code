"""Exact reference lookups and artifact-identifier reconciliation (evidence only, never verdicts).

The four read-only lookups answer from the authoritative ``data_catalog`` by EXACT,
case-insensitive name: no similarity, no fuzzy matching, no embeddings. Every fact carries its
provenance (catalog version + path in the catalog), so a judge or a reviewer can trace it.

``ArtifactReconciliation`` compares the base identifiers of one generated artifact with the
question's cheat sheet and the full catalog: ``valid_in_pack`` / ``valid_extra`` / ``invalid``
/ ``unresolved``. A cheat-sheet item the artifact does not use is listed, never a failure by
itself. ``ReferenceCheckRecord`` is the backward-compatible evidence stored on an evaluation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, model_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel


class LookupTool(StrEnum):
    LOOKUP_SCHEMA = "lookup_schema"
    LOOKUP_RULES = "lookup_rules"
    LOOKUP_RELATIONSHIPS = "lookup_relationships"
    LOOKUP_COLUMN_ALTERNATIVES = "lookup_column_alternatives"


class LookupStatus(StrEnum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"


class Provenance(StrictModel):
    catalog_version: NonEmptyStr  # the reference id: file stem + content hash
    path: NonEmptyStr  # e.g. "tables[v_lpc_details].columns[cg_id]"


class ColumnFact(StrictModel):
    table: NonEmptyStr  # canonical catalog name
    column: NonEmptyStr
    type: NonEmptyStr | None = None
    description: NonEmptyStr | None = None
    values: list[NonEmptyStr] = Field(default_factory=list)  # categorical values as written
    table_description: NonEmptyStr | None = None
    provenance: Provenance


class RuleFact(StrictModel):
    rule_id: NonEmptyStr  # "<table>#R<n>" (table rule n) or "R<n>" (catalog-wide rule n)
    scope: Literal["table", "catalog"]
    table: NonEmptyStr | None = None
    text: NonEmptyStr  # the user's original wording, never rewritten
    provenance: Provenance


class RelationshipFact(StrictModel):
    from_endpoint: NonEmptyStr  # "table.column" exactly as documented
    to_endpoint: NonEmptyStr
    verified: bool  # stated by the reference, never inferred
    description: NonEmptyStr | None = None
    provenance: Provenance


class SchemaLookup(StrictModel):
    tool: Literal[LookupTool.LOOKUP_SCHEMA] = LookupTool.LOOKUP_SCHEMA
    table_name: NonEmptyStr  # as asked (may be qualified)
    column_name: NonEmptyStr | None = None
    status: LookupStatus
    reason: NonEmptyStr | None = None  # why not found / ambiguous
    table: NonEmptyStr | None = None  # canonical table name when the table was found
    table_description: NonEmptyStr | None = None
    column: ColumnFact | None = None
    provenance: Provenance | None = None


class RulesLookup(StrictModel):
    tool: Literal[LookupTool.LOOKUP_RULES] = LookupTool.LOOKUP_RULES
    tables: list[NonEmptyStr] = Field(default_factory=list)
    rule_ids: list[NonEmptyStr] = Field(default_factory=list)
    rules: list[RuleFact] = Field(default_factory=list)
    not_found: list[NonEmptyStr] = Field(default_factory=list)
    note: NonEmptyStr = "Retrieved rules are candidates; whether one applies is for the judge."


class RelationshipsLookup(StrictModel):
    tool: Literal[LookupTool.LOOKUP_RELATIONSHIPS] = LookupTool.LOOKUP_RELATIONSHIPS
    tables: list[NonEmptyStr]
    pair: list[NonEmptyStr] | None = None
    relationships: list[RelationshipFact] = Field(default_factory=list)
    not_found: list[NonEmptyStr] = Field(default_factory=list)
    note: NonEmptyStr = (
        "Only documented relationships are listed. Cardinality, uniqueness and the validity of "
        "an undocumented join are not established by the reference; a documented relationship "
        "is not a mandatory join."
    )


class ColumnAlternativesLookup(StrictModel):
    tool: Literal[LookupTool.LOOKUP_COLUMN_ALTERNATIVES] = LookupTool.LOOKUP_COLUMN_ALTERNATIVES
    column_name: NonEmptyStr
    excluded_table: NonEmptyStr | None = None
    alternatives: list[ColumnFact] = Field(default_factory=list)
    note: NonEmptyStr = (
        "Exact column-name matches only; the same name in another table does not prove the "
        "same meaning."
    )


LookupResult = Annotated[
    SchemaLookup | RulesLookup | RelationshipsLookup | ColumnAlternativesLookup,
    Field(discriminator="tool"),
]


class LookupRequest(StrictModel):
    """One read-only lookup requested by the judge (LLM-facing; validated per tool)."""

    tool: LookupTool
    table: NonEmptyStr | None = None
    column: NonEmptyStr | None = None
    tables: list[NonEmptyStr] = Field(default_factory=list)
    rule_ids: list[NonEmptyStr] = Field(default_factory=list)
    pair: list[NonEmptyStr] | None = None
    excluded_table: NonEmptyStr | None = None
    reason: NonEmptyStr

    @model_validator(mode="after")
    def _arguments(self) -> Self:
        tool = self.tool
        if tool is LookupTool.LOOKUP_SCHEMA and self.table is None:
            raise ValueError("lookup_schema needs table (and optionally column)")
        if tool is LookupTool.LOOKUP_RULES and not (self.tables or self.rule_ids):
            raise ValueError("lookup_rules needs tables and/or rule_ids")
        if tool is LookupTool.LOOKUP_RELATIONSHIPS and not (self.tables or self.pair):
            raise ValueError("lookup_relationships needs tables or a pair")
        if self.pair is not None and len(self.pair) != 2:
            raise ValueError("pair names exactly two tables")
        if tool is LookupTool.LOOKUP_COLUMN_ALTERNATIVES and self.column is None:
            raise ValueError("lookup_column_alternatives needs column")
        return self

    def key(self) -> str:
        """Stable cache key (case-insensitive names; the reason is not part of the key)."""
        data = self.model_dump(mode="json", exclude={"reason"})
        return repr(sorted((k, str(v).casefold()) for k, v in data.items()))


class LookupOrigin(StrEnum):
    DETERMINISTIC = "deterministic"  # code looked it up before the judge call
    JUDGE_REQUEST = "judge_request"  # the judge asked for it (one batched round)


class LookupTraceEntry(StrictModel):
    origin: LookupOrigin
    request: LookupRequest
    result: LookupResult
    cached: bool = False  # answered from the per-catalog-version cache
    # Traceability (reforming, Part 9): the evidence id the judge cites (LOOKUP:<n>, in trace
    # order), the round (0 = precomputed by code, 1 = the judge's one requested batch) and
    # whether a final metric decision cited it. Absent in older artifacts.
    evidence_id: NonEmptyStr | None = None
    round: int | None = Field(default=None, ge=0, le=1)
    used_in_judgment: bool | None = None

    @property
    def status(self) -> str:
        """found / not_found / ambiguous (schema) or found / partly_found / not_found."""
        result = self.result
        if isinstance(result, SchemaLookup):
            return result.status.value
        found = bool(
            getattr(result, "rules", None)
            or getattr(result, "relationships", None)
            or getattr(result, "alternatives", None)
        )
        missing = bool(getattr(result, "not_found", None))
        if found and missing:
            return "partly_found"
        return "found" if found else "not_found"


# --- reconciliation of an artifact's identifiers -------------------------------------------


class IdentifierStatus(StrEnum):
    VALID_IN_PACK = "valid_in_pack"  # in the catalog and selected in the cheat sheet
    VALID_EXTRA = "valid_extra"  # in the catalog, not selected in the cheat sheet
    INVALID = "invalid"  # established as absent from the catalog
    UNRESOLVED = "unresolved"  # cannot be established (ambiguous scope, unknown qualifier)


class IdentifierDiagnosis(StrEnum):
    """Why an identifier is not valid, as far as the evidence goes (D253). The tool never
    queries a database: no diagnosis claims a database object exists or does not exist."""

    # A table the catalog does not list (it may still exist in the database, e.g. a physical
    # table behind a catalogued view): a mismatch with the catalog, not proven nonexistent.
    CATALOG_MISMATCH = "catalog_mismatch"
    # Cannot be checked: its table is not in the catalog, it is ambiguous, or its namespace is
    # not permitted.
    UNRESOLVED_IDENTIFIER = "unresolved_identifier"
    # A column absent from the column list of a catalogued table it is read from.
    CONFIRMED_INVALID_IDENTIFIER = "confirmed_invalid_identifier"


class IdentifierCheck(StrictModel):
    identifier: NonEmptyStr  # as written in the artifact (base table / table.column)
    kind: Literal["table", "column"]
    status: IdentifierStatus
    canonical: NonEmptyStr | None = None  # canonical catalog name when valid
    reason: NonEmptyStr
    # Invalid / unresolved identifiers only (None in older artifacts and for valid ones).
    diagnosis: IdentifierDiagnosis | None = None
    # A catalog-mismatch table: the cheat-sheet tables the SQL does not read (the expected
    # catalog tables). Listed, never assumed equivalent to the observed table.
    expected_catalog_tables: list[NonEmptyStr] = Field(default_factory=list)


class JoinStatus(StrEnum):
    DOCUMENTED_VERIFIED = "documented_verified"
    DOCUMENTED_UNVERIFIED = "documented_unverified"
    UNDOCUMENTED = "undocumented"  # no listed relationship between these tables is used
    # An extra equality between two tables the SQL already joins over a listed relationship
    # (a composite / additional join condition, not a new relationship).
    EXTRA_CONDITION = "extra_condition"
    UNRESOLVED = "unresolved"  # an endpoint could not be resolved to a catalog column


class JoinCheck(StrictModel):
    left: NonEmptyStr
    right: NonEmptyStr
    status: JoinStatus
    provenance: Provenance | None = None


class ArtifactReconciliation(StrictModel):
    artifact: int = Field(ge=1)  # 1-based position among the response's checked artifacts
    parsed: bool
    diagnostics: list[NonEmptyStr] = Field(default_factory=list)
    identifiers: list[IdentifierCheck] = Field(default_factory=list)
    extra_tables: list[NonEmptyStr] = Field(default_factory=list)
    pack_items_not_used: list[NonEmptyStr] = Field(default_factory=list)
    joins: list[JoinCheck] = Field(default_factory=list)
    # Rules of tables the artifact reads that the cheat sheet did not carry, plus catalog-wide
    # rules the cheat sheet did not select: candidates for the judge, never auto-applied.
    rules_not_in_pack: list[NonEmptyStr] = Field(default_factory=list)

    def with_status(self, status: IdentifierStatus) -> list[IdentifierCheck]:
        return [i for i in self.identifiers if i.status is status]


class ReferenceCheckRecord(StrictModel):
    """Evidence of the reference reconciliation and lookups behind one evaluation."""

    catalog_version: NonEmptyStr
    artifacts: list[ArtifactReconciliation] = Field(default_factory=list)
    lookup_trace: list[LookupTraceEntry] = Field(default_factory=list)
    judge_lookup_requests: int = Field(default=0, ge=0)
    general_judge_calls: int = Field(default=0, ge=0)
    # The judge's decisions on undocumented joins (D238), as returned: join, allowed, basis.
    join_decisions: list[dict[NonEmptyStr, JsonValue]] = Field(default_factory=list)
    # True when the judge requested its one optional batch of lookups (None: older artifact).
    additional_lookup_performed: bool | None = None
