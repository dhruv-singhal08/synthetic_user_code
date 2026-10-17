"""Reference data (kind "data_catalog"): the pre-built reference index and per-scenario snapshots.

The reference index is a JSON file supplied by the user and is the ONLY reference source
(tables, columns, types, descriptions, relationships + verification, rules). This tool never
generates it. Field names follow the real file; provenance/bookkeeping keys the tool does not
interpret are kept per level in ``extras`` (references/loader.py decides which), so nothing is
lost. Validation warnings (duplicates, dangling references) live in references/integrity.py.
Used only in grounded mode.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, JsonValue

from synthetic_user.models.base import NonEmptyStr, StrictModel


class ReferenceKind(StrEnum):
    DATA_CATALOG = "data_catalog"


DataCatalogKind = Literal[ReferenceKind.DATA_CATALOG]

Extras = dict[str, JsonValue]


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class Cardinality(StrEnum):
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"
    UNKNOWN = "unknown"


class IndexColumn(StrictModel):
    name: NonEmptyStr
    data_type: str | None = None
    description: str | None = None
    primary_key: bool = False
    pii: bool = False
    categorical: bool = False
    categorical_values_raw: str | None = None
    declared_cardinality: int | None = Field(default=None, ge=0)
    extras: Extras = Field(default_factory=dict)


class IndexMetric(StrictModel):
    name: NonEmptyStr
    description: str | None = None
    aggregation_guidance: str | None = None
    extras: Extras = Field(default_factory=dict)


class IndexDimension(StrictModel):
    name: NonEmptyStr
    description: str | None = None
    allowed_values_raw: str | None = None
    declared_cardinality: int | None = Field(default=None, ge=0)
    extras: Extras = Field(default_factory=dict)


class IndexTimeColumn(StrictModel):
    name: NonEmptyStr
    description: str | None = None
    data_type: str | None = None
    extras: Extras = Field(default_factory=dict)


class IndexRelationship(StrictModel):
    source_table: NonEmptyStr
    source_column: NonEmptyStr
    target_table: NonEmptyStr
    target_column: NonEmptyStr
    cardinality: Cardinality = Cardinality.UNKNOWN
    verification: VerificationStatus = VerificationStatus.UNKNOWN
    extras: Extras = Field(default_factory=dict)

    @property
    def source(self) -> str:
        return f"{self.source_table}.{self.source_column}"

    @property
    def target(self) -> str:
        return f"{self.target_table}.{self.target_column}"


class IndexTable(StrictModel):
    name: NonEmptyStr
    business_purpose: NonEmptyStr
    grain: str | None = None
    key_entities: list[NonEmptyStr] = Field(default_factory=list)
    primary_key_columns: list[NonEmptyStr] = Field(default_factory=list)
    important_columns: list[NonEmptyStr] = Field(default_factory=list)
    metrics: list[IndexMetric] = Field(default_factory=list)
    dimensions: list[IndexDimension] = Field(default_factory=list)
    time_columns: list[IndexTimeColumn] = Field(default_factory=list)
    relationships: list[IndexRelationship] = Field(default_factory=list)
    calculated_fields: list[NonEmptyStr] = Field(default_factory=list)
    business_rules: list[NonEmptyStr] = Field(default_factory=list)
    sensitive_columns: list[NonEmptyStr] = Field(default_factory=list)
    columns: list[IndexColumn] = Field(min_length=1)
    extras: Extras = Field(default_factory=dict)

    def get_column(self, name: str) -> IndexColumn | None:
        return next((c for c in self.columns if c.name == name), None)

    def has_column(self, name: str) -> bool:
        return self.get_column(name) is not None

    def rules(self) -> dict[str, list[str]]:
        """Non-empty business_rules / calculated_fields / sensitive_columns."""
        return {
            key: values
            for key, values in (
                ("business_rules", self.business_rules),
                ("calculated_fields", self.calculated_fields),
                ("sensitive_columns", self.sensitive_columns),
            )
            if values
        }


class IndexUncertainty(StrictModel):
    table: str | None = None
    field: str | None = None
    value: JsonValue = None
    reason: NonEmptyStr

    def text(self) -> str:
        where = ".".join(p for p in (self.table, self.field) if p) or "index"
        value = f" = {self.value!r}" if self.value is not None else ""
        return f"{where}{value}: {self.reason}"


class ReferenceIndex(StrictModel):
    reference_kind: DataCatalogKind = ReferenceKind.DATA_CATALOG
    index_id: NonEmptyStr
    index_version: NonEmptyStr
    source_catalog_id: NonEmptyStr
    source_catalog_version: NonEmptyStr
    generated_at: datetime
    tables: list[IndexTable] = Field(min_length=1)
    relationships: list[IndexRelationship] = Field(default_factory=list)
    global_semantic_rules: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    uncertainties: list[IndexUncertainty] = Field(default_factory=list)
    extras: Extras = Field(default_factory=dict)
    # Set by the loader: sha256[:8] of the file, so an edited file gets a new versioned_id.
    content_fingerprint: str | None = None

    @property
    def versioned_id(self) -> str:
        """index_id plus content fingerprint: recorded in run manifests and scenarios."""
        if self.content_fingerprint is None:
            return self.index_id
        return f"{self.index_id}-{self.content_fingerprint}"

    # --- lookup helpers -----------------------------------------------------------------

    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def get_table(self, name: str) -> IndexTable | None:
        return next((t for t in self.tables if t.name == name), None)

    def columns(self, table: str) -> list[IndexColumn]:
        found = self.get_table(table)
        return list(found.columns) if found else []

    def get_column(self, table: str, column: str) -> IndexColumn | None:
        found = self.get_table(table)
        return found.get_column(column) if found else None

    def has_column(self, table: str, column: str) -> bool:
        return self.get_column(table, column) is not None

    def relationships_of(self, table: str) -> list[IndexRelationship]:
        """Top-level relationships with ``table`` at either end."""
        return [r for r in self.relationships if table in (r.source_table, r.target_table)]

    def verification(self, left: str, right: str) -> VerificationStatus | None:
        """Verification of the relationship joining "t.c" ``left`` and ``right`` (either
        direction), or None when the index has no such relationship."""
        pair = {left, right}
        return next(
            (r.verification for r in self.relationships if {r.source, r.target} == pair), None
        )

    def rules(self, table: str) -> dict[str, list[str]]:
        found = self.get_table(table)
        return found.rules() if found else {}


class ReferenceSnapshot(StrictModel):
    """The slice of the index one scenario's cheat sheet points at (built by code)."""

    reference_kind: DataCatalogKind = ReferenceKind.DATA_CATALOG
    snapshot_id: NonEmptyStr
    source_index_id: NonEmptyStr
    tables: list[IndexTable] = Field(default_factory=list)
    relationships: list[IndexRelationship] = Field(default_factory=list)
    missing_references: list[NonEmptyStr] = Field(default_factory=list)
