"""Reference data: the authoritative catalog, the LLM-built index, and per-run snapshots.

The catalog is the only source of truth for tables, columns and relationships.
Cross-checking index/snapshot content against the catalog lives in references/integrity.py.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel, find_duplicates


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


class ColumnMetadata(StrictModel):
    name: NonEmptyStr
    data_type: str | None = None
    description: str | None = None
    nullable: bool | None = None
    allowed_values: list[str] | None = None


class TableMetadata(StrictModel):
    name: NonEmptyStr
    description: str | None = None
    grain: str | None = None
    primary_key: list[NonEmptyStr] | None = None
    columns: list[ColumnMetadata] = Field(min_length=1)

    @model_validator(mode="after")
    def _columns_consistent(self) -> Self:
        names = [c.name for c in self.columns]
        dupes = find_duplicates(names)
        if dupes:
            raise ValueError(f"table {self.name!r}: duplicate column names {dupes}")
        missing = [k for k in self.primary_key or [] if k not in names]
        if missing:
            raise ValueError(f"table {self.name!r}: primary_key columns not in table {missing}")
        return self

    def has_column(self, column: str) -> bool:
        return any(c.name == column for c in self.columns)


class Relationship(StrictModel):
    source_table: NonEmptyStr
    source_column: NonEmptyStr
    target_table: NonEmptyStr
    target_column: NonEmptyStr
    cardinality: Cardinality = Cardinality.UNKNOWN
    verification: VerificationStatus = VerificationStatus.UNKNOWN


class ReferenceCatalog(StrictModel):
    catalog_id: NonEmptyStr
    catalog_version: NonEmptyStr
    tables: list[TableMetadata] = Field(min_length=1)
    relationships: list[Relationship] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_table_names(self) -> Self:
        dupes = find_duplicates([t.name for t in self.tables])
        if dupes:
            raise ValueError(f"catalog {self.catalog_id!r}: duplicate table names {dupes}")
        return self

    def get_table(self, name: str) -> TableMetadata | None:
        return next((t for t in self.tables if t.name == name), None)

    def has_column(self, table: str, column: str) -> bool:
        found = self.get_table(table)
        return found is not None and found.has_column(column)


class IndexMetric(StrictModel):
    name: NonEmptyStr
    description: str | None = None
    columns: list[NonEmptyStr] = Field(default_factory=list)


class IndexRelationship(StrictModel):
    source_column: NonEmptyStr
    target_table: NonEmptyStr
    target_column: NonEmptyStr
    cardinality: Cardinality = Cardinality.UNKNOWN
    verification: VerificationStatus = VerificationStatus.UNKNOWN


class IndexedTable(StrictModel):
    name: NonEmptyStr
    business_purpose: NonEmptyStr
    grain: str | None = None
    key_entities: list[NonEmptyStr] = Field(default_factory=list)
    important_columns: list[NonEmptyStr] = Field(default_factory=list)
    metrics: list[IndexMetric] = Field(default_factory=list)
    dimensions: list[NonEmptyStr] = Field(default_factory=list)
    time_columns: list[NonEmptyStr] = Field(default_factory=list)
    relationships: list[IndexRelationship] = Field(default_factory=list)


class ReferenceIndex(StrictModel):
    index_id: NonEmptyStr
    index_version: NonEmptyStr
    source_catalog_id: NonEmptyStr
    source_catalog_version: NonEmptyStr
    tables: list[IndexedTable] = Field(default_factory=list)
    generated_at: datetime
    llm_provider: NonEmptyStr
    llm_model: NonEmptyStr
    uncertainties: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_table_names(self) -> Self:
        dupes = find_duplicates([t.name for t in self.tables])
        if dupes:
            raise ValueError(f"index {self.index_id!r}: duplicate table names {dupes}")
        return self

    def get_table(self, name: str) -> IndexedTable | None:
        return next((t for t in self.tables if t.name == name), None)


class ReferenceSnapshot(StrictModel):
    snapshot_id: NonEmptyStr
    source_catalog_id: NonEmptyStr
    source_catalog_version: NonEmptyStr
    tables: list[TableMetadata] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    missing_references: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_table_names(self) -> Self:
        dupes = find_duplicates([t.name for t in self.tables])
        if dupes:
            raise ValueError(f"snapshot {self.snapshot_id!r}: duplicate table names {dupes}")
        return self


class ReferenceIndexDraft(StrictModel):
    """LLM-facing shape of a reference index: no ids, versions, timestamps or provider info.

    Code adds those when building the ReferenceIndex.
    """

    tables: list[IndexedTable] = Field(default_factory=list)
    uncertainties: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_table_names(self) -> Self:
        dupes = find_duplicates([t.name for t in self.tables])
        if dupes:
            raise ValueError(f"duplicate table names {dupes}")
        return self
