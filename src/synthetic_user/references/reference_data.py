"""The ``reference_data`` input shape: a data catalog written with mappings (user format).

::

    kind: reference_data
    name: ...
    description: ...
    tables:
      <table>:
        description: ...            # or null
        rules: [<free text>, ...]
        columns:
          <column>:
            type: ...               # or null
            description: ...        # or null
            categorical_values: [...]
    relationships: [{from: table.column, to: table.column, verified: bool, description}]
    global_rules: [<free text>, ...]

It is mapped one-to-one onto ``data_catalog`` (tables in file order, ``categorical_values`` ->
``values``, ``global_rules`` -> catalog rules R1, R2, ...). Nothing is inferred, merged or
rewritten; an unexpected field or value type is refused with its path. Booleans in a value
list are refused because YAML reads unquoted yes / no / on / off as booleans: quote them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.reference import CatalogRelationship, ReferenceKind

REFERENCE_DATA_KIND = "reference_data"


class ReferenceDataColumn(StrictModel):
    type: NonEmptyStr | None = None
    description: NonEmptyStr | None = None
    categorical_values: list[NonEmptyStr] = Field(default_factory=list)

    @field_validator("categorical_values", mode="before")
    @classmethod
    def _scalar_values(cls, value: Any) -> Any:
        if value is None:
            return []
        if not isinstance(value, list):
            return value
        out: list[Any] = []
        for item in value:
            if isinstance(item, bool):
                raise ValueError(
                    f"categorical value {item!r} was read as a boolean; quote it in the YAML "
                    "(e.g. 'yes', 'no', 'on', 'off')"
                )
            # Numbers are written as in the file (1 -> "1"); anything else is refused below.
            out.append(str(item) if isinstance(item, int | float) else item)
        return out


class ReferenceDataTable(StrictModel):
    description: NonEmptyStr | None = None
    rules: list[NonEmptyStr] = Field(default_factory=list)
    columns: dict[NonEmptyStr, ReferenceDataColumn] = Field(default_factory=dict)

    @field_validator("rules", mode="before")
    @classmethod
    def _none_is_empty(cls, value: Any) -> Any:
        return [] if value is None else value

    @field_validator("columns", mode="before")
    @classmethod
    def _null_columns(cls, value: Any) -> Any:
        # "column_name:" with no body is a column without type / description.
        if isinstance(value, dict):
            return {name: {} if body is None else body for name, body in value.items()}
        return value


class ReferenceData(StrictModel):
    kind: Literal["reference_data"]
    name: NonEmptyStr
    description: NonEmptyStr | None = None
    tables: dict[NonEmptyStr, ReferenceDataTable] = Field(min_length=1)
    relationships: list[CatalogRelationship] = Field(default_factory=list)
    global_rules: list[NonEmptyStr] = Field(default_factory=list)

    @field_validator("relationships", "global_rules", mode="before")
    @classmethod
    def _none_is_empty(cls, value: Any) -> Any:
        return [] if value is None else value

    @field_validator("tables", mode="before")
    @classmethod
    def _null_tables(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {name: {} if body is None else body for name, body in value.items()}
        return value

    def to_data_catalog(self) -> dict[str, Any]:
        """The same content as a ``data_catalog`` mapping (validated by the caller)."""
        return {
            "kind": ReferenceKind.DATA_CATALOG.value,
            "name": self.name,
            "description": self.description,
            "tables": [
                {
                    "name": table_name,
                    "description": table.description,
                    "rules": list(table.rules),
                    "columns": [
                        {
                            "name": column_name,
                            "type": column.type,
                            "description": column.description,
                            "values": list(column.categorical_values),
                        }
                        for column_name, column in table.columns.items()
                    ],
                }
                for table_name, table in self.tables.items()
            ],
            "relationships": [r.model_dump(mode="json", by_alias=True) for r in self.relationships],
            "rules": list(self.global_rules),
        }
