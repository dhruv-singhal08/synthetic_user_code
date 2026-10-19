"""Reference data (design v4, D203): two fixed input formats and the per-scenario cheat sheet.

- ``data_catalog``: tables (description, rules, columns), relationships (joins; ``verified``
  is stated by the user, never by the LLM) and catalog-wide rules (ids R1, R2, ... by order).
- ``document_collection``: documents, each a name and a free-text summary.

Each grounded scenario gets a Reference Pack (cheat sheet): the LLM PICKS items by name / id
(tables + columns, joins, rule ids; or documents); code checks every pick exists and copies
the user's own text (descriptions, rules, summaries) into the pack. Unknown picks are rejected
with a reason, never fixed. Question generation and evaluation see the pack, never the whole
reference.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, JsonValue, model_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel, find_duplicates


class ReferenceKind(StrEnum):
    DATA_CATALOG = "data_catalog"
    DOCUMENT_COLLECTION = "document_collection"


# Used when the user gives no usage instructions (one per format).
DEFAULT_INSTRUCTIONS = {
    ReferenceKind.DATA_CATALOG: (
        "This is the data catalog the product answers from.\n"
        "- The tables, columns and relationships listed are the ones that exist; anything not "
        "listed does not exist.\n"
        "- Descriptions say what a table or column means: use them to decide which table and "
        "column answers a question.\n"
        "- Tables can only be combined over the listed relationships. A relationship marked "
        "NOT verified is uncertain: an answer that relies on it must not present the result "
        "as reliable.\n"
        "- Table rules and catalog rules (R1, R2, ...) are binding for every answer they "
        "concern (e.g. currency, date range, default values).\n"
        "- A correct answer uses the tables and columns whose meaning matches the question and "
        "respects the rules. Extra columns, filters or ordering are fine unless they change "
        "the answer or break a rule."
    ),
    ReferenceKind.DOCUMENT_COLLECTION: (
        "This is the document collection the product answers from; each document is given "
        "with a summary of its content.\n"
        "- The product answers from the documents' content and does not cite sources.\n"
        "- What a summary states is true. A statement that contradicts a summary is wrong.\n"
        "- Summaries are partial: a document may contain more than its summary says, so a "
        "statement the summaries do not cover is unverified, not wrong.\n"
        "- A good answer draws on the documents whose subject matches the question and does "
        "not present guesses as facts."
    ),
}

_NAME = re.compile(r"^[^.\s]+$")  # table / column names: no dots, no spaces


def _check_name(name: str, what: str) -> str:
    if not _NAME.fullmatch(name):
        raise ValueError(f"{what} name {name!r} must not contain dots or spaces")
    return name


def _normal(text: str) -> str:
    return " ".join(text.casefold().split())


# --- input format 1: data catalog ---------------------------------------------------------------


class CatalogColumn(StrictModel):
    name: NonEmptyStr
    type: NonEmptyStr | None = None
    description: NonEmptyStr | None = None
    values: list[NonEmptyStr] = Field(default_factory=list)  # allowed values / examples


class CatalogTable(StrictModel):
    name: NonEmptyStr
    description: NonEmptyStr | None = None
    rules: list[NonEmptyStr] = Field(default_factory=list)  # always travel with the table
    columns: list[CatalogColumn] = Field(default_factory=list)

    def column(self, name: str) -> CatalogColumn | None:
        return next((c for c in self.columns if c.name.casefold() == name.casefold()), None)


class Join(StrictModel):
    """A relationship between two columns, written ``table.column``."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    from_: NonEmptyStr = Field(alias="from")
    to: NonEmptyStr


class CatalogRelationship(Join):
    verified: bool = False  # stated by the user; the LLM can never mark a join verified
    description: NonEmptyStr | None = None


class DataCatalog(StrictModel):
    kind: Literal[ReferenceKind.DATA_CATALOG]
    name: NonEmptyStr
    description: NonEmptyStr | None = None
    tables: list[CatalogTable] = Field(min_length=1)
    relationships: list[CatalogRelationship] = Field(default_factory=list)
    rules: list[NonEmptyStr] = Field(default_factory=list)  # catalog-wide; ids R1, R2, ...

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        problems: list[str] = []
        names = [t.name.casefold() for t in self.tables]
        if dupes := find_duplicates(names):
            problems.append(f"duplicate table names: {dupes}")
        for table in self.tables:
            try:
                _check_name(table.name, "table")
                for column in table.columns:
                    _check_name(column.name, f"{table.name} column")
            except ValueError as exc:
                problems.append(str(exc))
            columns = [c.name.casefold() for c in table.columns]
            if dupes := find_duplicates(columns):
                problems.append(f"table {table.name}: duplicate columns {dupes}")
        for i, rel in enumerate(self.relationships, start=1):
            for end in (rel.from_, rel.to):
                if (problem := self.endpoint_problem(end)) is not None:
                    problems.append(f"relationship {i} ({rel.from_} -> {rel.to}): {problem}")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def table(self, name: str) -> CatalogTable | None:
        return next((t for t in self.tables if t.name.casefold() == name.casefold()), None)

    def endpoint_problem(self, endpoint: str) -> str | None:
        table_name, dot, column_name = endpoint.partition(".")
        if not dot or not column_name:
            return f"{endpoint!r} is not written table.column"
        table = self.table(table_name)
        if table is None:
            return f"unknown table {table_name!r}"
        if table.column(column_name) is None:
            return f"column {column_name!r} is not listed in table {table.name}"
        return None

    def relationship(self, a: str, b: str) -> CatalogRelationship | None:
        """The listed relationship between endpoints ``a`` and ``b`` (either direction)."""
        pair = {a.casefold(), b.casefold()}
        return next(
            (r for r in self.relationships if {r.from_.casefold(), r.to.casefold()} == pair),
            None,
        )

    def rule_ids(self) -> dict[str, str]:
        return {f"R{i}": rule for i, rule in enumerate(self.rules, start=1)}


# --- input format 2: document collection --------------------------------------------------------


class CatalogDocument(StrictModel):
    name: NonEmptyStr
    summary: NonEmptyStr  # free text, any length / structure


class DocumentCollection(StrictModel):
    kind: Literal[ReferenceKind.DOCUMENT_COLLECTION]
    name: NonEmptyStr
    description: NonEmptyStr | None = None
    documents: list[CatalogDocument] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> Self:
        if dupes := find_duplicates([_normal(d.name) for d in self.documents]):
            raise ValueError(f"duplicate document names: {dupes}")
        return self

    def document(self, name: str) -> CatalogDocument | None:
        return next((d for d in self.documents if _normal(d.name) == _normal(name)), None)


ReferenceContent = Annotated[DataCatalog | DocumentCollection, Field(discriminator="kind")]


class Reference(StrictModel):
    """The user's reference as the tool uses it."""

    # "<file stem>-<sha256[:8] of content + instructions>": any change gives a new id.
    reference_id: NonEmptyStr
    source_name: NonEmptyStr
    content: ReferenceContent
    instructions: NonEmptyStr | None = None  # uploaded or typed; None -> the default
    instructions_name: NonEmptyStr | None = None

    @property
    def kind(self) -> ReferenceKind:
        return self.content.kind

    def instructions_text(self) -> str:
        return self.instructions or DEFAULT_INSTRUCTIONS[self.kind]

    def render(self) -> str:
        """The whole reference as compact text (scenario prompt, mock product)."""
        content = self.content
        if isinstance(content, DataCatalog):
            return render_catalog(content)
        return render_documents(content)


def _head(kind: str, name: str, description: str | None) -> str:
    return f"{kind}: {name}" + (f" - {description}" if description else "")


def render_column(column: CatalogColumn) -> str:
    text = f"{column.name}" + (f" ({column.type})" if column.type else "")
    if column.description:
        text += f": {column.description}"
    if column.values:
        text += f" [values: {', '.join(column.values)}]"
    return text


def render_join(join: Join, verified: bool) -> str:
    return f"{join.from_} -> {join.to} ({'verified' if verified else 'NOT verified'})"


def render_catalog(catalog: DataCatalog) -> str:
    lines = [_head("Data catalog", catalog.name, catalog.description), "", "Tables:"]
    for table in catalog.tables:
        lines.append(f"- {table.name}: {table.description or '(no description)'}")
        lines.extend(f"    rule: {rule}" for rule in table.rules)
        lines.extend(f"    - {render_column(c)}" for c in table.columns)
    lines += ["", "Relationships (the only valid joins):"]
    lines += [f"- {render_join(r, r.verified)}" for r in catalog.relationships] or ["- (none)"]
    lines += ["", "Catalog rules:"]
    lines += [f"- {rid}: {rule}" for rid, rule in catalog.rule_ids().items()] or ["- (none)"]
    return "\n".join(lines)


def render_documents(collection: DocumentCollection) -> str:
    lines = [_head("Document collection", collection.name, collection.description)]
    for doc in collection.documents:
        lines += ["", f"### {doc.name}", doc.summary.strip()]
    return "\n".join(lines)


# --- cheat sheet: what the LLM picks (also what a person edits in the UI) ------------------------


class TablePick(StrictModel):
    table: NonEmptyStr = Field(description="table name exactly as in the catalog")
    columns: list[NonEmptyStr] = Field(
        default_factory=list, description="column names of that table the answer needs"
    )
    why: NonEmptyStr


class CatalogPackDraft(StrictModel):
    tables: list[TablePick] = Field(min_length=1)
    joins: list[Join] = Field(default_factory=list)
    rules: list[NonEmptyStr] = Field(default_factory=list, description="catalog rule ids")


class DocumentPick(StrictModel):
    name: NonEmptyStr = Field(description="document name exactly as in the collection")
    why: NonEmptyStr


class DocumentPackDraft(StrictModel):
    documents: list[DocumentPick] = Field(min_length=1)


ReferencePackDraft = CatalogPackDraft | DocumentPackDraft


# --- cheat sheet as stored: the picks with the user's text copied in by code --------------------


class PackTable(StrictModel):
    name: NonEmptyStr
    description: NonEmptyStr | None = None
    rules: list[NonEmptyStr] = Field(default_factory=list)
    columns: list[CatalogColumn] = Field(default_factory=list)
    why: NonEmptyStr


class PackJoin(Join):
    verified: bool


class PackRule(StrictModel):
    id: NonEmptyStr
    rule: NonEmptyStr


class PackDocument(StrictModel):
    name: NonEmptyStr
    summary: NonEmptyStr
    why: NonEmptyStr


class RejectedRefItem(StrictModel):
    item: dict[str, JsonValue]
    reason: NonEmptyStr


class ReferencePack(StrictModel):
    """The cheat sheet (authoritative answer key) of one grounded scenario."""

    pack_id: NonEmptyStr = Field(pattern=r"^RP_SCN_\d{3,}$")
    scenario_id: NonEmptyStr = Field(pattern=r"^SCN_\d{3,}$")
    reference_id: NonEmptyStr
    kind: ReferenceKind
    tables: list[PackTable] = Field(default_factory=list)
    joins: list[PackJoin] = Field(default_factory=list)
    rules: list[PackRule] = Field(default_factory=list)
    documents: list[PackDocument] = Field(default_factory=list)
    # Picks that named something the reference does not have.
    rejected_items: list[RejectedRefItem] = Field(default_factory=list)
    edited: bool = False  # True once a person changed it in the UI

    @model_validator(mode="after")
    def _fits_kind(self) -> Self:
        if self.kind is ReferenceKind.DATA_CATALOG:
            if not self.tables or self.documents:
                raise ValueError("a data_catalog pack has tables (at least one), no documents")
        elif not self.documents or self.tables or self.joins or self.rules:
            raise ValueError("a document_collection pack has documents only")
        return self

    def draft(self) -> ReferencePackDraft:
        """The editable picks (what a person changes in the UI)."""
        if self.kind is ReferenceKind.DATA_CATALOG:
            return CatalogPackDraft(
                tables=[
                    TablePick(table=t.name, columns=[c.name for c in t.columns], why=t.why)
                    for t in self.tables
                ],
                joins=[Join(**{"from": j.from_, "to": j.to}) for j in self.joins],
                rules=[r.id for r in self.rules],
            )
        return DocumentPackDraft(
            documents=[DocumentPick(name=d.name, why=d.why) for d in self.documents]
        )

    def item_names(self) -> list[str]:
        """Names of what the pack points at: ``table`` / ``table.column`` or document names."""
        if self.kind is ReferenceKind.DOCUMENT_COLLECTION:
            return [d.name for d in self.documents]
        names: list[str] = []
        for table in self.tables:
            names.append(table.name)
            names.extend(f"{table.name}.{c.name}" for c in table.columns)
        return names

    def render(self) -> str:
        """The cheat sheet as text for the question and evaluation prompts."""
        if self.kind is ReferenceKind.DOCUMENT_COLLECTION:
            lines = ["Documents the answer should draw on:"]
            for doc in self.documents:
                lines += ["", f"### {doc.name}", f"(why: {doc.why})", doc.summary.strip()]
            return "\n".join(lines)
        lines = ["Tables a correct answer must use:"]
        for table in self.tables:
            lines.append(f"- {table.name}: {table.description or '(no description)'}")
            lines.append(f"    why: {table.why}")
            lines.extend(f"    rule: {rule}" for rule in table.rules)
            lines.extend(f"    - {render_column(c)}" for c in table.columns)
        if self.joins:
            lines += ["", "Joins needed (listed relationships):"]
            lines += [f"- {render_join(j, j.verified)}" for j in self.joins]
        if self.rules:
            lines += ["", "Rules the answer must respect:"]
            lines += [f"- {r.id}: {r.rule}" for r in self.rules]
        return "\n".join(lines)


def pack_id_for(scenario_id: str) -> str:
    return f"RP_{scenario_id}"


# Straight, back and typographic quote marks (escaped: the source stays ASCII).
_QUOTE_MARKS = "\"'` \u201c\u201d\u2018\u2019"


def normalise_quote(text: str) -> str:
    """Whitespace-collapsed, case-folded text without surrounding quote marks or end
    punctuation, so a quote copied from wrapped text still matches."""
    collapsed = re.sub(r"\s+", " ", text).strip().casefold()
    return collapsed.strip(_QUOTE_MARKS).rstrip(".,;:").strip()


def quote_in(quote: str, text: str) -> bool:
    needle = normalise_quote(quote)
    return bool(needle) and needle in normalise_quote(text)
