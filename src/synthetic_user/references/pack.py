"""Reference Pack (cheat sheet) building: every pick must name something the reference has.

The scenario node keeps the valid picks and rejects (and records) the rest; a pack without a
valid table / document rejects its scenario. A person's edit in the UI is all-or-nothing, so
they see exactly what to fix. The pack holds the user's own text, copied by code.
"""

from __future__ import annotations

from typing import Any

from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.models.reference import (
    CatalogPackDraft,
    DataCatalog,
    DocumentCollection,
    DocumentPackDraft,
    PackDocument,
    PackJoin,
    PackRule,
    PackTable,
    Reference,
    ReferencePack,
    ReferencePackDraft,
    RejectedRefItem,
    pack_id_for,
)

NO_VALID_ITEM = "reference pack names no table / document that is in the reference"


def build_pack(
    draft: ReferencePackDraft, reference: Reference, scenario_id: str
) -> tuple[ReferencePack | None, list[RejectedRefItem]]:
    """Keep the picks the reference has; the rest are rejected with a reason.

    Returns (None, rejected) when no table / document survives or the pack does not fit the
    reference's format."""
    content = reference.content
    fields: dict[str, Any]
    if isinstance(content, DataCatalog) and isinstance(draft, CatalogPackDraft):
        fields, rejected = _catalog_pack(draft, content)
        ok = bool(fields["tables"])
    elif isinstance(content, DocumentCollection) and isinstance(draft, DocumentPackDraft):
        fields, rejected = _document_pack(draft, content)
        ok = bool(fields["documents"])
    else:
        reason = f"reference_pack does not fit a {reference.kind.value} reference"
        return None, [RejectedRefItem(item=draft.model_dump(mode="json"), reason=reason)]
    if not ok:
        return None, rejected
    pack = ReferencePack(
        pack_id=pack_id_for(scenario_id),
        scenario_id=scenario_id,
        reference_id=reference.reference_id,
        kind=reference.kind,
        rejected_items=rejected,
        **fields,
    )
    return pack, rejected


def edit_pack(
    pack: ReferencePack, draft: ReferencePackDraft, reference: Reference
) -> ReferencePack:
    """A person's edit: every pick must exist, else InputAssetError listing every problem
    (nothing is saved)."""
    if pack.reference_id != reference.reference_id:
        raise InputAssetError(
            f"{pack.pack_id} was built from reference {pack.reference_id}, "
            f"not {reference.reference_id}"
        )
    rebuilt, rejected = build_pack(draft, reference, pack.scenario_id)
    if rejected or rebuilt is None:
        problems = [f"{r.reason}" for r in rejected] or [NO_VALID_ITEM]
        raise InputAssetError(f"{pack.pack_id} not saved: " + "; ".join(problems))
    return rebuilt.model_copy(update={"edited": True})


def _catalog_pack(
    draft: CatalogPackDraft, catalog: DataCatalog
) -> tuple[dict[str, Any], list[RejectedRefItem]]:
    rejected: list[RejectedRefItem] = []
    tables: dict[str, PackTable] = {}
    for pick in draft.tables:
        table = catalog.table(pick.table)
        if table is None:
            rejected.append(_rejected(pick, f"unknown table {pick.table!r}"))
            continue
        columns = []
        for name in pick.columns:
            column = table.column(name)
            if column is None:
                item = {"table": table.name, "column": name}
                rejected.append(_rejected(item, f"column {name!r} is not in table {table.name}"))
            else:
                columns.append(column)
        if table.name in tables:  # the same table picked twice: merge its columns
            known = tables[table.name]
            extra = [c for c in columns if c not in known.columns]
            tables[table.name] = known.model_copy(update={"columns": [*known.columns, *extra]})
            continue
        tables[table.name] = PackTable(
            name=table.name,
            description=table.description,
            rules=table.rules,
            columns=columns,
            why=pick.why,
        )
    joins: list[PackJoin] = []
    for join in draft.joins:
        relationship = catalog.relationship(join.from_, join.to)
        if relationship is None:
            reason = f"{join.from_} -> {join.to} is not a listed relationship"
            rejected.append(_rejected(join, reason))
            continue
        joins.append(
            PackJoin(
                **{"from": relationship.from_, "to": relationship.to},
                verified=relationship.verified,
            )
        )
    known_rules = catalog.rule_ids()
    rules: list[PackRule] = []
    for rule_id in dict.fromkeys(r.strip().upper() for r in draft.rules):
        if rule_id in known_rules:
            rules.append(PackRule(id=rule_id, rule=known_rules[rule_id]))
        else:
            rejected.append(_rejected({"rule": rule_id}, f"unknown catalog rule id {rule_id!r}"))
    return {"tables": list(tables.values()), "joins": joins, "rules": rules}, rejected


def _document_pack(
    draft: DocumentPackDraft, collection: DocumentCollection
) -> tuple[dict[str, Any], list[RejectedRefItem]]:
    rejected: list[RejectedRefItem] = []
    documents: dict[str, PackDocument] = {}
    for pick in draft.documents:
        doc = collection.document(pick.name)
        if doc is None:
            rejected.append(_rejected(pick, f"unknown document {pick.name!r}"))
        elif doc.name not in documents:
            documents[doc.name] = PackDocument(name=doc.name, summary=doc.summary, why=pick.why)
    return {"documents": list(documents.values())}, rejected


def _rejected(item: Any, reason: str) -> RejectedRefItem:
    data = item if isinstance(item, dict) else item.model_dump(mode="json")
    return RejectedRefItem(item=data, reason=reason)
