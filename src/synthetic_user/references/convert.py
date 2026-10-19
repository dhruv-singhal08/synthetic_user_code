"""Convert an existing reference index (REFIDX JSON: tables with business_purpose, columns,
business_rules; relationships with verification; global_semantic_rules) into the
``data_catalog`` format (D203). Only what the catalog format holds is kept; bookkeeping
(index ids, generation info, integrity report) is dropped. The result is validated like any
uploaded reference."""

from __future__ import annotations

from typing import Any

from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.models.reference import ReferenceKind

_UNKNOWN = {"", "unknown", "none", "null"}


def refidx_to_catalog(index: dict[str, Any]) -> dict[str, Any]:
    tables = index.get("tables")
    if not isinstance(tables, list) or not tables:
        raise InputAssetError("Not a reference index: no 'tables' list")
    catalog: dict[str, Any] = {
        "kind": ReferenceKind.DATA_CATALOG.value,
        "name": str(index.get("source_catalog_id") or index.get("index_id") or "catalog"),
    }
    version = index.get("source_catalog_version") or index.get("index_version")
    if version:
        catalog["description"] = f"Converted from reference index version {version}"
    catalog["tables"] = [_table(t) for t in tables]
    catalog["relationships"] = [
        {
            "from": f"{r['source_table']}.{r['source_column']}",
            "to": f"{r['target_table']}.{r['target_column']}",
            "verified": str(r.get("verification", "")).lower() == "verified",
        }
        for r in index.get("relationships") or []
    ]
    rules = index.get("global_semantic_rules") or []
    catalog["rules"] = [
        str(v) for v in (rules.values() if isinstance(rules, dict) else rules) if _known(v)
    ]
    return catalog


def _table(table: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"name": table["name"]}
    purpose = table.get("business_purpose")
    grain = table.get("grain")
    parts = [str(purpose).strip().rstrip(".") + "." if _known(purpose) else ""]
    if _known(grain):
        parts.append(f"Grain: {str(grain).strip()}.")
    if description := " ".join(p for p in parts if p):
        out["description"] = description
    if rules := [str(r) for r in table.get("business_rules") or [] if _known(r)]:
        out["rules"] = rules
    out["columns"] = [_column(c) for c in table.get("columns") or []]
    return out


def _column(column: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"name": column["name"]}
    if _known(column.get("data_type")):
        out["type"] = str(column["data_type"])
    if _known(column.get("description")):
        out["description"] = str(column["description"]).strip()
    raw = column.get("categorical_values_raw")
    if _known(raw):
        out["values"] = [v.strip() for v in str(raw).split(",") if v.strip()]
    return out


def _known(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in _UNKNOWN
