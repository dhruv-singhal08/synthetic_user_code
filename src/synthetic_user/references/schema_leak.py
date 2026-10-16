"""Detect catalog identifiers leaking into natural-language questions.

Only identifiers containing an underscore are checked (e.g. ``card_transactions``,
``currency_code``): single words such as ``amount`` are ordinary English and would make
every question a "leak".
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from synthetic_user.models.reference import ReferenceCatalog


def catalog_identifiers(catalog: ReferenceCatalog) -> list[str]:
    names = {t.name for t in catalog.tables} | {c.name for t in catalog.tables for c in t.columns}
    return sorted(n for n in names if "_" in n)


def find_schema_leaks(text: str, identifiers: Iterable[str]) -> list[str]:
    """Identifiers found in ``text`` as whole words (case-insensitive), sorted."""
    return sorted(
        ident
        for ident in set(identifiers)
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(ident)}(?![A-Za-z0-9_])", text, re.IGNORECASE)
    )
