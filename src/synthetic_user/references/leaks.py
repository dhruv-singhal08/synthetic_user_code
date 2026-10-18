"""Detect reference identifiers leaking into natural-language questions.

Identifiers are the Reference Pack's core_ref names that look like internal names: a single
token containing ``_`` or ``.`` (e.g. ``card_transactions``, ``employees.full_name``) and
their ``_``-containing parts. Plain words such as ``amount`` or "Travel policy" are ordinary
language and are never treated as leaks.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from synthetic_user.models.reference import ReferencePack

_TOKEN = re.compile(r"^[A-Za-z0-9_.]+$")


def pack_identifiers(pack: ReferencePack) -> list[str]:
    found: set[str] = set()
    for ref in pack.core_refs:
        name = ref.name.strip()
        if not _TOKEN.fullmatch(name) or not ("_" in name or "." in name):
            continue
        found.add(name)
        found.update(part for part in name.split(".") if "_" in part)
    return sorted(found)


def find_leaks(text: str, identifiers: Iterable[str]) -> list[str]:
    """Identifiers found in ``text`` as whole words (case-insensitive), sorted."""
    return sorted(
        ident
        for ident in set(identifiers)
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(ident)}(?![A-Za-z0-9_])", text, re.IGNORECASE)
    )
