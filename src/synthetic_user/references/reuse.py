"""Reuse a stored reference index for the same catalog id + version unless forced."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from synthetic_user.core.exceptions import ReferenceIntegrityError
from synthetic_user.models.reference import ReferenceCatalog, ReferenceIndex
from synthetic_user.references.integrity import check_index_against_catalog
from synthetic_user.storage.json_store import JsonArtifactStore


@dataclass(frozen=True)
class IndexResult:
    index: ReferenceIndex
    path: Path
    reused: bool


async def load_or_build_reference_index(
    catalog: ReferenceCatalog,
    store: JsonArtifactStore,
    build: Callable[[ReferenceCatalog], Awaitable[ReferenceIndex]],
    *,
    force: bool = False,
) -> IndexResult:
    """Return the newest stored index for this catalog version, or build and store one.

    A reused index is re-checked against the catalog; a failing one raises
    ReferenceIntegrityError rather than being used or silently rebuilt.
    """
    existing = store.list_reference_indexes(catalog.catalog_id, catalog.catalog_version)
    if existing and not force:
        index_id = existing[-1]
        index = store.read_reference_index(catalog.catalog_id, catalog.catalog_version, index_id)
        errors = check_index_against_catalog(index, catalog)
        if errors:
            raise ReferenceIntegrityError(
                f"Stored reference index {index_id} no longer matches the catalog "
                f"(re-run with --force to rebuild): {'; '.join(errors)}"
            )
        path = store.reference_index_path(catalog.catalog_id, catalog.catalog_version, index_id)
        return IndexResult(index=index, path=path, reused=True)

    index = await build(catalog)
    return IndexResult(index=index, path=store.write_reference_index(index), reused=False)
