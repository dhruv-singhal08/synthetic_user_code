from __future__ import annotations

from pathlib import Path

import pytest

from synthetic_user.core.exceptions import ReferenceIntegrityError
from synthetic_user.models.reference import ReferenceCatalog, ReferenceIndex
from synthetic_user.references.reuse import load_or_build_reference_index
from synthetic_user.storage.json_store import JsonArtifactStore


class CountingBuilder:
    def __init__(self, index: ReferenceIndex) -> None:
        self.index = index
        self.calls = 0

    async def __call__(self, catalog: ReferenceCatalog) -> ReferenceIndex:
        self.calls += 1
        return self.index.model_copy(
            update={
                "index_id": f"IDX_20261013_12000{self.calls}_abcd",
                "source_catalog_version": catalog.catalog_version,
            }
        )


async def test_builds_then_reuses(
    tmp_path: Path, sample_catalog: ReferenceCatalog, sample_index: ReferenceIndex
) -> None:
    store, build = JsonArtifactStore(tmp_path), CountingBuilder(sample_index)

    first = await load_or_build_reference_index(sample_catalog, store, build)
    assert (first.reused, build.calls) == (False, 1)
    assert first.path.is_file()

    second = await load_or_build_reference_index(sample_catalog, store, build)
    assert (second.reused, build.calls) == (True, 1)
    assert second.index == first.index
    assert second.path == first.path


async def test_force_rebuilds_and_newest_is_reused(
    tmp_path: Path, sample_catalog: ReferenceCatalog, sample_index: ReferenceIndex
) -> None:
    store, build = JsonArtifactStore(tmp_path), CountingBuilder(sample_index)
    await load_or_build_reference_index(sample_catalog, store, build)
    forced = await load_or_build_reference_index(sample_catalog, store, build, force=True)
    assert (forced.reused, build.calls) == (False, 2)

    again = await load_or_build_reference_index(sample_catalog, store, build)
    assert again.reused
    assert again.index.index_id == forced.index.index_id


async def test_new_catalog_version_is_rebuilt(
    tmp_path: Path, sample_catalog: ReferenceCatalog, sample_index: ReferenceIndex
) -> None:
    store, build = JsonArtifactStore(tmp_path), CountingBuilder(sample_index)
    await load_or_build_reference_index(sample_catalog, store, build)
    v2 = sample_catalog.model_copy(update={"catalog_version": "2.0"})
    result = await load_or_build_reference_index(v2, store, build)
    assert (result.reused, build.calls) == (False, 2)
    assert "sample_card_spend_v2.0" in str(result.path)


async def test_stale_stored_index_is_an_error(
    tmp_path: Path, sample_catalog: ReferenceCatalog, sample_index: ReferenceIndex
) -> None:
    store, build = JsonArtifactStore(tmp_path), CountingBuilder(sample_index)
    await load_or_build_reference_index(sample_catalog, store, build)
    # Same id+version, but the catalog content changed underneath the stored index.
    changed = sample_catalog.model_copy(
        update={"tables": [t for t in sample_catalog.tables if t.name != "merchants"]}
    )
    with pytest.raises(ReferenceIntegrityError, match="--force"):
        await load_or_build_reference_index(changed, store, build)
