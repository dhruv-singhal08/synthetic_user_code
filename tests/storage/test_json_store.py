from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from synthetic_user.core.exceptions import ArtifactIntegrityError, ArtifactNotFoundError
from synthetic_user.models.execution import (
    ExecutionResult,
    SqlGeneratedResult,
    TimeoutResult,
)
from synthetic_user.models.reference import IndexedTable, ReferenceIndex
from synthetic_user.storage import json_store
from synthetic_user.storage.json_store import JsonArtifactStore


def _index(index_id: str = "IDX_001") -> ReferenceIndex:
    return ReferenceIndex(
        index_id=index_id,
        index_version="1",
        source_catalog_id="sample_card_spend",
        source_catalog_version="1.0",
        tables=[IndexedTable(name="card_transactions", business_purpose="card spend")],
        generated_at=datetime(2026, 10, 13, tzinfo=UTC),
        llm_provider="fake",
        llm_model="fake",
    )


def _files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_model_round_trip(tmp_path: Path) -> None:
    store = JsonArtifactStore(tmp_path)
    path = store.write_model("runs/r1/index.json", _index())
    assert path == tmp_path / "runs" / "r1" / "index.json"
    assert store.read_model("runs/r1/index.json", ReferenceIndex) == _index()


def test_models_round_trip(tmp_path: Path) -> None:
    store = JsonArtifactStore(tmp_path)
    items = [_index("IDX_001"), _index("IDX_002")]
    store.write_models("many.json", items, ReferenceIndex)
    assert store.read_models("many.json", ReferenceIndex) == items


def test_union_round_trip_via_adapter(tmp_path: Path) -> None:
    store = JsonArtifactStore(tmp_path)
    common = {
        "question_id": "Q_001",
        "product_id": "p",
        "submitted_question": "q",
        "started_at": datetime(2026, 10, 13, tzinfo=UTC),
    }
    results: list[ExecutionResult] = [
        SqlGeneratedResult(execution_id="EXE_001", sql="SELECT 1", **common),  # type: ignore[arg-type]
        TimeoutResult(execution_id="EXE_002", **common),  # type: ignore[arg-type]
    ]
    adapter: TypeAdapter[list[ExecutionResult]] = TypeAdapter(list[ExecutionResult])
    store.write_with("executions.json", results, adapter)
    loaded = store.read_with("executions.json", adapter)
    assert [type(r) for r in loaded] == [SqlGeneratedResult, TimeoutResult]


def test_overwrite_replaces_content(tmp_path: Path) -> None:
    store = JsonArtifactStore(tmp_path)
    store.write_model("a.json", _index("IDX_001"))
    store.write_model("a.json", _index("IDX_002"))
    assert store.read_model("a.json", ReferenceIndex).index_id == "IDX_002"


def test_no_temp_files_left(tmp_path: Path) -> None:
    store = JsonArtifactStore(tmp_path)
    store.write_model("d/a.json", _index())
    store.write_models("d/b.json", [_index()], ReferenceIndex)
    assert _files(tmp_path) == [os.path.join("d", "a.json"), os.path.join("d", "b.json")]


def test_failed_write_leaves_no_temp_and_keeps_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonArtifactStore(tmp_path)
    store.write_model("a.json", _index("IDX_001"))

    def boom(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(json_store.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        store.write_model("a.json", _index("IDX_002"))
    monkeypatch.undo()
    assert _files(tmp_path) == ["a.json"]
    assert store.read_model("a.json", ReferenceIndex).index_id == "IDX_001"


def test_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(ArtifactNotFoundError):
        JsonArtifactStore(tmp_path).read_model("nope.json", ReferenceIndex)


@pytest.mark.parametrize(
    "content",
    [b"{not json", b'{"index_id": "IDX_001"}', b"[]", b""],
)
def test_corrupt_file_is_integrity_error(tmp_path: Path, content: bytes) -> None:
    (tmp_path / "bad.json").write_bytes(content)
    with pytest.raises(ArtifactIntegrityError, match="invalid"):
        JsonArtifactStore(tmp_path).read_model("bad.json", ReferenceIndex)


@pytest.mark.parametrize("bad", ["../escape.json", "a/../../escape.json"])
def test_paths_cannot_escape_store(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="inside the store"):
        JsonArtifactStore(tmp_path).path_for(bad)


def test_absolute_path_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside the store"):
        JsonArtifactStore(tmp_path).path_for(tmp_path / "x.json")


# --- reusable reference indexes ----------------------------------------------------------


def test_reference_index_location_and_round_trip(tmp_path: Path) -> None:
    store = JsonArtifactStore(tmp_path)
    path = store.write_reference_index(_index())
    assert path == tmp_path / "reference_indexes" / "sample_card_spend_v1.0" / "IDX_001.json"
    assert store.read_reference_index("sample_card_spend", "1.0", "IDX_001") == _index()


def test_list_reference_indexes(tmp_path: Path) -> None:
    store = JsonArtifactStore(tmp_path)
    assert store.list_reference_indexes("sample_card_spend", "1.0") == []
    store.write_reference_index(_index("IDX_002"))
    store.write_reference_index(_index("IDX_001"))
    assert store.list_reference_indexes("sample_card_spend", "1.0") == ["IDX_001", "IDX_002"]


def test_reference_index_mismatch_detected(tmp_path: Path) -> None:
    store = JsonArtifactStore(tmp_path)
    src = store.write_reference_index(_index("IDX_001"))
    src.rename(src.with_name("IDX_009.json"))
    with pytest.raises(ArtifactIntegrityError, match="does not match"):
        store.read_reference_index("sample_card_spend", "1.0", "IDX_009")


@pytest.mark.parametrize("bad", ["../x", "a/b", "", ".hidden"])
def test_unsafe_index_components_rejected(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="not safe"):
        JsonArtifactStore(tmp_path).reference_index_path(bad, "1.0", "IDX_001")
