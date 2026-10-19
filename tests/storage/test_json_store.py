from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from synthetic_user.core.exceptions import ArtifactIntegrityError, ArtifactNotFoundError
from synthetic_user.models.execution import (
    AnsweredResponse,
    ExecutionRecord,
    TimeoutResponse,
)
from synthetic_user.models.reference import PackTable, ReferenceKind, ReferencePack
from synthetic_user.storage import json_store
from synthetic_user.storage.json_store import JsonArtifactStore


def _index(index_id: str = "IDX_001") -> ReferencePack:
    """A sample model (a Reference Pack); ``index_id`` goes into reference_id."""
    return ReferencePack(
        pack_id="RP_SCN_001",
        scenario_id="SCN_001",
        reference_id=index_id,
        kind=ReferenceKind.DATA_CATALOG,
        tables=[PackTable(name="card_transactions", why="spend")],
    )


def _files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_model_round_trip(tmp_path: Path) -> None:
    store = JsonArtifactStore(tmp_path)
    path = store.write_model("runs/r1/index.json", _index())
    assert path == tmp_path / "runs" / "r1" / "index.json"
    assert store.read_model("runs/r1/index.json", ReferencePack) == _index()


def test_models_round_trip(tmp_path: Path) -> None:
    store = JsonArtifactStore(tmp_path)
    items = [_index("IDX_001"), _index("IDX_002")]
    store.write_models("many.json", items, ReferencePack)
    assert store.read_models("many.json", ReferencePack) == items


def test_union_round_trip_via_adapter(tmp_path: Path) -> None:
    store = JsonArtifactStore(tmp_path)
    common = {
        "question_id": "Q_001",
        "product_id": "p",
        "adapter": "fixture",
        "submitted_question": "q",
        "started_at": datetime(2026, 10, 13, tzinfo=UTC),
    }
    records = [
        ExecutionRecord.model_validate(
            {
                "execution_id": "EXE_001",
                "response": {
                    "outcome": "answered",
                    "artifacts": [{"kind": "sql", "content": "SELECT 1"}],
                },
                **common,
            }
        ),
        ExecutionRecord.model_validate(
            {"execution_id": "EXE_002", "response": {"outcome": "timeout"}, **common}
        ),
    ]
    adapter: TypeAdapter[list[ExecutionRecord]] = TypeAdapter(list[ExecutionRecord])
    store.write_with("executions.json", records, adapter)
    loaded = store.read_with("executions.json", adapter)
    assert [type(r.response) for r in loaded] == [AnsweredResponse, TimeoutResponse]
    assert loaded == records


def test_overwrite_replaces_content(tmp_path: Path) -> None:
    store = JsonArtifactStore(tmp_path)
    store.write_model("a.json", _index("IDX_001"))
    store.write_model("a.json", _index("IDX_002"))
    assert store.read_model("a.json", ReferencePack).reference_id == "IDX_002"


def test_no_temp_files_left(tmp_path: Path) -> None:
    store = JsonArtifactStore(tmp_path)
    store.write_model("d/a.json", _index())
    store.write_models("d/b.json", [_index()], ReferencePack)
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
    assert store.read_model("a.json", ReferencePack).reference_id == "IDX_001"


def test_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(ArtifactNotFoundError):
        JsonArtifactStore(tmp_path).read_model("nope.json", ReferencePack)


@pytest.mark.parametrize(
    "content",
    [b"{not json", b'{"index_id": "IDX_001"}', b"[]", b""],
)
def test_corrupt_file_is_integrity_error(tmp_path: Path, content: bytes) -> None:
    (tmp_path / "bad.json").write_bytes(content)
    with pytest.raises(ArtifactIntegrityError, match="invalid"):
        JsonArtifactStore(tmp_path).read_model("bad.json", ReferencePack)


@pytest.mark.parametrize("bad", ["../escape.json", "a/../../escape.json"])
def test_paths_cannot_escape_store(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="inside the store"):
        JsonArtifactStore(tmp_path).path_for(bad)


def test_absolute_path_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside the store"):
        JsonArtifactStore(tmp_path).path_for(tmp_path / "x.json")
