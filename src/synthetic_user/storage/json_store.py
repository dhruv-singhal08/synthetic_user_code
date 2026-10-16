"""JSON artifact store with atomic writes and typed reads.

Reusable reference indexes live at
``<base_dir>/reference_indexes/<catalog_id>_v<catalog_version>/<index_id>.json``.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from synthetic_user.core.exceptions import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    format_validation_error,
)
from synthetic_user.models.reference import ReferenceIndex

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=BaseModel)

REFERENCE_INDEXES_DIR = "reference_indexes"
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class JsonArtifactStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    # --- generic -------------------------------------------------------------------------

    def path_for(self, relative: str | Path) -> Path:
        """Resolve ``relative`` under base_dir; refuses absolute paths and ``..`` escapes."""
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"artifact path must be relative and inside the store: {relative}")
        return self.base_dir / rel

    def write_with(self, relative: str | Path, value: T, adapter: TypeAdapter[T]) -> Path:
        path = self.path_for(relative)
        _atomic_write(path, adapter.dump_json(value, indent=2))
        return path

    def read_with(self, relative: str | Path, adapter: TypeAdapter[T]) -> T:
        path = self.path_for(relative)
        if not path.is_file():
            raise ArtifactNotFoundError(f"Artifact not found: {path}")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ArtifactIntegrityError(f"Cannot read artifact {path}: {exc}") from exc
        try:
            return adapter.validate_json(data)
        except ValidationError as exc:
            raise ArtifactIntegrityError(
                f"Artifact {path} is invalid: {format_validation_error(exc)}"
            ) from exc

    def write_model(self, relative: str | Path, model: BaseModel) -> Path:
        return self.write_with(relative, model, TypeAdapter(type(model)))

    def write_models(
        self, relative: str | Path, models: Sequence[ModelT], model_type: type[ModelT]
    ) -> Path:
        return self.write_with(relative, list(models), TypeAdapter(list[model_type]))  # type: ignore[valid-type]

    def read_model(self, relative: str | Path, model_type: type[ModelT]) -> ModelT:
        return self.read_with(relative, TypeAdapter(model_type))

    def read_models(self, relative: str | Path, model_type: type[ModelT]) -> list[ModelT]:
        adapter: TypeAdapter[list[ModelT]] = TypeAdapter(list[model_type])  # type: ignore[valid-type]
        return self.read_with(relative, adapter)

    def write_text(self, relative: str | Path, text: str) -> Path:
        """Atomically write a non-JSON artifact (e.g. results.csv)."""
        path = self.path_for(relative)
        _atomic_write(path, text.encode("utf-8"))
        return path

    def exists(self, relative: str | Path) -> bool:
        return self.path_for(relative).is_file()

    def delete(self, relative: str | Path) -> None:
        """Remove an artifact if present (used to drop stale downstream run artifacts)."""
        self.path_for(relative).unlink(missing_ok=True)

    # --- reusable reference indexes ------------------------------------------------------

    def reference_index_path(self, catalog_id: str, catalog_version: str, index_id: str) -> Path:
        return self.path_for(
            Path(REFERENCE_INDEXES_DIR)
            / _index_dir_name(catalog_id, catalog_version)
            / f"{_safe(index_id, 'index_id')}.json"
        )

    def write_reference_index(self, index: ReferenceIndex) -> Path:
        path = self.reference_index_path(
            index.source_catalog_id, index.source_catalog_version, index.index_id
        )
        return self.write_model(path.relative_to(self.base_dir), index)

    def read_reference_index(
        self, catalog_id: str, catalog_version: str, index_id: str
    ) -> ReferenceIndex:
        path = self.reference_index_path(catalog_id, catalog_version, index_id)
        index = self.read_model(path.relative_to(self.base_dir), ReferenceIndex)
        if (index.source_catalog_id, index.source_catalog_version, index.index_id) != (
            catalog_id,
            catalog_version,
            index_id,
        ):
            raise ArtifactIntegrityError(
                f"Reference index {path} does not match its location "
                f"({index.source_catalog_id} v{index.source_catalog_version} {index.index_id})"
            )
        return index

    def list_reference_indexes(self, catalog_id: str, catalog_version: str) -> list[str]:
        """Index ids stored for this catalog version, sorted."""
        directory = self.path_for(
            Path(REFERENCE_INDEXES_DIR) / _index_dir_name(catalog_id, catalog_version)
        )
        if not directory.is_dir():
            return []
        return sorted(p.stem for p in directory.glob("*.json"))


def _index_dir_name(catalog_id: str, catalog_version: str) -> str:
    return f"{_safe(catalog_id, 'catalog_id')}_v{_safe(catalog_version, 'catalog_version')}"


def _safe(value: str, name: str) -> str:
    if not _SAFE_COMPONENT.fullmatch(value) or ".." in value:
        raise ValueError(f"{name} is not safe for use in a path: {value!r}")
    return value


def _atomic_write(path: Path, data: bytes) -> None:
    """Write via a temp file in the same directory, fsync, then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
