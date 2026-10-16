"""Per-run artifact layout: ``<artifacts_dir>/runs/<run_id>/<name>``."""

from __future__ import annotations

from pathlib import Path

from synthetic_user.core.exceptions import ArtifactNotFoundError
from synthetic_user.core.identifiers import RUN_ID_PATTERN
from synthetic_user.models.run import RunManifest
from synthetic_user.storage.json_store import JsonArtifactStore

RUNS_DIR = "runs"
MANIFEST = "manifest.json"


def run_artifact(run_id: str, name: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"invalid run id: {run_id!r}")
    return Path(RUNS_DIR) / run_id / name


def save_run_manifest(store: JsonArtifactStore, manifest: RunManifest) -> Path:
    """Write the manifest; if one exists, keep its created_at and merge artifacts."""
    relative = run_artifact(manifest.run_id, MANIFEST)
    try:
        existing = store.read_model(relative, RunManifest)
    except ArtifactNotFoundError:
        existing = None
    if existing is not None:
        manifest = manifest.model_copy(
            update={
                "created_at": existing.created_at,
                "artifacts": {**existing.artifacts, **manifest.artifacts},
            }
        )
    return store.write_model(relative, manifest)
