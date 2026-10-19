"""Load the user's reference (design v4, D203): a ``data_catalog`` or ``document_collection``
in YAML or JSON, plus optional usage instructions (.md / .txt, or typed in the UI).

The file must match one of the two formats exactly (``kind`` says which); anything else is
refused with the field that is wrong. The tool never builds or rewrites a reference.

Mode: a reference given (``--reference`` or ``inputs.reference_file``) -> grounded; none ->
ungrounded (logged).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter, ValidationError

from synthetic_user.core.exceptions import InputAssetError, format_validation_error
from synthetic_user.models.reference import Reference, ReferenceContent, ReferenceKind
from synthetic_user.models.run import GenerationMode

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

REFERENCE_SUFFIXES = frozenset({".yaml", ".yml", ".json"})
GUIDE_SUFFIXES = frozenset({".txt", ".md", ".markdown"})
KINDS = ", ".join(k.value for k in ReferenceKind)
_CONTENT: TypeAdapter[Any] = TypeAdapter(ReferenceContent)  # type: ignore[arg-type]


def parse_reference(
    content: bytes | str,
    *,
    source_name: str,
    max_chars: int,
    guide: bytes | str | None = None,
    guide_name: str | None = None,
) -> Reference:
    """Build a Reference from file content (and optional instructions). InputAssetError on an
    unsupported file type, non-UTF-8 text, invalid YAML / JSON, a file that does not match
    either format, or a reference over ``max_chars`` characters (as rendered for prompts)."""
    suffix = Path(source_name).suffix.lower()
    if suffix not in REFERENCE_SUFFIXES:
        allowed = ", ".join(sorted(REFERENCE_SUFFIXES))
        raise InputAssetError(f"Reference {source_name}: use a YAML or JSON file ({allowed})")
    raw = _text(content, source_name)
    try:
        data = json.loads(raw) if suffix == ".json" else yaml.safe_load(raw)
    except (ValueError, yaml.YAMLError) as exc:
        raise InputAssetError(f"Reference {source_name} is not valid {suffix[1:]}: {exc}") from exc
    if not isinstance(data, dict):
        raise InputAssetError(f"Reference {source_name}: expected a mapping with 'kind' ({KINDS})")
    if data.get("kind") not in {k.value for k in ReferenceKind}:
        raise InputAssetError(
            f"Reference {source_name}: 'kind' must be one of {KINDS} "
            f"(got {data.get('kind')!r}); download a template from the UI"
        )
    try:
        parsed = _CONTENT.validate_python(data)
    except ValidationError as exc:
        raise InputAssetError(
            f"Reference {source_name} ({data['kind']}): {format_validation_error(exc)}"
        ) from exc
    instructions: str | None = None
    if guide is not None:
        name = guide_name or "instructions"
        if Path(name).suffix.lower() not in GUIDE_SUFFIXES:
            raise InputAssetError(f"Usage instructions {name}: use a .md or .txt file")
        instructions = _text(guide, name).strip() or None
    canonical = json.dumps(parsed.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(f"{canonical}\n\x00\n{instructions or ''}".encode()).hexdigest()[:8]
    reference = Reference(
        reference_id=f"{Path(source_name).stem}-{digest}",
        source_name=Path(source_name).name,
        content=parsed,
        instructions=instructions,
        instructions_name=Path(guide_name).name if guide_name and instructions else None,
    )
    size = len(reference.render())
    if size > max_chars:
        raise InputAssetError(
            f"Reference {source_name} has {size:,} characters as used in prompts; the limit is "
            f"{max_chars:,} (reference.max_chars). Trim it or split it per product area."
        )
    return reference


def load_reference(path: Path, *, max_chars: int, guide_path: Path | None = None) -> Reference:
    guide = _read(guide_path, "Usage instructions") if guide_path is not None else None
    return parse_reference(
        _read(path, "Reference"),
        source_name=path.name,
        max_chars=max_chars,
        guide=guide,
        guide_name=guide_path.name if guide_path is not None else None,
    )


def load_optional_reference(
    path: Path | None, logger: Logger, *, max_chars: int, guide_path: Path | None = None
) -> Reference | None:
    """grounded when a reference path is given, else ungrounded (logged either way)."""
    if path is None:
        if guide_path is not None:
            raise InputAssetError("Usage instructions were given without a reference file")
        logger.info("No reference given: mode %s", GenerationMode.UNGROUNDED)
        return None
    reference = load_reference(path, max_chars=max_chars, guide_path=guide_path)
    logger.info(
        "Reference %s (%s, %d chars, instructions: %s): mode %s",
        reference.reference_id,
        reference.kind.value,
        len(reference.render()),
        reference.instructions_name or "default",
        GenerationMode.GROUNDED,
    )
    return reference


def _read(path: Path, kind: str) -> bytes:
    if not path.is_file():
        raise InputAssetError(f"{kind} not found: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise InputAssetError(f"Cannot read {kind.lower()} {path}: {exc}") from exc


def _text(content: bytes | str, name: str) -> str:
    if isinstance(content, str):
        return content
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputAssetError(f"{name}: not UTF-8 text") from exc
