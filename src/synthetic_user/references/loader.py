"""Load the user's reference file (JSON / TXT / MD) and optional usage guide (design v3).

The tool never builds or rewrites a reference: the file is read as text (JSON is parsed only
to re-indent it compactly for prompts) and the guide is passed through as written.

Mode: a reference given (``--reference`` or ``inputs.reference_file``) -> grounded; none ->
ungrounded (logged).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.models.reference import Reference, ReferenceFormat
from synthetic_user.models.run import GenerationMode

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

FORMATS: dict[str, ReferenceFormat] = {
    ".json": ReferenceFormat.JSON,
    ".txt": ReferenceFormat.TEXT,
    ".md": ReferenceFormat.MARKDOWN,
    ".markdown": ReferenceFormat.MARKDOWN,
}
GUIDE_SUFFIXES = frozenset({".txt", ".md", ".markdown"})


def reference_format(name: str) -> ReferenceFormat:
    fmt = FORMATS.get(Path(name).suffix.lower())
    if fmt is None:
        raise InputAssetError(
            f"Reference {name}: unsupported file type; use one of {sorted(FORMATS)}"
        )
    return fmt


def parse_reference(
    content: bytes | str,
    *,
    source_name: str,
    max_chars: int,
    guide: bytes | str | None = None,
    guide_name: str | None = None,
) -> Reference:
    """Build a Reference from file content (and an optional guide). InputAssetError on an
    unsupported type, non-UTF-8 text, invalid JSON, an empty file or a reference over
    ``max_chars`` characters (as used in prompts)."""
    fmt = reference_format(source_name)
    raw = _text(content, source_name)
    if fmt is ReferenceFormat.JSON:
        try:
            text = json.dumps(json.loads(raw), indent=1, ensure_ascii=False)
        except ValueError as exc:
            raise InputAssetError(f"Reference {source_name} is not valid JSON: {exc}") from exc
    else:
        text = raw.strip()
    if not text.strip():
        raise InputAssetError(f"Reference {source_name} is empty")
    if len(text) > max_chars:
        raise InputAssetError(
            f"Reference {source_name} has {len(text):,} characters; the limit is "
            f"{max_chars:,} (reference.max_chars). Trim it or split it per product area."
        )
    guide_text: str | None = None
    if guide is not None:
        name = guide_name or "guide"
        if Path(name).suffix.lower() not in GUIDE_SUFFIXES:
            raise InputAssetError(f"Usage guide {name}: use a .md or .txt file")
        guide_text = _text(guide, name).strip() or None
    digest = hashlib.sha256(f"{text}\n\x00\n{guide_text or ''}".encode()).hexdigest()[:8]
    return Reference(
        reference_id=f"{Path(source_name).stem}-{digest}",
        source_name=Path(source_name).name,
        format=fmt,
        text=text,
        guide=guide_text,
        guide_name=Path(guide_name).name if guide_name and guide_text else None,
    )


def load_reference(path: Path, *, max_chars: int, guide_path: Path | None = None) -> Reference:
    guide = _read(guide_path, "Usage guide") if guide_path is not None else None
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
            raise InputAssetError("A usage guide was given without a reference file")
        logger.info("No reference given: mode %s", GenerationMode.UNGROUNDED)
        return None
    reference = load_reference(path, max_chars=max_chars, guide_path=guide_path)
    logger.info(
        "Reference %s (%s, %d chars, guide: %s): mode %s",
        reference.reference_id,
        reference.format,
        len(reference.text),
        reference.guide_name or "default",
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
