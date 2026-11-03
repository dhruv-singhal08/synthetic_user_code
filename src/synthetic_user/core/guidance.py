"""Evaluation guidance input: free text, or a .txt / .md / .yaml file, kept as plain text.

The text is stored exactly as given (trimmed), never rewritten or summarised. A YAML file must
parse (a broken file is refused with the parser's reason) but its original text is kept, so
comments and ordering survive. Empty guidance is "none": evaluation behaves as before.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.models.run import EvaluationGuidance

GUIDANCE_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".yaml", ".yml"})
TYPED_GUIDANCE = "typed in the UI"


def parse_evaluation_guidance(
    content: bytes | str, *, source: str, max_chars: int
) -> EvaluationGuidance | None:
    """None for empty guidance; InputAssetError for a wrong file type, non-UTF-8 text, invalid
    YAML or text longer than ``max_chars`` (evaluation.max_guidance_chars)."""
    suffix = Path(source).suffix.lower()
    if source != TYPED_GUIDANCE and suffix not in GUIDANCE_SUFFIXES:
        allowed = ", ".join(sorted(GUIDANCE_SUFFIXES))
        raise InputAssetError(f"Evaluation guidance {source}: use a text file ({allowed})")
    if isinstance(content, bytes):
        try:
            content = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InputAssetError(f"Evaluation guidance {source}: not UTF-8 text") from exc
    text = content.strip()
    if not text:
        return None
    if len(text) > max_chars:
        raise InputAssetError(
            f"Evaluation guidance {source} has {len(text):,} characters; the limit is "
            f"{max_chars:,} (evaluation.max_guidance_chars)"
        )
    if suffix in {".yaml", ".yml"}:
        try:
            yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise InputAssetError(f"Evaluation guidance {source} is not valid YAML: {exc}") from exc
    return EvaluationGuidance(
        text=text,
        source=Path(source).name if source != TYPED_GUIDANCE else source,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def load_evaluation_guidance(path: Path | None, *, max_chars: int) -> EvaluationGuidance | None:
    if path is None:
        return None
    if not path.is_file():
        raise InputAssetError(f"Evaluation guidance not found: {path}")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise InputAssetError(f"Cannot read evaluation guidance {path}: {exc}") from exc
    return parse_evaluation_guidance(content, source=path.name, max_chars=max_chars)
