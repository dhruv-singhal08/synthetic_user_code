"""Deterministic parsing of custom metric input (free text and/or a workbook).

Free text: one metric per paragraph (blank-line separated); a line starting with a bullet
("-", "*", "•") or a number ("1." / "1)") also starts a new metric. Bullet markers are removed.

Workbook (.xlsx, every sheet; or .csv): the first non-empty row of each sheet is the header;
every later row is one metric whose text is "Header: value" for each non-empty cell, so the
compiler sees which column each value came from. Provenance (file, sheet, row) is kept.

Nothing is interpreted here: no keyword matching, no merging, no duplicate detection. Blank
gaps and malformed rows are reported as explicit input errors, never silently dropped. All
text is untrusted user content; the compiler prompt treats it as data.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from synthetic_user.core.spreadsheet import SheetRow, read_rows
from synthetic_user.models.metrics import CustomMetricInput, CustomMetricInputError

_ITEM_START = re.compile(r"^\s*(?:[-*•]|\d{1,3}[.)])\s+")


@dataclass(frozen=True)
class CustomInputLimits:
    max_metrics: int
    max_chars_per_metric: int
    max_rows: int
    max_uncompressed_bytes: int


@dataclass
class ParsedCustomInput:
    inputs: list[CustomMetricInput] = field(default_factory=list)
    errors: list[CustomMetricInputError] = field(default_factory=list)


def parse_free_text(text: str, limits: CustomInputLimits) -> ParsedCustomInput:
    items: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            if current:
                items.append(current)
                current = []
            continue
        if _ITEM_START.match(line) and current:
            items.append(current)
            current = []
        current.append(_ITEM_START.sub("", line, count=1).strip())
    if current:
        items.append(current)
    parsed = ParsedCustomInput()
    for index, lines in enumerate(items, start=1):
        _add(parsed, " ".join(lines), f"free text item {index}", limits)
    return parsed


def parse_workbook(
    content: bytes, source_name: str, limits: CustomInputLimits
) -> ParsedCustomInput:
    rows = read_rows(
        content,
        source_name=source_name,
        max_rows=limits.max_rows,
        max_uncompressed_bytes=limits.max_uncompressed_bytes,
    )
    parsed = ParsedCustomInput()
    by_sheet: dict[str, list[SheetRow]] = {}
    for row in rows:
        by_sheet.setdefault(row.sheet, []).append(row)
    if not by_sheet:
        parsed.errors.append(
            CustomMetricInputError(provenance=source_name, reason="the file has no rows")
        )
    for sheet, sheet_rows in by_sheet.items():
        header, *data = sheet_rows
        labels = [cell.strip() for cell in header.cells]
        if not data:
            parsed.errors.append(
                CustomMetricInputError(
                    provenance=f"{source_name}: {header.provenance}",
                    reason="sheet has a header row but no metric rows",
                )
            )
            continue
        previous = header.row_number
        for row in data:
            if row.row_number > previous + 1:
                gap = (
                    f"row {previous + 1}"
                    if row.row_number == previous + 2
                    else f"rows {previous + 1}-{row.row_number - 1}"
                )
                where = f"{sheet} {gap}" if sheet else gap
                parsed.errors.append(
                    CustomMetricInputError(
                        provenance=f"{source_name}: {where}",
                        reason="blank row(s) inside the metric table were skipped",
                    )
                )
            previous = row.row_number
            _add(parsed, _row_text(row, labels), f"{source_name}: {row.provenance}", limits)
    return parsed


def merge(parts: Sequence[ParsedCustomInput], limits: CustomInputLimits) -> ParsedCustomInput:
    """Concatenate parsed inputs (text first, then workbook rows) and apply the count bound."""
    merged = ParsedCustomInput()
    for part in parts:
        merged.inputs.extend(part.inputs)
        merged.errors.extend(part.errors)
    if len(merged.inputs) > limits.max_metrics:
        for extra in merged.inputs[limits.max_metrics :]:
            merged.errors.append(
                CustomMetricInputError(
                    provenance=extra.provenance,
                    reason=f"more than {limits.max_metrics} custom metrics; not compiled",
                )
            )
        merged.inputs = merged.inputs[: limits.max_metrics]
    return merged


def _row_text(row: SheetRow, labels: Sequence[str]) -> str:
    parts: list[str] = []
    for index, cell in enumerate(row.cells):
        value = " ".join(cell.split())
        if not value:
            continue
        label = labels[index] if index < len(labels) and labels[index] else _column(index)
        parts.append(f"{label}: {value}")
    return "; ".join(parts)


def _column(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        letters = chr(ord("A") + rest) + letters
    return f"Column {letters}"


def _add(parsed: ParsedCustomInput, text: str, provenance: str, limits: CustomInputLimits) -> None:
    text = text.strip()
    if not text:
        parsed.errors.append(CustomMetricInputError(provenance=provenance, reason="empty metric"))
    elif len(text) > limits.max_chars_per_metric:
        parsed.errors.append(
            CustomMetricInputError(
                provenance=provenance,
                reason=f"longer than {limits.max_chars_per_metric:,} characters; not compiled",
            )
        )
    else:
        parsed.inputs.append(CustomMetricInput(text=text, provenance=provenance))
