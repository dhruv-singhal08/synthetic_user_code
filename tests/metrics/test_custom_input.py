"""Deterministic custom-metric input parsing: free text, workbooks (all sheets), errors."""

from __future__ import annotations

import io
import zipfile

import pytest

from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.metrics.custom_input import (
    CustomInputLimits,
    merge,
    parse_free_text,
    parse_workbook,
)

LIMITS = CustomInputLimits(
    max_metrics=5, max_chars_per_metric=200, max_rows=50, max_uncompressed_bytes=1_000_000
)


def xlsx(sheets: dict[str, list[list[str]]]) -> bytes:
    """A minimal .xlsx workbook (inline strings), built with the stdlib only."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        sheet_xml = []
        rels = []
        for i, (name, rows) in enumerate(sheets.items(), start=1):
            sheet_xml.append(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>')
            rels.append(
                f'<Relationship Id="rId{i}" Target="worksheets/sheet{i}.xml" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
                'worksheet"/>'
            )
            body = []
            for r, cells in enumerate(rows, start=1):
                cell_xml = "".join(
                    f'<c r="{chr(65 + c)}{r}" t="inlineStr"><is><t>{value}</t></is></c>'
                    for c, value in enumerate(cells)
                    if value
                )
                body.append(f'<row r="{r}">{cell_xml}</row>')
            archive.writestr(
                f"xl/worksheets/sheet{i}.xml",
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                f"<sheetData>{''.join(body)}</sheetData></worksheet>",
            )
        archive.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{''.join(sheet_xml)}</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{''.join(rels)}</Relationships>",
        )
    return buffer.getvalue()


def test_free_text_paragraphs_and_bullets() -> None:
    text = (
        "- Workflow adherence: follows the steps\n- Tone: polite\n\nDialect check:\nuses ANSI SQL"
    )
    parsed = parse_free_text(text, LIMITS)
    assert [(i.text, i.provenance) for i in parsed.inputs] == [
        ("Workflow adherence: follows the steps", "free text item 1"),
        ("Tone: polite", "free text item 2"),
        ("Dialect check: uses ANSI SQL", "free text item 3"),
    ]
    assert not parsed.errors


def test_workbook_reads_every_sheet_with_row_provenance_and_explicit_errors() -> None:
    content = xlsx(
        {
            "Quality": [
                ["Metric", "Definition"],
                ["Tone", "Polite"],
                ["", ""],
                ["Brevity", "Short"],
            ],
            "Grounding": [["Name", "Rule", ""], ["Sources", "Cites", "extra note"]],
            "Empty": [["Metric"]],
        }
    )
    parsed = parse_workbook(content, "metrics.xlsx", LIMITS)
    assert [(i.text, i.provenance) for i in parsed.inputs] == [
        ("Metric: Tone; Definition: Polite", "metrics.xlsx: Quality row 2"),
        ("Metric: Brevity; Definition: Short", "metrics.xlsx: Quality row 4"),
        ("Name: Sources; Rule: Cites; Column C: extra note", "metrics.xlsx: Grounding row 2"),
    ]
    assert [(e.provenance, e.reason) for e in parsed.errors] == [
        ("metrics.xlsx: Quality row 3", "blank row(s) inside the metric table were skipped"),
        ("metrics.xlsx: Empty row 1", "sheet has a header row but no metric rows"),
    ]


def test_mixed_inputs_bounded_and_overlong_rows_reported() -> None:
    text = parse_free_text("Tone: polite\n\nTone: polite", LIMITS)  # similar metrics both kept
    long_row = parse_free_text("x" * 300, LIMITS)
    merged = merge([text, long_row], LIMITS)
    assert [i.text for i in merged.inputs] == ["Tone: polite", "Tone: polite"]
    assert "longer than 200 characters" in merged.errors[0].reason
    many = parse_free_text("\n\n".join(f"m{i}" for i in range(8)), LIMITS)
    bounded = merge([many], LIMITS)
    assert len(bounded.inputs) == 5 and len(bounded.errors) == 3
    assert "more than 5 custom metrics" in bounded.errors[0].reason


@pytest.mark.parametrize(
    ("content", "name", "message"),
    [
        (b"not a zip", "m.xlsx", "not a valid .xlsx"),
        (b"a,b\n", "m.txt", "use an .xlsx or .csv"),
        (b"\xff\xfe", "m.csv", "CSV must be UTF-8"),
    ],
)
def test_malformed_files_are_explicit_errors(content: bytes, name: str, message: str) -> None:
    with pytest.raises(InputAssetError, match=message):
        parse_workbook(content, name, LIMITS)
