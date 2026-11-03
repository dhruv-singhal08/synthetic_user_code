"""Read workbook rows (.xlsx, stdlib only) or CSV rows, keeping each row's provenance.

No Excel plugin or extra dependency: an .xlsx file is a zip of XML parts. Every sheet is read;
each returned row records its sheet name and 1-based row number so callers can report errors
against the exact source row. Sizes are bounded before anything is decompressed.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from synthetic_user.core.exceptions import InputAssetError

_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS = {"m": _MAIN, "r": _REL}
_CELL_REF = re.compile(r"[A-Z]+")
SPREADSHEET_SUFFIXES = (".xlsx", ".csv")


@dataclass(frozen=True)
class SheetRow:
    sheet: str
    row_number: int  # 1-based, as shown by a spreadsheet program
    cells: tuple[str, ...]

    @property
    def provenance(self) -> str:
        """``Sheet1 row 4`` for a workbook sheet, ``row 4`` for a CSV file."""
        return f"{self.sheet} row {self.row_number}" if self.sheet else f"row {self.row_number}"


def read_rows(
    content: bytes,
    *,
    source_name: str,
    max_rows: int,
    max_uncompressed_bytes: int,
) -> list[SheetRow]:
    """All non-empty rows of every sheet (.xlsx) or of the file (.csv), in order."""
    suffix = Path(source_name).suffix.casefold()
    if suffix == ".csv":
        rows = _csv_rows(content, source_name)
    elif suffix == ".xlsx":
        rows = _xlsx_rows(content, source_name, max_uncompressed_bytes)
    else:
        raise InputAssetError(f"{source_name}: use an .xlsx or .csv file")
    if len(rows) > max_rows:
        raise InputAssetError(f"{source_name}: more than {max_rows} rows")
    return rows


def _csv_rows(content: bytes, source_name: str) -> list[SheetRow]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputAssetError(f"{source_name}: CSV must be UTF-8") from exc
    rows: list[SheetRow] = []
    for number, cells in enumerate(csv.reader(io.StringIO(text)), start=1):
        if any(cell.strip() for cell in cells):
            rows.append(SheetRow("", number, tuple(cells)))
    return rows


def _xlsx_rows(content: bytes, source_name: str, max_uncompressed: int) -> list[SheetRow]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise InputAssetError(f"{source_name}: not a valid .xlsx workbook") from exc
    try:
        if sum(info.file_size for info in archive.infolist()) > max_uncompressed:
            raise InputAssetError(
                f"{source_name}: workbook is larger than {max_uncompressed:,} bytes uncompressed"
            )
        names = set(archive.namelist())
        shared = _shared_strings(archive) if "xl/sharedStrings.xml" in names else []
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib.get("Id"): item.attrib.get("Target", "") for item in rels}
        rows: list[SheetRow] = []
        for sheet in workbook.findall("m:sheets/m:sheet", _NS):
            name = sheet.attrib.get("name", "Sheet")
            target = targets.get(sheet.attrib.get(f"{{{_REL}}}id"), "").lstrip("/")
            if not target:
                raise InputAssetError(f"{source_name}: sheet {name!r} has no worksheet part")
            path = target if target.startswith("xl/") else f"xl/{target}"
            rows.extend(_sheet_rows(name, ET.fromstring(archive.read(path)), shared))
        return rows
    except KeyError as exc:
        raise InputAssetError(f"{source_name}: incomplete .xlsx workbook ({exc})") from exc
    except ET.ParseError as exc:
        raise InputAssetError(f"{source_name}: unreadable .xlsx XML ({exc})") from exc
    finally:
        archive.close()


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(node.text or "" for node in item.findall(".//m:t", _NS)) for item in root]


def _sheet_rows(sheet: str, root: ET.Element, shared: list[str]) -> list[SheetRow]:
    result: list[SheetRow] = []
    for position, row in enumerate(root.findall("m:sheetData/m:row", _NS), start=1):
        values: dict[int, str] = {}
        for cell in row.findall("m:c", _NS):
            match = _CELL_REF.match(cell.attrib.get("r", "A1"))
            if match is None:
                continue
            kind = cell.attrib.get("t")
            if kind == "inlineStr":
                value = "".join(node.text or "" for node in cell.findall(".//m:t", _NS))
            else:
                node = cell.find("m:v", _NS)
                value = node.text if node is not None and node.text is not None else ""
                if kind == "s" and value:
                    index = int(value)
                    value = shared[index] if 0 <= index < len(shared) else ""
            values[_column_index(match.group())] = value
        if any(v.strip() for v in values.values()):
            number = int(row.attrib.get("r", position))
            result.append(
                SheetRow(sheet, number, tuple(values.get(i, "") for i in range(max(values) + 1)))
            )
    return result


def _column_index(letters: str) -> int:
    value = 0
    for char in letters:
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1
