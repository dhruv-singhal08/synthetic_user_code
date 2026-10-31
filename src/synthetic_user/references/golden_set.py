"""Parse, validate and render optional SME golden sets for test synthesis."""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from xml.etree import ElementTree as ET

from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.models.golden_set import GoldenCase, GoldenSet, GoldenSetPreview
from synthetic_user.models.question import MAX_REFERENCE_CONTEXT, MAX_SUCCESS_CRITERIA, Expectations
from synthetic_user.models.reference import AgentContext, Reference

GOLDEN_SET = "golden_set"

_HEADER_ALIASES = {
    "s no": "number",
    "sno": "number",
    "serial number": "number",
    "scenario type": "scenario_type",
    "question": "question",
    "expected journey": "expected_journey",
    "expected time period": "expected_time_period",
    "expected data source": "expected_data_sources",
    "expected data sources": "expected_data_sources",
    "expected answer response sme input": "expected_behavior",
    "expected answer response": "expected_behavior",
    "expected behavior": "expected_behavior",
    "expected behaviour": "expected_behavior",
    "expected success criteria": "success_criteria",
    "success criteria": "success_criteria",
    "expected tools": "expected_tools",
    "expected context evidence": "expected_context",
    "expected context": "expected_context",
    "prohibited behavior": "prohibited_behaviors",
    "prohibited behaviour": "prohibited_behaviors",
}
_REQUIRED = frozenset({"scenario_type", "question", "expected_behavior"})
_OBSERVED_RESPONSE_HEADERS = frozenset(
    {
        "voc actual response",
        "voc actual reponse",
        "actual response",
        "agent actual response",
        "product actual response",
    }
)


def parse_golden_set(content: bytes, *, source_name: str) -> GoldenSet:
    suffix = Path(source_name).suffix.casefold()
    if suffix == ".csv":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InputAssetError(f"Golden set {source_name}: CSV must be UTF-8") from exc
        rows = list(csv.reader(io.StringIO(text)))
    elif suffix == ".xlsx":
        rows = _xlsx_rows(content, source_name)
    else:
        raise InputAssetError(f"Golden set {source_name}: use an .xlsx or .csv file")
    return _rows_to_golden_set(rows, source_name)


def golden_set_preview(golden_set: GoldenSet) -> GoldenSetPreview:
    used: list[str] = ["question", "expected_behavior", "success_criteria"]
    optional = (
        "expected_tools",
        "expected_journey",
        "expected_time_period",
        "expected_data_sources",
        "expected_context",
        "prohibited_behaviors",
    )
    used.extend(name for name in optional if any(getattr(case, name) for case in golden_set.cases))
    return GoldenSetPreview(
        source_name=golden_set.source_name,
        cases=len(golden_set.cases),
        case_ids=[case.case_id for case in golden_set.cases],
        fields_used=used,
        uploaded_criteria_fields=list(
            dict.fromkeys(field for case in golden_set.cases for field in case.uploaded_criteria)
        ),
        cases_with_expected_tools=sum(bool(case.expected_tools) for case in golden_set.cases),
        cases_missing_expected_tools=[
            case.case_id for case in golden_set.cases if not case.expected_tools
        ],
        cases_with_expected_data_sources=sum(
            bool(case.expected_data_sources) for case in golden_set.cases
        ),
        cases_missing_expected_data_sources=[
            case.case_id for case in golden_set.cases if not case.expected_data_sources
        ],
    )


def validate_agent_contract(reference: Reference | None) -> None:
    """Require the authoritative operational reference needed by an Agent run.

    A golden set is optional inspiration. When one is supplied, ``ground_golden_set`` validates
    every tool/source name it contains against this reference. Generated expectations are always
    validated against the same reference, whether or not examples were uploaded.
    """

    if reference is None:
        raise InputAssetError(
            "Agent runs require Reference data with the canonical tools, data sources, workflow "
            "and rules. A golden set is optional."
        )
    if reference.agent_context is None:
        raise InputAssetError(
            f"Reference {reference.source_name} has no agent_context. Agent runs require "
            "Reference data with explicit tools, data_sources, workflow and rules."
        )


def ground_golden_set(golden_set: GoldenSet, reference: Reference) -> GoldenSet:
    """Cross-check a golden set against the structured Agent contract in the reference.

    The golden set remains authoritative for questions and expected behaviour. The reference is
    authoritative for which tools and data sources actually exist. Uploaded aliases are replaced
    by canonical names and the relevant definitions are copied into ``reference_context`` for the
    evaluator. Unknown names fail the run before any scenarios are created.
    """

    context = reference.agent_context
    if context is None:
        raise InputAssetError(
            f"Reference {reference.source_name} has no agent_context. Golden-set Agent runs "
            "require reference data with explicit tools, data_sources, workflow and rules."
        )

    cases: list[GoldenCase] = []
    problems: list[str] = []
    for case in golden_set.cases:
        tools, tool_errors = _canonical_tools(case.expected_tools, context)
        sources, source_errors = _canonical_sources(case.expected_data_sources, context)
        problems.extend(f"{case.case_id}: {problem}" for problem in (*tool_errors, *source_errors))
        if not tool_errors and not source_errors:
            compatibility = _source_compatibility(tools, sources, context)
            problems.extend(f"{case.case_id}: {problem}" for problem in compatibility)
        cases.append(
            case.model_copy(
                update={
                    "expected_tools": tools,
                    "expected_data_sources": sources,
                    "reference_context": _reference_context(tools, sources, context),
                }
            )
        )
    if problems:
        available_tools = ", ".join(tool.name for tool in context.tools) or "(none)"
        available_sources = ", ".join(source.name for source in context.data_sources)
        raise InputAssetError(
            f"Golden set {golden_set.source_name} does not match reference "
            f"{reference.source_name}: {'; '.join(problems)}. "
            f"Available tools: {available_tools}. Available data sources: {available_sources}."
        )
    return golden_set.model_copy(update={"cases": cases})


def ground_agent_expectations(
    expectations: Expectations,
    reference: Reference,
    *,
    require_tools: bool = True,
    require_sources: bool = True,
) -> Expectations:
    """Canonicalize an LLM-generated Agent answer key against the operational reference."""

    context = reference.agent_context
    if context is None:
        raise InputAssetError(f"Reference {reference.source_name} has no agent_context")
    tools, tool_errors = _canonical_tools(expectations.expected_tools, context)
    sources, source_errors = _canonical_sources(expectations.expected_data_sources, context)
    problems = [*tool_errors, *source_errors]
    if require_tools and context.tools and not tools:
        problems.append("expected_tools must name at least one reference tool")
    if require_sources and not sources:
        problems.append("expected_data_sources must name at least one reference data source")
    if not tool_errors and not source_errors:
        problems.extend(_source_compatibility(tools, sources, context))
    if problems:
        available_tools = ", ".join(tool.name for tool in context.tools) or "(none)"
        available_sources = ", ".join(source.name for source in context.data_sources)
        raise InputAssetError(
            f"Generated Agent expectations do not match Reference data: {'; '.join(problems)}. "
            f"Available tools: {available_tools}. Available data sources: {available_sources}."
        )
    return expectations.model_copy(
        update={
            "expected_tools": tools,
            "expected_data_sources": sources,
            "reference_context": _reference_context(tools, sources, context),
        }
    )


def golden_set_summary(golden_set: GoldenSet) -> dict[str, object]:
    """Compact coverage summary of every uploaded golden row."""

    return {
        "source": golden_set.source_name,
        "case_count": len(golden_set.cases),
        "scenario_types": dict(Counter(case.scenario_type for case in golden_set.cases)),
        "expected_tools": dict(
            Counter(tool for case in golden_set.cases for tool in case.expected_tools)
        ),
        "expected_data_sources": dict(
            Counter(source for case in golden_set.cases for source in case.expected_data_sources)
        ),
        "uploaded_criteria_fields": list(
            dict.fromkeys(field for case in golden_set.cases for field in case.uploaded_criteria)
        ),
    }


def golden_inspiration(
    golden_set: GoldenSet,
    *,
    query: str = "",
    limit: int = 8,
) -> list[dict[str, object]]:
    """Return a bounded, deterministic set of relevant golden examples for prompting."""

    cases = _select_inspiration_cases(golden_set, query=query, limit=limit)
    return [
        {
            "case_id": case.case_id,
            "scenario_type": case.scenario_type,
            "question": case.question,
            "expected_journey": case.expected_journey,
            "expected_time_period": case.expected_time_period,
            "expected_tools": case.expected_tools,
            "expected_data_sources": case.expected_data_sources,
            "expected_behavior": case.expected_behavior,
            "success_criteria": case.success_criteria,
            "expected_context": case.expected_context,
            "prohibited_behaviors": case.prohibited_behaviors,
            "uploaded_criteria": case.uploaded_criteria,
        }
        for case in cases
    ]


def _select_inspiration_cases(golden_set: GoldenSet, *, query: str, limit: int) -> list[GoldenCase]:
    if limit < 1:
        return []
    cases = golden_set.cases
    if len(cases) <= limit:
        return cases
    wanted = _search_tokens(query)
    if wanted:
        ranked = sorted(
            enumerate(cases),
            key=lambda item: (
                -len(wanted & _case_tokens(item[1])),
                -len(wanted & _case_tokens(item[1])) / max(1, len(wanted | _case_tokens(item[1]))),
                item[0],
            ),
        )
        return [case for _, case in ranked[:limit]]
    # Spread scenario-level examples across the workbook instead of overfitting to its first rows.
    indices = [min(len(cases) - 1, int(index * len(cases) / limit)) for index in range(limit)]
    return [cases[index] for index in dict.fromkeys(indices)]


def _case_tokens(case: GoldenCase) -> set[str]:
    return _search_tokens(
        " ".join(
            filter(
                None,
                (
                    case.scenario_type,
                    case.question,
                    case.expected_behavior,
                    case.expected_journey,
                    " ".join(case.expected_tools),
                    " ".join(case.expected_data_sources),
                ),
            )
        )
    )


def _search_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 2 and token not in {"the", "and", "for", "with", "from", "this", "that"}
    }


def _canonical_tools(values: Sequence[str], context: AgentContext) -> tuple[list[str], list[str]]:
    names: list[str] = []
    errors: list[str] = []
    for value in values:
        tool = context.tool(value)
        if tool is None:
            errors.append(f"unknown expected tool {value!r}")
        else:
            names.append(tool.name)
    return names, errors


def _canonical_sources(values: Sequence[str], context: AgentContext) -> tuple[list[str], list[str]]:
    names: list[str] = []
    errors: list[str] = []
    for value in values:
        source = context.data_source(value)
        if source is not None:
            names.append(source.name)
            continue
        group = context.source_group(value)
        if group is not None:
            for member in group.sources:
                canonical = context.data_source(member)
                assert canonical is not None  # AgentContext validates group members.
                names.append(canonical.name)
            continue
        errors.append(f"unknown expected data source {value!r}")
    return list(dict.fromkeys(names)), errors


def _source_compatibility(tools: list[str], sources: list[str], context: AgentContext) -> list[str]:
    if not tools or not sources:
        return []
    allowed: set[str] = set()
    constrained = False
    for name in tools:
        tool = context.tool(name)
        assert tool is not None
        if tool.data_sources:
            constrained = True
            for value in tool.data_sources:
                source = context.data_source(value)
                assert source is not None
                allowed.add(source.name)
    invalid = [source for source in sources if constrained and source not in allowed]
    if not invalid:
        return []
    unique_tools = list(dict.fromkeys(tools))
    return [f"data source(s) {invalid} are not supported by expected tool(s) {unique_tools}"]


def _reference_context(tools: list[str], sources: list[str], context: AgentContext) -> list[str]:
    lines: list[str] = []
    for name in dict.fromkeys(tools):
        tool = context.tool(name)
        assert tool is not None
        allowed = f" Allowed sources: {', '.join(tool.data_sources)}." if tool.data_sources else ""
        lines.append(f"Tool {tool.name}: {tool.description}{allowed}")
        lines.extend(f"Tool {tool.name} constraint: {rule}" for rule in tool.constraints)
    for name in sources:
        source = context.data_source(name)
        assert source is not None
        lines.append(f"Data source {source.name}: {source.description}")
        lines.extend(f"Data source {source.name} limitation: {item}" for item in source.limitations)
    lines.extend(f"Workflow {stage.name}: {stage.description}" for stage in context.workflow)
    lines.extend(f"Reference rule: {rule}" for rule in context.rules)
    return list(dict.fromkeys(lines))[:MAX_REFERENCE_CONTEXT]


def _rows_to_golden_set(rows: list[list[str]], source_name: str) -> GoldenSet:
    header_index = next((i for i, row in enumerate(rows) if any(str(v).strip() for v in row)), None)
    if header_index is None:
        raise InputAssetError(f"Golden set {source_name}: workbook is empty")
    headers: dict[int, str] = {}
    uploaded_headers: dict[int, str] = {}
    for col, value in enumerate(rows[header_index]):
        label = str(value).strip()
        normalized = _normal_header(label)
        if label and normalized not in _OBSERVED_RESPONSE_HEADERS:
            uploaded_headers[col] = label
        if alias := _HEADER_ALIASES.get(normalized):
            headers[col] = alias
    missing = sorted(_REQUIRED - set(headers.values()))
    if missing:
        raise InputAssetError(
            f"Golden set {source_name}: missing required columns {missing}; "
            "include Scenario Type, Question, and Expected Answer / Response (SME input)"
        )
    cases: list[GoldenCase] = []
    seen_questions: set[str] = set()
    for row_no, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
        values = {
            name: str(row[col]).strip() if col < len(row) else "" for col, name in headers.items()
        }
        if not any(values.values()):
            continue
        missing_values = [name for name in _REQUIRED if not values.get(name)]
        if missing_values:
            raise InputAssetError(
                f"Golden set {source_name}, row {row_no}: missing {sorted(missing_values)}"
            )
        key = " ".join(values["question"].casefold().split())
        if key in seen_questions:
            raise InputAssetError(f"Golden set {source_name}, row {row_no}: duplicate question")
        seen_questions.add(key)
        number = values.get("number") or str(len(cases) + 1)
        digits = re.sub(r"\D", "", number)
        case_id = f"GOLD_{int(digits or len(cases) + 1):03d}"
        criteria = _list(values.get("success_criteria", "")) or _criteria_from_behavior(
            values["expected_behavior"]
        )
        context = _list(values.get("expected_context", ""))
        source = values.get("expected_data_sources", "")
        uploaded_criteria = {
            label: value
            for col, label in uploaded_headers.items()
            if (value := (str(row[col]).strip() if col < len(row) else ""))
            and headers.get(col) not in {"number", "question"}
        }
        cases.append(
            GoldenCase(
                case_id=case_id,
                scenario_type=values["scenario_type"],
                question=values["question"],
                expected_journey=values.get("expected_journey") or None,
                expected_time_period=values.get("expected_time_period") or None,
                expected_data_sources=_sources(source),
                expected_behavior=values["expected_behavior"],
                success_criteria=criteria[:MAX_SUCCESS_CRITERIA],
                expected_tools=_list(values.get("expected_tools", "")),
                expected_context=context,
                prohibited_behaviors=_list(values.get("prohibited_behaviors", "")),
                uploaded_criteria=uploaded_criteria,
                is_follow_up=_is_follow_up(values),
            )
        )
    if not cases:
        raise InputAssetError(f"Golden set {source_name}: no data rows found")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise InputAssetError(f"Golden set {source_name}: duplicate S.no values")
    return GoldenSet(source_name=source_name, cases=cases)


def _xlsx_rows(content: bytes, source_name: str) -> list[list[str]]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise InputAssetError(f"Golden set {source_name}: not a valid .xlsx workbook") from exc
    try:
        names = set(archive.namelist())
        shared = _shared_strings(archive) if "xl/sharedStrings.xml" in names else []
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        ns = {
            "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
            "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        }
        sheets = workbook.findall("m:sheets/m:sheet", ns)
        target = next(
            (sheet for sheet in sheets if sheet.attrib.get("name", "").casefold() == "golden set"),
            None,
        )
        if target is None:
            raise InputAssetError(f"Golden set {source_name}: no sheet named 'Golden Set'")
        rel_id = target.attrib[f"{{{ns['r']}}}id"]
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel = next(item for item in rels if item.attrib.get("Id") == rel_id)
        sheet_path = rel.attrib["Target"].lstrip("/")
        if not sheet_path.startswith("xl/"):
            sheet_path = f"xl/{sheet_path}"
        root = ET.fromstring(archive.read(sheet_path))
        return _sheet_rows(root, shared)
    except KeyError as exc:
        raise InputAssetError(f"Golden set {source_name}: incomplete .xlsx workbook") from exc
    finally:
        archive.close()


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    return ["".join(node.text or "" for node in item.findall(".//m:t", ns)) for item in root]


def _sheet_rows(root: ET.Element, shared: list[str]) -> list[list[str]]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    result: list[list[str]] = []
    for row in root.findall("m:sheetData/m:row", ns):
        values: dict[int, str] = {}
        for cell in row.findall("m:c", ns):
            ref = cell.attrib.get("r", "A1")
            match = re.match(r"[A-Z]+", ref)
            if match is None:
                continue
            col = _column_index(match.group())
            kind = cell.attrib.get("t")
            if kind == "inlineStr":
                value = "".join(node.text or "" for node in cell.findall(".//m:t", ns))
            else:
                node = cell.find("m:v", ns)
                value = node.text if node is not None and node.text is not None else ""
                if kind == "s" and value:
                    value = shared[int(value)]
            values[col] = value
        if values:
            result.append([values.get(i, "") for i in range(max(values) + 1)])
    return result


def _column_index(letters: str) -> int:
    value = 0
    for char in letters:
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def _normal_header(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _list(value: str) -> list[str]:
    return [
        item
        for raw in re.split(r"[;\n]+", value)
        if (item := re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", raw).strip())
    ]


def _criteria_from_behavior(value: str) -> list[str]:
    listed = _list(value)
    return listed[:MAX_SUCCESS_CRITERIA] if len(listed) > 1 else [value.strip()]


def _sources(value: str) -> list[str]:
    normalized = value.casefold().strip()
    if normalized == "both":
        return ["Complaints", "CSAT"]
    if (
        "retain context" in normalized
        or "same as prior" in normalized
        or "retain prior" in normalized
    ):
        return [value]
    return [item for raw in re.split(r"[,;/]+", value) if (item := raw.strip())]


def _is_follow_up(values: dict[str, str]) -> bool:
    text = " ".join(values.values()).casefold()
    return "follow-up" in text or "retain prior" in text or "same as prior" in text
