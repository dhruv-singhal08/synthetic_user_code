"""Golden sets, the Agent operational context, profile selection and trace tags."""

from __future__ import annotations

import io
import zipfile
from typing import Any

import pytest

from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.evaluation_trace import (
    attach_trace_links,
    merge_finding_tags,
    trace_catalog,
)
from synthetic_user.golden_set import (
    golden_inspiration,
    golden_set_preview,
    ground_agent_expectations,
    ground_golden_set,
    parse_golden_set,
    validate_agent_contract,
)
from synthetic_user.models.evaluation import EvaluationResult
from synthetic_user.models.evaluation_profile import (
    AgentMetric,
    EvaluationProfile,
    EvaluationProfileSelection,
    normalize_agent_metrics,
    resolve_evaluation_profile,
)
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import Expectations, QuestionCase
from synthetic_user.models.reference import AgentContext, Reference, ReferencePack
from synthetic_user.references.loader import parse_reference

AGENT_CONTEXT: dict[str, Any] = {
    "data_sources": [
        {"name": "complaints", "description": "Complaint records", "aliases": ["Complaint data"]},
        {"name": "csat", "description": "Survey results", "aliases": ["Survey data"]},
    ],
    "source_groups": [{"name": "both", "sources": ["complaints", "csat"]}],
    "tools": [
        {
            "name": "aggregate_feedback",
            "description": "Counts and ranks feedback",
            "aliases": ["aggregate"],
            "data_sources": ["complaints", "csat"],
            "constraints": ["Never blend sources"],
        },
        {
            "name": "search_complaints",
            "description": "Finds complaint verbatims",
            "data_sources": ["complaints"],
        },
    ],
    "workflow": [{"name": "planning", "description": "Selects tools"}],
    "rules": ["Keep complaint and CSAT results separate."],
}
CATALOG = (
    "kind: data_catalog\nname: VoC\ntables:\n  - name: complaints_t\n    columns: [{name: theme}]\n"
)
GOLDEN_CSV = (
    "S.No,Scenario Type,Question,Expected Journey,Expected Data Source,"
    "Expected Answer / Response (SME input),Expected Tools,VoC Actual Response,Team Notes\n"
    "1,Ranking,Top complaint themes?,Card replacement,Complaints,"
    "Ranks themes by volume,aggregate,ignored,keep\n"
    "2,Comparison,How do complaints compare with CSAT?,,Both,"
    "1) Compare volumes; 2) Keep sources separate,aggregate_feedback,,\n"
)


@pytest.fixture
def agent_reference() -> Reference:
    import yaml

    text = CATALOG + yaml.safe_dump({"agent_context": AGENT_CONTEXT})
    return parse_reference(text.encode(), source_name="voc.yaml", max_chars=200_000)


# --- golden set parsing --------------------------------------------------------------------


def test_csv_golden_set_parsed_with_uploaded_criteria() -> None:
    golden = parse_golden_set(GOLDEN_CSV.encode(), source_name="gold.csv")
    first, second = golden.cases
    assert (first.case_id, first.expected_tools, first.expected_data_sources) == (
        "GOLD_001",
        ["aggregate"],
        ["Complaints"],
    )
    assert first.success_criteria == ["Ranks themes by volume"]
    # Observed product responses are never treated as SME criteria; unknown columns are kept.
    assert "VoC Actual Response" not in first.uploaded_criteria
    assert first.uploaded_criteria["Team Notes"] == "keep"
    assert second.expected_data_sources == ["Complaints", "CSAT"]  # "Both"
    assert second.success_criteria == ["Compare volumes", "Keep sources separate"]
    preview = golden_set_preview(golden)
    assert preview.cases == 2 and preview.cases_missing_expected_tools == []


@pytest.mark.parametrize(
    ("csv_text", "message"),
    [
        ("Question,Scenario Type\nq,t\n", "missing required columns"),
        (
            "Scenario Type,Question,Expected Behavior\nt,Same?,b\nt,  same? ,b\n",
            "duplicate question",
        ),
        ("Scenario Type,Question,Expected Behavior\nt,q,\n", "missing ['expected_behavior']"),
        ("Scenario Type,Question,Expected Behavior\n", "no data rows"),
    ],
)
def test_invalid_golden_sets_are_refused(csv_text: str, message: str) -> None:
    with pytest.raises(InputAssetError, match=message.replace("[", r"\[").replace("]", r"\]")):
        parse_golden_set(csv_text.encode(), source_name="gold.csv")


def test_unsupported_golden_file_type() -> None:
    with pytest.raises(InputAssetError, match=r"\.xlsx or \.csv"):
        parse_golden_set(b"x", source_name="gold.txt")


def _xlsx(rows: list[list[str]], sheet: str = "Golden Set") -> bytes:
    def cell(ref: str, value: str) -> str:
        return f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>'

    body = "".join(
        f'<row r="{r}">'
        + "".join(cell(f"{chr(65 + c)}{r}", v) for c, v in enumerate(values))
        + "</row>"
        for r, values in enumerate(rows, start=1)
    )
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets>'
            f'<sheet name="{sheet}" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{ns}"><sheetData>{body}</sheetData></worksheet>',
        )
    return buffer.getvalue()


def test_xlsx_golden_set_reads_the_golden_set_sheet() -> None:
    rows = [["Scenario Type", "Question", "Expected Behavior"], ["Ranking", "Top themes?", "Rank"]]
    golden = parse_golden_set(_xlsx(rows), source_name="gold.xlsx")
    assert [c.question for c in golden.cases] == ["Top themes?"]
    with pytest.raises(InputAssetError, match="no sheet named 'Golden Set'"):
        parse_golden_set(_xlsx(rows, sheet="Sheet1"), source_name="gold.xlsx")
    with pytest.raises(InputAssetError, match="not a valid .xlsx"):
        parse_golden_set(b"not a zip", source_name="gold.xlsx")


def test_inspiration_is_bounded_and_relevant() -> None:
    golden = parse_golden_set(GOLDEN_CSV.encode(), source_name="gold.csv")
    picked = golden_inspiration(golden, query="compare csat with complaints", limit=1)
    assert [p["case_id"] for p in picked] == ["GOLD_002"]
    assert golden_inspiration(golden, limit=0) == []


# --- grounding against the Agent context ---------------------------------------------------


def test_golden_names_are_canonicalised_against_the_reference(agent_reference: Reference) -> None:
    golden = ground_golden_set(
        parse_golden_set(GOLDEN_CSV.encode(), source_name="gold.csv"), agent_reference
    )
    first, second = golden.cases
    assert first.expected_tools == ["aggregate_feedback"]  # alias -> canonical
    assert first.expected_data_sources == ["complaints"]
    assert second.expected_data_sources == ["complaints", "csat"]  # group "both" expanded
    assert any("constraint: Never blend sources" in line for line in first.reference_context)


def test_unknown_or_incompatible_names_fail_before_any_scenario(
    agent_reference: Reference,
) -> None:
    unknown = GOLDEN_CSV.replace(",aggregate,", ",forecast_tool,")
    with pytest.raises(InputAssetError, match="unknown expected tool 'forecast_tool'"):
        ground_golden_set(parse_golden_set(unknown.encode(), source_name="g.csv"), agent_reference)
    expectations = Expectations(
        expected_behavior="b",
        success_criteria=["c"],
        expected_tools=["search_complaints"],
        expected_data_sources=["csat"],
    )
    with pytest.raises(InputAssetError, match="not supported by expected tool"):
        ground_agent_expectations(expectations, agent_reference)
    with pytest.raises(InputAssetError, match="must name at least one reference tool"):
        ground_agent_expectations(
            expectations.model_copy(update={"expected_tools": []}), agent_reference
        )


def test_agent_runs_need_an_agent_context(agent_reference: Reference) -> None:
    validate_agent_contract(agent_reference)
    with pytest.raises(InputAssetError, match="require Reference data"):
        validate_agent_contract(None)
    plain = parse_reference(CATALOG.encode(), source_name="c.yaml", max_chars=200_000)
    with pytest.raises(InputAssetError, match="has no agent_context"):
        validate_agent_contract(plain)


def test_agent_context_rejects_unknown_sources_and_duplicate_aliases() -> None:
    with pytest.raises(ValueError, match="names unknown data source 'sales'"):
        AgentContext.model_validate(
            {
                **AGENT_CONTEXT,
                "tools": [{"name": "t", "description": "d", "data_sources": ["sales"]}],
            }
        )
    # Names already match case-insensitively, so a case-only alias is a duplicate too.
    duplicate = {"name": "x", "description": "d", "aliases": ["COMPLAINTS"]}
    with pytest.raises(ValueError, match="duplicate data source name/alias"):
        AgentContext.model_validate(
            {**AGENT_CONTEXT, "data_sources": [*AGENT_CONTEXT["data_sources"], duplicate]}
        )


def test_agent_view_of_the_reference_hides_the_physical_schema(
    agent_reference: Reference,
) -> None:
    rendered = agent_reference.render_agent_context()
    assert "aggregate_feedback" in rendered and "complaints_t" not in rendered
    assert agent_reference.rendered_chars() == len(agent_reference.render()) + len(rendered)


def test_adding_no_agent_context_keeps_existing_reference_ids() -> None:
    plain = parse_reference(CATALOG.encode(), source_name="c.yaml", max_chars=200_000)
    explicit_null = parse_reference(
        (CATALOG + "agent_context: null\n").encode(), source_name="c.yaml", max_chars=200_000
    )
    assert plain.reference_id == explicit_null.reference_id


# --- profile selection ---------------------------------------------------------------------


def _product(**fields: Any) -> ProductDescription:
    base = {
        "product_id": "p",
        "product_version": "1",
        "name": "P",
        "description": "Answers questions about customer feedback.",
        "expected_outputs": ["Short answer"],
    }
    return ProductDescription.model_validate({**base, **fields})


def test_profile_auto_selected_from_sql_outputs_or_capabilities() -> None:
    auto = EvaluationProfileSelection.AUTO
    assert resolve_evaluation_profile(_product(expected_outputs=["SQL query"]), auto)[0] == "nl2sql"
    assert resolve_evaluation_profile(_product(supported=["writes SQL"]), auto)[0] == "nl2sql"
    profile, reason = resolve_evaluation_profile(_product(), auto)
    assert profile is EvaluationProfile.AGENT and "auto-selected" in reason
    explicit = resolve_evaluation_profile(_product(), EvaluationProfileSelection.NL2SQL)
    assert explicit == (EvaluationProfile.NL2SQL, "NL2SQL was selected by the user.")


def test_agent_metric_selection_is_normalised() -> None:
    chosen = [AgentMetric.CONTEXT_RECALL, AgentMetric.COMPLETION, AgentMetric.RELEVANCE]
    assert normalize_agent_metrics(chosen) == [
        AgentMetric.RELEVANCE,
        AgentMetric.COMPLETION,
        AgentMetric.CONTEXT_RECALL,
    ]
    with pytest.raises(ValueError, match="cannot disable required metrics: \\['completion'\\]"):
        normalize_agent_metrics([AgentMetric.RELEVANCE])
    assert EvaluationProfile("voc") is EvaluationProfile.AGENT  # former profile name


# --- trace tags ----------------------------------------------------------------------------


def test_trace_catalog_and_finding_links(
    sample_packs: list[ReferencePack], sample_scenarios: Any
) -> None:
    pack = sample_packs[0]
    question = QuestionCase.model_validate(
        {
            "question_id": "Q_001",
            "scenario_id": "SCN_001",
            "persona_id": "p",
            "mode": "grounded",
            "reference_pack_id": pack.pack_id,
            "utterance": "Top spenders?",
            "intended_goal": "g",
            "intent": {"task_type": "ranking"},
            "question_characteristics": {
                "style": "precise",
                "ambiguity_level": "low",
                "complexity": "simple",
            },
            "expectations": {
                "expected_behavior": "Rank employees.",
                "success_criteria": ["names the employees ranked by spend"],
            },
        }
    )
    tags = [item.tag for item in trace_catalog(question, pack)]
    assert tags[:3] == ["CRIT-01", "TABLE-01", "COLUMN-01.01"] and "JOIN-01" in tags
    assert "RULE-R1" in tags
    # Judge-suggested tags survive only when they exist in this question's catalog.
    assert merge_finding_tags(["JOIN-01", "TABLE-99"], [], question, pack) == ["JOIN-01"]

    evaluation = EvaluationResult.model_validate(
        {
            "evaluation_id": "EVAL_001",
            "question_id": "Q_001",
            "scenario_id": "SCN_001",
            "execution_id": "EXE_001",
            "product_outcome": "answered",
            "mode": "grounded",
            "verdict": "fail",
            "evaluation_status": "evaluated",
            "criteria_results": [
                {
                    "criterion": "names the employees ranked by spend",
                    "status": "not_met",
                    "rationale": "missing",
                }
            ],
            "findings": [
                {
                    "finding_id": "F_001",
                    "category": "schema_faithfulness",
                    "description": "Uses an unknown column full_nme instead of full_name",
                    "expected_behavior": "use employees.full_name",
                    "observed_behavior": "selects full_nme",
                    "severity": "medium",
                    "evidence_status": "supported",
                    "source": "llm",
                }
            ],
        }
    )
    linked = attach_trace_links(evaluation, question, pack)
    assert linked.criteria_results[0].tag == "CRIT-01"
    assert "SCHEMA-UNLISTED" in linked.findings[0].related_tags
    assert any(tag.startswith("TABLE-") for tag in linked.findings[0].related_tags)
