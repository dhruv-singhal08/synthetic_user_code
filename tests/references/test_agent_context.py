"""The reference's optional tool definitions (agent_context) and the report trace tags."""

from __future__ import annotations

import pytest

from synthetic_user.models.evaluation import EvaluationResult
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import AgentContext, Reference, ReferencePack
from synthetic_user.nodes.evaluator.trace_tags import (
    attach_trace_links,
    merge_finding_tags,
    trace_catalog,
)
from synthetic_user.references.loader import parse_reference
from tests_support import AGENT_CONTEXT, TOOL_CATALOG, evaluation_context, pack_spec


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


def test_tool_definitions_resolve_aliases_to_canonical_names(tool_reference: Reference) -> None:
    context = tool_reference.agent_context
    assert context is not None
    tool = context.tool("AGGREGATE")
    source = context.data_source("survey data")
    assert tool is not None and tool.name == "aggregate_feedback"
    assert source is not None and source.name == "csat"
    assert context.tool("made_up_tool") is None


def test_tool_view_of_the_reference_hides_the_physical_schema(tool_reference: Reference) -> None:
    rendered = tool_reference.render_agent_context()
    assert "aggregate_feedback" in rendered and "complaints_t" not in rendered
    assert tool_reference.rendered_chars() == len(tool_reference.render()) + len(rendered)


def test_adding_no_agent_context_keeps_existing_reference_ids() -> None:
    plain = parse_reference(TOOL_CATALOG.encode(), source_name="c.yaml", max_chars=200_000)
    explicit_null = parse_reference(
        (TOOL_CATALOG + "agent_context: null\n").encode(), source_name="c.yaml", max_chars=200_000
    )
    assert plain.reference_id == explicit_null.reference_id


# --- trace tags ----------------------------------------------------------------------------


def _question(pack: ReferencePack) -> QuestionCase:
    return QuestionCase.model_validate(
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
            "evaluation_context": evaluation_context(
                ["names the employees ranked by spend"], reference_spec=pack_spec(pack)
            ),
        }
    )


def _evaluation(metric_id: str = "schema_faithfulness") -> EvaluationResult:
    """A failed evaluation: criterion C1 failed in ``metric_id``."""
    return EvaluationResult.model_validate(
        {
            "schema_version": 2,
            "evaluation_id": "EVAL_001",
            "question_id": "Q_001",
            "scenario_id": "SCN_001",
            "execution_id": "EXE_001",
            "product_outcome": "answered",
            "mode": "grounded",
            "verdict": "fail",
            "evaluation_status": "evaluated",
            "metrics": {
                metric_id: {
                    "status": "scored",
                    "score": 0.5,
                    "rationale": "r",
                    "checks": [
                        {
                            "check_id": "CHECK:001",
                            "requirement_source": "success_criteria",
                            "requirement_id": "C1",
                            "status": "failed",
                            "reason": "missing",
                        }
                    ],
                }
            },
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


def test_trace_catalog_and_finding_links(sample_packs: list[ReferencePack]) -> None:
    pack = sample_packs[0]
    question = _question(pack)
    tags = [item.tag for item in trace_catalog(question, pack)]
    assert tags[:3] == ["C1", "TABLE-01", "COLUMN-01.01"] and "JOIN-01" in tags
    assert "RULE-R1" in tags
    # Judge-suggested tags survive only when they exist in this question's catalog.
    assert merge_finding_tags(["JOIN-01", "TABLE-99"], [], question, pack) == ["JOIN-01"]

    linked = attach_trace_links(_evaluation(), question, pack)
    assert "SCHEMA-UNLISTED" in linked.findings[0].related_tags
    assert any(tag.startswith("TABLE-") for tag in linked.findings[0].related_tags)
    # A failed criterion check is linked to the finding of the same metric (no regex guess).
    assert "C1" in linked.findings[0].related_tags


def test_failed_criterion_of_another_metric_is_not_linked(
    sample_packs: list[ReferencePack],
) -> None:
    pack = sample_packs[0]
    linked = attach_trace_links(_evaluation(metric_id="relevance"), _question(pack), pack)
    assert "C1" not in linked.findings[0].related_tags
