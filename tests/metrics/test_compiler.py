"""The metric compiler: one structured call per run with custom metrics, never semantic."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from synthetic_user.core.exceptions import LLMProviderError
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.metrics.registry import resolve_metric_config
from synthetic_user.models.metrics import CustomMetricInput, MetricSelection
from synthetic_user.models.product import ProductDescription

INPUTS = [
    CustomMetricInput(
        text="Workflow adherence: follows the approval steps", provenance="free text item 1"
    ),
    CustomMetricInput(text="Metric: Tone; Definition: Polite", provenance="m.xlsx: Sheet1 row 2"),
    CustomMetricInput(
        text="Metric: Politeness; Definition: Polite", provenance="m.xlsx: Sheet1 row 3"
    ),
    CustomMetricInput(text="SQL dialect adherence", provenance="free text item 2"),
]


def compiled(index: int, **extra: Any) -> dict[str, Any]:
    return {
        "input_index": index,
        "compilable": True,
        "display_name": f"Metric {index}",
        "definition": "compares the answer with the rule",
        "rubric": "decide by the rule",
        "score_anchors": {"1": "all", "0.5": "some", "0": "none"},
        "evidence_requirements": "the answer",
        "applicability": "every answer",
        "judge_group": "general",
        "required_execution_evidence": [],
        "required_reference_assets": [],
        "needs_question_expectations": False,
        "expectation_fields": [],
        "generation_instructions": None,
        **extra,
    }


def reply() -> dict[str, Any]:
    return {
        "metrics": [
            compiled(
                1,
                needs_question_expectations=True,
                expectation_fields=[
                    {"name": "expected_workflow", "type": "string_list", "description": "steps"}
                ],
                generation_instructions="list the documented steps for this question",
            ),
            compiled(2),
            compiled(3),  # resembles 2: compiled as written, never merged or rejected
            {
                "input_index": 4,
                "compilable": False,
                "uncompilable_reason": "no SQL dialect is configured for this run",
            },
        ]
    }


def resolve(llm: FakeLLMClient, product: ProductDescription, **kw: Any) -> Any:
    selection = MetricSelection(
        selected_metrics=kw.pop("builtin", ["relevance"]), custom_inputs=INPUTS
    )
    return asyncio.run(
        resolve_metric_config(
            selection,
            product,
            None,
            llm=llm,
            logger=logging.getLogger("t"),
            max_retries=kw.pop("retries", 1),
            sql_dialect=None,
        )
    )


def test_exactly_one_compiler_call_with_code_assigned_ids(
    sample_product: ProductDescription,
) -> None:
    llm = FakeLLMClient([reply()])
    config = resolve(llm, sample_product)
    assert len(llm.calls) == 1 and config.compiler_called
    assert config.selected_metric_ids == [
        "relevance",
        "custom_01",
        "custom_02",
        "custom_03",
        "custom_04",
    ]
    workflow = config.definition("custom_01")
    assert workflow is not None and workflow.origin == "custom" and workflow.category == "custom"
    assert (
        workflow.source_text == INPUTS[0].text and workflow.source_provenance == "free text item 1"
    )
    assert [f.name for f in workflow.question_expectation_schema] == ["expected_workflow"]
    tone, politeness = config.definition("custom_02"), config.definition("custom_03")
    assert tone is not None and politeness is not None and tone.available and politeness.available
    dialect = config.definition("custom_04")
    assert dialect is not None and not dialect.available
    assert dialect.unavailable_reason == "no SQL dialect is configured for this run"
    assert dialect.source_text == "SQL dialect adherence"  # the user's text is preserved
    prompt = llm.calls[0].user
    assert "untrusted user data" in prompt and "never merge" in prompt
    assert "m.xlsx: Sheet1 row 3" in prompt  # provenance reaches the compiler


def test_structural_problems_are_retried_never_fixed(sample_product: ProductDescription) -> None:
    bad = reply()
    bad["metrics"] = bad["metrics"][:3]  # one input missing
    retrieval_without_context = reply()
    retrieval_without_context["metrics"][1]["judge_group"] = "retrieval_grounding"
    llm = FakeLLMClient([bad, retrieval_without_context, reply()])
    config = resolve(llm, sample_product, retries=2)
    assert len(llm.calls) == 3
    assert "input_index 1..4 exactly once" in llm.calls[1].user
    assert "retrieval_grounding needs retrieved_context" in llm.calls[2].user
    assert config.definition("custom_02") is not None


def test_compiler_failure_marks_custom_metrics_unavailable_and_keeps_builtins(
    sample_product: ProductDescription,
) -> None:
    config = resolve(FakeLLMClient([LLMProviderError("down")]), sample_product)
    assert config.definition("relevance") is not None and config.definition("relevance").available
    unavailable = config.unavailable()
    assert [d.metric_id for d in unavailable] == [
        "custom_01",
        "custom_02",
        "custom_03",
        "custom_04",
    ]
    assert all("metric compiler failed" in (d.unavailable_reason or "") for d in unavailable)
    assert config.compiler_notes and "down" in config.compiler_notes[0]


def test_deterministic_judge_group_is_not_accepted_for_custom_metrics(
    sample_product: ProductDescription,
) -> None:
    bad = reply()
    bad["metrics"][0]["judge_group"] = "deterministic"
    llm = FakeLLMClient([bad, reply()])
    config = resolve(llm, sample_product)
    assert len(llm.calls) == 2 and "judge_group" in llm.calls[1].user
    assert config.definition("custom_01").judge_group == "general"  # type: ignore[union-attr]
