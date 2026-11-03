"""The built-in metric catalog (exactly 11) and the frozen run-level metric configuration."""

from __future__ import annotations

import logging

import pytest

from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.metrics.catalog import builtin_definitions, builtin_ids, category_labels
from synthetic_user.metrics.registry import builtin_metric_config, resolve_metric_config
from synthetic_user.models.metrics import (
    CustomMetricInput,
    MetricSelection,
    ResolvedMetricConfig,
)
from synthetic_user.models.product import ProductDescription

CATALOG = {
    "relevance": "answer_quality",
    "correctness_completeness": "answer_quality",
    "context_precision": "retrieval_quality",
    "context_recall": "retrieval_quality",
    "mrr": "retrieval_quality",
    "faithfulness": "grounding_citations",
    "citation_correctness": "grounding_citations",
    "semantic_correctness": "sql_quality",
    "rules_adherence": "sql_quality",
    "schema_faithfulness": "sql_quality",
    "tool_correctness": "agent",
}


def test_exactly_eleven_builtin_metrics_with_agreed_ids_and_categories() -> None:
    assert {d.metric_id: d.category.value for d in builtin_definitions()} == CATALOG
    for gone in ("completion", "answer_correctness", "tool_selection", "precision_at_k"):
        assert gone not in builtin_ids()
    assert set(category_labels().values()) >= {
        "Answer Quality / Task Fulfilment",
        "Retrieval Quality",
        "Grounding & Citations",
        "SQL Quality",
        "Agent",
        "Custom",
    }


def test_judge_routing_and_expectation_schemas_are_data() -> None:
    by_id = {d.metric_id: d for d in builtin_definitions()}
    retrieval = {m for m, d in by_id.items() if d.judge_group == "retrieval_grounding"}
    assert retrieval == {
        "context_precision",
        "context_recall",
        "mrr",
        "faithfulness",
        "citation_correctness",
    }
    needing = {m: [f.name for f in d.question_expectation_schema] for m, d in by_id.items()}
    assert {m for m, fields in needing.items() if fields} == {
        "context_recall",
        "citation_correctness",
        "tool_correctness",
    }
    assert needing["context_recall"] == ["required_information"]
    assert by_id["mrr"].continuous_score and not by_id["mrr"].counts_toward_verdict
    assert all(set(d.score_anchors) == {"1", "0.5", "0"} for d in by_id.values())


def test_builtin_only_selection_skips_the_compiler(
    sample_product: ProductDescription,
) -> None:
    import asyncio

    llm = FakeLLMClient()  # any call would fail: the queue is empty
    config = asyncio.run(
        resolve_metric_config(
            MetricSelection(selected_metrics=["tool_correctness", "relevance"]),
            sample_product,
            None,
            llm=llm,
            logger=logging.getLogger("t"),
            max_retries=0,
            sql_dialect=None,
        )
    )
    assert llm.calls == [] and not config.compiler_called
    assert config.selected_metric_ids == ["relevance", "tool_correctness"]  # catalog order
    assert config.config_id.startswith("MC_") and config.catalog_version


def test_config_is_frozen_and_round_trips() -> None:
    config = builtin_metric_config(["relevance"])
    with pytest.raises(ValueError):
        config.compiler_called = True  # type: ignore[misc]
    again = ResolvedMetricConfig.model_validate_json(config.model_dump_json())
    assert again == config and again.config_id == config.config_id


@pytest.mark.parametrize(
    ("selected", "message"),
    [
        (["completion"], "replaced by correctness_completeness"),
        (["tool_selection"], "replaced by tool_correctness"),
        (["answer_correctness"], "replaced by correctness_completeness"),
        (["nope"], "unknown built-in metric 'nope'"),
    ],
)
def test_legacy_and_unknown_ids_are_refused_never_mapped(selected: list[str], message: str) -> None:
    with pytest.raises(InputAssetError, match=message):
        builtin_metric_config(selected)


def test_selection_needs_a_metric_and_fingerprint_ignores_builtin_order() -> None:
    with pytest.raises(ValueError, match="select at least one metric"):
        MetricSelection()
    with pytest.raises(ValueError, match="more than once"):
        MetricSelection(selected_metrics=["relevance", "relevance"])
    a = MetricSelection(selected_metrics=["relevance", "mrr"])
    b = MetricSelection(selected_metrics=["mrr", "relevance"])
    custom = [CustomMetricInput(text="x", provenance="free text item 1")]
    c = MetricSelection(selected_metrics=["mrr", "relevance"], custom_inputs=custom)
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()
