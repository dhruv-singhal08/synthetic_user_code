"""Compact output contracts (D204): each example has exactly its model's fields, so the
example and the Pydantic model that validates the reply cannot drift apart."""

from __future__ import annotations

import json
import types
import typing
from typing import Any, get_args, get_origin

import pytest
from pydantic import BaseModel

from synthetic_user.metrics.catalog import builtin_definitions
from synthetic_user.metrics.compiler import CompiledMetricDraft, CompilerOutput
from synthetic_user.metrics.compiler_prompt import compiler_example
from synthetic_user.metrics.registry import builtin_metric_config
from synthetic_user.models.evaluation import (
    FindingDraft,
    GeneralJudgeDraft,
    JoinDecisionDraft,
    JudgedScore,
    RetrievalJudgeDraft,
)
from synthetic_user.models.execution import AnsweredResponse, RetrievedChunk, RetrievedContext
from synthetic_user.models.question import (
    MetricExpectationDraft,
    MetricMappingDraft,
    QuestionBatchDraft,
    QuestionCase,
)
from synthetic_user.models.reference import CatalogPackDraft, DocumentPackDraft, ReferenceKind
from synthetic_user.models.scenario import ScenarioBatchDraft
from synthetic_user.nodes.evaluator.general_prompt import judge_example
from synthetic_user.nodes.evaluator.retrieval_prompt import RetrievalJudgeInput, retrieval_example
from synthetic_user.nodes.question.context import ContextRequest
from synthetic_user.nodes.question.prompt import question_example
from synthetic_user.nodes.scenario.prompt import scenario_example
from synthetic_user.prompts.contracts import render_contract


def _models(annotation: Any) -> list[type[BaseModel]]:
    """Pydantic models inside an annotation (list[X], X | None, X | Y ...)."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    if get_origin(annotation) in (list, typing.Union, types.UnionType):
        return [m for arg in get_args(annotation) for m in _models(arg)]
    return []


def _assert_matches(example: Any, model: type[BaseModel], path: str) -> None:
    assert isinstance(example, dict), path
    keys = {f.alias or name for name, f in model.model_fields.items()}
    assert set(example) == keys, f"{path}: example {sorted(example)} != model {sorted(keys)}"
    for name, field in model.model_fields.items():
        value = example[field.alias or name]
        if isinstance(value, list) and value and isinstance(value[0], dict):
            value = value[0]
        if not isinstance(value, dict):
            continue
        candidates = _models(field.annotation)
        if not candidates:  # a plain mapping such as golden_set_criteria: dict[str, str]
            continue
        if len(candidates) > 1:  # a union: the example shows the right member
            candidates = [m for m in candidates if set(value) == set(m.model_fields)]
        assert len(candidates) == 1, f"{path}.{name}: no single model fits {sorted(value)}"
        _assert_matches(value, candidates[0], f"{path}.{name}")


@pytest.mark.parametrize("kind", [None, *ReferenceKind])
def test_scenario_example_matches_the_model(kind: ReferenceKind | None) -> None:
    example = scenario_example(kind)
    scenario = example["scenarios"][0]
    if kind is None:
        assert "reference_pack" not in scenario
        scenario = {**scenario, "reference_pack": None}
    _assert_matches({"scenarios": [scenario]}, ScenarioBatchDraft, "scenario")
    if kind is ReferenceKind.DATA_CATALOG:
        _assert_matches(scenario["reference_pack"], CatalogPackDraft, "pack")
    if kind is ReferenceKind.DOCUMENT_COLLECTION:
        _assert_matches(scenario["reference_pack"], DocumentPackDraft, "pack")


QUESTION_EXAMPLE = question_example(ContextRequest())
GENERAL = [d for d in builtin_definitions() if d.judge_group == "general"]


def test_question_example_matches_the_model() -> None:
    _assert_matches(QUESTION_EXAMPLE, QuestionBatchDraft, "questions")
    context = QUESTION_EXAMPLE["questions"][0]["evaluation_context"]
    assert context["metric_mapping"] == {}
    assert context["success_criteria"][0]["id"] == "C1"


def test_question_example_maps_criteria_metrics_and_expectation_metrics() -> None:
    config = builtin_metric_config(
        ["relevance", "context_recall", "tool_correctness", "schema_faithfulness"]
    )
    by_id = {d.metric_id: d for d in config.definitions}
    request = ContextRequest(
        selected=list(config.definitions),
        criteria_metrics=[by_id["relevance"], by_id["tool_correctness"]],
        requested=[by_id["context_recall"], by_id["tool_correctness"]],
    )
    example = question_example(request)
    entries = example["questions"][0]["evaluation_context"]["metric_mapping"]
    # schema_faithfulness takes no criteria and no expectation: code maps it
    assert list(entries) == ["relevance", "context_recall", "tool_correctness"]
    assert entries["relevance"]["expectation"] is None and entries["relevance"]["criteria"]
    assert entries["context_recall"]["criteria"] == []
    _assert_matches(entries["tool_correctness"], MetricMappingDraft, "mapping")
    expectation = entries["tool_correctness"]["expectation"]
    _assert_matches(expectation, MetricExpectationDraft, "expectation")
    assert set(expectation["value"]) == {
        "expected_tools",
        "expected_data_sources",
        "expected_arguments",
    }


@pytest.mark.parametrize("classify", [True, False])
def test_general_judge_example_matches_the_model(classify: bool) -> None:
    example = judge_example(GENERAL, needs_classification=classify)
    expected = {"metrics", "findings", "goal_completion", "overall_verdict"}
    if classify:
        expected |= {"response_classification", "classification_note"}
    assert set(example) == expected
    assert sorted(example["metrics"]) == sorted(d.metric_id for d in GENERAL)
    for metric_id, score in example["metrics"].items():
        keys = set(JudgedScore.model_fields)
        if metric_id != "correctness_completeness":
            keys -= {"factual_verification"}
        assert set(score) == keys, metric_id
    # "unresolved" and "evidence_requests" are offered only with a catalog reconciliation;
    # "exempted_by" only with evaluation guidance (D243)
    assert set(example["findings"][0]) == set(FindingDraft.model_fields) - {
        "unresolved",
        "exempted_by",
    }
    assert set(example["goal_completion"]) == {"status", "reason", "evidence"}
    assert set(GeneralJudgeDraft.model_fields) - {"evidence_requests", "join_decisions"} >= set(
        example
    )


def test_reconciled_general_judge_example_shows_every_field() -> None:
    example = judge_example(
        GENERAL,
        needs_classification=True,
        unresolved_findings=True,
        lookup_offer=True,
        join_decisions=True,
        guidance=True,
    )
    assert set(example) == set(GeneralJudgeDraft.model_fields)
    assert set(example["findings"][0]) == set(FindingDraft.model_fields)
    assert example["evidence_requests"] == []
    _assert_matches(example["join_decisions"][0], JoinDecisionDraft, "join_decision")
    _assert_matches(example["findings"][0], FindingDraft, "finding")


def _question() -> QuestionCase:
    return QuestionCase.model_validate(
        {
            "question_id": "Q_001",
            "scenario_id": "SCN_001",
            "persona_id": "p",
            "mode": "ungrounded",
            "utterance": "Compare January and February complaints",
            "intended_goal": "g",
            "intent": {"task_type": "comparison"},
            "question_characteristics": {
                "style": "precise",
                "ambiguity_level": "low",
                "complexity": "simple",
            },
            "evaluation_context": {
                "success_criteria": [{"id": "C1", "requirement": "covers both months"}],
                "metric_mapping": {
                    "context_recall": {
                        "comparison_basis": "required_information_vs_chunks",
                        "expectation": {
                            "status": "generated",
                            "value": {"required_information": ["January", "February"]},
                        },
                    }
                },
            },
        }
    )


def test_retrieval_judge_example_matches_the_model(sample_product: Any) -> None:
    config = builtin_metric_config(
        ["context_precision", "context_recall", "mrr", "faithfulness", "citation_correctness"]
    )
    judged = [d for d in config.definitions if not d.continuous_score]
    data = RetrievalJudgeInput(
        product=sample_product,
        question=_question(),
        response=AnsweredResponse(text="January had more."),
        definitions=judged,
        needs_labels=True,
        context=RetrievedContext(
            availability="available",
            chunks=[RetrievedChunk(chunk_id="c1", rank=1, text="Jan: 10")],
            citations=[{"chunk_id": "c1", "resolved": True}],
            citations_exposed=True,
            rank_exposed=True,
        ),
        tool_trace=None,
        artifact_max_chars=8000,
    )
    example = retrieval_example(data)
    assert set(example) == set(RetrievalJudgeDraft.model_fields)
    assert "mrr" not in example["metrics"]  # MRR is computed by code, never scored by the judge


def test_compiler_example_matches_the_model() -> None:
    example = compiler_example()
    assert set(example) == set(CompilerOutput.model_fields)
    assert set(example["metrics"][0]) == set(CompiledMetricDraft.model_fields)


@pytest.mark.parametrize(
    "example",
    [
        scenario_example(ReferenceKind.DATA_CATALOG),
        scenario_example(None),
        QUESTION_EXAMPLE,
        judge_example(GENERAL, needs_classification=True),
    ],
)
def test_contracts_stay_short(example: dict[str, Any]) -> None:
    contract = render_contract(example)
    # + goal_completion (D243), + check-level decisions shown once (reforming)
    assert len(contract.splitlines()) < 180  # six general metrics in this example
    assert "$defs" not in contract and json.loads(contract.split("\n", 1)[1]) == example
