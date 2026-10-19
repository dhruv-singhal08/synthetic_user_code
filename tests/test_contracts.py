"""Compact output contracts (D204): each example has exactly its model's fields, so the
example and the Pydantic model that validates the reply cannot drift apart."""

from __future__ import annotations

import json
import types
import typing
from typing import Any, get_args, get_origin

import pytest
from pydantic import BaseModel

from synthetic_user.models.evaluation import EvaluationDraft
from synthetic_user.models.question import QuestionBatchDraft
from synthetic_user.models.reference import CatalogPackDraft, DocumentPackDraft, ReferenceKind
from synthetic_user.models.scenario import ScenarioBatchDraft
from synthetic_user.prompts.contracts import render_contract
from synthetic_user.prompts.question import QUESTION_EXAMPLE
from synthetic_user.prompts.response_evaluation import evaluation_example
from synthetic_user.prompts.scenario import scenario_example


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


def test_question_example_matches_the_model() -> None:
    _assert_matches(QUESTION_EXAMPLE, QuestionBatchDraft, "questions")


@pytest.mark.parametrize("grounded", [True, False])
def test_evaluation_example_matches_the_model(grounded: bool) -> None:
    example = evaluation_example(grounded)
    assert (example["correctness"] is None) is (not grounded)
    _assert_matches({**example, "correctness": example["relevancy"]}, EvaluationDraft, "eval")


@pytest.mark.parametrize(
    "example",
    [
        scenario_example(ReferenceKind.DATA_CATALOG),
        scenario_example(None),
        QUESTION_EXAMPLE,
        evaluation_example(True),
    ],
)
def test_contracts_stay_short(example: dict[str, Any]) -> None:
    contract = render_contract(example)
    assert len(contract.splitlines()) < 60
    assert "$defs" not in contract and json.loads(contract.split("\n", 1)[1]) == example
