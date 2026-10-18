from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from synthetic_user.benchmark import load_cases, load_scenarios, run_benchmark
from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.core.identifiers import SequentialIdGenerator
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import Reference
from synthetic_user.nodes.evaluator import Evaluator

BENCH = Path(__file__).resolve().parent / "fixtures" / "benchmark"


def test_labeled_cases_are_valid() -> None:
    cases = load_cases(BENCH / "labeled_cases.jsonl")
    scenario_list, packs = load_scenarios(BENCH / "scenarios.json")
    scenarios = {s.id: s for s in scenario_list}
    assert {p.pack_id for p in packs} == {
        s.reference_pack_id for s in scenario_list if s.reference_pack_id
    }
    assert 14 <= len(cases) <= 18
    assert len({c.case_id for c in cases}) == len(cases)
    assert {c.scenario_id for c in cases} <= set(scenarios)
    assert {c.label for c in cases} == {"pass", "fail", "inconclusive"}
    # both modes and both product types are covered
    assert {scenarios[c.scenario_id].mode for c in cases} == {"grounded", "ungrounded"}
    assert {c.product_id for c in cases} == {"sample_nl2sql", "sample_policy_chatbot"}
    assert {c.outcome for c in cases} == {"answered", "clarification", "denied"}


def test_bad_case_line_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"case_id": "X"}\n', encoding="utf-8")
    with pytest.raises(InputAssetError, match=r"cases.jsonl:1"):
        load_cases(path)


def _approve_generic(user: str) -> dict[str, Any]:
    """A generic judge that approves everything (criteria copied from the prompt)."""
    ok = {"status": "pass", "rationale": "fine"}
    na = {"status": "not_applicable", "rationale": "n/a"}
    block = user.split("What was asked and what is expected:\n", 1)[1]
    criteria = json.loads(block.split("\n\nProduct description", 1)[0])["success_criteria"]
    clarification = "Product response (clarification)" in user
    grounded = "Reference pack (AUTHORITATIVE" in user
    if "Product response (denied)" in user:
        na = {"status": "not_applicable", "rationale": "n/a"}
        return {
            **{k: na for k in ("relevance", "goal_completeness", "persona_fit")},
            "capability_consistency": ok,
            "clarification_handling": na,
            "safety_and_honesty": na,
            "reference_alignment": na if grounded else None,
            "overall_verdict": "pass",
        }
    return {
        "relevance": ok,
        "goal_completeness": na if clarification else ok,
        "criteria": []
        if clarification
        else [{"criterion": c, "status": "met", "rationale": "fine"} for c in criteria],
        "persona_fit": ok,
        "capability_consistency": ok,
        "clarification_handling": ok if clarification else na,
        "clarification": {
            "relevance": "relevant",
            "necessity": "necessary",
            "reason": "fine",
        }
        if clarification
        else None,
        "safety_and_honesty": ok,
        "reference_alignment": (na if clarification else ok) if grounded else None,
        "findings": [],
        "overall_verdict": "pass",
    }


async def test_agreement_computed_with_scripted_llm(
    app_config: AppConfig,
    sample_reference: Reference,
    sample_product: ProductDescription,
    sample_chatbot: ProductDescription,
    sample_persona: Persona,
) -> None:
    """With a judge that approves everything, every case passes: agreement = pass labels.
    (There are no deterministic SQL rules any more - design v3.)"""

    def approve(system: str, user: str) -> dict[str, Any]:
        return _approve_generic(user)

    evaluator = Evaluator(
        FakeLLMClient([approve] * 20),
        app_config,
        logging.getLogger("test"),
        reference=sample_reference,
        id_generator=SequentialIdGenerator(),
    )
    cases = load_cases(BENCH / "labeled_cases.jsonl")
    scenarios, packs = load_scenarios(BENCH / "scenarios.json")
    report = await run_benchmark(
        evaluator,
        cases,
        scenarios,
        packs,
        {p.product_id: p for p in (sample_product, sample_chatbot)},
        sample_persona,
    )
    assert {r.verdict for r in report.rows} == {"pass"}
    assert {(r.mode, r.product_id) for r in report.rows} == {
        ("grounded", "sample_nl2sql"),
        ("ungrounded", "sample_policy_chatbot"),
    }
    passes = sum(1 for c in cases if c.label == "pass")
    assert report.agreed == passes
    assert report.agreement == pytest.approx(passes / len(cases))
