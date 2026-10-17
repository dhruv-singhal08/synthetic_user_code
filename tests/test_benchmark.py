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
from synthetic_user.models.reference import ReferenceIndex
from synthetic_user.nodes.evaluator import Evaluator
from synthetic_user.nodes.evaluators.plugins.sql.models import SqlChecks

BENCH = Path(__file__).resolve().parent / "fixtures" / "benchmark"


def test_labeled_cases_are_valid() -> None:
    cases = load_cases(BENCH / "labeled_cases.jsonl")
    scenarios = {s.id: s for s in load_scenarios(BENCH / "scenarios.json")}
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
    block = user.split("Success criteria (for goal_completeness):", 1)[1]
    criteria = json.loads(block.split("Product response", 1)[0])
    clarification = "Product response (clarification)" in user
    if "Product response (denied)" in user:
        na = {"status": "not_applicable", "rationale": "n/a"}
        return {
            **{k: na for k in ("relevance", "goal_completeness", "persona_fit")},
            "capability_consistency": ok,
            "clarification_handling": na,
            "safety_and_honesty": na,
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
        "findings": [],
        "overall_verdict": "pass",
    }


async def test_agreement_computed_with_scripted_llm(
    app_config: AppConfig,
    sample_index: ReferenceIndex,
    sample_product: ProductDescription,
    sample_chatbot: ProductDescription,
    sample_persona: Persona,
) -> None:
    """With an LLM that always approves, only deterministic rules can disagree with it."""
    all_pass = dict.fromkeys(SqlChecks.model_fields, "pass")

    def approve(system: str, user: str) -> dict[str, Any]:
        if "Success criteria (for goal_completeness)" in user:
            return _approve_generic(user)
        return {"checks": all_pass, "findings": [], "verdict": "pass"}

    evaluator = Evaluator(
        FakeLLMClient([approve] * 60),
        app_config,
        logging.getLogger("test"),
        index=sample_index,
        id_generator=SequentialIdGenerator(),
    )
    report = await run_benchmark(
        evaluator,
        load_cases(BENCH / "labeled_cases.jsonl"),
        load_scenarios(BENCH / "scenarios.json"),
        sample_index,
        {p.product_id: p for p in (sample_product, sample_chatbot)},
        sample_persona,
    )
    verdicts = {r.case_id: r.verdict for r in report.rows}
    # Deterministic rules decide these regardless of the LLM:
    assert verdicts["B03"] == "fail"  # wrong join vs verified relationship
    assert verdicts["B09"] == "inconclusive"  # table not in the reference index
    assert verdicts["B13"] == "fail"  # syntax error
    assert verdicts["B14"] == "fail"  # unknown column
    assert verdicts["B18"] == "pass"  # justified denial (capability judged only)
    assert {(r.mode, r.product_id) for r in report.rows} == {
        ("grounded", "sample_nl2sql"),
        ("ungrounded", "sample_policy_chatbot"),
    }
    assert report.agreed == 8
    assert report.agreement == pytest.approx(8 / 16)
