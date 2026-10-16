from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from synthetic_user.benchmark import load_cases, load_scenarios, run_benchmark
from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.core.identifiers import SequentialIdGenerator
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.models.evaluation import EvaluationChecks
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import ReferenceCatalog
from synthetic_user.nodes.sql_evaluator import SqlEvaluator

BENCH = Path(__file__).resolve().parent / "fixtures" / "benchmark"


def test_labeled_cases_are_valid() -> None:
    cases = load_cases(BENCH / "labeled_cases.jsonl")
    scenarios = {s.id for s in load_scenarios(BENCH / "scenarios.json")}
    assert 12 <= len(cases) <= 15
    assert len({c.case_id for c in cases}) == len(cases)
    assert {c.scenario_id for c in cases} <= scenarios
    assert {c.label for c in cases} == {"pass", "fail", "inconclusive"}


def test_bad_case_line_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"case_id": "X"}\n', encoding="utf-8")
    with pytest.raises(InputAssetError, match=r"cases.jsonl:1"):
        load_cases(path)


async def test_agreement_computed_with_scripted_llm(
    app_config: AppConfig, sample_catalog: ReferenceCatalog, sample_product: ProductDescription
) -> None:
    """With an LLM that always approves, only deterministic rules can disagree with it."""
    all_pass = dict.fromkeys(EvaluationChecks.model_fields, "pass")

    def approve(system: str, user: str) -> dict[str, Any]:
        if "clarification" in system:
            return {"relevance": "relevant", "necessity": "necessary", "reason": "fine"}
        return {"checks": all_pass, "findings": [], "verdict": "pass"}

    evaluator = SqlEvaluator(
        FakeLLMClient([approve] * 20),
        app_config,
        logging.getLogger("test"),
        catalog=sample_catalog,
        id_generator=SequentialIdGenerator(),
    )
    report = await run_benchmark(
        evaluator,
        load_cases(BENCH / "labeled_cases.jsonl"),
        load_scenarios(BENCH / "scenarios.json"),
        sample_catalog,
        sample_product,
    )
    verdicts = {r.case_id: r.verdict for r in report.rows}
    # Deterministic rules decide these regardless of the LLM:
    assert verdicts["B03"] == "fail"  # wrong join vs verified relationship
    assert verdicts["B09"] == "inconclusive"  # table not in catalog
    assert verdicts["B13"] == "fail"  # syntax error
    assert verdicts["B14"] == "fail"  # unknown column
    assert report.agreed == 9
    assert report.agreement == pytest.approx(9 / 15)
