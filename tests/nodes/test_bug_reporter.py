"""Bug Reporter (D244): one bug per root cause, from validated failures only.

Live review: a run produced BUG_001..BUG_005 for five failed questions - one bug per failed
evaluation, all of its findings merged, no cross-question deduplication, and findings that did
not cause the failure (lower severity, a metric scored perfect) inside the bug.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from nodes.test_evaluator import (
    ANSWER,
    SQL,
    SQL_METRICS,
    _execution,
    _question,
    all_pass,
    finding,
    score,
)
from synthetic_user.core.config import AppConfig
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.metrics.registry import builtin_metric_config
from synthetic_user.models.evaluation import BugCandidate, EvaluationResult, Severity, Verdict
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator import EvaluationCase, Evaluator
from synthetic_user.nodes.evaluator.bugs import consolidate_bugs
from synthetic_user.nodes.evaluator.kpis import compute_kpis

SQL_TEXT = SQL
UTTERANCES = ("Who were our top 10 card spenders last quarter?", "Top card spenders in Q3?")
RULE = {"mechanism": "rule_violation", "subject": "R1", "hypothesis": "Currencies are mixed."}
SOURCE = {
    "mechanism": "wrong_source",
    "subject": "employees.full_name",
    "hypothesis": "The name column is picked without the documented join key.",
}


def _cases(scenario: Scenario, pack: ReferencePack, count: int) -> list[EvaluationCase]:
    cases = []
    for index in range(1, count + 1):
        question = _question(scenario, pack).model_copy(
            update={"question_id": f"Q_00{index}", "utterance": UTTERANCES[index - 1]}
        )
        execution = _execution(ANSWER).model_copy(
            update={"execution_id": f"EXE_00{index}", "question_id": f"Q_00{index}"}
        )
        cases.append(EvaluationCase(question, scenario, execution))
    return cases


def _reply(scores: dict[str, float], findings: list[dict[str, Any]]) -> dict[str, Any]:
    reply = all_pass(*SQL_METRICS)
    for metric_id, value in scores.items():
        reply["metrics"][metric_id] = score(value)
    reply["findings"] = findings
    return reply


async def _evaluate(
    settings: AppConfig,
    product: ProductDescription,
    reference: Reference,
    cases: list[EvaluationCase],
    replies: dict[str, dict[str, Any]],
) -> list[EvaluationResult]:
    """``replies``: the General Judge reply per question utterance (calls run concurrently)."""

    def respond(system: str, user: str) -> dict[str, Any]:
        return next(reply for utterance, reply in replies.items() if utterance in user)

    evaluator = Evaluator(
        FakeLLMClient([respond] * 10),
        settings,
        logging.getLogger("t"),
        reference=reference,
        metric_config=builtin_metric_config(list(SQL_METRICS)),
    )
    return await evaluator.evaluate_many(cases, product)


async def test_one_root_cause_across_questions_and_metrics_is_one_bug(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
) -> None:
    reply = _reply(
        {"rules_adherence": 0, "semantic_correctness": 0.5},
        [
            finding("rules_adherence", "SUM(c.amount)", severity="critical", root_cause=RULE),
            finding("semantic_correctness", "SUM(c.amount)", root_cause=RULE),
        ],
    )
    cases = _cases(sample_scenarios[0], sample_packs[0], 2)
    results = await _evaluate(
        app_config,
        sample_product,
        sample_reference,
        cases,
        dict.fromkeys(UTTERANCES, reply),
    )
    assert [r.verdict for r in results] == [Verdict.FAIL, Verdict.FAIL]
    # Each evaluation keeps its own occurrence (its own finding ids) under the SAME bug id.
    assert [[b.bug_id for b in r.bug_candidates] for r in results] == [["BUG_001"], ["BUG_001"]]
    for result in results:
        [bug] = result.bug_candidates
        assert set(bug.finding_ids) == {f.finding_id for f in result.findings}
    [bug] = consolidate_bugs(results)
    assert [o.question_id for o in bug.occurrences] == ["Q_001", "Q_002"]
    assert bug.severity is Severity.CRITICAL
    assert bug.metric_ids == ["rules_adherence", "semantic_correctness"]
    assert bug.root_cause_key == "rule_violation|r1"
    assert bug.title.startswith("rule violation (R1): ")
    assert bug.evidence == ['Q_001: "SUM(c.amount)"', 'Q_002: "SUM(c.amount)"']
    lines = bug.description.split("\n")
    assert [line.split(":", 1)[0] for line in lines] == [
        "Expected",
        "Actual",
        "Root-cause hypothesis",
        "Impact",
        "Evidence",
    ]
    assert bug.impact == (
        "critical severity; 2 affected question(s) (Q_001, Q_002); "
        "metric(s): rules_adherence, semantic_correctness"
    )
    assert len(bug.finding_ids) == 4  # traceable to every original finding
    kpis = compute_kpis(
        run_id="RUN_1",
        product_id=sample_product.product_id,
        mode=GenerationMode.GROUNDED,
        scenarios=sample_scenarios,
        executions=[c.execution for c in cases],
        evaluations=results,
    )
    assert (kpis.new_bugs, kpis.new_bugs_by_severity["critical"]) == (1, 1)  # counted once


async def test_distinct_defects_stay_separate_even_in_one_question(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
) -> None:
    reply = _reply(
        {"rules_adherence": 0, "schema_faithfulness": 0},
        [
            finding("rules_adherence", "SUM(c.amount)", root_cause=RULE),
            finding("schema_faithfulness", "e.full_name", root_cause=SOURCE),
        ],
    )
    [result] = await _evaluate(
        app_config,
        sample_product,
        sample_reference,
        _cases(sample_scenarios[0], sample_packs[0], 1),
        {UTTERANCES[0]: reply},
    )
    assert [b.root_cause_key for b in result.bug_candidates] == [
        "rule_violation|r1",
        "wrong_source|employees.full_name",  # canonicalised through the catalog
    ]
    assert len({b.bug_id for b in result.bug_candidates}) == 2


async def test_one_defect_labelled_differently_per_metric_is_one_bug(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
) -> None:
    """Live Gemini: the same wrong column came back as wrong_source, wrong_logic and a
    rule_violation on a pack-relative tag, plus the goal finding - one defect, one bug."""
    reply = _reply(
        {"rules_adherence": 0, "semantic_correctness": 0, "schema_faithfulness": 0.5},
        [
            finding(
                "semantic_correctness",
                "SUM(c.amount)",
                root_cause={**SOURCE, "mechanism": "wrong_logic", "subject": "c.amount"},
            ),
            finding(
                "rules_adherence",
                "c.amount",
                root_cause={**RULE, "subject": "TABLE-RULE-01.01"},
            ),
            finding(
                "schema_faithfulness",
                "SUM(c.amount) AS total_spend",
                root_cause={**SOURCE, "subject": "card_transactions.amount"},
            ),
        ],
    )
    reply["goal_completion"] = {
        "status": "not_completed",
        "reason": "Spend is summed across currencies.",
        "evidence": SQL_TEXT,
    }
    [result] = await _evaluate(
        app_config,
        sample_product,
        sample_reference,
        _cases(sample_scenarios[0], sample_packs[0], 1),
        {UTTERANCES[0]: reply},
    )
    assert result.verdict is Verdict.FAIL
    [bug] = result.bug_candidates
    assert len(bug.finding_ids) == 4  # three metrics + the goal outcome
    assert not bug.title.startswith("wrong result")  # a metric finding names the bug


@pytest.mark.parametrize("subject", ["R1", "TABLE-RULE-01.01"])
async def test_the_same_rule_violation_in_two_questions_is_one_bug(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
    subject: str,
) -> None:
    reply = _reply(
        {"rules_adherence": 0},
        [finding("rules_adherence", "SUM(c.amount)", root_cause={**RULE, "subject": subject})],
    )
    results = await _evaluate(
        app_config,
        sample_product,
        sample_reference,
        _cases(sample_scenarios[0], sample_packs[0], 2),
        dict.fromkeys(UTTERANCES, reply),
    )
    assert [[b.bug_id for b in r.bug_candidates] for r in results] == [["BUG_001"], ["BUG_001"]]


async def test_a_whole_artifact_quote_does_not_merge_distinct_defects(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
) -> None:
    reply = _reply(
        {"rules_adherence": 0, "schema_faithfulness": 0},
        [
            finding("rules_adherence", "SUM(c.amount)", root_cause=RULE),
            finding("schema_faithfulness", "e.full_name", root_cause=SOURCE),
        ],
    )
    reply["goal_completion"] = {
        "status": "not_completed",
        "reason": "Wrong result.",
        "evidence": SQL_TEXT,  # contains both quotes: never a link by itself
    }
    [result] = await _evaluate(
        app_config,
        sample_product,
        sample_reference,
        _cases(sample_scenarios[0], sample_packs[0], 1),
        {UTTERANCES[0]: reply},
    )
    assert len(result.bug_candidates) == 2
    assert sum(len(b.finding_ids) for b in result.bug_candidates) == 3  # goal joins one of them


@pytest.mark.parametrize(
    "cause",
    [
        {
            "mechanism": "wrong_logic",
            "subject": "the monthly trend",
            "hypothesis": "h",
        },  # unverifiable
        {"mechanism": "wrong_logic", "subject": None, "hypothesis": "h"},  # no subject
        None,  # no root cause given
    ],
)
async def test_without_a_verifiable_subject_questions_are_never_merged(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
    cause: dict[str, Any] | None,
) -> None:
    extra = {"root_cause": cause} if cause else {}
    reply = _reply(
        {"semantic_correctness": 0}, [finding("semantic_correctness", "SUM(c.amount)", **extra)]
    )
    results = await _evaluate(
        app_config,
        sample_product,
        sample_reference,
        _cases(sample_scenarios[0], sample_packs[0], 2),
        dict.fromkeys(UTTERANCES, reply),
    )
    assert [[b.bug_id for b in r.bug_candidates] for r in results] == [["BUG_001"], ["BUG_002"]]
    assert len(consolidate_bugs(results)) == 2


async def test_a_subjectless_finding_joins_the_same_defect_in_its_question(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
) -> None:
    reply = _reply(
        {"rules_adherence": 0, "semantic_correctness": 0.5},
        [
            finding("rules_adherence", "SUM(c.amount)", root_cause=RULE),
            finding(
                "semantic_correctness",
                "SUM(c.amount)",
                root_cause={**RULE, "subject": None},
            ),
        ],
    )
    [result] = await _evaluate(
        app_config,
        sample_product,
        sample_reference,
        _cases(sample_scenarios[0], sample_packs[0], 1),
        {UTTERANCES[0]: reply},
    )
    [bug] = result.bug_candidates
    assert len(bug.finding_ids) == 2


@pytest.mark.parametrize(
    ("scores", "findings", "verdict", "bug_findings"),
    [
        # Below the fail line: a lower-severity issue never becomes a bug on its own.
        (
            {"semantic_correctness": 0.5},
            [finding("semantic_correctness", "SUM(c.amount)", severity="medium")],
            Verdict.PASS,
            0,
        ),
        # Not proven / not settled: inconclusive, no bug.
        (
            {"semantic_correctness": 0.5},
            [finding("semantic_correctness", "SUM(c.amount)", evidence_status="weak")],
            Verdict.INCONCLUSIVE,
            0,
        ),
        (
            {"semantic_correctness": 0.5},
            [finding("semantic_correctness", "SUM(c.amount)", unresolved=True)],
            Verdict.INCONCLUSIVE,
            0,
        ),
        # A finding on a metric the judge scored perfect is not part of the failure.
        (
            {"rules_adherence": 0},
            [
                finding("rules_adherence", "SUM(c.amount)", root_cause=RULE),
                finding("relevance", "SUM(c.amount)", root_cause=SOURCE),
            ],
            Verdict.FAIL,
            1,
        ),
    ],
)
async def test_false_positive_findings_never_become_bugs(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
    scores: dict[str, float],
    findings: list[dict[str, Any]],
    verdict: Verdict,
    bug_findings: int,
) -> None:
    [result] = await _evaluate(
        app_config,
        sample_product,
        sample_reference,
        _cases(sample_scenarios[0], sample_packs[0], 1),
        {UTTERANCES[0]: _reply(scores, findings)},
    )
    assert result.verdict is verdict
    assert sum(len(b.finding_ids) for b in result.bug_candidates) == bug_findings


def test_bugs_written_before_root_causes_are_kept_as_they_are() -> None:
    legacy = {
        "bug_id": "BUG_001",
        "title": "t",
        "description": "d",
        "reproduction_question": "q",
        "product_id": "p",
        "question_id": "Q_001",
        "scenario_id": "SCN_001",
        "execution_id": "EXE_001",
        "finding_ids": ["F_001"],
    }
    bug = BugCandidate.model_validate(legacy)
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
            "findings": [
                {
                    "finding_id": "F_001",
                    "category": "relevance",
                    "description": "d",
                    "expected_behavior": "e",
                    "observed_behavior": "o",
                    "severity": "high",
                    "evidence_status": "supported",
                    "source": "llm",
                }
            ],
            "bug_candidates": [legacy],
        }
    )
    assert consolidate_bugs([evaluation]) == [bug]
    assert json.loads(bug.model_dump_json())["occurrences"] == []
