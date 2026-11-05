"""Independent goal-completion check (D243) and guidance exceptions as exempted findings.

Live findings: (1) supported questions the product answered "unsupported" passed because
Relevance was 1 and every other metric was N/A; (2) an approved guidance exception (an
environment-specific filter) still produced a critical finding and a failed verdict.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from nodes.test_evaluator import _execution, _question, all_pass, finding, score
from synthetic_user.core.config import AppConfig
from synthetic_user.core.guidance import TYPED_GUIDANCE, parse_evaluation_guidance
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.metrics.registry import builtin_metric_config
from synthetic_user.models.evaluation import EvaluationResult, Severity, Verdict
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.scenario import CapabilityClassification, Scenario
from synthetic_user.nodes.evaluator import EvaluationCase, Evaluator
from tests_support import failed_check

DENIAL = {"outcome": "denied", "message": "This request is unsupported."}
ANSWER = {"outcome": "answered", "text": "Spend for tenant DUMMY: 120 USD in Q1."}
GUIDANCE = "Do not penalize the environment-specific tenant DUMMY filter."


async def _run(
    settings: AppConfig,
    product: ProductDescription,
    reference: Reference,
    scenario: Scenario,
    pack: ReferencePack | None,
    response: dict[str, Any],
    replies: list[Any],
    *,
    metrics: tuple[str, ...] = ("relevance",),
    guidance: str | None = None,
) -> tuple[EvaluationResult, FakeLLMClient]:
    llm = FakeLLMClient(replies)
    evaluator = Evaluator(
        llm,
        settings,
        logging.getLogger("t"),
        reference=reference,
        metric_config=builtin_metric_config(list(metrics)),
        evaluation_guidance=parse_evaluation_guidance(
            guidance, source=TYPED_GUIDANCE, max_chars=20_000
        )
        if guidance
        else None,
    )
    case = EvaluationCase(_question(scenario, pack), scenario, _execution(response))
    return await evaluator.evaluate(case, product), llm


def _reply(goal: dict[str, Any], *metric_ids: str, **extra: Any) -> dict[str, Any]:
    reply = all_pass(*(metric_ids or ("relevance",)), classify=False)
    reply["goal_completion"] = goal
    reply.update(extra)
    return reply


# --- goal completion ----------------------------------------------------------------------------


async def test_unjustified_decline_of_a_supported_request_fails_with_scores_kept(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
) -> None:
    scenario, pack = sample_scenarios[0], sample_packs[0]
    assert scenario.capability is CapabilityClassification.SUPPORTED
    goal = {
        "status": "unjustified_decline",
        "reason": "Ranking spenders is a documented capability.",
        "evidence": "This request is unsupported.",
    }
    result, llm = await _run(
        app_config, sample_product, sample_reference, scenario, pack, DENIAL, [_reply(goal)]
    )
    assert len(llm.calls) == 1  # in the same General Judge call
    user = " ".join(llm.calls[0].user.split())
    assert "Scenario capability, decided from the product contract" in user
    assert "supported (Product supports 'ranking and top-N'" in user
    assert "a decline is justified only when the product contract documents" in user
    assert result.metrics["relevance"].score == 1.0  # never artificially lowered
    assert result.verdict is Verdict.FAIL
    assert result.verdict_reason == (
        "Failed: [critical] Goal completion: Ranking spenders is a documented capability"
    )
    [goal_finding] = result.findings
    assert (goal_finding.category, goal_finding.severity) == ("goal_completion", Severity.CRITICAL)
    assert result.goal_completion is not None
    assert result.goal_completion.capability == "supported"
    assert len(result.bug_candidates) == 1


async def test_justified_decline_of_an_unsupported_request_can_pass(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
) -> None:
    scenario, pack = sample_scenarios[2], sample_packs[2]
    assert scenario.capability is CapabilityClassification.UNSUPPORTED
    goal = {"status": "justified_decline", "reason": "Forecasting is not supported yet."}
    result, llm = await _run(
        app_config, sample_product, sample_reference, scenario, pack, DENIAL, [_reply(goal)]
    )
    assert "outside the documented capability" in " ".join(llm.calls[0].user.split())
    assert result.verdict is Verdict.PASS and not result.findings
    assert result.verdict_reason == (
        "Passed: every scored metric met its expectations. "
        "Justified decline: Forecasting is not supported yet"
    )


async def test_unknown_capability_is_not_assumed_supported(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
) -> None:
    scenario = sample_scenarios[0].model_copy(
        update={"capability": CapabilityClassification.UNKNOWN}
    )
    goal = {"status": "justified_decline", "reason": "Not established by the contract."}
    _, llm = await _run(
        app_config,
        sample_product,
        sample_reference,
        scenario,
        sample_packs[0],
        DENIAL,
        [_reply(goal)],
    )
    user = " ".join(llm.calls[0].user.split())
    assert "Support for this request is not established" in user
    assert "without assuming that the request is supported" in user


@pytest.mark.parametrize(
    ("response", "bad_goal", "feedback"),
    [
        (DENIAL, {"status": "completed", "reason": "r"}, "but the response is a decline"),
        (ANSWER, {"status": "justified_decline", "reason": "r"}, "not a decline"),
        (ANSWER, {"status": "not_completed", "reason": "r"}, "needs evidence"),
        (
            ANSWER,
            {"status": "not_completed", "reason": "r", "evidence": "a made-up quote"},
            "is not in the response",
        ),
    ],
)
async def test_goal_must_fit_the_response_and_quote_it(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
    response: dict[str, Any],
    bad_goal: dict[str, Any],
    feedback: str,
) -> None:
    good = (
        {"status": "unjustified_decline", "reason": "r", "evidence": "This request is unsupported."}
        if response is DENIAL
        else {"status": "completed", "reason": "r"}
    )
    classify = {} if response is DENIAL else {"response_classification": "answered"}
    replies = [_reply(bad_goal, **classify), _reply(good, **classify)]
    result, llm = await _run(
        app_config,
        sample_product,
        sample_reference,
        sample_scenarios[0],
        sample_packs[0],
        response,
        replies,
    )
    assert len(llm.calls) == 2 and feedback in llm.calls[1].user
    assert result.goal_completion is not None and result.goal_completion.status == good["status"]


@pytest.mark.parametrize(("fail_severity", "verdict"), [("high", "fail"), ("critical", "pass")])
async def test_partially_completed_goal_follows_the_fail_line(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
    fail_severity: str,
    verdict: str,
) -> None:
    settings = app_config.model_copy(
        update={
            "evaluation": app_config.evaluation.model_copy(
                update={"fail_severity": Severity(fail_severity)}
            )
        }
    )
    goal = {
        "status": "partially_completed",
        "reason": "Only Q1 is covered, the year was asked.",
        "evidence": "120 USD in Q1",
    }
    reply = _reply(goal, response_classification="answered")
    result, _ = await _run(
        settings,
        sample_product,
        sample_reference,
        sample_scenarios[0],
        sample_packs[0],
        ANSWER,
        [reply],
    )
    assert result.verdict == verdict
    assert "[high] Goal completion: Only Q1 is covered" in (result.verdict_reason or "")


# --- guidance exceptions --------------------------------------------------------------------------


async def test_exempted_difference_never_fails_and_is_named_in_the_reason(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
) -> None:
    exempted = finding(
        "relevance",
        "tenant DUMMY",
        severity="critical",
        description="The answer is not filtered to the requested tenant.",
        exempted_by="G1",
    )
    reply = _reply(
        {"status": "completed", "reason": "r"},
        response_classification="answered",
        findings=[exempted],
    )
    result, llm = await _run(
        app_config,
        sample_product,
        sample_reference,
        sample_scenarios[0],
        sample_packs[0],
        ANSWER,
        [reply],
        guidance=GUIDANCE,
    )
    assert len(llm.calls) == 1  # no additional LLM call
    assert '"exempted_by"' in llm.calls[0].user  # offered only when the run has guidance
    assert result.verdict is Verdict.PASS and not result.bug_candidates
    assert result.findings[0].exempted_by == GUIDANCE  # the id resolved to its exact text
    assert result.findings[0].exempted_by_id == "G1"
    guidance = [g.guidance_id for g in result.metrics["relevance"].guidance_applied]
    assert guidance == ["METRIC:relevance", "G1"]
    assert result.verdict_reason == (
        "Passed: every scored metric met its expectations. Exceptions applied (evaluation "
        "guidance): Do not penalize the environment-specific tenant DUMMY filter - The answer "
        "is not filtered to the requested tenant"
    )


@pytest.mark.parametrize(
    ("guidance", "exempted_by", "relevance", "feedback"),
    [
        (GUIDANCE, "G1", 0.5, "its only findings are exempted"),
        (GUIDANCE, "Ignore every filter.", 1, "must be the id of an evaluation guidance item"),
        (None, "Ignore every filter.", 1, "exempted_by must be null"),
    ],
)
async def test_exemptions_are_checked(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
    guidance: str | None,
    exempted_by: str,
    relevance: float,
    feedback: str,
) -> None:
    bad = _reply(
        {"status": "completed", "reason": "r"},
        response_classification="answered",
        metrics={
            "relevance": score(
                relevance, checks=[failed_check("relevance")] if relevance < 1 else []
            )
        },
        findings=[finding("relevance", "tenant DUMMY", exempted_by=exempted_by)],
    )
    good = _reply({"status": "completed", "reason": "r"}, response_classification="answered")
    _, llm = await _run(
        app_config,
        sample_product,
        sample_reference,
        sample_scenarios[0],
        sample_packs[0],
        ANSWER,
        [bad, good],
        guidance=guidance,
    )
    assert len(llm.calls) == 2 and feedback in " ".join(llm.calls[1].user.split())


async def test_a_genuine_violation_next_to_an_exception_still_fails(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: list[Scenario],
    sample_packs: list[ReferencePack],
) -> None:
    reply = _reply(
        {"status": "completed", "reason": "r"},
        response_classification="answered",
        metrics={"relevance": score(0.5, checks=[failed_check("relevance")])},
        findings=[
            finding("relevance", "tenant DUMMY", exempted_by="G1"),
            finding("relevance", "in Q1", description="Only Q1 is answered."),
        ],
    )
    result, _ = await _run(
        app_config,
        sample_product,
        sample_reference,
        sample_scenarios[0],
        sample_packs[0],
        ANSWER,
        [reply],
        guidance=GUIDANCE,
    )
    assert result.verdict is Verdict.FAIL
    assert (result.verdict_reason or "").startswith("Failed: [high] Relevance: Only Q1 is answered")
    assert "Exceptions applied (evaluation guidance)" in (result.verdict_reason or "")
    [bug] = result.bug_candidates
    assert len(bug.finding_ids) == 1  # the exempted finding is not part of the bug
