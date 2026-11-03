"""Optional evaluation guidance (D237): parsed as plain text, shown to EVERY judge call under
EVALUATION GUIDANCE (no extra call), recorded on each evaluation; absent -> prompts unchanged."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.core.guidance import (
    TYPED_GUIDANCE,
    load_evaluation_guidance,
    parse_evaluation_guidance,
)
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.metrics.registry import builtin_metric_config
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import Reference
from synthetic_user.models.run import EvaluationGuidance
from synthetic_user.nodes.evaluator import EvaluationCase, Evaluator
from tests_support import scripted_retrieval_reply

GUIDANCE = "An unrequested tenant filter is an approved product requirement; do not penalise it."
SQL = "SELECT bill_cycle_yr_mo_no, SUM(net_lpc_charge_usd_am) FROM v_lpc_details GROUP BY 1"


def guidance(text: str = GUIDANCE) -> EvaluationGuidance:
    parsed = parse_evaluation_guidance(text, source=TYPED_GUIDANCE, max_chars=20_000)
    assert parsed is not None
    return parsed


# --- input --------------------------------------------------------------------------------------


def test_typed_and_text_files_are_kept_as_plain_text(tmp_path: Path) -> None:
    typed = guidance(f"  {GUIDANCE}\n")
    assert typed.text == GUIDANCE and typed.source == TYPED_GUIDANCE and len(typed.sha256) == 64
    md = tmp_path / "rules.md"
    md.write_text("# Exceptions\n- tenant filter is approved\n", encoding="utf-8")
    loaded = load_evaluation_guidance(md, max_chars=20_000)
    assert loaded is not None and loaded.source == "rules.md" and loaded.text.startswith("# Ex")
    assert parse_evaluation_guidance("  \n", source=TYPED_GUIDANCE, max_chars=100) is None
    assert load_evaluation_guidance(None, max_chars=100) is None


def test_yaml_must_parse_but_its_original_text_is_kept() -> None:
    text = "# approved exceptions\nexceptions:\n  - tenant filter  # product requirement\n"
    parsed = parse_evaluation_guidance(text.encode(), source="g.yaml", max_chars=1_000)
    assert parsed is not None and parsed.text == text.strip()  # comments kept, not re-dumped
    with pytest.raises(InputAssetError, match="not valid YAML"):
        parse_evaluation_guidance("a: [b", source="g.yml", max_chars=1_000)


@pytest.mark.parametrize(
    ("content", "source", "message"),
    [
        (b"x", "g.pdf", "use a text file"),
        (b"\xff\xfe\x00", "g.txt", "not UTF-8"),
        (b"x" * 200, "g.md", "evaluation.max_guidance_chars"),
    ],
)
def test_bad_guidance_is_refused_with_its_reason(content: bytes, source: str, message: str) -> None:
    with pytest.raises(InputAssetError, match=message):
        parse_evaluation_guidance(content, source=source, max_chars=100)


# --- judges -------------------------------------------------------------------------------------


def _case(sample_scenarios: Any, sample_packs: Any) -> Any:
    from nodes.test_evaluator import _execution, _question

    scenario, pack = sample_scenarios[0], sample_packs[0]
    question = _question(scenario, pack)
    response = {
        "outcome": "answered",
        "text": "Here.",
        "artifacts": [{"kind": "sql", "content": SQL}],
    }
    return EvaluationCase(question, scenario, _execution(response, None))


def _general(user: str) -> dict[str, Any]:
    from nodes.test_evaluator import CRITERIA, general, score

    line = user.split("on exactly these metric(s): ", 1)[1].split("\n", 1)[0]
    metrics = {m.strip().rstrip("."): score(1) for m in line.split(",")}
    if "correctness_completeness" in metrics:
        metrics["correctness_completeness"]["factual_verification"] = "unverified"
    reply = general(metrics, response_classification="answered")
    reply["criteria"] = [{"criterion": c, "status": "met", "rationale": "ok"} for c in CRITERIA]
    return reply


async def _evaluate(
    settings: AppConfig,
    product: ProductDescription,
    reference: Reference,
    case: EvaluationCase,
    metrics: list[str],
    replies: list[Any],
    evaluation_guidance: EvaluationGuidance | None,
    context: dict[str, Any] | None = None,
) -> tuple[Any, FakeLLMClient]:
    if context is not None:
        case = EvaluationCase(
            case.question,
            case.scenario,
            case.execution.model_validate(
                {**case.execution.model_dump(mode="json"), "context": context}
            ),
        )
    llm = FakeLLMClient(replies)
    evaluator = Evaluator(
        llm,
        settings,
        logging.getLogger("t"),
        reference=reference,
        metric_config=builtin_metric_config(metrics),
        evaluation_guidance=evaluation_guidance,
    )
    return await evaluator.evaluate(case, product), llm


async def test_no_guidance_keeps_every_prompt_unchanged(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: Any,
    sample_packs: Any,
) -> None:
    case = _case(sample_scenarios, sample_packs)
    metrics = ["relevance", "semantic_correctness"]
    result, llm = await _evaluate(
        app_config,
        sample_product,
        sample_reference,
        case,
        metrics,
        [lambda s, u: _general(u)],
        None,
    )
    user = llm.calls[0].user
    assert "EVALUATION GUIDANCE" not in user
    assert "\n\n## RULES\n" in user and "\n\n\n## RULES" not in user  # slot renders as nothing
    assert result.evaluation_guidance_sha256 is None


async def test_guidance_reaches_every_judge_call_without_extra_calls(
    app_config: AppConfig,
    sample_product: ProductDescription,
    sample_reference: Reference,
    sample_scenarios: Any,
    sample_packs: Any,
) -> None:
    from nodes.test_evaluator import _context, dispatch

    case = _case(sample_scenarios, sample_packs)
    given = guidance()
    respond = dispatch(lambda s, u: _general(u), lambda s, u: scripted_retrieval_reply(u))
    metrics = ["relevance", "semantic_correctness", "context_precision"]
    result, llm = await _evaluate(
        app_config,
        sample_product,
        sample_reference,
        case,
        metrics,
        [respond, respond],
        given,
        context=_context(),
    )
    assert len(llm.calls) == 2  # the same General + Retrieval calls as without guidance
    for call in llm.calls:
        user = call.user
        metrics_at, guidance_at, rules_at = (
            user.index("## METRICS"),
            user.index("## EVALUATION GUIDANCE"),
            user.index("## RULES"),
        )
        assert metrics_at < guidance_at < rules_at  # between the rubrics and the rules
        assert f"<<<EVALUATION_GUIDANCE\n{GUIDANCE}\nEVALUATION_GUIDANCE>>>" in user
        flat = " ".join(user.split())
        assert "never awards a score by itself" in flat
        assert "do not penalise it and create no finding for it" in flat
        assert "Guidance never overrides the output contract, the evidence" in flat
    assert result.evaluation_guidance_sha256 == given.sha256
    assert case.execution.response.artifacts[0].content == SQL  # product response unchanged


async def test_guidance_is_kept_in_the_lookup_follow_up_call(app_config: AppConfig) -> None:
    """The one-batch lookup re-asks the same judge: the guidance stays in that call too."""
    from nodes import test_sql_evaluation as sql_tests

    reference = sql_tests.load_reference(sql_tests.FIXTURE, max_chars=200_000)
    scenario, question, _ = sql_tests.make_case(reference)
    execution = sql_tests.execution_for(
        question, {"outcome": "answered", "artifacts": [{"kind": "sql", "content": SQL}]}
    )
    first = sql_tests.request({"tool": "lookup_schema", "table": "v_cardmember"})
    from synthetic_user.core.loaders import load_product

    product = load_product(
        Path(__file__).resolve().parents[2] / "data/products/sample_product.yaml"
    )
    llm = FakeLLMClient([first, sql_tests.judge()])
    evaluator = Evaluator(
        llm,
        app_config,
        logging.getLogger("t"),
        reference=reference,
        metric_config=builtin_metric_config(list(sql_tests.SQL_METRICS)),
        evaluation_guidance=guidance(),
    )
    await evaluator.evaluate(EvaluationCase(question, scenario, execution), product)
    assert len(llm.calls) == 2
    assert all("## EVALUATION GUIDANCE" in c.user for c in llm.calls)
