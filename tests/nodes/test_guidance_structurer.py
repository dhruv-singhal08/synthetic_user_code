"""Evaluation guidance -> verbatim, typed items G1, G2, ... by ONE call (user decision): no
information lost or added, checked by code; cached per guidance version by the pipeline."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from synthetic_user.core.exceptions import ArtifactIntegrityError, LLMOutputValidationError
from synthetic_user.core.guidance import parse_evaluation_guidance
from synthetic_user.guidance.structurer import GuidanceStructurer
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.models.run import EvaluationGuidance

TEXT = """# Exceptions
- Do not penalize the tenant DUMMY filter.

# Scoring
1. Semantic correctness: a CTE is as good as a subquery."""
METRICS = ["relevance", "semantic_correctness"]


def guidance(text: str = TEXT) -> EvaluationGuidance:
    parsed = parse_evaluation_guidance(text, source="guidance.md", max_chars=10_000)
    assert parsed is not None
    return parsed


def items(*entries: tuple[str, list[str], str]) -> dict[str, Any]:
    return {"items": [{"type": t, "metrics": m, "text": x} for t, m, x in entries]}


GOOD = items(
    ("exception", [], "# Exceptions\n- Do not penalize the tenant DUMMY filter."),
    (
        "interpretation_rule",
        ["semantic_correctness"],
        "# Scoring\n1. Semantic correctness: a CTE is as good as a subquery.",
    ),
)


async def test_items_are_verbatim_typed_and_located() -> None:
    llm = FakeLLMClient([GOOD])
    structured = await GuidanceStructurer(llm, logging.getLogger("t"), max_retries=2).structure(
        guidance(), METRICS
    )
    assert len(llm.calls) == 1
    assert "   2 | - Do not penalize the tenant DUMMY filter." in llm.calls[0].user
    assert [(i.guidance_id, i.type.value, i.metrics, i.location) for i in structured.items] == [
        ("G1", "exception", [], "guidance.md lines 1-2"),
        ("G2", "interpretation_rule", ["semantic_correctness"], "guidance.md lines 4-5"),
    ]
    assert structured.text == TEXT and structured.sha256 == guidance().sha256


async def test_lost_rewritten_or_invented_content_is_retried_then_refused() -> None:
    lossy = items(("exception", ["faithfulness"], "Never penalise DUMMY filters."))
    llm = FakeLLMClient([lossy, GOOD])
    structured = await GuidanceStructurer(llm, logging.getLogger("t"), max_retries=2).structure(
        guidance(), METRICS
    )
    feedback = " ".join(llm.calls[1].user.split())
    assert "item 1: its text is not an exact excerpt of the guidance" in feedback
    assert "item 1: unknown metric ids ['faithfulness']" in feedback
    assert "lines [1, 2, 4, 5] of the guidance are not in any item" in feedback
    assert len(structured.items) == 2

    with pytest.raises(LLMOutputValidationError):
        await GuidanceStructurer(
            FakeLLMClient([lossy] * 3), logging.getLogger("t"), max_retries=2
        ).structure(guidance(), METRICS)


async def test_list_markers_may_be_left_out_of_an_item() -> None:
    text = "- Approve tenant filters.\n- Ignore column order."
    llm = FakeLLMClient(
        [
            items(
                ("exception", [], "Approve tenant filters."),
                ("interpretation_rule", [], "Ignore column order."),
            )
        ]
    )
    structured = await GuidanceStructurer(llm, logging.getLogger("t"), max_retries=0).structure(
        guidance(text), METRICS
    )
    assert [i.location for i in structured.items] == ["guidance.md line 1", "guidance.md line 2"]


async def test_pipeline_structures_each_guidance_version_once(
    tmp_path: Any, app_config: Any
) -> None:
    from synthetic_user.adapters.base import UnavailableAdapter
    from synthetic_user.pipeline import STRUCTURED_GUIDANCE, SyntheticUserPipeline
    from synthetic_user.storage.json_store import JsonArtifactStore
    from synthetic_user.storage.runs import run_artifact

    store = JsonArtifactStore(tmp_path)
    run_id = "RUN_20261105_120000_abcd"
    llm = FakeLLMClient([GOOD])
    pipeline = SyntheticUserPipeline(
        app_config, llm, UnavailableAdapter(), store, logger=logging.getLogger("t")
    )
    first = await pipeline.structure_guidance(run_id, guidance())
    again = await pipeline.structure_guidance(run_id, guidance())  # cached: no second call
    assert first is not None and first == again
    assert len(llm.calls) == 1 and len(first.items) == 2
    assert store.exists(run_artifact(run_id, STRUCTURED_GUIDANCE))
    assert await pipeline.structure_guidance(run_id, None) is None


async def test_legacy_questions_are_refused_before_execution(
    tmp_path: Any, app_config: Any
) -> None:
    from synthetic_user.models.question import QuestionCase
    from synthetic_user.pipeline import _refuse_legacy

    legacy = QuestionCase.model_validate(
        {
            "question_id": "Q_001",
            "scenario_id": "SCN_001",
            "persona_id": "p",
            "mode": "ungrounded",
            "utterance": "u",
            "intended_goal": "g",
            "intent": {"task_type": "ranking"},
            "question_characteristics": {
                "style": "precise",
                "ambiguity_level": "low",
                "complexity": "simple",
            },
            "expectations": {"expected_behavior": "b", "success_criteria": {"general": ["c"]}},
        }
    )
    with pytest.raises(ArtifactIntegrityError, match="Regenerate the questions step"):
        _refuse_legacy([legacy])
