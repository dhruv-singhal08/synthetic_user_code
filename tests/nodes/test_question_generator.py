from __future__ import annotations

import logging
from typing import Any

import pytest

from synthetic_user.core.config import AppConfig
from synthetic_user.core.identifiers import SequentialIdGenerator
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.models.base import AmbiguityLevel
from synthetic_user.models.persona import ExpertiseLevel, Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionGenerationResult
from synthetic_user.models.reference import ReferenceCatalog
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.question_generator import SURPLUS, VAGUE_NOT_ALLOWED, QuestionGenerator
from synthetic_user.prompts.question import QuestionPromptOptions, build_question_prompt
from synthetic_user.references.snapshot import build_snapshot


def q(
    utterance: str, style: str = "precise", ambiguity: str = "low", **extra: Any
) -> dict[str, Any]:
    return {
        "utterance": utterance,
        "intent": {"task_type": "ranking", "metrics": ["spend"]},
        "question_characteristics": {"style": style, "ambiguity_level": ambiguity},
        "expectations": {
            "clarification": "conditional",
            "acceptable_behaviors": ["asks which currency to use"],
            "unacceptable_behaviors": ["sums across currencies silently"],
        },
        "interpretation_notes": ["spend could mean gross or net"],
        **extra,
    }


def batch(*questions: dict[str, Any]) -> dict[str, Any]:
    return {"questions": list(questions)}


def _settings(app_config: AppConfig, **question: Any) -> AppConfig:
    gen = app_config.generation
    return app_config.model_copy(
        update={
            "generation": gen.model_copy(
                update={"question": gen.question.model_copy(update=question)}
            )
        }
    )


class Env:
    def __init__(
        self,
        persona: Persona,
        product: ProductDescription,
        scenarios: list[Scenario],
        catalog: ReferenceCatalog,
        settings: AppConfig,
    ) -> None:
        self.persona, self.product, self.scenarios = persona, product, scenarios
        self.catalog, self.settings = catalog, settings

    async def run(
        self,
        llm: FakeLLMClient,
        *,
        scenarios: int = 1,
        count: int = 2,
        persona: Persona | None = None,
        **question: Any,
    ) -> QuestionGenerationResult:
        generator = QuestionGenerator(
            llm,
            _settings(self.settings, **question),
            logging.getLogger("test"),
            id_generator=SequentialIdGenerator(),
        )
        return await generator.generate(
            persona or self.persona,
            self.product,
            self.scenarios[:scenarios],
            self.catalog,
            count=count,
        )


@pytest.fixture
def env(
    sample_persona: Persona,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
    sample_catalog: ReferenceCatalog,
    app_config: AppConfig,
) -> Env:
    return Env(sample_persona, sample_product, sample_scenarios, sample_catalog, app_config)


# --- ids and goal ---------------------------------------------------------------------------


async def test_ids_and_goal_set_by_code(env: Env) -> None:
    llm = FakeLLMClient(
        [
            batch(
                q("Who are our biggest card spenders this quarter?"),
                q("Which staff spend most on travel?"),
            ),
            batch(
                q("How is monthly spend per category trending?"),
                q("Is software spend creeping up?"),
            ),
        ]
    )
    result = await env.run(llm, scenarios=2)
    assert [x.question_id for x in result.questions] == ["Q_001", "Q_002", "Q_003", "Q_004"]
    assert [x.scenario_id for x in result.questions] == ["SCN_001", "SCN_001", "SCN_002", "SCN_002"]
    assert {x.persona_id for x in result.questions} == {"finance_controller"}
    assert [x.reference_snapshot_id for x in result.questions[::2]] == [
        "SNAP_SCN_001",
        "SNAP_SCN_002",
    ]
    assert result.questions[0].intended_goal == env.scenarios[0].business_goal
    assert result.questions[2].intended_goal == env.scenarios[1].business_goal
    assert [s.snapshot_id for s in result.snapshots] == ["SNAP_SCN_001", "SNAP_SCN_002"]
    assert not result.rejected


async def test_llm_cannot_set_ids_or_goal(env: Env) -> None:
    llm = FakeLLMClient(
        [
            batch(
                q("Who spends the most?", question_id="Q_999"),
                q("Top spenders?", intended_goal="x"),
            )
        ]
    )
    result = await env.run(llm)
    assert not result.questions
    assert all("Extra inputs are not permitted" in r.reasons[0] for r in result.rejected)


async def test_stable_order_under_concurrency(env: Env) -> None:
    def respond(system: str, user: str) -> dict[str, Any]:
        for scenario in env.scenarios:
            if scenario.business_goal in user:
                return batch(q(f"Question about {scenario.title}"))
        raise AssertionError("unknown scenario")

    llm = FakeLLMClient([respond, respond, respond])
    result = await env.run(llm, scenarios=3, count=1)
    assert [x.scenario_id for x in result.questions] == ["SCN_001", "SCN_002", "SCN_003"]
    assert [x.question_id for x in result.questions] == ["Q_001", "Q_002", "Q_003"]


# --- schema leaks -----------------------------------------------------------------------------


async def test_leak_retried_then_clean_passes(env: Env) -> None:
    llm = FakeLLMClient(
        [
            batch(q("Top 5 spenders in card_transactions?")),
            batch(q("Who are our top 5 card spenders?")),
        ]
    )
    result = await env.run(llm, count=1)
    assert [x.utterance for x in result.questions] == ["Who are our top 5 card spenders?"]
    assert result.questions[0].schema_leak_warnings == []
    assert len(llm.calls) == 2
    assert "card_transactions" in llm.calls[1].user.split("previous output invalid:", 1)[1]


async def test_persistent_leak_rejected_text_unchanged(env: Env) -> None:
    leaky = "Sum amount from card_transactions by employee"
    llm = FakeLLMClient([batch(q(leaky), q("Who spends the most on cards?"))] * 3)
    result = await env.run(llm, count=2)
    assert len(llm.calls) == 3
    assert [x.utterance for x in result.questions] == ["Who spends the most on cards?"]
    assert result.rejected[0].payload["utterance"] == leaky
    assert result.rejected[0].reasons == ["schema leak: ['card_transactions']"]


async def test_leak_kept_as_warning_when_not_rejecting(env: Env) -> None:
    llm = FakeLLMClient([batch(q("Spend per currency_code last month?"))])
    result = await env.run(llm, count=1, reject_schema_leaks=False)
    assert result.questions[0].schema_leak_warnings == [
        "mentions catalog identifier 'currency_code'"
    ]
    assert len(llm.calls) == 1


async def test_highly_technical_persona_may_use_identifiers(env: Env) -> None:
    expertise = env.persona.expertise.model_copy(update={"sql": ExpertiseLevel.HIGH})
    technical = env.persona.model_copy(update={"expertise": expertise})
    llm = FakeLLMClient([batch(q("Sum amount from card_transactions by employee_id"))])
    result = await env.run(llm, count=1, persona=technical)
    assert len(llm.calls) == 1
    assert len(result.questions[0].schema_leak_warnings) == 2


# --- vague / ambiguity ------------------------------------------------------------------------


async def test_vague_question_needs_ambiguity(env: Env) -> None:
    llm = FakeLLMClient(
        [
            batch(
                q("Anything odd in spend?", "vague", "low"),
                q("Is spend looking weird lately?", "vague", "medium"),
            )
        ]
    )
    result = await env.run(llm, allow_vague_questions=True)
    assert [x.utterance for x in result.questions] == ["Is spend looking weird lately?"]
    assert result.questions[0].question_characteristics.ambiguity_level is AmbiguityLevel.MEDIUM
    assert result.rejected[0].reasons == ["vague question must have ambiguity_level medium or high"]


async def test_vague_rejected_when_not_allowed(env: Env) -> None:
    llm = FakeLLMClient(
        [
            batch(
                q("Is spend looking weird lately?", "vague", "high"),
                q("Why did travel spend jump in May?", "investigative", "medium"),
            )
        ]
    )
    result = await env.run(llm, allow_vague_questions=False)
    assert [x.question_characteristics.style for x in result.questions] == ["investigative"]
    assert result.rejected[0].reasons == [VAGUE_NOT_ALLOWED]


# --- dedupe -----------------------------------------------------------------------------------


async def test_near_duplicates_dropped_and_recorded(env: Env) -> None:
    llm = FakeLLMClient(
        [
            batch(
                q("Who are our top card spenders this quarter?"),
                q("Who are our top card spenders, this quarter?!"),
            ),
            batch(
                q("who are our TOP card spenders this quarter"),
                q("How is category spend trending?"),
            ),
        ]
    )
    result = await env.run(llm, scenarios=2)
    assert [x.utterance for x in result.questions] == [
        "Who are our top card spenders this quarter?",
        "How is category spend trending?",
    ]
    assert [r.reasons for r in result.rejected] == [
        ["near-duplicate of Q_001"],
        ["near-duplicate of Q_001"],
    ]
    assert [r.scenario_id for r in result.rejected] == ["SCN_001", "SCN_002"]


async def test_copying_persona_example_rejected(env: Env) -> None:
    example = env.persona.example_utterances[0]
    llm = FakeLLMClient([batch(q(example))])
    result = await env.run(llm, count=1)
    assert not result.questions
    assert result.rejected[0].reasons == ["near-duplicate of persona example utterance"]


# --- other ------------------------------------------------------------------------------------


async def test_surplus_recorded(env: Env) -> None:
    llm = FakeLLMClient(
        [batch(q("Who spends most?"), q("Which team spends most?"), q("Top merchants?"))]
    )
    result = await env.run(llm, count=2)
    assert len(result.questions) == 2
    assert result.rejected[0].reasons == [SURPLUS]


async def test_bad_json_records_scenario_rejection(env: Env) -> None:
    llm = FakeLLMClient(["no json", "still none", "nope"])
    result = await env.run(llm, count=1)
    assert not result.questions
    assert result.rejected[0].scenario_id == "SCN_001"
    assert result.rejected[0].payload == {}
    assert len(result.snapshots) == 1


def test_prompt_contents(
    sample_persona: Persona,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
    sample_catalog: ReferenceCatalog,
) -> None:
    scenario = sample_scenarios[0]
    snapshot = build_snapshot(sample_catalog, scenario.cheat_sheet, "SNAP_SCN_001")
    options = QuestionPromptOptions(
        count=3, ambiguity_level=AmbiguityLevel.HIGH, language="en", allow_vague_questions=False
    )
    _system, user = build_question_prompt(
        sample_persona, sample_product, scenario, snapshot, options
    )
    for section in (
        "## TASK",
        "## INPUT",
        "## CONSTRAINTS",
        "## OUTPUT CONTRACT",
        "## PROHIBITED",
        "## MISSING INFO",
    ):
        assert section in user
    assert scenario.business_goal in user
    assert sample_persona.behavior.style in user
    assert "Mostly vague or investigative" in user
    assert 'style "vague" is not allowed' in user
    assert "No table names, no column names" in user
    assert "several valid" in user
    for example in sample_persona.example_utterances:
        assert f"- {example}" in user.split("## PROHIBITED", 1)[1]
    contract = user.split("## OUTPUT CONTRACT", 1)[1].split("## PROHIBITED", 1)[0]
    for forbidden in ('"question_id":', '"intended_goal":', '"schema_leak_warnings":'):
        assert forbidden not in contract
