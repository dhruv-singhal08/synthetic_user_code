from __future__ import annotations

import logging
from typing import Any

import pytest

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import ArtifactIntegrityError
from synthetic_user.core.identifiers import SequentialIdGenerator
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.models.base import AmbiguityLevel
from synthetic_user.models.persona import ExpertiseLevel, Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionGenerationResult
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario, TaskType
from synthetic_user.nodes.question_generator import SURPLUS, VAGUE_NOT_ALLOWED, QuestionGenerator
from synthetic_user.prompts.question import QuestionPromptOptions, build_question_prompt


def q(
    utterance: str,
    style: str = "precise",
    ambiguity: str = "low",
    complexity: str = "simple",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "utterance": utterance,
        "intent": {"task_type": "ranking"},
        "question_characteristics": {
            "style": style,
            "ambiguity_level": ambiguity,
            "complexity": complexity,
        },
        "expectations": {
            "expected_behavior": "Rank the top spenders, keeping currencies separate.",
            "success_criteria": ["names the top spenders", "does not mix currencies"],
        },
        **extra,
    }


def batch(*questions: dict[str, Any]) -> dict[str, Any]:
    return {"questions": list(questions)}


def with_prefs(persona: Persona, **prefs: Any) -> Persona:
    """``persona`` with question_preferences changed (validated like a persona file)."""
    data = persona.model_dump(mode="json")
    data["question_preferences"] = {**data["question_preferences"], **prefs}
    return Persona.model_validate(data)


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
        reference: Reference,
        packs: list[ReferencePack],
        settings: AppConfig,
    ) -> None:
        self.persona, self.product, self.scenarios = persona, product, scenarios
        self.reference, self.packs, self.settings = reference, packs, settings

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
            self.reference,
            self.packs,
            count=count,
        )


@pytest.fixture
def env(
    sample_persona: Persona,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
    sample_reference: Reference,
    sample_packs: list[ReferencePack],
    app_config: AppConfig,
) -> Env:
    return Env(
        sample_persona,
        sample_product,
        sample_scenarios,
        sample_reference,
        sample_packs,
        app_config,
    )


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
    assert [x.reference_pack_id for x in result.questions[::2]] == ["RP_SCN_001", "RP_SCN_002"]
    assert result.questions[0].intended_goal == env.scenarios[0].business_goal
    assert result.questions[2].intended_goal == env.scenarios[1].business_goal
    assert not result.rejected
    assert {x.mode for x in result.questions} == {"grounded"}


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


# --- reference leaks -----------------------------------------------------------------------------


async def test_leak_retried_then_clean_passes(env: Env) -> None:
    llm = FakeLLMClient(
        [
            batch(q("Top 5 spenders in card_transactions?")),
            batch(q("Who are our top 5 card spenders?")),
        ]
    )
    result = await env.run(llm, count=1)
    assert [x.utterance for x in result.questions] == ["Who are our top 5 card spenders?"]
    assert result.questions[0].leak_warnings == []
    assert len(llm.calls) == 2
    assert "card_transactions" in llm.calls[1].user.split("previous output invalid:", 1)[1]


async def test_persistent_leak_rejected_text_unchanged(env: Env) -> None:
    leaky = "Sum amount from card_transactions by employee"
    llm = FakeLLMClient([batch(q(leaky), q("Who spends the most on cards?"))] * 3)
    result = await env.run(llm, count=2)
    assert len(llm.calls) == 3
    assert [x.utterance for x in result.questions] == ["Who spends the most on cards?"]
    assert result.rejected[0].payload["utterance"] == leaky
    assert result.rejected[0].reasons == ["reference leak: ['card_transactions']"]


async def test_leak_kept_as_warning_when_not_rejecting(env: Env) -> None:
    llm = FakeLLMClient([batch(q("Spend per currency_code last month?"))])
    result = await env.run(llm, count=1, reject_reference_leaks=False)
    assert result.questions[0].leak_warnings == ["mentions reference name 'currency_code'"]
    assert len(llm.calls) == 1


async def test_highly_technical_persona_may_use_identifiers(env: Env) -> None:
    expertise = env.persona.expertise.model_copy(update={"sql": ExpertiseLevel.HIGH})
    technical = env.persona.model_copy(update={"expertise": expertise})
    llm = FakeLLMClient([batch(q("Sum amount from card_transactions by full_name"))])
    result = await env.run(llm, count=1, persona=technical)
    assert len(llm.calls) == 1
    assert len(result.questions[0].leak_warnings) == 2


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
    config_driven = with_prefs(env.persona, ambiguity=None)  # persona leaves ambiguity to config
    result = await env.run(llm, persona=config_driven, allow_vague_questions=True)
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
    config_driven = with_prefs(env.persona, ambiguity=None)
    result = await env.run(llm, persona=config_driven, allow_vague_questions=False)
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


async def test_copying_a_typical_user_question_rejected(env: Env) -> None:
    typical = "What are the biggest customer pain points this quarter?"
    product = env.product.model_copy(update={"typical_user_questions": [typical]})
    llm = FakeLLMClient([batch(q("what are the biggest customer pain points this quarter"))])
    generator = QuestionGenerator(
        llm, env.settings, logging.getLogger("test"), id_generator=SequentialIdGenerator()
    )
    result = await generator.generate(
        env.persona, product, env.scenarios[:1], env.reference, env.packs, count=1
    )
    assert not result.questions
    assert result.rejected[0].reasons == ["near-duplicate of product typical user question"]


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


def test_prompt_contents(
    sample_persona: Persona,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
    sample_reference: Reference,
    sample_packs: list[ReferencePack],
) -> None:
    scenario = sample_scenarios[0]
    options = QuestionPromptOptions(
        count=3, ambiguity_level=AmbiguityLevel.HIGH, language="en", allow_vague_questions=False
    )
    _system, user = build_question_prompt(
        sample_persona, sample_product, scenario, (sample_packs[0], sample_reference), options
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
    assert "Plain language only: no internal identifiers" in user
    assert "Cheat sheet of this scenario (for YOUR understanding only" in user
    assert "Transaction amount in the transaction currency" in user  # copied by code
    assert sample_reference.render() not in user  # the cheat sheet, not the whole reference
    assert "merchant names cannot be joined in" in user  # the usage instructions
    assert "expected_behavior: 1-3 plain sentences" in user
    assert "success_criteria: 1-4 short" in user
    assert "acceptable_variations" not in user  # removed (D204)
    assert "The persona decides HOW the question is asked" in user
    assert "Do not reuse the wording of the scenario's business_goal" in user
    assert "JSON Schema" not in user and '"$defs"' not in user  # compact contract
    style_examples = user.split("Style examples", 1)[1].split("How to use the persona", 1)[0]
    for example in sample_persona.example_utterances:
        assert f"- {example}" in style_examples  # shown as style to match, not to copy
    assert "Copying or lightly rewording the persona's style examples" in user
    contract = user.split("## OUTPUT CONTRACT", 1)[1].split("## PROHIBITED", 1)[0]
    for forbidden in ('"question_id":', '"intended_goal":', '"leak_warnings":'):
        assert forbidden not in contract


# --- v2.1: success criteria ------------------------------------------------------------------


async def test_question_owns_its_expectations(env: Env) -> None:
    llm = FakeLLMClient([batch(q("Who are our biggest card spenders this quarter?"))])
    result = await env.run(llm, count=1)
    expectations = result.questions[0].expectations
    assert expectations.success_criteria == ["names the top spenders", "does not mix currencies"]
    assert expectations.success_criteria != env.scenarios[0].success_criteria
    assert expectations.expected_behavior.startswith("Rank the top spenders")


@pytest.mark.parametrize(
    "expectations",
    [
        {"expected_behavior": "x"},  # no criteria: never copied from the scenario
        {"expected_behavior": "x", "success_criteria": ["a", "b", "c", "d", "e"]},
        {"clarification": "conditional", "acceptable_behaviors": ["x"], "success_criteria": ["a"]},
    ],
)
async def test_bad_expectations_rejected(env: Env, expectations: dict[str, Any]) -> None:
    llm = FakeLLMClient([batch(q("Who spends the most on cards?", expectations=expectations))])
    result = await env.run(llm, count=1)
    assert not result.questions and len(result.rejected) == 1


def _ungrounded(scenario: Scenario) -> Scenario:
    return scenario.model_copy(
        update={
            "mode": GenerationMode.UNGROUNDED,
            "reference_id": None,
            "reference_pack_id": None,
        }
    )


async def _generate(
    env: Env,
    llm: FakeLLMClient,
    scenarios: list[Scenario],
    reference: Reference | None,
    **cfg: Any,
) -> QuestionGenerationResult:
    generator = QuestionGenerator(
        llm,
        _settings(env.settings, **cfg),
        logging.getLogger("t"),
        id_generator=SequentialIdGenerator(),
    )
    return await generator.generate(
        env.persona, env.product, scenarios, reference, env.packs, count=2
    )


async def test_ungrounded_questions_have_mode_and_no_pack(env: Env) -> None:
    llm = FakeLLMClient([batch(q("Who spends most on cards?"), q("Which team overspends?"))])
    result = await _generate(env, llm, [_ungrounded(env.scenarios[0])], None)
    assert [x.question_id for x in result.questions] == ["Q_001", "Q_002"]
    assert all(x.mode == "ungrounded" and x.reference_pack_id is None for x in result.questions)
    user = llm.calls[0].user
    assert "Reference pack" not in user and "Usage guide" not in user
    assert "No reference data is available" in user


async def test_leaks_only_checked_in_grounded_mode(env: Env) -> None:
    leaky = "What is the total amount by currency_code in card_transactions?"
    grounded = await _generate(
        env, FakeLLMClient([batch(q(leaky))] * 3), env.scenarios[:1], env.reference
    )
    assert not grounded.questions and "reference leak" in grounded.rejected[0].reasons[0]
    ungrounded = await _generate(
        env, FakeLLMClient([batch(q(leaky))]), [_ungrounded(env.scenarios[0])], env.reference
    )
    assert [x.leak_warnings for x in ungrounded.questions] == [[]]


async def test_mixed_modes_pack_only_for_grounded(env: Env) -> None:
    llm = FakeLLMClient([batch(q("Top spenders this quarter?")), batch(q("Monthly trend please?"))])
    scenarios = [env.scenarios[0], _ungrounded(env.scenarios[1])]
    result = await _generate(env, llm, scenarios, env.reference)
    assert [(x.mode, x.reference_pack_id) for x in result.questions] == [
        ("grounded", "RP_SCN_001"),
        ("ungrounded", None),
    ]


async def test_grounded_scenario_without_reference_or_pack_is_an_error(env: Env) -> None:
    with pytest.raises(ArtifactIntegrityError, match="RP_SCN_001 is not available"):
        await _generate(env, FakeLLMClient([]), env.scenarios[:1], None)
    other = env.reference.model_copy(update={"reference_id": "other-00000000"})
    with pytest.raises(ArtifactIntegrityError, match="was built from reference"):
        await _generate(env, FakeLLMClient([]), env.scenarios[:1], other)


async def test_chatbot_product_questions_generated(
    env: Env, sample_chatbot: ProductDescription
) -> None:
    scenario = _ungrounded(env.scenarios[0]).model_copy(
        update={
            "product_id": "sample_policy_chatbot",
            "business_goal": "Find out the hotel cap for a Paris business trip",
            "title": "Hotel limit for a Paris trip",
            "task_type": TaskType.EXPLANATION,
            "success_criteria": ["states the nightly hotel limit", "says if taxes are included"],
        }
    )
    llm = FakeLLMClient([batch(q("How much can I spend per night on a hotel in Paris?"))])
    generator = QuestionGenerator(
        llm, env.settings, logging.getLogger("t"), id_generator=SequentialIdGenerator()
    )
    result = await generator.generate(env.persona, sample_chatbot, [scenario], None, count=1)
    question = result.questions[0]
    assert question.intended_goal == scenario.business_goal
    system, user = llm.calls[0].system, llm.calls[0].user
    assert '"name": "Sample Travel & Expense Policy Assistant"' in user
    assert "SQL" not in system
    template = user.split("## CONSTRAINTS", 1)[1].split("## OUTPUT CONTRACT", 1)[0]
    assert "SQL" not in template


# --- persona schema v2: question_preferences drive generation (D200) --------------------------


async def test_persona_ambiguity_overrides_config(env: Env) -> None:
    vague = q("Is spend looking weird lately?", "vague", "medium")
    occasional = with_prefs(env.persona, ambiguity="occasional")
    allowed = await env.run(
        FakeLLMClient([batch(vague)]), persona=occasional, count=1, allow_vague_questions=False
    )
    assert [x.utterance for x in allowed.questions] == ["Is spend looking weird lately?"]
    never = with_prefs(env.persona, ambiguity="none")
    refused = await env.run(
        FakeLLMClient([batch(vague)]), persona=never, count=1, allow_vague_questions=True
    )
    assert not refused.questions and refused.rejected[0].reasons == [VAGUE_NOT_ALLOWED]


async def test_fixed_complexity_is_enforced_with_a_retry(env: Env) -> None:
    simple_only = with_prefs(env.persona, complexity="simple")

    def with_complexity(utterance: str, complexity: str | None) -> dict[str, Any]:
        question = q(utterance)
        if complexity:
            question["question_characteristics"]["complexity"] = complexity
        return question

    llm = FakeLLMClient(
        [
            batch(with_complexity("Who spent most last month?", "complex")),
            batch(with_complexity("Who spent most last month?", "complex")),
            batch(
                with_complexity("Who spent most last month?", "complex"),
                with_complexity("Which team spent most?", "simple"),
            ),
        ]
    )
    result = await env.run(llm, persona=simple_only, count=2)
    assert "persona asks for simple questions" in llm.calls[1].user  # retry feedback
    assert [x.utterance for x in result.questions] == ["Which team spent most?"]
    assert result.questions[0].question_characteristics.complexity == "simple"
    assert result.rejected[0].reasons == [
        "complexity complex: the persona asks for simple questions"
    ]


async def test_mixed_complexity_accepts_any(env: Env) -> None:
    question = q("Who spent most last month?")
    question["question_characteristics"]["complexity"] = "complex"
    result = await env.run(FakeLLMClient([batch(question, q("Top merchants?"))]))
    assert len(result.questions) == 2 and not result.rejected


def test_technical_knowledge_high_makes_the_persona_technical(sample_persona: Persona) -> None:
    data = sample_persona.model_dump(mode="json")
    assert not sample_persona.is_technical
    data["expertise"]["technical_knowledge"] = "high"
    assert Persona.model_validate(data).is_technical
