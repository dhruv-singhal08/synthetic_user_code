"""Persona schema v2 (D200): every persona field reaches the prompts with a rule for using it,
so changing a value changes what the LLM is asked to generate."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from synthetic_user.models.base import AmbiguityLevel
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.scenario import Scenario
from synthetic_user.prompts.question import QuestionPromptOptions, build_question_prompt
from synthetic_user.prompts.scenario import ScenarioPromptOptions, build_scenario_prompt

PERSONAS = Path(__file__).resolve().parents[1] / "data" / "personas"
OPTIONS = QuestionPromptOptions(
    count=1, ambiguity_level=AmbiguityLevel.LOW, language="en", allow_vague_questions=False
)


def _edit(persona: Persona, change: Callable[[dict[str, Any]], None]) -> Persona:
    data = persona.model_dump(mode="json")
    change(data)
    return Persona.model_validate(data)


def _scenario_prompt(persona: Persona, product: ProductDescription) -> str:
    options = ScenarioPromptOptions(
        count=3, include_unsupported_scenarios=True, diversity_preference="balanced"
    )
    return build_scenario_prompt(persona, product, None, options, avoid_signatures=[])[1]


def _question_prompt(persona: Persona, product: ProductDescription, scenario: Scenario) -> str:
    ungrounded = scenario.model_copy(
        update={"mode": "ungrounded", "reference_id": None, "reference_pack_id": None}
    )
    return build_question_prompt(persona, product, ungrounded, None, OPTIONS)[1]


def _set(path: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def change(data: dict[str, Any]) -> None:
        *parents, leaf = path.split(".")
        for key in parents:
            data = data[key]
        data[leaf] = value

    return change


# (field changed, new value, text that must then appear, in scenarios?, in questions?)
CHANGES = [
    ("identity.role", "Head of Treasury", "Head of Treasury", True, True),
    ("identity.description", "Runs the liquidity desk.", "Runs the liquidity desk.", True, True),
    ("expertise.domain", "low", '"domain": "low"', True, True),
    ("expertise.data_analysis", "high", '"data_analysis": "high"', True, True),
    ("expertise.notes", "Never reads dashboards.", "Never reads dashboards.", True, True),
    ("behavior.style", "chatty and informal", "chatty and informal", False, True),
    ("behavior.traits", ["always asks for the source"], "always asks for the source", True, True),
    ("typical_tasks", ["Audit hotel bookings"], "Audit hotel bookings", True, False),
    ("primary_motivation", "Cut travel cost by 10%.", "Cut travel cost by 10%.", True, True),
    ("decision_inputs", ["Supplier contracts"], "Supplier contracts", True, True),
    (
        "question_preferences.complexity",
        "complex",
        "every question is complex - several parts",
        False,
        True,
    ),
    (
        "question_preferences.follow_up_questions",
        True,
        "follow_up_questions is true",
        False,
        True,
    ),
    ("example_utterances", ["Yo, who blew the budget?"], "Yo, who blew the budget?", False, True),
]


@pytest.mark.parametrize(("path", "value", "expected", "in_scenarios", "in_questions"), CHANGES)
def test_changing_a_persona_value_changes_the_prompts(
    sample_persona: Persona,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
    path: str,
    value: Any,
    expected: str,
    in_scenarios: bool,
    in_questions: bool,
) -> None:
    changed = _edit(sample_persona, _set(path, value))
    scenario = sample_scenarios[0]
    before = (
        _scenario_prompt(sample_persona, sample_product),
        _question_prompt(sample_persona, sample_product, scenario),
    )
    after = (
        _scenario_prompt(changed, sample_product),
        _question_prompt(changed, sample_product, scenario),
    )
    assert expected not in before[0] and expected not in before[1]
    assert (expected in after[0]) is in_scenarios, "scenario prompt"
    assert (expected in after[1]) is in_questions, "question prompt"


def test_every_persona_field_has_a_usage_rule(
    sample_persona: Persona, sample_product: ProductDescription, sample_scenarios: list[Scenario]
) -> None:
    scenario_prompt = _scenario_prompt(sample_persona, sample_product)
    question_prompt = _question_prompt(sample_persona, sample_product, sample_scenarios[0])
    scenario_rules = scenario_prompt.split("How to use the persona:", 1)[1]
    question_rules = question_prompt.split("How to use the persona:", 1)[1]
    for key in (
        "typical_tasks:",
        "primary_motivation:",
        "decision_inputs:",
        "traits:",
        "expertise.levels:",
        "question_complexity:",
    ):
        assert f"- {key}" in scenario_rules, key
    for key in (
        "style:",
        "traits:",
        "expertise.levels:",
        "primary_motivation:",
        "decision_inputs:",
        "follow_up_questions is",
        "ambiguity:",
    ):
        assert f"- {key}" in question_rules, key
    assert "complexity is mixed: spread the scenarios" in scenario_rules
    assert "complexity is mixed: use the one that fits" in question_rules
    for prompt in (scenario_prompt, question_prompt):  # metadata only
        assert '"finance_controller"' not in prompt and '"version"' not in prompt


def test_ambiguity_rule_reaches_the_question_prompt(
    sample_persona: Persona, sample_product: ProductDescription, sample_scenarios: list[Scenario]
) -> None:
    options = QuestionPromptOptions(
        count=1, ambiguity_level=AmbiguityLevel.MEDIUM, language="en", allow_vague_questions=True
    )
    user = build_question_prompt(
        sample_persona, sample_product, sample_scenarios[0], None, options
    )[1]
    assert 'ambiguity: level "medium" - A mix of precise, vague and investigative' in user
    assert "Vague questions are allowed." in user


# --- schema ---------------------------------------------------------------------------------


def test_both_persona_files_load() -> None:
    owner = Persona.model_validate(
        yaml.safe_load((PERSONAS / "gcs_product_owner.yaml").read_text(encoding="utf-8"))
    )
    assert owner.expertise.levels()["product_management"] == "high"
    assert owner.question_preferences.follow_up_questions is True
    assert owner.question_preferences.ambiguity == "occasional"
    assert len(owner.typical_tasks) == 12 and len(owner.decision_inputs) == 9
    assert not owner.is_technical  # technical_knowledge medium, sql low
    finance = Persona.model_validate(
        yaml.safe_load((PERSONAS / "finance_controller.yaml").read_text(encoding="utf-8"))
    )
    assert finance.question_preferences.ambiguity == "none"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("expertise.data_analysis", "expert", "Input should be 'low', 'medium' or 'high'"),
        ("expertise.Data-Analysis", "high", "snake_case"),
        ("question_preferences.complexity", "hard", "Input should be"),
        ("question_preferences.ambiguity", "sometimes", "Input should be"),
        ("question_preferences.tone", "formal", "Extra inputs are not permitted"),
        ("decision_inputs", [""], "at least 1 character"),
    ],
)
def test_invalid_persona_values_are_rejected(
    sample_persona: Persona, path: str, value: Any, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _edit(sample_persona, _set(path, value))


def test_optional_blocks_have_defaults(sample_persona: Persona) -> None:
    def drop(data: dict[str, Any]) -> None:
        for key in ("decision_inputs", "question_preferences", "example_utterances"):
            del data[key]

    bare = _edit(sample_persona, drop)
    assert bare.decision_inputs == [] and bare.example_utterances == []
    prefs = bare.question_preferences
    assert (prefs.complexity, prefs.follow_up_questions, prefs.ambiguity) == ("mixed", False, None)


def test_product_fields_reach_the_prompts(sample_persona: Persona) -> None:
    import yaml

    from synthetic_user.models.execution import AnsweredResponse
    from synthetic_user.models.question import QuestionCase
    from synthetic_user.prompts.response_evaluation import build_response_evaluation_prompt

    voc = ProductDescription.model_validate(
        yaml.safe_load(
            (PERSONAS.parent / "products" / "voice_of_customer_agent.yaml").read_text("utf-8")
        )
    )
    scenario_prompt = _scenario_prompt(sample_persona, voc)
    for text in (
        "CSATResult",  # data_sources
        "single interface for customer insight discovery",  # business_problem
        "Which complaint categories are increasing most rapidly?",  # typical question
        "Executive summaries",  # expected_outputs
        "never copy them",
    ):
        assert text in scenario_prompt, text
    scenario = Scenario.model_validate(
        {
            "id": "SCN_001",
            "persona_id": "p",
            "product_id": voc.product_id,
            "mode": "ungrounded",
            "business_goal": "Find top pain points",
            "title": "Pain points",
            "task_type": "summarization",
            "capability": "supported",
            "capability_rationale": "supported",
            "success_criteria": ["names the top themes"],
            "question_intent": "top pain points",
            "signature": "s",
        }
    )
    question_prompt = _question_prompt(sample_persona, voc, scenario)
    assert "CSATResult" in question_prompt and "typical_user_questions" not in question_prompt
    question = QuestionCase.model_validate(
        {
            "question_id": "Q_001",
            "scenario_id": "SCN_001",
            "persona_id": "p",
            "mode": "ungrounded",
            "utterance": "Top pain points?",
            "intended_goal": "g",
            "intent": {"task_type": "summarization"},
            "question_characteristics": {
                "style": "precise",
                "ambiguity_level": "low",
                "complexity": "simple",
            },
            "expectations": {"expected_behavior": "b", "success_criteria": ["c"]},
        }
    )
    _system, judge = build_response_evaluation_prompt(
        voc, scenario, question, AnsweredResponse(text="Themes: delays.")
    )
    assert "should not be interpreted as confirmed causation" in judge  # known_limitations
    assert "expected to consist of: Executive summaries" in judge
    assert "under known_limitations" in judge
