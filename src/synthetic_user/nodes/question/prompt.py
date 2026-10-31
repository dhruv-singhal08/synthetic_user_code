"""Assemble the Node 3 (question) prompt; the prompt text itself is in ``prompt.yaml``.

Both modes: persona (read as prompts/persona.py says) + product + scenario. The persona alone
decides HOW the question is asked; the scenario says only WHAT the user wants (D204).
Grounded adds the scenario's cheat sheet (Reference Pack) and the usage instructions, for the
model's understanding only - reference names are never quoted. Each question owns its
expectations: expected_behavior (written) and 1-4 success_criteria (design v3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from synthetic_user.models.base import AmbiguityLevel
from synthetic_user.models.evaluation_profile import EvaluationProfile
from synthetic_user.models.golden_set import GoldenSet
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import (
    MAX_DELIVERABLES,
    MAX_SUCCESS_CRITERIA,
    DeliverableAction,
    QuestionComplexity,
    QuestionStyle,
)
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.scenario import CapabilityClassification, Scenario, TaskType
from synthetic_user.prompts.contracts import choices, render_contract
from synthetic_user.prompts.persona import question_persona_block
from synthetic_user.prompts.templates import prompts_beside
from synthetic_user.references.golden_set import golden_inspiration, golden_set_summary

PROMPTS = prompts_beside(__file__)
SYSTEM = PROMPTS.text("system")
QUESTION_EXAMPLE: dict[str, Any] = PROMPTS.data(
    "question_example",
    deliverable_actions=choices(DeliverableAction),
    task_types=choices(TaskType),
    styles=choices(QuestionStyle),
    ambiguity_levels=choices(AmbiguityLevel),
    max_success_criteria=MAX_SUCCESS_CRITERIA,
)
# Fields of the product description and the scenario shown to the question generator.
PRODUCT_FIELDS = {
    "name",
    "description",
    "data_sources",
    "supported",
    "not_supported",
    "known_limitations",
    "user_visible_capabilities",
    "expected_outputs",
}
SCENARIO_FIELDS = {
    "business_goal",
    "title",
    "typical_task",
    "task_type",
    "capability",
    "capability_rationale",
    "success_criteria",
    "question_intent",
}
# Fields of each golden-set example shown as inspiration (NL2SQL; Agent shows whole examples).
GOLDEN_EXAMPLE_FIELDS = (
    "case_id",
    "scenario_type",
    "question",
    "expected_behavior",
    "success_criteria",
    "uploaded_criteria",
)


@dataclass(frozen=True)
class QuestionPromptOptions:
    count: int
    ambiguity_level: AmbiguityLevel
    language: str
    allow_vague_questions: bool
    agent_golden_examples: int = 6
    max_deliverables: int = MAX_DELIVERABLES
    max_sources: int = 1
    max_success_criteria: int = MAX_SUCCESS_CRITERIA
    max_words: int = 40
    allowed_complexities: tuple[QuestionComplexity, ...] = (
        QuestionComplexity.SIMPLE,
        QuestionComplexity.MEDIUM,
    )
    allow_follow_ups: bool = False
    allow_cross_source_comparison: bool = False
    agent_min_tool_calls: int = 1
    agent_min_data_sources: int = 1
    # True when this persona would normally use technical identifiers (e.g. a data engineer).
    technical_persona: bool = False


def _golden_question_guidance(
    golden_set: GoldenSet | None,
    scenario: Scenario,
    options: QuestionPromptOptions,
) -> str:
    if golden_set is None:
        return ""
    examples = golden_inspiration(
        golden_set, query=_golden_query(scenario), limit=options.agent_golden_examples
    )
    patterns = [{field: example[field] for field in GOLDEN_EXAMPLE_FIELDS} for example in examples]
    return PROMPTS.text(
        "golden_guidance",
        summary=json.dumps(golden_set_summary(golden_set), indent=2),
        examples=json.dumps(patterns, indent=2),
    )


def _golden_query(scenario: Scenario) -> str:
    """The text golden-set examples are ranked against."""
    return " ".join((scenario.business_goal, scenario.question_intent, scenario.title))


def _allowed(flag: bool) -> str:
    return PROMPTS.text("allowed" if flag else "not_allowed")


def build_question_prompt(
    persona: Persona,
    product: ProductDescription,
    scenario: Scenario,
    grounding: tuple[ReferencePack, Reference] | None,
    options: QuestionPromptOptions,
    evaluation_profile: EvaluationProfile = EvaluationProfile.NL2SQL,
    agent_reference: Reference | None = None,
    golden_set: GoldenSet | None = None,
) -> tuple[str, str]:
    """Return (system, user) asking for ``options.count`` questions for ``scenario``.

    ``grounding`` (pack + reference) is given for grounded scenarios, None for ungrounded.
    """
    product_json = product.model_dump(mode="json", include=PRODUCT_FIELDS)
    scenario_json = scenario.model_dump(mode="json", include=SCENARIO_FIELDS)
    ambiguity_rule = PROMPTS.text(
        "ambiguity_rule",
        level=options.ambiguity_level.value,
        style_mix=PROMPTS.data("style_mix")[options.ambiguity_level.value],
        vague_rule=PROMPTS.text(
            "vague_allowed" if options.allow_vague_questions else "vague_not_allowed"
        ),
    )
    is_agent = evaluation_profile is EvaluationProfile.AGENT
    if is_agent:
        reference_block, missing = _agent_reference(scenario, options, agent_reference, golden_set)
    elif grounding is not None:
        pack, reference = grounding
        reference_block = PROMPTS.text(
            "grounded_reference",
            pack=pack.render(),
            instructions=reference.instructions_text(),
            golden_guidance=_golden_question_guidance(golden_set, scenario, options),
        )
        missing = PROMPTS.text("grounded_missing")
    else:
        reference_block, missing = "", PROMPTS.text("ungrounded_missing")

    user = PROMPTS.text(
        "core",
        count=options.count,
        language=options.language,
        persona=question_persona_block(persona, ambiguity_rule=ambiguity_rule),
        product=json.dumps(product_json, indent=2),
        scenario=json.dumps(scenario_json, indent=2),
        reference=reference_block,
        max_deliverables=options.max_deliverables,
        max_sources=options.max_sources,
        max_words=options.max_words,
        max_success_criteria=options.max_success_criteria,
        allowed_complexities=", ".join(c.value for c in options.allowed_complexities),
        follow_ups=_allowed(options.allow_follow_ups),
        cross_source_comparison=_allowed(options.allow_cross_source_comparison),
        identifiers_rule=PROMPTS.text(
            "identifiers_technical" if options.technical_persona else "identifiers_plain"
        ),
        profile_rules=_agent_rules(scenario, options, golden_set) if is_agent else "",
        golden_rules=PROMPTS.text("golden_rules") if golden_set is not None else "",
        contract=render_contract(QUESTION_EXAMPLE),
        missing=missing,
    )
    return SYSTEM, user


def _agent_reference(
    scenario: Scenario,
    options: QuestionPromptOptions,
    agent_reference: Reference | None,
    golden_set: GoldenSet | None,
) -> tuple[str, str]:
    """The Agent operational reference (+ golden-set inspiration) and the missing-info rule."""
    if agent_reference is not None and agent_reference.agent_context is not None:
        operational = agent_reference.render_agent_context()
    else:
        operational = PROMPTS.text("agent_no_context")
    block = PROMPTS.text("agent_reference", context=operational)
    if golden_set is not None:
        inspiration = golden_inspiration(
            golden_set, query=_golden_query(scenario), limit=options.agent_golden_examples
        )
        block += PROMPTS.text(
            "agent_inspiration",
            summary=json.dumps(golden_set_summary(golden_set), indent=2),
            examples=json.dumps(inspiration, indent=2),
        )
    return block, PROMPTS.text("agent_missing")


def _agent_rules(
    scenario: Scenario, options: QuestionPromptOptions, golden_set: GoldenSet | None
) -> str:
    """The Agent answer-key rules, with this scenario's operational complexity floor."""
    if scenario.capability is CapabilityClassification.SUPPORTED:
        minimum = PROMPTS.text(
            "agent_minimum_supported",
            min_data_sources=options.agent_min_data_sources,
            min_tool_calls=options.agent_min_tool_calls,
        )
    else:
        minimum = PROMPTS.text("agent_minimum_unsupported")
    return PROMPTS.text(
        "agent_rules",
        operational_minimum=minimum,
        golden_rule=PROMPTS.text(
            "agent_golden_rule" if golden_set is not None else "agent_no_golden_rule"
        ),
    )
