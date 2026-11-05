"""Assemble the Node 3 (question) prompt; the prompt text itself is in ``prompt.yaml``.

ONE scenario per prompt. Both modes: persona (read as prompts/persona.py says) + product +
scenario. The persona alone decides HOW the question is asked; the scenario says only WHAT the
user wants (D204). Grounded adds the scenario's cheat sheet (Reference Pack), the usage
instructions and - when a selected metric needs them - the authoritative tool definitions, for
the model's understanding only. Each question owns its Evaluation Context: the success
criteria (C1..Cn) and a metric_mapping entry for every selected metric that maps criteria (their
ids) or needs a question-specific expectation (its value); code adds the Reference Spec and the
rest of the mapping (context.py). Without a valid question the model returns a
generation_failure instead of fabricating requirements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from synthetic_user.models.base import AmbiguityLevel
from synthetic_user.models.metrics import ExpectationFieldType
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import (
    MAX_DELIVERABLES,
    MAX_SUCCESS_CRITERIA,
    DeliverableAction,
    MetricExpectationStatus,
    QuestionComplexity,
    QuestionStyle,
)
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.scenario import Scenario, TaskType
from synthetic_user.nodes.question.context import ContextRequest
from synthetic_user.prompts.contracts import choices, render_contract
from synthetic_user.prompts.persona import question_persona_block
from synthetic_user.prompts.templates import prompts_beside

PROMPTS = prompts_beside(__file__)
SYSTEM = PROMPTS.text("system")
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
_FIELD_PLACEHOLDERS = {
    ExpectationFieldType.STRING: "<text>",
    ExpectationFieldType.STRING_LIST: ["<item>"],
    ExpectationFieldType.BOOLEAN: "<true | false>",
}


@dataclass(frozen=True)
class QuestionPromptOptions:
    count: int
    ambiguity_level: AmbiguityLevel
    language: str
    allow_vague_questions: bool
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
    # True when this persona would normally use technical identifiers (e.g. a data engineer).
    technical_persona: bool = False
    # Utterances already generated for this scenario (diversity context, D239).
    previous_questions: tuple[str, ...] = ()


def question_example(request: ContextRequest) -> dict[str, Any]:
    example: dict[str, Any] = PROMPTS.data(
        "question_example",
        deliverable_actions=choices(DeliverableAction),
        task_types=choices(TaskType),
        styles=choices(QuestionStyle),
        ambiguity_levels=choices(AmbiguityLevel),
    )
    mapped = request.mapped
    if mapped:
        context = example["questions"][0]["evaluation_context"]
        expectation = PROMPTS.data(
            "metric_expectation_example", statuses=choices(MetricExpectationStatus)
        )
        criteria_ids = {d.metric_id for d in request.criteria_metrics}
        requested_ids = {d.metric_id for d in request.requested}
        mapping: dict[str, Any] = {}
        for m in mapped:
            entry: dict[str, Any] = (
                dict(PROMPTS.data("metric_mapping_example"))
                if m.metric_id in criteria_ids
                else {"criteria": []}
            )
            entry["expectation"] = (
                {
                    "status": expectation["status"],
                    "value": {
                        f.name: _FIELD_PLACEHOLDERS[f.type] for f in m.question_expectation_schema
                    },
                    "reason": expectation["reason"],
                }
                if m.metric_id in requested_ids
                else None
            )
            mapping[m.metric_id] = entry
        context["metric_mapping"] = mapping
    return example


def _allowed(flag: bool) -> str:
    return PROMPTS.text("allowed" if flag else "not_allowed")


def _metric_rules(request: ContextRequest) -> str:
    mapped = request.mapped
    if not mapped:
        return PROMPTS.text("no_metric_rules")
    criteria_ids = {d.metric_id for d in request.criteria_metrics}
    requested_ids = {d.metric_id for d in request.requested}
    requirements = "\n".join(
        PROMPTS.text(
            "metric_requirement",
            metric_id=m.metric_id,
            display_name=m.display_name,
            criteria_rule=(
                PROMPTS.text("metric_criteria_rule", basis=m.basis)
                if m.metric_id in criteria_ids
                else PROMPTS.text("metric_no_criteria_rule")
            ),
            expectation_rule=(
                PROMPTS.text(
                    "metric_expectation_rule",
                    instructions=m.generation_instructions or m.description,
                    fields="; ".join(
                        PROMPTS.text(
                            "metric_field",
                            name=f.name,
                            type=f.type.value,
                            required=PROMPTS.text(
                                "field_required" if f.required else "field_optional"
                            ),
                            description=f.description,
                        )
                        for f in m.question_expectation_schema
                    ),
                )
                if m.metric_id in requested_ids
                else ""
            ),
        )
        for m in mapped
    )
    return PROMPTS.text("metric_rules", metrics=requirements)


def _diversity(options: QuestionPromptOptions) -> str:
    """Diversity rules for the batch (several questions) or the single question, plus the
    scenario's earlier questions as context when there are any (same call, D239)."""
    text = (
        PROMPTS.text("diversity_multi", count=options.count)
        if options.count > 1
        else PROMPTS.text("diversity_single")
    )
    if options.previous_questions:
        text += PROMPTS.text(
            "diversity_previous",
            count=options.count,
            previous=json.dumps(list(options.previous_questions), indent=2, ensure_ascii=False),
        )
    return text


def build_question_prompt(
    persona: Persona,
    product: ProductDescription,
    scenario: Scenario,
    grounding: tuple[ReferencePack, Reference] | None,
    options: QuestionPromptOptions,
    request: ContextRequest | None = None,
    *,
    tool_definitions: Reference | None = None,
) -> tuple[str, str]:
    """Return (system, user) asking for ``options.count`` questions for ``scenario``.

    ``grounding`` (pack + reference) is given for grounded scenarios, None for ungrounded.
    ``request`` = what the metric_mapping must hold for the run's selected metrics.
    ``tool_definitions`` = the reference whose agent_context a requested metric needs.
    """
    request = request or ContextRequest()
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
    tools = (
        PROMPTS.text("grounded_tools", tools=tool_definitions.render_agent_context())
        if tool_definitions is not None and tool_definitions.agent_context is not None
        else ""
    )
    if grounding is not None:
        pack, reference = grounding
        reference_block = PROMPTS.text(
            "grounded_reference",
            pack=pack.render(),
            instructions=reference.instructions_text(),
            tools=tools,
        )
        missing = PROMPTS.text("grounded_missing", failure=PROMPTS.text("generation_failure"))
    else:
        reference_block = tools
        missing = PROMPTS.text("ungrounded_missing", failure=PROMPTS.text("generation_failure"))

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
        metric_rules=_metric_rules(request),
        diversity=_diversity(options),
        contract=render_contract(question_example(request)),
        missing=missing,
    )
    return SYSTEM, user
