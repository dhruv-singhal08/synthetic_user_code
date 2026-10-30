"""Prompt for Node 3: natural-language questions for one scenario (any product type).

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

from synthetic_user.golden_set import golden_inspiration, golden_set_summary
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

SYSTEM = (
    "You role-play a real user typing questions into a conversational product, in their own "
    "words. You never expose internal identifiers, IDs, code or query language unless the "
    "persona would naturally do so. You reply with a single JSON object and nothing else."
)

_STYLE_MIX = {
    AmbiguityLevel.LOW: "Mostly precise questions; at most one vague or investigative one.",
    AmbiguityLevel.MEDIUM: "A mix of precise, vague and investigative questions.",
    AmbiguityLevel.HIGH: "Mostly vague or investigative questions that need interpretation "
    "or clarification.",
}

QUESTION_EXAMPLE: dict[str, Any] = {
    "questions": [
        {
            "utterance": "<the question exactly as this persona would type it>",
            "plan": {
                "deliverables": [
                    {
                        "action": choices(DeliverableAction),
                        "subject": "<one output the product must return>",
                        "sources": ["<product data source, when needed>"],
                        "limit": "<positive integer or null>",
                    }
                ],
                "filters": [{"name": "<filter name>", "value": "<filter value>"}],
                "is_follow_up": False,
            },
            "intent": {"task_type": choices(TaskType)},
            "question_characteristics": {
                "style": choices(QuestionStyle),
                "ambiguity_level": choices(AmbiguityLevel),
            },
            "expectations": {
                "expected_behavior": "<1-3 sentences: what the product should do for this "
                "question>",
                "success_criteria": [f"<1-{MAX_SUCCESS_CRITERIA} short, checkable must-haves>"],
                "expected_tools": ["<tool capability expected, or empty>"],
                "expected_journey": "<journey/domain scope expected, or null>",
                "expected_time_period": "<time period/duration expected, or null>",
                "expected_data_sources": ["<data source expected, or empty>"],
                "expected_context": ["<evidence/context expected, or empty>"],
                "prohibited_behaviors": ["<observable behavior that must not occur, or empty>"],
                "reference_context": [],
                "golden_set_criteria": {},
            },
        }
    ]
}


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
    fields = (
        "case_id",
        "scenario_type",
        "question",
        "expected_behavior",
        "success_criteria",
        "uploaded_criteria",
    )
    examples = golden_inspiration(
        golden_set,
        query=" ".join((scenario.business_goal, scenario.question_intent, scenario.title)),
        limit=options.agent_golden_examples,
    )
    patterns = [{field: example[field] for field in fields} for example in examples]
    return f"""
Optional SME golden-set coverage:
{json.dumps(golden_set_summary(golden_set), indent=2)}

Relevant intent and evaluation patterns (INSPIRATION only):
{json.dumps(patterns, indent=2)}

Use these patterns together with Persona, Product and the authoritative Reference to make the
new question realistic and its expected behavior and success criteria specific. Do not replay or
lightly reword any uploaded question or expected outcome. Reference data overrides any conflict.
"""


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
    product_json = product.model_dump(
        mode="json",
        include={
            "name",
            "description",
            "data_sources",
            "supported",
            "not_supported",
            "known_limitations",
            "user_visible_capabilities",
            "expected_outputs",
        },
    )
    scenario_json = scenario.model_dump(
        mode="json",
        include={
            "business_goal",
            "title",
            "typical_task",
            "task_type",
            "capability",
            "capability_rationale",
            "success_criteria",
            "question_intent",
        },
    )
    vague_rule = (
        "Vague questions are allowed."
        if options.allow_vague_questions
        else 'Do NOT write vague questions (style "vague" is not allowed).'
    )
    identifiers_rule = (
        "This persona is technical: internal names or identifiers are acceptable where they "
        "would naturally use them, but plain language is still preferred."
        if options.technical_persona
        else "Plain language only: no internal identifiers, IDs, field or table names, "
        "snake_case words, code or query language."
    )
    ambiguity_rule = (
        f'level "{options.ambiguity_level.value}" - {_STYLE_MIX[options.ambiguity_level]} '
        f"{vague_rule}"
    )
    persona_block = question_persona_block(persona, ambiguity_rule=ambiguity_rule)

    if evaluation_profile is EvaluationProfile.AGENT:
        if agent_reference is not None and agent_reference.agent_context is not None:
            operational = agent_reference.render_agent_context()
        else:
            operational = "No structured Agent context was supplied."
        reference_block = f"""\
Agent operational reference (AUTHORITATIVE):
<<<AGENT_CONTEXT
{operational}
AGENT_CONTEXT>>>
"""
        if golden_set is not None:
            inspiration = golden_inspiration(
                golden_set,
                query=" ".join((scenario.business_goal, scenario.question_intent, scenario.title)),
                limit=options.agent_golden_examples,
            )
            reference_block += f"""
Golden-set coverage across all uploaded examples:
{json.dumps(golden_set_summary(golden_set), indent=2)}

Relevant SME examples (INSPIRATION only):
{json.dumps(inspiration, indent=2)}

The uploaded examples teach valid intent patterns and how detailed an evaluation contract should
be. Generate a NEW question and a NEW contract for this scenario. Never copy or lightly reword a
golden question, expected behaviour, success criterion, or planner output.
"""
        missing = (
            "The operational reference is the source of truth. If it does not establish a needed "
            "tool or source, revise the question to use supported capabilities rather than "
            "inventing an implementation detail."
        )
    elif grounding is not None:
        pack, reference = grounding
        reference_block = f"""
Cheat sheet of this scenario (for YOUR understanding only - never use its names):
{pack.render()}

Usage instructions of the reference (for YOUR understanding):
{reference.instructions_text()}
{_golden_question_guidance(golden_set, scenario, options)}
"""
        missing = (
            "If the cheat sheet shows the goal depends on something ambiguous, say in "
            "expected_behavior\nwhether the product should ask or state its assumption. Do not "
            "guess silently."
        )
    else:
        reference_block = ""
        missing = (
            "No reference data is available: do not invent names, fields or numbers. If the "
            "goal could\nbe read several ways, say in expected_behavior which readings are fine "
            "and whether the\nproduct should ask for clarification."
        )

    if scenario.capability is CapabilityClassification.SUPPORTED:
        operational_minimum = f"""- COMPLEXITY FLOOR FOR THIS SUPPORTED AGENT SCENARIO: every
  question must genuinely use at least {options.agent_min_data_sources} distinct Agent data
  source(s) and require at least {options.agent_min_tool_calls} operational tool call(s). Repeating
  a canonical tool name is valid only for separate calls with different sources or purposes.
  Record every call in expected_tools, in execution order, and every source in
  expected_data_sources. Do not pad either list."""
    else:
        operational_minimum = (
            "- This scenario is unsupported: do not invent tool calls or data sources merely to "
            "meet a complexity target. The expected behaviour should explain the limitation."
        )
    agent_golden_rule = (
        "- golden_set_criteria: adapt relevant organization-specific criterion LABELS from the "
        "SME examples to this new question. Write new values; never copy an example value "
        "blindly."
        if golden_set is not None
        else "- golden_set_criteria must be {} because no golden set was supplied."
    )
    profile_rules = (
        f"""- Agent answer key: expected_behavior and success_criteria must completely describe
  the observable answer/action required by the exact question. Do not include tables, columns,
  fields, joins, SQL, or physical-schema requirements.
{operational_minimum}
- expected_tools: list the exact canonical tool name from AGENT_CONTEXT for every expected call,
  in execution order. Repeat a tool name when separate calls are expected. Never use a capability
  phrase or a golden-set label as a tool name.
- expected_data_sources: list exact canonical source names from AGENT_CONTEXT. These may differ
  from the user-facing labels used in plan.deliverables[].sources.
- expected_journey and expected_time_period: capture the exact scope needed by this new question,
  or null when absent. expected_context describes the evidence the tool results should expose.
- prohibited_behaviors: include only behaviours that would materially violate this question or
  an applicable reference rule.
{agent_golden_rule}
- reference_context must be []; code fills it from the canonical tools and sources after
  validating the generated contract against Reference data."""
        if evaluation_profile is EvaluationProfile.AGENT
        else ""
    )
    golden_rules = (
        "- The optional golden set is guidance, not a replay list. Use relevant patterns to "
        "refine expected_behavior and success_criteria, while keeping the new question and "
        "contract distinct from every uploaded example."
        if golden_set is not None
        else ""
    )

    user = f"""## TASK
Write {options.count} question(s) this persona would type into the product to pursue the
scenario's business goal. Language: {options.language}.

## INPUT
{persona_block}
Product:
{json.dumps(product_json, indent=2)}

Scenario (WHAT the user wants):
{json.dumps(scenario_json, indent=2)}
{reference_block}
## CONSTRAINTS
- HARD POLICY (overrides persona preferences): at most {options.max_deliverables} deliverables,
  {options.max_sources} distinct data source(s), {options.max_words} words, and
  {options.max_success_criteria} success criteria. Allowed computed complexities:
  {", ".join(c.value for c in options.allowed_complexities)}. Follow-ups are
  {"allowed" if options.allow_follow_ups else "not allowed"}; cross-source comparison is
  {"allowed" if options.allow_cross_source_comparison else "not allowed"}.
- plan.deliverables is the complete machine-checkable list of outputs requested by the utterance.
  Each item has exactly one action and one subject. Do not combine actions such as "rank and quote"
  in one item. Give each item only the source(s) it actually needs. Code counts this list.
- A separately requested list, metric, comparison, ranking, trend, breakdown, overlap analysis,
  explanation, recommendation or set of quotes is one deliverable. The same output requested
  separately for two sources counts as two. Journey, period and segment restrictions belong in
  plan.filters and do not count as deliverables.
- Do not hide extra outputs in a subject. "Themes with representative comments" is two
  deliverables (rank themes + quote comments). "Complaint volume and top categories" is two
  deliverables (aggregate volume + rank categories), even when both use one source. When the
  hard policy permits only one deliverable, ask for only one of those outputs.
- Set plan.is_follow_up honestly. The utterance must not ask for anything absent from the plan.
- Example of a prohibited five-deliverable question: asking for complaint themes, CSAT themes,
  overlapping theme names, complaint quotes and CSAT quotes in one response. Generate only a
  focused subset that fits the configured limit.
- The persona decides HOW the question is asked: wording, tone, length, complexity and
  ambiguity come only from "How to use the persona". The scenario only says WHAT is wanted.
- Do not reuse the wording of the scenario's business_goal, title or question_intent; write
  the question as this person would type it.
- {identifiers_rule}
- Set question_characteristics style and ambiguity_level honestly. Code computes complexity from
  the structured plan; do not output complexity.
- expected_behavior: 1-3 plain sentences: what the product should do for THIS question (what
  it should return or do; whether asking for clarification or declining would be right or
  wrong). If the question can fairly be read in more than one way, say which readings are
  acceptable.
- expected_behavior and success_criteria must respect the product's expected_outputs and scope.
  Do not require execution, returned data values, or another artifact that the product says it
  does not provide. When the product returns an artifact that computes the requested answer,
  require that artifact to be complete and valid for the request.
- success_criteria: 1-{options.max_success_criteria} short, checkable must-haves of a goal-complete
  answer, refined from the scenario's success_criteria to this exact question. Every criterion
  must be entailed by this exact question, its structured plan, the product answer contract or a
  necessary rule in the supplied reference. A generally useful practice is not automatically a
  must-have. Word each criterion so that every acceptable reading passes it.
{profile_rules}
{golden_rules}
- A ranking without a numeric plan.deliverables[].limit requires ordering, not an invented top-N
  cutoff. Never require LIMIT, top N, top 5, top 10 or another result count unless that exact
  limit is requested by the utterance and recorded in the plan.
- Do not add a currency, unit, conversion, precision, filter, grouping, dimension or output-format
  requirement that this question does not request. If a reference rule is necessary only under a
  condition not established by the question (for example, unnormalised multiple currencies),
  write it explicitly as a conditional criterion beginning with "If ..."; never make it an
  unconditional reason to fail.
- Negative criteria are allowed only when their prohibited behaviour would make the requested
  result wrong. Do not add generic defensive checks merely because they sound prudent.
- A sensitive identifier used only inside COUNT or COUNT(DISTINCT) is not exposed by the result.
  A privacy criterion may prohibit returning, grouping, or displaying raw identifier values, but
  must not prohibit using a documented identifier internally to calculate a requested aggregate.
- The intended goal is the scenario's business_goal (code copies it; do not restate it).
- Questions must differ meaningfully from each other.

## OUTPUT CONTRACT
{render_contract(QUESTION_EXAMPLE)}

## PROHIBITED
- Copying or lightly rewording the persona's style examples.
- ids, goals, leak warnings, timestamps (code adds these).
- Any text outside the JSON object.

## MISSING INFO
{missing}
"""
    return SYSTEM, user
