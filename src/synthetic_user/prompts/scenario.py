"""Prompts for Node 2 (scenarios): two builders sharing a common core.

COMMON (both modes): persona (prompts/persona.py) + product description + settings; business
goal first; realistic tasks this persona would do WITH THIS PRODUCT; capability with a
rationale citing the product description; 2-5 checkable success_criteria; a one-line
question_intent (never the wording); no user questions.
GROUNDED adds the reference (data_catalog or document_collection, rendered by code) + usage
instructions and asks for the scenario's cheat sheet (reference_pack): picks by exact name /
id that code checks and fills with the reference's own text (D203).
UNGROUNDED has no reference and no pack, and must not invent data structures or numbers.
The output contract is a compact JSON example (D204).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from synthetic_user.golden_set import golden_inspiration, golden_set_summary
from synthetic_user.models.evaluation_profile import EvaluationProfile
from synthetic_user.models.golden_set import GoldenSet
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import DataCatalog, Reference, ReferenceKind
from synthetic_user.models.scenario import CapabilityClassification, TaskType
from synthetic_user.prompts.contracts import choices, render_contract
from synthetic_user.prompts.persona import scenario_persona_block

SYSTEM = (
    "You design realistic test scenarios for a synthetic user who will later test a "
    "conversational product. You plan what the user wants to achieve with THIS product; you "
    "never write the user's questions. You reply with a single JSON object and nothing else."
)
GROUNDED_SYSTEM = SYSTEM + (
    " You ground every scenario in the reference supplied by the user and pick items from it "
    "by their exact names; you never invent tables, columns, relationships, documents, facts "
    "or rules it does not contain."
)
UNGROUNDED_SYSTEM = SYSTEM + (
    " No data schema is available: you never invent tables, fields, data structures or numbers."
)
AGENT_SYSTEM = SYSTEM + (
    " This is an Agent evaluation: work in user-visible product, tool, and data-source terms. "
    "Never request or emit database tables, columns, joins, SQL, or a reference_pack."
)

KIND_LABELS = {
    ReferenceKind.DATA_CATALOG: "data catalog",
    ReferenceKind.DOCUMENT_COLLECTION: "document collection",
}

# The cheat-sheet part of the output example, per reference format.
PACK_EXAMPLES: dict[ReferenceKind, dict[str, Any]] = {
    ReferenceKind.DATA_CATALOG: {
        "tables": [
            {
                "table": "<table name, exactly as in the catalog>",
                "columns": ["<column name of that table the answer needs>"],
                "why": "<why the goal needs it>",
            }
        ],
        "joins": [{"from": "<table.column>", "to": "<table.column>"}],
        "rules": ["<catalog rule id, e.g. R1>"],
    },
    ReferenceKind.DOCUMENT_COLLECTION: {
        "documents": [
            {"name": "<document name, exactly as listed>", "why": "<why the goal needs it>"}
        ]
    },
}

PACK_RULES = {
    ReferenceKind.DATA_CATALOG: """\
- reference_pack is this scenario's cheat sheet: the answer key the evaluator will use. Pick
  from the catalog by exact name; code copies the descriptions and rules in and drops any
  name the catalog does not have.
  - tables (1-4): every table a correct answer must use, with the columns it needs, and why.
    Include columns used only inside a required aggregate, including a sensitive identifier needed
    for COUNT(DISTINCT ...). Aggregate-only use does not expose the underlying identifier values.
  - joins (0-3): the listed relationships needed to combine those tables, "from" / "to"
    exactly as listed. Only listed relationships exist.
  - rules (0-4): ids of the catalog rules the answer must respect (table rules come with
    their table automatically).
  - Only what the goal really needs; what merely could help is not part of the answer key.
- A goal that needs a table, column or join the catalog does not list is unsupported or
  unknown: say what is missing in capability_rationale; never invent it.""",
    ReferenceKind.DOCUMENT_COLLECTION: """\
- reference_pack is this scenario's cheat sheet: the answer key the evaluator will use.
  - documents (1-3): the documents whose content answers the goal, names exactly as listed,
    and why. Code copies their summaries in and drops any name the collection does not have.
- Goals must be answerable from what the summaries say the documents contain; a goal no
  document covers is unsupported or unknown.
- success_criteria may name facts from the summaries the answer must get right.""",
}


@dataclass(frozen=True)
class ScenarioPromptOptions:
    count: int
    include_unsupported_scenarios: bool
    diversity_preference: str
    max_request_steps: int = 3
    agent_golden_examples: int = 12
    agent_min_tool_calls: int = 1
    agent_min_data_sources: int = 1


def scenario_example(kind: ReferenceKind | None) -> dict[str, Any]:
    scenario: dict[str, Any] = {
        "business_goal": "<what the persona wants to achieve, in business terms>",
        "title": "<short title>",
        "typical_task": "<one of the persona's typical_tasks, copied exactly, or null>",
        "task_type": choices(TaskType),
        "capability": choices(CapabilityClassification),
        "capability_rationale": "<cites the product description item it relies on>",
        "success_criteria": ["<2-5 short, checkable statements in business terms>"],
        "question_intent": "<one sentence: what the user will ask for - not the wording>",
    }
    if kind is not None:
        scenario["reference_pack"] = PACK_EXAMPLES[kind]
    return {"scenarios": [scenario]}


def _golden_scenario_guidance(golden_set: GoldenSet | None, options: ScenarioPromptOptions) -> str:
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
    examples = [
        {field: example[field] for field in fields}
        for example in golden_inspiration(golden_set, limit=options.agent_golden_examples)
    ]
    return f"""\
Optional SME golden set (INSPIRATION, not cases to replay):
Coverage across all uploaded examples:
{json.dumps(golden_set_summary(golden_set), indent=2)}

Representative intent and evaluation patterns:
{json.dumps(examples, indent=2)}

Use these patterns together with Persona, Product and the authoritative Reference to improve
coverage and success criteria. Create new business goals and intents; never copy or lightly
reword an uploaded question, expected behaviour, or success criterion.

"""


def build_scenario_prompt(
    persona: Persona,
    product: ProductDescription,
    reference: Reference | None,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str] = (),
    evaluation_profile: EvaluationProfile = EvaluationProfile.NL2SQL,
    golden_set: GoldenSet | None = None,
) -> tuple[str, str]:
    """Grounded prompt when a reference is given, else the ungrounded one."""
    if reference is None:
        return build_ungrounded_scenario_prompt(persona, product, options, avoid_signatures)
    if evaluation_profile is EvaluationProfile.AGENT:
        return build_agent_scenario_prompt(
            persona, product, reference, options, avoid_signatures, golden_set=golden_set
        )
    return build_grounded_scenario_prompt(
        persona,
        product,
        reference,
        options,
        avoid_signatures,
        golden_set=golden_set,
    )


def build_agent_scenario_prompt(
    persona: Persona,
    product: ProductDescription,
    reference: Reference,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str] = (),
    *,
    golden_set: GoldenSet | None = None,
) -> tuple[str, str]:
    """Grounded Agent prompt without physical schema or a scenario-level cheat sheet.

    A data catalog may contain an operational description of tools and data sources. Agent
    generation receives only that description and catalog-wide behavioural rules; table,
    column, and join metadata remain exclusive to NL2SQL.
    """
    content = reference.content
    if reference.agent_context is not None:
        rendered = reference.render_agent_context()
    elif isinstance(content, DataCatalog):
        description = content.description or "No additional operational description was supplied."
        description = description.split("\n\nTABLE DETAILS", 1)[0].strip()
        context_lines = [
            f"Operational reference: {content.name}",
            description,
        ]
        if content.rules:
            context_lines += ["", "Product/tool constraints:"]
            context_lines += [f"- {rule}" for rule in content.rules]
        rendered = "\n".join(context_lines)
    else:
        rendered = reference.render()
    inspiration_block = ""
    if golden_set is not None:
        inspiration_block = f"""
SME golden set (INSPIRATION, not test cases to replay):
Coverage across all uploaded cases:
{json.dumps(golden_set_summary(golden_set), indent=2)}

Representative examples:
{json.dumps(golden_inspiration(golden_set, limit=options.agent_golden_examples), indent=2)}

Use these examples to learn valid intents, evaluation detail, and realistic combinations. Create
NEW business goals and question intents. Do not copy or lightly reword a golden question, its
expected behaviour, or its success criteria.
"""
    reference_block = f"""Agent context (AUTHORITATIVE; no physical schema):
<<<AGENT_CONTEXT
{rendered}
AGENT_CONTEXT>>>
{inspiration_block}
"""
    golden_constraint = (
        "- Base goals and success_criteria on the product, operational context, and patterns in "
        "the optional golden set above.\n"
        "- Generate novel coverage beyond the uploaded examples; do not replay a golden-set case."
        if golden_set is not None
        else "- Base goals and success_criteria on the product and operational context."
    )
    constraints = f"""\
- There is no reference_pack for Agent scenarios. The evaluator will use each generated
  question's expected_behavior and success_criteria as the answer key.
- Use only user-visible product capabilities, tools, and data sources. Never mention or select
  database tables, columns, fields, joins, SQL, or other storage implementation details.
- Every supported goal must be achievable through at least one tool and data source in the Agent
  context.
{golden_constraint}
- For this run's selected complexity, every supported goal must genuinely require at least
  {options.agent_min_data_sources} distinct Agent data source(s) and
  {options.agent_min_tool_calls} operational tool call(s). A repeated tool is valid only when
  separate calls with different sources or purposes are actually needed. Keep source results
  separate when the operational rules prohibit blending them. Unsupported goals are exempt."""
    prohibited = (
        "- Database tables, columns, fields, joins, SQL, reference_pack, or physical-schema "
        "requirements."
    )
    missing = """\
If the product description and operational context do not establish that a goal is supported,
classify it as unknown or unsupported and explain the missing capability."""
    user = _core(
        persona,
        product,
        options,
        avoid_signatures,
        kind=None,
        reference=reference_block,
        constraints=constraints,
        prohibited=prohibited,
        missing=missing,
    )
    return AGENT_SYSTEM, user


def build_grounded_scenario_prompt(
    persona: Persona,
    product: ProductDescription,
    reference: Reference,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str] = (),
    *,
    golden_set: GoldenSet | None = None,
) -> tuple[str, str]:
    label = f'{KIND_LABELS[reference.kind]} "{reference.source_name}"'
    reference_block = f"""Reference ({label}; AUTHORITATIVE):
<<<REFERENCE
{reference.render()}
REFERENCE>>>

Usage instructions for the reference (follow them):
{reference.instructions_text()}

{_golden_scenario_guidance(golden_set, options)}
"""
    constraints = PACK_RULES[reference.kind] + (
        "\n- Base goals and success_criteria on what the reference actually contains."
        " Golden-set patterns are optional guidance and never override Reference data."
    )
    prohibited = (
        "- Tables, columns, relationships, documents, facts or rules that are not in the reference."
    )
    missing = """\
If the reference lacks something a goal needs, choose another goal, or classify the capability
as unknown and say what is missing in capability_rationale."""
    user = _core(
        persona,
        product,
        options,
        avoid_signatures,
        kind=reference.kind,
        reference=reference_block,
        constraints=constraints,
        prohibited=prohibited,
        missing=missing,
    )
    return GROUNDED_SYSTEM, user


def build_ungrounded_scenario_prompt(
    persona: Persona,
    product: ProductDescription,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str] = (),
) -> tuple[str, str]:
    constraints = """\
- There is NO reference: there is no reference_pack.
- Do not invent data structures, tables, fields, columns, IDs or numbers; describe goals and
  success_criteria at business level only (what the user must learn or be able to do)."""
    prohibited = "- Internal names (tables, fields, IDs), code, schemas, made-up figures."
    missing = """\
If the product description does not say whether something is supported, classify the
capability as unknown and say what is missing in capability_rationale."""
    user = _core(
        persona,
        product,
        options,
        avoid_signatures,
        kind=None,
        reference="",
        constraints=constraints,
        prohibited=prohibited,
        missing=missing,
    )
    return UNGROUNDED_SYSTEM, user


# --- common core --------------------------------------------------------------------------------


def _core(
    persona: Persona,
    product: ProductDescription,
    options: ScenarioPromptOptions,
    avoid_signatures: Sequence[str],
    *,
    kind: ReferenceKind | None,
    reference: str,
    constraints: str,
    prohibited: str,
    missing: str,
) -> str:
    product_json = product.model_dump(
        mode="json",
        include={
            "name",
            "description",
            "business_problem",
            "data_sources",
            "supported",
            "not_supported",
            "known_limitations",
            "user_visible_capabilities",
            "typical_user_questions",
            "expected_outputs",
        },
        exclude_none=True,
    )
    unsupported = (
        "Where the count allows, include scenarios the product does NOT support (see "
        "not_supported / known_limitations) and classify them as unsupported."
        if options.include_unsupported_scenarios
        else "Only produce scenarios the product supports."
    )
    avoid = "\n".join(f"- {s}" for s in avoid_signatures) or "- (none)"
    parts = "title, typical_task, task_type, capability with rationale, success_criteria, "
    parts += "question_intent" + (" and the cheat sheet (reference_pack)" if kind else "")
    return f"""## TASK
Write {options.count} distinct scenario(s) this persona would realistically pursue WITH THIS
PRODUCT ("{product.name}"). For each: the business goal FIRST, then
{parts}.

## INPUT
{scenario_persona_block(persona)}
Product:
{json.dumps(product_json, indent=2)}

{reference}Settings: count={options.count}; diversity_preference={options.diversity_preference!r}.

Existing scenario signatures to AVOID repeating:
{avoid}

## CONSTRAINTS
- Design each business goal so it can be asked and answered with no more than
  {options.max_request_steps} distinct deliverables. Outputs requested separately for different
  sources count separately. Do not bundle a chain of analyses, follow-up investigations,
  cross-source overlap analysis and supporting quotes into one scenario.
- business_goal is what the persona wants to achieve, in business terms. Write it first.
- The task must be something this persona would really do with this product; follow "How
  to use the persona" above.
- capability_rationale must cite the specific product description item (supported /
  not_supported / known_limitations) it relies on.
- Goals must be answerable from the product's data_sources (when listed); a goal that needs
  data the product does not have is unsupported.
- typical_user_questions show what real users ask: use them to choose realistic goals and to
  cover the product's main uses - never copy them (questions are written later).
- {unsupported}
- success_criteria: 2-5 short, checkable statements of what a goal-complete answer must
  contain (an evaluator will tick each one off), in business terms. Do not invent a result count,
  top-N cutoff, currency conversion, filter, dimension or format that the business_goal does not
  require. Express a reference-dependent safeguard as an explicit "If ..." condition when its
  triggering data state is not guaranteed.
- question_intent: WHAT the user will ask for, in one plain sentence. Never suggest wording,
  tone or phrasing: the question is written later, in the persona's own voice.
{constraints}
- Vary task types and goals; do not repeat an avoided signature.

## OUTPUT CONTRACT
{render_contract(scenario_example(kind))}

## PROHIBITED
- Writing the user's question, an example question or phrasing hints.
- ids, persona/product/reference ids, mode, signatures, timestamps.
{prohibited}
- Any text outside the JSON object.

## MISSING INFO
{missing} Fewer good scenarios are better than padded ones.
"""
