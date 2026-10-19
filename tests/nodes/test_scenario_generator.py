from __future__ import annotations

import copy
import logging
from typing import Any

import pytest

from synthetic_user.core.config import AppConfig
from synthetic_user.core.identifiers import SequentialIdGenerator
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import Reference
from synthetic_user.models.scenario import ScenarioDraft, ScenarioGenerationResult, TaskType
from synthetic_user.nodes.scenario_generator import (
    FILTERED_BY_CONFIG,
    MISSING_PACK,
    NO_VALID_ITEM,
    SURPLUS,
    UNEXPECTED_PACK,
    UNKNOWN_TYPICAL_TASK,
    ScenarioGenerator,
    scenario_signature,
    ungrounded_signature,
)
from synthetic_user.prompts.scenario import ScenarioPromptOptions, build_scenario_prompt


def _settings(app_config: AppConfig, **scenario: Any) -> AppConfig:
    gen = app_config.generation
    new_scenario = gen.scenario.model_copy(update=scenario)
    return app_config.model_copy(
        update={"generation": gen.model_copy(update={"scenario": new_scenario})}
    )


def _batch(*drafts: dict[str, Any]) -> dict[str, Any]:
    return {"scenarios": [copy.deepcopy(d) for d in drafts]}


class Env:
    def __init__(
        self,
        persona: Persona,
        product: ProductDescription,
        reference: Reference,
        settings: AppConfig,
    ) -> None:
        self.persona, self.product, self.reference = persona, product, reference
        self.settings = settings

    async def run(
        self, llm: FakeLLMClient, count: int, existing: tuple[str, ...] = (), **scenario: Any
    ) -> ScenarioGenerationResult:
        generator = ScenarioGenerator(
            llm,
            _settings(self.settings, **scenario),
            logging.getLogger("test"),
            id_generator=SequentialIdGenerator(),
        )
        return await generator.generate(
            self.persona, self.product, self.reference, count=count, existing_signatures=existing
        )


@pytest.fixture
def env(
    sample_persona: Persona,
    sample_product: ProductDescription,
    sample_reference: Reference,
    app_config: AppConfig,
) -> Env:
    return Env(sample_persona, sample_product, sample_reference, app_config)


# --- happy path, goal present, ids by code ----------------------------------------------


async def test_accepts_valid_scenarios_with_code_assigned_ids(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    llm = FakeLLMClient([_batch(*scenario_drafts)])
    result = await env.run(llm, count=3)
    assert [s.id for s in result.accepted] == ["SCN_001", "SCN_002", "SCN_003"]
    assert not result.rejected and not result.duplicates
    for scenario in result.accepted:
        assert scenario.business_goal
        assert scenario.persona_id == "finance_controller"
        assert scenario.product_id == "sample_nl2sql"
        assert scenario.reference_id == env.reference.reference_id
        assert scenario.reference_pack_id == f"RP_{scenario.id}"
        assert scenario.mode == "grounded"
        assert scenario.success_criteria
    assert [p.pack_id for p in result.packs] == ["RP_SCN_001", "RP_SCN_002", "RP_SCN_003"]
    for scenario, pack in zip(result.accepted, result.packs, strict=True):
        assert pack.scenario_id == scenario.id
        assert pack.reference_id == env.reference.reference_id
        assert scenario.signature == scenario_signature(
            scenario.task_type, scenario.business_goal, scenario.capability, pack
        )
    assert result.packs[0].rules[0].rule.startswith("Amounts in different currency_code values")
    assert result.packs[0].joins[0].verified is True
    assert len(llm.calls) == 1


async def test_missing_business_goal_rejected(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    del scenario_drafts[0]["business_goal"]
    result = await env.run(FakeLLMClient([_batch(scenario_drafts[0])]), count=1, max_topup_rounds=0)
    assert not result.accepted
    assert "business_goal" in result.rejected[0].reasons[0]


async def test_llm_supplied_ids_rejected(env: Env, scenario_drafts: list[dict[str, Any]]) -> None:
    scenario_drafts[0]["id"] = "SCN_999"
    scenario_drafts[0]["signature"] = "made-up"
    result = await env.run(FakeLLMClient([_batch(scenario_drafts[0])]), count=1, max_topup_rounds=0)
    assert not result.accepted
    assert "Extra inputs are not permitted" in result.rejected[0].reasons[0]


# --- (a) cheat-sheet picks are checked against the reference ------------------------------


INVENTED = {"table": "payroll", "columns": ["salary"], "why": "invented"}


async def test_unknown_picks_are_dropped_and_recorded(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    pack_draft = scenario_drafts[0]["reference_pack"]
    pack_draft["tables"].append(INVENTED)
    pack_draft["tables"][1]["columns"].append("salary")
    pack_draft["joins"].append(
        {"from": "card_transactions.merchant_id", "to": "merchants.merchant_id"}
    )
    pack_draft["rules"].append("R7")
    result = await env.run(FakeLLMClient([_batch(scenario_drafts[0])]), count=1)
    pack = result.packs[0]
    assert pack.item_names() == [
        "card_transactions",
        "card_transactions.amount",
        "card_transactions.currency_code",
        "employees",
        "employees.full_name",
    ]
    assert [r.id for r in pack.rules] == ["R1"]
    assert [d.reason for d in pack.rejected_items] == [
        "column 'salary' is not in table employees",
        "unknown table 'payroll'",
        "card_transactions.merchant_id -> merchants.merchant_id is not a listed relationship",
        "unknown catalog rule id 'R7'",
    ]


async def test_pack_without_a_valid_table_rejects_the_scenario(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    scenario_drafts[1]["reference_pack"] = {"tables": [INVENTED]}
    result = await env.run(FakeLLMClient([_batch(scenario_drafts[1])]), count=1, max_topup_rounds=0)
    assert not result.accepted and not result.packs
    assert result.rejected[0].reasons == [NO_VALID_ITEM, "unknown table 'payroll'"]


async def test_one_bad_scenario_does_not_sink_the_batch(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    scenario_drafts[0]["reference_pack"] = {"tables": [INVENTED]}
    result = await env.run(FakeLLMClient([_batch(*scenario_drafts)]), count=2, max_topup_rounds=0)
    assert [s.title for s in result.accepted] == [
        "Monthly spend trend by merchant category",
        "Card spend forecast",
    ]
    # ids stay dense and the packs follow their scenarios
    assert [s.id for s in result.accepted] == ["SCN_001", "SCN_002"]
    assert [p.scenario_id for p in result.packs] == ["SCN_001", "SCN_002"]
    assert len(result.rejected) == 1


async def test_document_reference_grounds_a_chatbot(
    env: Env, sample_chatbot: ProductDescription, policy_reference: Reference
) -> None:
    draft = {
        **copy.deepcopy(CHATBOT_DRAFTS[0]),
        "reference_pack": {"documents": [{"name": "Hotel policy", "why": "Paris is Tier 1"}]},
    }
    wrong_kind = {
        **copy.deepcopy(CHATBOT_DRAFTS[1]),
        "reference_pack": {"tables": [{"table": "card_transactions", "why": "x"}]},
    }
    generator = ScenarioGenerator(
        FakeLLMClient([_batch(draft, wrong_kind)]),
        _settings(env.settings, max_topup_rounds=0),
        logging.getLogger("t"),
        id_generator=SequentialIdGenerator(),
    )
    result = await generator.generate(env.persona, sample_chatbot, policy_reference, count=2)
    assert result.accepted[0].mode == "grounded"
    (pack,) = result.packs
    assert pack.kind == "document_collection" and pack.documents[0].name == "Hotel policy"
    assert "USD 250 in Tier 1" in pack.documents[0].summary  # the user's text, copied by code
    assert "does not fit a document_collection reference" in result.rejected[0].reasons[1]


# --- (b) unsupported ------------------------------------------------------------------------


async def test_unsupported_filtered_by_config(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    result = await env.run(
        FakeLLMClient([_batch(*scenario_drafts)]),
        count=3,
        include_unsupported_scenarios=False,
        max_topup_rounds=0,
    )
    assert [s.capability for s in result.accepted] == ["supported", "supported"]
    assert result.rejected[0].reasons == [FILTERED_BY_CONFIG]


async def test_unsupported_kept_when_allowed(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    result = await env.run(
        FakeLLMClient([_batch(scenario_drafts[2])]), count=1, include_unsupported_scenarios=True
    )
    assert result.accepted[0].capability == "unsupported"


# --- (e) duplicates -------------------------------------------------------------------------


async def test_identical_scenarios_are_duplicates(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    result = await env.run(
        FakeLLMClient([_batch(scenario_drafts[0], scenario_drafts[0])]),
        count=2,
        max_topup_rounds=0,
    )
    assert len(result.accepted) == 1
    assert result.duplicates[0].reasons[0].startswith("duplicate of SCN_001")


async def test_same_shape_different_goal_is_duplicate(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    reworded = copy.deepcopy(scenario_drafts[0])
    reworded["business_goal"] = "Find the biggest card spenders among staff for audit"
    reworded["reference_pack"]["tables"].reverse()
    result = await env.run(
        FakeLLMClient([_batch(scenario_drafts[0], reworded)]), count=2, max_topup_rounds=0
    )
    assert len(result.accepted) == 1
    assert len(result.duplicates) == 1


async def test_existing_signatures_are_avoided(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    first = await env.run(FakeLLMClient([_batch(scenario_drafts[1])]), count=1)
    existing = scenario_signature(
        TaskType.TREND, "anything else", first.accepted[0].capability, first.packs[0]
    )
    llm = FakeLLMClient([_batch(scenario_drafts[1])])
    result = await env.run(llm, count=1, existing=(existing,), max_topup_rounds=0)
    assert not result.accepted
    assert result.duplicates[0].reasons[0].startswith("duplicate of existing scenario")
    assert existing in llm.calls[0].user


# --- top-up ---------------------------------------------------------------------------------


async def test_top_up_is_bounded_and_never_pads(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    llm = FakeLLMClient([_batch(scenario_drafts[0]), _batch(scenario_drafts[1])])
    result = await env.run(llm, count=3, max_topup_rounds=1)
    assert len(llm.calls) == 2
    assert [s.id for s in result.accepted] == ["SCN_001", "SCN_002"]
    assert "Write 3 distinct scenario(s)" in llm.calls[0].user
    assert "Write 2 distinct scenario(s)" in llm.calls[1].user
    assert result.accepted[0].signature in llm.calls[1].user
    assert llm.remaining == 0


async def test_no_top_up_when_rounds_zero(env: Env, scenario_drafts: list[dict[str, Any]]) -> None:
    llm = FakeLLMClient([_batch(scenario_drafts[0]), _batch(scenario_drafts[1])])
    result = await env.run(llm, count=3, max_topup_rounds=0)
    assert len(llm.calls) == 1
    assert len(result.accepted) == 1


async def test_no_top_up_when_satisfied(env: Env, scenario_drafts: list[dict[str, Any]]) -> None:
    llm = FakeLLMClient([_batch(*scenario_drafts)])
    await env.run(llm, count=3, max_topup_rounds=5)
    assert len(llm.calls) == 1


async def test_surplus_recorded_not_accepted(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    result = await env.run(FakeLLMClient([_batch(*scenario_drafts)]), count=2)
    assert len(result.accepted) == 2
    assert result.rejected[0].reasons == [SURPLUS]


# --- prompt ---------------------------------------------------------------------------------


def test_prompt_contents(
    sample_persona: Persona, sample_product: ProductDescription, sample_reference: Reference
) -> None:
    options = ScenarioPromptOptions(
        count=4, include_unsupported_scenarios=True, diversity_preference="broad"
    )
    _system, user = build_scenario_prompt(
        sample_persona, sample_product, sample_reference, options, avoid_signatures=["sig-1"]
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
    assert "business goal FIRST" in user
    assert "Pick\n  from the catalog by exact name" in user and "drops any" in user
    assert "Writing the user's question" in user
    assert "- sig-1" in user
    for item in sample_product.not_supported + sample_product.known_limitations:
        assert item in user
    assert sample_reference.render() in user  # the whole reference, as rendered by code
    assert "Usage instructions for the reference" in user
    assert "merchant names cannot be joined in" in user  # the user's instructions
    contract = user.split("## OUTPUT CONTRACT", 1)[1].split("## PROHIBITED", 1)[0]
    for forbidden in ('"id":', '"persona_id":', '"signature":', '"reference_id":', "$defs"):
        assert forbidden not in contract
    assert '"joins"' in contract and '"documents"' not in contract  # the catalog's pack
    assert "question_guidance" not in user and "phrasing" not in contract
    assert next(iter(ScenarioDraft.model_json_schema()["properties"])) == "business_goal"


# --- success criteria and reference pack presence ------------------------------------------


async def test_missing_success_criteria_rejected(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    scenario_drafts[0]["success_criteria"] = []
    result = await env.run(FakeLLMClient([_batch(scenario_drafts[0])]), count=1, max_topup_rounds=0)
    assert not result.accepted
    assert "success_criteria" in result.rejected[0].reasons[0]


async def test_grounded_draft_without_pack_rejected(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    del scenario_drafts[0]["reference_pack"]
    result = await env.run(FakeLLMClient([_batch(scenario_drafts[0])]), count=1, max_topup_rounds=0)
    assert not result.accepted
    assert result.rejected[0].reasons == [MISSING_PACK]


# --- C05: ungrounded mode -------------------------------------------------------------------


def _ungrounded(draft: dict[str, Any]) -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in draft.items() if k != "reference_pack"}


CHATBOT_DRAFTS: list[dict[str, Any]] = [
    {
        "business_goal": "Find out the hotel cap for a business trip to Paris before booking",
        "title": "Hotel limit for a Paris trip",
        "task_type": "explanation",
        "capability": "supported",
        "capability_rationale": "Product supports 'explain per-diem and hotel limits'",
        "success_criteria": [
            "states the nightly hotel limit that applies to Paris",
            "says whether the limit includes taxes",
            "points to the policy section the answer relies on",
        ],
        "question_intent": "learn the hotel cap for Paris",
    },
    {
        "business_goal": "Get an expense report approved by the manager directly in the chat",
        "title": "Approve my expense report",
        "task_type": "how_to",
        "capability": "unsupported",
        "capability_rationale": "Product lists 'submitting or approving expense reports' under "
        "not_supported",
        "success_criteria": [
            "clearly says the assistant cannot approve reports",
            "explains where approval is done instead",
        ],
        "question_intent": "approve a report via chat",
    },
]


async def _run_ungrounded(
    env: Env, llm: FakeLLMClient, count: int, product: ProductDescription | None = None, **cfg: Any
) -> ScenarioGenerationResult:
    generator = ScenarioGenerator(
        llm,
        _settings(env.settings, **cfg),
        logging.getLogger("t"),
        id_generator=SequentialIdGenerator(),
    )
    return await generator.generate(env.persona, product or env.product, None, count=count)


async def test_ungrounded_accepts_goal_and_criteria_without_pack(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    drafts = [_ungrounded(d) for d in scenario_drafts]
    llm = FakeLLMClient([_batch(*drafts)])
    result = await _run_ungrounded(env, llm, count=3)
    assert [s.id for s in result.accepted] == ["SCN_001", "SCN_002", "SCN_003"]
    for scenario in result.accepted:
        assert scenario.mode == "ungrounded"
        assert scenario.business_goal and scenario.success_criteria
        assert scenario.reference_pack_id is None and scenario.reference_id is None
        assert scenario.signature == ungrounded_signature(
            scenario.task_type, scenario.business_goal, scenario.capability
        )
    assert not result.packs
    system, user = llm.calls[0].system, llm.calls[0].user
    assert "never invent tables" in system
    assert "there is no reference_pack" in user and '"reference_pack"' not in user
    assert "card_transactions" not in user and "<<<REFERENCE" not in user


async def test_ungrounded_pack_from_llm_is_rejected(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    llm = FakeLLMClient([_batch(scenario_drafts[0], _ungrounded(scenario_drafts[1]))])
    result = await _run_ungrounded(env, llm, count=2, max_topup_rounds=0)
    assert [s.title for s in result.accepted] == [scenario_drafts[1]["title"]]
    assert result.rejected[0].reasons == [UNEXPECTED_PACK]
    assert result.rejected[0].payload["title"] == scenario_drafts[0]["title"]


async def test_ungrounded_unsupported_filter_and_empty_criteria(env: Env) -> None:
    empty = {**copy.deepcopy(CHATBOT_DRAFTS[0]), "success_criteria": []}
    llm = FakeLLMClient([_batch(empty, CHATBOT_DRAFTS[1])])
    result = await _run_ungrounded(
        env, llm, count=2, max_topup_rounds=0, include_unsupported_scenarios=False
    )
    assert not result.accepted
    assert "success_criteria" in result.rejected[0].reasons[0]
    assert result.rejected[1].reasons == [FILTERED_BY_CONFIG]


async def test_non_sql_product_sample_chatbot_works(
    env: Env, sample_chatbot: ProductDescription
) -> None:
    llm = FakeLLMClient([_batch(*CHATBOT_DRAFTS)])
    result = await _run_ungrounded(env, llm, count=2, product=sample_chatbot)
    assert [(s.product_id, s.task_type, s.capability) for s in result.accepted] == [
        ("sample_policy_chatbot", TaskType.EXPLANATION, "supported"),
        ("sample_policy_chatbot", TaskType.HOW_TO, "unsupported"),
    ]
    user = llm.calls[0].user
    assert 'PRODUCT ("Sample Travel & Expense Policy Assistant")' in user
    assert "submitting or approving expense reports" in user
    assert "SQL" not in llm.calls[0].system


async def test_ungrounded_duplicates_by_task_goal_and_capability(env: Env) -> None:
    same_goal = {
        **copy.deepcopy(CHATBOT_DRAFTS[0]),
        "title": "Paris hotel cap (again)",
        "business_goal": "Before booking, find out the hotel cap for a Paris business trip",
    }
    other_capability = {**copy.deepcopy(CHATBOT_DRAFTS[0]), "capability": "unknown"}
    llm = FakeLLMClient([_batch(CHATBOT_DRAFTS[0], same_goal, other_capability)])
    result = await _run_ungrounded(env, llm, count=3, max_topup_rounds=0)
    assert [s.capability for s in result.accepted] == ["supported", "unknown"]
    assert len(result.duplicates) == 1
    assert result.duplicates[0].reasons[0].startswith("duplicate of SCN_001")


async def test_ungrounded_top_up_bounded_never_padded(env: Env) -> None:
    llm = FakeLLMClient([_batch(CHATBOT_DRAFTS[0]), _batch(), _batch()])
    result = await _run_ungrounded(env, llm, count=3, max_topup_rounds=1)
    assert len(result.accepted) == 1 and len(llm.calls) == 2


def test_both_prompts_share_the_common_core(
    sample_persona: Persona, sample_product: ProductDescription, sample_reference: Reference
) -> None:
    options = ScenarioPromptOptions(
        count=2, include_unsupported_scenarios=True, diversity_preference="x"
    )
    _gs, grounded = build_scenario_prompt(sample_persona, sample_product, sample_reference, options)
    _us, ungrounded = build_scenario_prompt(sample_persona, sample_product, None, options)
    for text in (grounded, ungrounded):
        assert "business goal FIRST" in text
        assert "WITH THIS\nPRODUCT" in text
        assert "success_criteria: 2-5 short, checkable statements" in text
        assert "capability_rationale must cite" in text
        assert "Writing the user's question" in text
    assert "AUTHORITATIVE" in grounded and '"reference_pack"' in grounded
    assert "REFIDX" not in ungrounded and "Do not invent data structures" in ungrounded


# --- persona schema v2: typical_task links a scenario to the persona (D200) -------------------


async def test_typical_task_is_matched_to_the_persona_wording(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    named, unknown, none = (copy.deepcopy(d) for d in scenario_drafts)
    named["typical_task"] = "identify top merchants or merchant categories by spend."
    unknown["typical_task"] = "Plan the office party"
    result = await env.run(
        FakeLLMClient([_batch(named, unknown, none)]), count=3, max_topup_rounds=0
    )
    assert [s.typical_task for s in result.accepted] == [
        "Identify top merchants or merchant categories by spend",  # the persona's own text
        None,
    ]
    assert result.rejected[0].reasons == [f"{UNKNOWN_TYPICAL_TASK}: 'Plan the office party'"]
