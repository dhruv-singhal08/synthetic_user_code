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
from synthetic_user.models.reference import ReferenceIndex, VerificationStatus
from synthetic_user.models.scenario import ScenarioDraft, ScenarioGenerationResult, TaskType
from synthetic_user.nodes.scenario_generator import (
    FILTERED_BY_CONFIG,
    MISSING_CHEAT_SHEET,
    SURPLUS,
    UNEXPECTED_CHEAT_SHEET,
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
        index: ReferenceIndex,
        settings: AppConfig,
    ) -> None:
        self.persona, self.product, self.index, self.settings = persona, product, index, settings

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
            self.persona, self.product, self.index, count=count, existing_signatures=existing
        )


@pytest.fixture
def env(
    sample_persona: Persona,
    sample_product: ProductDescription,
    sample_index: ReferenceIndex,
    app_config: AppConfig,
) -> Env:
    return Env(sample_persona, sample_product, sample_index, app_config)


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
        assert scenario.reference_index_id == env.index.versioned_id
        assert scenario.mode == "grounded"
        assert scenario.success_criteria
        assert scenario.signature == scenario_signature(
            scenario.task_type, scenario.business_goal, scenario.reference_cheat_sheet
        )
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


# --- (a) tables / columns ------------------------------------------------------------------


async def test_unknown_table_rejected(env: Env, scenario_drafts: list[dict[str, Any]]) -> None:
    cheat = scenario_drafts[1]["reference_cheat_sheet"]
    cheat["tables"].append("budgets")
    cheat["columns"].append("budgets.limit_amount")
    result = await env.run(FakeLLMClient([_batch(scenario_drafts[1])]), count=1, max_topup_rounds=0)
    assert result.rejected[0].reasons == ["table 'budgets' is not in the reference index"]


@pytest.mark.parametrize(
    "bad_column",
    ["card_transactions.department_name", "card_transactions.amount_usd"],
)
async def test_column_not_on_table_rejected(
    env: Env, scenario_drafts: list[dict[str, Any]], bad_column: str
) -> None:
    scenario_drafts[1]["reference_cheat_sheet"]["columns"].append(bad_column)
    result = await env.run(FakeLLMClient([_batch(scenario_drafts[1])]), count=1, max_topup_rounds=0)
    assert not result.accepted
    assert bad_column in result.rejected[0].reasons[0]


async def test_unqualified_column_rejected(env: Env, scenario_drafts: list[dict[str, Any]]) -> None:
    scenario_drafts[1]["reference_cheat_sheet"]["columns"].append("amount")
    result = await env.run(FakeLLMClient([_batch(scenario_drafts[1])]), count=1, max_topup_rounds=0)
    assert not result.accepted
    assert "columns" in result.rejected[0].reasons[0]


async def test_one_bad_scenario_does_not_sink_the_batch(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    scenario_drafts[0]["reference_cheat_sheet"]["columns"].append("employees.salary")
    result = await env.run(FakeLLMClient([_batch(*scenario_drafts)]), count=2, max_topup_rounds=0)
    assert [s.title for s in result.accepted] == [
        "Monthly spend trend by merchant category",
        "Card spend forecast",
    ]
    assert len(result.rejected) == 1


# --- (b) no promotion ---------------------------------------------------------------------


async def test_verification_upgrade_downgraded_with_note(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    draft = scenario_drafts[0]
    cheat = draft["reference_cheat_sheet"]
    cheat["tables"].append("departments")
    cheat["columns"].append("departments.department_name")
    cheat["relationships"].append(
        {
            "source": "employees.department_id",
            "target": "departments.department_id",
            "verification": "verified",
        }
    )
    result = await env.run(FakeLLMClient([_batch(draft)]), count=1)
    rels = result.accepted[0].reference_cheat_sheet.relationships
    assert rels[0].verification is VerificationStatus.VERIFIED
    assert rels[1].verification is VerificationStatus.UNKNOWN
    assert any(
        "was marked verified but the reference index says unknown" in c
        for c in result.accepted[0].reference_cheat_sheet.semantic_considerations
    )


async def test_join_without_relationship_becomes_unknown(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    draft = scenario_drafts[0]
    draft["reference_cheat_sheet"]["relationships"] = [
        {
            "source": "card_transactions.employee_id",
            "target": "employees.department_id",
            "verification": "inferred",
        }
    ]
    result = await env.run(FakeLLMClient([_batch(draft)]), count=1)
    cheat = result.accepted[0].reference_cheat_sheet
    assert cheat.relationships[0].verification is VerificationStatus.UNKNOWN
    assert any(
        "has no relationship in the reference index" in c for c in cheat.semantic_considerations
    )


# --- (c) multi-table, (d) unsupported -----------------------------------------------------


async def test_multi_table_rejected_when_not_allowed(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    llm = FakeLLMClient([_batch(*scenario_drafts)])
    result = await env.run(llm, count=3, allow_multi_table=False, max_topup_rounds=0)
    assert len(result.accepted) == 2
    assert result.rejected[0].reasons[0].startswith("multi_table_not_allowed")
    assert "MUST use exactly ONE table" in llm.calls[0].user


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
    reworded["reference_cheat_sheet"]["operations"] = ["join", "sum", "rank"]
    result = await env.run(
        FakeLLMClient([_batch(scenario_drafts[0], reworded)]), count=2, max_topup_rounds=0
    )
    assert len(result.accepted) == 1
    assert len(result.duplicates) == 1


async def test_existing_signatures_are_avoided(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    existing = scenario_signature(
        TaskType.TREND,
        "anything else",
        (await env.run(FakeLLMClient([_batch(scenario_drafts[1])]), count=1))
        .accepted[0]
        .reference_cheat_sheet,
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
    sample_persona: Persona, sample_product: ProductDescription, sample_index: ReferenceIndex
) -> None:
    options = ScenarioPromptOptions(
        count=4,
        allow_multi_table=True,
        include_unsupported_scenarios=True,
        diversity_preference="broad",
    )
    _system, user = build_scenario_prompt(
        sample_persona, sample_product, sample_index, options, avoid_signatures=["sig-1"]
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
    assert "NOT SQL" in user
    assert "copy the relationship's verification EXACTLY" in user
    assert "Writing the user's question" in user
    assert "- sig-1" in user
    for item in sample_product.not_supported_yet + sample_product.known_gaps:
        assert item in user
    assert '"card_transactions.currency_code"' in user
    contract = user.split("## OUTPUT CONTRACT", 1)[1].split("## PROHIBITED", 1)[0]
    for forbidden in ('"id":', '"persona_id":', '"signature":', '"reference_index_id":'):
        assert forbidden not in contract
    assert next(iter(ScenarioDraft.model_json_schema()["properties"])) == "business_goal"


# --- v2.1: success criteria and cheat sheet presence ---------------------------------------


async def test_missing_success_criteria_rejected(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    scenario_drafts[0]["success_criteria"] = []
    result = await env.run(FakeLLMClient([_batch(scenario_drafts[0])]), count=1, max_topup_rounds=0)
    assert not result.accepted
    assert "success_criteria" in result.rejected[0].reasons[0]


async def test_grounded_draft_without_cheat_sheet_rejected(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    del scenario_drafts[0]["reference_cheat_sheet"]
    result = await env.run(FakeLLMClient([_batch(scenario_drafts[0])]), count=1, max_topup_rounds=0)
    assert not result.accepted
    assert result.rejected[0].reasons == [MISSING_CHEAT_SHEET]


# --- C05: ungrounded mode -------------------------------------------------------------------


def _ungrounded(draft: dict[str, Any]) -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in draft.items() if k != "reference_cheat_sheet"}


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
        "question_guidance": {"intent": "learn the hotel cap for Paris"},
    },
    {
        "business_goal": "Get an expense report approved by the manager directly in the chat",
        "title": "Approve my expense report",
        "task_type": "how_to",
        "capability": "unsupported",
        "capability_rationale": "Product lists 'submitting or approving expense reports' under "
        "not_supported_yet",
        "success_criteria": [
            "clearly says the assistant cannot approve reports",
            "explains where approval is done instead",
        ],
        "question_guidance": {"intent": "approve a report via chat"},
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


async def test_ungrounded_accepts_goal_and_criteria_without_cheat_sheet(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    drafts = [_ungrounded(d) for d in scenario_drafts]
    llm = FakeLLMClient([_batch(*drafts)])
    result = await _run_ungrounded(env, llm, count=3)
    assert [s.id for s in result.accepted] == ["SCN_001", "SCN_002", "SCN_003"]
    for scenario in result.accepted:
        assert scenario.mode == "ungrounded"
        assert scenario.business_goal and scenario.success_criteria
        assert scenario.reference_cheat_sheet is None and scenario.reference_index_id is None
        assert scenario.signature == ungrounded_signature(
            scenario.task_type, scenario.business_goal, scenario.capability
        )
    system, user = llm.calls[0].system, llm.calls[0].user
    assert "never invent tables" in system
    assert "reference_cheat_sheet (leave it out)" in user
    assert "card_transactions" not in user and "Reference index" not in user


async def test_ungrounded_cheat_sheet_from_llm_is_rejected(
    env: Env, scenario_drafts: list[dict[str, Any]]
) -> None:
    llm = FakeLLMClient([_batch(scenario_drafts[0], _ungrounded(scenario_drafts[1]))])
    result = await _run_ungrounded(env, llm, count=2, max_topup_rounds=0)
    assert [s.title for s in result.accepted] == [scenario_drafts[1]["title"]]
    assert result.rejected[0].reasons == [UNEXPECTED_CHEAT_SHEET]
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
    assert "product type: chatbot" in user
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
    sample_persona: Persona, sample_product: ProductDescription, sample_index: ReferenceIndex
) -> None:
    options = ScenarioPromptOptions(
        count=2,
        allow_multi_table=True,
        include_unsupported_scenarios=True,
        diversity_preference="x",
    )
    _gs, grounded = build_scenario_prompt(sample_persona, sample_product, sample_index, options)
    _us, ungrounded = build_scenario_prompt(sample_persona, sample_product, None, options)
    for text in (grounded, ungrounded):
        assert "business goal FIRST" in text
        assert "WITH THIS\nPRODUCT" in text
        assert "success_criteria: 2-5 short, checkable statements" in text
        assert "capability_rationale must cite" in text
        assert "Writing the user's question" in text
    assert "Reference index (AUTHORITATIVE" in grounded and "reference_cheat_sheet:" in grounded
    assert "REFIDX" not in ungrounded and "Do not invent data structures" in ungrounded
