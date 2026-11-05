from __future__ import annotations

import logging
from typing import Any

import pytest

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import ArtifactIntegrityError, LLMProviderError
from synthetic_user.core.identifiers import SequentialIdGenerator
from synthetic_user.llm.fake import FakeCall, FakeLLMClient
from synthetic_user.metrics.registry import builtin_metric_config
from synthetic_user.models.base import AmbiguityLevel
from synthetic_user.models.persona import ExpertiseLevel, Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionGenerationResult
from synthetic_user.models.reference import CatalogPackDraft, Reference, ReferencePack, TablePick
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario, TaskType
from synthetic_user.nodes.question.generator import (
    SURPLUS,
    VAGUE_LOW,
    VAGUE_NOT_ALLOWED,
    QuestionGenerator,
)
from synthetic_user.nodes.question.prompt import QuestionPromptOptions, build_question_prompt
from synthetic_user.references.pack import build_pack


def plan_for(complexity: str, limit: int | None = None) -> dict[str, Any]:
    """A structured plan whose computed complexity is ``complexity``."""
    rank = {"action": "rank", "subject": "employees by card spend", "sources": [], "limit": limit}
    trend = {"action": "trend", "subject": "monthly card spend", "sources": []}
    deliverables = {"simple": [rank], "medium": [rank, trend], "complex": [rank, trend, rank]}
    return {"deliverables": deliverables[complexity], "filters": [], "is_follow_up": False}


CRITERIA = [
    "names the spenders in ranked order",
    "if amounts use several currencies, keeps them separate",
]


def q(
    utterance: str,
    style: str = "precise",
    ambiguity: str = "low",
    complexity: str = "simple",
    limit: int | None = None,
    *,
    criteria: list[str] | None = None,
    mapping: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "utterance": utterance,
        "plan": plan_for(complexity, limit),
        "intent": {"task_type": "ranking"},
        "question_characteristics": {"style": style, "ambiguity_level": ambiguity},
        "evaluation_context": {
            "success_criteria": [
                {"id": f"C{i}", "requirement": text, "observed_in": "sql"}
                for i, text in enumerate(criteria or CRITERIA, start=1)
            ],
            "metric_mapping": mapping or {},
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
        metrics: tuple[str, ...] = (),
        reference: Reference | None = None,
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
            reference or self.reference,
            self.packs,
            count=count,
            metrics=builtin_metric_config(list(metrics)).definitions if metrics else (),
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
        * 3
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
            batch(q("Top 5 spenders in card_transactions?", limit=5)),
            batch(q("Who are our top 5 card spenders?", limit=5)),
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
        * 3
    )
    config_driven = with_prefs(env.persona, ambiguity=None)  # persona leaves ambiguity to config
    result = await env.run(llm, persona=config_driven, allow_vague_questions=True)
    assert [x.utterance for x in result.questions] == ["Is spend looking weird lately?"]
    assert result.questions[0].question_characteristics.ambiguity_level is AmbiguityLevel.MEDIUM
    assert result.rejected[0].reasons == [VAGUE_LOW]


async def test_vague_rejected_when_not_allowed(env: Env) -> None:
    llm = FakeLLMClient(
        [
            batch(
                q("Is spend looking weird lately?", "vague", "high"),
                q("Why did travel spend jump in May?", "investigative", "medium"),
            )
        ]
        * 3
    )
    config_driven = with_prefs(env.persona, ambiguity=None)
    result = await env.run(llm, persona=config_driven, allow_vague_questions=False)
    assert [x.question_characteristics.style for x in result.questions] == ["investigative"]
    assert result.rejected[0].reasons == [VAGUE_NOT_ALLOWED]


# --- dedupe -----------------------------------------------------------------------------------


async def test_exact_duplicates_are_rejected(env: Env) -> None:
    """Exact (case / punctuation / spacing normalized) duplicates, also across scenarios. The
    similarity-based diversity check (D239) is switched off here: its own tests are below."""
    llm = FakeLLMClient(
        [
            batch(
                q("Who are our top card spenders this quarter?"),
                q("Who are our top card spenders, this quarter?!"),
            ),
            batch(
                q("Who were our top card spenders over this quarter?"),  # similar, not equal
                q("How is category spend trending?"),
            ),
        ]
    )
    result = await env.run(llm, scenarios=2, diversity_check=False)
    assert [x.utterance for x in result.questions] == [
        "Who are our top card spenders this quarter?",
        "Who were our top card spenders over this quarter?",
        "How is category spend trending?",
    ]
    assert [r.reasons for r in result.rejected] == [["exact duplicate of Q_001"]]


async def test_copying_persona_example_rejected(env: Env) -> None:
    example = env.persona.example_utterances[0]
    llm = FakeLLMClient([batch(q(example))] * 3)
    result = await env.run(llm, count=1)
    assert not result.questions and len(llm.calls) == 3  # fed back, not found after acceptance
    assert result.rejected[0].reasons == [
        "exact duplicate of persona example utterance: write a new question"
    ]


async def test_copying_a_typical_user_question_rejected(env: Env) -> None:
    typical = "What are the biggest customer pain points this quarter?"
    product = env.product.model_copy(update={"typical_user_questions": [typical]})
    llm = FakeLLMClient([batch(q("what are the biggest customer pain points this quarter"))] * 3)
    generator = QuestionGenerator(
        llm, env.settings, logging.getLogger("test"), id_generator=SequentialIdGenerator()
    )
    result = await generator.generate(
        env.persona, product, env.scenarios[:1], env.reference, env.packs, count=1
    )
    assert not result.questions
    assert result.rejected[0].reasons == [
        "exact duplicate of product typical user question: write a new question"
    ]


# --- one call per scenario, concurrency, isolation ---------------------------------------------


class _CountingLLM(FakeLLMClient):
    """Tracks how many generation calls are in flight at once."""

    def __init__(self, reply: dict[str, Any], fail_on: str | None = None) -> None:
        super().__init__()
        self._reply, self._fail_on = reply, fail_on
        self.in_flight = 0
        self.peak = 0

    async def generate_json(self, system: str, user: str) -> dict[str, Any]:
        import asyncio
        import copy

        self.calls.append(FakeCall("generate_json", system, user))
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        if self._fail_on is not None and self._fail_on in user:
            raise LLMProviderError("provider down for this scenario")
        reply = copy.deepcopy(self._reply)
        title = user.split('"title": "', 1)[1].split('"', 1)[0]
        reply["questions"][0]["utterance"] = f"Question about {title}?"
        return reply


def _many_scenarios(env: Env, n: int) -> list[Scenario]:
    base = env.scenarios[0]
    return [
        base.model_copy(update={"id": f"SCN_{i:03d}", "title": f"Goal {i}", "signature": f"s{i}"})
        for i in range(1, n + 1)
    ]


async def _generate_for(env: Env, llm: FakeLLMClient, scenarios: list[Scenario], **cfg: Any) -> Any:
    generator = QuestionGenerator(
        llm,
        _settings(env.settings, **cfg),
        logging.getLogger("test"),
        id_generator=SequentialIdGenerator(),
    )
    packs = [
        env.packs[0].model_copy(update={"pack_id": s.reference_pack_id, "scenario_id": s.id})
        for s in scenarios
    ]
    scenarios = [s.model_copy(update={"reference_pack_id": f"RP_{s.id}"}) for s in scenarios]
    packs = [p.model_copy(update={"pack_id": f"RP_{p.scenario_id}"}) for p in packs]
    return await generator.generate(
        env.persona, env.product, scenarios, env.reference, packs, count=1
    )


async def test_one_call_per_scenario_concurrency_default_five(env: Env) -> None:
    llm = _CountingLLM(batch(q("x")))
    result = await _generate_for(env, llm, _many_scenarios(env, 8))
    assert len(llm.calls) == 8  # exactly one generation call per accepted scenario
    assert llm.peak == 5  # generation.question.max_parallel_scenarios default
    # results keep the input scenario order whatever order the calls finish in
    assert [x.scenario_id for x in result.questions] == [f"SCN_{i:03d}" for i in range(1, 9)]
    assert [x.question_id for x in result.questions] == [f"Q_{i:03d}" for i in range(1, 9)]


async def test_concurrency_is_configurable(env: Env) -> None:
    llm = _CountingLLM(batch(q("x")))
    await _generate_for(env, llm, _many_scenarios(env, 4), max_parallel_scenarios=2)
    assert llm.peak == 2


async def test_failed_scenario_call_is_isolated(env: Env) -> None:
    llm = _CountingLLM(batch(q("x")), fail_on='"title": "Goal 2"')
    result = await _generate_for(env, llm, _many_scenarios(env, 3))
    assert [x.scenario_id for x in result.questions] == ["SCN_001", "SCN_003"]
    assert [(r.scenario_id, r.reasons) for r in result.rejected] == [
        ("SCN_002", ["question generation call failed: provider down for this scenario"])
    ]


# --- metric expectations -----------------------------------------------------------------------


def with_metric_expectations(
    question: dict[str, Any],
    entries: dict[str, Any],
    criteria_metrics: tuple[str, ...] = ("semantic_correctness",),
    observed_in: str = "sql",
) -> dict[str, Any]:
    """The question with a metric_mapping: every criterion (observed in ``observed_in``) mapped
    to ``criteria_metrics`` and ``entries`` (metric id -> expectation) as the expectations."""
    stated = question["evaluation_context"]["success_criteria"]
    criteria = [{**c, "observed_in": observed_in} for c in stated]
    ids = [c["id"] for c in criteria]
    mapping: dict[str, Any] = {m: {"criteria": ids, "expectation": None} for m in criteria_metrics}
    for metric_id, entry in entries.items():
        mapping.setdefault(metric_id, {"criteria": []})["expectation"] = entry
    question = {**question, "evaluation_context": {**question["evaluation_context"]}}
    question["evaluation_context"]["success_criteria"] = criteria
    question["evaluation_context"]["metric_mapping"] = mapping
    return question


async def test_criteria_metrics_are_mapped_and_code_completes_the_mapping(env: Env) -> None:
    mapped = with_metric_expectations(q("Who spends most?"), {})
    mapped["evaluation_context"]["metric_mapping"]["semantic_correctness"]["criteria"] = ["C1"]
    llm = FakeLLMClient([batch(mapped)])
    metrics = ("relevance", "correctness", "semantic_correctness", "schema_faithfulness")
    result = await env.run(llm, count=1, metrics=metrics)
    context = result.questions[0].context
    semantic = context.metric_mapping["semantic_correctness"]
    assert (semantic.criteria, semantic.expectation) == (["C1"], None)
    # correctness: EVERY criterion and the Reference Spec (schema + available rules), by code
    correctness = context.metric_mapping["correctness"]
    assert (correctness.criteria, correctness.reference_requirements) == (
        ["C1", "C2"],
        ["reference_spec.schema"],
    )
    assert correctness.available_rules == ["TABLE-RULE-01.01", "RULE-R1"]
    assert correctness.comparison_basis == "criteria_and_reference_spec_vs_response"
    assert context.unmapped_criteria() == []  # no coverage gap while correctness is selected
    # relevance compares the question with the response only: never success criteria
    relevance = context.metric_mapping["relevance"]
    assert (relevance.criteria, relevance.reference_requirements) == ([], [])
    assert relevance.comparison_basis == "question_vs_response"
    # schema_faithfulness takes no criteria: code maps it to the Reference Spec's schema
    schema = context.metric_mapping["schema_faithfulness"]
    assert (schema.criteria, schema.reference_requirements) == ([], ["reference_spec.schema"])
    assert schema.comparison_basis == "schema_vs_sql" and schema.evidence_types == ["sql"]
    prompt = llm.calls[0].user
    assert "- semantic_correctness (Semantic Correctness): criteria = the" in prompt
    rules = prompt.split("metric_mapping: one entry", 1)[1].split("## OUTPUT CONTRACT", 1)[0]
    assert "schema_faithfulness (" not in rules and "relevance (" not in rules
    assert "- correctness (" not in rules  # code maps it


async def test_mapped_criterion_ids_must_exist(env: Env) -> None:
    bad = q(
        "Who spends most?",
        mapping={"semantic_correctness": {"criteria": ["C1", "C9"], "expectation": None}},
    )
    llm = FakeLLMClient([batch(bad)] * 3)
    result = await env.run(llm, count=1, metrics=("semantic_correctness",))
    assert not result.questions
    reasons = " ".join(result.rejected[0].reasons)
    assert "criteria ['C9'] are not success_criteria ids" in reasons
    assert "not mapped" not in reasons  # an unmapped C2 is a coverage gap, not a rejection


async def test_criterion_ids_are_normalised_by_code(env: Env) -> None:
    question = q("Who spends most?", mapping={"semantic_correctness": {"criteria": ["b", "a"]}})
    question["evaluation_context"]["success_criteria"] = [
        {"id": "a", "requirement": "names the spenders", "observed_in": "sql"},
        {"id": "b", "requirement": "ranks them", "observed_in": "sql"},
    ]
    result = await env.run(
        FakeLLMClient([batch(question)]), count=1, metrics=("semantic_correctness",)
    )
    context = result.questions[0].context
    assert [(c.id, c.requirement) for c in context.success_criteria] == [
        ("C1", "names the spenders"),
        ("C2", "ranks them"),
    ]
    assert context.metric_mapping["semantic_correctness"].criteria == ["C2", "C1"]


async def test_generation_failure_is_recorded_not_fabricated(env: Env) -> None:
    llm = FakeLLMClient([{"questions": [], "generation_failure": "the cheat sheet has no period"}])
    result = await env.run(llm, count=1)
    assert not result.questions
    assert result.rejected[0].reasons == ["generation failure: the cheat sheet has no period"]
    assert "generation_failure" in llm.calls[0].user


async def test_context_recall_checklist_generated_and_stored(env: Env) -> None:
    entry = {
        "context_recall": {
            "status": "generated",
            "value": {"required_information": ["card spend per employee", "the last quarter"]},
        }
    }
    llm = FakeLLMClient([batch(with_metric_expectations(q("Who spends most?"), entry))])
    result = await env.run(llm, count=1, metrics=("semantic_correctness", "context_recall"))
    stored = result.questions[0].context.expectation("context_recall")
    assert stored is not None and stored.status == "generated"
    assert stored.value == {"required_information": ["card spend per employee", "the last quarter"]}
    prompt = llm.calls[0].user
    assert "required_information (string_list, required)" in prompt
    assert "never invent" in prompt.lower()
    assert '"context_recall": {' in prompt  # only the metric that needs an expectation


async def test_cannot_generate_is_recorded_and_question_kept(env: Env) -> None:
    entry = {
        "context_recall": {
            "status": "cannot_generate",
            "value": None,
            "reason": "the question does not say which period",
        }
    }
    llm = FakeLLMClient([batch(with_metric_expectations(q("Who spends most?"), entry))])
    result = await env.run(llm, count=1, metrics=("semantic_correctness", "context_recall"))
    assert len(result.questions) == 1 and not result.rejected
    stored = result.questions[0].context.expectation("context_recall")
    assert stored is not None
    assert (stored.status, stored.value, stored.reason) == (
        "cannot_generate",
        None,
        "the question does not say which period",
    )


@pytest.mark.parametrize(
    ("entries", "problem"),
    [
        ({}, "metric_mapping is missing ['context_recall']"),
        (
            {
                "context_recall": {"status": "generated", "value": {"required_information": []}},
            },
            "required field 'required_information' is missing",
        ),
        (
            {
                "context_recall": {"status": "generated", "value": {"other": ["x"]}},
            },
            "unknown field 'other'",
        ),
        (
            {
                "context_recall": {"status": "cannot_generate", "value": None},
            },
            "needs a specific reason",
        ),
        (
            {
                "context_recall": {"status": "not_applicable", "value": None, "reason": "n/a"},
                "faithfulness": {"status": "not_applicable", "value": None, "reason": "n/a"},
            },
            "must not include ['faithfulness']",
        ),
    ],
)
async def test_malformed_metric_expectations_rejected(
    env: Env, entries: dict[str, Any], problem: str
) -> None:
    bad = with_metric_expectations(q("Who spends most?"), entries)
    llm = FakeLLMClient([batch(bad)] * 3)
    result = await env.run(llm, count=1, metrics=("semantic_correctness", "context_recall"))
    assert not result.questions
    assert any(problem in reason for reason in result.rejected[0].reasons), result.rejected


async def test_tool_expectation_cannot_generate_without_tool_definitions(env: Env) -> None:
    """No agent_context in the reference: code records cannot_generate, the LLM is not asked."""
    question = with_metric_expectations(
        q("Who spends most?"), {"tool_correctness": None}, (), "answer"
    )
    llm = FakeLLMClient([batch(question)])
    result = await env.run(llm, count=1, metrics=("correctness", "tool_correctness"))
    stored = result.questions[0].context.expectation("tool_correctness")
    assert stored is not None and stored.status == "cannot_generate"
    assert "no authoritative tool definitions" in (stored.reason or "")
    assert "expected_tools" not in llm.calls[0].user  # its expectation is not asked for


async def test_tool_expectation_canonical_names_and_unknown_tools(
    env: Env, tool_reference: Reference
) -> None:
    pack, _ = build_pack(
        CatalogPackDraft(tables=[TablePick(table="complaints_t", columns=["theme"], why="x")]),
        tool_reference,
        "SCN_001",
    )
    assert pack is not None
    scenario = env.scenarios[0].model_copy(update={"reference_id": tool_reference.reference_id})
    tools_env = Env(env.persona, env.product, [scenario], tool_reference, [pack], env.settings)
    good = {
        "tool_correctness": {
            "status": "generated",
            "value": {"expected_tools": ["aggregate"], "expected_data_sources": ["Survey data"]},
        }
    }
    bad = {
        "tool_correctness": {
            "status": "generated",
            "value": {"expected_tools": ["invented_tool"]},
        }
    }
    tools = ("tool_correctness",)
    trace = "tool_trace"
    llm = FakeLLMClient(
        [batch(with_metric_expectations(q("Which complaints rank top?"), good, tools, trace))]
    )
    result = await tools_env.run(llm, count=1, metrics=("tool_correctness",))
    stored_good = result.questions[0].context.expectation("tool_correctness")
    value = stored_good.value if stored_good is not None else None
    # aliases are stored as the canonical names of the authoritative tool definitions
    assert value == {"expected_tools": ["aggregate_feedback"], "expected_data_sources": ["csat"]}
    assert "aggregate_feedback" in llm.calls[0].user  # authoritative definitions are shown

    llm = FakeLLMClient(
        [batch(with_metric_expectations(q("Which complaints rank top?"), bad, tools, trace))] * 3
    )
    result = await tools_env.run(llm, count=1, metrics=("tool_correctness",))
    assert len(llm.calls) == 3  # retried with the membership problem as feedback
    assert "unknown tool 'invented_tool'" in llm.calls[1].user
    # never fabricated or silently fixed: recorded as cannot_generate, the question is kept
    stored = result.questions[0].context.expectation("tool_correctness")
    assert stored is not None
    assert stored.status == "cannot_generate" and "invented_tool" in (stored.reason or "")


# --- other ------------------------------------------------------------------------------------


async def test_surplus_recorded(env: Env) -> None:
    llm = FakeLLMClient(
        [batch(q("Who spends most?"), q("Which team spends most?"), q("Top merchants?"))]
    )
    # q() gives every question the same plan: not a diversity test (D239)
    result = await env.run(llm, count=2, diversity_check=False)
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
    assert "evaluation_context.success_criteria: every requirement" in user
    assert "with ids C1, C2, ... in order" in user
    assert "expected_behavior" not in user  # removed from the contract (reforming)
    assert "Code attaches it to every question as the Reference Spec" in user
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


async def test_question_owns_its_evaluation_context(env: Env) -> None:
    llm = FakeLLMClient([batch(q("Who are our biggest card spenders this quarter?"))])
    result = await env.run(llm, count=1)
    context = result.questions[0].context
    assert [c.requirement for c in context.success_criteria] == CRITERIA
    assert [c.requirement for c in context.success_criteria] != env.scenarios[0].success_criteria
    # The Reference Spec is the scenario's pack, reused by code with its identity
    pack = next(p for p in env.packs if p.pack_id == env.scenarios[0].reference_pack_id)
    assert context.reference_spec is not None and context.reference_spec.pack == pack
    assert context.reference_spec.reference_id == env.reference.reference_id
    assert result.questions[0].legacy_expectations is None


@pytest.mark.parametrize(
    "evaluation_context",
    [
        {"success_criteria": []},  # no criteria: never copied from the scenario
        {"success_criteria": [{"id": f"C{i}", "requirement": "x"} for i in range(11)]},
        {"success_criteria": [{"id": "C1", "requirement": "a"}], "expected_behavior": "x"},
    ],
)
async def test_bad_evaluation_context_rejected(
    env: Env, evaluation_context: dict[str, Any]
) -> None:
    bad = q("Who spends the most on cards?", evaluation_context=evaluation_context)
    llm = FakeLLMClient([batch(bad)])
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
        FakeLLMClient([batch(vague)] * 3), persona=never, count=1, allow_vague_questions=True
    )
    assert not refused.questions and refused.rejected[0].reasons == [VAGUE_NOT_ALLOWED]


async def test_complexity_is_computed_and_hard_policy_enforced_with_a_retry(env: Env) -> None:
    # Complexity comes from the structured plan, never from a model-provided label.
    llm = FakeLLMClient(
        [
            batch(q("Who spent most, and how did spend trend?", complexity="medium")),
            batch(q("Who spent most, and how did spend trend?", complexity="medium")),
            batch(
                q("Who spent most, and how did spend trend?", complexity="medium"),
                q("Which team spent most?"),
            ),
        ]
    )
    result = await env.run(llm, count=2, allowed_complexities=["simple"])
    assert "computed complexity medium is not allowed" in llm.calls[1].user  # retry feedback
    assert [x.utterance for x in result.questions] == ["Which team spent most?"]
    assert result.questions[0].question_characteristics.complexity == "simple"
    assert any("computed complexity medium" in r for r in result.rejected[0].reasons)


async def test_mixed_complexity_accepts_any(env: Env) -> None:
    question = q("Who spent most last month?", complexity="complex")
    result = await env.run(
        FakeLLMClient([batch(question, q("Which merchants?"))]),
        allowed_complexities=["simple", "medium", "complex"],
        max_deliverables=3,
        diversity_check=False,  # q() gives every question the same success criteria
    )
    assert not result.rejected
    assert [x.question_characteristics.complexity for x in result.questions] == [
        "complex",
        "simple",
    ]


async def test_model_supplied_complexity_label_is_rejected(env: Env) -> None:
    question = q("Who spent most last month?")
    question["question_characteristics"]["complexity"] = "complex"
    result = await env.run(FakeLLMClient([batch(question)] * 3), count=1)
    assert not result.questions
    assert "complexity: Extra inputs are not permitted" in result.rejected[0].reasons[0]


def test_technical_knowledge_high_makes_the_persona_technical(sample_persona: Persona) -> None:
    data = sample_persona.model_dump(mode="json")
    assert not sample_persona.is_technical
    data["expertise"]["technical_knowledge"] = "high"
    assert Persona.model_validate(data).is_technical


# --- hard question policy --------------------------------------------------------------------


def _draft(utterance: str, **changes: Any) -> Any:
    from synthetic_user.models.question import QuestionDraft

    raw = q(utterance)
    for path, value in changes.items():
        target = raw
        *parents, leaf = path.split("__")
        for key in parents:
            target = target[key]
        target[leaf] = value
    return QuestionDraft.model_validate(raw)


@pytest.mark.parametrize(
    ("utterance", "changes", "problem"),
    [
        (
            "Who spent most?",
            {"plan__deliverables": [{"action": "rank", "subject": "spend", "sources": ["CRM"]}]},
            "unknown product data sources",
        ),
        ("Who spent most?", {"plan__is_follow_up": True}, "follow-up questions are disabled"),
        (" ".join(["word"] * 45), {}, "words; maximum is 40"),
    ],
)
def test_hard_question_policy(
    env: Env, utterance: str, changes: dict[str, Any], problem: str
) -> None:
    from synthetic_user.nodes.question.generator import question_policy_problems

    problems = question_policy_problems(
        _draft(utterance, **changes), env.product, env.settings.generation.question
    )
    assert any(problem in p for p in problems), problems


def test_clean_question_has_no_policy_problems(env: Env) -> None:
    from synthetic_user.nodes.question.generator import question_policy_problems

    assert (
        question_policy_problems(
            _draft("Who spent most last month?"), env.product, env.settings.generation.question
        )
        == []
    )


@pytest.mark.parametrize(
    ("utterance", "changes"),
    [
        (
            "Show complaint themes?",
            {"plan__deliverables": [{"action": "rank", "subject": "themes with comments"}]},
        ),
        ("Who spent most and what did they buy?", {}),
        ("Top 5 spenders?", {}),
        (
            "Who spent most?",
            {
                "evaluation_context__success_criteria": [
                    {"id": "C1", "requirement": "Return the top 10 spenders."}
                ]
            },
        ),
        (
            "Who spent most?",
            {
                "evaluation_context__success_criteria": [
                    {"id": "C1", "requirement": "shows amounts in USD"}
                ]
            },
        ),
    ],
)
def test_no_regex_semantic_rejection(env: Env, utterance: str, changes: dict[str, Any]) -> None:
    """Former keyword/regex semantic checks (compound outputs, second request, top-N, currency)
    are gone: semantic fairness is the prompt's job, not a regex gate."""
    from synthetic_user.nodes.question.generator import question_policy_problems

    assert (
        question_policy_problems(
            _draft(utterance, **changes), env.product, env.settings.generation.question
        )
        == []
    )


def test_prompt_guards_replace_the_former_regex_gates(env: Env) -> None:
    """Regressions of earlier completed runs (invented top-N limits, unconditional currency or
    privacy duties, unsupported drift) are now prevented by prompt instructions."""
    options = QuestionPromptOptions(
        count=1, ambiguity_level=AmbiguityLevel.LOW, language="en", allow_vague_questions=False
    )
    unsupported = env.scenarios[2]
    _system, user = build_question_prompt(
        env.persona, env.product, unsupported, (env.packs[2], env.reference), options
    )
    for guard in (
        "Never require LIMIT, top N, top 5, top 10 or another result count",
        "Do not add a currency, unit, conversion",
        'write it explicitly as a conditional criterion beginning with "If ..."',
        "used only internally to compute a requested aggregate",
        "keep the unsupported request",
        "Write the utterance FIRST",
    ):
        assert guard in user, guard
    assert "COUNT(DISTINCT" not in user  # no SQL wording in the generic question prompt


# --- semantic diversity (D239) ----------------------------------------------------------------


def _with(
    question: dict[str, Any],
    *,
    filters: list[dict[str, str]] | None = None,
    criteria: list[str] | None = None,
) -> dict[str, Any]:
    if filters is not None:
        question["plan"]["filters"] = filters
    if criteria is not None:
        question["evaluation_context"] = {
            **question["evaluation_context"],
            "success_criteria": [
                {"id": f"C{i}", "requirement": text} for i, text in enumerate(criteria, start=1)
            ],
        }
    return question


PERIOD = [{"name": "period", "value": "last quarter"}]


async def test_paraphrase_is_a_near_duplicate_fixed_by_the_existing_retry(env: Env) -> None:
    paraphrase = batch(
        _with(q("Who were our top card spenders last quarter?"), filters=PERIOD),
        _with(q("Who were the top card spenders for last quarter?"), filters=PERIOD),
    )
    diverse = batch(
        _with(q("Who were our top card spenders last quarter?"), filters=PERIOD),
        _with(
            q("Which employees spent the most on cards so far this year?"),
            filters=[{"name": "period", "value": "year to date"}],
        ),
    )
    llm = FakeLLMClient([paraphrase, diverse])
    result = await env.run(llm)
    assert len(llm.calls) == 2  # one retry of the SAME call, no extra call
    feedback = llm.calls[1].user
    assert "question 2: near-duplicate of question 1" in feedback
    assert (
        "varying ambiguity, terminology, filters, time expression, grouping or phrasing" in feedback
    )
    assert len(result.questions) == 2 and not result.rejected


@pytest.mark.parametrize(
    "second",
    [
        # a different time expression (filter value), nearly the same words
        _with(
            q("Who were our top card spenders last month?"),
            filters=[{"name": "period", "value": "previous month"}],
        ),
        # a different ambiguity / style, nearly the same words
        _with(
            q("Why were those our top card spenders last quarter?", "investigative", "medium"),
            filters=PERIOD,
        ),
        # a different result count (its own expected behaviour, as a real run writes it)
        _with(
            q("Who were our top 5 card spenders last quarter?", limit=5),
            filters=PERIOD,
            criteria=["ranks the five biggest card spenders of last quarter"],
        ),
        # the same request in clearly different terminology
        _with(q("Show the biggest corporate-card users for Q3 please"), filters=PERIOD),
    ],
)
async def test_real_variation_is_never_a_near_duplicate(env: Env, second: dict[str, Any]) -> None:
    first = _with(q("Who were our top card spenders last quarter?"), filters=PERIOD)
    llm = FakeLLMClient([batch(first, second)])
    result = await env.run(llm)
    assert len(llm.calls) == 1 and len(result.questions) == 2 and not result.rejected


async def test_persistent_near_duplicate_is_rejected_with_its_reason(env: Env) -> None:
    same = batch(
        _with(q("Who were our top card spenders last quarter?"), filters=PERIOD),
        _with(q("Who were the top card spenders last quarter?"), filters=PERIOD),
    )
    llm = FakeLLMClient([same, same, same])
    result = await env.run(llm)
    assert [x.utterance for x in result.questions] == [
        "Who were our top card spenders last quarter?"
    ]
    (rejected,) = result.rejected
    assert rejected.reasons[0].startswith("near-duplicate of question 1")


async def test_criteria_copied_from_a_sibling_are_retried(env: Env) -> None:
    copied = batch(
        q("Who spent most last quarter?"),
        q("How did monthly card spend move?", complexity="medium"),  # same success criteria
    )
    fixed = batch(
        q("Who spent most last quarter?"),
        _with(
            q("How did monthly card spend move?", complexity="medium"),
            criteria=["shows the monthly card spend trend", "ranks the spenders"],
        ),
    )
    llm = FakeLLMClient([copied, fixed])
    result = await env.run(llm)
    assert "success_criteria are copied word for word from question 1" in llm.calls[1].user
    assert len(result.questions) == 2


async def test_previous_questions_are_diversity_context_in_the_same_call(env: Env) -> None:
    first = await env.run(
        FakeLLMClient(
            [batch(_with(q("Who were our top card spenders last quarter?"), filters=PERIOD))]
        ),
        count=1,
    )
    previous = {"SCN_001": first.questions}
    repeat = batch(_with(q("Who were the top card spenders for last quarter?"), filters=PERIOD))
    new = batch(
        _with(
            q("Which employees spent most on cards this year?"),
            filters=[{"name": "period", "value": "year to date"}],
        )
    )
    llm = FakeLLMClient([repeat, new])
    generator = QuestionGenerator(
        llm, env.settings, logging.getLogger("test"), id_generator=SequentialIdGenerator()
    )
    result = await generator.generate(
        env.persona,
        env.product,
        env.scenarios[:1],
        env.reference,
        env.packs,
        count=1,
        previous=previous,
    )
    prompt = llm.calls[0].user
    assert "Questions already generated for this scenario" in prompt
    assert "Who were our top card spenders last quarter?" in prompt
    assert "Still write 1 NEW question(s)" in " ".join(prompt.split())
    assert "near-duplicate of earlier question Q_001" in llm.calls[1].user
    assert [x.utterance for x in result.questions] == [
        "Which employees spent most on cards this year?"
    ]


def test_diversity_rules_in_the_prompt(
    sample_persona: Persona,
    sample_product: ProductDescription,
    sample_scenarios: list[Scenario],
    sample_reference: Reference,
    sample_packs: list[ReferencePack],
) -> None:
    def prompt(count: int) -> str:
        options = QuestionPromptOptions(
            count=count,
            ambiguity_level=AmbiguityLevel.LOW,
            language="en",
            allow_vague_questions=False,
        )
        _system, user = build_question_prompt(
            sample_persona,
            sample_product,
            sample_scenarios[0],
            (sample_packs[0], sample_reference),
            options,
        )
        return " ".join(user.split())

    several = prompt(3)
    for text in (
        "the 3 questions must differ meaningfully, never only by paraphrase or word substitution",
        "ambiguity (within the persona's allowed style mix), terminology",
        "filters, time expressions",
        "grouping or breakdown, and phrasing",
        "Diversity never changes the scenario's business goal or capability",
        "the persona's characteristics, the hard policy or the allowed complexities",
        "Never add a deliverable, an unsupported requirement or a fact that depends on the",
        "describe THAT question's own utterance",
    ):
        assert text in several, text
    single = prompt(1)
    assert "DIVERSITY" not in single and "Questions already generated" not in single
    assert "describe this exact utterance" in single


async def test_diversity_check_can_be_switched_off(env: Env) -> None:
    same = batch(
        _with(q("Who were our top card spenders last quarter?"), filters=PERIOD),
        _with(q("Who were the top card spenders last quarter?"), filters=PERIOD),
    )
    llm = FakeLLMClient([same])
    result = await env.run(llm, diversity_check=False)
    assert len(llm.calls) == 1 and len(result.questions) == 2
