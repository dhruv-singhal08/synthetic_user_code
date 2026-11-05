from __future__ import annotations

import asyncio
import inspect
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from synthetic_user.core.config import AppConfig, load_config
from synthetic_user.core.loaders import load_persona, load_product
from synthetic_user.metrics.registry import builtin_metric_config
from synthetic_user.models.metrics import ResolvedMetricConfig
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import CatalogPackDraft, Reference, ReferencePack
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario
from synthetic_user.references.loader import load_reference, parse_reference
from synthetic_user.references.pack import build_pack
from synthetic_user.storage.json_store import JsonArtifactStore
from synthetic_user.storage.runs import run_artifact
from tests_support import (
    AGENT_CONTEXT,
    TOOL_CATALOG,
    checks_for,
    failed_check,
    goal_for,
    scripted_retrieval_reply,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_CONFIG = REPO_ROOT / "config" / "app.yaml"
REFERENCES = REPO_ROOT / "data" / "references"
SAMPLE_REFERENCE = REFERENCES / "sample_card_spend_catalog.yaml"
SAMPLE_GUIDE = REFERENCES / "sample_card_spend_instructions.md"
SAMPLE_POLICY = REFERENCES / "sample_travel_policy_docs.yaml"
SAMPLE_POLICY_GUIDE = REFERENCES / "sample_travel_policy_instructions.md"
MAX_CHARS = 200_000

# Cheat-sheet picks against data/references/sample_card_spend_catalog.yaml.
EMPLOYEE_JOIN = {"from": "card_transactions.employee_id", "to": "employees.employee_id"}


@pytest.fixture(autouse=True)
def _close_leftover_event_loop(request: pytest.FixtureRequest) -> None:
    """pytest-asyncio 0.24 (pinned for Amex, D119) installs a fresh, never-closed loop after
    each async test; a later sync test's asyncio.run() drops it and GC warns. Close it first."""
    if inspect.iscoroutinefunction(request.function):
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            return
    if not loop.is_running():
        loop.close()
        asyncio.set_event_loop(None)


DEFAULT_TEST_METRICS = ("relevance", "correctness_completeness")


def seed_metric_config(
    store: JsonArtifactStore, run_id: str, metric_ids: tuple[str, ...] = DEFAULT_TEST_METRICS
) -> ResolvedMetricConfig:
    """Write a built-in-only frozen metric_config.json into a hand-built test run."""
    config = builtin_metric_config(list(metric_ids))
    store.write_model(run_artifact(run_id, "metric_config.json"), config)
    return config


@pytest.fixture
def seed_config() -> Callable[..., ResolvedMetricConfig]:
    """``seed_config(store, run_id, metric_ids)`` -> writes a frozen built-in metric config."""
    return seed_metric_config


@pytest.fixture
def app_config() -> AppConfig:
    return load_config(REPO_CONFIG, env={})


@pytest.fixture
def tool_reference() -> Reference:
    """A data_catalog whose agent_context defines the canonical tools and data sources."""
    import yaml

    text = TOOL_CATALOG + yaml.safe_dump({"agent_context": AGENT_CONTEXT})
    return parse_reference(text.encode(), source_name="voc.yaml", max_chars=MAX_CHARS)


@pytest.fixture
def sample_reference() -> Reference:
    """The sample card-spend data_catalog + its usage instructions, loaded like a user's."""
    return load_reference(SAMPLE_REFERENCE, max_chars=MAX_CHARS, guide_path=SAMPLE_GUIDE)


@pytest.fixture
def policy_reference() -> Reference:
    """A document_collection: the sample travel policy documents + usage instructions."""
    return load_reference(SAMPLE_POLICY, max_chars=MAX_CHARS, guide_path=SAMPLE_POLICY_GUIDE)


SAMPLE_PERSONA = REPO_ROOT / "data" / "personas" / "finance_controller.yaml"
SAMPLE_PRODUCT = REPO_ROOT / "data" / "products" / "sample_product.yaml"


@pytest.fixture
def sample_persona() -> Persona:
    return load_persona(SAMPLE_PERSONA)


@pytest.fixture
def sample_product() -> ProductDescription:
    return load_product(SAMPLE_PRODUCT)


@pytest.fixture
def sample_chatbot() -> ProductDescription:
    """A non-SQL product (text answers only)."""
    return load_product(REPO_ROOT / "data" / "products" / "sample_chatbot.yaml")


@pytest.fixture
def scenario_drafts() -> list[dict[str, Any]]:
    """Three distinct, valid grounded scenario drafts for the sample reference:
    [0] supported ranking, [1] supported trend, [2] unsupported."""
    return [
        {
            "business_goal": "Identify which employees drive the most corporate card spend "
            "so finance can review outliers",
            "title": "Top spenders by employee",
            "success_criteria": [
                "names the employees ranked by total card spend",
                "keeps amounts separate per currency or states which currency is used",
                "covers the requested time period",
            ],
            "task_type": "ranking",
            "capability": "supported",
            "capability_rationale": "Product supports 'ranking and top-N' and 'joins over "
            "documented relationships'",
            "reference_pack": {
                "tables": [
                    {
                        "table": "card_transactions",
                        "columns": ["amount", "currency_code"],
                        "why": "spend is the sum of amount, kept per currency",
                    },
                    {
                        "table": "employees",
                        "columns": ["full_name"],
                        "why": "the ranking is by employee",
                    },
                ],
                "joins": [EMPLOYEE_JOIN],
                "rules": ["R1"],
            },
            "question_intent": "rank employees by total card spend",
        },
        {
            "business_goal": "Track how monthly card spend per merchant category evolves "
            "to spot budget drift",
            "title": "Monthly spend trend by merchant category",
            "success_criteria": [
                "shows card spend per merchant category for each month",
                "covers the requested period",
            ],
            "task_type": "trend",
            "capability": "supported",
            "capability_rationale": "Product supports 'single-table aggregation' and "
            "'filtering by date range'",
            "reference_pack": {
                "tables": [
                    {
                        "table": "card_transactions",
                        "columns": ["amount", "transaction_date", "merchant_category"],
                        "why": "monthly spend per category comes from the transactions",
                    }
                ]
            },
            "question_intent": "monthly spend trend by category",
        },
        {
            "business_goal": "Forecast next quarter's card spend to plan the budget",
            "title": "Card spend forecast",
            "success_criteria": [
                "states clearly whether a forecast can be produced",
                "does not present invented future figures as facts",
            ],
            "task_type": "other",
            "capability": "unsupported",
            "capability_rationale": "Product lists 'forecasting' under not_supported_yet",
            "reference_pack": {
                "tables": [
                    {"table": "card_transactions", "why": "history a forecast would start from"}
                ]
            },
            "question_intent": "forecast spend",
        },
    ]


@pytest.fixture
def sample_packs(
    scenario_drafts: list[dict[str, Any]], sample_reference: Reference
) -> list[ReferencePack]:
    """The drafts' packs as accepted ReferencePacks RP_SCN_001..RP_SCN_003."""
    packs = []
    for i, draft in enumerate(scenario_drafts, start=1):
        pack, _ = build_pack(
            CatalogPackDraft.model_validate(draft["reference_pack"]),
            sample_reference,
            f"SCN_{i:03d}",
        )
        assert pack is not None
        packs.append(pack)
    return packs


@pytest.fixture
def sample_scenarios(
    scenario_drafts: list[dict[str, Any]], sample_reference: Reference
) -> list[Scenario]:
    """The three scenario drafts as accepted grounded Scenarios SCN_001..SCN_003."""
    return [
        Scenario(
            id=f"SCN_{i:03d}",
            persona_id="finance_controller",
            product_id="sample_nl2sql",
            mode=GenerationMode.GROUNDED,
            reference_id=sample_reference.reference_id,
            reference_pack_id=f"RP_SCN_{i:03d}",
            signature=f"sig-{i}",
            **{k: v for k, v in draft.items() if k != "reference_pack"},
        )
        for i, draft in enumerate(scenario_drafts, start=1)
    ]


Judge = Callable[[str, str], dict[str, Any]]


def general_metrics_in(user: str) -> list[str]:
    """The metric ids a General Judge prompt asks for."""
    line = user.split("on exactly these metric(s): ", 1)[1].split("\n", 1)[0]
    return [m.strip().rstrip(".") for m in line.split(",")]


def scripted_judge(
    fail_on: str | None = None, *, failing: str | None = None, classification: str = "answered"
) -> Judge:
    """A FakeLLM responder for both judge prompts.

    General Judge: every requested metric scores 1 (a passed check for every requirement its
    evaluation contract assigns), except
    answers containing ``fail_on``: then ``failing`` (default: semantic_correctness when
    requested, else correctness_completeness, else the first metric) scores 0 with a supported
    finding and a failed derived check. The response classification is returned when the
    prompt asks for it.
    Retrieval Judge: every requested metric 1, every chunk relevant, checklist covered.
    """

    def judge(system: str, user: str) -> dict[str, Any]:
        if "Retrieval and Grounding Judge" in system:
            return scripted_retrieval_reply(user)
        metrics = general_metrics_in(user)
        answer = user.split("Product response (", 1)[1]
        bad = fail_on is not None and fail_on in answer
        target = failing or next(
            (m for m in ("semantic_correctness", "correctness_completeness") if m in metrics),
            metrics[0],
        )
        failed = {"score": 0, "rationale": "wrong", "evidence": fail_on, "what_didnt": ["wrong"]}
        ok = {"score": 1, "rationale": "fine", "what_worked": ["fine"]}
        scores: dict[str, Any] = {}
        for metric in metrics:
            score = dict(failed if bad and metric == target else ok)
            score["checks"] = checks_for(user, metric)
            if bad and metric == target:
                score["checks"].append(failed_check(metric))
            if (
                f"- {metric} (" in user
                and "[factual verification required]"
                in user.split(f"- {metric} (", 1)[1].split("\n", 1)[0]
            ):
                score["factual_verification"] = "unverified"
            scores[metric] = score
        reply: dict[str, Any] = {
            "metrics": scores,
            "findings": [
                {
                    "category": target,
                    "description": "the answer is wrong",
                    "expected_behavior": "a correct answer",
                    "observed_behavior": "a wrong answer",
                    "evidence": fail_on,
                    "severity": "high",
                    "evidence_status": "supported",
                    "related_tags": [],
                }
            ]
            if bad
            else [],
            "overall_verdict": "fail" if bad else "pass",
        }
        asked = "Also classify the response kind" in user
        if asked:
            reply["response_classification"] = classification
        reply["goal_completion"] = goal_for(user, classification if asked else None)
        return reply

    return judge


@pytest.fixture
def judge_factory() -> Callable[..., Judge]:
    """``judge_factory(fail_on=...)`` -> a scripted evaluation judge (see scripted_judge)."""
    return scripted_judge
