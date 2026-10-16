from __future__ import annotations

import asyncio
import copy
import inspect
import warnings
from pathlib import Path
from typing import Any

import pytest

from synthetic_user.core.config import AppConfig, load_config
from synthetic_user.core.loaders import load_catalog, load_persona, load_product
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import ReferenceCatalog, ReferenceIndex
from synthetic_user.models.scenario import Scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_CONFIG = REPO_ROOT / "config" / "app.yaml"
SAMPLE_CATALOG = REPO_ROOT / "data" / "references" / "sample_catalog.yaml"
SAMPLE_INDEX = REPO_ROOT / "tests" / "fixtures" / "sample_reference_index.json"


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


@pytest.fixture
def app_config() -> AppConfig:
    return load_config(REPO_CONFIG, env={})


@pytest.fixture
def sample_catalog() -> ReferenceCatalog:
    return load_catalog(SAMPLE_CATALOG)


@pytest.fixture
def sample_index() -> ReferenceIndex:
    return ReferenceIndex.model_validate_json(SAMPLE_INDEX.read_text(encoding="utf-8"))


@pytest.fixture
def sample_draft(sample_index: ReferenceIndex) -> dict[str, Any]:
    """What a well-behaved LLM returns for the sample catalog (no ids/versions/timestamps)."""
    return copy.deepcopy(sample_index.model_dump(mode="json", include={"tables", "uncertainties"}))


SAMPLE_PERSONA = REPO_ROOT / "data" / "personas" / "finance_controller.yaml"
SAMPLE_PRODUCT = REPO_ROOT / "data" / "products" / "sample_product.yaml"


@pytest.fixture
def sample_persona() -> Persona:
    return load_persona(SAMPLE_PERSONA)


@pytest.fixture
def sample_product() -> ProductDescription:
    return load_product(SAMPLE_PRODUCT)


@pytest.fixture
def scenario_drafts() -> list[dict[str, Any]]:
    """Three distinct, valid scenario drafts for the sample index:
    [0] supported multi-table ranking, [1] supported single-table trend, [2] unsupported."""
    return [
        {
            "business_goal": "Identify which employees drive the most corporate card spend "
            "so finance can review outliers",
            "title": "Top spenders by employee",
            "task_type": "ranking",
            "capability": "supported",
            "capability_rationale": "Product supports 'ranking and top-N' and 'joins over "
            "documented relationships'",
            "cheat_sheet": {
                "tables": ["card_transactions", "employees"],
                "columns": [
                    "card_transactions.amount",
                    "card_transactions.currency_code",
                    "employees.full_name",
                ],
                "relationships": [
                    {
                        "source": "card_transactions.employee_id",
                        "target": "employees.employee_id",
                        "verification": "verified",
                    }
                ],
                "metrics": [
                    {
                        "name": "total_spend",
                        "definition": "sum of amount within one currency",
                        "columns": ["card_transactions.amount"],
                    }
                ],
                "dimensions": ["employees.full_name", "card_transactions.currency_code"],
                "operations": ["join", "group by", "sum", "order by desc", "limit"],
                "optional_filters": ["transaction_date range"],
                "semantic_considerations": ["amounts are in mixed currencies"],
            },
            "question_guidance": {
                "intent": "rank employees by total card spend",
                "expected_ambiguity": "medium",
            },
        },
        {
            "business_goal": "Track how monthly card spend per merchant category evolves "
            "to spot budget drift",
            "title": "Monthly spend trend by merchant category",
            "task_type": "trend",
            "capability": "supported",
            "capability_rationale": "Product supports 'single-table aggregation' and "
            "'filtering by date range'",
            "cheat_sheet": {
                "tables": ["card_transactions"],
                "columns": [
                    "card_transactions.amount",
                    "card_transactions.transaction_date",
                    "card_transactions.merchant_category",
                ],
                "metrics": [
                    {
                        "name": "total_spend",
                        "definition": "sum of amount per month",
                        "columns": ["card_transactions.amount"],
                    }
                ],
                "dimensions": ["card_transactions.merchant_category"],
                "operations": ["group by month", "sum"],
            },
            "question_guidance": {"intent": "monthly spend trend by category"},
        },
        {
            "business_goal": "Forecast next quarter's card spend to plan the budget",
            "title": "Card spend forecast",
            "task_type": "other",
            "capability": "unsupported",
            "capability_rationale": "Product lists 'forecasting' under not_supported_yet",
            "cheat_sheet": {
                "tables": ["card_transactions"],
                "columns": ["card_transactions.amount", "card_transactions.transaction_date"],
                "metrics": [
                    {
                        "name": "projected_spend",
                        "definition": "future spend projection",
                        "columns": ["card_transactions.amount"],
                    }
                ],
                "dimensions": ["card_transactions.transaction_date"],
                "operations": ["forecast"],
            },
            "question_guidance": {"intent": "forecast spend"},
        },
    ]


@pytest.fixture
def sample_scenarios(
    scenario_drafts: list[dict[str, Any]], sample_index: ReferenceIndex
) -> list[Scenario]:
    """The three scenario drafts as accepted Scenarios SCN_001..SCN_003."""
    return [
        Scenario(
            id=f"SCN_{i:03d}",
            persona_id="finance_controller",
            product_id="sample_nl2sql",
            reference_index_id=sample_index.index_id,
            signature=f"sig-{i}",
            **draft,
        )
        for i, draft in enumerate(scenario_drafts, start=1)
    ]
