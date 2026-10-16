from __future__ import annotations

import copy
import logging
from datetime import UTC, datetime
from typing import Any

import pytest

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import ReferenceIntegrityError
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.models.reference import (
    ReferenceCatalog,
    ReferenceIndex,
    VerificationStatus,
)
from synthetic_user.nodes.reference_index import REFERENCE_INDEX_VERSION, ReferenceIndexGenerator
from synthetic_user.prompts.reference_index import build_reference_index_prompt

NOW = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)


def _generator(llm: FakeLLMClient, settings: AppConfig) -> ReferenceIndexGenerator:
    return ReferenceIndexGenerator(
        llm,
        settings,
        logging.getLogger("test"),
        clock=lambda: NOW,
        id_factory=lambda now: "IDX_20261013_120000_abcd",
    )


def _table(draft: dict[str, Any], name: str) -> dict[str, Any]:
    return next(t for t in draft["tables"] if t["name"] == name)


def _with_retries(settings: AppConfig, retries: int) -> AppConfig:
    llm = settings.llm.model_copy(update={"max_validation_retries": retries})
    return settings.model_copy(update={"llm": llm})


async def test_names_preserved_and_code_sets_metadata(
    app_config: AppConfig,
    sample_catalog: ReferenceCatalog,
    sample_draft: dict[str, Any],
    sample_index: ReferenceIndex,
) -> None:
    llm = FakeLLMClient([sample_draft], model_name="fake-model")
    index = await _generator(llm, app_config).generate(sample_catalog)

    assert [t.name for t in index.tables] == [t.name for t in sample_catalog.tables]
    assert index.index_id == "IDX_20261013_120000_abcd"
    assert index.index_version == REFERENCE_INDEX_VERSION
    assert index.generated_at == NOW
    assert (index.source_catalog_id, index.source_catalog_version) == ("sample_card_spend", "1.0")
    assert (index.llm_provider, index.llm_model) == ("fake", "fake-model")
    assert index.tables == sample_index.tables
    assert len(llm.calls) == 1


async def test_tables_returned_out_of_order_are_reordered(
    app_config: AppConfig, sample_catalog: ReferenceCatalog, sample_draft: dict[str, Any]
) -> None:
    sample_draft["tables"].reverse()
    index = await _generator(FakeLLMClient([sample_draft]), app_config).generate(sample_catalog)
    assert [t.name for t in index.tables] == [t.name for t in sample_catalog.tables]


async def test_invented_column_rejected(
    app_config: AppConfig, sample_catalog: ReferenceCatalog, sample_draft: dict[str, Any]
) -> None:
    _table(sample_draft, "card_transactions")["important_columns"].append("amount_usd")
    llm = FakeLLMClient([sample_draft] * 3)
    with pytest.raises(ReferenceIntegrityError) as exc_info:
        await _generator(llm, app_config).generate(sample_catalog)
    assert "after 3 attempt(s)" in str(exc_info.value)
    assert "'amount_usd' is not a catalog column" in str(exc_info.value)
    assert len(llm.calls) == 3


async def test_fixed_on_retry_passes(
    app_config: AppConfig, sample_catalog: ReferenceCatalog, sample_draft: dict[str, Any]
) -> None:
    bad = copy.deepcopy(sample_draft)
    _table(bad, "employees")["dimensions"].append("department_name")
    llm = FakeLLMClient([bad, sample_draft])
    index = await _generator(llm, app_config).generate(sample_catalog)
    assert "department_name" not in index.tables[1].dimensions
    assert len(llm.calls) == 2
    assert "previous output invalid:" in llm.calls[1].user
    assert "found on departments" in llm.calls[1].user


async def test_verified_upgrade_downgraded(
    app_config: AppConfig, sample_catalog: ReferenceCatalog, sample_draft: dict[str, Any]
) -> None:
    _table(sample_draft, "employees")["relationships"][0]["verification"] = "verified"
    index = await _generator(FakeLLMClient([sample_draft]), app_config).generate(sample_catalog)
    rel = index.tables[1].relationships[0]
    assert rel.verification is VerificationStatus.UNKNOWN
    assert any("downgraded to unknown" in u for u in index.uncertainties)


async def test_no_relationships_ok(
    app_config: AppConfig, sample_catalog: ReferenceCatalog, sample_draft: dict[str, Any]
) -> None:
    catalog = sample_catalog.model_copy(update={"relationships": []})
    for table in sample_draft["tables"]:
        table["relationships"] = []
    llm = FakeLLMClient([sample_draft])
    index = await _generator(llm, app_config).generate(catalog)
    assert all(not t.relationships for t in index.tables)
    assert '"relationships": []' in llm.calls[0].user


async def test_bad_json_rejected(app_config: AppConfig, sample_catalog: ReferenceCatalog) -> None:
    llm = FakeLLMClient(["Sure! Here is the index:", '{"tables": [', "```json\n{oops}\n```"])
    with pytest.raises(ReferenceIntegrityError, match="after 3 attempt"):
        await _generator(llm, app_config).generate(sample_catalog)


async def test_missing_table_rejected(
    app_config: AppConfig, sample_catalog: ReferenceCatalog, sample_draft: dict[str, Any]
) -> None:
    sample_draft["tables"] = [t for t in sample_draft["tables"] if t["name"] != "merchants"]
    llm = FakeLLMClient([sample_draft])
    with pytest.raises(ReferenceIntegrityError, match="missing table 'merchants'"):
        await _generator(llm, _with_retries(app_config, 0)).generate(sample_catalog)


async def test_llm_cannot_supply_ids_or_timestamps(
    app_config: AppConfig, sample_catalog: ReferenceCatalog, sample_draft: dict[str, Any]
) -> None:
    sample_draft["index_id"] = "IDX_LLM"
    llm = FakeLLMClient([sample_draft])
    with pytest.raises(ReferenceIntegrityError, match="Extra inputs are not permitted"):
        await _generator(llm, _with_retries(app_config, 0)).generate(sample_catalog)


async def test_batches_when_over_max_tables_per_call(
    app_config: AppConfig, sample_catalog: ReferenceCatalog, sample_draft: dict[str, Any]
) -> None:
    settings = app_config.model_copy(
        update={
            "reference_index": app_config.reference_index.model_copy(
                update={"max_tables_per_call": 2}
            )
        }
    )

    def respond(system: str, user: str) -> dict[str, Any]:
        wanted = [t for t in sample_draft["tables"] if f'"name": "{t["name"]}"' in user]
        return {"tables": wanted, "uncertainties": [f"batch of {len(wanted)}"]}

    llm = FakeLLMClient([respond, respond])
    index = await _generator(llm, settings).generate(sample_catalog)
    assert len(llm.calls) == 2
    assert "card_transactions, employees." in llm.calls[0].user
    assert "departments, merchants." in llm.calls[1].user
    assert [t.name for t in index.tables] == [t.name for t in sample_catalog.tables]
    assert index.uncertainties == ["batch of 2", "batch of 2"]


# --- prompt ------------------------------------------------------------------------------


def test_prompt_sections_and_contract(sample_catalog: ReferenceCatalog) -> None:
    system, user = build_reference_index_prompt(sample_catalog.tables, sample_catalog.relationships)
    assert "JSON" in system
    for section in (
        "## TASK",
        "## INPUT",
        "## CONSTRAINTS",
        "## OUTPUT CONTRACT",
        "## PROHIBITED",
        "## MISSING INFO",
    ):
        assert section in user
    contract = user.split("## OUTPUT CONTRACT", 1)[1].split("## PROHIBITED", 1)[0]
    for forbidden in ("index_id", "index_version", "generated_at", "llm_provider", "llm_model"):
        assert forbidden not in contract
    for table in sample_catalog.tables:
        for column in table.columns:
            assert f'"{column.name}"' in user
    assert '"unknown"' in user
