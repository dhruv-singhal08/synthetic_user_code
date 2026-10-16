"""Node 1: build a ReferenceIndex for a catalog with the LLM, grounded by integrity checks."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import LLMOutputValidationError, ReferenceIntegrityError
from synthetic_user.core.identifiers import new_index_id
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.reference import (
    ReferenceCatalog,
    ReferenceIndex,
    ReferenceIndexDraft,
    Relationship,
    TableMetadata,
)
from synthetic_user.prompts.reference_index import build_reference_index_prompt
from synthetic_user.references.integrity import (
    check_index_against_catalog,
    check_tables_against_catalog,
    enforce_relationship_verification,
)

REFERENCE_INDEX_VERSION = "1"

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ReferenceIndexGenerator:
    def __init__(
        self,
        llm: LLMClient,
        settings: AppConfig,
        logger: Logger,
        *,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[datetime], str] = lambda now: new_index_id(now=now),
    ) -> None:
        self._llm = llm
        self._settings = settings
        self._log = logger
        self._clock = clock
        self._id_factory = id_factory

    async def generate(self, catalog: ReferenceCatalog) -> ReferenceIndex:
        batches = _chunks(catalog.tables, self._settings.reference_index.max_tables_per_call)
        self._log.info(
            "Building reference index for %s v%s: %d table(s) in %d call(s)",
            catalog.catalog_id,
            catalog.catalog_version,
            len(catalog.tables),
            len(batches),
        )
        semaphore = asyncio.Semaphore(self._settings.llm.max_concurrency)
        drafts = await asyncio.gather(
            *(self._generate_batch(catalog, batch, semaphore) for batch in batches)
        )

        by_name = {t.name: t for draft in drafts for t in draft.tables}
        now = self._clock()
        index = ReferenceIndex(
            index_id=self._id_factory(now),
            index_version=REFERENCE_INDEX_VERSION,
            source_catalog_id=catalog.catalog_id,
            source_catalog_version=catalog.catalog_version,
            tables=[by_name[t.name] for t in catalog.tables],
            generated_at=now,
            llm_provider=self._llm.provider_name,
            llm_model=self._llm.model_name,
            uncertainties=[u for draft in drafts for u in draft.uncertainties],
        )
        index = enforce_relationship_verification(index, catalog)
        errors = check_index_against_catalog(index, catalog)
        if errors:
            raise ReferenceIntegrityError(f"Reference index failed integrity: {'; '.join(errors)}")
        self._log.info(
            "Reference index %s built (%d uncertainties)", index.index_id, len(index.uncertainties)
        )
        return index

    async def _generate_batch(
        self,
        catalog: ReferenceCatalog,
        batch: Sequence[TableMetadata],
        semaphore: asyncio.Semaphore,
    ) -> ReferenceIndexDraft:
        names = {t.name for t in batch}
        system, user = build_reference_index_prompt(batch, _relationships_for(catalog, names))

        def post_validate(draft: ReferenceIndexDraft) -> list[str]:
            return check_tables_against_catalog(draft.tables, catalog, expected_tables=names)

        async with semaphore:
            try:
                return await generate_structured(
                    self._llm,
                    system,
                    user,
                    ReferenceIndexDraft,
                    max_retries=self._settings.llm.max_validation_retries,
                    post_validate=post_validate,
                )
            except LLMOutputValidationError as exc:
                raise ReferenceIntegrityError(
                    f"Reference index for tables {sorted(names)} rejected after "
                    f"{exc.attempts} attempt(s): {'; '.join(exc.errors)}"
                ) from exc


def _chunks(tables: Sequence[TableMetadata], size: int) -> list[list[TableMetadata]]:
    return [list(tables[i : i + size]) for i in range(0, len(tables), size)]


def _relationships_for(catalog: ReferenceCatalog, names: set[str]) -> list[Relationship]:
    return [r for r in catalog.relationships if r.source_table in names or r.target_table in names]
