"""Node 2: generate scenarios + reference cheat sheets (one LLM call per round).

Per-scenario deterministic checks decide accept / reject / duplicate. Short results are
topped up at most ``max_topup_rounds`` times and never padded.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from pydantic import Field, ValidationError

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import format_validation_error
from synthetic_user.core.identifiers import IdKind, SequentialIdGenerator
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.base import StrictModel
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import ReferenceIndex
from synthetic_user.models.scenario import (
    CapabilityClassification,
    ReferenceCheatSheet,
    RejectedScenario,
    Scenario,
    ScenarioDraft,
    ScenarioGenerationResult,
    TaskType,
)
from synthetic_user.prompts.scenario import ScenarioPromptOptions, build_scenario_prompt
from synthetic_user.references.cheat_sheet import (
    align_relationship_verification,
    check_cheat_sheet_against_index,
)

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

FILTERED_BY_CONFIG = "filtered_by_config"
SURPLUS = "surplus: more scenarios returned than requested"

_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "which",
        "what",
        "into",
        "over",
        "per",
        "their",
        "they",
        "them",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "our",
        "you",
        "your",
        "can",
        "will",
        "should",
        "would",
        "been",
        "being",
        "its",
        "out",
        "all",
        "any",
        "each",
        "how",
        "who",
        "whom",
        "why",
        "when",
        "where",
        "than",
        "then",
        "also",
        "only",
        "more",
        "most",
        "less",
        "least",
        "very",
    }
)


class _RawScenarioBatch(StrictModel):
    """Parsed per call; each scenario is validated individually so one bad item is
    rejected on its own instead of failing the whole batch."""

    scenarios: list[dict[str, Any]] = Field(default_factory=list)


def scenario_signature(task_type: TaskType, business_goal: str, cheat: ReferenceCheatSheet) -> str:
    """task_type | goal keywords | tables | metrics | dimensions | operations (all sorted)."""
    keywords = sorted(
        {
            w
            for w in re.findall(r"[a-z0-9]+", business_goal.lower())
            if len(w) > 2 and w not in _STOPWORDS
        }
    )
    parts = [
        task_type.value,
        " ".join(keywords),
        ",".join(sorted(set(cheat.tables))),
        ",".join(sorted({m.name.strip().lower() for m in cheat.metrics})),
        ",".join(sorted(set(cheat.dimensions))),
        ",".join(sorted({o.strip().lower() for o in cheat.operations})),
    ]
    return "|".join(p.replace("|", "/") for p in parts)


def _shape(signature: str) -> tuple[str, ...] | None:
    """(task_type, tables, metrics, dimensions) from a signature, or None if malformed."""
    parts = signature.split("|")
    return (parts[0], parts[2], parts[3], parts[4]) if len(parts) == 6 else None


class ScenarioGenerator:
    def __init__(
        self,
        llm: LLMClient,
        settings: AppConfig,
        logger: Logger,
        *,
        id_generator: SequentialIdGenerator,
    ) -> None:
        self._llm = llm
        self._settings = settings
        self._log = logger
        self._ids = id_generator

    async def generate(
        self,
        persona: Persona,
        product: ProductDescription,
        index: ReferenceIndex,
        *,
        count: int | None = None,
        existing_signatures: Sequence[str] = (),
    ) -> ScenarioGenerationResult:
        cfg = self._settings.generation.scenario
        requested = count if count is not None else cfg.scenario_count
        if requested < 1:
            raise ValueError("scenario count must be >= 1")

        state = _DedupState(existing_signatures)
        result = ScenarioGenerationResult()
        for round_no in range(1 + cfg.max_topup_rounds):
            needed = requested - len(result.accepted)
            if needed <= 0:
                break
            options = ScenarioPromptOptions(
                count=needed,
                allow_multi_table=cfg.allow_multi_table,
                include_unsupported_scenarios=cfg.include_unsupported_scenarios,
                diversity_preference=cfg.diversity_preference,
            )
            system, user = build_scenario_prompt(
                persona, product, index, options, avoid_signatures=state.signatures()
            )
            batch = await generate_structured(
                self._llm,
                system,
                user,
                _RawScenarioBatch,
                max_retries=self._settings.llm.max_validation_retries,
            )
            for raw in batch.scenarios:
                if len(result.accepted) >= requested:
                    result.rejected.append(RejectedScenario(payload=raw, reasons=[SURPLUS]))
                    continue
                self._process(raw, persona, product, index, state, result)
            self._log.info(
                "Scenario round %d: %d returned; totals accepted=%d rejected=%d duplicates=%d",
                round_no + 1,
                len(batch.scenarios),
                len(result.accepted),
                len(result.rejected),
                len(result.duplicates),
            )
        if len(result.accepted) < requested:
            self._log.warning(
                "Only %d of %d requested scenarios accepted (not padded)",
                len(result.accepted),
                requested,
            )
        return result

    def _process(
        self,
        raw: dict[str, Any],
        persona: Persona,
        product: ProductDescription,
        index: ReferenceIndex,
        state: _DedupState,
        result: ScenarioGenerationResult,
    ) -> None:
        cfg = self._settings.generation.scenario
        try:
            draft = ScenarioDraft.model_validate(raw)
        except ValidationError as exc:
            result.rejected.append(
                RejectedScenario(payload=raw, reasons=[format_validation_error(exc)])
            )
            return

        reasons = check_cheat_sheet_against_index(draft.cheat_sheet, index)
        if not cfg.allow_multi_table and len(set(draft.cheat_sheet.tables)) > 1:
            reasons.append(f"multi_table_not_allowed: uses {sorted(set(draft.cheat_sheet.tables))}")
        if (
            not cfg.include_unsupported_scenarios
            and draft.capability is CapabilityClassification.UNSUPPORTED
        ):
            reasons.append(FILTERED_BY_CONFIG)
        if reasons:
            result.rejected.append(RejectedScenario(payload=raw, reasons=reasons))
            return

        cheat, _notes = align_relationship_verification(draft.cheat_sheet, index)
        signature = scenario_signature(draft.task_type, draft.business_goal, cheat)
        duplicate_of = state.find(signature)
        if duplicate_of is not None:
            result.duplicates.append(
                RejectedScenario(payload=raw, reasons=[f"duplicate of {duplicate_of}: {signature}"])
            )
            return

        scenario = Scenario(
            id=self._ids.next(IdKind.SCENARIO),
            persona_id=persona.id,
            product_id=product.product_id,
            reference_index_id=index.index_id,
            business_goal=draft.business_goal,
            title=draft.title,
            task_type=draft.task_type,
            capability=draft.capability,
            capability_rationale=draft.capability_rationale,
            cheat_sheet=cheat,
            question_guidance=draft.question_guidance,
            signature=signature,
        )
        state.add(signature, scenario.id)
        result.accepted.append(scenario)


class _DedupState:
    """Signatures and (task, tables, metrics, dimensions) shapes seen so far."""

    def __init__(self, existing: Sequence[str]) -> None:
        self._by_signature: dict[str, str] = {}
        self._by_shape: dict[tuple[str, ...], str] = {}
        for signature in existing:
            self.add(signature, "existing scenario")

    def add(self, signature: str, owner: str) -> None:
        self._by_signature.setdefault(signature, owner)
        shape = _shape(signature)
        if shape is not None:
            self._by_shape.setdefault(shape, owner)

    def find(self, signature: str) -> str | None:
        if signature in self._by_signature:
            return self._by_signature[signature]
        shape = _shape(signature)
        return self._by_shape.get(shape) if shape is not None else None

    def signatures(self) -> list[str]:
        return list(self._by_signature)
