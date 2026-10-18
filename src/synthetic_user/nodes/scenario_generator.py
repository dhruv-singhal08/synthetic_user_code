"""Node 2: generate scenarios (one LLM call per round), in two modes.

grounded (reference given): each scenario comes with a reference_pack built in the same call;
code keeps only core_refs / rules whose quote is in the reference (the others are recorded in
the pack's rejected_items) and rejects the scenario when no core_ref survives.
ungrounded (no reference): a pack must be absent (rejected otherwise), unsupported filter.
Per-scenario checks decide accept / reject / duplicate. Short results are topped up at most
``max_topup_rounds`` times and never padded.
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
from synthetic_user.models.reference import Reference, ReferencePack, pack_id_for
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import (
    CapabilityClassification,
    RejectedScenario,
    Scenario,
    ScenarioDraft,
    ScenarioGenerationResult,
    TaskType,
)
from synthetic_user.prompts.scenario import ScenarioPromptOptions, build_scenario_prompt
from synthetic_user.references.pack import NO_VALID_CORE_REF, build_pack

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

FILTERED_BY_CONFIG = "filtered_by_config"
SURPLUS = "surplus: more scenarios returned than requested"
MISSING_PACK = "mode grounded requires a reference_pack"
UNEXPECTED_PACK = "mode ungrounded must not have a reference_pack"

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


def _goal_keywords(business_goal: str) -> str:
    words = re.findall(r"[a-z0-9]+", business_goal.lower())
    return " ".join(sorted({w for w in words if len(w) > 2 and w not in _STOPWORDS}))


def scenario_signature(
    task_type: TaskType,
    business_goal: str,
    capability: CapabilityClassification,
    pack: ReferencePack,
) -> str:
    """Grounded: task_type | goal keywords | capability | core_ref names."""
    parts = [
        task_type.value,
        _goal_keywords(business_goal),
        capability.value,
        ",".join(sorted({r.name.strip().lower() for r in pack.core_refs})),
    ]
    return "|".join(p.replace("|", "/") for p in parts)


def ungrounded_signature(
    task_type: TaskType, business_goal: str, capability: CapabilityClassification
) -> str:
    """Ungrounded: task_type | goal keywords | capability."""
    parts = [task_type.value, _goal_keywords(business_goal), capability.value]
    return "|".join(p.replace("|", "/") for p in parts)


def _shape(signature: str) -> tuple[str, ...] | None:
    """(task_type, core_ref names) of a grounded signature; None for ungrounded ones."""
    parts = signature.split("|")
    return (parts[0], parts[3]) if len(parts) == 4 else None


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
        reference: Reference | None,
        *,
        count: int | None = None,
        existing_signatures: Sequence[str] = (),
    ) -> ScenarioGenerationResult:
        """mode = grounded if ``reference`` else ungrounded."""
        cfg = self._settings.generation.scenario
        requested = count if count is not None else cfg.scenario_count
        if requested < 1:
            raise ValueError("scenario count must be >= 1")

        mode = GenerationMode.GROUNDED if reference is not None else GenerationMode.UNGROUNDED
        self._log.info("Generating %d scenario(s), mode %s", requested, mode)
        state = _DedupState(existing_signatures)
        result = ScenarioGenerationResult()
        for round_no in range(1 + cfg.max_topup_rounds):
            needed = requested - len(result.accepted)
            if needed <= 0:
                break
            options = ScenarioPromptOptions(
                count=needed,
                include_unsupported_scenarios=cfg.include_unsupported_scenarios,
                diversity_preference=cfg.diversity_preference,
            )
            system, user = build_scenario_prompt(
                persona, product, reference, options, avoid_signatures=state.signatures()
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
                self._process(raw, persona, product, reference, state, result)
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
        reference: Reference | None,
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

        reasons: list[str] = []
        pack: ReferencePack | None = None
        scenario_id = self._ids.peek(IdKind.SCENARIO)
        if reference is None:
            if draft.reference_pack is not None:
                reasons.append(UNEXPECTED_PACK)
        elif draft.reference_pack is None:
            reasons.append(MISSING_PACK)
        else:
            pack, dropped = build_pack(draft.reference_pack, reference, scenario_id)
            if pack is None:
                reasons.append(NO_VALID_CORE_REF)
                reasons.extend(
                    f"{d.reason} ({d.item.get('name') or d.item.get('rule')})" for d in dropped
                )
        if (
            not cfg.include_unsupported_scenarios
            and draft.capability is CapabilityClassification.UNSUPPORTED
        ):
            reasons.append(FILTERED_BY_CONFIG)
        if reasons:
            result.rejected.append(RejectedScenario(payload=raw, reasons=reasons))
            return

        if pack is not None:
            signature = scenario_signature(
                draft.task_type, draft.business_goal, draft.capability, pack
            )
        else:
            signature = ungrounded_signature(draft.task_type, draft.business_goal, draft.capability)
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
            mode=GenerationMode.GROUNDED if reference is not None else GenerationMode.UNGROUNDED,
            reference_id=reference.reference_id if reference is not None else None,
            reference_pack_id=pack_id_for(scenario_id) if pack is not None else None,
            business_goal=draft.business_goal,
            title=draft.title,
            task_type=draft.task_type,
            capability=draft.capability,
            capability_rationale=draft.capability_rationale,
            success_criteria=draft.success_criteria,
            question_guidance=draft.question_guidance,
            signature=signature,
        )
        assert scenario.id == scenario_id
        state.add(signature, scenario.id)
        result.accepted.append(scenario)
        if pack is not None:
            result.packs.append(pack)


class _DedupState:
    """Signatures seen so far, plus grounded (task, tables, metrics, dimensions) shapes."""

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
