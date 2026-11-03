"""Node 2: generate scenarios in ONE LLM call for the whole requested batch.

Grounded scenarios carry a validated Reference Pack (cheat sheet); ungrounded ones have neither
a reference nor a pack. No top-up rounds and no separate validation call: each returned item is
validated individually (schema, persona task, pack integrity, config filter) and rejected with
its reasons, never fixed. Duplicates are only EXACT structural duplicates (identical signature:
task type | capability | cheat-sheet items | hash of the normalized business goal); there is no
keyword or similarity-based duplicate detection. The run's selected metrics reach the prompt as
coverage guidance only; the generation mode never depends on them.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from typing import Any

from pydantic import Field, ValidationError

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import format_validation_error
from synthetic_user.core.identifiers import IdKind, SequentialIdGenerator
from synthetic_user.core.trace import trace_subject
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.base import StrictModel
from synthetic_user.models.metrics import MetricDefinition
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
from synthetic_user.nodes.scenario.prompt import ScenarioPromptOptions, build_scenario_prompt
from synthetic_user.references.pack import NO_VALID_ITEM, build_pack

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

FILTERED_BY_CONFIG = "filtered_by_config"
SURPLUS = "surplus: more scenarios returned than requested"
MISSING_PACK = "mode grounded requires a reference_pack"
UNEXPECTED_PACK = "mode ungrounded must not have a reference_pack"
UNKNOWN_TYPICAL_TASK = "typical_task is not one of the persona's typical_tasks"


class _RawScenarioBatch(StrictModel):
    """Parsed per call; each scenario is validated individually so one bad item is
    rejected on its own instead of failing the whole batch."""

    scenarios: list[dict[str, Any]] = Field(default_factory=list)


def _goal_hash(business_goal: str) -> str:
    """Hash of the whitespace/case-normalized goal: exact identity, no keyword heuristics."""
    normalized = " ".join(business_goal.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def scenario_signature(
    task_type: TaskType,
    business_goal: str,
    capability: CapabilityClassification,
    pack: ReferencePack | None = None,
) -> str:
    """Structural id: task_type | capability | cheat-sheet item names | goal hash."""
    items = ",".join(sorted({n.strip().casefold() for n in pack.item_names()})) if pack else ""
    parts = [task_type.value, capability.value, items, _goal_hash(business_goal)]
    return "|".join(p.replace("|", "/") for p in parts)


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
        metrics: Sequence[MetricDefinition] = (),
    ) -> ScenarioGenerationResult:
        """mode = grounded if ``reference`` else ungrounded. Exactly one LLM call (+ bounded
        validation retries of that call)."""
        cfg = self._settings.generation.scenario
        requested = count if count is not None else cfg.scenario_count
        if requested < 1:
            raise ValueError("scenario count must be >= 1")

        mode = GenerationMode.GROUNDED if reference is not None else GenerationMode.UNGROUNDED
        self._log.info("Generating %d scenario(s) in one call, mode %s", requested, mode)
        seen: dict[str, str] = dict.fromkeys(existing_signatures, "existing scenario")
        result = ScenarioGenerationResult()
        options = ScenarioPromptOptions(
            count=requested,
            include_unsupported_scenarios=cfg.include_unsupported_scenarios,
            diversity_preference=cfg.diversity_preference,
            max_request_steps=self._settings.generation.question.max_deliverables,
        )
        system, user = build_scenario_prompt(
            persona,
            product,
            reference,
            options,
            avoid_signatures=list(existing_signatures),
            metrics=metrics,
        )
        with trace_subject("scenario batch"):
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
            self._process(raw, persona, product, reference, seen, result)
        self._log.info(
            "Scenarios: %d returned; accepted=%d rejected=%d duplicates=%d",
            len(batch.scenarios),
            len(result.accepted),
            len(result.rejected),
            len(result.duplicates),
        )
        if len(result.accepted) < requested:
            self._log.warning(
                "Only %d of %d requested scenarios accepted (partial result, not padded)",
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
        seen: dict[str, str],
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
        typical_task = persona_task(persona, draft.typical_task)
        if draft.typical_task is not None and typical_task is None:
            reasons.append(f"{UNKNOWN_TYPICAL_TASK}: {draft.typical_task!r}")
        if reference is None:
            if draft.reference_pack is not None:
                reasons.append(UNEXPECTED_PACK)
        elif draft.reference_pack is None:
            reasons.append(MISSING_PACK)
        else:
            pack, dropped = build_pack(draft.reference_pack, reference, scenario_id)
            if pack is None:
                reasons.append(NO_VALID_ITEM)
                reasons.extend(d.reason for d in dropped)
        if (
            not cfg.include_unsupported_scenarios
            and draft.capability is CapabilityClassification.UNSUPPORTED
        ):
            reasons.append(FILTERED_BY_CONFIG)
        if reasons:
            result.rejected.append(RejectedScenario(payload=raw, reasons=reasons))
            return

        signature = scenario_signature(draft.task_type, draft.business_goal, draft.capability, pack)
        duplicate_of = seen.get(signature)
        if duplicate_of is not None:
            result.duplicates.append(
                RejectedScenario(
                    payload=raw, reasons=[f"exact duplicate of {duplicate_of}: {signature}"]
                )
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
            typical_task=typical_task,
            task_type=draft.task_type,
            capability=draft.capability,
            capability_rationale=draft.capability_rationale,
            success_criteria=draft.success_criteria,
            question_intent=draft.question_intent,
            signature=signature,
        )
        assert scenario.id == scenario_id
        seen[signature] = scenario.id
        result.accepted.append(scenario)
        if pack is not None:
            result.packs.append(pack)


def persona_task(persona: Persona, text: str | None) -> str | None:
    """The persona's typical task ``text`` names (case / spacing / end punctuation ignored),
    returned in the persona's own wording; None if ``text`` is None or names none of them."""
    if text is None:
        return None
    wanted = _task_key(text)
    return next((t for t in persona.typical_tasks if _task_key(t) == wanted), None)


def _task_key(text: str) -> str:
    return " ".join(text.casefold().split()).rstrip(".")
