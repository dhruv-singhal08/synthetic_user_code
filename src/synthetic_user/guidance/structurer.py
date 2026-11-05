"""Evaluation guidance -> verbatim, typed items with stable ids (G1, G2, ...).

ONE structured LLM call per guidance version, made when the user saves guidance (UI) or, for a
CLI run, the first time the evaluate step sees that version; the result is cached by the text's
sha256, so an unchanged guidance never costs another call. No information may be lost and none
added, checked by code (retry feedback, bounded):

- every item's text is an exact excerpt of the guidance (case, whitespace and line breaks
  aside: the same check the judges' quotes use);
- every non-empty line of the guidance is inside some item (a list marker or heading marker
  at the start of a line may be left out);
- metric ids are among the run's metric ids.

Code assigns the ids in order and computes each item's line range. When the call still fails
the save is refused with the reason (nothing is stored half-structured).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from pydantic import Field

from synthetic_user.core.trace import trace_subject
from synthetic_user.guidance.structurer_prompt import PROMPTS, build_structurer_prompt
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.run import EvaluationGuidance, GuidanceItem, GuidanceType

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

# A list / heading marker at the start of a line (may be left out of an item's text).
_MARKER = re.compile(r"^\s*(?:[-*+•]|\d+[.)]|#+|>)\s*")
_JSON_ESCAPES = (("\\n", " "), ("\\r", " "), ("\\t", " "), ('\\"', '"'))


class GuidanceItemDraft(StrictModel):
    type: GuidanceType
    metrics: list[NonEmptyStr] = Field(default_factory=list)
    text: NonEmptyStr


class GuidanceStructureDraft(StrictModel):
    items: list[GuidanceItemDraft] = Field(min_length=1)


def compact(text: str) -> str:
    for escape, char in _JSON_ESCAPES:
        text = text.replace(escape, char)
    return re.sub(r"\s+", "", text).casefold()


def _line_key(line: str) -> str:
    return compact(_MARKER.sub("", line, count=1))


def structure_problems(
    draft: GuidanceStructureDraft, guidance: EvaluationGuidance, metric_ids: Sequence[str]
) -> list[str]:
    source = compact(guidance.text)
    known = set(metric_ids)
    problems: list[str] = []
    for index, item in enumerate(draft.items, start=1):
        if compact(item.text) not in source:
            problems.append(PROMPTS.text("not_verbatim", index=index))
        if unknown := [m for m in item.metrics if m not in known]:
            problems.append(PROMPTS.text("unknown_metrics", index=index, metrics=unknown))
    covered = [compact(item.text) for item in draft.items]
    missing = [
        number
        for number, line in enumerate(guidance.text.splitlines(), start=1)
        if (key := _line_key(line)) and re.search(r"\w", key) and not any(key in c for c in covered)
    ]
    if missing:
        problems.append(PROMPTS.text("lines_not_covered", lines=missing))
    return problems


def _location(text: str, guidance: EvaluationGuidance) -> str:
    item = compact(text)
    lines = [
        number
        for number, line in enumerate(guidance.text.splitlines(), start=1)
        if (key := _line_key(line)) and key in item
    ]
    if not lines:
        return guidance.source
    first, last = min(lines), max(lines)
    span = f"line {first}" if first == last else f"lines {first}-{last}"
    return f"{guidance.source} {span}"


class GuidanceStructurer:
    def __init__(self, llm: LLMClient, logger: Logger, *, max_retries: int) -> None:
        self._llm = llm
        self._log = logger
        self._max_retries = max_retries

    async def structure(
        self, guidance: EvaluationGuidance, metric_ids: Sequence[str]
    ) -> EvaluationGuidance:
        """The guidance with its verbatim items (LLMOutputValidationError / LLMProviderError
        when it cannot be structured; the caller refuses the save)."""
        system, user = build_structurer_prompt(guidance, metric_ids)
        with trace_subject("evaluation guidance"):
            draft = await generate_structured(
                self._llm,
                system,
                user,
                GuidanceStructureDraft,
                max_retries=self._max_retries,
                post_validate=lambda d: structure_problems(d, guidance, metric_ids),
            )
        items = [
            GuidanceItem(
                guidance_id=f"G{index}",
                type=item.type,
                metrics=list(dict.fromkeys(item.metrics)),
                text=item.text.strip(),
                location=_location(item.text, guidance),
            )
            for index, item in enumerate(draft.items, start=1)
        ]
        self._log.info(
            "Evaluation guidance structured into %d item(s): %s",
            len(items),
            ", ".join(f"{i.guidance_id}={i.type.value}" for i in items),
        )
        return guidance.model_copy(update={"items": items})
