"""The Retrieval/Grounding Judge: ONE separate LLM call per evaluable response that has
applicable retrieval metrics (context_precision, context_recall, faithfulness,
citation_correctness, relevance labels for MRR, and custom metrics that need retrieved context).

Kept apart from the General Judge so prompts stay small and metrics do not interfere. MRR is
never scored by the LLM: the judge labels every chunk relevant / not relevant and code computes
the reciprocal rank (1 / rank of the first relevant chunk, 0 when none is relevant) from the
exposed ranks. Validation is structural: metric keys, quotes found in the answer or a chunk,
known chunk ids, each label / checklist item / citation exactly once, checklist-score
consistency, and a finding for every score below 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from synthetic_user.core.exceptions import LLMOutputValidationError, LLMProviderError
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.evaluation import CoverageStatus, RetrievalJudgeDraft
from synthetic_user.models.execution import RetrievedContext
from synthetic_user.nodes.evaluator.judge_common import (
    check_scores_and_findings,
    guidance_items,
    normalise,
    response_text,
)
from synthetic_user.nodes.evaluator.retrieval_prompt import (
    REQUIRED_INFORMATION_FIELD,
    RetrievalJudgeInput,
    build_retrieval_prompt,
    citation_label,
)

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]


@dataclass
class RetrievalJudgement:
    draft: RetrievalJudgeDraft | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReciprocalRank:
    value: float
    first_relevant_rank: int | None
    relevant_chunk_ids: list[str]


def reciprocal_rank(context: RetrievedContext, labels: dict[str, bool]) -> ReciprocalRank | None:
    """1 / rank of the first relevant chunk (0 when none is relevant). None when the chunks
    carry no explicit rank or a chunk has no label: arbitrary order is never a rank."""
    if not context.rank_exposed or any(c.chunk_id not in labels for c in context.chunks):
        return None
    ranked = sorted(context.chunks, key=lambda c: c.rank or 0)
    relevant = [c for c in ranked if labels[c.chunk_id]]
    if not relevant:
        return ReciprocalRank(0.0, None, [])
    first = relevant[0].rank
    assert first is not None
    return ReciprocalRank(round(1 / first, 6), first, [c.chunk_id for c in relevant])


def check_retrieval(draft: RetrievalJudgeDraft, data: RetrievalJudgeInput) -> list[str]:
    ids = [d.metric_id for d in data.definitions]
    chunk_ids = {c.chunk_id for c in data.context.chunks}
    quotable = "\n".join([response_text(data.response), *(c.text for c in data.context.chunks)])
    errors = check_scores_and_findings(
        draft.metrics,
        draft.findings,
        ids,
        quotable,
        guidance_ids=[i.guidance_id for i in guidance_items(data.guidance)]
        if data.guidance is not None
        else [],
    )
    tool_refs = {f"tool-{c.order}" for c in data.tool_trace or ()}
    for name, judged in draft.metrics.items():
        unknown = sorted(set(judged.evidence_ids) - chunk_ids - tool_refs)
        if unknown:
            errors.append(f"{name}: evidence_ids {unknown} are not retrieved chunks / tool calls")
    if data.needs_labels:
        labelled = [label.chunk_id for label in draft.chunk_relevance]
        if sorted(labelled) != sorted(chunk_ids):
            errors.append(
                "chunk_relevance must label every retrieved chunk exactly once: "
                f"expected {sorted(chunk_ids)}, got {labelled}"
            )
    elif draft.chunk_relevance and any(
        label.chunk_id not in chunk_ids for label in draft.chunk_relevance
    ):
        errors.append("chunk_relevance names unknown chunk ids")
    checklist = data.checklist()
    if checklist:
        wanted = {normalise(item): item for item in checklist}
        got = [normalise(item.item) for item in draft.required_information]
        if sorted(got) != sorted(wanted):
            errors.append(
                "required_information must list every checklist item exactly once: "
                f"expected {checklist}, got {[i.item for i in draft.required_information]}"
            )
        for item in draft.required_information:
            if unknown := sorted(set(item.chunk_ids) - chunk_ids):
                errors.append(f"required_information {item.item!r}: unknown chunk ids {unknown}")
        errors.extend(_checklist_consistency(draft, data))
    if data.checks_citations:
        wanted_citations = [
            citation_label(i, c.citation_id) for i, c in enumerate(data.context.citations, 1)
        ]
        got_citations = [check.citation for check in draft.citation_checks]
        if sorted(got_citations) != sorted(wanted_citations):
            errors.append(
                "citation_checks must cover every listed citation exactly once: expected "
                f"{wanted_citations}, got {got_citations}"
            )
        for check in draft.citation_checks:
            if check.chunk_id is not None and check.chunk_id not in chunk_ids:
                errors.append(f"citation {check.citation}: unknown chunk id {check.chunk_id!r}")
    return errors


def _checklist_consistency(draft: RetrievalJudgeDraft, data: RetrievalJudgeInput) -> list[str]:
    statuses = [item.status for item in draft.required_information]
    if not statuses:
        return []
    all_covered = all(s is CoverageStatus.COVERED for s in statuses)
    none_covered = all(s is CoverageStatus.NOT_COVERED for s in statuses)
    errors: list[str] = []
    for definition in data.definitions:
        value = data.generated(definition.metric_id)
        if not value or REQUIRED_INFORMATION_FIELD not in value:
            continue
        judged = draft.metrics.get(definition.metric_id)
        if judged is None or judged.score is None:
            continue
        if judged.score == 1 and not all_covered:
            errors.append(f"{definition.metric_id}: score 1 needs every checklist item covered")
        if judged.score == 0 and not none_covered:
            errors.append(f"{definition.metric_id}: score 0 means no checklist item is covered")
        if judged.score == 0.5 and (all_covered or none_covered):
            errors.append(f"{definition.metric_id}: score 0.5 needs partial checklist coverage")
    return errors


class RetrievalJudge:
    def __init__(self, llm: LLMClient, logger: Logger, *, max_retries: int) -> None:
        self._llm = llm
        self._log = logger
        self._max_retries = max_retries

    async def judge(self, data: RetrievalJudgeInput) -> RetrievalJudgement:
        system, user = build_retrieval_prompt(data)
        judgement = RetrievalJudgement()
        try:
            judgement.draft = await generate_structured(
                self._llm,
                system,
                user,
                RetrievalJudgeDraft,
                max_retries=self._max_retries,
                post_validate=lambda draft: check_retrieval(draft, data),
                cumulative_feedback=True,
            )
        except LLMOutputValidationError as exc:
            self._log.warning("%s: retrieval judge output invalid", data.question.question_id)
            judgement.error = (
                f"retrieval judge output invalid after {exc.attempts} attempt(s): "
                + "; ".join(exc.errors)
            )
        except LLMProviderError as exc:
            self._log.warning(
                "%s: retrieval judge provider call failed: %s", data.question.question_id, exc
            )
            judgement.error = f"retrieval judge provider call failed: {exc}"
        return judgement
