"""The General Judge: ONE LLM call per evaluable terminal response that has general metrics.

Scores the run's applicable general metrics (relevance, correctness_completeness, the SQL
metrics, tool_correctness and custom metrics routed to ``general``), rates each success
criterion, and - only when the adapter exposes no typed response kind for an answered payload -
returns the response classification within the same call (no extra classifier call).

Validation is structural: exactly the requested metric keys, 0 / 0.5 / 1 scores, every quote
found in the response / artifacts / tool trace, every success criterion listed once, a finding
for every score below 1, a factual-verification state where the metric asks for one, and no
schema finding that names an SQL output alias, CTE or function (from the parser, not a regex).
Invalid output is retried (bounded); exhausted retries or a provider failure produce
``evaluation_error`` results with the reason, never a guessed score.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from synthetic_user.core.exceptions import LLMOutputValidationError, LLMProviderError
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.evaluation import GeneralJudgeDraft
from synthetic_user.models.metrics import EvidenceKind, ReferenceAsset
from synthetic_user.nodes.evaluator.general_prompt import GeneralJudgeInput, build_general_prompt
from synthetic_user.nodes.evaluator.judge_common import (
    COPY_HINT,
    check_scores_and_findings,
    normalise,
    quoted,
    response_text,
)

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]


@dataclass
class GeneralJudgement:
    draft: GeneralJudgeDraft | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def quotable_text(data: GeneralJudgeInput) -> str:
    parts = [response_text(data.response)]
    if data.tool_trace is not None:
        parts.append(json.dumps([c.model_dump(mode="json") for c in data.tool_trace]))
    return "\n".join(parts)


def check_general(draft: GeneralJudgeDraft, data: GeneralJudgeInput, quotable: str) -> list[str]:
    ids = [d.metric_id for d in data.definitions]
    errors = check_scores_and_findings(draft.metrics, draft.findings, ids, quotable)
    if data.needs_classification and draft.response_classification is None:
        errors.append("response_classification is required (answered | unable_to_answer)")
    for definition in data.definitions:
        judged = draft.metrics.get(definition.metric_id)
        if judged is None:
            continue
        if definition.asks_factual_verification and judged.factual_verification is None:
            errors.append(f"{definition.metric_id}: factual_verification is required")
    wanted = {_criterion_key(c): c for c in data.question.expectations.success_criteria}
    got = [_criterion_key(r.criterion) for r in draft.criteria]
    missing = [c for key, c in wanted.items() if key not in got]
    extra = [r.criterion for r in draft.criteria if _criterion_key(r.criterion) not in wanted]
    repeated = sorted({g for g in got if got.count(g) > 1})
    if missing or extra or repeated:
        errors.append(
            f"criteria must list each success criterion once: missing {missing}, "
            f"unknown {extra}, repeated {repeated}"
        )
    for result in draft.criteria:
        if result.metric_id is not None and result.metric_id not in ids:
            errors.append(f"criterion {result.criterion!r}: metric_id must be one of {ids} or null")
        if result.evidence and not quoted(result.evidence, quotable):
            errors.append(
                f"criterion {result.criterion!r}: evidence {result.evidence!r} is not in the "
                f"response ({COPY_HINT})"
            )
    errors.extend(_non_base_identifier_findings(draft, data))
    return errors


def _non_base_identifier_findings(draft: GeneralJudgeDraft, data: GeneralJudgeInput) -> list[str]:
    """A schema finding whose evidence is only an output alias / CTE / function name."""
    schema_metrics = {
        d.metric_id
        for d in data.definitions
        if ReferenceAsset.CATALOG_SCHEMA in d.required_reference_assets
        and EvidenceKind.SQL in d.required_execution_evidence
    }
    names = {
        normalise(name)
        for item in data.sql
        for name in (
            *item.structure.output_aliases,
            *item.structure.ctes,
            *item.structure.functions,
        )
    }
    errors: list[str] = []
    for index, finding in enumerate(draft.findings, start=1):
        if (
            finding.category in schema_metrics
            and finding.evidence
            and normalise(finding.evidence.strip(' `"[]')) in names
        ):
            errors.append(
                f"finding {index}: {finding.evidence!r} is an SQL output alias, CTE name or "
                "function, not a base table/column identifier; it cannot lower "
                f"{finding.category}"
            )
    return errors


def _criterion_key(text: str) -> str:
    """Tolerate an LLM adding an ellipsis inside otherwise verbatim notation."""
    return normalise(re.sub(r"\s*\.\.\.\s*", "", text))


class GeneralJudge:
    def __init__(self, llm: LLMClient, logger: Logger, *, max_retries: int) -> None:
        self._llm = llm
        self._log = logger
        self._max_retries = max_retries

    async def judge(self, data: GeneralJudgeInput) -> GeneralJudgement:
        system, user = build_general_prompt(data)
        quotable = quotable_text(data)
        judgement = GeneralJudgement()
        try:
            judgement.draft = await generate_structured(
                self._llm,
                system,
                user,
                GeneralJudgeDraft,
                max_retries=self._max_retries,
                post_validate=lambda draft: check_general(draft, data, quotable),
            )
        except LLMOutputValidationError as exc:
            self._log.warning("%s: general judge output invalid", data.question.question_id)
            judgement.error = (
                f"general judge output invalid after {exc.attempts} attempt(s): "
                + "; ".join(exc.errors)
            )
        except LLMProviderError as exc:
            # A transient outage must not discard other completed judgements in the batch.
            self._log.warning(
                "%s: general judge provider call failed: %s", data.question.question_id, exc
            )
            judgement.error = f"general judge provider call failed: {exc}"
        return judgement
