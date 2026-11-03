"""The General Judge: ONE LLM call per evaluable terminal response that has general metrics
(at most TWO when it uses the one-batch evidence lookup, D231).

Scores the run's applicable general metrics (relevance, correctness_completeness, the SQL
metrics, tool_correctness and custom metrics routed to ``general``), rates each success
criterion, and - only when the adapter exposes no typed response kind for an answered payload -
returns the response classification within the same call (no extra classifier call).

Validation is structural: exactly the requested metric keys, 0 / 0.5 / 1 scores, every quote
found in the response / artifacts / tool trace, every success criterion listed once, a finding
for every score below 1, a factual-verification state where the metric asks for one, and no
schema finding that names an SQL output alias, CTE or function (from the parser, not a regex),
and a goal-completion assessment consistent with the response kind (D243).
Invalid output is retried (bounded); exhausted retries or a provider failure produce
``evaluation_error`` results with the reason, never a guessed score.

Bounded lookup (data-catalog runs with SQL, ``evaluation.reference_lookup``): code has already
reconciled the SQL with the catalog and run the obvious exact lookups. The first call may then
return ``evidence_requests`` (ALL needed lookups, at most ``max_requests``) instead of scores;
code runs them through the read-only ``LookupSession`` and makes ONE more call, which must
score (a second request is refused by validation). No loops, no further rounds.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, replace

from synthetic_user.core.exceptions import LLMOutputValidationError, LLMProviderError
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.evaluation import (
    DECLINE_STATUSES,
    GOAL_PROBLEM_SEVERITY,
    GeneralJudgeDraft,
    JoinBasis,
)
from synthetic_user.models.execution import DeniedResponse, ResponseClassification
from synthetic_user.models.lookup import LookupOrigin
from synthetic_user.models.metrics import EvidenceKind, ReferenceAsset
from synthetic_user.nodes.evaluator.general_prompt import (
    PROMPTS,
    GeneralJudgeInput,
    build_general_prompt,
)
from synthetic_user.nodes.evaluator.judge_common import (
    COPY_HINT,
    EvaluableResponse,
    check_scores_and_findings,
    normalise,
    quoted,
    response_text,
)
from synthetic_user.references.catalog_index import LookupSession

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]


@dataclass
class GeneralJudgement:
    draft: GeneralJudgeDraft | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    calls: int = 0  # structured calls made (1, or 2 after a lookup request)
    lookup_requests: int = 0  # lookups the judge requested (0 when it scored at once)


def quotable_text(data: GeneralJudgeInput) -> str:
    parts = [response_text(data.response)]
    if data.tool_trace is not None:
        parts.append(json.dumps([c.model_dump(mode="json") for c in data.tool_trace]))
    return "\n".join(parts)


def check_general(draft: GeneralJudgeDraft, data: GeneralJudgeInput, quotable: str) -> list[str]:
    if draft.evidence_requests:
        if data.max_lookup_requests <= 0:
            return [
                "evidence_requests are not available in this reply: score every listed metric "
                "now and return evidence_requests []"
            ]
        if len(draft.evidence_requests) > data.max_lookup_requests:
            return [
                f"at most {data.max_lookup_requests} evidence_requests (got "
                f"{len(draft.evidence_requests)}): keep only the lookups needed to score"
            ]
        return []  # a valid lookup request; the scoring reply is validated afterwards
    if draft.overall_verdict is None:
        return ["overall_verdict is required"]
    ids = [d.metric_id for d in data.definitions]
    errors = check_scores_and_findings(
        draft.metrics, draft.findings, ids, quotable, guidance=data.guidance
    )
    if data.needs_classification and draft.response_classification is None:
        errors.append("response_classification is required (answered | unable_to_answer)")
    for definition in data.definitions:
        judged = draft.metrics.get(definition.metric_id)
        if judged is None:
            continue
        if definition.asks_factual_verification and judged.factual_verification is None:
            errors.append(f"{definition.metric_id}: factual_verification is required")
    wanted = {_criterion_key(c): c for c in data.question.expectations.success_criteria.general}
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
    errors.extend(_unaddressed_undocumented_joins(draft, data))
    errors.extend(_goal_problems(draft, data, quotable))
    return errors


def _goal_problems(draft: GeneralJudgeDraft, data: GeneralJudgeInput, quotable: str) -> list[str]:
    """The goal-completion assessment (D243): present, a decline status exactly when the
    response is a decline (typed, or as the judge classified it), and a quoted fragment for
    every problem status. Whether a decline is justified is the judge's reading of the
    contract; code never decides it."""
    goal = draft.goal_completion
    if goal is None:
        return [PROMPTS.text("goal_required")]
    kind = _response_kind(data.response, draft)
    errors: list[str] = []
    if kind is ResponseClassification.UNABLE_TO_ANSWER and goal.status not in DECLINE_STATUSES:
        errors.append(PROMPTS.text("goal_decline_status", status=goal.status.value))
    if kind is ResponseClassification.ANSWERED and goal.status in DECLINE_STATUSES:
        errors.append(PROMPTS.text("goal_answer_status", status=goal.status.value))
    if goal.status in GOAL_PROBLEM_SEVERITY:
        if not goal.evidence:
            errors.append(PROMPTS.text("goal_evidence_required", status=goal.status.value))
        elif not quoted(goal.evidence, quotable):
            errors.append(
                PROMPTS.text(
                    "goal_evidence_not_quoted", evidence=repr(goal.evidence), hint=COPY_HINT
                )
            )
    return errors


def _response_kind(
    response: EvaluableResponse, draft: GeneralJudgeDraft
) -> ResponseClassification | None:
    """Answered / unable_to_answer from the typed outcome, the adapter or the judge's own
    classification; None when nothing says."""
    if isinstance(response, DeniedResponse):
        return ResponseClassification.UNABLE_TO_ANSWER
    if response.response_classification is not None:
        return response.response_classification
    if draft.response_classification is not None:
        return ResponseClassification(draft.response_classification)
    return None


def _unaddressed_undocumented_joins(draft: GeneralJudgeDraft, data: GeneralJudgeInput) -> list[str]:
    """Every undocumented join needs the judge's decision, and the schema score must agree
    with it (D238). Code knows the join is not listed; only the judge can read whether the
    reference, its usage instructions or the guidance allow it. Code never sets the score."""
    wanted = data.undocumented_joins()
    if not wanted:
        return []

    def key(text: str) -> frozenset[str]:
        return frozenset(normalise(part) for part in text.split("="))

    decisions = {key(d.join): d for d in draft.join_decisions}
    errors: list[str] = []
    missing = [j for j in wanted if key(j) not in decisions]
    if missing:
        errors.append(
            f"join_decisions must decide every undocumented join; missing: {missing} (copy "
            "each as 'left = right', with allowed and its basis)"
        )
    schema_metrics = [
        d.metric_id
        for d in data.definitions
        if ReferenceAsset.CATALOG_SCHEMA in d.required_reference_assets
        and EvidenceKind.SQL in d.required_execution_evidence
    ]
    for join in wanted:
        decision = decisions.get(key(join))
        if decision is None:
            continue
        for metric_id in schema_metrics:
            judged = draft.metrics.get(metric_id)
            if judged is None or judged.score is None:
                continue
            categories = {f.category: f for f in draft.findings if f.category == metric_id}
            if not decision.allowed and decision.basis is not JoinBasis.NOT_ESTABLISHED:
                if judged.score == 1:
                    errors.append(
                        f"join_decisions says {join} is not allowed ({decision.basis.value}) "
                        f"but {metric_id} is scored 1: score it below 1 with a finding, or "
                        "correct the decision"
                    )
            elif decision.basis is JoinBasis.NOT_ESTABLISHED and not any(
                f.unresolved for f in categories.values()
            ):
                errors.append(
                    f"join_decisions says the validity of {join} is not established: add an "
                    f"unresolved finding with category {metric_id!r}"
                )
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

    async def judge(
        self, data: GeneralJudgeInput, lookups: LookupSession | None = None
    ) -> GeneralJudgement:
        judgement = GeneralJudgement()
        if lookups is None and data.max_lookup_requests:
            data = replace(data, max_lookup_requests=0)
        draft = await self._call(data, judgement)
        if draft is not None and draft.evidence_requests and lookups is not None:
            judgement.lookup_requests = len(draft.evidence_requests)
            results = lookups.run_all(draft.evidence_requests, LookupOrigin.JUDGE_REQUEST)
            self._log.info(
                "%s: general judge requested %d lookup(s): %s",
                data.question.question_id,
                len(draft.evidence_requests),
                ", ".join(r.tool.value for r in draft.evidence_requests),
            )
            draft = await self._call(
                replace(data, max_lookup_requests=0, requested_lookups=results), judgement
            )
        judgement.draft = draft
        return judgement

    async def _call(
        self, data: GeneralJudgeInput, judgement: GeneralJudgement
    ) -> GeneralJudgeDraft | None:
        system, user = build_general_prompt(data)
        quotable = quotable_text(data)
        judgement.calls += 1
        try:
            return await generate_structured(
                self._llm,
                system,
                user,
                GeneralJudgeDraft,
                max_retries=self._max_retries,
                post_validate=lambda draft: check_general(draft, data, quotable),
                cumulative_feedback=True,
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
        return None
