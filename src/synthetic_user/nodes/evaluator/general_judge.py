"""The General Judge: ONE LLM call per evaluable terminal response that has general metrics
(at most TWO when it uses the one-batch evidence lookup, D231).

Scores the run's applicable general metrics (relevance, correctness, the SQL
metrics, tool_correctness and custom metrics routed to ``general``), rates each success
criterion, and - only when the adapter exposes no typed response kind for an answered payload -
returns the response classification within the same call (no extra classifier call).

Validation is structural: exactly the requested metric keys, 0 / 0.5 / 1 scores, every quote
found in the response / artifacts / tool trace, a check-level decision for EVERY requirement the
question's Metric Mapping assigns to each scored metric (no silent omission), every evidence id
and guidance id resolvable (no invented evidence or guidance), a derived check only for a
resolvable rule / guidance / rubric id, a score of 1 without a failed check (a score below 1
without a failed check gets its findings as failed derived checks, by code), a finding for every
score below 1, a factual-verification state where the metric asks
for one, no schema finding that names an SQL output alias, CTE or function (from the parser,
not a regex), and a goal-completion assessment consistent with the response kind (D243).
Invalid output is retried (bounded). After the retries, metrics are validated independently
(D251): when only some metrics' own output is invalid (e.g. Rules Adherence omits a rule
decision), those become ``evaluation_error`` with their own problems and the valid metrics'
scores, checks and findings are kept; a shared problem (goal completion, classification, join
decisions, ...), every metric invalid, or a provider failure makes the whole call an
``evaluation_error`` with one deduplicated reason - never a guessed score or a fabricated check.

Rules (D250-D251): only the criteria and mandatory Reference Spec items are mandatory checks.
The rules of the Reference Spec are AVAILABLE to a rule metric: the judge gives each applicable
rule a check and lists the others in ``not_applicable_rules`` (ids only), so an unrelated rule
needs no check but none is silently dropped.

Bounded lookup (data-catalog runs with SQL, ``evaluation.reference_lookup``): code has already
reconciled the SQL with the catalog and run the obvious exact lookups. The first call may then
return ``evidence_requests`` (ALL needed lookups, at most ``max_requests``) instead of scores;
code runs them through the read-only ``LookupSession`` and makes ONE more call, which must
score (a second request is refused by validation). No loops, no further rounds.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from typing import Any

from synthetic_user.core.exceptions import LLMOutputValidationError, LLMProviderError
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.evaluation import (
    DECLINE_STATUSES,
    GOAL_PROBLEM_SEVERITY,
    GeneralJudgeDraft,
    JoinBasis,
    RequirementSource,
    RequirementStatus,
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
    # LLM attempts per structured call (validation retries included), in call order.
    attempts: list[int] = field(default_factory=list)
    # The evidence ids the final (scoring) call could cite.
    evidence_ids: list[str] = field(default_factory=list)
    # Metrics whose own output stayed invalid after the retries (D251): metric id -> its
    # problems. The other metrics of the reply are kept; ``error`` stays None.
    invalid_metrics: dict[str, list[str]] = field(default_factory=dict)


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
        draft.metrics,
        draft.findings,
        ids,
        quotable,
        guidance_ids=[g for g, ref in data.guidance_refs.items() if ref.user],
    )
    if data.needs_classification and draft.response_classification is None:
        errors.append("response_classification is required (answered | unable_to_answer)")
    for definition in data.definitions:
        judged = draft.metrics.get(definition.metric_id)
        if judged is None:
            continue
        if definition.asks_factual_verification and judged.factual_verification is None:
            errors.append(f"{definition.metric_id}: factual_verification is required")
    errors.extend(check_problems(draft, data))
    errors.extend(_non_base_identifier_findings(draft, data))
    errors.extend(_unaddressed_undocumented_joins(draft, data))
    errors.extend(_goal_problems(draft, data, quotable))
    return errors


def derived_ids(data: GeneralJudgeInput) -> set[str]:
    """What a derived check may name: a guidance id, a metric rubric (METRIC:<id>), a catalog
    rule id the reconciliation or a lookup surfaced, or a Reference Spec requirement."""
    ids = set(data.guidance_refs)
    for item in data.sql:
        if item.reconciliation is not None:
            ids.update(item.reconciliation.rules_not_in_pack)
    for entry in [*data.catalog_lookups, *(data.requested_lookups or ())]:
        ids.update(getattr(rule, "rule_id", "") for rule in getattr(entry.result, "rules", []))
    spec = data.question.context.reference_spec
    if spec is not None:
        ids.update(r.id for r in spec.requirements())
    return {i for i in ids if i}


def check_problems(draft: GeneralJudgeDraft, data: GeneralJudgeInput) -> list[str]:
    """Check-level decisions (reforming, Parts 8 and 10): every requirement the Metric Mapping
    assigns to a judged metric is decided exactly once with the right source, derived checks
    name a resolvable id, evidence and guidance ids resolve, and a score of 1 never hides a failed
    check (a 0 / 0.5 without one is explained by its findings, traceability.py). One check may
    decide several requirements (requirement_ids). Code never sets a score."""
    registry = data.registry()
    derivable = derived_ids(data)
    errors: list[str] = []
    for definition in data.definitions:
        metric_id = definition.metric_id
        judged = draft.metrics.get(metric_id)
        if judged is None:
            continue
        assigned = data.assigned(metric_id)
        criteria = (
            set(data.question.context.metric_mapping[metric_id].criteria)
            if (metric_id in data.question.context.metric_mapping)
            else set()
        )
        available = data.available_rules(metric_id)
        decided: list[str] = []
        for check in judged.checks:
            for rid in (r.strip() for r in check.requirement_ids):
                if check.source is RequirementSource.DERIVED:
                    if rid in available:
                        errors.append(
                            f"{metric_id}: {rid!r} is an available rule of this metric: decide "
                            "it with source reference_spec (or list it in not_applicable_rules)"
                        )
                    elif rid not in derivable:
                        errors.append(
                            f"{metric_id}: derived check {rid!r} names no catalog rule, "
                            "guidance id, Reference Spec id or METRIC:<id> shown above"
                        )
                    continue
                decided.append(rid)
                if rid not in assigned and rid not in available:
                    errors.append(
                        f"{metric_id}: check {rid!r} is not assigned to this metric (assigned: "
                        f"{assigned}); use source derived only for an extra check"
                    )
                elif (rid in criteria) != (check.source is RequirementSource.SUCCESS_CRITERIA):
                    errors.append(f"{metric_id}: check {rid!r} has the wrong source")
        not_applicable = [r.strip() for r in judged.not_applicable_rules]
        mandatory = [r for r in decided if r not in available]
        if missing := [r for r in assigned if r not in mandatory]:
            errors.append(
                f"{metric_id}: checks must decide every assigned requirement; missing {missing} "
                "(requirements with the same decision may share one entry: list their ids in "
                "requirement_ids)"
            )
        if unknown := [r for r in not_applicable if r not in available]:
            errors.append(
                f"{metric_id}: not_applicable_rules {unknown} are not available rules of this "
                f"metric ({available})"
            )
        if undecided := [r for r in available if r not in decided and r not in not_applicable]:
            errors.append(
                f"{metric_id}: decide every available rule: give each applicable rule a check "
                f"and list the others in not_applicable_rules; missing {undecided}"
            )
        both = [r for r in available if r in decided and r in not_applicable]
        if both:
            errors.append(
                f"{metric_id}: rules {both} are both checked and listed in not_applicable_rules"
            )
        listed = [*decided, *not_applicable]
        if repeated := sorted({r for r in listed if listed.count(r) > 1} - set(both)):
            errors.append(f"{metric_id}: requirements decided more than once: {repeated}")
        for check in judged.checks:
            if (
                check.source is RequirementSource.DERIVED
                and check.status is RequirementStatus.FAILED
                and not check.evidence_ids
            ):
                errors.append(
                    f"{metric_id}: the failed derived check {check.requirement_ids} needs the "
                    "evidence_ids that show the defect"
                )
        cited = [*judged.evidence_ids, *(i for c in judged.checks for i in c.evidence_ids)]
        if unknown := registry.unknown(cited):
            errors.append(
                f"{metric_id}: unknown evidence ids {sorted(set(unknown))}; cite only ids shown "
                "above (RESPONSE, ARTIFACT:<n>, SQL:..., RECON:<k>, LOOKUP:<n>, TOOL:<n>, TURN:<n>)"
            )
        guidance = [
            *(g.guidance_id for g in judged.guidance_applied),
            *(i for c in judged.checks for i in c.guidance_ids),
        ]
        if unknown := [g for g in guidance if g not in data.guidance_refs]:
            errors.append(f"{metric_id}: unknown guidance ids {sorted(set(unknown))}")
        failed = any(c.status is RequirementStatus.FAILED for c in judged.checks)
        if judged.score == 1 and failed:
            errors.append(
                f"{metric_id} is scored 1 but a check failed: score it below 1, or correct the "
                "check (a passed check never hides a failed requirement)"
            )
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


def metric_problems(
    draft: GeneralJudgeDraft, data: GeneralJudgeInput, quotable: str
) -> tuple[list[str], dict[str, list[str]]]:
    """The validation problems of a scoring reply split into shared ones (goal completion,
    classification, join decisions, unknown metric keys or finding categories, ...) and the
    ones of each metric, by validating each metric on its own (its score, checks and findings)
    with the SAME checks as ``check_general``. Code only, no LLM call (D251)."""
    ids = {d.metric_id for d in data.definitions}
    shared_draft = draft.model_copy(
        update={
            "metrics": {k: v for k, v in draft.metrics.items() if k not in ids},
            "findings": [f for f in draft.findings if f.category not in ids],
        }
    )
    shared = check_general(shared_draft, replace(data, definitions=[]), quotable)
    by_metric: dict[str, list[str]] = {}
    for definition in data.definitions:
        metric_id = definition.metric_id
        own = draft.model_copy(
            update={
                "metrics": {k: v for k, v in draft.metrics.items() if k == metric_id},
                "findings": [f for f in draft.findings if f.category == metric_id],
            }
        )
        problems = [
            p
            for p in check_general(own, replace(data, definitions=[definition]), quotable)
            if p not in shared
        ]
        if problems:
            by_metric[metric_id] = problems
    covered = {*shared, *(p for ps in by_metric.values() for p in ps)}
    shared.extend(p for p in check_general(draft, data, quotable) if p not in covered)
    return shared, by_metric


def salvage(
    draft: GeneralJudgeDraft, data: GeneralJudgeInput, quotable: str
) -> tuple[GeneralJudgeDraft, dict[str, list[str]]] | None:
    """After the bounded retries: keep the metrics whose own output is valid and drop the
    invalid ones (their scores, checks and findings), returning each invalid metric's
    problems. None when a shared part is invalid or no metric is valid (the whole call is then
    an evaluation error, as before). Never fabricates a check or a score."""
    if draft.evidence_requests:
        return None
    shared, invalid = metric_problems(draft, data, quotable)
    if shared or not invalid or len(invalid) == len(data.definitions):
        return None
    kept = draft.model_copy(
        update={
            "metrics": {k: v for k, v in draft.metrics.items() if k not in invalid},
            "findings": [f for f in draft.findings if f.category not in invalid],
        }
    )
    return kept, invalid


def dedupe_errors(errors: list[str]) -> list[str]:
    """``attempt n: problem`` lines -> each problem once, with the attempts it occurred in."""
    seen: dict[str, list[str]] = {}
    for error in errors:
        head, sep, problem = error.partition(": ")
        if sep and head.startswith("attempt "):
            seen.setdefault(problem, []).append(head.removeprefix("attempt "))
        else:
            seen.setdefault(error, [])
    return [
        f"{problem} (attempt{'s' if len(at) > 1 else ''} {', '.join(at)})" if at else problem
        for problem, at in seen.items()
    ]


class _CountingLLM:
    """Counts the LLM attempts of the current structured call (retries included) for the
    evaluation trace; everything else is passed through unchanged."""

    def __init__(self, llm: LLMClient, attempts: list[int]) -> None:
        self._llm = llm
        self._attempts = attempts

    @property
    def provider_name(self) -> str:
        return self._llm.provider_name

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    async def generate_json(self, system: str, user: str) -> dict[str, Any]:
        self._attempts[-1] += 1
        return await self._llm.generate_json(system, user)

    async def generate_text(self, system: str, user: str) -> str:
        self._attempts[-1] += 1
        return await self._llm.generate_text(system, user)


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
        judgement.attempts.append(0)
        judgement.evidence_ids = data.registry().ids()
        last: list[GeneralJudgeDraft] = []

        def post_validate(draft: GeneralJudgeDraft) -> list[str]:
            last[:] = [draft]
            return check_general(draft, data, quotable)

        try:
            return await generate_structured(
                _CountingLLM(self._llm, judgement.attempts),
                system,
                user,
                GeneralJudgeDraft,
                max_retries=self._max_retries,
                post_validate=post_validate,
                cumulative_feedback=True,
            )
        except LLMOutputValidationError as exc:
            kept = salvage(last[0], data, quotable) if last else None
            if kept is not None:
                draft, invalid = kept
                judgement.invalid_metrics = invalid
                self._log.warning(
                    "%s: general judge output invalid for %s only; other metrics kept",
                    data.question.question_id,
                    ", ".join(invalid),
                )
                return draft
            self._log.warning("%s: general judge output invalid", data.question.question_id)
            judgement.error = (
                f"general judge output invalid after {exc.attempts} attempt(s): "
                + "; ".join(dedupe_errors(list(exc.errors)))
            )
        except LLMProviderError as exc:
            # A transient outage must not discard other completed judgements in the batch.
            self._log.warning(
                "%s: general judge provider call failed: %s", data.question.question_id, exc
            )
            judgement.error = f"general judge provider call failed: {exc}"
        return None
