"""Generic evaluation of one ANSWERED response (all products, both modes): one LLM judgement,
then metric scores and the verdict computed by code.

Draft checks before acceptance (bounded retry via llm/structured.py):
- correctness given exactly for grounded questions;
- one criterion result per success criterion;
- every quote (scores, criteria, claims, findings) occurs in the response; a claim's
  reference_quote occurs in the scenario's cheat sheet or the usage instructions; claim
  statuses fit the mode;
- finding categories are metric names; required findings: relevancy 0, correctness 0, a
  not_met criterion, every hallucinated claim.
After the last retry the evaluation is inconclusive with the errors as notes (never guessed).

Scores (MetricName): relevancy / correctness = the judge's 0 / 0.5 / 1; completeness = mean
of criteria (met 1, partly 0.5, not_met 0; unverifiable left out); groundedness = supported /
all claims; hallucination_rate = (contradicted + fabricated) / all claims. No claims ->
groundedness and hallucination_rate not_applicable.

Verdict: a metric FAILS when relevancy, correctness or completeness is 0, or
hallucination_rate > 0 (groundedness alone never fails: statements the reference does not
cover are not errors). A failing metric backed by a supported finding of that category ->
fail; otherwise any failing or inconclusive metric -> inconclusive; else pass.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import LLMOutputValidationError
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.evaluation import (
    GROUNDED_CLAIM_STATUSES,
    HALLUCINATED,
    UNGROUNDED_CLAIM_STATUSES,
    ClaimStatus,
    CriterionStatus,
    EvaluationDraft,
    EvidenceStatus,
    Finding,
    JudgedScore,
    MetricName,
    MetricResult,
    Metrics,
    MetricStatus,
    Verdict,
)
from synthetic_user.models.execution import AnsweredResponse
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import Reference, ReferencePack, quote_in
from synthetic_user.models.scenario import Scenario
from synthetic_user.prompts.response_evaluation import (
    build_response_evaluation_prompt,
    response_text,
)

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

METRIC_NAMES = tuple(m.value for m in MetricName)
# Added to quote errors so the retry knows how to fix them.
COPY_HINT = (
    'copy ONE shorter contiguous fragment character for character from the answer, without "..."'
)
_CRITERION_SCORE = {
    CriterionStatus.MET: 1.0,
    CriterionStatus.PARTLY: 0.5,
    CriterionStatus.NOT_MET: 0.0,
}


@dataclass
class GenericJudgement:
    draft: EvaluationDraft | None = None
    notes: list[str] = field(default_factory=list)


# --- draft checks -----------------------------------------------------------------------------


def check_evaluation(
    draft: EvaluationDraft,
    criteria: Sequence[str],
    response: str,
    pack: ReferencePack | None = None,
    instructions: str = "",
) -> list[str]:
    """Problems with an evaluation draft (empty list = acceptable). ``pack`` (the scenario's
    cheat sheet) is given exactly for grounded questions, with the usage ``instructions``
    the judge also saw (a reference_quote may come from either)."""
    grounded = pack is not None
    quotable_reference = f"{pack.render()}\n{instructions}" if pack is not None else ""
    errors: list[str] = []
    if grounded and draft.correctness is None:
        errors.append("correctness is required for a grounded question")
    if not grounded and draft.correctness is not None:
        errors.append("correctness must be null: the question has no reference")

    scores = {"relevancy": draft.relevancy, "correctness": draft.correctness}
    for name, judged in scores.items():
        if judged and judged.evidence and not _quoted(judged.evidence, response):
            errors.append(
                f"{name}: evidence {judged.evidence!r} is not in the response ({COPY_HINT})"
            )

    wanted = {_normalise(c): c for c in criteria}
    got = [_normalise(r.criterion) for r in draft.criteria]
    missing = [c for key, c in wanted.items() if key not in got]
    extra = [r.criterion for r in draft.criteria if _normalise(r.criterion) not in wanted]
    repeated = sorted({g for g in got if got.count(g) > 1})
    if missing or extra or repeated:
        errors.append(
            f"criteria must list each success criterion once: missing {missing}, "
            f"unknown {extra}, repeated {repeated}"
        )
    for result in draft.criteria:
        if result.evidence and not _quoted(result.evidence, response):
            errors.append(
                f"criterion {result.criterion!r}: evidence {result.evidence!r} is not in the "
                f"response ({COPY_HINT})"
            )

    allowed = GROUNDED_CLAIM_STATUSES if grounded else UNGROUNDED_CLAIM_STATUSES
    for i, claim in enumerate(draft.claims, start=1):
        if not _quoted(claim.text, response):
            errors.append(
                f"claim {i}: text {claim.text!r} is not copied from the response ({COPY_HINT})"
            )
        if claim.status not in allowed:
            errors.append(
                f"claim {i}: status {claim.status.value!r} must be one of "
                f"{sorted(s.value for s in allowed)}"
            )
        if claim.reference_quote:
            if pack is None:
                errors.append(f"claim {i}: reference_quote must be null (no reference)")
            elif not quote_in(claim.reference_quote, quotable_reference):
                errors.append(
                    f"claim {i}: reference_quote is not in the cheat sheet or the usage "
                    "instructions (copy one short passage character for character, or null)"
                )

    for i, finding in enumerate(draft.findings, start=1):
        if finding.category not in METRIC_NAMES:
            errors.append(
                f"finding {i}: category {finding.category!r} is not one of {METRIC_NAMES}"
            )
        if finding.evidence and not _quoted(finding.evidence, response):
            errors.append(
                f"finding {i}: evidence {finding.evidence!r} is not in the response ({COPY_HINT})"
            )
    categories = {f.category for f in draft.findings}
    required = {
        "relevancy": draft.relevancy.score == 0,
        "correctness": draft.correctness is not None and draft.correctness.score == 0,
        "completeness": any(r.status is CriterionStatus.NOT_MET for r in draft.criteria),
        "hallucination_rate": any(c.status in HALLUCINATED for c in draft.claims),
    }
    for name, needed in required.items():
        if needed and name not in categories:
            errors.append(f"a finding with category {name!r} is required")
    return errors


# --- scores and verdict -----------------------------------------------------------------------


def compute_metrics(draft: EvaluationDraft, *, grounded: bool) -> Metrics:
    """Metric results from an accepted draft (pure, deterministic)."""
    na = MetricStatus.NOT_APPLICABLE

    def judged(score: JudgedScore) -> MetricResult:
        if score.score is None:
            return MetricResult(
                status=MetricStatus.INCONCLUSIVE, rationale=score.rationale, evidence=score.evidence
            )
        return MetricResult(
            status=MetricStatus.SCORED,
            score=float(score.score),
            rationale=score.rationale,
            evidence=score.evidence,
        )

    correctness = (
        judged(draft.correctness)
        if grounded and draft.correctness is not None
        else MetricResult(status=na, rationale="no reference (ungrounded)")
    )

    counted = [_CRITERION_SCORE[r.status] for r in draft.criteria if r.status in _CRITERION_SCORE]
    if counted:
        met = sum(1 for r in draft.criteria if r.status is CriterionStatus.MET)
        completeness = MetricResult(
            status=MetricStatus.SCORED,
            score=_round(sum(counted) / len(counted)),
            rationale=f"{met} of {len(draft.criteria)} success criteria met "
            f"({len(draft.criteria) - len(counted)} unverifiable)",
        )
    else:
        completeness = MetricResult(
            status=MetricStatus.INCONCLUSIVE, rationale="no success criterion could be verified"
        )

    claims = draft.claims
    if not claims:
        no_claims = "the answer makes no factual claims"
        groundedness = MetricResult(status=na, rationale=no_claims)
        hallucination = MetricResult(status=na, rationale=no_claims)
    else:
        supported = sum(1 for c in claims if c.status is ClaimStatus.SUPPORTED)
        bad = [c for c in claims if c.status in HALLUCINATED]
        groundedness = (
            MetricResult(
                status=MetricStatus.SCORED,
                score=_round(supported / len(claims)),
                rationale=f"{supported} of {len(claims)} claims backed by the reference",
            )
            if grounded
            else MetricResult(status=na, rationale="no reference (ungrounded)")
        )
        hallucination = MetricResult(
            status=MetricStatus.SCORED,
            score=_round(len(bad) / len(claims)),
            rationale=f"{len(bad)} of {len(claims)} claims "
            f"{'contradict the reference' if grounded else 'are fabricated'}",
            evidence=bad[0].text if bad else None,
        )
    return Metrics(
        relevancy=judged(draft.relevancy),
        correctness=correctness,
        groundedness=groundedness,
        hallucination_rate=hallucination,
        completeness=completeness,
    )


def failing_metrics(metrics: Metrics) -> list[MetricName]:
    """Metrics that fail the question (see the module docstring)."""
    failing: list[MetricName] = []
    for name, result in metrics.by_name().items():
        if result.status is not MetricStatus.SCORED or result.score is None:
            continue
        if name is MetricName.HALLUCINATION_RATE:
            if result.score > 0:
                failing.append(name)
        elif name is not MetricName.GROUNDEDNESS and result.score == 0:
            failing.append(name)
    return failing


def metrics_verdict(metrics: Metrics, findings: Sequence[Finding]) -> Verdict:
    supported = {f.category for f in findings if f.evidence_status is EvidenceStatus.SUPPORTED}
    failing = failing_metrics(metrics)
    if any(name.value in supported for name in failing):
        return Verdict.FAIL
    undecided = any(r.status is MetricStatus.INCONCLUSIVE for r in metrics.by_name().values())
    if failing or undecided:
        return Verdict.INCONCLUSIVE
    return Verdict.PASS


_GAP_STOPWORDS = frozenset(
    {"when", "with", "from", "that", "this", "into", "over", "than", "then", "they", "them"}
    | {"have", "been", "being", "were", "will", "would", "should", "could", "does", "done"}
    | {"link", "data", "rows", "value", "values", "product", "query", "queries", "user"}
)


def matches_known_gap(text: str, known_gaps: Sequence[str]) -> bool:
    """True if at least two keywords of a known gap (or its only keyword) occur in text."""
    words = _keywords(text)
    for gap in known_gaps:
        gap_words = _keywords(gap)
        if gap_words and len(gap_words & words) >= min(2, len(gap_words)):
            return True
    return False


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) >= 4} - _GAP_STOPWORDS


def _round(value: float) -> float:
    return round(value, 3)


def _quoted(fragment: str, text: str) -> bool:
    return _normalise(fragment) in _normalise(text)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


# --- evaluator --------------------------------------------------------------------------------


class GenericEvaluator:
    def __init__(self, llm: LLMClient, settings: AppConfig, logger: Logger) -> None:
        self._llm = llm
        self._settings = settings
        self._log = logger

    async def judge(
        self,
        product: ProductDescription,
        scenario: Scenario,
        question: QuestionCase,
        response: AnsweredResponse,
        grounding: tuple[ReferencePack, Reference] | None = None,
    ) -> GenericJudgement:
        criteria = question.expectations.success_criteria
        quotable = response_text(response)
        pack = grounding[0] if grounding is not None else None
        instructions = grounding[1].instructions_text() if grounding is not None else ""
        system, user = build_response_evaluation_prompt(
            product, scenario, question, response, grounding
        )
        judgement = GenericJudgement()
        try:
            judgement.draft = await generate_structured(
                self._llm,
                system,
                user,
                EvaluationDraft,
                max_retries=self._settings.llm.max_validation_retries,
                post_validate=lambda d: check_evaluation(d, criteria, quotable, pack, instructions),
            )
        except LLMOutputValidationError as exc:
            self._log.warning("%s: evaluation failed", question.question_id)
            judgement.notes.append(
                f"evaluation failed after {exc.attempts} attempt(s): " + "; ".join(exc.errors)
            )
        return judgement
