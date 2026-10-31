"""Evaluate one answered response with three observable NL2SQL metrics.

The judge separately scores semantic correctness, rules adherence, and schema faithfulness,
and also rates each explicit success criterion for explainability. Code computes the verdict.
Every quoted fragment must occur in the product response, and every sub-perfect score needs a
supported finding.
Invalid judge output is retried; exhausted retries produce an inconclusive evaluation.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import LLMOutputValidationError, LLMProviderError
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.evaluation import (
    NL2SQL_METRICS,
    CriterionStatus,
    EvaluationDraft,
    EvidenceStatus,
    Finding,
    MetricName,
    MetricResult,
    Metrics,
    MetricStatus,
    Verdict,
)
from synthetic_user.models.execution import AnsweredResponse, ConversationTurn
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator.nl2sql_prompt import (
    COPY_HINT,
    IGNORED_DUMMY_FILTER_NOTE,
    build_response_evaluation_prompt,
    missing_finding_message,
    response_text,
)
from synthetic_user.nodes.evaluator.no_sql import NO_SQL_REASON

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

METRIC_NAMES = tuple(m.value for m in NL2SQL_METRICS)
_SCHEMA_ONLY_LANGUAGE = re.compile(
    r"\b(?:unknown|undefined|unspecified|undocumented|unlisted|unallowed|disallowed|"
    r"fabricated|hallucinated)\s+(?:database\s+)?(?:date\s+|time\s+|year\s+)?"
    r"(?:tables?|columns?|fields?|identifiers?)\b|"
    r"\b(?:tables?|columns?|fields?|identifiers?)\b.{0,80}\bnot\s+(?:otherwise\s+)?"
    r"(?:defined|established|available|valid|listed|documented|allowed)\b|"
    r"\bnot\s+(?:in|present\s+in|listed\s+in)\s+(?:the\s+)?(?:allowed\s+)?"
    r"(?:schema|table\s+list|column\s+list|identifier\s+list)\b|"
    r"\b(?:allowed|reference)\s+(?:schema|tables?|columns?|identifiers?)\b",
    re.IGNORECASE,
)
_RULE_OR_JOIN_LANGUAGE = re.compile(
    r"\b(?:business|table|reference|privacy|currency)\s+rules?\b|"
    r"\b(?:violat(?:e|es|ed|ion)|conflict(?:s|ed|ing)?)\b.{0,80}\brules?\b|"
    r"\b(?:join|relationship|foreign\s+key)\b",
    re.IGNORECASE,
)
_UNVERIFIABLE_PENALTY_LANGUAGE = re.compile(
    r"\bunverifiable\b|\bcannot\s+(?:be\s+)?verif(?:y|ied)\b|"
    r"\bnot\s+possible\s+to\s+verif(?:y|ied)\b",
    re.IGNORECASE,
)
_BILLING_LABEL_ONLY_LANGUAGE = re.compile(
    r"(?:not|does not|without|rather than).{0,100}explicit(?:ly)?.{0,60}billing[\s-]+months?",
    re.IGNORECASE,
)
_YTD_LANGUAGE = re.compile(r"\b(?:year[\s-]+to[\s-]+date|ytd|this year so far)\b", re.IGNORECASE)
_CURRENT_YEAR_EQUALITY = re.compile(
    r"\b(?:[A-Za-z_][A-Za-z0-9_$]*\s*\.\s*)?(?:year|yr)\s*=\s*"
    r"(?:CAST\s*\(\s*)?EXTRACT\s*\(\s*YEAR\s+FROM\s+CURRENT_DATE(?:\s*\(\s*\))?\s*\)",
    re.IGNORECASE,
)
_YTD_OVERSTRICT_LANGUAGE = re.compile(
    r"\b(?:only|merely|just)\b.{0,100}\bcurrent\s+year\b|"
    r"\b(?:not|does\s+not|without|rather\s+than)\b.{0,100}"
    r"\bexplicit(?:ly)?\b.{0,70}\b(?:year[\s-]+to[\s-]+date|ytd|date\s+(?:range|window))\b",
    re.IGNORECASE,
)
_SQL_ALIAS = re.compile(r"\bAS\s+[`\"\[]?([A-Za-z_][A-Za-z0-9_$]*)[`\"\]]?", re.IGNORECASE)
_COUNT_IDENTIFIER = re.compile(
    r"\bCOUNT\s*\(\s*(?:DISTINCT\s+)?"
    r"(?:[A-Za-z_][A-Za-z0-9_$]*\s*\.\s*)?"
    r"[`\"\[]?([A-Za-z_][A-Za-z0-9_$]*)[`\"\]]?\s*\)",
    re.IGNORECASE,
)
_AGGREGATE_PRIVACY_PENALTY = re.compile(
    r"\b(?:expos[a-z]*|reveal[a-z]*|"
    r"(?:return|display|group)[a-z]*\s+(?:the\s+)?(?:raw\s+)?(?:card\s+account\s+)?"
    r"identif[a-z]*|select[a-z]*\s+(?:a\s+)?(?:card\s+account\s+)?identif[a-z]*)\b",
    re.IGNORECASE,
)
_SQL_TYPE_NAMES = frozenset(
    {"bigint", "bool", "boolean", "date", "datetime", "decimal", "float", "int", "int64"}
    | {"integer", "numeric", "string", "timestamp", "varchar"}
)
_DUMMY_CG_ID_PREDICATE = (
    r"(?:(?:[A-Za-z_][A-Za-z0-9_$]*)\s*\.\s*)?"
    r"(?:`cg_id`|\"cg_id\"|\[cg_id\]|cg_id)\s+IN\s*\(\s*'DUMMY'\s*\)"
)
_DUMMY_CG_ID_BEFORE_CONNECTOR = re.compile(
    rf"{_DUMMY_CG_ID_PREDICATE}\s+(?:AND|OR)\s+", re.IGNORECASE
)
_DUMMY_CG_ID_AFTER_CONNECTOR = re.compile(
    rf"\s+(?:AND|OR)\s+{_DUMMY_CG_ID_PREDICATE}", re.IGNORECASE
)
_DUMMY_CG_ID_ONLY_WHERE = re.compile(rf"\bWHERE\s+{_DUMMY_CG_ID_PREDICATE}", re.IGNORECASE)


@dataclass
class Nl2SqlJudgement:
    draft: EvaluationDraft | None = None
    notes: list[str] = field(default_factory=list)


def without_ignored_sql_workarounds(response: AnsweredResponse) -> tuple[AnsweredResponse, bool]:
    """Return the judge-only response after removing the exact approved dummy cg_id filter."""

    changed = False

    def clean(value: str | None) -> str | None:
        nonlocal changed
        if value is None:
            return None
        cleaned = _DUMMY_CG_ID_BEFORE_CONNECTOR.sub("", value)
        cleaned = _DUMMY_CG_ID_AFTER_CONNECTOR.sub("", cleaned)
        cleaned = _DUMMY_CG_ID_ONLY_WHERE.sub("", cleaned)
        changed = changed or cleaned != value
        return cleaned

    artifacts = []
    for artifact in response.artifacts:
        content = (
            clean(artifact.content)
            if artifact.kind.casefold() == "sql" and isinstance(artifact.content, str)
            else artifact.content
        )
        artifacts.append(artifact.model_copy(update={"content": content}))
    judged = response.model_copy(update={"text": clean(response.text), "artifacts": artifacts})
    return judged, changed


# --- draft checks -----------------------------------------------------------------------------


def check_evaluation(
    draft: EvaluationDraft,
    criteria: Sequence[str],
    response: str,
) -> list[str]:
    """Return observable contract violations in a judge draft."""
    errors: list[str] = []
    for name in NL2SQL_METRICS:
        judged = getattr(draft, name.value)
        if judged.evidence and not _quoted(judged.evidence, response):
            errors.append(
                f"{name.value}: evidence {judged.evidence!r} is not in the response ({COPY_HINT})"
            )

    wanted = {_criterion_key(c): c for c in criteria}
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
        if result.evidence and not _quoted(result.evidence, response):
            errors.append(
                f"criterion {result.criterion!r}: evidence {result.evidence!r} is not in the "
                f"response ({COPY_HINT})"
            )

    semantic_texts = [
        ("semantic correctness rationale", draft.semantic_correctness.rationale),
        *(
            ("semantic correctness what_worked", item)
            for item in draft.semantic_correctness.what_worked
        ),
        *(
            ("semantic correctness what_didnt", item)
            for item in draft.semantic_correctness.what_didnt
        ),
        *(
            (f"semantic correctness finding {index}", text)
            for index, finding in enumerate(draft.findings, start=1)
            if finding.category == MetricName.SEMANTIC_CORRECTNESS.value
            for text in (
                finding.description,
                finding.expected_behavior,
                finding.observed_behavior,
            )
        ),
    ]
    schema_only = [label for label, text in semantic_texts if _SCHEMA_ONLY_LANGUAGE.search(text)]
    if schema_only:
        errors.append(
            "schema-name allow-list problems belong only to schema faithfulness, not semantic "
            "correctness "
            f"({', '.join(schema_only)})"
        )
    rule_or_join_in_semantics = [
        label for label, text in semantic_texts if _RULE_OR_JOIN_LANGUAGE.search(text)
    ]
    if rule_or_join_in_semantics:
        errors.append(
            "business rules and documented joins belong only to rules adherence, not semantic "
            f"correctness ({', '.join(rule_or_join_in_semantics)})"
        )
    billing_wording_penalties = [
        label
        for label, text in [
            *semantic_texts,
            *(
                (f"criterion {result.criterion!r}", result.rationale)
                for result in draft.criteria
                if "billing month" in _normalise(result.criterion)
            ),
        ]
        if _BILLING_LABEL_ONLY_LANGUAGE.search(text)
    ]
    if (
        re.search(r"\byr_mo_no\b", response, re.IGNORECASE)
        and any("billing month" in _normalise(criterion) for criterion in criteria)
        and billing_wording_penalties
    ):
        errors.append(
            "using yr_mo_no with calculated month boundaries is a billing-month filter; do not "
            "penalize it merely for not repeating the words 'billing month' "
            f"({', '.join(billing_wording_penalties)})"
        )
    ytd_penalties = [
        label
        for label, text in [
            *semantic_texts,
            *(
                (f"criterion {result.criterion!r}", result.rationale)
                for result in draft.criteria
                if _YTD_LANGUAGE.search(result.criterion)
            ),
        ]
        if _YTD_OVERSTRICT_LANGUAGE.search(text)
    ]
    if (
        _CURRENT_YEAR_EQUALITY.search(response)
        and any(_YTD_LANGUAGE.search(criterion) for criterion in criteria)
        and ytd_penalties
    ):
        errors.append(
            "current-year equality is a reasonable semantic implementation of year-to-date; "
            "do not penalize it solely for lacking an explicit date range "
            f"({', '.join(ytd_penalties)})"
        )
    if any(result.status is CriterionStatus.UNVERIFIABLE for result in draft.criteria):
        unverifiable_penalties = [
            label for label, text in semantic_texts if _UNVERIFIABLE_PENALTY_LANGUAGE.search(text)
        ]
        if unverifiable_penalties:
            errors.append(
                "an unverifiable criterion is neutral and cannot be used as a semantic "
                "correctness gap "
                f"({', '.join(unverifiable_penalties)})"
            )

    aliases = {
        alias.casefold()
        for alias in _SQL_ALIAS.findall(response)
        if alias.casefold() not in _SQL_TYPE_NAMES
    }
    rules_penalty_texts = [
        ("rules adherence rationale", draft.rules_adherence.rationale),
        *(("rules adherence what_didnt", item) for item in draft.rules_adherence.what_didnt),
        *(
            (f"rules adherence finding {index}", text)
            for index, finding in enumerate(draft.findings, start=1)
            if finding.category == MetricName.RULES_ADHERENCE.value
            for text in (
                finding.description,
                finding.expected_behavior,
                finding.observed_behavior,
            )
        ),
    ]
    schema_in_rules = [
        label for label, text in rules_penalty_texts if _SCHEMA_ONLY_LANGUAGE.search(text)
    ]
    if schema_in_rules:
        errors.append(
            "table and column allow-list problems belong only to schema faithfulness, not rules "
            f"adherence ({', '.join(schema_in_rules)})"
        )
    schema_penalty_texts = [
        ("schema faithfulness rationale", draft.schema_faithfulness.rationale),
        *(
            ("schema faithfulness what_didnt", item)
            for item in draft.schema_faithfulness.what_didnt
        ),
        *(
            (f"schema faithfulness finding {index}", text)
            for index, finding in enumerate(draft.findings, start=1)
            if finding.category == MetricName.SCHEMA_FAITHFULNESS.value
            for text in (
                finding.description,
                finding.expected_behavior,
                finding.observed_behavior,
            )
        ),
    ]
    rules_in_schema = [
        label for label, text in schema_penalty_texts if _RULE_OR_JOIN_LANGUAGE.search(text)
    ]
    if rules_in_schema:
        errors.append(
            "applicable rules and joins belong only to rules adherence; schema faithfulness "
            f"checks only allowed table and column names ({', '.join(rules_in_schema)})"
        )
    aggregate_only_identifiers = _aggregate_only_identifiers(response)
    aggregate_privacy_penalties = [
        label
        for label, text in rules_penalty_texts
        if _AGGREGATE_PRIVACY_PENALTY.search(text)
        and (
            any(
                re.search(rf"\b{re.escape(identifier)}\b", text, re.IGNORECASE)
                for identifier in aggregate_only_identifiers
            )
            or re.search(r"\b(?:identifier|card number|account number)\b", text, re.IGNORECASE)
        )
    ]
    if aggregate_privacy_penalties:
        errors.append(
            "an identifier used only inside COUNT or COUNT(DISTINCT) is not exposed in the "
            "result; do not penalize aggregate-only use under a raw-identifier privacy rule "
            f"({', '.join(aggregate_privacy_penalties)})"
        )
    alias_penalties = [
        label
        for label, text in schema_penalty_texts
        if any(re.search(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE) for alias in aliases)
    ]
    if (
        draft.schema_faithfulness.score is not None
        and draft.schema_faithfulness.score < 1
        and alias_penalties
    ):
        errors.append(
            "SQL output aliases are not database columns and cannot lower schema faithfulness "
            f"({', '.join(alias_penalties)})"
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
    for name in NL2SQL_METRICS:
        score = getattr(draft, name.value).score
        if score is not None and score < 1 and name.value not in categories:
            errors.append(missing_finding_message(name.value, score))
    return errors


def ignore_sql_output_alias_penalties(draft: EvaluationDraft, response: str) -> None:
    """Remove schema penalties that treat SQL output aliases as database columns."""

    aliases = {
        alias.casefold()
        for alias in _SQL_ALIAS.findall(response)
        if alias.casefold() not in _SQL_TYPE_NAMES
    }
    if not aliases:
        return

    def alias_only(finding: object) -> bool:
        evidence = getattr(finding, "evidence", None)
        if not isinstance(evidence, str):
            return False
        normalized = evidence.strip(' `"[]').casefold()
        return normalized in aliases

    kept = [
        finding
        for finding in draft.findings
        if finding.category != MetricName.SCHEMA_FAITHFULNESS.value or not alias_only(finding)
    ]
    if len(kept) == len(draft.findings):
        return
    draft.findings = kept
    schema_findings = [
        finding for finding in kept if finding.category == MetricName.SCHEMA_FAITHFULNESS.value
    ]
    filtered_gaps = [
        gap
        for gap in draft.schema_faithfulness.what_didnt
        if not any(re.search(rf"\b{re.escape(alias)}\b", gap, re.IGNORECASE) for alias in aliases)
    ]
    if schema_findings:
        draft.schema_faithfulness.what_didnt = filtered_gaps
        return
    draft.schema_faithfulness = draft.schema_faithfulness.model_copy(
        update={
            "score": 1,
            "rationale": (
                "All explicit database table and column identifiers are allowed; SQL output "
                "aliases are not schema columns."
            ),
            "what_didnt": [],
        }
    )


# --- scores and verdict -----------------------------------------------------------------------


def compute_metrics(
    draft: EvaluationDraft,
    *,
    rules_applicable: bool = True,
    schema_applicable: bool = True,
    sql_generated: bool = True,
) -> Metrics:
    """Metric results from an accepted draft (pure, deterministic)."""

    def judged(name: MetricName) -> MetricResult:
        score = getattr(draft, name.value)
        if not sql_generated:
            if name is MetricName.SEMANTIC_CORRECTNESS:
                return MetricResult(
                    status=MetricStatus.SCORED,
                    score=0,
                    rationale=NO_SQL_REASON,
                    what_didnt=[NO_SQL_REASON],
                )
            return MetricResult(
                status=MetricStatus.NOT_APPLICABLE,
                rationale="no SQL query was generated to assess",
            )
        if name is MetricName.RULES_ADHERENCE and not rules_applicable:
            return MetricResult(
                status=MetricStatus.NOT_APPLICABLE,
                rationale="no applicable table/business rules or documented joins were supplied",
            )
        if name is MetricName.SCHEMA_FAITHFULNESS and not schema_applicable:
            return MetricResult(
                status=MetricStatus.NOT_APPLICABLE,
                rationale="no table/column reference pack was supplied for this question",
            )
        if score.score is None:
            return MetricResult(
                status=MetricStatus.INCONCLUSIVE,
                rationale=score.rationale,
                evidence=score.evidence,
                what_worked=score.what_worked,
                what_didnt=score.what_didnt,
            )
        return MetricResult(
            status=MetricStatus.SCORED,
            score=float(score.score),
            rationale=score.rationale,
            evidence=score.evidence,
            what_worked=score.what_worked,
            what_didnt=score.what_didnt,
        )

    return Metrics(
        semantic_correctness=judged(MetricName.SEMANTIC_CORRECTNESS),
        rules_adherence=judged(MetricName.RULES_ADHERENCE),
        schema_faithfulness=judged(MetricName.SCHEMA_FAITHFULNESS),
    )


def failing_metrics(metrics: Metrics) -> list[MetricName]:
    """Metrics that fail the question (see the module docstring)."""
    failing: list[MetricName] = []
    for name, result in metrics.by_name().items():
        if result.status is not MetricStatus.SCORED or result.score is None:
            continue
        if result.score < 1:
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


def _quoted(fragment: str, text: str) -> bool:
    return _normalise(fragment) in _normalise(text)


def _aggregate_only_identifiers(text: str) -> set[str]:
    identifiers = {match.group(1).casefold() for match in _COUNT_IDENTIFIER.finditer(text)}
    without_aggregates = _COUNT_IDENTIFIER.sub(" ", text)
    return {
        identifier
        for identifier in identifiers
        if not re.search(rf"\b{re.escape(identifier)}\b", without_aggregates, re.IGNORECASE)
    }


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _criterion_key(text: str) -> str:
    """Tolerate an LLM adding an ellipsis inside otherwise verbatim SQL notation."""

    return _normalise(re.sub(r"\s*\.\.\.\s*", "", text))


# --- evaluator --------------------------------------------------------------------------------


class Nl2SqlEvaluator:
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
        conversation: Sequence[ConversationTurn] = (),
    ) -> Nl2SqlJudgement:
        ignored_workaround = False
        judged_response = response
        if self._settings.evaluation.ignore_dummy_cg_id_filter:
            judged_response, ignored_workaround = without_ignored_sql_workarounds(response)
        criteria = question.expectations.success_criteria
        quotable = response_text(judged_response)
        system, user = build_response_evaluation_prompt(
            product,
            scenario,
            question,
            judged_response,
            grounding,
            conversation,
            artifact_max_chars=self._settings.evaluation.artifact_max_chars,
        )
        judgement = Nl2SqlJudgement(notes=[IGNORED_DUMMY_FILTER_NOTE] if ignored_workaround else [])
        try:

            def validate(draft: EvaluationDraft) -> list[str]:
                ignore_sql_output_alias_penalties(draft, quotable)
                return check_evaluation(draft, criteria, quotable)

            judgement.draft = await generate_structured(
                self._llm,
                system,
                user,
                EvaluationDraft,
                max_retries=self._settings.llm.max_validation_retries,
                post_validate=validate,
            )
        except LLMOutputValidationError as exc:
            self._log.warning("%s: evaluation failed", question.question_id)
            judgement.notes.append(
                f"evaluation failed after {exc.attempts} attempt(s): " + "; ".join(exc.errors)
            )
        except LLMProviderError as exc:
            # A transient judge outage must not discard the other completed judgements in the
            # batch. The answered response remains visible as an inconclusive evaluation and can
            # be retried by re-running this step.
            self._log.warning("%s: evaluation provider call failed: %s", question.question_id, exc)
            judgement.notes.append(f"evaluation provider call failed: {exc}")
        return judgement
