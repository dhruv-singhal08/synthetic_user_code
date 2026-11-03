"""Compute run KPIs from stored artifacts (pure code, no LLM) and render them as CSV.

Definitions:
- one KPI row per SELECTED metric (frozen metric config; legacy runs: the ids they recorded);
  mean / pass rate over responses where the metric was scored (n = the denominator); every
  non-scored status (not_applicable, cannot_generate, evidence_unavailable, inconclusive,
  evaluation_error, unavailable) is counted separately with its most frequent reasons, never
  averaged in;
- metric pass rate = share scoring the best value;
- MRR = mean reciprocal rank over questions with ranked chunks + relevance labels; per-question
  rows and exclusions are reported;
- categories are descriptive groupings only (no composite category score);
- accuracy = pass verdicts / judged responses (provisional verdict policy);
- availability = 1 - (error + timeout) / questions asked;
- latency p50 / p95 (nearest rank) over executions with a measured latency;
- a bug is a known-gap bug when every one of its findings matches a known gap, else new;
  its severity is its most severe finding;
- inconclusive rate = inconclusive verdicts / judged answers.
"""

from __future__ import annotations

import csv
import io
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from synthetic_user.metrics.catalog import builtin, builtin_definitions, category_labels
from synthetic_user.models.evaluation import (
    EvaluationResult,
    Finding,
    MetricStatus,
    Severity,
    Verdict,
)
from synthetic_user.models.execution import ExecutionRecord, ProductOutcome
from synthetic_user.models.kpi import (
    MAX_ISSUE_EXAMPLES,
    CompilerFailure,
    ConsolidatedIssue,
    GroupKpi,
    IssueSummary,
    MetricKpi,
    MrrKpi,
    ReciprocalRankRow,
    RunKpis,
)
from synthetic_user.models.metrics import EvidenceKind, ResolvedMetricConfig
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator.bugs import consolidate_bugs
from synthetic_user.nodes.evaluator.no_sql import NO_SQL_REASON

_SEVERITY_RANK = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
MRR_METRIC_ID = "mrr"
MAX_REASONS = 5
CONTEXT_RECALL_NOTE = (
    "expectation-based coverage of each question's required-information checklist, not "
    "exhaustive corpus recall"
)
# Display names of metric ids written by earlier versions (kept, never relabelled).
LEGACY_METRIC_LABELS = {
    "completion": "Completion (legacy)",
    "tool_selection": "Tool Selection (legacy)",
}


@dataclass(frozen=True)
class _IssueTheme:
    key: str
    title: str
    recommended_action: str
    pattern: re.Pattern[str]


# Ordered, product-actionable taxonomy. A finding belongs to exactly one theme; wording similarity
# never creates a new category, so reports remain comparable from run to run.
_ISSUE_THEMES = (
    _IssueTheme(
        "missing_requested_output",
        "Requested output was not produced",
        "Return the requested SQL/result, or ask one targeted clarification when required.",
        re.compile(
            r"\brefus(?:e|es|ed|al)\b|\bno sql\b|\bwithout (?:any )?sql\b|"
            r"(?:fails?|failed) to (?:provide|produce|generate|return).{0,45}\bsql\b|"
            r"does not (?:provide|produce|generate|return|include).{0,45}\bsql\b|"
            r"generic (?:unsupported|inability|refusal) message",
            re.IGNORECASE,
        ),
    ),
    _IssueTheme(
        "schema_and_identifiers",
        "Uses undocumented tables or columns",
        "Generate SQL using only tables and columns allowed by the question's cheat sheet.",
        re.compile(
            r"\b(?:undocumented|unallowed|unlisted|unknown)\b.{0,55}"
            r"\b(?:column|table|field|identifier|schema)\b|"
            r"\b(?:column|table|field|identifier)\b.{0,70}"
            r"(?:not (?:in|present in|included in|listed in)|outside).{0,35}"
            r"(?:allowed|schema|cheat sheet)|"
            r"\ballowed (?:schema|column|table).{0,45}\b(?:missing|mismatch|inconsistent)",
            re.IGNORECASE,
        ),
    ),
    _IssueTheme(
        "joins_and_relationships",
        "Uses a missing or invalid join",
        "Use the documented relationship and join keys required for the requested dimensions.",
        re.compile(r"\bjoin(?:s|ed|ing)?\b|\brelationship\b|\bjoin keys?\b", re.IGNORECASE),
    ),
    _IssueTheme(
        "reference_rule_compliance",
        "Violates an applicable business or data rule",
        "Apply the relevant cheat-sheet rule without introducing restrictions unrelated to "
        "the ask.",
        re.compile(
            r"\b(?:violat(?:e|es|ed|ion)|contrary|conflict)\b.{0,80}\b(?:rule|requirement)\b|"
            r"\b(?:sensitive|privacy|currency|currencies|date availability)\b|"
            r"\bexpos(?:e|es|ed|ing)\b.{0,50}\bidentifier\b",
            re.IGNORECASE,
        ),
    ),
    _IssueTheme(
        "unsupported_request_handling",
        "Unsupported request was handled incorrectly",
        "State the limitation and ask for the missing input or offer a supported alternative.",
        re.compile(
            r"\bunsupported\b|\bnot supported\b|\bmissing (?:dimension|metric)\b|"
            r"\bpropose an alternative\b|\bflag (?:the|that).{0,40}\blimitation\b",
            re.IGNORECASE,
        ),
    ),
    _IssueTheme(
        "clarification_handling",
        "Clarification was missing or mishandled",
        "Ask only for information needed to resolve a material ambiguity, then use the answer.",
        re.compile(r"\bclarif(?:y|ies|ied|ication)\b|\bambig(?:uous|uity)\b", re.IGNORECASE),
    ),
    _IssueTheme(
        "time_window_and_period",
        "Uses the wrong or incomplete time window",
        "Apply the requested date, month, quarter, or comparison period using documented fields.",
        re.compile(
            r"\b(?:year[- ]to[- ]date|ytd|month|quarter|date|period|recent|time window|"
            r"year[- ]over[- ]year)\b",
            re.IGNORECASE,
        ),
    ),
    _IssueTheme(
        "filters_and_scope",
        "Applies an incorrect filter or scope",
        "Align filters with the requested population and remove scope-changing extras.",
        re.compile(r"\bfilter(?:s|ed|ing)?\b|\bscope\b|\brestrict(?:s|ed|ing)?\b", re.IGNORECASE),
    ),
    _IssueTheme(
        "aggregation_and_calculation",
        "Uses an incorrect aggregation or calculation",
        "Use the requested measure, aggregation grain, and documented formula.",
        re.compile(
            r"\b(?:sum|count|average|avg|aggregate|aggregation|calculation|formula|ratio)\b",
            re.IGNORECASE,
        ),
    ),
    _IssueTheme(
        "grouping_ranking_and_limits",
        "Groups, ranks, or limits results incorrectly",
        "Use the requested grouping dimension, ordering direction, and top-N limit.",
        re.compile(
            r"\b(?:group(?:s|ed|ing)?|rank(?:s|ed|ing)?|order(?:s|ed|ing)?|top[- ]?n|limit)\b",
            re.IGNORECASE,
        ),
    ),
    _IssueTheme(
        "wrong_intent_or_dimension",
        "Answers a different intent, metric, or dimension",
        "Align the generated answer with the requested business question and dimensions.",
        re.compile(
            r"\bdoes not (?:address|respond|engage)\b|\binstead (?:returns?|uses?|provides?)\b|"
            r"\bwrong (?:intent|metric|dimension|question)\b",
            re.IGNORECASE,
        ),
    ),
    _IssueTheme(
        "response_format",
        "Required response format is missing",
        "Return all explicitly requested response components in the product's supported format.",
        re.compile(r"\b(?:format|explanation|widget|artifact|output component)\b", re.IGNORECASE),
    ),
)

# Fallback theme wording for built-in (and legacy) metric ids; any other id - e.g. a compiled
# custom metric - gets a generic theme, so consolidation never fails on an unknown category.
_FALLBACK_TEXT = {
    "semantic_correctness": (
        "Query does not produce the requested result",
        "Implement every applicable expected criterion for the requested result.",
    ),
    "rules_adherence": (
        "Query violates an applicable rule or join",
        "Follow the applicable table/business rules and documented joins.",
    ),
    "schema_faithfulness": (
        "Query uses undocumented tables or columns",
        "Use only the allowed table and column names.",
    ),
    "relevance": ("Answer does not address the request", "Respond directly to the user's request."),
    "correctness_completeness": (
        "Requested output is incomplete or incorrect",
        "Deliver every requested part, consistent with the trusted evidence.",
    ),
    "completion": ("Requested task is incomplete", "Complete every requested output and action."),
    "tool_correctness": (
        "Tool, source or arguments do not match the task",
        "Align tool, source, arguments, grouping, filters and time window with the request.",
    ),
    "tool_selection": (
        "Tool or data-source selection does not match the task",
        "Align tool, source, metric, grouping, filters and time window with the request.",
    ),
    "faithfulness": (
        "Answer claims are not supported by retrieved evidence",
        "Ground every material claim in the retrieved context or tool results.",
    ),
    "citation_correctness": (
        "Citations do not support the cited claims",
        "Cite the specific retrieved chunk that supports each claim.",
    ),
    "context_precision": (
        "Retrieved context is not sufficiently relevant",
        "Retrieve focused context that is relevant to the user's request.",
    ),
    "context_recall": (
        "Retrieved context misses required information",
        "Retrieve the information the question's checklist requires.",
    ),
}


def _fallback_theme(category: str, labels: Mapping[str, str]) -> _IssueTheme:
    title, action = _FALLBACK_TEXT.get(
        category,
        (
            f"{labels.get(category, category)} issues",
            f"Review the findings for {labels.get(category, category)}.",
        ),
    )
    key = "other_" + re.sub(r"[^a-z0-9_]+", "_", category.casefold()).strip("_")
    return _IssueTheme(key, title, action, re.compile(r"$^"))


def consolidate_findings(
    evaluations: Sequence[EvaluationResult],
    executions: Sequence[ExecutionRecord] = (),
    metric_config: ResolvedMetricConfig | None = None,
) -> IssueSummary:
    """Group findings into stable, product-actionable themes without another LLM call."""

    labels = metric_labels(metric_config, evaluations)
    no_sql_questions = {e.question_id for e in evaluations if NO_SQL_REASON in e.notes}
    no_sql_findings = [
        finding
        for evaluation in evaluations
        if evaluation.question_id in no_sql_questions
        for finding in evaluation.findings
        if finding.exempted_by is None  # an applied guidance exception is not an issue (D243)
    ]

    rows = sorted(
        (
            (evaluation.question_id, finding)
            for evaluation in evaluations
            if evaluation.question_id not in no_sql_questions
            for finding in evaluation.findings
            if finding.exempted_by is None
        ),
        key=lambda item: (
            item[1].matches_known_gap,
            -_SEVERITY_RANK[item[1].severity],
            item[1].description.casefold(),
            item[0],
        ),
    )
    groups: dict[str, tuple[_IssueTheme, list[tuple[str, Finding]]]] = {}
    for question_id, finding in rows:
        theme = _issue_theme(finding, labels)
        groups.setdefault(theme.key, (theme, []))[1].append((question_id, finding))

    issues = []
    ordered_groups = sorted(
        groups.values(),
        key=lambda group: (
            -max(_SEVERITY_RANK[finding.severity] for _, finding in group[1]),
            -len({question_id for question_id, _ in group[1]}),
            group[0].title,
        ),
    )
    for theme, items in ordered_groups:
        descriptions = sorted(
            {finding.description for _, finding in items}, key=lambda text: (len(text), text)
        )
        questions = sorted({question_id for question_id, _ in items}, key=_numeric_id)
        worst = max(
            (finding.severity for _, finding in items),
            key=lambda value: _SEVERITY_RANK[value],
        )
        issues.append(
            ConsolidatedIssue(
                issue_id="ISSUE_999",
                theme=theme.key,
                categories=sorted({finding.category for _, finding in items}),
                title=theme.title,
                recommended_action=theme.recommended_action,
                severity=worst,
                occurrences=len(items),
                affected_question_ids=questions,
                finding_ids=[finding.finding_id for _, finding in items],
                examples=descriptions[:MAX_ISSUE_EXAMPLES],
                known_gap=all(finding.matches_known_gap for _, finding in items),
            )
        )
    if no_sql_questions:
        categories = {finding.category for finding in no_sql_findings}
        if any(
            execution.question_id in no_sql_questions
            and execution.outcome is ProductOutcome.TIMEOUT
            for execution in executions
        ):
            categories.add("availability")
        if not categories:
            categories.add("semantic_correctness")
        worst = max(
            (finding.severity for finding in no_sql_findings),
            key=lambda value: _SEVERITY_RANK[value],
            default=Severity.HIGH,
        )
        issues.append(
            ConsolidatedIssue(
                issue_id="ISSUE_999",
                theme="no_sql_generated",
                categories=sorted(categories),
                title="No SQL generated",
                recommended_action="Return a SQL query before the request times out.",
                severity=worst,
                occurrences=len(no_sql_questions),
                affected_question_ids=sorted(no_sql_questions, key=_numeric_id),
                finding_ids=sorted(
                    {finding.finding_id for finding in no_sql_findings}, key=_numeric_id
                ),
                examples=[NO_SQL_REASON],
                known_gap=False,
            )
        )
    issues = [
        issue.model_copy(update={"issue_id": f"ISSUE_{index:03d}"})
        for index, issue in enumerate(
            sorted(
                issues,
                key=lambda issue: (
                    -_SEVERITY_RANK[issue.severity],
                    -len(issue.affected_question_ids),
                    issue.title,
                ),
            ),
            start=1,
        )
    ]
    return IssueSummary(
        total_findings=len(rows) + len(no_sql_findings),
        affected_questions=len({question_id for question_id, _ in rows} | no_sql_questions),
        issues=issues,
    )


def _issue_theme(finding: Finding, labels: Mapping[str, str]) -> _IssueTheme:
    """Report grouping only (not a validation gate): first matching theme, else the metric's."""
    text = " ".join(
        (
            finding.description,
            finding.expected_behavior,
            finding.observed_behavior,
        )
    )
    return next(
        (theme for theme in _ISSUE_THEMES if theme.pattern.search(text)),
        _fallback_theme(finding.category, labels),
    )


def _numeric_id(value: str) -> tuple[str, int]:
    prefix, _, suffix = value.rpartition("_")
    return prefix, int(suffix) if suffix.isdigit() else 0


def compute_kpis(
    *,
    run_id: str,
    product_id: str,
    mode: GenerationMode,
    scenarios: Sequence[Scenario],
    executions: Sequence[ExecutionRecord],
    evaluations: Sequence[EvaluationResult],
    metric_config: ResolvedMetricConfig | None = None,
    now: datetime | None = None,
) -> RunKpis:
    outcomes = Counter(e.outcome.value for e in executions)
    judged = [e for e in evaluations if e.verdict is not None]
    verdicts = Counter(e.verdict.value for e in judged if e.verdict is not None)
    asked = len(executions)
    failed_calls = outcomes[ProductOutcome.ERROR.value] + outcomes[ProductOutcome.TIMEOUT.value]
    latencies = sorted(e.latency_ms for e in executions if e.latency_ms is not None)

    new_bugs: Counter[str] = Counter()
    known = 0
    # One count per bug (root cause), however many questions it affects (D244).
    by_id = {f.finding_id: f for ev in evaluations for f in ev.findings}
    for bug in consolidate_bugs(evaluations):
        findings = [by_id[i] for i in bug.finding_ids if i in by_id]
        if findings and all(f.matches_known_gap for f in findings):
            known += 1
            continue
        worst = max(findings, key=lambda f: _SEVERITY_RANK[f.severity])
        new_bugs[worst.severity.value] += 1

    by_scenario = {s.id: s for s in scenarios}
    metric_ids = _metric_ids(metric_config, evaluations)
    labels = metric_labels(metric_config, evaluations)
    accuracy = _ratio(verdicts[Verdict.PASS.value], len(judged))
    classifications = Counter(
        e.response_classification.value if e.response_classification else "none"
        for e in evaluations
    )
    policies = sorted({e.verdict_policy for e in judged if e.verdict_policy})
    return RunKpis(
        run_id=run_id,
        product_id=product_id,
        mode=mode,
        computed_at=now or datetime.now(UTC),
        questions_asked=asked,
        outcomes={o.value: outcomes.get(o.value, 0) for o in ProductOutcome},
        response_classifications=dict(sorted(classifications.items())),
        judged=len(judged),
        verdicts={v.value: verdicts.get(v.value, 0) for v in Verdict},
        accuracy=accuracy,
        task_success_rate=accuracy,
        metric_config_id=metric_config.config_id if metric_config is not None else None,
        verdict_policy=", ".join(policies) or None,
        selected_metrics=list(metric_config.selected_metric_ids) if metric_config else [],
        category_groups=_category_groups(metric_config),
        compiler_failures=[
            CompilerFailure(
                metric_id=d.metric_id,
                source_text=d.source_text or d.description,
                provenance=d.source_provenance or "custom metric",
                reason=d.unavailable_reason or "not compiled",
            )
            for d in (metric_config.unavailable() if metric_config is not None else [])
        ],
        metrics=[_metric_kpi(metric_id, judged, metric_config, labels) for metric_id in metric_ids],
        mrr=_mrr(evaluations, metric_config),
        availability=_round(1 - failed_calls / asked) if asked else None,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        new_bugs=sum(new_bugs.values()),
        new_bugs_by_severity={s.value: new_bugs.get(s.value, 0) for s in Severity},
        known_gap_bugs=known,
        inconclusive_rate=_ratio(verdicts[Verdict.INCONCLUSIVE.value], len(judged)),
        by_task_type=_breakdown(
            evaluations, lambda ev: _scenario_attr(by_scenario, ev, "task_type"), metric_ids
        ),
        by_capability=_breakdown(
            evaluations, lambda ev: _scenario_attr(by_scenario, ev, "capability"), metric_ids
        ),
    )


def metric_labels(
    metric_config: ResolvedMetricConfig | None, evaluations: Sequence[EvaluationResult] = ()
) -> dict[str, str]:
    """metric id -> display name (frozen config first; built-in catalog; legacy names)."""
    labels = {d.metric_id: d.display_name for d in builtin_definitions()}
    labels.update(LEGACY_METRIC_LABELS)
    if metric_config is not None:
        labels.update({d.metric_id: d.display_name for d in metric_config.definitions})
    for evaluation in evaluations:
        for metric_id in evaluation.metrics:
            labels.setdefault(metric_id, metric_id.replace("_", " ").title())
    return labels


def _metric_ids(
    metric_config: ResolvedMetricConfig | None, evaluations: Sequence[EvaluationResult]
) -> list[str]:
    if metric_config is not None:
        return list(metric_config.selected_metric_ids)
    # Legacy runs: the metrics their evaluations actually recorded, original ids kept.
    return list(dict.fromkeys(metric_id for ev in evaluations for metric_id in ev.metrics))


def _category_groups(metric_config: ResolvedMetricConfig | None) -> dict[str, list[str]]:
    if metric_config is None:
        return {}
    names = category_labels()
    groups: dict[str, list[str]] = {}
    for definition in metric_config.definitions:
        groups.setdefault(names[definition.category], []).append(definition.metric_id)
    return groups


def kpis_csv(kpis: RunKpis) -> str:
    """One row per KPI, for spreadsheets: kpi, value, n, excluded, note."""
    rows: list[list[str]] = [["kpi", "value", "n", "excluded", "note"]]

    def row(
        kpi: str, value: float | None, n: int | str = "", excluded: str = "", note: str = ""
    ) -> None:
        rows.append([kpi, "" if value is None else f"{value:g}", str(n), excluded, note])

    row(
        "Accuracy",
        kpis.accuracy,
        kpis.judged,
        note="pass verdicts / evaluated responses (answers + substantive denials); "
        f"verdict policy {kpis.verdict_policy or 'n/a'}. task_success_rate in "
        "kpis.json is the same value, kept for older readers",
    )
    for m in kpis.metrics:
        label = m.display_name or m.metric
        excluded = (
            f"{m.not_applicable} not applicable, {m.cannot_generate} cannot generate, "
            f"{m.evidence_unavailable} evidence unavailable, {m.inconclusive} inconclusive, "
            f"{m.evaluation_error} evaluation error, {m.unavailable} unavailable"
        )
        direction = "lower is better" if m.lower_is_better else "higher is better"
        note = "; ".join([direction, *m.notes])
        row(f"{label} (mean)", m.mean, m.n, excluded, note)
        best = "share scoring 0" if m.lower_is_better else "share scoring 1"
        row(f"{label} (pass rate)", m.pass_rate, m.n, excluded, best)
    if kpis.mrr is not None:
        row(
            "MRR (run level)",
            kpis.mrr.value,
            kpis.mrr.denominator,
            f"{kpis.mrr.excluded_missing_rank} missing rank, {kpis.mrr.excluded_other} other",
            kpis.mrr.definition,
        )
    for failure in kpis.compiler_failures:
        row(f"Custom metric {failure.metric_id} (unavailable)", None, note=failure.reason)
    row(
        "Availability",
        kpis.availability,
        kpis.questions_asked,
        note="1 - (error + timeout) / asked",
    )
    row("Latency p50 (ms)", kpis.latency_p50_ms)
    row("Latency p95 (ms)", kpis.latency_p95_ms)
    severities = ", ".join(f"{k} {v}" for k, v in kpis.new_bugs_by_severity.items() if v)
    row("New bugs", kpis.new_bugs, note=severities or "none")
    row("Known-gap bugs", kpis.known_gap_bugs, note="every finding matches a known gap")
    row("Inconclusive rate", kpis.inconclusive_rate, kpis.judged)
    not_judged = ", ".join(
        f"{k} {v}"
        for k, v in kpis.outcomes.items()
        if k
        in (
            ProductOutcome.CLARIFICATION.value,
            ProductOutcome.ERROR.value,
            ProductOutcome.TIMEOUT.value,
        )
        and v
    )
    row("Questions asked", kpis.questions_asked, note=f"recorded only: {not_judged or 'none'}")
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    return buffer.getvalue()


def _metric_kpi(
    metric_id: str,
    judged: Sequence[EvaluationResult],
    metric_config: ResolvedMetricConfig | None,
    labels: Mapping[str, str],
) -> MetricKpi:
    results = [r for ev in judged if (r := ev.metrics.get(metric_id)) is not None]
    scores = [r.score for r in results if r.status is MetricStatus.SCORED and r.score is not None]
    definition = metric_config.definition(metric_id) if metric_config is not None else None
    builtin_definition = builtin(metric_id)
    lower = definition is not None and not definition.higher_is_better
    best = 0.0 if lower else 1.0
    counts = Counter(r.status for r in results)
    reasons = Counter(
        f"{r.status.value}: {r.rationale}" for r in results if r.status is not MetricStatus.SCORED
    )
    notes: list[str] = []
    if definition is not None and not definition.available:
        notes.append(f"unavailable: {definition.unavailable_reason}")
    if any("disclosure" in r.details for r in results):
        notes.append(CONTEXT_RECALL_NOTE)
    if definition is not None and definition.continuous_score:
        notes.append("mean of per-question reciprocal ranks over scored questions (= MRR)")
    origin = (
        definition.origin.value
        if definition is not None
        else ("builtin" if builtin_definition is not None else "legacy")
    )
    category = definition or builtin_definition
    return MetricKpi(
        metric=metric_id,
        display_name=labels.get(metric_id, metric_id),
        category=category_labels()[category.category] if category is not None else None,
        origin=origin,
        mean=_round(sum(scores) / len(scores)) if scores else None,
        pass_rate=_ratio(sum(1 for s in scores if s == best), len(scores)),
        n=len(scores),
        not_applicable=counts[MetricStatus.NOT_APPLICABLE],
        inconclusive=counts[MetricStatus.INCONCLUSIVE],
        cannot_generate=counts[MetricStatus.CANNOT_GENERATE],
        evidence_unavailable=counts[MetricStatus.EVIDENCE_UNAVAILABLE],
        evaluation_error=counts[MetricStatus.EVALUATION_ERROR],
        unavailable=counts[MetricStatus.UNAVAILABLE],
        lower_is_better=lower,
        reasons=dict(reasons.most_common(MAX_REASONS)),
        notes=notes,
    )


def _mrr(
    evaluations: Sequence[EvaluationResult], metric_config: ResolvedMetricConfig | None
) -> MrrKpi | None:
    if metric_config is None or metric_config.definition(MRR_METRIC_ID) is None:
        return None
    rows: list[ReciprocalRankRow] = []
    for ev in evaluations:
        result = ev.metrics.get(MRR_METRIC_ID)
        if result is None:
            rows.append(
                ReciprocalRankRow(
                    question_id=ev.question_id,
                    reciprocal_rank=None,
                    status=f"excluded: {ev.product_outcome.value} (not evaluated)",
                )
            )
            continue
        first = result.details.get("first_relevant_rank")
        rows.append(
            ReciprocalRankRow(
                question_id=ev.question_id,
                reciprocal_rank=result.score,
                first_relevant_rank=first if isinstance(first, int) else None,
                status="scored"
                if result.status is MetricStatus.SCORED
                else f"excluded: {result.status.value}",
            )
        )
    scored = [r.reciprocal_rank for r in rows if r.reciprocal_rank is not None]
    missing_rank = sum(
        1
        for ev in evaluations
        if (r := ev.metrics.get(MRR_METRIC_ID)) is not None
        and r.missing_requirement == EvidenceKind.RANKED_CONTEXT.value
    )
    return MrrKpi(
        value=_round(sum(scored) / len(scored)) if scored else None,
        denominator=len(scored),
        excluded_missing_rank=missing_rank,
        excluded_other=len(rows) - len(scored) - missing_rank,
        per_question=rows,
        definition=(
            "mean over questions with ranked retrieved chunks and relevance labels of "
            "1 / rank of the first relevant chunk (0 when none is relevant or nothing was "
            "retrieved); questions without exposed ranks or not evaluated are excluded"
        ),
    )


def _breakdown(
    evaluations: Sequence[EvaluationResult],
    key: Callable[[EvaluationResult], str],
    metric_ids: Sequence[str],
) -> list[GroupKpi]:
    groups: dict[str, list[EvaluationResult]] = {}
    for ev in evaluations:
        groups.setdefault(key(ev), []).append(ev)
    out = []
    for group, items in sorted(groups.items()):
        judged = [e for e in items if e.verdict is not None]
        passed = sum(1 for e in judged if e.verdict is Verdict.PASS)
        out.append(
            GroupKpi(
                group=group,
                questions=len(items),
                judged=len(judged),
                pass_rate=_ratio(passed, len(judged)),
                metric_means={m: _mean(m, judged) for m in metric_ids},
            )
        )
    return out


def _mean(metric_id: str, judged: Sequence[EvaluationResult]) -> float | None:
    scores = [
        r.score
        for ev in judged
        if (r := ev.metrics.get(metric_id)) is not None
        and r.status is MetricStatus.SCORED
        and r.score is not None
    ]
    return _round(sum(scores) / len(scores)) if scores else None


def _scenario_attr(by_scenario: dict[str, Scenario], ev: EvaluationResult, attr: str) -> str:
    scenario = by_scenario.get(ev.scenario_id)
    return str(getattr(scenario, attr).value) if scenario is not None else "unknown"


def _percentile(values: Sequence[float], pct: int) -> float | None:
    if not values:
        return None
    rank = max(1, math.ceil(pct / 100 * len(values)))
    return _round(values[rank - 1])


def _ratio(part: int, whole: int) -> float | None:
    return _round(part / whole) if whole else None


def _round(value: float) -> float:
    return round(value, 3)
