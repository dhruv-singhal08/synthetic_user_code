"""Compute run KPIs from stored artifacts (pure code, no LLM) and render them as CSV.

Definitions (docs/DESIGN_V3_REFERENCE_AND_EVALUATION.md section 11):
- metric mean / pass rate over answers where the metric was scored; not_applicable and
  inconclusive answers are counted separately, never averaged in;
- metric pass rate = share scoring 1;
- final accuracy = pass verdicts / judged answers (every applicable evaluation driver passes);
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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from synthetic_user.models.evaluation import (
    LOWER_IS_BETTER,
    NL2SQL_METRICS,
    EvaluationResult,
    Finding,
    MetricName,
    MetricStatus,
    Severity,
    Verdict,
)
from synthetic_user.models.execution import ExecutionRecord, ProductOutcome
from synthetic_user.models.kpi import (
    MAX_ISSUE_EXAMPLES,
    ConsolidatedIssue,
    GroupKpi,
    IssueSummary,
    MetricKpi,
    RunKpis,
)
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.evaluator.no_sql import NO_SQL_REASON, no_sql_execution, run_expects_sql

_SEVERITY_RANK = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
METRIC_LABELS = {
    MetricName.SEMANTIC_CORRECTNESS: "Semantic Correctness",
    MetricName.RULES_ADHERENCE: "Rules Adherence",
    MetricName.SCHEMA_FAITHFULNESS: "Schema Faithfulness",
    MetricName.RELEVANCE: "Relevance",
    MetricName.COMPLETION: "Completion",
    MetricName.TOOL_SELECTION: "Tool Selection",
    MetricName.FAITHFULNESS: "Faithfulness",
    MetricName.CONTEXT_PRECISION: "Context Precision",
    MetricName.CONTEXT_RECALL: "Context Recall",
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
            r"\bunsupported\b|\bnot supported\b|\bmissing (?:merchant|dimension|metric)\b|"
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
            r"\b(?:sum|count|average|avg|aggregate|aggregation|calculation|formula|ratio|cpm)\b",
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

_FALLBACK_THEMES = {
    MetricName.SEMANTIC_CORRECTNESS.value: _IssueTheme(
        "other_semantic_correctness",
        "Query does not produce the requested result",
        "Implement every applicable expected criterion for the requested result.",
        re.compile(r"$^"),
    ),
    MetricName.RULES_ADHERENCE.value: _IssueTheme(
        "other_rules_adherence",
        "Query violates an applicable rule or join",
        "Follow the applicable table/business rules and documented joins.",
        re.compile(r"$^"),
    ),
    MetricName.SCHEMA_FAITHFULNESS.value: _IssueTheme(
        "other_schema_faithfulness",
        "Query uses undocumented tables or columns",
        "Use only the allowed table and column names.",
        re.compile(r"$^"),
    ),
    MetricName.RELEVANCE.value: _IssueTheme(
        "other_relevance",
        "Answer does not address the request",
        "Respond directly to the user's request.",
        re.compile(r"$^"),
    ),
    MetricName.COMPLETION.value: _IssueTheme(
        "other_completion",
        "Requested task is incomplete",
        "Complete every requested output and action.",
        re.compile(r"$^"),
    ),
    MetricName.TOOL_SELECTION.value: _IssueTheme(
        "other_tool_selection",
        "Tool or data-source selection does not match the task",
        "Align tool, source, metric, grouping, filters and time window with the request.",
        re.compile(r"$^"),
    ),
    MetricName.FAITHFULNESS.value: _IssueTheme(
        "other_faithfulness",
        "Answer claims are not supported by exposed context",
        "Ground every material claim in the retrieved context or tool results.",
        re.compile(r"$^"),
    ),
    MetricName.CONTEXT_PRECISION.value: _IssueTheme(
        "other_context_precision",
        "Retrieved context is not sufficiently relevant",
        "Retrieve focused context that is relevant to the user's request.",
        re.compile(r"$^"),
    ),
    MetricName.CONTEXT_RECALL.value: _IssueTheme(
        "other_context_recall",
        "Retrieved context misses required information",
        "Retrieve all context necessary to complete the task.",
        re.compile(r"$^"),
    ),
}


def consolidate_findings(
    evaluations: Sequence[EvaluationResult],
    executions: Sequence[ExecutionRecord] = (),
) -> IssueSummary:
    """Group findings into stable, product-actionable themes without another LLM call."""

    sql_expected = run_expects_sql(executions, evaluations)
    no_sql_questions = {
        execution.question_id
        for execution in executions
        if no_sql_execution(execution, sql_expected=sql_expected)
    }
    no_sql_findings = [
        finding
        for evaluation in evaluations
        if evaluation.question_id in no_sql_questions
        for finding in evaluation.findings
    ]

    rows = sorted(
        (
            (evaluation.question_id, finding)
            for evaluation in evaluations
            if evaluation.question_id not in no_sql_questions
            for finding in evaluation.findings
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
        theme = _issue_theme(finding)
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
            categories.add(MetricName.SEMANTIC_CORRECTNESS.value)
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


def _issue_theme(finding: Finding) -> _IssueTheme:
    text = " ".join(
        (
            finding.description,
            finding.expected_behavior,
            finding.observed_behavior,
        )
    )
    return next(
        (theme for theme in _ISSUE_THEMES if theme.pattern.search(text)),
        _FALLBACK_THEMES[finding.category],
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
    for ev in evaluations:
        by_id = {f.finding_id: f for f in ev.findings}
        for bug in ev.bug_candidates:
            findings = [by_id[i] for i in bug.finding_ids if i in by_id]
            if findings and all(f.matches_known_gap for f in findings):
                known += 1
                continue
            worst = max(findings, key=lambda f: _SEVERITY_RANK[f.severity])
            new_bugs[worst.severity.value] += 1

    by_scenario = {s.id: s for s in scenarios}
    active_metrics = (
        tuple(
            dict.fromkeys(
                metric
                for evaluation in evaluations
                for metric in (
                    evaluation.metrics.by_name() if evaluation.metrics is not None else ()
                )
            )
        )
        or NL2SQL_METRICS
    )
    return RunKpis(
        run_id=run_id,
        product_id=product_id,
        mode=mode,
        computed_at=now or datetime.now(UTC),
        questions_asked=asked,
        outcomes={o.value: outcomes.get(o.value, 0) for o in ProductOutcome},
        judged=len(judged),
        verdicts={v.value: verdicts.get(v.value, 0) for v in Verdict},
        accuracy=_ratio(verdicts[Verdict.PASS.value], len(judged)),
        task_success_rate=_ratio(verdicts[Verdict.PASS.value], len(judged)),
        metrics=[_metric_kpi(name, judged) for name in active_metrics],
        availability=_round(1 - failed_calls / asked) if asked else None,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        new_bugs=sum(new_bugs.values()),
        new_bugs_by_severity={s.value: new_bugs.get(s.value, 0) for s in Severity},
        known_gap_bugs=known,
        inconclusive_rate=_ratio(verdicts[Verdict.INCONCLUSIVE.value], len(judged)),
        by_task_type=_breakdown(
            evaluations, lambda ev: _scenario_attr(by_scenario, ev, "task_type")
        ),
        by_capability=_breakdown(
            evaluations, lambda ev: _scenario_attr(by_scenario, ev, "capability")
        ),
    )


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
        note="answers passing every applicable evaluation check / evaluated answers",
    )
    for m in kpis.metrics:
        excluded = f"{m.not_applicable} not applicable, {m.inconclusive} inconclusive"
        direction = "lower is better" if m.lower_is_better else "higher is better"
        row(f"{METRIC_LABELS[m.metric]} (mean)", m.mean, m.n, excluded, direction)
        best = "share scoring 0" if m.lower_is_better else "share scoring 1"
        row(f"{METRIC_LABELS[m.metric]} (pass rate)", m.pass_rate, m.n, excluded, best)
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
        f"{k} {v}" for k, v in kpis.outcomes.items() if k != ProductOutcome.ANSWERED.value and v
    )
    row("Questions asked", kpis.questions_asked, note=f"recorded only: {not_judged or 'none'}")
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    return buffer.getvalue()


def _metric_kpi(name: MetricName, judged: Sequence[EvaluationResult]) -> MetricKpi:
    results = [
        result
        for ev in judged
        if ev.metrics is not None
        if (result := ev.metrics.by_name().get(name)) is not None
    ]
    scores = [r.score for r in results if r.status is MetricStatus.SCORED and r.score is not None]
    lower = name in LOWER_IS_BETTER
    best = 0.0 if lower else 1.0
    return MetricKpi(
        metric=name,
        mean=_round(sum(scores) / len(scores)) if scores else None,
        pass_rate=_ratio(sum(1 for s in scores if s == best), len(scores)),
        n=len(scores),
        not_applicable=sum(1 for r in results if r.status is MetricStatus.NOT_APPLICABLE),
        inconclusive=sum(1 for r in results if r.status is MetricStatus.INCONCLUSIVE),
        lower_is_better=lower,
    )


def _breakdown(
    evaluations: Sequence[EvaluationResult], key: Callable[[EvaluationResult], str]
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
                metric_means={
                    m: _metric_kpi(m, judged).mean
                    for m in dict.fromkeys(
                        metric
                        for evaluation in judged
                        for metric in (
                            evaluation.metrics.by_name() if evaluation.metrics is not None else ()
                        )
                    )
                },
            )
        )
    return out


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
