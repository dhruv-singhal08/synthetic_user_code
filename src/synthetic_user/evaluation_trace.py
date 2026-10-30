"""Stable trace tags linking evaluation results to the question and its cheat sheet."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from synthetic_user.models.evaluation import (
    CriterionResult,
    CriterionStatus,
    EvaluationResult,
    Finding,
    FindingDraft,
    MetricName,
)
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import ReferencePack


@dataclass(frozen=True)
class TraceItem:
    tag: str
    kind: str
    label: str
    text: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def criterion_tag(index: int) -> str:
    return f"CRIT-{index:02d}"


def trace_catalog(question: QuestionCase, pack: ReferencePack | None) -> list[TraceItem]:
    """Return the stable, ordered tags visible to the judge and report UI."""

    items = [
        TraceItem(criterion_tag(index), "criterion", "Expected criterion", criterion)
        for index, criterion in enumerate(question.expectations.success_criteria, start=1)
    ]
    if pack is None:
        return items
    for table_index, table in enumerate(pack.tables, start=1):
        table_tag = f"TABLE-{table_index:02d}"
        items.append(TraceItem(table_tag, "table", "Allowed table", table.name))
        for column_index, column in enumerate(table.columns, start=1):
            items.append(
                TraceItem(
                    f"COLUMN-{table_index:02d}.{column_index:02d}",
                    "column",
                    "Allowed column",
                    f"{table.name}.{column.name}",
                )
            )
        for rule_index, table_rule in enumerate(table.rules, start=1):
            items.append(
                TraceItem(
                    f"TABLE-RULE-{table_index:02d}.{rule_index:02d}",
                    "table_rule",
                    "Table rule",
                    table_rule,
                )
            )
    for join_index, join in enumerate(pack.joins, start=1):
        items.append(
            TraceItem(
                f"JOIN-{join_index:02d}",
                "join",
                "Allowed join",
                f"{join.from_} -> {join.to} ({'verified' if join.verified else 'not verified'})",
            )
        )
    for selected_rule in pack.rules:
        items.append(
            TraceItem(
                f"RULE-{selected_rule.id}",
                "rule",
                "Selected rule",
                selected_rule.rule,
            )
        )
    for document_index, document in enumerate(pack.documents, start=1):
        items.append(
            TraceItem(
                f"DOCUMENT-{document_index:02d}",
                "document",
                "Reference document",
                document.name,
            )
        )
    return items


_SCHEMA_PROBLEM = re.compile(
    r"\b(?:schema|table|column|field|identifier)\b.{0,90}"
    r"\b(?:unknown|undefined|undocumented|unlisted|unallowed|not (?:allowed|listed|present))\b|"
    r"\b(?:unknown|undefined|undocumented|unlisted|unallowed)\b.{0,60}"
    r"\b(?:schema|table|column|field|identifier)\b|"
    r"\b(?:wrong table alias|only documented on)\b",
    re.IGNORECASE,
)
_RULE_SIGNAL = re.compile(
    r"\b(?:rule|requirement|violat(?:e|es|ed|ion)|contrary|conflict|sensitive|privacy|"
    r"currency|currencies|availability|default)\b",
    re.IGNORECASE,
)
_JOIN_SIGNAL = re.compile(r"\b(?:join|relationship|foreign key)\b", re.IGNORECASE)
_SCHEMA_CRITERION = re.compile(
    r"\b(?:allowed|documented|listed)\b.{0,60}\b(?:table|column|field|identifier|schema)\b|"
    r"\b(?:table|column|field|identifier|schema)\b.{0,60}\b(?:allowed|documented|listed)\b",
    re.IGNORECASE,
)
_WORDS = re.compile(r"[a-z][a-z0-9_]+")
_STOPWORDS = frozenset(
    {
        "about",
        "allowed",
        "answer",
        "because",
        "column",
        "data",
        "does",
        "expected",
        "field",
        "from",
        "into",
        "only",
        "query",
        "request",
        "requested",
        "response",
        "result",
        "rule",
        "should",
        "table",
        "that",
        "their",
        "this",
        "uses",
        "using",
        "with",
        "without",
    }
)


def link_finding_tags(
    finding: FindingDraft | Finding,
    question: QuestionCase,
    pack: ReferencePack | None,
    criteria_results: list[CriterionResult],
) -> list[str]:
    """Link a finding to trace items using exact names and conservative semantic overlap."""

    text = " ".join(
        filter(
            None,
            (
                finding.description,
                finding.expected_behavior,
                finding.observed_behavior,
                finding.evidence,
            ),
        )
    )
    linked: set[str] = set()
    failed_criteria = [
        result
        for result in criteria_results
        if result.status in {CriterionStatus.PARTLY, CriterionStatus.NOT_MET}
        and _criterion_metric(result.criterion) == finding.category
    ]
    if failed_criteria:
        if len(failed_criteria) == 1:
            linked.add(failed_criteria[0].tag or "")
        else:
            finding_words = _keywords(text)
            scores = [
                (
                    len(finding_words & _keywords(f"{result.criterion} {result.rationale}")),
                    result,
                )
                for result in failed_criteria
            ]
            best = max((score for score, _ in scores), default=0)
            for score, result in scores:
                if score == best and score >= 2:
                    linked.add(result.tag or "")

    catalog = trace_catalog(question, pack)
    if pack is not None:
        for table_index, table in enumerate(pack.tables, start=1):
            table_tag = f"TABLE-{table_index:02d}"
            table_linked = _mentions_identifier(text, table.name)
            for column_index, column in enumerate(table.columns, start=1):
                if _mentions_identifier(text, column.name):
                    linked.add(f"COLUMN-{table_index:02d}.{column_index:02d}")
                    table_linked = True
            if table_linked:
                linked.add(table_tag)

    finding_words = _keywords(text)
    for item in catalog:
        if item.tag.casefold() in text.casefold():
            linked.add(item.tag)
            continue
        if item.kind in {"table_rule", "rule"}:
            overlap = finding_words & _keywords(item.text)
            if _RULE_SIGNAL.search(text) and len(overlap) >= 2:
                linked.add(item.tag)
        elif item.kind == "join" and _JOIN_SIGNAL.search(text):
            overlap = finding_words & _keywords(item.text)
            if len(overlap) >= 2:
                linked.add(item.tag)
        elif item.kind == "document" and _mentions_identifier(text, item.text):
            linked.add(item.tag)

    if finding.category == MetricName.SCHEMA_FAITHFULNESS.value and _SCHEMA_PROBLEM.search(text):
        linked.add("SCHEMA-UNLISTED")

    order = {item.tag: index for index, item in enumerate(catalog)}
    return sorted((tag for tag in linked if tag), key=lambda tag: (order.get(tag, 10_000), tag))


def _criterion_metric(criterion: str) -> str:
    if _JOIN_SIGNAL.search(criterion) or _RULE_SIGNAL.search(criterion):
        return MetricName.RULES_ADHERENCE.value
    if _SCHEMA_CRITERION.search(criterion):
        return MetricName.SCHEMA_FAITHFULNESS.value
    return MetricName.SEMANTIC_CORRECTNESS.value


def merge_finding_tags(
    suggested: list[str],
    inferred: list[str],
    question: QuestionCase,
    pack: ReferencePack | None,
) -> list[str]:
    """Keep judge-supplied trace tags only when they exist in this question's catalog."""

    catalog = trace_catalog(question, pack)
    order = {item.tag: index for index, item in enumerate(catalog)}
    allowed = set(order) | {"SCHEMA-UNLISTED"}
    combined = {tag for tag in [*suggested, *inferred] if tag in allowed}
    return sorted(combined, key=lambda tag: (order.get(tag, 10_000), tag))


def attach_trace_links(
    evaluation: EvaluationResult,
    question: QuestionCase,
    pack: ReferencePack | None,
) -> EvaluationResult:
    """Add deterministic tags to a stored result, including historical run artifacts."""

    criteria = [
        result.model_copy(update={"tag": result.tag or criterion_tag(index)})
        for index, result in enumerate(evaluation.criteria_results, start=1)
    ]
    findings = [
        finding.model_copy(
            update={
                "related_tags": merge_finding_tags(
                    finding.related_tags,
                    link_finding_tags(finding, question, pack, criteria),
                    question,
                    pack,
                )
            }
        )
        for finding in evaluation.findings
    ]
    return evaluation.model_copy(update={"criteria_results": criteria, "findings": findings})


def _mentions_identifier(text: str, name: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9_$]){re.escape(name)}(?![A-Za-z0-9_$])", text, re.I))


def _keywords(text: str) -> set[str]:
    return {
        word
        for word in _WORDS.findall(text.casefold())
        if len(word) >= 4 and word not in _STOPWORDS
    }
