"""Stable trace tags linking evaluation results to the question and its cheat sheet.

Reporting aid only (never a validation gate). Success criteria are tagged with their own ids
(C1, C2, ...) and Reference Spec items with the ids of ``reference_requirements`` (one scheme for
the judge, the checks and the report). A finding is linked to the criteria whose check FAILED in
the finding's metric; with several failed criteria, keyword overlap picks among them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from synthetic_user.models.evaluation import (
    EvaluationResult,
    Finding,
    FindingDraft,
    RequirementStatus,
)
from synthetic_user.models.question import (
    QuestionCase,
    ReferenceRequirementKind,
    reference_requirements,
)
from synthetic_user.models.reference import ReferencePack


@dataclass(frozen=True)
class TraceItem:
    tag: str
    kind: str
    label: str
    text: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_LABELS = {
    ReferenceRequirementKind.TABLE: ("table", "Allowed table"),
    ReferenceRequirementKind.COLUMN: ("column", "Allowed column"),
    ReferenceRequirementKind.TABLE_RULE: ("table_rule", "Table rule"),
    ReferenceRequirementKind.JOIN: ("join", "Allowed join"),
    ReferenceRequirementKind.RULE: ("rule", "Selected rule"),
    ReferenceRequirementKind.DOCUMENT: ("document", "Reference document"),
}


def trace_catalog(question: QuestionCase, pack: ReferencePack | None) -> list[TraceItem]:
    """Return the stable, ordered tags visible to the judge and report UI."""

    items = [
        TraceItem(c.id, "criterion", "Success criterion", c.requirement)
        for c in (
            question.evaluation_context.success_criteria if question.evaluation_context else []
        )
    ]
    if pack is None:
        return items
    for requirement in reference_requirements(pack):
        if requirement.kind in _LABELS:
            kind, label = _LABELS[requirement.kind]
            items.append(TraceItem(requirement.id, kind, label, requirement.text))
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
# The built-in metric whose findings may carry the SCHEMA-UNLISTED report tag.
SCHEMA_METRIC = "schema_faithfulness"
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
    failed_criteria: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    """``failed_criteria`` = metric id -> the criterion ids whose check failed in it."""
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
    context = question.evaluation_context
    failed = [
        c
        for c in (failed_criteria or {}).get(finding.category, [])
        if context is not None and context.criterion(c) is not None
    ]
    if failed and context is not None:
        if len(failed) == 1:
            linked.add(failed[0])
        else:
            finding_words = _keywords(text)
            scores = [
                (len(finding_words & _keywords(context.requirement_text(c) or "")), c)
                for c in failed
            ]
            best = max((score for score, _ in scores), default=0)
            for score, criterion_id in scores:
                if score == best and score >= 2:
                    linked.add(criterion_id)

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

    if finding.category == SCHEMA_METRIC and _SCHEMA_PROBLEM.search(text):
        linked.add("SCHEMA-UNLISTED")

    order = {item.tag: index for index, item in enumerate(catalog)}
    return sorted((tag for tag in linked if tag), key=lambda tag: (order.get(tag, 10_000), tag))


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

    failed = failed_criteria_by_metric(evaluation)
    findings = [
        finding.model_copy(
            update={
                "related_tags": merge_finding_tags(
                    finding.related_tags,
                    link_finding_tags(finding, question, pack, failed),
                    question,
                    pack,
                )
            }
        )
        for finding in evaluation.findings
    ]
    return evaluation.model_copy(update={"findings": findings})


def failed_criteria_by_metric(evaluation: EvaluationResult) -> dict[str, list[str]]:
    return {
        metric_id: [
            c.requirement_id
            for c in result.checks
            if c.status is RequirementStatus.FAILED and c.requirement_id.startswith("C")
        ]
        for metric_id, result in evaluation.metrics.items()
    }


def _mentions_identifier(text: str, name: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9_$]){re.escape(name)}(?![A-Za-z0-9_$])", text, re.I))


def _keywords(text: str) -> set[str]:
    return {
        word
        for word in _WORDS.findall(text.casefold())
        if len(word) >= 4 and word not in _STOPWORDS
    }
