"""Artifact plugins: grounded checks of product-specific response artifacts (e.g. SQL).

Plugins run ONLY in grounded mode (a cheat sheet + reference snapshot exist) and only for
answered responses they apply to; ungrounded evaluations never run them. Plugins are listed
in config ``evaluation.plugins`` (registry.py). A plugin returns PluginResult without ids;
nodes/evaluator.py assigns finding ids and combines verdicts (fail > inconclusive > pass).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.evaluation import (
    CheckStatus,
    EvidenceStatus,
    FindingSource,
    Severity,
    Verdict,
)
from synthetic_user.models.execution import AnsweredResponse, ProductResponse
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import ReferenceIndex, ReferenceSnapshot
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario


@dataclass(frozen=True)
class GroundedContext:
    question: QuestionCase
    scenario: Scenario
    snapshot: ReferenceSnapshot
    index: ReferenceIndex
    product: ProductDescription
    response: AnsweredResponse


class PluginFinding(StrictModel):
    """A Finding without its id (the evaluator assigns F_ ids in input order)."""

    category: NonEmptyStr
    description: NonEmptyStr
    expected_behavior: NonEmptyStr
    observed_behavior: NonEmptyStr
    evidence: str | None = None
    severity: Severity
    confidence: float | None = None
    evidence_status: EvidenceStatus
    source: FindingSource
    matches_known_gap: bool = False


@dataclass
class PluginResult:
    checks: dict[str, CheckStatus]
    findings: list[PluginFinding]
    analysis: dict[str, JsonValue] | None
    verdict: Verdict
    notes: list[str] = field(default_factory=list)
    cheat_sheet_conflicts: list[str] = field(default_factory=list)
    llm_verdict: Verdict | None = None
    check_rationale: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class ArtifactPlugin(Protocol):
    name: str

    def applies(
        self, response: ProductResponse, question: QuestionCase, scenario: Scenario
    ) -> bool: ...

    async def evaluate(self, ctx: GroundedContext) -> PluginResult: ...


def is_grounded_answer(
    response: ProductResponse, question: QuestionCase, scenario: Scenario
) -> bool:
    return (
        isinstance(response, AnsweredResponse)
        and question.mode is GenerationMode.GROUNDED
        and scenario.reference_cheat_sheet is not None
    )


def checks_verdict(checks: Mapping[str, CheckStatus], findings: Sequence[PluginFinding]) -> Verdict:
    """A failed check backed by a supported finding of that category -> fail; else any failed
    (unsupported) or inconclusive check -> inconclusive; else pass."""
    supported = {f.category for f in findings if f.evidence_status is EvidenceStatus.SUPPORTED}
    failed = [name for name, status in checks.items() if status is CheckStatus.FAIL]
    if any(name in supported for name in failed):
        return Verdict.FAIL
    if failed or CheckStatus.INCONCLUSIVE in checks.values():
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
