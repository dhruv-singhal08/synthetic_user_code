"""Artifact plugin hook (DISCONNECTED in design v3; kept for later).

A plugin would add product-specific checks for one kind of response artifact (e.g. a query
language, a chart spec) on top of the generic evaluation. Nothing calls plugins today: the
evaluator (nodes/evaluator.py) runs only the generic judge, and the registry is empty. To
connect one later: implement ArtifactPlugin, register it in registry.py, and call the
applicable plugins from the evaluator for grounded answers, combining verdicts with the
"worst of" rule (fail > inconclusive > pass).
"""

from __future__ import annotations

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
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.scenario import Scenario


@dataclass(frozen=True)
class GroundedContext:
    question: QuestionCase
    scenario: Scenario
    pack: ReferencePack
    reference: Reference
    product: ProductDescription
    response: AnsweredResponse


class PluginFinding(StrictModel):
    """A Finding without its id (the evaluator would assign F_ ids in input order)."""

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
    verdict: Verdict
    analysis: dict[str, JsonValue] | None = None
    notes: list[str] = field(default_factory=list)
    check_rationale: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class ArtifactPlugin(Protocol):
    name: str

    def applies(
        self, response: ProductResponse, question: QuestionCase, scenario: Scenario
    ) -> bool: ...

    async def evaluate(self, ctx: GroundedContext) -> PluginResult: ...
