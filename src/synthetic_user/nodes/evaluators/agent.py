"""Evaluate Agent answers with configurable metrics and deterministic applicability."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import LLMOutputValidationError, LLMProviderError
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.models.evaluation import (
    AGENT_REQUIRED_METRICS,
    AgentEvaluationDraft,
    MetricName,
    MetricResult,
    Metrics,
    MetricStatus,
)
from synthetic_user.models.evaluation_profile import AgentMetric, normalize_agent_metrics
from synthetic_user.models.execution import AnsweredResponse, ConversationTurn, ResponseArtifact
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.scenario import Scenario
from synthetic_user.prompts.agent_response_evaluation import build_agent_evaluation_prompt
from synthetic_user.prompts.response_evaluation import response_text

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

_TOOL_MARKERS = frozenset(
    {
        "tool",
        "tools",
        "tool_name",
        "tool_call",
        "tool_calls",
        "function_name",
        "function_call",
        "function_calls",
    }
)
_CONTEXT_MARKERS = frozenset(
    {
        "context",
        "contexts",
        "retrieval",
        "retrieval_result",
        "retrieval_results",
        "retrieved_context",
        "retrieved_contexts",
        "retrieved_chunks",
        "chunks",
        "documents",
        "sources",
        "citations",
        "tool_result",
        "tool_results",
    }
)


@dataclass(frozen=True)
class AgentMetricPlan:
    enabled: tuple[MetricName, ...]
    applicable: tuple[MetricName, ...]
    not_applicable: dict[MetricName, str]


@dataclass
class AgentJudgement:
    draft: AgentEvaluationDraft | None = None
    plan: AgentMetricPlan | None = None
    notes: list[str] = field(default_factory=list)


def plan_agent_metrics(
    enabled: Sequence[AgentMetric],
    response: AnsweredResponse,
    grounding: tuple[ReferencePack, Reference] | None,
    *,
    has_expected_context: bool = False,
) -> AgentMetricPlan:
    normalized = normalize_agent_metrics(list(enabled))
    selected = tuple(MetricName(metric.value) for metric in normalized)
    has_tools = any(
        _artifact_has_marker(artifact, _TOOL_MARKERS) for artifact in response.artifacts
    )
    has_context = any(
        _artifact_has_marker(artifact, _CONTEXT_MARKERS) for artifact in response.artifacts
    )
    not_applicable: dict[MetricName, str] = {}
    if MetricName.TOOL_SELECTION in selected and not has_tools:
        not_applicable[MetricName.TOOL_SELECTION] = (
            "the response exposes no structured tool-call or function-call trace"
        )
    for metric in (
        MetricName.FAITHFULNESS,
        MetricName.CONTEXT_PRECISION,
        MetricName.CONTEXT_RECALL,
    ):
        if metric in selected and not has_context:
            not_applicable[metric] = (
                "the response exposes no retrieved-context, source, citation, or "
                "tool-result artifact"
            )
    if (
        MetricName.CONTEXT_RECALL in selected
        and MetricName.CONTEXT_RECALL not in not_applicable
        and grounding is None
        and not has_expected_context
    ):
        not_applicable[MetricName.CONTEXT_RECALL] = (
            "no expected reference context is available for measuring retrieval coverage"
        )
    applicable = tuple(metric for metric in selected if metric not in not_applicable)
    return AgentMetricPlan(selected, applicable, not_applicable)


def check_agent_evaluation(
    draft: AgentEvaluationDraft,
    criteria: Sequence[str],
    response: str,
    applicable: Sequence[MetricName],
) -> list[str]:
    errors: list[str] = []
    applicable_set = set(applicable)
    for metric in MetricName:
        if metric not in set(AGENT_REQUIRED_METRICS) | {
            MetricName.TOOL_SELECTION,
            MetricName.FAITHFULNESS,
            MetricName.CONTEXT_PRECISION,
            MetricName.CONTEXT_RECALL,
        }:
            continue
        judged = getattr(draft, metric.value)
        if metric in applicable_set and judged is None:
            errors.append(f"{metric.value}: required for this response")
        elif metric not in applicable_set and judged is not None:
            errors.append(f"{metric.value}: must be omitted because it is not applicable")
        elif judged is not None and judged.evidence and not _quoted(judged.evidence, response):
            errors.append(f"{metric.value}: evidence is not an exact quote from the response")

    wanted = {_key(criterion): criterion for criterion in criteria}
    got = [_key(result.criterion) for result in draft.criteria]
    missing = [criterion for key, criterion in wanted.items() if key not in got]
    extra = [result.criterion for result in draft.criteria if _key(result.criterion) not in wanted]
    repeated = sorted({key for key in got if got.count(key) > 1})
    if missing or extra or repeated:
        errors.append(
            f"criteria must list each success criterion once: missing {missing}, "
            f"unknown {extra}, repeated {repeated}"
        )
    for result in draft.criteria:
        if result.evidence and not _quoted(result.evidence, response):
            errors.append(f"criterion {result.criterion!r}: evidence is not in the response")

    categories: set[str] = set()
    allowed_categories = {metric.value for metric in applicable}
    for index, finding in enumerate(draft.findings, start=1):
        if finding.category not in allowed_categories:
            errors.append(
                f"finding {index}: category {finding.category!r} is not one of "
                f"{sorted(allowed_categories)}"
            )
        else:
            categories.add(finding.category)
        if finding.evidence and not _quoted(finding.evidence, response):
            errors.append(f"finding {index}: evidence is not an exact quote from the response")
        if finding.related_tags:
            errors.append(f"finding {index}: related_tags must be empty for the Agent profile")
    for metric in applicable:
        judged = getattr(draft, metric.value)
        if (
            judged is not None
            and judged.score is not None
            and judged.score < 1
            and metric.value not in categories
        ):
            errors.append(f"a finding with category {metric.value!r} is required")
    return errors


def compute_agent_metrics(draft: AgentEvaluationDraft, plan: AgentMetricPlan) -> Metrics:
    values: dict[str, MetricResult] = {}
    for metric in plan.enabled:
        if reason := plan.not_applicable.get(metric):
            values[metric.value] = MetricResult(
                status=MetricStatus.NOT_APPLICABLE,
                rationale=reason,
            )
            continue
        score = getattr(draft, metric.value)
        assert score is not None
        values[metric.value] = MetricResult(
            status=MetricStatus.INCONCLUSIVE if score.score is None else MetricStatus.SCORED,
            score=None if score.score is None else float(score.score),
            rationale=score.rationale,
            evidence=score.evidence,
            what_worked=score.what_worked,
            what_didnt=score.what_didnt,
        )
    return Metrics(**values)


class AgentEvaluator:
    def __init__(
        self,
        llm: LLMClient,
        settings: AppConfig,
        logger: Logger,
        *,
        enabled_metrics: Sequence[AgentMetric] | None = None,
    ) -> None:
        self._llm = llm
        self._settings = settings
        self._log = logger
        self._enabled_metrics = normalize_agent_metrics(
            list(enabled_metrics or settings.evaluation.agent_metrics)
        )

    async def judge(
        self,
        product: ProductDescription,
        scenario: Scenario,
        question: QuestionCase,
        response: AnsweredResponse,
        grounding: tuple[ReferencePack, Reference] | None = None,
        conversation: Sequence[ConversationTurn] = (),
    ) -> AgentJudgement:
        expected = question.expectations
        plan = plan_agent_metrics(
            self._enabled_metrics,
            response,
            grounding,
            has_expected_context=bool(expected.expected_context or expected.expected_data_sources),
        )
        system, user = build_agent_evaluation_prompt(
            product,
            scenario,
            question,
            response,
            plan.applicable,
            grounding,
            conversation,
            artifact_max_chars=self._settings.evaluation.artifact_max_chars,
        )
        judgement = AgentJudgement(plan=plan)
        quotable = response_text(response)
        try:
            judgement.draft = await generate_structured(
                self._llm,
                system,
                user,
                AgentEvaluationDraft,
                max_retries=self._settings.llm.max_validation_retries,
                post_validate=lambda draft: check_agent_evaluation(
                    draft,
                    question.expectations.success_criteria,
                    quotable,
                    plan.applicable,
                ),
            )
        except LLMOutputValidationError as exc:
            self._log.warning("%s: Agent evaluation failed", question.question_id)
            judgement.notes.append(
                f"Agent evaluation failed after {exc.attempts} attempt(s): " + "; ".join(exc.errors)
            )
        except LLMProviderError as exc:
            self._log.warning(
                "%s: Agent evaluation provider call failed: %s", question.question_id, exc
            )
            judgement.notes.append(f"Agent evaluation provider call failed: {exc}")
        return judgement


def _artifact_has_marker(artifact: ResponseArtifact, markers: frozenset[str]) -> bool:
    tokens = {_normal_token(artifact.kind), *(_normal_token(key) for key in artifact.metadata)}
    tokens.update(_structure_keys(artifact.content))
    return bool(tokens & markers)


def _structure_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {_normal_token(str(key)) for key in value} | {
            token for child in value.values() for token in _structure_keys(child)
        }
    if isinstance(value, list):
        return {token for child in value for token in _structure_keys(child)}
    return set()


def _normal_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _quoted(fragment: str, response: str) -> bool:
    return _key(fragment) in _key(response)
