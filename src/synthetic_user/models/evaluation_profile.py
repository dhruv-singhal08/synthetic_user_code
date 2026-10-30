"""Run-level evaluation profiles, Agent metrics, and selection policy.

Profiles are deliberately separate from ``ProductDescription``. A product describes what it
does; a run records how its answers are judged. NL2SQL is the specialised SQL path; Agent is the
general path for direct-answer, retrieval, and tool-using agents. Optional Agent metrics are
enabled per run and can still be not applicable when the response exposes no required telemetry.
"""

from __future__ import annotations

import re
from enum import StrEnum

from synthetic_user.models.product import ProductDescription


class EvaluationProfile(StrEnum):
    NL2SQL = "nl2sql"
    AGENT = "agent"

    @classmethod
    def _missing_(cls, value: object) -> EvaluationProfile | None:
        # Stored pre-Agent runs remain readable after the former VoC profile was generalised.
        return cls.AGENT if value == "voc" else None


class EvaluationProfileSelection(StrEnum):
    AUTO = "auto"
    NL2SQL = "nl2sql"
    AGENT = "agent"

    @classmethod
    def _missing_(cls, value: object) -> EvaluationProfileSelection | None:
        return cls.AGENT if value == "voc" else None


class AgentMetric(StrEnum):
    RELEVANCE = "relevance"
    COMPLETION = "completion"
    TOOL_SELECTION = "tool_selection"
    FAITHFULNESS = "faithfulness"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"


REQUIRED_AGENT_METRICS = (AgentMetric.RELEVANCE, AgentMetric.COMPLETION)
OPTIONAL_AGENT_METRICS = (
    AgentMetric.TOOL_SELECTION,
    AgentMetric.FAITHFULNESS,
    AgentMetric.CONTEXT_PRECISION,
    AgentMetric.CONTEXT_RECALL,
)
ALL_AGENT_METRICS = (*REQUIRED_AGENT_METRICS, *OPTIONAL_AGENT_METRICS)
# The general Agent path evaluates the observable answer by default. Tool/retrieval metrics are
# opt-in because they require structured telemetry that many agents do not expose.
DEFAULT_AGENT_METRICS = REQUIRED_AGENT_METRICS

# One config-driven source of truth for how the expected-behaviour contract feeds the Agent
# evaluator. The API exposes this mapping so the UI and judge prompt explain the same contract.
AGENT_EXPECTATION_METRIC_MAP: dict[str, tuple[AgentMetric, ...]] = {
    "question": (AgentMetric.RELEVANCE, AgentMetric.CONTEXT_PRECISION),
    "expected_behavior": (AgentMetric.RELEVANCE, AgentMetric.COMPLETION),
    "success_criteria": (AgentMetric.COMPLETION, AgentMetric.CONTEXT_RECALL),
    "expected_tools": (AgentMetric.TOOL_SELECTION,),
    "expected_journey": (
        AgentMetric.COMPLETION,
        AgentMetric.TOOL_SELECTION,
        AgentMetric.CONTEXT_PRECISION,
        AgentMetric.CONTEXT_RECALL,
    ),
    "expected_time_period": (
        AgentMetric.COMPLETION,
        AgentMetric.TOOL_SELECTION,
        AgentMetric.CONTEXT_PRECISION,
        AgentMetric.CONTEXT_RECALL,
    ),
    "expected_data_sources": (
        AgentMetric.TOOL_SELECTION,
        AgentMetric.FAITHFULNESS,
        AgentMetric.CONTEXT_PRECISION,
        AgentMetric.CONTEXT_RECALL,
    ),
    "expected_context": (
        AgentMetric.FAITHFULNESS,
        AgentMetric.CONTEXT_PRECISION,
        AgentMetric.CONTEXT_RECALL,
    ),
    # Supporting definitions from the reference upload. The evaluator uses these to interpret
    # canonical tools/sources; unlike success criteria, they are not outputs the answer must quote.
    "reference_context": (
        AgentMetric.TOOL_SELECTION,
        AgentMetric.FAITHFULNESS,
        AgentMetric.CONTEXT_PRECISION,
        AgentMetric.CONTEXT_RECALL,
    ),
    "prohibited_behaviors": (
        AgentMetric.COMPLETION,
        AgentMetric.TOOL_SELECTION,
        AgentMetric.FAITHFULNESS,
    ),
    # Unknown organization-specific golden-set fields are never dropped. Completion always
    # consumes them; telemetry-aware metrics may additionally use them when relevant.
    "golden_set_criteria": (
        AgentMetric.COMPLETION,
        AgentMetric.TOOL_SELECTION,
        AgentMetric.FAITHFULNESS,
        AgentMetric.CONTEXT_PRECISION,
        AgentMetric.CONTEXT_RECALL,
    ),
}


def normalize_agent_metrics(
    metrics: list[AgentMetric] | tuple[AgentMetric, ...],
) -> list[AgentMetric]:
    """Return stable, unique metrics and enforce the two mandatory Agent checks."""

    selected = set(metrics)
    missing = [metric.value for metric in REQUIRED_AGENT_METRICS if metric not in selected]
    if missing:
        raise ValueError(f"Agent metrics cannot disable required metrics: {missing}")
    return [metric for metric in ALL_AGENT_METRICS if metric in selected]


ENABLED_EVALUATION_PROFILES = (EvaluationProfile.NL2SQL, EvaluationProfile.AGENT)


def resolve_evaluation_profile(
    product: ProductDescription,
    selection: EvaluationProfileSelection,
) -> tuple[EvaluationProfile, str]:
    """Resolve a supported profile and return a user-facing audit reason."""
    if selection is EvaluationProfileSelection.NL2SQL:
        return EvaluationProfile.NL2SQL, "NL2SQL was selected by the user."
    if selection is EvaluationProfileSelection.AGENT:
        return EvaluationProfile.AGENT, "Agent was selected by the user."

    expected = " ".join(product.expected_outputs)
    if re.search(r"\bsql\b|structured query language", expected, flags=re.IGNORECASE):
        return (
            EvaluationProfile.NL2SQL,
            "Auto-selected from the product's expected outputs, which explicitly require SQL.",
        )

    capabilities = " ".join(
        [product.description, *product.supported, *product.user_visible_capabilities]
    )
    if re.search(r"\bsql\b|structured query language", capabilities, flags=re.IGNORECASE):
        return (
            EvaluationProfile.NL2SQL,
            "Auto-selected from the product description and capabilities, which explicitly "
            "mention SQL.",
        )

    return (
        EvaluationProfile.AGENT,
        "Agent was auto-selected because the product does not explicitly require SQL output.",
    )
