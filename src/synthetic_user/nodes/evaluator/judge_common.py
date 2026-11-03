"""Helpers shared by the General Judge and the Retrieval/Grounding Judge.

Rendering of the response view, exact-quote checks, the "score < 1 needs a finding" retry
feedback, metric rubric blocks (from the frozen metric definitions) and the known-gap match.
Prompt text is in ``judge_common.yaml``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from typing import Any

from synthetic_user.models.evaluation import (
    EvidenceStatus,
    FindingDraft,
    JudgedScore,
    Severity,
)
from synthetic_user.models.execution import AnsweredResponse, DeniedResponse
from synthetic_user.models.metrics import MetricDefinition
from synthetic_user.prompts.contracts import choices
from synthetic_user.prompts.templates import prompts_beside

PROMPTS = prompts_beside(__file__, "judge_common_prompt.yaml")
COPY_HINT = PROMPTS.text("copy_hint")
EvaluableResponse = AnsweredResponse | DeniedResponse


def response_text(response: EvaluableResponse) -> str:
    """Everything a judge may quote from the response: text / message and artifact contents."""
    if isinstance(response, DeniedResponse):
        return response.message
    parts = [response.text or ""]
    for artifact in response.artifacts:
        content = artifact.content
        parts.append(content if isinstance(content, str) else json.dumps(content))
    return "\n".join(p for p in parts if p)


def bounded(text: str, max_chars: int) -> str:
    return (
        text if len(text) <= max_chars else text[:max_chars] + PROMPTS.text("artifact_cut_marker")
    )


def render_response(response: EvaluableResponse, *, artifact_max_chars: int) -> str:
    """The judge's view of the response, every artifact bounded to ``artifact_max_chars``."""
    if isinstance(response, DeniedResponse):
        return json.dumps({"denial_message": response.message}, indent=2)
    artifacts: list[dict[str, Any]] = []
    for artifact in response.artifacts:
        content = (
            artifact.content if isinstance(artifact.content, str) else json.dumps(artifact.content)
        )
        artifacts.append({"kind": artifact.kind, "content": bounded(content, artifact_max_chars)})
    return json.dumps({"text": response.text, "artifacts": artifacts}, indent=2)


def missing_finding_message(metric: str, score: float) -> str:
    """Retry feedback that allows BOTH fixes (D208: naming only "add a finding" made a live
    judge lower a correct score to 0 just to satisfy the check)."""
    return PROMPTS.text(
        "missing_finding", metric=metric, score=f"{score:g}", metric_repr=repr(metric)
    )


def guidance_block(guidance: str | None) -> str:
    """The EVALUATION GUIDANCE section, or "" without guidance (prompt unchanged, D237)."""
    return PROMPTS.text("evaluation_guidance", guidance=guidance) if guidance else ""


def metric_blocks(definitions: Iterable[MetricDefinition]) -> str:
    return "\n".join(
        PROMPTS.text(
            "metric_block",
            metric_id=d.metric_id,
            display_name=d.display_name,
            rubric=d.rubric,
            anchor_1=d.score_anchors["1"],
            anchor_half=d.score_anchors["0.5"],
            anchor_0=d.score_anchors["0"],
            evidence_requirements=d.evidence_requirements,
        )
        for d in definitions
    )


def score_example() -> dict[str, Any]:
    example: dict[str, Any] = PROMPTS.data("score_example")
    return example


def finding_example(metric_ids: Sequence[str]) -> dict[str, Any]:
    example: dict[str, Any] = PROMPTS.data(
        "finding_example",
        metrics=choices(metric_ids),
        severities=choices(Severity),
        evidence_statuses=choices(EvidenceStatus),
    )
    return example


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


# The judge reads artifacts JSON-encoded (newlines shown as \n), so it may copy an escape.
_JSON_ESCAPES = (("\\n", " "), ("\\r", " "), ("\\t", " "), ('\\"', '"'))


def _compact(text: str) -> str:
    for escape, char in _JSON_ESCAPES:
        text = text.replace(escape, char)
    return re.sub(r"\s+", "", text).casefold()


def quoted(fragment: str, source: str) -> bool:
    """The fragment occurs in the source, ignoring case, whitespace and JSON escapes: a judge
    re-spacing SQL (``a,' ',b`` vs ``a, ' ', b``, a line break before a comma) still quotes it;
    every other character must match in order."""
    return _compact(fragment) in _compact(source)


def check_scores_and_findings(
    metrics: dict[str, JudgedScore],
    findings: Sequence[FindingDraft],
    expected: Sequence[str],
    quotable: str,
) -> list[str]:
    """Shared structural checks: exactly the expected metric keys, quotes found in the source,
    finding categories among the judged metrics, and a finding for every score below 1."""
    errors: list[str] = []
    wanted = set(expected)
    if missing := sorted(wanted - set(metrics)):
        errors.append(f"metrics must include {missing}")
    if extra := sorted(set(metrics) - wanted):
        errors.append(f"metrics must not include {extra} (not judged in this call)")
    for name, judged in metrics.items():
        if judged.evidence and not quoted(judged.evidence, quotable):
            errors.append(
                f"{name}: evidence {judged.evidence!r} is not in the source ({COPY_HINT})"
            )
    categories: set[str] = set()
    for index, finding in enumerate(findings, start=1):
        if finding.category not in wanted:
            errors.append(
                f"finding {index}: category {finding.category!r} is not one of {sorted(wanted)}"
            )
        else:
            categories.add(finding.category)
        if finding.evidence and not quoted(finding.evidence, quotable):
            errors.append(
                f"finding {index}: evidence {finding.evidence!r} is not in the source ({COPY_HINT})"
            )
    for name, judged in metrics.items():
        if (
            name in wanted
            and judged.score is not None
            and judged.score < 1
            and name not in categories
        ):
            errors.append(missing_finding_message(name, judged.score))
    return errors


_GAP_STOPWORDS = frozenset(
    {"when", "with", "from", "that", "this", "into", "over", "than", "then", "they", "them"}
    | {"have", "been", "being", "were", "will", "would", "should", "could", "does", "done"}
    | {"link", "data", "rows", "value", "values", "product", "query", "queries", "user"}
)


def matches_known_gap(text: str, known_gaps: Sequence[str]) -> bool:
    """True if at least two keywords of a known gap (or its only keyword) occur in text.

    Reporting aid only (known-gap tracking of findings); never a validation gate.
    """
    words = _keywords(text)
    for gap in known_gaps:
        gap_words = _keywords(gap)
        if gap_words and len(gap_words & words) >= min(2, len(gap_words)):
            return True
    return False


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) >= 4} - _GAP_STOPWORDS
