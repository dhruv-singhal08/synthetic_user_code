"""Assemble the Retrieval/Grounding Judge prompt; the text itself is in ``retrieval_prompt.yaml``.

The judge sees the question, its expectations, the answer, the normalized ranked chunks, the
exposed citations (with their resolution) and tool results - never the whole reference.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from synthetic_user.models.evaluation import CoverageStatus, SupportStatus
from synthetic_user.models.execution import RetrievedContext, ToolCall
from synthetic_user.models.metrics import EvidenceKind, MetricDefinition
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.nodes.evaluator.judge_common import (
    EvaluableResponse,
    bounded,
    finding_example,
    guidance_block,
    metric_blocks,
    render_response,
    score_example,
    severity_rule,
)
from synthetic_user.prompts.contracts import choices, render_contract
from synthetic_user.prompts.templates import prompts_beside

PROMPTS = prompts_beside(__file__, "retrieval_prompt.yaml")
SYSTEM = PROMPTS.text("system")
PRODUCT_FIELDS = {"name", "description", "expected_outputs"}
# The checklist field of an expectation that the judge rates item by item (Context Recall).
REQUIRED_INFORMATION_FIELD = "required_information"


@dataclass(frozen=True)
class RetrievalJudgeInput:
    product: ProductDescription
    question: QuestionCase
    response: EvaluableResponse
    definitions: Sequence[MetricDefinition]  # judged metrics (scores)
    needs_labels: bool  # per-chunk relevance labels for code-computed metrics (MRR)
    context: RetrievedContext
    tool_trace: Sequence[ToolCall] | None
    artifact_max_chars: int
    # Optional run-level evaluation guidance (D237); None -> prompt unchanged.
    guidance: str | None = None

    def checklist(self) -> list[str]:
        """The required-information items of the judged metrics (empty when none)."""
        items: list[str] = []
        for definition in self.definitions:
            value = self.generated(definition.metric_id)
            raw = value.get(REQUIRED_INFORMATION_FIELD) if value else None
            if isinstance(raw, list):
                items.extend(str(item) for item in raw if str(item).strip())
        return list(dict.fromkeys(items))

    def generated(self, metric_id: str) -> dict[str, Any] | None:
        return self.question.expectations.success_criteria.generated(metric_id)

    @property
    def checks_citations(self) -> bool:
        return bool(self.context.citations) and any(
            EvidenceKind.CITATIONS in d.required_execution_evidence for d in self.definitions
        )


def citation_label(index: int, citation_id: str | None) -> str:
    return citation_id or f"#{index}"


def build_retrieval_prompt(data: RetrievalJudgeInput) -> tuple[str, str]:
    definitions = list(data.definitions)
    ids = [d.metric_id for d in definitions]
    task_parts: list[str] = []
    if ids:
        task_parts.append(PROMPTS.text("task_metrics", metric_ids=", ".join(ids)))
    if data.needs_labels:
        task_parts.append(PROMPTS.text("task_labels"))
    rules: list[str] = []
    if data.needs_labels:
        rules.append(PROMPTS.text("rule_labels"))
    if data.checklist():
        rules.append(PROMPTS.text("rule_required_information"))
    if data.checks_citations:
        rules.append(PROMPTS.text("rule_citations"))
    sections: list[str] = []
    expectations = {d.metric_id: v for d in definitions if (v := data.generated(d.metric_id))}
    if expectations:
        sections.append(
            PROMPTS.text(
                "section_metric_expectations", expectations=json.dumps(expectations, indent=2)
            )
        )
    if data.context.citations:
        citations = [
            {
                "citation": citation_label(i, c.citation_id),
                "chunk_id": c.chunk_id,
                "source_id": c.source_id,
                "claim": c.claim,
                "resolved": c.resolved,
            }
            for i, c in enumerate(data.context.citations, start=1)
        ]
        sections.append(
            PROMPTS.text("section_citations", citations=json.dumps(citations, indent=2))
        )
    results = [
        {"call": f"tool-{c.order}", "name": c.name, "result": c.result, "error": c.error}
        for c in data.tool_trace or ()
        if c.result is not None or c.error
    ]
    if results:
        sections.append(
            PROMPTS.text(
                "section_tool_results",
                tools=bounded(json.dumps(results, indent=2), data.artifact_max_chars),
            )
        )
    chunks = [
        {
            "chunk_id": chunk.chunk_id,
            "rank": chunk.rank,
            "source_id": chunk.source_id,
            "text": bounded(chunk.text, data.artifact_max_chars),
        }
        for chunk in data.context.chunks
    ]
    question = data.question
    expected = {
        "question": question.utterance,
        "expected_behavior": question.expectations.expected_behavior,
        "success_criteria": question.expectations.success_criteria.general,
    }
    user = PROMPTS.text(
        "user",
        task=PROMPTS.text("task_join").join(task_parts),
        product=json.dumps(data.product.model_dump(mode="json", include=PRODUCT_FIELDS), indent=2),
        expected=json.dumps(expected, indent=2),
        sections="".join(sections),
        response=render_response(data.response, artifact_max_chars=data.artifact_max_chars),
        chunk_note=PROMPTS.text(
            "chunk_note_ranked" if data.context.rank_exposed else "chunk_note_unranked"
        ),
        chunks=json.dumps(chunks, indent=2),
        metric_rules=metric_blocks(definitions),
        guidance=guidance_block(data.guidance),
        severity_rule=severity_rule(),
        rules="".join(f"{rule}\n" for rule in rules),
        contract=render_contract(retrieval_example(data)),
    )
    return SYSTEM, user


def retrieval_example(data: RetrievalJudgeInput) -> dict[str, Any]:
    ids = [d.metric_id for d in data.definitions]
    example: dict[str, Any] = {"metrics": {metric_id: score_example() for metric_id in ids}}
    if data.needs_labels:
        example["chunk_relevance"] = [PROMPTS.data("chunk_relevance_example")]
    if data.checklist():
        example["required_information"] = [
            PROMPTS.data("required_information_example", coverage=choices(CoverageStatus))
        ]
    if data.checks_citations:
        example["citation_checks"] = [
            PROMPTS.data("citation_check_example", support=choices(SupportStatus))
        ]
    example["findings"] = [finding_example(ids, guidance=bool(data.guidance))] if ids else []
    return example
