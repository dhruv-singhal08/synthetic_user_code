"""Shared test data importable from any test module (``tests/`` is on sys.path)."""

from __future__ import annotations

import json
from typing import Any

# Authoritative tool definitions (reference agent_context) for tool-dependent metrics.
AGENT_CONTEXT: dict[str, Any] = {
    "data_sources": [
        {"name": "complaints", "description": "Complaint records", "aliases": ["Complaint data"]},
        {"name": "csat", "description": "Survey results", "aliases": ["Survey data"]},
    ],
    "source_groups": [{"name": "both", "sources": ["complaints", "csat"]}],
    "tools": [
        {
            "name": "aggregate_feedback",
            "description": "Counts and ranks feedback",
            "aliases": ["aggregate"],
            "data_sources": ["complaints", "csat"],
            "constraints": ["Never blend sources"],
        },
        {
            "name": "search_complaints",
            "description": "Finds complaint verbatims",
            "data_sources": ["complaints"],
        },
    ],
    "workflow": [{"name": "planning", "description": "Selects tools"}],
    "rules": ["Keep complaint and CSAT results separate."],
}
TOOL_CATALOG = (
    "kind: data_catalog\nname: VoC\ntables:\n  - name: complaints_t\n"
    "    rules: [Count each complaint once]\n    columns: [{name: theme}]\n"
)


def scripted_retrieval_reply(user: str, relevant: set[str] | None = None) -> dict[str, Any]:
    """Retrieval Judge reply: metrics 1, chunks relevant (or only ``relevant``), all covered."""
    chunks = json.loads(user.split("Retrieved chunks (", 1)[1].split("):\n", 1)[1].split("\n\n")[0])
    task = user.split("## TASK\n", 1)[1].split("\n\n", 1)[0]
    metrics: list[str] = []
    if "the metric(s) " in task:
        metrics = [
            m.strip()
            for m in task.split("the metric(s) ", 1)[1]
            .split(" and per-chunk", 1)[0]
            .rstrip(".")
            .split(",")
        ]
    reply: dict[str, Any] = {
        "metrics": {m: {"score": 1, "rationale": "fine"} for m in metrics},
        "findings": [],
    }
    if "per-chunk relevance labels" in task:
        reply["chunk_relevance"] = [
            {
                "chunk_id": c["chunk_id"],
                "relevant": relevant is None or c["chunk_id"] in relevant,
                "rationale": "label",
            }
            for c in chunks
        ]
    if "required_information:" in user and "Question-specific expectations" in user:
        expectations = json.loads(
            user.split("Question-specific expectations of the listed metrics:\n", 1)[1].split(
                "\n\n", 1
            )[0]
        )
        items = [i for v in expectations.values() for i in v.get("required_information", [])]
        reply["required_information"] = [
            {"item": i, "status": "covered", "chunk_ids": [], "rationale": "covered"} for i in items
        ]
    if "citation_checks:" in user and "Citations exposed with the answer" in user:
        citations = json.loads(
            user.split("Citations exposed with the answer", 1)[1]
            .split(":\n", 1)[1]
            .split("\n\n", 1)[0]
        )
        reply["citation_checks"] = [
            {
                "citation": c["citation"],
                "chunk_id": c["chunk_id"],
                "support": "supported",
                "rationale": "ok",
            }
            for c in citations
        ]
    return reply


def goal_for(user: str, classification: str | None = None) -> dict[str, Any]:
    """A consistent goal_completion for a scripted General Judge reply (D243): a decline is a
    justified decline, anything else a completed goal."""
    declined = (
        "Product response (typed denial" in user
        or "typed by the product as unable to answer" in user
        or classification == "unable_to_answer"
    )
    if declined:
        return {"status": "justified_decline", "reason": "the contract justifies it"}
    return {"status": "completed", "reason": "the goal is met"}
