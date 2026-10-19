"""Prompt for the evaluation of one ANSWERED product response (all products, both modes).

The judge sees only the evaluation contract: the question, the goal, expected_behavior,
success_criteria, the product description and the full response (text + artifacts).
Grounded questions add the scenario's cheat sheet (Reference Pack, never the whole reference)
and the usage instructions, with correctness / claim rules per reference format (D203). The
judge scores relevancy and (grounded) correctness 0 / 0.5 / 1, rates every success criterion
and labels every factual claim; code computes completeness, groundedness and
hallucination_rate and the verdict from that. The output contract is a compact JSON example
(D204).
"""

from __future__ import annotations

import json
from typing import Any

from synthetic_user.models.evaluation import (
    GROUNDED_CLAIM_STATUSES,
    UNGROUNDED_CLAIM_STATUSES,
    CriterionStatus,
    EvidenceStatus,
    MetricName,
    Severity,
    Verdict,
)
from synthetic_user.models.execution import AnsweredResponse
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import Reference, ReferenceKind, ReferencePack
from synthetic_user.models.scenario import Scenario
from synthetic_user.prompts.contracts import choices, render_contract

SYSTEM = (
    "You evaluate how well a conversational product answered a user's question, for any kind "
    "of product. You judge only what the response, the question, the product description and "
    "the cheat sheet (when given) show. You quote evidence exactly as written in the "
    "response. You reply with a single JSON object and nothing else."
)

# Per artifact (e.g. SQL, a table); longer artifacts are cut with a marker.
ARTIFACT_MAX_CHARS = 8000

_CORRECTNESS = {
    ReferenceKind.DATA_CATALOG: """\
- correctness: is the answer (e.g. its query) right according to the cheat sheet?
  1 = uses the cheat sheet's tables and columns (or columns whose descriptions show the same
  meaning), joins tables only over the listed joins and respects every rule; 0.5 = partly
  right, or relies on a NOT verified join while presenting the result as reliable; 0 = wrong
  (a column with a different meaning, a join that is not listed, a broken rule); null = the
  response does not show it. Extra columns, filters or ordering are never wrong by themselves.""",
    ReferenceKind.DOCUMENT_COLLECTION: """\
- correctness: are the answer's key facts right according to the documents in the cheat
  sheet? 1 = consistent with the summaries and gives what the question needs; 0.5 = partly
  right or misses a key fact the summaries state; 0 = states something a summary contradicts;
  null = the answer states nothing checkable. The product does not cite sources: judge the
  content only.""",
}

_CLAIMS = {
    ReferenceKind.DATA_CATALOG: """\
- claims: every factual statement and every meaningful part of an artifact (a join condition,
  a filter, an aggregation, a grouping), labelled supported (the cheat sheet backs it),
  contradicted (the cheat sheet says otherwise: a join that is not listed, a column used for
  something its description excludes, a broken rule) or not_in_reference (the cheat sheet
  neither backs nor contradicts it).""",
    ReferenceKind.DOCUMENT_COLLECTION: """\
- claims: every factual statement, labelled supported (a summary states it), contradicted (a
  summary states otherwise) or not_in_reference (the summaries do not cover it - they are
  partial, so this is NOT an error).""",
}


def response_text(response: AnsweredResponse) -> str:
    """Everything the judge may quote from: the text and the artifact contents."""
    parts = [response.text or ""]
    for artifact in response.artifacts:
        content = artifact.content
        parts.append(content if isinstance(content, str) else json.dumps(content))
    return "\n".join(p for p in parts if p)


def evaluation_example(grounded: bool) -> dict[str, Any]:
    score = {
        "score": "<0 | 0.5 | 1 | null>",
        "rationale": "<why>",
        "evidence": "<one short fragment copied from the answer, or null>",
    }
    statuses = GROUNDED_CLAIM_STATUSES if grounded else UNGROUNDED_CLAIM_STATUSES
    return {
        "relevancy": score,
        "correctness": score if grounded else None,
        "criteria": [
            {
                "criterion": "<a success criterion, copied exactly>",
                "status": choices(CriterionStatus),
                "rationale": "<why>",
                "evidence": "<one short fragment copied from the answer, or null>",
            }
        ],
        "claims": [
            {
                "text": "<one short fragment copied from the answer>",
                "status": choices(sorted(statuses)),
                "reason": "<why>",
                "reference_quote": "<short passage from the cheat sheet or instructions, or null>"
                if grounded
                else None,
            }
        ],
        "findings": [
            {
                "category": choices(MetricName),
                "description": "<the problem>",
                "expected_behavior": "<what should have happened>",
                "observed_behavior": "<what happened>",
                "evidence": "<one short fragment copied from the answer, or null>",
                "severity": choices(Severity),
                "confidence": "<0-1, or null>",
                "evidence_status": choices(EvidenceStatus),
                "matches_known_gap": "<true | false>",
            }
        ],
        "overall_verdict": choices(Verdict),
    }


def build_response_evaluation_prompt(
    product: ProductDescription,
    scenario: Scenario,
    question: QuestionCase,
    response: AnsweredResponse,
    grounding: tuple[ReferencePack, Reference] | None = None,
) -> tuple[str, str]:
    expectations = question.expectations
    product_json = product.model_dump(
        mode="json",
        include={
            "name",
            "description",
            "data_sources",
            "supported",
            "not_supported",
            "known_limitations",
            "expected_outputs",
        },
    )
    contract = {
        "question": question.utterance,
        "goal": scenario.business_goal,
        "expected_behavior": expectations.expected_behavior,
        "success_criteria": expectations.success_criteria,
    }
    if grounding is not None:
        pack, reference = grounding
        grounded = f"""
Cheat sheet of this scenario (AUTHORITATIVE answer key):
{pack.render()}

Usage instructions of the reference (follow their evaluation guidance):
{reference.instructions_text()}
"""
        correctness = _CORRECTNESS[pack.kind]
        claims = (
            _CLAIMS[pack.kind]
            + """
  For supported / contradicted give reference_quote: one short passage copied character for
  character from the cheat sheet or the usage instructions (or null)."""
        )
    else:
        grounded = ""
        correctness = """\
- correctness: must be null - there is no reference (ungrounded run)."""
        claims = """\
- claims: every factual statement the response makes, labelled unverified (plausible but
  cannot be checked - there is NO ground truth) or fabricated (a specific fact or figure
  stated as certain with no basis in the question or the product description).
  reference_quote is null. Never claim facts are correct."""
    artifacts_rule = (
        "This product's answers are expected to consist of: "
        f"{', '.join(product.expected_outputs)}. Judge the response against what such an "
        "answer delivers: e.g. a product whose answer is a query or other artifact is not "
        "expected to also return the resulting data."
        if product.expected_outputs
        else "The product description does not say what its answers consist of; judge the "
        "response as a direct answer to the user."
    )
    user = f"""## TASK
Evaluate the product's answer below: score relevancy{" and correctness" if grounding else ""},
rate every success criterion, label every factual claim, report findings.

## INPUT
What was asked and what is expected:
{json.dumps(contract, indent=2)}

Product description:
{json.dumps(product_json, indent=2)}
{grounded}
Product answer:
{_response_view(response)}

## SCORING
- relevancy: does the answer address the question that was asked? 1 = fully, 0.5 = partly
  or with a lot of off-topic content, 0 = not at all; null only if you cannot tell.
{correctness}
- criteria: one entry per success criterion (copy each criterion text exactly), status
  met | partly | not_met | unverifiable. A reading the expected_behavior calls acceptable
  must not lower a score.
{claims}
  How to extract claims: claim text is a SHORT fragment copied character for character
  from the answer (a few words up to one sentence) - never reworded, never several pieces
  joined with "...". Framing sentences that state no fact (e.g. "Here is the query for ...")
  are not claims.
- {artifacts_rule}

## FINDINGS
- A finding explains a problem; its category is the metric it concerns.
- Required: a finding of category relevancy when relevancy is 0, correctness when
  correctness is 0, completeness when a criterion is not_met, hallucination_rate for every
  {"contradicted" if grounding else "fabricated"} claim.
- evidence (in scores, criteria and findings) is ONE short contiguous fragment copied
  character for character from the product answer (a few words is enough), or null - never
  reworded, never several pieces joined with "...". evidence_status: supported (the quote
  shows it), weak, or unsupported.
- matches_known_gap: true when the problem is one the product description already lists
  under known_limitations (or not_supported), else false.
- overall_verdict: your own opinion; recorded, not decisive.

## OUTPUT CONTRACT
{render_contract(evaluation_example(grounding is not None))}

## PROHIBITED
- Inventing quotes, facts or ground truth.
- Penalising the answer for things the product description says are out of scope, except
  when the answer pretends to do them.
- Any text outside the JSON object.
"""
    return SYSTEM, user


def _response_view(response: AnsweredResponse) -> str:
    artifacts = []
    for a in response.artifacts:
        content = a.content if isinstance(a.content, str) else json.dumps(a.content)
        if len(content) > ARTIFACT_MAX_CHARS:
            content = content[:ARTIFACT_MAX_CHARS] + "\n[... cut: artifact too long]"
        artifacts.append({"kind": a.kind, "content": content})
    return json.dumps({"text": response.text, "artifacts": artifacts}, indent=2)
