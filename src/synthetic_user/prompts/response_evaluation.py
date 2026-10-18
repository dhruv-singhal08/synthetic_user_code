"""Prompt for the evaluation of one ANSWERED product response (all products, both modes).

The judge sees only the evaluation contract (design v3): the question, the goal,
expected_behavior, success_criteria, acceptable_variations, the product description and the
full response (text + artifacts). Grounded questions add the scenario's Reference Pack
(authoritative) and the usage guide. The judge scores relevancy and (grounded) correctness
0 / 0.5 / 1, rates every success criterion and labels every factual claim; code computes
completeness, groundedness and hallucination_rate and the verdict from that.
"""

from __future__ import annotations

import json

from synthetic_user.models.evaluation import EvaluationDraft
from synthetic_user.models.execution import AnsweredResponse
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.reference import Reference, ReferencePack
from synthetic_user.models.scenario import Scenario

SYSTEM = (
    "You evaluate how well a conversational product answered a user's question, for any kind "
    "of product. You judge only what the response, the question, the product description and "
    "the reference pack (when given) show. You quote evidence exactly as written in the "
    "response. You reply with a single JSON object and nothing else."
)

# Per artifact (e.g. SQL, a table); longer artifacts are cut with a marker.
ARTIFACT_MAX_CHARS = 8000


def response_text(response: AnsweredResponse) -> str:
    """Everything the judge may quote from: the text and the artifact contents."""
    parts = [response.text or ""]
    for artifact in response.artifacts:
        content = artifact.content
        parts.append(content if isinstance(content, str) else json.dumps(content))
    return "\n".join(p for p in parts if p)


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
            "product_type",
            "supported",
            "not_supported_yet",
            "known_gaps",
            "response_artifacts_expected",
        },
    )
    contract = {
        "question": question.utterance,
        "goal": scenario.business_goal,
        "expected_behavior": expectations.expected_behavior,
        "success_criteria": expectations.success_criteria,
        "acceptable_variations": expectations.acceptable_variations,
    }
    if grounding is not None:
        pack, reference = grounding
        grounded = f"""
Reference pack (AUTHORITATIVE answer key for this scenario):
{json.dumps(pack.view(), indent=2)}

Usage guide of the reference (follow its evaluation guidance):
{reference.guide_text()}
"""
        correctness = """\
- correctness: are the key facts / the query right for this question according to the
  reference pack? 1 = right (relies on the core_refs, or an equivalent the reference
  supports, and respects the rules); 0.5 = partly right; 0 = wrong (relies on something with
  a different meaning, or breaks a rule); null = the response does not show it. Using MORE
  than the pack (extra items, fields, filters, details) is never wrong by itself."""
        claims = """\
- claims: every factual statement the response makes, labelled supported (the reference
  pack / reference backs it), contradicted (the reference says otherwise) or
  not_in_reference (the reference neither backs nor contradicts it). For supported /
  contradicted give reference_quote: one of the pack's quote values copied as it is (or
  null)."""
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
        f"{', '.join(product.response_artifacts_expected)}. Judge the response against what "
        "such an answer delivers: e.g. a product whose answer is a query or other artifact is "
        "not expected to also return the resulting data."
        if product.response_artifacts_expected
        else "The product description does not say what its answers consist of; judge the "
        "response as a direct answer to the user."
    )
    schema = json.dumps(EvaluationDraft.model_json_schema(), indent=2)
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
  met | partly | not_met | unverifiable.
{claims}
  How to extract claims: claim text is a SHORT fragment copied character for character
  from the answer (a few words up to one sentence) - never reworded, never several pieces
  joined with "...". For an artifact such
  as a query, each meaningful part is a claim (e.g. a join condition, a filter, an
  aggregation, a grouping), copied from the artifact. Framing sentences that state no fact
  (e.g. "Here is the SQL query for ...") are not claims.
- acceptable_variations are valid: never lower a score for one of them.
- {artifacts_rule}

## FINDINGS
- A finding explains a problem; its category is the metric it concerns: relevancy |
  correctness | groundedness | hallucination_rate | completeness.
- Required: a finding of category relevancy when relevancy is 0, correctness when
  correctness is 0, completeness when a criterion is not_met, hallucination_rate for every
  {"contradicted" if grounding else "fabricated"} claim.
- evidence (in scores, criteria and findings) is ONE short contiguous fragment copied
  character for character from the product answer (a few words is enough), or null - never
  reworded, never several pieces joined with "...".
  evidence_status: supported (the quote shows it), weak, or unsupported.
- overall_verdict: your own opinion (pass | fail | inconclusive); recorded, not decisive.

## OUTPUT CONTRACT
Return one JSON object matching this JSON Schema:
{schema}

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
