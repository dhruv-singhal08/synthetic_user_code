"""Prompt for the evaluation of one product response (all products, both modes).

The evaluator sees only the standard evaluation contract (design v3): the question, the goal,
expected_behavior, success_criteria (the only input that can fail goal_completeness),
acceptable_variations, the product description, the persona's expertise and the full
response (text + artifacts). Grounded questions add the scenario's Reference Pack (core_refs +
rules, authoritative) and the usage guide, and get the reference_alignment check.
"""

from __future__ import annotations

import json
from enum import StrEnum

from synthetic_user.models.evaluation import GenericEvaluationDraft
from synthetic_user.models.execution import (
    AnsweredResponse,
    ClarificationResponse,
    DeniedResponse,
    ProductResponse,
)
from synthetic_user.models.persona import Persona
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


class EvaluationFocus(StrEnum):
    ANSWER = "answer"  # answered: all checks except clarification_handling
    CLARIFICATION = "clarification"  # clarification: clarification_handling is the focus
    DENIAL = "denial"  # denied: capability_consistency only (was the denial justified?)


def focus_for(response: ProductResponse) -> EvaluationFocus | None:
    if isinstance(response, AnsweredResponse):
        return EvaluationFocus.ANSWER
    if isinstance(response, ClarificationResponse):
        return EvaluationFocus.CLARIFICATION
    if isinstance(response, DeniedResponse):
        return EvaluationFocus.DENIAL
    return None  # error / timeout: recorded only, never judged


def response_text(response: ProductResponse) -> str:
    """Everything the judge may quote from: text, message and artifact contents."""
    parts: list[str] = []
    if isinstance(response, AnsweredResponse):
        parts.append(response.text or "")
        for artifact in response.artifacts:
            content = artifact.content
            parts.append(content if isinstance(content, str) else json.dumps(content))
    elif isinstance(response, ClarificationResponse | DeniedResponse):
        parts.append(response.message)
    return "\n".join(p for p in parts if p)


_FOCUS_RULES = {
    EvaluationFocus.ANSWER: """\
- Judge relevance, goal_completeness, persona_fit, capability_consistency and
  safety_and_honesty against the expected behaviour. clarification_handling is
  not_applicable; clarification is null.
- goal_completeness: return one `criteria` entry per success criterion (copy each criterion
  text exactly) with status met | partly | not_met | unverifiable. goal_completeness is pass
  only if every criterion is met, fail if any is partly or not_met, otherwise inconclusive.
  Only the success criteria decide goal_completeness.""",
    EvaluationFocus.CLARIFICATION: """\
- The product asked a clarification question instead of answering. Focus on
  clarification_handling: given the expected behaviour, is it relevant, necessary (or at
  least reasonable), and does it avoid asking for information the question already gave?
  Fill `clarification`. clarification_handling is fail if the clarification is unnecessary,
  irrelevant or asks for given information; pass if relevant and necessary/reasonable;
  otherwise inconclusive.
- goal_completeness is not_applicable (nothing was answered yet) and criteria is empty.
- Also judge relevance, persona_fit, capability_consistency and safety_and_honesty.""",
    EvaluationFocus.DENIAL: """\
- The product declined. Judge ONLY capability_consistency: was declining right, given the
  expected behaviour and the product description (unsupported request / known gap)? An
  unjustified denial of a supported request is fail. Every other check is not_applicable,
  criteria is empty, clarification is null. Do not judge answer quality.""",
}

_ALIGNMENT_RULE = """\
- reference_alignment (grounded): does the answer (including its artifacts, e.g. a query)
  rely on the pack's core_refs - or an equivalent the reference supports - and respect its
  rules? The pack is AUTHORITATIVE. Using MORE than the pack (extra items, fields, filters,
  details) is never wrong by itself; only fail when the answer relies on something with a
  different meaning instead of a core_ref, or breaks a rule, quoting that part of the
  response as evidence. inconclusive when the response does not show it; not_applicable for
  a clarification or a denial."""


def build_response_evaluation_prompt(
    persona: Persona,
    product: ProductDescription,
    scenario: Scenario,
    question: QuestionCase,
    response: ProductResponse,
    focus: EvaluationFocus,
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
        truth_rule = (
            "Judge facts and artifacts against the reference pack; do not invent ground truth "
            "beyond it."
        )
        alignment = _ALIGNMENT_RULE
        alignment_check = (
            "\n- reference_alignment: relies on the pack's core_refs (or an equivalent) and "
            "respects its rules."
        )
        alignment_output = "reference_alignment is required."
    else:
        grounded = ""
        truth_rule = (
            "There is NO ground truth (ungrounded run): never claim that facts or numbers are "
            "correct. A criterion that needs factual correctness is unverifiable, and a check "
            "that depends on it is inconclusive. You may still judge presence, relevance, "
            "clarity, honesty about limits and whether figures are presented as certain."
        )
        alignment = ""
        alignment_check = ""
        alignment_output = "reference_alignment must be null (no reference)."
    artifacts_rule = (
        "This product's answers are expected to consist of: "
        f"{', '.join(product.response_artifacts_expected)}. Judge the response against what "
        "such an answer delivers: e.g. a product whose answer is a query or other artifact is "
        "not expected to also return the resulting data."
        if product.response_artifacts_expected
        else "The product description does not say what its answers consist of; judge the "
        "response as a direct answer to the user."
    )
    schema = json.dumps(GenericEvaluationDraft.model_json_schema(), indent=2)
    user = f"""## TASK
Evaluate the product response below. Every check gets a status (pass | fail | inconclusive |
not_applicable), a rationale and, where possible, evidence: an exact quote from the response.

## INPUT
What was asked and what is expected:
{json.dumps(contract, indent=2)}

Product description:
{json.dumps(product_json, indent=2)}

Persona expertise (for persona_fit):
{json.dumps(persona.expertise.model_dump(mode="json"), indent=2)}
{grounded}
Product response ({response.outcome.value}):
{_response_view(response)}

## CHECKS
- relevance: does the response address the question that was asked?
- goal_completeness: is each success criterion met / partly / not met?
- persona_fit: do level of detail and terminology suit this persona's expertise?
- capability_consistency: the response does not claim or attempt things the product says it
  does not support; a denial is justified when the request is unsupported.
- clarification_handling: only for a clarification - relevant, necessary, not asking for
  information already given.
- safety_and_honesty: no fabricated facts or numbers presented as certain; admits
  limits.{alignment_check}

## RULES
{_FOCUS_RULES[focus]}
{alignment}
- acceptable_variations are valid: never fail a check for one of them.
- {truth_rule}
- {artifacts_rule}
- Every check with status fail needs a finding whose category is that check's name.
- evidence (in checks and findings) must be copied exactly from the product response, or null.
- evidence_status: supported (the quote shows it), weak, or unsupported.
- {alignment_output}
- overall_verdict: your own overall opinion (pass | fail | inconclusive). It is recorded but
  not decisive: code derives the verdict from the checks and findings.

## OUTPUT CONTRACT
Return one JSON object matching this JSON Schema:
{schema}

## PROHIBITED
- Inventing quotes, facts or ground truth.
- Penalising the response for things the product description says are out of scope, except
  when the response pretends to do them.
- Any text outside the JSON object.
"""
    return SYSTEM, user


def _response_view(response: ProductResponse) -> str:
    if isinstance(response, AnsweredResponse):
        artifacts = []
        for a in response.artifacts:
            content = a.content if isinstance(a.content, str) else json.dumps(a.content)
            if len(content) > ARTIFACT_MAX_CHARS:
                content = content[:ARTIFACT_MAX_CHARS] + "\n[... cut: artifact too long]"
            artifacts.append({"kind": a.kind, "content": content})
        return json.dumps({"text": response.text, "artifacts": artifacts}, indent=2)
    if isinstance(response, ClarificationResponse | DeniedResponse):
        return json.dumps({"message": response.message}, indent=2)
    return json.dumps(response.model_dump(mode="json", exclude={"raw_response"}), indent=2)
