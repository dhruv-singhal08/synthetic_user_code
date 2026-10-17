"""Prompt for the generic (all products, both modes) evaluation of one product response.

Inputs: persona, product description (supported / not supported yet / known gaps), scenario
goal + success_criteria, the question (+ intended_goal, interpretation_notes, expectations)
and the product response (text + artifacts summary). The judge fills every generic check with
pass | fail | inconclusive | not_applicable + rationale + evidence quoted from the response.
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
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import Scenario

SYSTEM = (
    "You evaluate how well a conversational product answered a user's question, for any kind "
    "of product. You judge only what the response, the question and the product description "
    "show. You quote evidence exactly as written in the response. "
    "You reply with a single JSON object and nothing else."
)

ARTIFACT_PREVIEW_CHARS = 1500


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
  safety_and_honesty. clarification_handling is not_applicable; clarification is null.
- goal_completeness: return one `criteria` entry per success criterion listed under
  "Success criteria" (copy each criterion text exactly) with status met | partly | not_met |
  unverifiable. goal_completeness is pass only if every criterion is met, fail if any is
  partly or not_met, otherwise inconclusive.""",
    EvaluationFocus.CLARIFICATION: """\
- The product asked a clarification question instead of answering. Focus on
  clarification_handling: is it relevant, necessary (or at least reasonable), and does it
  avoid asking for information the question already gave? Fill `clarification`.
  clarification_handling is fail if the clarification is unnecessary, irrelevant or asks for
  given information; pass if relevant and necessary/reasonable; otherwise inconclusive.
- goal_completeness is not_applicable (nothing was answered yet) and criteria is empty.
- Also judge relevance, persona_fit, capability_consistency and safety_and_honesty.""",
    EvaluationFocus.DENIAL: """\
- The product declined. Judge ONLY capability_consistency: was the denial justified by the
  product description (the request is not supported / a known gap)? An unjustified denial
  of a supported request is fail. Every other check is not_applicable, criteria is empty,
  clarification is null. Do not judge answer quality.""",
}


def build_response_evaluation_prompt(
    persona: Persona,
    product: ProductDescription,
    scenario: Scenario,
    question: QuestionCase,
    response: ProductResponse,
    focus: EvaluationFocus,
) -> tuple[str, str]:
    persona_json = persona.model_dump(
        mode="json", include={"identity", "expertise", "behavior", "primary_motivation"}
    )
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
    scenario_json = scenario.model_dump(
        mode="json", include={"business_goal", "title", "capability", "capability_rationale"}
    )
    question_json = question.model_dump(
        mode="json",
        include={"utterance", "intended_goal", "interpretation_notes", "expectations"},
    )
    criteria = question.expectations.success_criteria or scenario.success_criteria
    grounded = question.mode is GenerationMode.GROUNDED
    truth_rule = (
        "Technical artifacts (e.g. SQL) are checked separately against reference data; judge "
        "here only what the response communicates."
        if grounded
        else "There is NO ground truth (ungrounded run): never claim that facts or numbers are "
        "correct. A criterion that needs factual correctness is unverifiable, and a check that "
        "depends on it is inconclusive. You may still judge presence, relevance, clarity, "
        "honesty about limits and whether figures are presented as certain."
    )
    artifacts_rule = (
        "This product's answers are expected to consist of: "
        f"{', '.join(product.response_artifacts_expected)}. Judge relevance, goal_completeness "
        "and persona_fit against what such an answer delivers: e.g. a product whose answer is "
        "a query or other artifact is not expected to also return the resulting data, and "
        "receiving that artifact is normal for its users."
        if product.response_artifacts_expected
        else "The product description does not say what its answers consist of; judge the "
        "response as a direct answer to the user."
    )
    schema = json.dumps(GenericEvaluationDraft.model_json_schema(), indent=2)
    user = f"""## TASK
Evaluate the product response below with the generic checks. Every check gets a status
(pass | fail | inconclusive | not_applicable), a rationale and, where possible, evidence: an
exact quote from the response.

## INPUT
Persona (who asked):
{json.dumps(persona_json, indent=2)}

Product description:
{json.dumps(product_json, indent=2)}

Scenario:
{json.dumps(scenario_json, indent=2)}

Question:
{json.dumps(question_json, indent=2)}

Success criteria (for goal_completeness):
{json.dumps(criteria, indent=2)}

Product response ({response.outcome.value}):
{_response_view(response)}

## CHECKS
- relevance: does the response address the question that was asked?
- goal_completeness: is each success criterion met / partly / not met?
- persona_fit: do level of detail and terminology suit this persona?
- capability_consistency: the response does not claim or attempt things the product says it
  does not support; a denial is justified when the request is unsupported.
- clarification_handling: only for a clarification - relevant, necessary, not asking for
  information already given.
- safety_and_honesty: no fabricated facts or numbers presented as certain; admits limits.

## RULES
{_FOCUS_RULES[focus]}
- {truth_rule}
- {artifacts_rule}
- Every check with status fail needs a finding whose category is that check's name.
- evidence (in checks and findings) must be copied exactly from the product response, or null.
- evidence_status: supported (the quote shows it), weak, or unsupported.
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
        artifacts = [
            {
                "kind": a.kind,
                "content": (a.content if isinstance(a.content, str) else json.dumps(a.content))[
                    :ARTIFACT_PREVIEW_CHARS
                ],
            }
            for a in response.artifacts
        ]
        return json.dumps({"text": response.text, "artifacts": artifacts}, indent=2)
    if isinstance(response, ClarificationResponse | DeniedResponse):
        return json.dumps({"message": response.message}, indent=2)
    return json.dumps(response.model_dump(mode="json", exclude={"raw_response"}), indent=2)
