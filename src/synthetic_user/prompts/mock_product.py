"""Prompt for the MOCK product: an LLM stand-in for ANY product, until the real one is connected.

It plays the product described by the product description, answering from the user's
reference when one is given. Artifacts the product returns (e.g. a query) go in fenced blocks
labelled with the artifact kind.
"""

from __future__ import annotations

import json

from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import Reference

SYSTEM = (
    "You are a MOCK of the conversational product described below, used for testing. Answer "
    "the user's message as that product would. Reply in exactly one of these forms and "
    "nothing else:\n"
    "1. the answer: plain text, plus any artifacts the product returns, each in a fenced code "
    "block whose label is the artifact kind (e.g. ```sql ... ```)\n"
    "2. CLARIFY: <one clarification question> - only if the message is genuinely ambiguous\n"
    "3. DENIED: <reason> - if the request is something the product does not support"
)


def build_mock_product_prompt(
    question: str, product: ProductDescription, reference: Reference | None
) -> tuple[str, str]:
    product_json = product.model_dump(
        mode="json",
        include={
            "name",
            "description",
            "product_type",
            "supported",
            "not_supported_yet",
            "response_artifacts_expected",
        },
    )
    reference_block = (
        f"Reference the product works from ({reference.format.value}):\n"
        f"<<<REFERENCE\n{reference.text}\nREFERENCE>>>\n\n"
        if reference is not None
        else "No reference data: answer only what the product description makes plausible.\n\n"
    )
    user = (
        f"Product:\n{json.dumps(product_json, indent=2)}\n\n"
        f"{reference_block}"
        f"User message: {question}"
    )
    return SYSTEM, user
