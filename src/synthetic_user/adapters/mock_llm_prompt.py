"""Prompt for the MOCK product: an LLM stand-in for ANY product, until the real one is connected.

It plays the product described by the product description, answering from the user's
reference when one is given. Artifacts the product returns (e.g. a query) go in fenced blocks
labelled with the artifact kind. The prompt text is in ``mock_llm_prompt.yaml``.
"""

from __future__ import annotations

import json

from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import Reference
from synthetic_user.prompts.templates import prompts_beside

PROMPTS = prompts_beside(__file__, "mock_llm_prompt.yaml")
SYSTEM = PROMPTS.text("system")
# Fields of the product description the mock product plays.
PRODUCT_FIELDS = {
    "name",
    "description",
    "data_sources",
    "supported",
    "not_supported",
    "known_limitations",
    "user_visible_capabilities",
    "expected_outputs",
}


def build_mock_product_prompt(
    question: str, product: ProductDescription, reference: Reference | None
) -> tuple[str, str]:
    product_json = product.model_dump(mode="json", include=PRODUCT_FIELDS)
    reference_block = (
        PROMPTS.text("reference", kind=reference.kind.value, reference=reference.render())
        if reference is not None
        else PROMPTS.text("no_reference")
    )
    user = PROMPTS.text(
        "user",
        product=json.dumps(product_json, indent=2),
        reference=reference_block,
        question=question,
    )
    return SYSTEM, user
