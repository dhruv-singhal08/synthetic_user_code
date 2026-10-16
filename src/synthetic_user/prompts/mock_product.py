"""Prompt for the MOCK NL2SQL product (stand-in until the real product is connected)."""

from __future__ import annotations

import json

from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import ReferenceCatalog

SYSTEM = (
    "You are a MOCK NL2SQL product used for testing. Answer the user's question about the "
    "database below. Reply in exactly one of these forms and nothing else:\n"
    "1. one SQL SELECT statement inside a ```sql code block\n"
    "2. CLARIFY: <one clarification question> - only if the question is genuinely ambiguous\n"
    "3. DENIED: <reason> - if the request is something the product does not support"
)


def build_mock_product_prompt(
    question: str, catalog: ReferenceCatalog, product: ProductDescription, sql_dialect: str
) -> tuple[str, str]:
    schema = {
        "tables": [
            {
                "name": t.name,
                "description": t.description,
                "columns": [
                    {"name": c.name, "type": c.data_type, "description": c.description}
                    for c in t.columns
                ],
            }
            for t in catalog.tables
        ],
        "relationships": [r.model_dump(mode="json") for r in catalog.relationships],
    }
    capabilities = product.model_dump(mode="json", include={"supported", "not_supported_yet"})
    user = (
        f"SQL dialect: {sql_dialect}\n\n"
        f"Product capabilities:\n{json.dumps(capabilities, indent=2)}\n\n"
        f"Database schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"Question: {question}"
    )
    return SYSTEM, user
