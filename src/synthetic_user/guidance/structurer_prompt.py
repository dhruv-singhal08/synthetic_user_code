"""Assemble the guidance-structurer prompt; the text itself is in ``structurer_prompt.yaml``."""

from __future__ import annotations

import json
from collections.abc import Sequence

from synthetic_user.models.run import EvaluationGuidance, GuidanceType
from synthetic_user.prompts.contracts import choices, render_contract
from synthetic_user.prompts.templates import prompts_beside

PROMPTS = prompts_beside(__file__, "structurer_prompt.yaml")
SYSTEM = PROMPTS.text("system")


def build_structurer_prompt(
    guidance: EvaluationGuidance, metric_ids: Sequence[str]
) -> tuple[str, str]:
    numbered = "\n".join(
        f"{number:>4} | {line}" for number, line in enumerate(guidance.text.splitlines(), start=1)
    )
    user = PROMPTS.text(
        "user",
        source=guidance.source,
        numbered=numbered,
        metric_ids=json.dumps(list(metric_ids)),
        contract=render_contract(PROMPTS.data("example", types=choices(GuidanceType))),
    )
    return SYSTEM, user
