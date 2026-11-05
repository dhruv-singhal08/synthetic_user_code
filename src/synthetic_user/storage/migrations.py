"""Explicit, opt-in migration of runs created before the metric registry (schema-1 evaluations).

Reading old runs never rewrites them: metric ids stay as judged (``completion``,
``tool_selection``, ...). The only relabelling offered is the one the metric catalog allows:
``completion`` -> ``correctness`` (the merged definition), and only when the user
asks for it. Every migrated result is marked in its notes; ``tool_selection`` is never mapped
to ``tool_correctness`` because the new definition is broader than what was judged.
"""

from __future__ import annotations

from collections.abc import Sequence

from synthetic_user.core.exceptions import ArtifactIntegrityError
from synthetic_user.models.evaluation import EvaluationResult, Finding

# old id -> new id; the only mapping the catalog allows (instruction section 11).
ALLOWED_MAPPINGS = {"completion": "correctness"}
REFUSED_MAPPINGS = {
    "tool_selection": "tool_correctness is an expanded definition: historical tool_selection "
    "scores are not relabelled",
}


def migration_note(old: str, new: str) -> str:
    return (
        f"Explicit migration: {old} -> {new}. The score was judged under the legacy {old} "
        "rubric, not re-evaluated."
    )


def map_legacy_metric(
    evaluations: Sequence[EvaluationResult], old: str, new: str
) -> tuple[list[EvaluationResult], int]:
    """Rename metric ``old`` to ``new`` in schema-1 evaluations (results + finding categories).

    Returns the evaluations and how many were changed. Current-schema evaluations are refused:
    their ids come from a frozen metric configuration and are never rewritten.
    """
    if old in REFUSED_MAPPINGS:
        raise ArtifactIntegrityError(REFUSED_MAPPINGS[old])
    if ALLOWED_MAPPINGS.get(old) != new:
        raise ArtifactIntegrityError(
            f"Unsupported metric migration {old} -> {new}; allowed: {ALLOWED_MAPPINGS}"
        )
    current = [e.evaluation_id for e in evaluations if e.schema_version >= 2]
    if current:
        raise ArtifactIntegrityError(
            f"{len(current)} evaluation(s) already use the metric registry (e.g. {current[0]}); "
            "only legacy (schema 1) runs can be migrated"
        )
    migrated: list[EvaluationResult] = []
    changed = 0
    for evaluation in evaluations:
        if old not in evaluation.metrics:
            migrated.append(evaluation)
            continue
        if new in evaluation.metrics:
            raise ArtifactIntegrityError(
                f"{evaluation.evaluation_id} already has both {old} and {new}"
            )
        metrics = {(new if k == old else k): v for k, v in evaluation.metrics.items()}
        findings = [
            f.model_copy(update={"category": new}) if f.category == old else f
            for f in evaluation.findings
        ]
        migrated.append(
            evaluation.model_copy(
                update={
                    "metrics": metrics,
                    "findings": [Finding.model_validate(f.model_dump()) for f in findings],
                    "notes": [*evaluation.notes, migration_note(old, new)],
                }
            )
        )
        changed += 1
    return migrated, changed
