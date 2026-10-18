"""Reference Pack validation: every core_ref and rule must quote the reference verbatim.

Used by the scenario node (items with an unknown quote are rejected and recorded; a pack
without any valid core_ref rejects its scenario) and by pack edits from the UI (an edit with
any unknown quote is refused as a whole, so a person sees exactly what to fix).
"""

from __future__ import annotations

from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.models.reference import (
    CoreRef,
    Reference,
    ReferencePack,
    ReferencePackDraft,
    ReferenceRule,
    RejectedRefItem,
    pack_id_for,
)

NO_VALID_CORE_REF = "reference pack has no core_ref whose quote is in the reference"


def quote_problem(quote: str, reference: Reference) -> str | None:
    if reference.contains(quote):
        return None
    return f"quote not found in the reference: {quote[:120]!r}"


def build_pack(
    draft: ReferencePackDraft, reference: Reference, scenario_id: str
) -> tuple[ReferencePack | None, list[RejectedRefItem]]:
    """Keep the items whose quote is in the reference; the rest are rejected with a reason.

    Returns (None, rejected) when no core_ref survives."""
    core: list[CoreRef] = []
    rules: list[ReferenceRule] = []
    rejected: list[RejectedRefItem] = []
    for item in draft.core_refs:
        problem = quote_problem(item.quote, reference)
        if problem:
            rejected.append(RejectedRefItem(item=item.model_dump(mode="json"), reason=problem))
        else:
            core.append(item)
    for rule in draft.rules:
        problem = quote_problem(rule.quote, reference)
        if problem:
            rejected.append(RejectedRefItem(item=rule.model_dump(mode="json"), reason=problem))
        else:
            rules.append(rule)
    if not core:
        return None, rejected
    pack = ReferencePack(
        pack_id=pack_id_for(scenario_id),
        scenario_id=scenario_id,
        reference_id=reference.reference_id,
        core_refs=core,
        rules=rules,
        rejected_items=rejected,
    )
    return pack, rejected


def edit_pack(
    pack: ReferencePack, draft: ReferencePackDraft, reference: Reference
) -> ReferencePack:
    """A person's edit: all quotes must be in the reference, else InputAssetError listing
    every problem (nothing is saved)."""
    if pack.reference_id != reference.reference_id:
        raise InputAssetError(
            f"{pack.pack_id} was built from reference {pack.reference_id}, "
            f"not {reference.reference_id}"
        )
    problems = [
        f"{kind} {i} ({name}): {problem}"
        for kind, items in (
            ("core_ref", [(c.name, c.quote) for c in draft.core_refs]),
            ("rule", [(r.rule, r.quote) for r in draft.rules]),
        )
        for i, (name, quote) in enumerate(items, start=1)
        if (problem := quote_problem(quote, reference))
    ]
    if problems:
        raise InputAssetError(f"{pack.pack_id} not saved: " + "; ".join(problems))
    return pack.model_copy(
        update={"core_refs": draft.core_refs, "rules": draft.rules, "edited": True}
    )
