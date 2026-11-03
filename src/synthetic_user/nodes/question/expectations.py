"""Question-specific metric expectations: what is requested, and deterministic validation.

Only selected metrics whose definition declares a ``question_expectation_schema`` get an entry.
Code decides ``cannot_generate`` up front when the authoritative asset the expectation must be
grounded in is absent (e.g. no tool definitions in the reference): the LLM is then not asked.

Validation is structural and deterministic: exactly the requested metric ids, a valid status
shape, required fields present, declared field types, no unknown fields, and canonical
membership of tool / data-source names in the reference's tool definitions (aliases map to the
canonical name). A generated value that fails canonical membership after the bounded retries is
recorded as ``cannot_generate`` with the exact problem - never used, never silently fixed - and
the question itself is kept (one metric's expectation never rejects an otherwise valid
question). Nothing here inspects the meaning of text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from synthetic_user.models.metrics import (
    CHEAT_SHEET_KEY,
    GENERAL_CRITERIA_KEY,
    ExpectationFieldType,
    MetricDefinition,
    ReferenceAsset,
)
from synthetic_user.models.question import (
    Expectations,
    ExpectationsDraft,
    MetricExpectation,
    MetricExpectationDraft,
    MetricExpectationStatus,
    SuccessCriteria,
)
from synthetic_user.models.reference import Reference, ReferenceKind, ReferencePack

# Expectation fields whose values must be canonical names in the reference tool definitions.
TOOL_FIELD = "expected_tools"
SOURCE_FIELD = "expected_data_sources"
ASSET_MISSING = {
    ReferenceAsset.TOOL_DEFINITIONS: (
        "no authoritative tool definitions: the reference has no agent_context tools"
    ),
    ReferenceAsset.CATALOG_SCHEMA: "no table/column cheat sheet for this scenario",
    ReferenceAsset.CATALOG_RULES: "no rules or documented joins in this scenario's cheat sheet",
}


@dataclass(frozen=True)
class ExpectationRequest:
    requested: list[MetricDefinition] = field(default_factory=list)
    preset: dict[str, MetricExpectation] = field(default_factory=dict)

    @property
    def needs_tool_definitions(self) -> bool:
        return any(
            ReferenceAsset.TOOL_DEFINITIONS in d.required_reference_assets for d in self.requested
        )


def expectation_request(
    metrics: Sequence[MetricDefinition],
    reference: Reference | None,
    pack: ReferencePack | None,
) -> ExpectationRequest:
    requested: list[MetricDefinition] = []
    preset: dict[str, MetricExpectation] = {}
    for definition in metrics:
        if not definition.available or not definition.needs_expectations:
            continue
        missing = [
            asset
            for asset in definition.required_reference_assets
            if _asset_missing(asset, reference, pack)
        ]
        if missing:
            preset[definition.metric_id] = MetricExpectation(
                status=MetricExpectationStatus.CANNOT_GENERATE,
                reason="; ".join(ASSET_MISSING[asset] for asset in missing),
            )
        else:
            requested.append(definition)
    return ExpectationRequest(requested, preset)


def _asset_missing(
    asset: ReferenceAsset, reference: Reference | None, pack: ReferencePack | None
) -> bool:
    if asset is ReferenceAsset.TOOL_DEFINITIONS:
        context = reference.agent_context if reference is not None else None
        return context is None or not context.tools
    catalog = pack is not None and pack.kind is ReferenceKind.DATA_CATALOG
    if asset is ReferenceAsset.CATALOG_SCHEMA:
        return not (catalog and pack is not None and pack.tables)
    return not (
        catalog
        and pack is not None
        and (pack.rules or pack.joins or any(t.rules for t in pack.tables))
    )


def shape_problems(draft: ExpectationsDraft, request: ExpectationRequest) -> list[str]:
    """Structural problems of the draft's metric_expectations (keys, statuses, fields, types)."""
    wanted = {d.metric_id: d for d in request.requested}
    got = draft.metric_expectations
    problems: list[str] = []
    if missing := sorted(set(wanted) - set(got)):
        problems.append(f"metric_expectations is missing {missing}")
    if extra := sorted(set(got) - set(wanted)):
        problems.append(
            f"metric_expectations must not include {extra} (not requested for this run)"
        )
    for metric_id, entry in got.items():
        definition = wanted.get(metric_id)
        if definition is not None:
            problems.extend(f"{metric_id}: {p}" for p in _entry_problems(entry, definition))
    return problems


def _entry_problems(entry: MetricExpectationDraft, definition: MetricDefinition) -> list[str]:
    status = entry.status
    if status is not MetricExpectationStatus.GENERATED:
        problems = [] if entry.value is None else [f"a {status.value} expectation has value null"]
        if entry.reason is None:
            problems.append(f"a {status.value} expectation needs a specific reason")
        return problems
    if not entry.value:
        return ["a generated expectation needs a value"]
    fields = {f.name: f for f in definition.question_expectation_schema}
    problems = [f"unknown field {name!r}" for name in entry.value if name not in fields]
    for name, spec in fields.items():
        value = entry.value.get(name)
        if value is None or value == "" or value == []:
            if spec.required:
                problems.append(f"required field {name!r} is missing")
            continue
        if not _type_ok(value, spec.type):
            problems.append(f"field {name!r} must be a {spec.type.value}")
    return problems


def _type_ok(value: Any, kind: ExpectationFieldType) -> bool:
    if kind is ExpectationFieldType.BOOLEAN:
        return isinstance(value, bool)
    if kind is ExpectationFieldType.STRING:
        return isinstance(value, str) and bool(value.strip())
    return isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value)


def canonical_problems(
    draft: ExpectationsDraft, request: ExpectationRequest, reference: Reference | None
) -> dict[str, list[str]]:
    """metric id -> canonical-membership problems of its generated value."""
    result: dict[str, list[str]] = {}
    context = reference.agent_context if reference is not None else None
    for definition in request.requested:
        entry = draft.metric_expectations.get(definition.metric_id)
        if (
            entry is None
            or entry.status is not MetricExpectationStatus.GENERATED
            or not entry.value
            or ReferenceAsset.TOOL_DEFINITIONS not in definition.required_reference_assets
            or context is None
        ):
            continue
        problems: list[str] = []
        for name in _strings(entry.value.get(TOOL_FIELD)):
            if context.tool(name) is None:
                problems.append(f"unknown tool {name!r} (not in the reference tool definitions)")
        for name in _strings(entry.value.get(SOURCE_FIELD)):
            if context.data_source(name) is None and context.source_group(name) is None:
                problems.append(f"unknown data source {name!r} (not in the tool definitions)")
        if problems:
            result[definition.metric_id] = problems
    return result


def build_expectations(
    draft: ExpectationsDraft,
    request: ExpectationRequest,
    reference: Reference | None,
    pack: ReferencePack | None,
) -> Expectations:
    """The stored expectations: canonical names applied, invalid generated values recorded as
    cannot_generate with the reason, code-decided entries added. Everything except
    expected_behavior is collected into one structured success_criteria (D240): the draft's
    criteria under ``general``, a copy of the scenario's pack under ``cheat_sheet`` and the
    metric expectations under their metric ids."""
    invalid = canonical_problems(draft, request, reference)
    context = reference.agent_context if reference is not None else None
    entries: dict[str, MetricExpectation] = {}
    for definition in request.requested:
        entry = draft.metric_expectations[definition.metric_id]
        metric_id = definition.metric_id
        if metric_id in invalid:
            entries[metric_id] = MetricExpectation(
                status=MetricExpectationStatus.CANNOT_GENERATE,
                reason="generated expectation rejected: " + "; ".join(invalid[metric_id]),
            )
            continue
        value = dict(entry.value) if entry.value is not None else None
        if (
            value is not None
            and context is not None
            and (ReferenceAsset.TOOL_DEFINITIONS in definition.required_reference_assets)
        ):
            if TOOL_FIELD in value:
                value[TOOL_FIELD] = [
                    tool.name for n in _strings(value[TOOL_FIELD]) if (tool := context.tool(n))
                ]
            if SOURCE_FIELD in value:
                value[SOURCE_FIELD] = [
                    found.name
                    for n in _strings(value[SOURCE_FIELD])
                    if (found := context.data_source(n) or context.source_group(n))
                ]
        entries[metric_id] = MetricExpectation(
            status=entry.status, value=value, reason=entry.reason
        )
    entries.update(request.preset)
    criteria = SuccessCriteria.model_validate(
        {
            GENERAL_CRITERIA_KEY: draft.success_criteria,
            CHEAT_SHEET_KEY: pack,
            **entries,
        }
    )
    return Expectations(expected_behavior=draft.expected_behavior, success_criteria=criteria)


def _strings(value: Any) -> list[str]:
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []
