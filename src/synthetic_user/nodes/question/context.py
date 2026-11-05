"""The question's Evaluation Context: what is requested from the generator, deterministic
integrity checks, and assembly (reforming, Part 4).

The generator proposes, in the SAME question-generation call: the Success Criteria (each with an
id) and a Metric Mapping entry for every selected metric that maps criteria (the criterion ids it
evaluates) or needs a question-specific expectation (its value). Code does the rest:

- criterion ids are normalised to C1..Cn in the order given (the mapping is renumbered with
  them), so ids are stable whatever the model wrote;
- the Reference Spec is the scenario's Reference Pack, reused as is (never reselected);
- each selected metric's mapping gets its Reference Spec requirements from the metric's
  required reference assets (``catalog_schema`` -> the mandatory ``reference_spec.schema``,
  ``catalog_rules`` -> every table / catalog rule of the pack as AVAILABLE rules: the judge
  decides which apply, only those become scoring checks), its comparison basis and evidence
  types from the frozen metric definition, and its expectation (generated, or decided by code
  up front when the authoritative asset is absent: the LLM is then not asked).

Integrity checks are structural only (no catalog validation, no meaning of text): unique
criterion ids, exactly the requested mapping keys, mapped criterion ids that exist, each mapped
criterion observable in evidence the metric reads (``observed_in`` vs the metric's required
execution evidence), expectation shape / fields / types, and canonical membership of tool /
data-source names in the reference's tool definitions. A criterion no selected metric can
observe stays in the context unmapped (a coverage gap, reported), never forced into a metric
and never a reason to reject the question. A generated expectation that fails canonical
membership after the bounded retries is recorded as ``cannot_generate`` with the exact problem -
never used, never silently fixed - and the question itself is kept.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from synthetic_user.models.metrics import (
    EvidenceKind,
    ExpectationFieldType,
    MetricDefinition,
    ReferenceAsset,
)
from synthetic_user.models.question import (
    RULE_KINDS,
    SCHEMA_REQUIREMENT,
    EvaluationContext,
    EvaluationContextDraft,
    MetricExpectation,
    MetricExpectationDraft,
    MetricExpectationStatus,
    MetricMapping,
    MetricMappingDraft,
    ReferenceSpec,
    SuccessCriterion,
    SuccessCriterionDraft,
    reference_requirements,
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
class ContextRequest:
    """What one question-generation call must return for the run's selected metrics."""

    # Every selected metric (frozen order), available or not: each gets a mapping.
    selected: list[MetricDefinition] = field(default_factory=list)
    # Available metrics the question's Success Criteria are mapped to.
    criteria_metrics: list[MetricDefinition] = field(default_factory=list)
    # Available metrics whose question-specific expectation the generator writes.
    requested: list[MetricDefinition] = field(default_factory=list)
    # Expectations decided by code (the authoritative asset is missing).
    preset: dict[str, MetricExpectation] = field(default_factory=dict)

    @property
    def needs_tool_definitions(self) -> bool:
        return any(
            ReferenceAsset.TOOL_DEFINITIONS in d.required_reference_assets for d in self.requested
        )

    @property
    def mapped(self) -> list[MetricDefinition]:
        """The metrics the generator writes a metric_mapping entry for, in frozen order."""
        ids = {d.metric_id for d in [*self.criteria_metrics, *self.requested]}
        return [d for d in self.selected if d.metric_id in ids]

    @property
    def evidence_kinds(self) -> list[EvidenceKind]:
        """The ``observed_in`` values offered for a criterion: the answer, plus the evidence the
        run's criteria metrics read (e.g. sql, tool_trace)."""
        kinds = [EvidenceKind.ANSWER]
        for definition in self.criteria_metrics:
            kinds.extend(definition.required_execution_evidence)
        return list(dict.fromkeys(kinds))


def observes(definition: MetricDefinition, kind: EvidenceKind) -> bool:
    """Whether a metric reads the evidence a criterion is observed in. A metric that reads the
    answer also reads its artifacts (an SQL artifact is part of the response); a metric whose
    definition names no evidence (some custom metrics) reads everything."""
    evidence = definition.required_execution_evidence
    if not evidence or kind in evidence:
        return True
    return kind is EvidenceKind.SQL and EvidenceKind.ANSWER in evidence


def context_request(
    metrics: Sequence[MetricDefinition],
    reference: Reference | None,
    pack: ReferencePack | None,
) -> ContextRequest:
    request = ContextRequest(selected=list(metrics))
    for definition in metrics:
        if not definition.available:
            continue
        if definition.maps_criteria and not definition.all_success_criteria:
            request.criteria_metrics.append(definition)
        if not definition.needs_expectations:
            continue
        missing = [
            asset
            for asset in definition.required_reference_assets
            if _asset_missing(asset, reference, pack)
        ]
        if missing:
            request.preset[definition.metric_id] = MetricExpectation(
                status=MetricExpectationStatus.CANNOT_GENERATE,
                reason="; ".join(ASSET_MISSING[asset] for asset in missing),
            )
        else:
            request.requested.append(definition)
    return request


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


# --- integrity checks (retry feedback) ---------------------------------------------------------


def mapping_problems(draft: EvaluationContextDraft, request: ContextRequest) -> list[str]:
    """Structural problems of the draft's success_criteria and metric_mapping."""
    problems: list[str] = []
    ids = [c.id for c in draft.success_criteria]
    if repeated := sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"success_criteria ids must be unique; repeated {repeated}")
    known = set(ids)
    wanted = {d.metric_id: d for d in request.mapped}
    got = draft.metric_mapping
    if missing := [m for m in wanted if m not in got]:
        problems.append(f"metric_mapping is missing {missing}")
    if extra := sorted(set(got) - set(wanted)):
        problems.append(
            f"metric_mapping must not include {extra} (not selected for this run, or code "
            "maps them)"
        )
    criteria_ids = {d.metric_id for d in request.criteria_metrics}
    requested_ids = {d.metric_id for d in request.requested}
    observed = {c.id: c.observed_in for c in draft.success_criteria}
    if criteria_ids:
        offered = request.evidence_kinds
        names = " | ".join(k.value for k in offered)
        for criterion in draft.success_criteria:
            if criterion.observed_in is None:
                problems.append(f"{criterion.id}: observed_in is required ({names})")
            elif criterion.observed_in not in offered:
                problems.append(
                    f"{criterion.id}: observed_in {criterion.observed_in.value!r} is not one of "
                    f"{names}"
                )
    for metric_id, entry in got.items():
        definition = wanted.get(metric_id)
        if definition is None:
            continue
        if unknown := [c for c in entry.criteria if c not in known]:
            problems.append(f"{metric_id}: criteria {unknown} are not success_criteria ids")
        if metric_id in criteria_ids:
            for criterion_id in entry.criteria:
                kind = observed.get(criterion_id)
                if kind is not None and not observes(definition, kind):
                    reads = ", ".join(k.value for k in definition.required_execution_evidence)
                    problems.append(
                        f"{metric_id}: {criterion_id} is observed in the {kind.value}, but "
                        f"{metric_id} evaluates only the {reads}: leave {criterion_id} out of "
                        f"{metric_id} (a criterion no listed metric can observe stays unmapped "
                        "and is reported as a coverage gap)"
                    )
        elif entry.criteria:
            problems.append(f"{metric_id}: criteria must be [] (this metric does not use them)")
        if metric_id in requested_ids:
            if entry.expectation is None:
                problems.append(f"{metric_id}: expectation is required")
            else:
                problems.extend(
                    f"{metric_id}: {p}" for p in _entry_problems(entry.expectation, definition)
                )
        elif entry.expectation is not None:
            problems.append(f"{metric_id}: expectation must be null (none is needed)")
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
    draft: EvaluationContextDraft, request: ContextRequest, reference: Reference | None
) -> dict[str, list[str]]:
    """metric id -> canonical-membership problems of its generated expectation."""
    result: dict[str, list[str]] = {}
    context = reference.agent_context if reference is not None else None
    for definition in request.requested:
        mapping = draft.metric_mapping.get(definition.metric_id)
        entry = mapping.expectation if mapping is not None else None
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


# --- assembly -----------------------------------------------------------------------------------


def build_context(
    draft: EvaluationContextDraft,
    request: ContextRequest,
    reference: Reference | None,
    pack: ReferencePack | None,
) -> EvaluationContext:
    """The stored Evaluation Context (the draft passed ``mapping_problems``)."""
    renumber = {c.id: f"C{i}" for i, c in enumerate(draft.success_criteria, start=1)}
    criteria = [
        SuccessCriterion(id=renumber[c.id], requirement=c.requirement, observed_in=c.observed_in)
        for c in draft.success_criteria
    ]
    spec = (
        ReferenceSpec(pack_id=pack.pack_id, reference_id=pack.reference_id, pack=pack)
        if pack is not None
        else None
    )
    expectations = _expectations(draft, request, reference)
    mapping: dict[str, MetricMapping] = {}
    for definition in request.selected:
        entry = draft.metric_mapping.get(definition.metric_id)
        if definition.all_success_criteria:
            mapped_criteria = [c.id for c in criteria]  # every criterion, by code (D256)
        elif entry is not None and definition.maps_criteria:
            mapped_criteria = [renumber[c] for c in entry.criteria if c in renumber]
        else:
            mapped_criteria = []
        mapping[definition.metric_id] = MetricMapping(
            criteria=list(dict.fromkeys(mapped_criteria)),
            reference_requirements=metric_reference_requirements(definition, pack),
            available_rules=metric_available_rules(definition, pack),
            comparison_basis=definition.basis,
            evidence_types=list(definition.required_execution_evidence),
            expectation=expectations.get(definition.metric_id),
        )
    return EvaluationContext(success_criteria=criteria, reference_spec=spec, metric_mapping=mapping)


def metric_reference_requirements(
    definition: MetricDefinition, pack: ReferencePack | None
) -> list[str]:
    """The MANDATORY Reference Spec requirement ids of a metric, from its required assets: the
    selected schema as one requirement. Rules are available, not mandatory (below)."""
    if pack is None:
        return []
    if ReferenceAsset.CATALOG_SCHEMA in definition.reference_assets and pack.tables:
        return [SCHEMA_REQUIREMENT]
    return []


def metric_available_rules(definition: MetricDefinition, pack: ReferencePack | None) -> list[str]:
    """Every table rule and selected catalog rule of the pack, for a metric that checks rules.
    A table rule travels with its table whatever it is about, so the judge decides which rules
    apply to the question and the response; only those become scoring checks."""
    if pack is None or ReferenceAsset.CATALOG_RULES not in definition.reference_assets:
        return []
    return [item.id for item in reference_requirements(pack) if item.kind in RULE_KINDS]


def _expectations(
    draft: EvaluationContextDraft, request: ContextRequest, reference: Reference | None
) -> dict[str, MetricExpectation]:
    """Canonical names applied, invalid generated values recorded as cannot_generate with the
    reason, code-decided entries added."""
    invalid = canonical_problems(draft, request, reference)
    context = reference.agent_context if reference is not None else None
    entries: dict[str, MetricExpectation] = {}
    for definition in request.requested:
        metric_id = definition.metric_id
        mapping = draft.metric_mapping.get(metric_id)
        entry = mapping.expectation if mapping is not None else None
        if entry is None:
            continue
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
            and ReferenceAsset.TOOL_DEFINITIONS in definition.required_reference_assets
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
    return entries


def _strings(value: Any) -> list[str]:
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def fixed_context(
    criteria: Sequence[str],
    pack: ReferencePack | None,
    metrics: Sequence[MetricDefinition],
) -> EvaluationContext:
    """An Evaluation Context for given criteria (no generation call, e.g. labeled benchmark
    cases): every criterion is mapped to every selected criteria metric; no question-specific
    expectation exists, so a metric that needs one is reported as cannot_generate."""
    ids = [f"C{i}" for i in range(1, len(criteria) + 1)]
    request = context_request(metrics, None, pack)
    draft = EvaluationContextDraft(
        success_criteria=[
            SuccessCriterionDraft(id=i, requirement=text)
            for i, text in zip(ids, criteria, strict=True)
        ],
        metric_mapping={
            d.metric_id: MetricMappingDraft(criteria=ids) for d in request.criteria_metrics
        },
    )
    return build_context(draft, ContextRequest(selected=list(metrics)), None, pack)
