"""Question cases: one natural-language question the synthetic user asks the product."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Self

from pydantic import Field, JsonValue, model_validator

from synthetic_user.models.base import AmbiguityLevel, NonEmptyStr, StrictModel
from synthetic_user.models.metrics import METRIC_ID_PATTERN, EvidenceKind
from synthetic_user.models.reference import ReferencePack
from synthetic_user.models.run import GenerationMode
from synthetic_user.models.scenario import TaskType


class QuestionStyle(StrEnum):
    PRECISE = "precise"
    VAGUE = "vague"
    INVESTIGATIVE = "investigative"


class QuestionComplexity(StrEnum):
    """Computed from the number of deliverables and sources in the question plan."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class DeliverableAction(StrEnum):
    """One observable output requested from the product."""

    LOOKUP = "lookup"
    RANK = "rank"
    TREND = "trend"
    AGGREGATE = "aggregate"
    COMPARE = "compare"
    EXPLAIN = "explain"
    SUMMARIZE = "summarize"
    QUOTE = "quote"
    RECOMMEND = "recommend"


class QuestionIntent(StrictModel):
    task_type: TaskType


class QuestionCharacteristics(StrictModel):
    style: QuestionStyle
    ambiguity_level: AmbiguityLevel
    complexity: QuestionComplexity


MAX_SUCCESS_CRITERIA = 10
MAX_DELIVERABLES = 3


class QuestionDeliverable(StrictModel):
    """A single action, kept separate so several outputs cannot be hidden in one sentence."""

    action: DeliverableAction
    subject: NonEmptyStr
    sources: list[NonEmptyStr] = Field(default_factory=list, max_length=2)
    limit: int | None = Field(default=None, ge=1, le=100)


class QuestionFilter(StrictModel):
    name: NonEmptyStr
    value: NonEmptyStr


class QuestionPlan(StrictModel):
    """Machine-checkable plan behind an utterance; filters do not count as deliverables."""

    deliverables: list[QuestionDeliverable] = Field(min_length=1, max_length=MAX_DELIVERABLES)
    filters: list[QuestionFilter] = Field(default_factory=list, max_length=8)
    is_follow_up: bool = False


class QuestionDraftCharacteristics(StrictModel):
    """The model describes style and ambiguity; code computes complexity from the plan."""

    style: QuestionStyle
    ambiguity_level: AmbiguityLevel


class MetricExpectationStatus(StrEnum):
    GENERATED = "generated"  # a validated, question-specific value exists
    NOT_APPLICABLE = "not_applicable"  # the metric does not apply to this question
    # The metric may apply, but its expectation cannot be established from the question and
    # the authoritative assets: no value is fabricated and the metric is never scored 0.
    CANNOT_GENERATE = "cannot_generate"


class MetricExpectation(StrictModel):
    status: MetricExpectationStatus
    value: dict[NonEmptyStr, JsonValue] | None = None
    reason: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if self.status is MetricExpectationStatus.GENERATED:
            if not self.value:
                raise ValueError("a generated metric expectation needs a value")
        elif self.value is not None:
            raise ValueError(f"a {self.status.value} metric expectation has value null")
        if self.status is MetricExpectationStatus.CANNOT_GENERATE and self.reason is None:
            raise ValueError("cannot_generate needs a specific reason")
        return self


# --- the Evaluation Context (reforming, Part 4) -------------------------------------------------

CRITERION_ID_PATTERN = r"^C\d+$"
# The Reference Spec requirement that stands for the whole selected schema (tables, columns and
# documented relationships): schema validity is judged against the full catalog through the
# catalog reconciliation, so it is one requirement, not one per identifier.
SCHEMA_REQUIREMENT = "reference_spec.schema"


class SuccessCriterion(StrictModel):
    """One atomic, observable must-have of THIS question (id assigned by code: C1, C2, ...).
    ``observed_in`` = the execution evidence it is observable in (answer, sql, tool_trace,
    ...): a criterion is mapped only to metrics that read that evidence. None in questions
    generated before it, or in fixed criteria (labeled benchmark cases)."""

    id: NonEmptyStr = Field(pattern=CRITERION_ID_PATTERN)
    requirement: NonEmptyStr
    observed_in: EvidenceKind | None = None


class ReferenceRequirementKind(StrEnum):
    SCHEMA = "schema"  # the selected tables, columns and joins as one requirement
    TABLE = "table"
    COLUMN = "column"
    JOIN = "join"
    TABLE_RULE = "table_rule"
    RULE = "rule"  # a catalog-level rule selected for the scenario
    DOCUMENT = "document"


RULE_KINDS = frozenset({ReferenceRequirementKind.TABLE_RULE, ReferenceRequirementKind.RULE})
# Id prefixes of the rule kinds (TABLE-RULE-01.02, RULE-R1): read mappings stored before
# ``available_rules`` existed, when every rule was a mandatory reference requirement.
RULE_ID_PREFIXES = ("TABLE-RULE-", "RULE-")


class ReferenceRequirement(StrictModel):
    """An addressable item of the Reference Spec; ``id`` is its stable trace tag."""

    id: NonEmptyStr
    kind: ReferenceRequirementKind
    text: NonEmptyStr
    # Provenance inside the pack (e.g. "tables[card_transactions].rules[1]").
    source: NonEmptyStr


def reference_requirements(pack: ReferencePack) -> list[ReferenceRequirement]:
    """Every addressable item of a Reference Pack, in pack order, with the stable tags the
    judges and the report use (TABLE-01, COLUMN-01.02, TABLE-RULE-01.02, JOIN-01, RULE-<id>,
    DOCUMENT-01) plus ``reference_spec.schema`` for a data-catalog pack. Code only."""
    kind = ReferenceRequirementKind
    items: list[ReferenceRequirement] = []
    if pack.tables:
        names = ", ".join(t.name for t in pack.tables)
        items.append(
            ReferenceRequirement(
                id=SCHEMA_REQUIREMENT,
                kind=kind.SCHEMA,
                text=f"tables {names} with their selected columns and documented joins",
                source="tables, joins",
            )
        )
    for ti, table in enumerate(pack.tables, start=1):
        at = f"tables[{table.name}]"
        items.append(
            ReferenceRequirement(id=f"TABLE-{ti:02d}", kind=kind.TABLE, text=table.name, source=at)
        )
        for ci, column in enumerate(table.columns, start=1):
            items.append(
                ReferenceRequirement(
                    id=f"COLUMN-{ti:02d}.{ci:02d}",
                    kind=kind.COLUMN,
                    text=f"{table.name}.{column.name}",
                    source=f"{at}.columns[{column.name}]",
                )
            )
        for ri, rule in enumerate(table.rules, start=1):
            items.append(
                ReferenceRequirement(
                    id=f"TABLE-RULE-{ti:02d}.{ri:02d}",
                    kind=kind.TABLE_RULE,
                    text=rule,
                    source=f"{at}.rules[{ri}]",
                )
            )
    for ji, join in enumerate(pack.joins, start=1):
        state = "verified" if join.verified else "not verified"
        items.append(
            ReferenceRequirement(
                id=f"JOIN-{ji:02d}",
                kind=kind.JOIN,
                text=f"{join.from_} -> {join.to} ({state})",
                source=f"joins[{ji}]",
            )
        )
    for selected in pack.rules:
        items.append(
            ReferenceRequirement(
                id=f"RULE-{selected.id}",
                kind=kind.RULE,
                text=selected.rule,
                source=f"rules[{selected.id}]",
            )
        )
    for di, document in enumerate(pack.documents, start=1):
        items.append(
            ReferenceRequirement(
                id=f"DOCUMENT-{di:02d}",
                kind=kind.DOCUMENT,
                text=document.name,
                source=f"documents[{document.name}]",
            )
        )
    return items


class ReferenceSpec(StrictModel):
    """The scenario's Reference Pack reused as this question's authoritative technical scope:
    never reselected by question generation. ``pack`` is a code copy (editing a pack already
    invalidates its questions, so the copy never goes stale); identity and provenance are kept
    in ``pack_id`` / ``reference_id`` (the catalog version: file stem + content hash)."""

    pack_id: NonEmptyStr
    reference_id: NonEmptyStr
    pack: ReferencePack

    @model_validator(mode="after")
    def _same_pack(self) -> Self:
        if (self.pack.pack_id, self.pack.reference_id) != (self.pack_id, self.reference_id):
            raise ValueError("reference_spec identity must match its pack")
        return self

    def requirements(self) -> list[ReferenceRequirement]:
        return reference_requirements(self.pack)

    def requirement(self, requirement_id: str) -> ReferenceRequirement | None:
        return next((r for r in self.requirements() if r.id == requirement_id), None)


class MetricMapping(StrictModel):
    """Which requirements one selected metric evaluates for this question, and how.

    ``criteria`` are proposed by the question generator (validated by code); every other field
    is filled by code: the reference requirements from the Reference Spec and the metric's
    required assets, the comparison basis and evidence types from the frozen metric definition,
    and ``expectation`` = the metric's question-specific value when its definition needs one.

    ``reference_requirements`` are mandatory checks (e.g. ``reference_spec.schema``).
    ``available_rules`` are the Reference Spec rules the metric may apply (every table rule and
    selected catalog rule of the pack): the judge decides which apply to the question and the
    response; only the applicable ones become scoring checks, the rest are recorded as not
    applicable (none is silently dropped)."""

    criteria: list[NonEmptyStr] = Field(default_factory=list)
    reference_requirements: list[NonEmptyStr] = Field(default_factory=list)
    available_rules: list[NonEmptyStr] = Field(default_factory=list)
    comparison_basis: NonEmptyStr
    evidence_types: list[EvidenceKind] = Field(default_factory=list)
    expectation: MetricExpectation | None = None

    @model_validator(mode="before")
    @classmethod
    def _split_rules(cls, value: Any) -> Any:
        """A mapping stored before ``available_rules``: its rule ids become available rules."""
        if not isinstance(value, dict) or "available_rules" in value:
            return value
        ids = list(value.get("reference_requirements") or [])
        rules = [i for i in ids if isinstance(i, str) and i.startswith(RULE_ID_PREFIXES)]
        if not rules:
            return value
        return {
            **value,
            "reference_requirements": [i for i in ids if i not in rules],
            "available_rules": rules,
        }

    @model_validator(mode="after")
    def _unique(self) -> Self:
        for name, ids in (
            ("criteria", self.criteria),
            ("reference", self.reference_requirements),
            ("rule", self.available_rules),
        ):
            if len(ids) != len(set(ids)):
                raise ValueError(f"metric mapping lists a {name} id twice: {ids}")
        return self


class EvaluationContext(StrictModel):
    """The question-specific evaluation contract: what the product must accomplish (Success
    Criteria), the authoritative technical scope (Reference Spec; grounded questions only) and
    which of them each selected metric evaluates (Metric Mapping). Evaluation Guidance is NOT
    part of it: it belongs to the evaluator configuration of the run."""

    success_criteria: list[SuccessCriterion] = Field(min_length=1)
    reference_spec: ReferenceSpec | None = None
    metric_mapping: dict[NonEmptyStr, MetricMapping] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _integrity(self) -> Self:
        """Structural integrity only (never catalog validation): ids C1..Cn in order, mapping
        keys are metric ids, and every mapped criterion / reference requirement exists."""
        ids = [c.id for c in self.success_criteria]
        expected = [f"C{i}" for i in range(1, len(ids) + 1)]
        if ids != expected:
            raise ValueError(f"success criteria ids must be {expected}, got {ids}")
        known = set(ids)
        spec = {r.id for r in self.reference_spec.requirements()} if self.reference_spec else set()
        problems: list[str] = []
        for metric_id, mapping in self.metric_mapping.items():
            if not re.fullmatch(METRIC_ID_PATTERN, metric_id):
                problems.append(f"{metric_id!r} is not a metric id")
            if unknown := [c for c in mapping.criteria if c not in known]:
                problems.append(f"{metric_id}: unknown criteria {unknown}")
            if unknown := [
                r
                for r in [*mapping.reference_requirements, *mapping.available_rules]
                if r not in spec
            ]:
                problems.append(
                    f"{metric_id}: reference requirements {unknown} are not in the spec"
                )
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @property
    def pack(self) -> ReferencePack | None:
        return self.reference_spec.pack if self.reference_spec is not None else None

    def criterion(self, criterion_id: str) -> SuccessCriterion | None:
        return next((c for c in self.success_criteria if c.id == criterion_id), None)

    def unmapped_criteria(self) -> list[SuccessCriterion]:
        """Criteria no selected metric evaluates (no selected metric can observe them): kept in
        the context and recorded as coverage gaps, never forced into an unsuitable metric."""
        mapped = {c for m in self.metric_mapping.values() for c in m.criteria}
        return [c for c in self.success_criteria if c.id not in mapped]

    def requirement_text(self, requirement_id: str) -> str | None:
        """The text of a criterion or Reference Spec requirement id; None when unknown."""
        if (criterion := self.criterion(requirement_id)) is not None:
            return criterion.requirement
        if self.reference_spec is not None:
            found = self.reference_spec.requirement(requirement_id)
            return found.text if found is not None else None
        return None

    def expectation(self, metric_id: str) -> MetricExpectation | None:
        mapping = self.metric_mapping.get(metric_id)
        return mapping.expectation if mapping is not None else None

    def generated(self, metric_id: str) -> dict[str, JsonValue] | None:
        """The metric's question-specific value when it was generated, else None."""
        entry = self.expectation(metric_id)
        if entry is None or entry.status is not MetricExpectationStatus.GENERATED:
            return None
        return dict(entry.value or {})


LEGACY_QUESTION_MESSAGE = (
    "{question_id} was generated before the Evaluation Context (it has expected_behavior / the "
    "old success_criteria): it is kept for display only. Regenerate the questions step to "
    "execute or evaluate it."
)


class QuestionCase(StrictModel):
    question_id: NonEmptyStr = Field(pattern=r"^Q_\d{3,}$")
    scenario_id: NonEmptyStr = Field(pattern=r"^SCN_\d{3,}$")
    persona_id: NonEmptyStr
    mode: GenerationMode
    reference_pack_id: NonEmptyStr | None = None
    utterance: NonEmptyStr
    intended_goal: NonEmptyStr
    intent: QuestionIntent
    question_characteristics: QuestionCharacteristics
    # The evaluation contract; None only for a question stored before it existed (legacy).
    evaluation_context: EvaluationContext | None = None
    # Legacy questions: their former ``expectations`` (expected_behavior + success_criteria),
    # kept verbatim for display. Such a question is never executed or evaluated (user decision).
    legacy_expectations: dict[NonEmptyStr, JsonValue] | None = None
    # Optional only for backward compatibility with artifacts created before the plan existed.
    plan: QuestionPlan | None = None
    # Reference names (identifier-like core_ref names) the utterance repeats.
    leak_warnings: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _read_legacy(cls, value: Any) -> Any:
        """Profile / golden-set era fields are dropped (the profile is not a question property
        any more); the pre-Evaluation-Context ``expectations`` are kept as legacy text."""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        data.pop("evaluation_profile", None)
        data.pop("golden_case_id", None)
        if "expectations" in data:
            data["legacy_expectations"] = data.pop("expectations")
        return data

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        # Agent-profile questions of earlier runs were grounded without a pack: accepted on
        # read only (no new question is created that way).
        if self.mode is GenerationMode.UNGROUNDED and self.reference_pack_id is not None:
            raise ValueError(f"question {self.question_id}: mode ungrounded must not have a pack")
        if (self.evaluation_context is None) == (self.legacy_expectations is None):
            raise ValueError(
                f"question {self.question_id}: needs exactly one of evaluation_context and "
                "legacy_expectations"
            )
        context = self.evaluation_context
        if context is not None:
            spec_pack = context.reference_spec.pack_id if context.reference_spec else None
            if self.mode is GenerationMode.GROUNDED and spec_pack != self.reference_pack_id:
                raise ValueError(
                    f"question {self.question_id}: reference_spec {spec_pack} does not match "
                    f"reference pack {self.reference_pack_id}"
                )
            if self.mode is GenerationMode.UNGROUNDED and context.reference_spec is not None:
                raise ValueError(f"question {self.question_id}: ungrounded, no reference_spec")
        return self

    @property
    def is_legacy(self) -> bool:
        return self.evaluation_context is None

    @property
    def context(self) -> EvaluationContext:
        """The evaluation contract; ValueError for a legacy (display-only) question."""
        if self.evaluation_context is None:
            raise ValueError(LEGACY_QUESTION_MESSAGE.format(question_id=self.question_id))
        return self.evaluation_context

    @property
    def criteria_texts(self) -> list[str]:
        return (
            [c.requirement for c in self.evaluation_context.success_criteria]
            if (self.evaluation_context is not None)
            else []
        )


# --- LLM-facing drafts ---------------------------------------------------------------------------


class MetricExpectationDraft(StrictModel):
    status: MetricExpectationStatus
    value: dict[NonEmptyStr, JsonValue] | None = None
    reason: NonEmptyStr | None = None


class SuccessCriterionDraft(StrictModel):
    id: NonEmptyStr
    requirement: NonEmptyStr
    # Required by code when the run maps criteria to metrics (context.py).
    observed_in: EvidenceKind | None = None


class MetricMappingDraft(StrictModel):
    """The generator's proposal for one selected metric: the criterion ids it evaluates and,
    when the metric needs one, its question-specific expectation."""

    criteria: list[NonEmptyStr] = Field(default_factory=list)
    expectation: MetricExpectationDraft | None = None


class EvaluationContextDraft(StrictModel):
    success_criteria: list[SuccessCriterionDraft] = Field(
        min_length=1, max_length=MAX_SUCCESS_CRITERIA
    )
    metric_mapping: dict[NonEmptyStr, MetricMappingDraft] = Field(default_factory=dict)


class QuestionDraft(StrictModel):
    """LLM-facing question: no ids, no goal (copied from the scenario), no leak warnings, no
    Reference Spec (code reuses the scenario's pack)."""

    utterance: NonEmptyStr
    plan: QuestionPlan
    intent: QuestionIntent
    question_characteristics: QuestionDraftCharacteristics
    evaluation_context: EvaluationContextDraft


class QuestionBatchDraft(StrictModel):
    """Output contract for one question-generation call (one scenario)."""

    questions: list[QuestionDraft]


class RejectedQuestion(StrictModel):
    scenario_id: NonEmptyStr
    payload: dict[str, Any]
    reasons: list[NonEmptyStr] = Field(min_length=1)


class QuestionGenerationResult(StrictModel):
    questions: list[QuestionCase] = Field(default_factory=list)
    rejected: list[RejectedQuestion] = Field(default_factory=list)
