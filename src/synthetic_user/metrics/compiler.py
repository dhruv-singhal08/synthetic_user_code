"""Metric compiler: ONE structured LLM call per run that has custom metrics; none otherwise.

Code assigns ids (``custom_01``, ``custom_02``, ... in input order), keeps the user's exact
text and provenance, and validates the reply structurally. A metric the model marks
uncompilable, or the whole call failing, yields ``unavailable`` definitions with the reason:
nothing is dropped and the built-in metrics are never blocked. No semantic merging,
de-duplication or conflict judgement happens anywhere.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal, Self

from pydantic import Field, model_validator

from synthetic_user.core.exceptions import LLMOutputValidationError, LLMProviderError
from synthetic_user.core.trace import trace_subject
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.structured import generate_structured
from synthetic_user.metrics.compiler_prompt import UNIVERSAL_QUESTION_FIELDS, build_compiler_prompt
from synthetic_user.models.base import NonEmptyStr, StrictModel, find_duplicates
from synthetic_user.models.metrics import (
    EXPECTATION_FIELD_PATTERN,
    CustomMetricInput,
    EvidenceKind,
    ExpectationField,
    ExpectationFieldType,
    JudgeGroup,
    MetricAvailability,
    MetricCategory,
    MetricDefinition,
    MetricOrigin,
    ReferenceAsset,
)
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.reference import Reference

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]

COMPILER_VERSION = "metric-compiler-1"
CUSTOM_VERSION = "1.0"
_RETRIEVAL_EVIDENCE = frozenset(
    {EvidenceKind.RETRIEVED_CONTEXT, EvidenceKind.CITATIONS, EvidenceKind.GROUNDING_EVIDENCE}
)


def custom_metric_id(index: int) -> str:
    return f"custom_{index:02d}"


class CompiledFieldDraft(StrictModel):
    name: NonEmptyStr = Field(pattern=EXPECTATION_FIELD_PATTERN)
    type: ExpectationFieldType
    description: NonEmptyStr
    required: bool = True


class CompiledMetricDraft(StrictModel):
    input_index: int = Field(ge=1)
    compilable: bool
    uncompilable_reason: NonEmptyStr | None = None
    display_name: NonEmptyStr | None = None
    definition: NonEmptyStr | None = None
    rubric: NonEmptyStr | None = None
    score_anchors: dict[NonEmptyStr, NonEmptyStr] | None = None
    evidence_requirements: NonEmptyStr | None = None
    applicability: NonEmptyStr | None = None
    judge_group: Literal["general", "retrieval_grounding"] | None = None
    required_execution_evidence: list[EvidenceKind] = Field(default_factory=list)
    required_reference_assets: list[ReferenceAsset] = Field(default_factory=list)
    needs_question_expectations: bool = False
    expectation_fields: list[CompiledFieldDraft] = Field(default_factory=list)
    generation_instructions: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _complete(self) -> Self:
        if not self.compilable:
            if self.uncompilable_reason is None:
                raise ValueError("an uncompilable metric needs uncompilable_reason")
            return self
        missing = [
            name
            for name in (
                "display_name",
                "definition",
                "rubric",
                "score_anchors",
                "evidence_requirements",
                "applicability",
                "judge_group",
            )
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(f"a compilable metric needs {missing}")
        if self.needs_question_expectations != bool(self.expectation_fields):
            raise ValueError("needs_question_expectations is true exactly when fields are listed")
        if self.expectation_fields and self.generation_instructions is None:
            raise ValueError("expectation fields need generation_instructions")
        return self


class CompilerOutput(StrictModel):
    metrics: list[CompiledMetricDraft]


def check_compiler_output(output: CompilerOutput, count: int) -> list[str]:
    """Structural checks only (indices, ids, fields, routing). Never semantic."""
    problems: list[str] = []
    indices = [m.input_index for m in output.metrics]
    if sorted(indices) != list(range(1, count + 1)):
        problems.append(
            f"metrics must contain input_index 1..{count} exactly once each, got {sorted(indices)}"
        )
    for metric in output.metrics:
        label = f"metric {metric.input_index}"
        if not metric.compilable:
            continue
        names = [f.name for f in metric.expectation_fields]
        if dupes := find_duplicates(names):
            problems.append(f"{label}: duplicate expectation fields {dupes}")
        if clash := sorted(set(names) & set(UNIVERSAL_QUESTION_FIELDS)):
            problems.append(f"{label}: expectation fields redefine universal fields {clash}")
        anchors = metric.score_anchors or {}
        if missing := sorted({"1", "0.5", "0"} - set(anchors)):
            problems.append(f"{label}: score_anchors miss {missing}")
        if metric.judge_group == "retrieval_grounding" and not (
            set(metric.required_execution_evidence) & _RETRIEVAL_EVIDENCE
        ):
            problems.append(
                f"{label}: retrieval_grounding needs retrieved_context, citations or "
                "grounding_evidence in required_execution_evidence"
            )
    return problems


class MetricCompiler:
    def __init__(self, llm: LLMClient, logger: Logger, *, max_retries: int) -> None:
        self._llm = llm
        self._log = logger
        self._max_retries = max_retries

    async def compile(
        self,
        inputs: Sequence[CustomMetricInput],
        product: ProductDescription,
        reference: Reference | None,
        *,
        sql_dialect: str | None,
    ) -> tuple[list[MetricDefinition], list[str]]:
        """(definitions in input order, compiler notes). Exactly one LLM call (+ bounded
        validation retries); never called with an empty input list."""
        if not inputs:
            raise ValueError("the metric compiler needs at least one custom metric")
        system, user = build_compiler_prompt(inputs, product, reference, sql_dialect=sql_dialect)
        try:
            with trace_subject("metric compiler"):
                output = await generate_structured(
                    self._llm,
                    system,
                    user,
                    CompilerOutput,
                    max_retries=self._max_retries,
                    post_validate=lambda out: check_compiler_output(out, len(inputs)),
                )
        except (LLMOutputValidationError, LLMProviderError) as exc:
            errors = getattr(exc, "errors", None) or [str(exc)]
            reason = "metric compiler failed: " + "; ".join(errors)
            self._log.warning("Custom metric compilation failed: %s", reason)
            return [_unavailable(i, item, reason) for i, item in enumerate(inputs, 1)], [reason]
        by_index = {m.input_index: m for m in output.metrics}
        definitions = [
            _definition(index, item, by_index[index]) for index, item in enumerate(inputs, 1)
        ]
        unavailable = sum(1 for d in definitions if not d.available)
        self._log.info(
            "Compiled %d custom metric(s): %d available, %d unavailable",
            len(definitions),
            len(definitions) - unavailable,
            unavailable,
        )
        return definitions, []


def _definition(
    index: int, item: CustomMetricInput, draft: CompiledMetricDraft
) -> MetricDefinition:
    if not draft.compilable:
        assert draft.uncompilable_reason is not None
        return _unavailable(index, item, draft.uncompilable_reason)
    assert draft.display_name and draft.definition and draft.rubric and draft.score_anchors
    assert draft.evidence_requirements and draft.applicability and draft.judge_group
    return MetricDefinition(
        metric_id=custom_metric_id(index),
        display_name=draft.display_name,
        category=MetricCategory.CUSTOM,
        version=CUSTOM_VERSION,
        origin=MetricOrigin.CUSTOM,
        description=draft.definition,
        rubric=draft.rubric,
        score_anchors=draft.score_anchors,
        judge_group=JudgeGroup(draft.judge_group),
        question_expectation_schema=[
            ExpectationField.model_validate(f.model_dump()) for f in draft.expectation_fields
        ],
        generation_instructions=draft.generation_instructions,
        required_reference_assets=list(dict.fromkeys(draft.required_reference_assets)),
        required_execution_evidence=_evidence(draft.required_execution_evidence),
        applicability_policy=draft.applicability,
        evidence_requirements=draft.evidence_requirements,
        source_text=item.text,
        source_provenance=item.provenance,
    )


def _evidence(listed: Sequence[EvidenceKind]) -> list[EvidenceKind]:
    """The compiled evidence, with ``answer`` added unless the metric judges an SQL artifact
    only: an SQL artifact already implies an answered response, and keeping ``sql`` alone is
    what routes the metric as SQL-only (D234). Listed ``answer`` is always kept."""
    if EvidenceKind.SQL in listed:
        return list(dict.fromkeys(listed))
    return list(dict.fromkeys([EvidenceKind.ANSWER, *listed]))


def _unavailable(index: int, item: CustomMetricInput, reason: str) -> MetricDefinition:
    """A custom metric that cannot be evaluated: kept with the user's text and the reason."""
    return MetricDefinition(
        metric_id=custom_metric_id(index),
        display_name=f"Custom metric {index}",
        category=MetricCategory.CUSTOM,
        version=CUSTOM_VERSION,
        origin=MetricOrigin.CUSTOM,
        description=item.text,
        rubric=item.text,
        score_anchors={"1": "not compiled", "0.5": "not compiled", "0": "not compiled"},
        judge_group=JudgeGroup.GENERAL,
        applicability_policy="not evaluated: the metric could not be compiled",
        evidence_requirements="not evaluated: the metric could not be compiled",
        availability=MetricAvailability.UNAVAILABLE,
        unavailable_reason=reason,
        source_text=item.text,
        source_provenance=item.provenance,
    )
