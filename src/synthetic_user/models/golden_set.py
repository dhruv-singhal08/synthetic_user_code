"""SME-authored golden questions and their observable evaluation contract."""

from __future__ import annotations

from pydantic import Field

from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.question import (
    MAX_GOLDEN_SET_CRITERIA,
    MAX_REFERENCE_CONTEXT,
    MAX_SUCCESS_CRITERIA,
)


class GoldenCase(StrictModel):
    case_id: NonEmptyStr = Field(pattern=r"^GOLD_\d{3,}$")
    scenario_type: NonEmptyStr
    question: NonEmptyStr
    expected_journey: NonEmptyStr | None = None
    expected_time_period: NonEmptyStr | None = None
    expected_data_sources: list[NonEmptyStr] = Field(default_factory=list, max_length=8)
    expected_behavior: NonEmptyStr
    success_criteria: list[NonEmptyStr] = Field(min_length=1, max_length=MAX_SUCCESS_CRITERIA)
    expected_tools: list[NonEmptyStr] = Field(default_factory=list, max_length=12)
    expected_context: list[NonEmptyStr] = Field(default_factory=list, max_length=12)
    prohibited_behaviors: list[NonEmptyStr] = Field(default_factory=list, max_length=12)
    # Canonical tool/source definitions copied from the uploaded reference after cross-validation.
    # This is evaluator context, not another SME requirement.
    reference_context: list[NonEmptyStr] = Field(
        default_factory=list, max_length=MAX_REFERENCE_CONTEXT
    )
    # Exact non-empty criteria columns from the uploaded row. This preserves organization-
    # specific fields that the standard parser does not yet know by name.
    uploaded_criteria: dict[NonEmptyStr, NonEmptyStr] = Field(
        default_factory=dict, max_length=MAX_GOLDEN_SET_CRITERIA
    )
    is_follow_up: bool = False


class GoldenSet(StrictModel):
    source_name: NonEmptyStr
    cases: list[GoldenCase] = Field(min_length=1, max_length=500)


class GoldenSetPreview(StrictModel):
    source_name: NonEmptyStr
    cases: int
    case_ids: list[NonEmptyStr]
    fields_used: list[NonEmptyStr]
    uploaded_criteria_fields: list[NonEmptyStr]
    cases_with_expected_tools: int
    cases_missing_expected_tools: list[NonEmptyStr]
    cases_with_expected_data_sources: int
    cases_missing_expected_data_sources: list[NonEmptyStr]
