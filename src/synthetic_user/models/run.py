"""Run manifest: what a run used and which artifacts it produced."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import Field, model_validator

from synthetic_user.models.base import NonEmptyStr, StrictModel

_SECRET_KEY_MARKERS = ("secret", "api_key", "password")


class GenerationMode(StrEnum):
    """grounded: a reference file was given; ungrounded: persona + product only."""

    GROUNDED = "grounded"
    UNGROUNDED = "ungrounded"


def mode_mismatch(mode: GenerationMode, has_reference: bool, what: str) -> str | None:
    """Problem text when ``what`` is present in ungrounded mode or missing in grounded mode."""
    if mode is GenerationMode.GROUNDED and not has_reference:
        return f"mode grounded requires {what}"
    if mode is GenerationMode.UNGROUNDED and has_reference:
        return f"mode ungrounded must not have {what}"
    return None


class GuidanceType(StrEnum):
    """What a guidance item does to an evaluation decision (reforming, Part 5)."""

    EVALUATION_REQUIREMENT = "evaluation_requirement"  # an extra mandatory check
    INTERPRETATION_RULE = "interpretation_rule"  # how to read a metric, a term or the reference
    EXCEPTION = "exception"  # an approved difference that is never a defect
    SCORING_POLICY = "scoring_policy"  # how a metric is scored
    EVIDENCE_POLICY = "evidence_policy"  # what evidence suffices / what cannot be verified


GUIDANCE_ID_PATTERN = r"^G\d+$"


class GuidanceItem(StrictModel):
    """One item of the user's evaluation guidance: an exact, verbatim excerpt of the text the
    user gave (never rewritten), classified by one structuring call when the guidance is saved.
    ``metrics`` = the metric ids it concerns; [] = every metric."""

    guidance_id: NonEmptyStr = Field(pattern=GUIDANCE_ID_PATTERN)
    type: GuidanceType
    metrics: list[NonEmptyStr] = Field(default_factory=list)
    text: NonEmptyStr
    # Where the excerpt is in the guidance: "<source> line a-b" (code computes it).
    location: NonEmptyStr


class EvaluationGuidance(StrictModel):
    """Optional free-text evaluation guidance of a run (typed or a .txt / .md / .yaml file),
    kept as the plain text the user gave. Read only by the evaluation judges (D237). ``items``
    is its structured form (one LLM call per guidance version, every line covered verbatim);
    empty until it was structured."""

    text: NonEmptyStr
    source: NonEmptyStr  # "typed in the UI" or the uploaded / configured file name
    sha256: NonEmptyStr = Field(pattern=r"^[0-9a-f]{64}$")
    items: list[GuidanceItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [item.guidance_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError(f"guidance item ids must be unique: {ids}")
        return self

    def item(self, guidance_id: str) -> GuidanceItem | None:
        return next((i for i in self.items if i.guidance_id == guidance_id), None)


class RunManifest(StrictModel):
    run_id: NonEmptyStr = Field(pattern=r"^RUN_\d{8}_\d{6}_[0-9a-f]{4}$")
    created_at: datetime
    persona_id: NonEmptyStr
    product_id: NonEmptyStr
    mode: GenerationMode
    reference_id: NonEmptyStr | None = None
    # The frozen metric configuration (metric_config.json) and the selection it came from.
    # None = a run created before run-level metric selection (legacy, read-only for evaluation).
    metric_config_id: NonEmptyStr | None = None
    metric_selection_fingerprint: NonEmptyStr | None = None
    # Profile of a pre-registry run, kept as a label only; never used for routing.
    legacy_evaluation_profile: NonEmptyStr | None = None
    llm_provider: NonEmptyStr
    llm_model: NonEmptyStr
    pipeline_version: NonEmptyStr
    effective_config: dict[str, Any]
    artifacts: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    # The evaluation guidance the stored evaluations were judged with (None = none given).
    evaluation_guidance: EvaluationGuidance | None = None

    @model_validator(mode="before")
    @classmethod
    def _read_legacy_profile(cls, value: Any) -> Any:
        if isinstance(value, dict) and "evaluation_profile" in value:
            data = dict(value)
            profile = data.pop("evaluation_profile")
            if profile is not None:
                data.setdefault("legacy_evaluation_profile", str(profile))
            return data
        return value

    @model_validator(mode="after")
    def _reference_matches_mode(self) -> Self:
        problem = mode_mismatch(self.mode, self.reference_id is not None, "a reference")
        if problem:
            raise ValueError(problem)
        return self

    @model_validator(mode="after")
    def _no_secrets_in_config(self) -> Self:
        leaked = _secret_keys(self.effective_config)
        if leaked:
            raise ValueError(f"effective_config must not contain secrets: {leaked}")
        return self


def _secret_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}{key}"
            if any(marker in str(key).lower() for marker in _SECRET_KEY_MARKERS):
                found.append(path)
            found.extend(_secret_keys(child, f"{path}."))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            found.extend(_secret_keys(child, f"{prefix}{i}."))
    return found
