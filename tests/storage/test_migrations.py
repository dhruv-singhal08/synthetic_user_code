"""Legacy (profile-era) runs: readable as written; relabelled only by explicit request."""

from __future__ import annotations

from typing import Any

import pytest

from synthetic_user.core.exceptions import ArtifactIntegrityError
from synthetic_user.models.evaluation import EvaluationResult
from synthetic_user.storage.migrations import map_legacy_metric
from synthetic_user.web.workspace import UiRunState


def _legacy(**overrides: Any) -> EvaluationResult:
    return EvaluationResult.model_validate(
        {
            "evaluation_id": "EVAL_001",
            "question_id": "Q_001",
            "scenario_id": "SCN_001",
            "execution_id": "EXE_001",
            "product_outcome": "answered",
            "mode": "grounded",
            "evaluation_profile": "agent",
            "verdict": "fail",
            "evaluation_status": "evaluated",
            "metrics": {
                "relevance": {"status": "scored", "score": 1.0, "rationale": "ok"},
                "completion": {"status": "scored", "score": 0.5, "rationale": "partly"},
                "tool_selection": {"status": "scored", "score": 1.0, "rationale": "ok"},
            },
            "findings": [
                {
                    "finding_id": "F_001",
                    "category": "completion",
                    "description": "misses a part",
                    "expected_behavior": "e",
                    "observed_behavior": "o",
                    "severity": "high",
                    "evidence_status": "supported",
                    "source": "llm",
                }
            ],
            **overrides,
        }
    )


def test_completion_is_mapped_only_explicitly_and_marked() -> None:
    old = _legacy()
    assert "completion" in old.metrics  # reading never relabels
    (migrated,), changed = map_legacy_metric([old], "completion", "correctness")
    assert changed == 1
    assert set(migrated.metrics) == {"relevance", "correctness", "tool_selection"}
    assert migrated.metrics["correctness"].score == 0.5  # not re-evaluated
    assert migrated.findings[0].category == "correctness"
    assert "judged under the legacy completion rubric" in migrated.notes[-1]
    assert migrated.legacy_evaluation_profile == "agent" and migrated.schema_version == 1


def test_tool_selection_is_never_relabelled() -> None:
    with pytest.raises(ArtifactIntegrityError, match="expanded definition"):
        map_legacy_metric([_legacy()], "tool_selection", "tool_correctness")


def test_current_evaluations_are_never_rewritten() -> None:
    current = _legacy().model_copy(update={"schema_version": 2})
    with pytest.raises(ArtifactIntegrityError, match="only legacy"):
        map_legacy_metric([current], "completion", "correctness")


def test_profile_era_ui_state_reads_with_labels_only() -> None:
    state = UiRunState.model_validate(
        {
            "run_id": "RUN_20261001_120000_abcd",
            "created_at": "2026-10-01T12:00:00Z",
            "sources": {"persona": "sample"},
            "evaluation_profile": "agent",
            "evaluation_profile_selection": "auto",
            "evaluation_profile_reason": "auto",
            "agent_metrics": ["relevance", "completion", "context_recall"],
        }
    )
    assert state.metric_selection is None  # never a hidden profile fallback
    assert state.legacy_evaluation_profile == "agent"
    assert state.legacy_agent_metrics == ["relevance", "completion", "context_recall"]
