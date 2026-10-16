"""UI run workspace: inputs, UI state and the results table for one run.

Layout inside ``<artifacts_dir>/runs/<run_id>/``:
    inputs/persona.yaml, inputs/product.yaml, inputs/catalog.yaml   (uploaded or sample copy)
    inputs/persona_queries.json                                      (optional pasted queries)
    ui_state.json, results.csv, + the normal pipeline artifacts
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import InputAssetError
from synthetic_user.core.identifiers import new_run_id
from synthetic_user.core.loaders import parse_asset
from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.evaluation import EvaluationResult
from synthetic_user.models.execution import ExecutionResult
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase, RejectedQuestion
from synthetic_user.models.reference import ReferenceCatalog
from synthetic_user.models.scenario import RejectedScenario, Scenario
from synthetic_user.pipeline import RunInputs, Step, steps_from
from synthetic_user.product.contract import ProductContract
from synthetic_user.storage.json_store import JsonArtifactStore
from synthetic_user.storage.runs import run_artifact
from synthetic_user.web.results_csv import build_rows, to_csv

INPUT_KINDS: dict[str, tuple[type[BaseModel], str]] = {
    "persona": (Persona, "Persona file"),
    "product": (ProductDescription, "Product file"),
    "catalog": (ReferenceCatalog, "Catalog file"),
}
MAX_UPLOAD_BYTES = 1_000_000
MAX_PERSONA_QUERIES = 50
RESULTS_CSV = "results.csv"
UI_STATE = "ui_state.json"


class UiRunState(StrictModel):
    run_id: NonEmptyStr
    created_at: datetime
    sources: dict[str, str]  # kind -> "sample" | "uploaded: <filename>"
    persona_queries: list[str] = Field(default_factory=list)
    reference_index_id: str | None = None
    contract: ProductContract | None = None
    steps_done: list[Step] = Field(default_factory=list)


@dataclass(frozen=True)
class Upload:
    filename: str
    content: bytes


def parse_upload(kind: str, upload: Upload) -> BaseModel:
    """Validate an uploaded YAML file as persona / product / catalog."""
    if kind not in INPUT_KINDS:
        raise InputAssetError(f"Unknown input kind {kind!r}")
    if len(upload.content) > MAX_UPLOAD_BYTES:
        raise InputAssetError(f"{upload.filename}: larger than {MAX_UPLOAD_BYTES:,} bytes")
    try:
        text = upload.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputAssetError(f"{upload.filename}: not UTF-8 text") from exc
    model, label = INPUT_KINDS[kind]
    return parse_asset(text, model, kind=label, source=upload.filename)


def clean_queries(text: str) -> list[str]:
    """One query per line; blanks and duplicates dropped, order kept."""
    queries = list(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))
    if len(queries) > MAX_PERSONA_QUERIES:
        raise InputAssetError(f"At most {MAX_PERSONA_QUERIES} persona queries, got {len(queries)}")
    return queries


class Workspace:
    def __init__(self, settings: AppConfig, store: JsonArtifactStore) -> None:
        self._settings = settings
        self._store = store

    # --- samples / inputs ----------------------------------------------------------------

    def sample_path(self, kind: str) -> Path:
        paths = {
            "persona": self._settings.inputs.persona_path,
            "product": self._settings.inputs.product_path,
            "catalog": self._settings.inputs.catalog_path,
        }
        return paths[kind]

    def sample(self, kind: str) -> tuple[str, BaseModel]:
        path = self.sample_path(kind)
        if not path.is_file():
            raise InputAssetError(f"Sample {kind} not found: {path}")
        text = path.read_text(encoding="utf-8")
        model, label = INPUT_KINDS[kind]
        return text, parse_asset(text, model, kind=label, source=str(path))

    def create_run(self, uploads: dict[str, Upload | None], persona_queries: str) -> UiRunState:
        """Validate every input (uploaded or sample), then save them into a new run."""
        texts: dict[str, str] = {}
        sources: dict[str, str] = {}
        for kind in INPUT_KINDS:
            upload = uploads.get(kind)
            if upload is not None:
                parse_upload(kind, upload)
                texts[kind] = upload.content.decode("utf-8")
                sources[kind] = f"uploaded: {upload.filename}"
            else:
                texts[kind], _ = self.sample(kind)
                sources[kind] = "sample"
        queries = clean_queries(persona_queries)

        run_id = new_run_id()
        for kind, text in texts.items():
            self._store.write_text(self._input(run_id, f"{kind}.yaml"), text)
        self._store.write_with(
            self._input(run_id, "persona_queries.json"), queries, TypeAdapter(list[str])
        )
        state = UiRunState(
            run_id=run_id, created_at=datetime.now(UTC), sources=sources, persona_queries=queries
        )
        self.save_state(state)
        return state

    def load_inputs(self, run_id: str) -> RunInputs:
        def parse(kind: str) -> BaseModel:
            path = self._store.path_for(self._input(run_id, f"{kind}.yaml"))
            model, label = INPUT_KINDS[kind]
            return parse_asset(path.read_text(encoding="utf-8"), model, kind=label, source=kind)

        persona = parse("persona")
        product = parse("product")
        catalog = parse("catalog")
        assert isinstance(persona, Persona)
        assert isinstance(product, ProductDescription)
        assert isinstance(catalog, ReferenceCatalog)
        queries = self.state(run_id).persona_queries
        if queries:
            examples = list(dict.fromkeys([*persona.example_utterances, *queries]))
            persona = persona.model_copy(update={"example_utterances": examples})
        return RunInputs(persona=persona, product=product, catalog=catalog)

    # --- state -----------------------------------------------------------------------------

    def state(self, run_id: str) -> UiRunState:
        return self._store.read_model(run_artifact(run_id, UI_STATE), UiRunState)

    def save_state(self, state: UiRunState) -> None:
        self._store.write_model(run_artifact(state.run_id, UI_STATE), state)

    def mark_done(self, state: UiRunState, step: Step) -> UiRunState:
        """Record ``step``; later steps are no longer done (the pipeline cascades)."""
        later = steps_from(step) - {step}
        done = [s for s in state.steps_done if s not in later and s is not step]
        updated = state.model_copy(update={"steps_done": [*done, step]})
        self.save_state(updated)
        return updated

    # --- results ---------------------------------------------------------------------------

    def results(self, run_id: str) -> dict[str, Any]:
        """Current results table (+ rejected items), and refresh results.csv."""
        scenarios = self._read(run_id, "scenarios.json", list[Scenario]) or []
        questions = self._read(run_id, "questions.json", list[QuestionCase]) or []
        executions = self._read(run_id, "executions.json", list[ExecutionResult]) or []
        evaluations = self._read(run_id, "evaluations.json", list[EvaluationResult]) or []
        rows = build_rows(run_id, scenarios, questions, executions, evaluations)
        self._store.write_text(run_artifact(run_id, RESULTS_CSV), to_csv(rows))
        rejected_scenarios = self._read(run_id, "scenarios_rejected.json", list[RejectedScenario])
        rejected_questions = self._read(run_id, "questions_rejected.json", list[RejectedQuestion])
        return {
            "rows": rows,
            "rejected_scenarios": [r.model_dump(mode="json") for r in rejected_scenarios or []],
            "rejected_questions": [r.model_dump(mode="json") for r in rejected_questions or []],
            "artifacts": sorted(
                p.name
                for p in self._store.path_for(run_artifact(run_id, "")).iterdir()
                if p.is_file()
            ),
        }

    def artifact_path(self, run_id: str, name: str) -> Path:
        relative = run_artifact(run_id, name)
        if Path(name).name != name or not self._store.exists(relative):
            raise InputAssetError(f"No artifact {name!r} in run {run_id}")
        return self._store.path_for(relative)

    def _read(self, run_id: str, name: str, kind: Any) -> Any:
        relative = run_artifact(run_id, name)
        if not self._store.exists(relative):
            return None
        return self._store.read_with(relative, TypeAdapter(kind))

    @staticmethod
    def _input(run_id: str, name: str) -> Path:
        return run_artifact(run_id, f"inputs/{name}")
