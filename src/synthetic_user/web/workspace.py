"""UI run workspace: inputs, UI state and the results table for one run.

Layout inside ``<artifacts_dir>/runs/<run_id>/``:
    inputs/persona.yaml, inputs/product.yaml      (uploaded or sample copy)
    inputs/reference_index.json                    (uploaded or sample copy; absent -> ungrounded)
    inputs/persona_queries.json                    (optional pasted queries)
    inputs/product_contract.yaml                   (uploaded product contract, when connected)
    ui_state.json, results.csv, + the normal pipeline artifacts
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from synthetic_user.adapters.contract import ProductContract
from synthetic_user.adapters.http_json import HttpJsonOptions
from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import InputAssetError, format_validation_error
from synthetic_user.core.identifiers import new_run_id
from synthetic_user.core.loaders import parse_asset, parse_yaml_mapping
from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.evaluation import EvaluationResult
from synthetic_user.models.execution import ExecutionRecord
from synthetic_user.models.persona import Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase, RejectedQuestion
from synthetic_user.models.reference import ReferenceIndex
from synthetic_user.models.scenario import RejectedScenario, Scenario
from synthetic_user.pipeline import RunInputs, Step, steps_from
from synthetic_user.references.loader import parse_reference_index
from synthetic_user.storage.json_store import JsonArtifactStore
from synthetic_user.storage.runs import run_artifact
from synthetic_user.web.results_csv import build_rows, to_csv

REFERENCE_INDEX = "reference_index"  # optional input kind (JSON); absent -> ungrounded
INPUT_KINDS: dict[str, tuple[type[BaseModel], str]] = {
    "persona": (Persona, "Persona file"),
    "product": (ProductDescription, "Product file"),
    REFERENCE_INDEX: (ReferenceIndex, "Reference index"),
}
MAX_UPLOAD_BYTES = 10_000_000  # full pre-built reference index JSONs are large
MAX_PERSONA_QUERIES = 50
RESULTS_CSV = "results.csv"
UI_STATE = "ui_state.json"
CONTRACT_FILE = "product_contract.yaml"


class ProductConnection(StrictModel):
    """A real product connection: product URL + uploaded contract (http_json, D180).

    Only a summary is kept here (header NAMES, never values); the contract itself stays in
    inputs/product_contract.yaml and is re-validated before every execute step."""

    product_url: NonEmptyStr
    contract_file: NonEmptyStr
    method: NonEmptyStr
    header_names: list[str] = Field(default_factory=list)
    timeout: float
    request_template: dict[str, Any] | list[Any]
    response_mapping: dict[str, Any]


class UiRunState(StrictModel):
    run_id: NonEmptyStr
    created_at: datetime
    sources: dict[str, str]  # kind -> "sample" | "uploaded: <filename>" | "none"
    persona_queries: list[str] = Field(default_factory=list)
    reference_index_id: str | None = None
    contract: ProductContract | None = None  # mock contract (D113)
    connection: ProductConnection | None = None  # real product: URL + uploaded contract
    steps_done: list[Step] = Field(default_factory=list)


@dataclass(frozen=True)
class Upload:
    filename: str
    content: bytes


def parse_upload(kind: str, upload: Upload) -> BaseModel:
    """Validate an uploaded file as persona / product (YAML) or reference index (JSON)."""
    if kind not in INPUT_KINDS:
        raise InputAssetError(f"Unknown input kind {kind!r}")
    if len(upload.content) > MAX_UPLOAD_BYTES:
        raise InputAssetError(f"{upload.filename}: larger than {MAX_UPLOAD_BYTES:,} bytes")
    try:
        text = upload.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputAssetError(f"{upload.filename}: not UTF-8 text") from exc
    return _parse(kind, text, upload.filename)


def _parse(kind: str, text: str, source: str) -> BaseModel:
    if kind == REFERENCE_INDEX:
        return parse_reference_index(text, source=source)
    model, label = INPUT_KINDS[kind]
    return parse_asset(text, model, kind=label, source=source)


def _file_name(kind: str) -> str:
    return f"{kind}.json" if kind == REFERENCE_INDEX else f"{kind}.yaml"


def parse_contract(text: str, url: str, *, source: str) -> HttpJsonOptions:
    """Validate a product contract (YAML or JSON): how to call the product and read its reply.

    Shape = the http_json adapter options (method, headers, timeout, request_template,
    response_mapping), optionally wrapped as ``{type: http_json, options: {...}}`` like a
    ``*.adapter.yaml`` file. ``url`` (the UI field) wins over a ``url`` in the contract.
    """
    data = parse_yaml_mapping(text, InputAssetError, kind="Product contract", source=source)
    if "options" in data:
        extra = set(data) - {"type", "options"}
        if extra or data.get("type", "http_json") != "http_json":
            raise InputAssetError(
                f"Product contract {source}: a wrapped contract must be "
                f"{{type: http_json, options: {{...}}}} (got keys {sorted(data)})"
            )
        data = data["options"] if isinstance(data["options"], dict) else {}
    product_url = url.strip() or str(data.get("url") or "").strip()
    if not product_url:
        raise InputAssetError("Enter the product URL (or set url in the contract)")
    try:
        return HttpJsonOptions.model_validate({**data, "url": product_url})
    except ValidationError as exc:
        raise InputAssetError(f"Product contract {source}: {format_validation_error(exc)}") from exc


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

    def sample_path(self, kind: str) -> Path | None:
        paths = {
            "persona": self._settings.inputs.persona_path,
            "product": self._settings.inputs.product_path,
            REFERENCE_INDEX: self._settings.inputs.reference_index_file,
        }
        return paths[kind]

    def sample(self, kind: str) -> tuple[str, BaseModel] | None:
        """(text, parsed) of the configured sample; None for an unconfigured reference index."""
        path = self.sample_path(kind)
        if path is None:
            return None
        if not path.is_file():
            raise InputAssetError(f"Sample {kind} not found: {path}")
        text = path.read_text(encoding="utf-8")
        return text, _parse(kind, text, str(path))

    def create_run(
        self,
        uploads: dict[str, Upload | None],
        persona_queries: str,
        *,
        use_reference_index: bool = False,
    ) -> UiRunState:
        """Validate every input, then save them into a new run.

        Persona: upload or the configured sample. Product description: upload only (no
        sample). Reference index: only when switched on, and then an upload is required;
        switched off -> ungrounded run.
        """
        texts: dict[str, str] = {}
        sources: dict[str, str] = {}
        for kind in INPUT_KINDS:
            upload = uploads.get(kind)
            if kind == REFERENCE_INDEX and not use_reference_index:
                sources[kind] = "off"
                continue
            if upload is not None:
                parse_upload(kind, upload)
                texts[kind] = upload.content.decode("utf-8")
                sources[kind] = f"uploaded: {upload.filename}"
            elif kind == "persona" and (sample := self.sample(kind)) is not None:
                texts[kind] = sample[0]
                sources[kind] = "sample"
            elif kind == "product":
                raise InputAssetError("Upload a product description (YAML); no sample is used")
            else:
                raise InputAssetError(
                    "Reference index is switched on: upload the pre-built index JSON, or switch "
                    "it off for an ungrounded run"
                )
        queries = clean_queries(persona_queries)

        run_id = new_run_id()
        for kind, text in texts.items():
            self._store.write_text(self._input(run_id, _file_name(kind)), text)
        self._store.write_with(
            self._input(run_id, "persona_queries.json"), queries, TypeAdapter(list[str])
        )
        state = UiRunState(
            run_id=run_id, created_at=datetime.now(UTC), sources=sources, persona_queries=queries
        )
        self.save_state(state)
        return state

    def load_inputs(self, run_id: str) -> RunInputs:
        def parse(kind: str) -> BaseModel | None:
            path = self._store.path_for(self._input(run_id, _file_name(kind)))
            if kind == REFERENCE_INDEX and not path.is_file():
                return None
            return _parse(kind, path.read_text(encoding="utf-8"), kind)

        persona = parse("persona")
        product = parse("product")
        index = parse(REFERENCE_INDEX)
        assert isinstance(persona, Persona)
        assert index is None or isinstance(index, ReferenceIndex)
        assert isinstance(product, ProductDescription)
        queries = self.state(run_id).persona_queries
        if queries:
            examples = list(dict.fromkeys([*persona.example_utterances, *queries]))
            persona = persona.model_copy(update={"example_utterances": examples})
        return RunInputs(persona=persona, product=product, reference_index=index)

    # --- product connection ----------------------------------------------------------------

    def connect(self, state: UiRunState, url: str, upload: Upload) -> UiRunState:
        """Validate URL + contract, save the contract and switch the run to the real product
        (the mock contract, if any, is dropped)."""
        if len(upload.content) > MAX_UPLOAD_BYTES:
            raise InputAssetError(f"{upload.filename}: larger than {MAX_UPLOAD_BYTES:,} bytes")
        try:
            text = upload.content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InputAssetError(f"{upload.filename}: not UTF-8 text") from exc
        options = parse_contract(text, url, source=upload.filename)
        self._store.write_text(self._input(state.run_id, CONTRACT_FILE), text)
        connection = ProductConnection(
            product_url=options.url,
            contract_file=upload.filename,
            method=options.method,
            header_names=sorted(options.headers),
            timeout=options.timeout,
            request_template=options.request_template,
            response_mapping=options.response_mapping.model_dump(mode="json", exclude_none=True),
        )
        updated = state.model_copy(update={"connection": connection, "contract": None})
        self.save_state(updated)
        return updated

    def connection_options(self, state: UiRunState) -> HttpJsonOptions | None:
        if state.connection is None:
            return None
        path = self._store.path_for(self._input(state.run_id, CONTRACT_FILE))
        return parse_contract(
            path.read_text(encoding="utf-8"), state.connection.product_url, source=CONTRACT_FILE
        )

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
        executions = self._read(run_id, "executions.json", list[ExecutionRecord]) or []
        evaluations = self._read(run_id, "evaluations.json", list[EvaluationResult]) or []
        rows = build_rows(run_id, scenarios, questions, executions, evaluations)
        self._store.write_text(run_artifact(run_id, RESULTS_CSV), to_csv(rows))
        details = {
            "scenarios": [s.model_dump(mode="json") for s in scenarios],
            "evaluations": [e.model_dump(mode="json") for e in evaluations],
        }
        rejected_scenarios = self._read(run_id, "scenarios_rejected.json", list[RejectedScenario])
        rejected_questions = self._read(run_id, "questions_rejected.json", list[RejectedQuestion])
        return {
            "rows": rows,
            **details,
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
