"""UI run workspace: inputs, UI state and the results table for one run.

Layout inside ``<artifacts_dir>/runs/<run_id>/``:
    inputs/persona.yaml, inputs/product.yaml      (uploaded or sample copy)
    inputs/reference/<file>                        (uploaded .json/.txt/.md; absent -> ungrounded)
    inputs/reference_guide/<file>                  (optional uploaded usage guide .md/.txt)
    inputs/custom_metrics/<file>                   (optional custom-metric workbook .xlsx/.csv)
    inputs/evaluation_guidance/<file>              (optional evaluation guidance .txt/.md/.yaml)
    inputs/persona_queries.json                    (optional pasted queries)
    inputs/product_contract.yaml                   (uploaded product contract, when connected)
    ui_state.json, results.csv, + the normal pipeline artifacts
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator

from synthetic_user.adapters.contract import ProductContract
from synthetic_user.adapters.http_common import HttpOptionsBase
from synthetic_user.adapters.http_json import HttpJsonOptions
from synthetic_user.adapters.http_sse import HttpSseOptions
from synthetic_user.core.config import AppConfig, ProductAdapterType
from synthetic_user.core.exceptions import (
    ArtifactNotFoundError,
    InputAssetError,
    format_validation_error,
)
from synthetic_user.core.guidance import TYPED_GUIDANCE, parse_evaluation_guidance
from synthetic_user.core.identifiers import RUN_ID_PATTERN, new_run_id
from synthetic_user.core.loaders import parse_asset, parse_yaml_mapping
from synthetic_user.core.spreadsheet import SPREADSHEET_SUFFIXES
from synthetic_user.metrics.custom_input import (
    CustomInputLimits,
    ParsedCustomInput,
    merge,
    parse_free_text,
    parse_workbook,
)
from synthetic_user.metrics.registry import validate_selected_ids
from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.evaluation import EvaluationResult
from synthetic_user.models.execution import ContextSourceMode, ExecutionRecord
from synthetic_user.models.kpi import RunKpis
from synthetic_user.models.metrics import (
    CustomMetricInput,
    CustomMetricInputError,
    MetricSelection,
    ResolvedMetricConfig,
)
from synthetic_user.models.persona import ComplexityPreference, Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase, RejectedQuestion
from synthetic_user.models.reference import DataCatalog, Reference, ReferenceKind, ReferencePack
from synthetic_user.models.run import EvaluationGuidance
from synthetic_user.models.scenario import RejectedScenario, Scenario
from synthetic_user.nodes.evaluator.kpis import consolidate_findings
from synthetic_user.nodes.evaluator.no_sql import NO_SQL_REASON, no_sql_execution, run_expects_sql
from synthetic_user.nodes.evaluator.trace_tags import attach_trace_links
from synthetic_user.pipeline import STRUCTURED_GUIDANCE, RunInputs, Step, steps_from
from synthetic_user.references.loader import GUIDE_SUFFIXES, parse_reference
from synthetic_user.storage.json_store import JsonArtifactStore
from synthetic_user.storage.runs import RUNS_DIR, run_artifact
from synthetic_user.web.results_csv import build_rows, columns_for, to_csv

REFERENCE = "reference"  # data_catalog / document_collection (.yaml/.json); absent -> ungrounded
GUIDE = "reference_guide"  # optional usage instructions (.md/.txt upload, or typed in the UI)
CUSTOM_METRICS = "custom_metrics"  # optional custom-metric workbook (.xlsx / .csv)
EVALUATION_GUIDANCE = "evaluation_guidance"  # optional guidance for the judges (.txt/.md/.yaml)
TYPED_GUIDANCE_FILE = "guidance.md"  # file name used for guidance typed in the UI
TYPED_INSTRUCTIONS = "instructions.md"  # file name used for instructions typed in the UI
YAML_KINDS: dict[str, tuple[type[BaseModel], str]] = {
    "persona": (Persona, "Persona file"),
    "product": (ProductDescription, "Product file"),
}
INPUT_KINDS = (*YAML_KINDS, REFERENCE, GUIDE, CUSTOM_METRICS)
RESULTS_CSV = "results.csv"
UI_STATE = "ui_state.json"
CONTRACT_FILE = "product_contract.yaml"


class ProductConnection(StrictModel):
    """A real product connection: product URL + uploaded HTTP adapter contract.

    Only a summary is kept here (header NAMES, never values); the contract itself stays in
    inputs/product_contract.yaml and is re-validated before every execute step."""

    product_url: NonEmptyStr
    adapter_type: ProductAdapterType = ProductAdapterType.HTTP_JSON
    contract_file: NonEmptyStr
    method: NonEmptyStr
    header_names: list[str] = Field(default_factory=list)
    timeout: float
    request_template: dict[str, Any] | list[Any]
    response_mapping: dict[str, Any]
    stream: dict[str, Any] | None = None
    # Where retrieved context comes from (chosen in the connection UI, checked against the
    # contract): the product response itself, a separate context API, or none.
    context_source: ContextSourceMode = ContextSourceMode.NONE
    context_api_url: str | None = None


HttpConnectionOptions = HttpJsonOptions | HttpSseOptions


@dataclass(frozen=True)
class ParsedHttpConnection:
    adapter_type: ProductAdapterType
    options: HttpConnectionOptions


class UiRunState(StrictModel):
    run_id: NonEmptyStr
    created_at: datetime
    sources: dict[str, str]  # kind -> "sample" | "uploaded: <filename>" | "none"
    persona_queries: list[str] = Field(default_factory=list)
    # A UI-selected, per-run override. None preserves the persona file's preference for
    # backwards compatibility with runs created before this control existed.
    question_complexity: ComplexityPreference | None = None
    # Run-level metric selection from the Evaluation tab (frozen at the scenarios step).
    # None only for runs created before metric selection existed.
    metric_selection: MetricSelection | None = None
    # The Evaluation tab (metrics, custom metrics, counts, complexity) is a draft until the user
    # locks it; scenarios need it locked. Runs saved before this field were locked at creation.
    evaluation_locked: bool = True
    scenario_count: int | None = Field(default=None, ge=1)
    question_count: int | None = Field(default=None, ge=1)
    # Profile-era runs: kept as labels only, never used for routing.
    legacy_evaluation_profile: str | None = None
    legacy_agent_metrics: list[str] = Field(default_factory=list)
    reference_id: str | None = None
    contract: ProductContract | None = None  # mock contract (D113)
    connection: ProductConnection | None = None  # real product: URL + uploaded contract
    steps_done: list[Step] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _read_legacy(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "evaluation_profile" not in value:
            return value
        data = dict(value)
        profile = data.pop("evaluation_profile", None)
        data.pop("evaluation_profile_selection", None)
        data.pop("evaluation_profile_reason", None)
        metrics = data.pop("agent_metrics", None) or []
        data["legacy_evaluation_profile"] = str(profile) if profile else None
        data["legacy_agent_metrics"] = [str(m) for m in metrics] if profile == "agent" else []
        return data


@dataclass(frozen=True)
class Upload:
    filename: str
    content: bytes


class GuidePreview(StrictModel):
    name: NonEmptyStr
    chars: int


class CustomMetricsPreview(StrictModel):
    source_name: NonEmptyStr
    metrics: list[CustomMetricInput]
    errors: list[CustomMetricInputError]


class ReferencePreview(StrictModel):
    reference_id: NonEmptyStr
    source_name: NonEmptyStr
    kind: ReferenceKind
    name: NonEmptyStr
    chars: int  # as rendered for prompts
    tables: int | None = None
    relationships: int | None = None
    rules: int | None = None
    documents: int | None = None
    agent_tools: int | None = None
    agent_data_sources: int | None = None
    agent_workflow_stages: int | None = None


def reference_preview(reference: Reference) -> ReferencePreview:
    content = reference.content
    counts: dict[str, int] = (
        {
            "tables": len(content.tables),
            "relationships": len(content.relationships),
            "rules": len(content.rules),
        }
        if isinstance(content, DataCatalog)
        else {"documents": len(content.documents)}
    )
    agent = reference.agent_context
    return ReferencePreview(
        reference_id=reference.reference_id,
        source_name=reference.source_name,
        kind=reference.kind,
        name=content.name,
        chars=reference.rendered_chars(),
        agent_tools=len(agent.tools) if agent is not None else None,
        agent_data_sources=len(agent.data_sources) if agent is not None else None,
        agent_workflow_stages=len(agent.workflow) if agent is not None else None,
        **counts,
    )


def custom_input_limits(settings: AppConfig) -> CustomInputLimits:
    return CustomInputLimits(
        max_metrics=settings.evaluation.max_custom_metrics,
        max_chars_per_metric=settings.evaluation.max_custom_metric_chars,
        max_rows=settings.evaluation.max_custom_metric_rows,
        max_uncompressed_bytes=settings.web.max_upload_bytes * 10,
    )


def parse_upload(
    kind: str,
    upload: Upload,
    *,
    max_chars: int,
    max_upload_bytes: int,
    limits: CustomInputLimits | None = None,
) -> BaseModel:
    """Validate persona/product YAML, reference YAML/JSON, instructions or a metric workbook."""
    if kind not in INPUT_KINDS:
        raise InputAssetError(f"Unknown input kind {kind!r}")
    if len(upload.content) > max_upload_bytes:
        raise InputAssetError(f"{upload.filename}: larger than {max_upload_bytes:,} bytes")
    if kind == CUSTOM_METRICS:
        if limits is None:
            raise InputAssetError("custom metric limits are required")
        parsed = parse_workbook(upload.content, upload.filename, limits)
        return CustomMetricsPreview(
            source_name=upload.filename, metrics=parsed.inputs, errors=parsed.errors
        )
    if kind == REFERENCE:
        return reference_preview(
            parse_reference(upload.content, source_name=upload.filename, max_chars=max_chars)
        )
    text = _decode(upload)
    if kind == GUIDE:
        if Path(upload.filename).suffix.lower() not in GUIDE_SUFFIXES:
            raise InputAssetError(f"Usage instructions {upload.filename}: use a .md or .txt file")
        return GuidePreview(name=upload.filename, chars=len(text))
    model, label = YAML_KINDS[kind]
    return parse_asset(text, model, kind=label, source=upload.filename)


def _decode(upload: Upload) -> str:
    try:
        return upload.content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputAssetError(f"{upload.filename}: not UTF-8 text") from exc


def parse_contract(text: str, url: str, *, source: str) -> ParsedHttpConnection:
    """Validate a product contract (YAML or JSON): how to call the product and read its reply.

    Shape = http_json/http_sse adapter options, optionally wrapped as
    ``{type: http_sse, options: {...}}`` like a ``*.adapter.yaml`` file. ``url`` (the UI field)
    wins over a ``url`` in the contract.
    """
    data = parse_yaml_mapping(text, InputAssetError, kind="Product contract", source=source)
    adapter_type = ProductAdapterType.HTTP_JSON
    if "options" in data:
        extra = set(data) - {"type", "options"}
        raw_type = data.get("type", ProductAdapterType.HTTP_JSON.value)
        try:
            adapter_type = ProductAdapterType(raw_type)
        except ValueError as exc:
            raise InputAssetError(
                f"Product contract {source}: type must be http_json or http_sse"
            ) from exc
        if extra or adapter_type not in (
            ProductAdapterType.HTTP_JSON,
            ProductAdapterType.HTTP_SSE,
        ):
            raise InputAssetError(
                f"Product contract {source}: a wrapped contract must be "
                f"{{type: http_json|http_sse, options: {{...}}}} (got keys {sorted(data)})"
            )
        data = data["options"] if isinstance(data["options"], dict) else {}
    product_url = url.strip() or str(data.get("url") or "").strip()
    if not product_url:
        raise InputAssetError("Enter the product URL (or set url in the contract)")
    model = HttpSseOptions if adapter_type is ProductAdapterType.HTTP_SSE else HttpJsonOptions
    try:
        options = model.model_validate({**data, "url": product_url})
    except ValidationError as exc:
        raise InputAssetError(f"Product contract {source}: {format_validation_error(exc)}") from exc
    return ParsedHttpConnection(adapter_type, options)


def check_context_source(options: HttpOptionsBase, chosen: ContextSourceMode | None) -> None:
    """The UI's context-source choice must match the contract (never guessed or mixed)."""
    if chosen is None or chosen is options.context_source:
        return
    hints = {
        ContextSourceMode.RESPONSE_FIELD: "add response_mapping.context (path to the retrieved "
        "chunks in the product response)",
        ContextSourceMode.CONTEXT_API: "add a context_api section (endpoint, headers and the "
        "correlation mapping for the same execution)",
        ContextSourceMode.NONE: "remove the context mapping from the contract",
    }
    raise InputAssetError(
        f"Context source '{chosen.value}' was selected but the contract configures "
        f"'{options.context_source.value}': {hints[chosen]}"
    )


def clean_queries(text: str, *, max_queries: int) -> list[str]:
    """One query per line; blanks and duplicates dropped, order kept."""
    queries = list(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))
    if len(queries) > max_queries:
        raise InputAssetError(f"At most {max_queries} persona queries, got {len(queries)}")
    return queries


class Workspace:
    def __init__(self, settings: AppConfig, store: JsonArtifactStore) -> None:
        self._settings = settings
        self._store = store

    # --- samples / inputs ----------------------------------------------------------------

    def sample(self, kind: str) -> tuple[str, BaseModel] | None:
        """(text, parsed preview) of a configured persona, product or reference, if any."""
        paths = {
            "persona": self._settings.inputs.persona_path,
            "product": self._settings.inputs.product_path,
            REFERENCE: self._settings.inputs.reference_file,
        }
        path = paths.get(kind)
        if path is None:
            return None
        if not path.is_file():
            raise InputAssetError(f"Sample {kind} not found: {path}")
        text = path.read_text(encoding="utf-8")
        if kind == REFERENCE:
            parsed = parse_upload(
                REFERENCE,
                Upload(path.name, text.encode("utf-8")),
                max_chars=self._settings.reference.max_chars,
                max_upload_bytes=self._settings.web.max_upload_bytes,
            )
            return text, parsed
        model, label = YAML_KINDS[kind]
        return text, parse_asset(text, model, kind=label, source=str(path))

    def create_run(
        self,
        uploads: dict[str, Upload | None],
        persona_queries: str,
        *,
        question_complexity: ComplexityPreference | None = None,
        selected_metrics: list[str] | None = None,
        custom_metrics_text: str = "",
        scenario_count: int | None = None,
        question_count: int | None = None,
        use_reference: bool = False,
        reference_instructions: str = "",
    ) -> UiRunState:
        """Validate every input, then save them into a new run.

        Persona and product: upload or configured default. Reference: when switched on, upload
        or configured default (optional: no reference = ungrounded). Metrics: the built-in ids
        and custom metrics (typed text and/or a workbook) chosen in the Evaluation tab apply to
        the whole run; they are frozen at the scenarios step. Usage instructions are optional,
        as a file OR typed text (not both).
        """
        max_chars = self._settings.reference.max_chars
        selection = self._metric_selection(
            selected_metrics, custom_metrics_text, uploads.get(CUSTOM_METRICS)
        )
        self._check_counts(scenario_count, question_count)
        texts: dict[str, str] = {}
        sources: dict[str, str] = {}
        for kind in YAML_KINDS:
            upload = uploads.get(kind)
            if upload is not None:
                parse_upload(
                    kind,
                    upload,
                    max_chars=max_chars,
                    max_upload_bytes=self._settings.web.max_upload_bytes,
                )
                texts[kind] = _decode(upload)
                sources[kind] = f"uploaded: {upload.filename}"
            elif (sample := self.sample(kind)) is not None:
                texts[kind] = sample[0]
                sources[kind] = "sample"
            else:
                raise InputAssetError(f"Upload a {kind} YAML; no configured default is available")
        reference_upload, guide_upload = uploads.get(REFERENCE), uploads.get(GUIDE)
        typed = reference_instructions.strip()
        files: dict[str, Upload] = {}
        if use_reference and typed:
            if guide_upload is not None:
                raise InputAssetError(
                    "Give the usage instructions as a file or as typed text, not both"
                )
            guide_upload = Upload(TYPED_INSTRUCTIONS, typed.encode("utf-8"))
        if not use_reference:
            sources[REFERENCE] = sources[GUIDE] = "off"
        elif reference_upload is None and (sample := self.sample(REFERENCE)) is not None:
            path = self._settings.inputs.reference_file
            assert path is not None
            reference_upload = Upload(path.name, sample[0].encode("utf-8"))
            _reference_of(reference_upload, guide_upload, max_chars)
            files[REFERENCE] = reference_upload
            sources[REFERENCE] = "sample"
        elif reference_upload is None:
            raise InputAssetError(
                "Reference is switched on: upload the reference (a data_catalog or "
                "document_collection, .yaml / .json), configure inputs.reference_file, or "
                "switch it off for an ungrounded run"
            )
        else:
            _reference_of(reference_upload, guide_upload, max_chars)
            files[REFERENCE] = reference_upload
            sources[REFERENCE] = f"uploaded: {reference_upload.filename}"
        if use_reference:
            if guide_upload is not None:
                files[GUIDE] = guide_upload
                sources[GUIDE] = (
                    "typed in the UI" if typed else f"uploaded: {guide_upload.filename}"
                )
            else:
                sources[GUIDE] = "default"
        workbook = uploads.get(CUSTOM_METRICS)
        sources[CUSTOM_METRICS] = f"uploaded: {workbook.filename}" if workbook else "none"
        queries = clean_queries(persona_queries, max_queries=self._settings.web.max_persona_queries)

        run_id = new_run_id()
        for kind, text in texts.items():
            self._store.write_text(self._input(run_id, f"{kind}.yaml"), text)
        for kind, upload in files.items():
            name = Path(upload.filename).name
            self._store.write_text(self._input(run_id, f"{kind}/{name}"), _decode(upload))
        if workbook is not None:
            self._save_workbook(run_id, workbook)
        self._store.write_with(
            self._input(run_id, "persona_queries.json"), queries, TypeAdapter(list[str])
        )
        state = UiRunState(
            run_id=run_id,
            created_at=datetime.now(UTC),
            sources=sources,
            persona_queries=queries,
            question_complexity=question_complexity,
            metric_selection=selection,
            evaluation_locked=False,
            scenario_count=scenario_count,
            question_count=question_count,
        )
        self.save_state(state)
        return state

    # --- evaluation (draft until locked) ---------------------------------------------------

    def update_evaluation(
        self,
        state: UiRunState,
        *,
        selected_metrics: list[str] | None,
        custom_metrics_text: str,
        workbook: Upload | None,
        remove_workbook: bool,
        scenario_count: int | None,
        question_count: int | None,
        question_complexity: ComplexityPreference | None,
        lock: bool,
    ) -> UiRunState:
        """Replace the run's evaluation draft (and lock it when ``lock``).

        Without a new workbook the run's saved workbook is kept, unless ``remove_workbook``.
        A locked evaluation is refused: unlock it first (possible until scenarios exist).
        """
        if state.evaluation_locked:
            raise InputAssetError(
                f"Run {state.run_id}: the evaluation is locked; unlock it to change it"
            )
        if workbook is not None and remove_workbook:
            raise InputAssetError("Upload a new workbook or remove the saved one, not both")
        saved = self._single_file(state.run_id, CUSTOM_METRICS)
        if workbook is None and saved is not None and not remove_workbook:
            workbook = Upload(saved.name, saved.read_bytes())
        selection = self._metric_selection(selected_metrics, custom_metrics_text, workbook)
        self._check_counts(scenario_count, question_count)
        if saved is not None and (remove_workbook or workbook is not None):
            saved.unlink()
        if workbook is not None:
            self._save_workbook(state.run_id, workbook)
        sources = {
            **state.sources,
            CUSTOM_METRICS: f"uploaded: {workbook.filename}" if workbook else "none",
        }
        updated = state.model_copy(
            update={
                "metric_selection": selection,
                "scenario_count": scenario_count,
                "question_count": question_count,
                "question_complexity": question_complexity,
                "sources": sources,
                "evaluation_locked": lock,
            }
        )
        self.save_state(updated)
        return updated

    # --- evaluation guidance (judges only; editable at any time) ---------------------------

    def pending_guidance(
        self, *, text: str, upload: Upload | None, remove: bool
    ) -> EvaluationGuidance | None:
        """The guidance a save request would store, validated exactly like the save (None for
        a removal or empty text), so it can be structured BEFORE anything is stored."""
        typed = text.strip()
        if sum((bool(typed), upload is not None, remove)) > 1:
            raise InputAssetError(
                "Give the evaluation guidance as typed text or as a file (or remove it), not more"
            )
        max_chars = self._settings.evaluation.max_guidance_chars
        if upload is not None:
            if len(upload.content) > self._settings.web.max_upload_bytes:
                raise InputAssetError(f"{upload.filename}: file is too large")
            return parse_evaluation_guidance(
                upload.content, source=upload.filename, max_chars=max_chars
            )
        if typed:
            return parse_evaluation_guidance(typed, source=TYPED_GUIDANCE, max_chars=max_chars)
        return None

    def update_evaluation_guidance(
        self, state: UiRunState, *, text: str, upload: Upload | None, remove: bool
    ) -> UiRunState:
        """Replace (or remove) the run's evaluation guidance: typed text OR a .txt / .md / .yaml
        file, stored as given. Not part of the frozen metric selection, so it may change after
        scenarios exist; the next evaluate step uses it (D237)."""
        self.pending_guidance(text=text, upload=upload, remove=remove)
        typed = text.strip()
        new: Upload | None = None
        if upload is not None:
            new = upload
        elif typed:
            new = Upload(TYPED_GUIDANCE_FILE, typed.encode("utf-8"))
        saved = self._single_file(state.run_id, EVALUATION_GUIDANCE)
        if saved is not None:
            saved.unlink()
        source = "none"
        if new is not None:
            name = Path(new.filename).name
            path = self._store.path_for(self._input(state.run_id, f"{EVALUATION_GUIDANCE}/{name}"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(new.content)  # kept byte for byte (its provenance)
            source = TYPED_GUIDANCE if upload is None else f"uploaded: {upload.filename}"
        updated = state.model_copy(
            update={"sources": {**state.sources, EVALUATION_GUIDANCE: source}}
        )
        self.save_state(updated)
        return updated

    def evaluation_guidance(self, run_id: str) -> EvaluationGuidance | None:
        """The saved guidance, with its structured items when this version was structured
        (the pipeline's cache, keyed by the text's sha256)."""
        saved = self._single_file(run_id, EVALUATION_GUIDANCE)
        if saved is None:
            return None
        guidance = parse_evaluation_guidance(
            saved.read_bytes(),
            source=TYPED_GUIDANCE if saved.name == TYPED_GUIDANCE_FILE else saved.name,
            max_chars=self._settings.evaluation.max_guidance_chars,
        )
        cache = run_artifact(run_id, STRUCTURED_GUIDANCE)
        if guidance is not None and self._store.exists(cache):
            cached = self._store.read_model(cache, EvaluationGuidance)
            if cached.sha256 == guidance.sha256:
                return guidance.model_copy(update={"items": cached.items})
        return guidance

    def unlock_evaluation(self, state: UiRunState) -> UiRunState:
        """Make the evaluation editable again; refused once scenarios froze the metric config."""
        if Step.SCENARIOS in state.steps_done:
            raise InputAssetError(
                f"Run {state.run_id}: scenarios were generated with this evaluation (its metric "
                "configuration is frozen); start a new run to change it"
            )
        updated = state.model_copy(update={"evaluation_locked": False})
        self.save_state(updated)
        return updated

    def _check_counts(self, scenario_count: int | None, question_count: int | None) -> None:
        for label, value in (("scenario count", scenario_count), ("questions", question_count)):
            if value is not None and not 1 <= value <= self._settings.web.max_step_count:
                raise InputAssetError(
                    f"{label} must be between 1 and {self._settings.web.max_step_count}"
                )

    def _save_workbook(self, run_id: str, workbook: Upload) -> None:
        # Kept byte for byte: the workbook is the provenance of the custom metrics.
        path = self._store.path_for(
            self._input(run_id, f"{CUSTOM_METRICS}/{Path(workbook.filename).name}")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(workbook.content)

    def _metric_selection(
        self, selected: list[str] | None, text: str, workbook: Upload | None
    ) -> MetricSelection:
        limits = custom_input_limits(self._settings)
        parts: list[ParsedCustomInput] = []
        if text.strip():
            if len(text) > limits.max_chars_per_metric * limits.max_metrics:
                raise InputAssetError("custom metric text is too long")
            parts.append(parse_free_text(text, limits))
        if workbook is not None:
            if Path(workbook.filename).suffix.casefold() not in SPREADSHEET_SUFFIXES:
                raise InputAssetError(f"{workbook.filename}: custom metrics need .xlsx or .csv")
            if len(workbook.content) > self._settings.web.max_upload_bytes:
                raise InputAssetError(f"{workbook.filename}: file is too large")
            parts.append(parse_workbook(workbook.content, workbook.filename, limits))
        parsed = merge(parts, limits)
        # The user's selection wins; a request that names no metric at all (no built-in, no
        # custom) falls back to the configured evaluation.default_metrics.
        if selected:
            ids = list(selected)
        elif parsed.inputs:
            ids = []
        else:
            ids = list(self._settings.evaluation.default_metrics)
        validate_selected_ids(ids)
        try:
            return MetricSelection(
                selected_metrics=ids, custom_inputs=parsed.inputs, input_errors=parsed.errors
            )
        except ValidationError as exc:
            raise InputAssetError(
                f"Invalid metric selection: {format_validation_error(exc)}"
            ) from exc

    def load_inputs(self, run_id: str) -> RunInputs:
        def parse(kind: str) -> BaseModel:
            path = self._store.path_for(self._input(run_id, f"{kind}.yaml"))
            model, label = YAML_KINDS[kind]
            return parse_asset(path.read_text(encoding="utf-8"), model, kind=label, source=kind)

        persona = parse("persona")
        product = parse("product")
        assert isinstance(persona, Persona)
        assert isinstance(product, ProductDescription)
        reference_file = self._single_file(run_id, REFERENCE)
        reference = None
        if reference_file is not None:
            guide_file = self._single_file(run_id, GUIDE)
            reference = _reference_of(
                Upload(reference_file.name, reference_file.read_bytes()),
                Upload(guide_file.name, guide_file.read_bytes()) if guide_file else None,
                self._settings.reference.max_chars,
            )
        state = self.state(run_id)
        if state.question_complexity is not None:
            preferences = persona.question_preferences.model_copy(
                update={"complexity": state.question_complexity}
            )
            persona = persona.model_copy(update={"question_preferences": preferences})
        queries = state.persona_queries
        if queries:
            examples = list(dict.fromkeys([*persona.example_utterances, *queries]))
            persona = persona.model_copy(update={"example_utterances": examples})
        return RunInputs(
            persona=persona,
            product=product,
            reference=reference,
            metric_selection=state.metric_selection,
            evaluation_guidance=self.evaluation_guidance(run_id),
        )

    def _single_file(self, run_id: str, kind: str) -> Path | None:
        folder = self._store.path_for(self._input(run_id, kind))
        files = sorted(p for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []
        if len(files) > 1:
            raise InputAssetError(f"Run {run_id}: expected one file in inputs/{kind}/")
        return files[0] if files else None

    # --- product connection ----------------------------------------------------------------

    def connect(
        self,
        state: UiRunState,
        url: str,
        upload: Upload,
        context_source: ContextSourceMode | None = None,
    ) -> UiRunState:
        """Validate URL + contract, save the contract and switch the run to the real product
        (the mock contract, if any, is dropped). ``context_source`` is the UI's choice of where
        retrieved context comes from; it must match what the contract configures."""
        max_upload_bytes = self._settings.web.max_upload_bytes
        if len(upload.content) > max_upload_bytes:
            raise InputAssetError(f"{upload.filename}: larger than {max_upload_bytes:,} bytes")
        try:
            text = upload.content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InputAssetError(f"{upload.filename}: not UTF-8 text") from exc
        parsed = parse_contract(text, url, source=upload.filename)
        options = parsed.options
        check_context_source(options, context_source)
        self._store.write_text(self._input(state.run_id, CONTRACT_FILE), text)
        connection = ProductConnection(
            product_url=options.url,
            adapter_type=parsed.adapter_type,
            contract_file=upload.filename,
            method=options.method,
            header_names=sorted(options.headers),
            timeout=options.timeout,
            request_template=options.request_template,
            response_mapping=options.response_mapping.model_dump(mode="json", exclude_none=True),
            stream=(
                options.stream.model_dump(mode="json")
                if isinstance(options, HttpSseOptions)
                else None
            ),
            context_source=options.context_source,
            context_api_url=options.context_api.url if options.context_api else None,
        )
        updated = state.model_copy(update={"connection": connection, "contract": None})
        self.save_state(updated)
        return updated

    def connection_options(self, state: UiRunState) -> ParsedHttpConnection | None:
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

    def recent_runs(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Newest UI runs first (runs without a readable ui_state.json are skipped)."""
        limit = self._settings.web.recent_runs_limit if limit is None else limit
        runs_dir = self._store.path_for(Path(RUNS_DIR))
        names = (
            sorted(
                (p.name for p in runs_dir.iterdir() if RUN_ID_PATTERN.fullmatch(p.name)),
                reverse=True,
            )
            if runs_dir.is_dir()
            else []
        )
        runs: list[dict[str, Any]] = []
        for run_id in names:
            if len(runs) >= limit:
                break
            if not self._store.exists(run_artifact(run_id, UI_STATE)):
                continue
            try:
                state = self.state(run_id)
                kpis = self._read(run_id, "kpis.json", RunKpis)
            except (ArtifactNotFoundError, ValidationError, ValueError):
                continue
            runs.append(
                {
                    "run_id": run_id,
                    "created_at": state.created_at.isoformat(),
                    "sources": state.sources,
                    "question_complexity": state.question_complexity,
                    "selected_metrics": (
                        state.metric_selection.selected_metrics if state.metric_selection else []
                    ),
                    "custom_metrics": (
                        len(state.metric_selection.custom_inputs) if state.metric_selection else 0
                    ),
                    "legacy_evaluation_profile": state.legacy_evaluation_profile,
                    "steps_done": [s.value for s in state.steps_done],
                    "task_success_rate": kpis.task_success_rate if kpis else None,
                    "accuracy": kpis.accuracy if kpis else None,
                    "judged": kpis.judged if kpis else None,
                    "product_id": kpis.product_id if kpis else None,
                }
            )
        return runs

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
        packs = self._read(run_id, "reference_packs.json", list[ReferencePack]) or []
        config: ResolvedMetricConfig | None = self._read(
            run_id, "metric_config.json", ResolvedMetricConfig
        )
        question_by_id = {question.question_id: question for question in questions}
        pack_by_id = {pack.pack_id: pack for pack in packs}
        evaluations = [
            attach_trace_links(
                evaluation,
                question_by_id[evaluation.question_id],
                pack_by_id.get(question_by_id[evaluation.question_id].reference_pack_id or ""),
            )
            if evaluation.question_id in question_by_id
            else evaluation
            for evaluation in evaluations
        ]
        if config is None:
            # Legacy runs: notes were added at render time before the evaluator wrote them.
            execution_by_id = {execution.execution_id: execution for execution in executions}
            sql_expected = run_expects_sql(None, evaluations)
            evaluations = [
                evaluation.model_copy(update={"notes": [*evaluation.notes, NO_SQL_REASON]})
                if (
                    NO_SQL_REASON not in evaluation.notes
                    and (execution := execution_by_id.get(evaluation.execution_id)) is not None
                    and no_sql_execution(execution, sql_expected=sql_expected)
                )
                else evaluation
                for evaluation in evaluations
            ]
        kpis = self._read(run_id, "kpis.json", RunKpis)
        metric_ids = (
            list(config.selected_metric_ids)
            if config is not None
            else list(dict.fromkeys(m for ev in evaluations for m in ev.metrics))
        )
        rows = build_rows(run_id, scenarios, questions, executions, evaluations, packs, metric_ids)
        columns = columns_for(metric_ids)
        self._store.write_text(run_artifact(run_id, RESULTS_CSV), to_csv(rows, columns))
        details = {
            "scenarios": [s.model_dump(mode="json") for s in scenarios],
            "questions": [q.model_dump(mode="json") for q in questions],
            "reference_packs": [p.model_dump(mode="json") for p in packs],
            "metric_config": config.model_dump(mode="json") if config is not None else None,
            "kpis": kpis.model_dump(mode="json") if kpis is not None else None,
            "executions": [e.model_dump(mode="json") for e in executions],
            "evaluations": [e.model_dump(mode="json") for e in evaluations],
            "issue_summary": consolidate_findings(evaluations, executions, config).model_dump(
                mode="json"
            ),
        }
        rejected_scenarios = self._read(run_id, "scenarios_rejected.json", list[RejectedScenario])
        rejected_questions = self._read(run_id, "questions_rejected.json", list[RejectedQuestion])
        return {
            "rows": rows,
            "columns": columns,
            "metric_ids": metric_ids,
            **details,
            "rejected_scenarios": [r.model_dump(mode="json") for r in rejected_scenarios or []],
            "rejected_questions": [r.model_dump(mode="json") for r in rejected_questions or []],
            "evaluation_guidance": (
                guidance.model_dump(mode="json")
                if (guidance := self.evaluation_guidance(run_id)) is not None
                else None
            ),
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


def _reference_of(upload: Upload, guide: Upload | None, max_chars: int) -> Reference:
    return parse_reference(
        upload.content,
        source_name=upload.filename,
        max_chars=max_chars,
        guide=guide.content if guide else None,
        guide_name=guide.filename if guide else None,
    )
