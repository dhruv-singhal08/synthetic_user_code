"""UI run workspace: inputs, UI state and the results table for one run.

Layout inside ``<artifacts_dir>/runs/<run_id>/``:
    inputs/persona.yaml, inputs/product.yaml      (uploaded or sample copy)
    inputs/reference/<file>                        (uploaded .json/.txt/.md; absent -> ungrounded)
    inputs/reference_guide/<file>                  (optional uploaded usage guide .md/.txt)
    inputs/golden_set.json                         (optional normalized SME examples)
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
from synthetic_user.adapters.http_sse import HttpSseOptions
from synthetic_user.core.config import AppConfig, ProductAdapterType
from synthetic_user.core.exceptions import (
    ArtifactNotFoundError,
    InputAssetError,
    format_validation_error,
)
from synthetic_user.core.identifiers import RUN_ID_PATTERN, new_run_id
from synthetic_user.core.loaders import parse_asset, parse_yaml_mapping
from synthetic_user.evaluation_trace import attach_trace_links
from synthetic_user.golden_set import (
    GOLDEN_SET,
    golden_set_preview,
    ground_golden_set,
    parse_golden_set,
    validate_agent_contract,
)
from synthetic_user.kpis import consolidate_findings
from synthetic_user.models.base import NonEmptyStr, StrictModel
from synthetic_user.models.evaluation import EvaluationResult
from synthetic_user.models.evaluation_profile import (
    DEFAULT_AGENT_METRICS,
    AgentMetric,
    EvaluationProfile,
    EvaluationProfileSelection,
    normalize_agent_metrics,
    resolve_evaluation_profile,
)
from synthetic_user.models.execution import ExecutionRecord
from synthetic_user.models.golden_set import GoldenCase, GoldenSet
from synthetic_user.models.kpi import RunKpis
from synthetic_user.models.persona import ComplexityPreference, Persona
from synthetic_user.models.product import ProductDescription
from synthetic_user.models.question import QuestionCase, RejectedQuestion
from synthetic_user.models.reference import DataCatalog, Reference, ReferenceKind, ReferencePack
from synthetic_user.models.scenario import RejectedScenario, Scenario
from synthetic_user.no_sql import NO_SQL_REASON, no_sql_execution, run_expects_sql
from synthetic_user.pipeline import RunInputs, Step, steps_from
from synthetic_user.references.loader import GUIDE_SUFFIXES, parse_reference
from synthetic_user.storage.json_store import JsonArtifactStore
from synthetic_user.storage.runs import RUNS_DIR, run_artifact
from synthetic_user.web.results_csv import build_rows, to_csv

REFERENCE = "reference"  # data_catalog / document_collection (.yaml/.json); absent -> ungrounded
GUIDE = "reference_guide"  # optional usage instructions (.md/.txt upload, or typed in the UI)
TYPED_INSTRUCTIONS = "instructions.md"  # file name used for instructions typed in the UI
YAML_KINDS: dict[str, tuple[type[BaseModel], str]] = {
    "persona": (Persona, "Persona file"),
    "product": (ProductDescription, "Product file"),
}
INPUT_KINDS = (*YAML_KINDS, REFERENCE, GUIDE, GOLDEN_SET)
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
    # The resolved profile is immutable run metadata. ``selection`` records whether it was
    # auto-selected or explicitly chosen so reopening a run never has to infer this again.
    evaluation_profile: EvaluationProfile = EvaluationProfile.NL2SQL
    evaluation_profile_selection: EvaluationProfileSelection = EvaluationProfileSelection.AUTO
    evaluation_profile_reason: str = "Legacy run: NL2SQL was the default enabled profile."
    # Defaults keep older UI runs readable. Relevance/completion are mandatory; optional metrics
    # may be disabled per run and record N/A when enabled but their telemetry is unavailable.
    agent_metrics: list[AgentMetric] = Field(default_factory=lambda: list(DEFAULT_AGENT_METRICS))
    reference_id: str | None = None
    contract: ProductContract | None = None  # mock contract (D113)
    connection: ProductConnection | None = None  # real product: URL + uploaded contract
    steps_done: list[Step] = Field(default_factory=list)


@dataclass(frozen=True)
class Upload:
    filename: str
    content: bytes


class GuidePreview(StrictModel):
    name: NonEmptyStr
    chars: int


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


def parse_upload(kind: str, upload: Upload, *, max_chars: int, max_upload_bytes: int) -> BaseModel:
    """Validate persona/product YAML, reference YAML/JSON, a golden set or instructions."""
    if kind not in INPUT_KINDS:
        raise InputAssetError(f"Unknown input kind {kind!r}")
    if len(upload.content) > max_upload_bytes:
        raise InputAssetError(f"{upload.filename}: larger than {max_upload_bytes:,} bytes")
    if kind == GOLDEN_SET:
        return golden_set_preview(parse_golden_set(upload.content, source_name=upload.filename))
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
        evaluation_profile: EvaluationProfileSelection = EvaluationProfileSelection.AUTO,
        agent_metrics: list[AgentMetric] | None = None,
        use_reference: bool = False,
        reference_instructions: str = "",
    ) -> UiRunState:
        """Validate every input, then save them into a new run.

        Persona and product: upload or configured default. Reference: when switched on, upload
        or configured default. Golden sets are optional and require Reference data when supplied.
        Agent runs require structured Agent context so tool/source names can be checked before
        execution. Usage instructions are optional, as a file OR typed text (not both).
        """
        max_chars = self._settings.reference.max_chars
        texts: dict[str, str] = {}
        sources: dict[str, str] = {}
        parsed_product: ProductDescription | None = None
        for kind in YAML_KINDS:
            upload = uploads.get(kind)
            if upload is not None:
                parsed = parse_upload(
                    kind,
                    upload,
                    max_chars=max_chars,
                    max_upload_bytes=self._settings.web.max_upload_bytes,
                )
                if kind == "product":
                    assert isinstance(parsed, ProductDescription)
                    parsed_product = parsed
                texts[kind] = _decode(upload)
                sources[kind] = f"uploaded: {upload.filename}"
            elif (sample := self.sample(kind)) is not None:
                texts[kind] = sample[0]
                sources[kind] = "sample"
                if kind == "product":
                    assert isinstance(sample[1], ProductDescription)
                    parsed_product = sample[1]
            else:
                raise InputAssetError(f"Upload a {kind} YAML; no configured default is available")
        assert parsed_product is not None
        try:
            resolved_profile, profile_reason = resolve_evaluation_profile(
                parsed_product, evaluation_profile
            )
            selected_agent_metrics = normalize_agent_metrics(
                agent_metrics or self._settings.evaluation.agent_metrics
            )
        except ValueError as exc:
            raise InputAssetError(str(exc)) from exc
        golden_upload = uploads.get(GOLDEN_SET)
        golden_set: GoldenSet | None = None
        if golden_upload is not None:
            if len(golden_upload.content) > self._settings.web.max_upload_bytes:
                raise InputAssetError(
                    f"{golden_upload.filename}: larger than "
                    f"{self._settings.web.max_upload_bytes:,} bytes"
                )
            golden_set = parse_golden_set(golden_upload.content, source_name=golden_upload.filename)
            sources[GOLDEN_SET] = f"uploaded: {golden_upload.filename}"
        else:
            sources[GOLDEN_SET] = "none"
        reference_upload, guide_upload = uploads.get(REFERENCE), uploads.get(GUIDE)
        resolved_reference: Reference | None = None
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
            resolved_reference = _reference_of(reference_upload, guide_upload, max_chars)
            files[REFERENCE] = reference_upload
            sources[REFERENCE] = "sample"
            if guide_upload is not None:
                files[GUIDE] = guide_upload
                sources[GUIDE] = (
                    "typed in the UI" if typed else f"uploaded: {guide_upload.filename}"
                )
            else:
                sources[GUIDE] = "default"
        elif reference_upload is None:
            raise InputAssetError(
                "Reference is switched on: upload the reference (a data_catalog or "
                "document_collection, .yaml / .json), configure inputs.reference_file, or "
                "switch it off for an ungrounded run"
            )
        else:
            resolved_reference = _reference_of(reference_upload, guide_upload, max_chars)
            files[REFERENCE] = reference_upload
            sources[REFERENCE] = f"uploaded: {reference_upload.filename}"
            if guide_upload is not None:
                files[GUIDE] = guide_upload
                sources[GUIDE] = (
                    "typed in the UI" if typed else f"uploaded: {guide_upload.filename}"
                )
            else:
                sources[GUIDE] = "default"
        if golden_set is not None and resolved_reference is None:
            raise InputAssetError(
                "A golden set must be used together with Reference data. Switch Reference on "
                "and upload the authoritative data, schema, tools, sources and rules."
            )
        if resolved_profile is EvaluationProfile.AGENT:
            validate_agent_contract(resolved_reference)
        if golden_set is not None and resolved_profile is EvaluationProfile.AGENT:
            assert resolved_reference is not None
            golden_set = ground_golden_set(golden_set, resolved_reference)
        queries = clean_queries(persona_queries, max_queries=self._settings.web.max_persona_queries)

        run_id = new_run_id()
        for kind, text in texts.items():
            self._store.write_text(self._input(run_id, f"{kind}.yaml"), text)
        for kind, upload in files.items():
            name = Path(upload.filename).name
            self._store.write_text(self._input(run_id, f"{kind}/{name}"), _decode(upload))
        self._store.write_with(
            self._input(run_id, "persona_queries.json"), queries, TypeAdapter(list[str])
        )
        if golden_set is not None:
            self._store.write_model(self._input(run_id, "golden_set.json"), golden_set)
        state = UiRunState(
            run_id=run_id,
            created_at=datetime.now(UTC),
            sources=sources,
            persona_queries=queries,
            question_complexity=question_complexity,
            evaluation_profile=resolved_profile,
            evaluation_profile_selection=evaluation_profile,
            evaluation_profile_reason=profile_reason,
            agent_metrics=selected_agent_metrics,
        )
        self.save_state(state)
        return state

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
        golden_path = self._input(run_id, "golden_set.json")
        golden_set = (
            self._store.read_model(golden_path, GoldenSet)
            if self._store.exists(golden_path)
            else None
        )
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
            golden_set=golden_set,
            evaluation_profile=state.evaluation_profile,
        )

    def _single_file(self, run_id: str, kind: str) -> Path | None:
        folder = self._store.path_for(self._input(run_id, kind))
        files = sorted(p for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []
        if len(files) > 1:
            raise InputAssetError(f"Run {run_id}: expected one file in inputs/{kind}/")
        return files[0] if files else None

    # --- product connection ----------------------------------------------------------------

    def connect(self, state: UiRunState, url: str, upload: Upload) -> UiRunState:
        """Validate URL + contract, save the contract and switch the run to the real product
        (the mock contract, if any, is dropped)."""
        max_upload_bytes = self._settings.web.max_upload_bytes
        if len(upload.content) > max_upload_bytes:
            raise InputAssetError(f"{upload.filename}: larger than {max_upload_bytes:,} bytes")
        try:
            text = upload.content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InputAssetError(f"{upload.filename}: not UTF-8 text") from exc
        parsed = parse_contract(text, url, source=upload.filename)
        options = parsed.options
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
                    "evaluation_profile": state.evaluation_profile,
                    "evaluation_profile_selection": state.evaluation_profile_selection,
                    "agent_metrics": state.agent_metrics,
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
        questions = self._questions_with_golden_contract(run_id, questions)
        executions = self._read(run_id, "executions.json", list[ExecutionRecord]) or []
        evaluations = self._read(run_id, "evaluations.json", list[EvaluationResult]) or []
        packs = self._read(run_id, "reference_packs.json", list[ReferencePack]) or []
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
        execution_by_id = {execution.execution_id: execution for execution in executions}
        sql_expected = run_expects_sql(executions, evaluations)
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
        rows = build_rows(run_id, scenarios, questions, executions, evaluations, packs)
        self._store.write_text(run_artifact(run_id, RESULTS_CSV), to_csv(rows))
        details = {
            "scenarios": [s.model_dump(mode="json") for s in scenarios],
            "reference_packs": [p.model_dump(mode="json") for p in packs],
            "kpis": kpis.model_dump(mode="json") if kpis is not None else None,
            "executions": [e.model_dump(mode="json") for e in executions],
            "evaluations": [e.model_dump(mode="json") for e in evaluations],
            "issue_summary": consolidate_findings(evaluations, executions).model_dump(mode="json"),
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

    def _questions_with_golden_contract(
        self, run_id: str, questions: list[QuestionCase]
    ) -> list[QuestionCase]:
        """Hydrate the UI view from the saved golden input.

        This also makes runs created by the earlier implementation immediately display journey,
        duration, source and the complete uploaded contract without requiring regeneration.
        """
        golden_path = self._input(run_id, "golden_set.json")
        if not self._store.exists(golden_path):
            return questions
        golden_set = self._store.read_model(golden_path, GoldenSet)
        by_id = {case.case_id: case for case in golden_set.cases}
        hydrated: list[QuestionCase] = []
        for question in questions:
            case = by_id.get(question.golden_case_id or "")
            if case is None:
                hydrated.append(question)
                continue
            expectations = question.expectations.model_copy(
                update={
                    "expected_behavior": case.expected_behavior,
                    "success_criteria": case.success_criteria,
                    "expected_tools": case.expected_tools,
                    "expected_journey": case.expected_journey,
                    "expected_time_period": case.expected_time_period,
                    "expected_data_sources": case.expected_data_sources,
                    "expected_context": case.expected_context,
                    "prohibited_behaviors": case.prohibited_behaviors,
                    "reference_context": case.reference_context,
                    "golden_set_criteria": (
                        case.uploaded_criteria or _legacy_uploaded_criteria(case)
                    ),
                }
            )
            hydrated.append(question.model_copy(update={"expectations": expectations}))
        return hydrated

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


def _legacy_uploaded_criteria(case: GoldenCase) -> dict[str, str]:
    """Reconstruct standard labels for golden inputs saved before exact columns were retained."""
    values: tuple[tuple[str, str | None], ...] = (
        ("Scenario Type", case.scenario_type),
        ("Expected Journey", case.expected_journey),
        ("Expected Time Period", case.expected_time_period),
        ("Expected Data Source", "; ".join(case.expected_data_sources) or None),
        ("Expected Answer / Response (SME input)", case.expected_behavior),
        ("Expected Success Criteria", "\n".join(case.success_criteria)),
        ("Expected Tools", "; ".join(case.expected_tools) or None),
        ("Expected Context / Evidence", "\n".join(case.expected_context) or None),
        ("Prohibited Behavior", "\n".join(case.prohibited_behaviors) or None),
    )
    return {label: value for label, value in values if value}
