"""FastAPI server: JSON endpoints for the pipeline + the single-page UI in ./static.

Local tool (binds 127.0.0.1 by default). Each step endpoint runs one pipeline step,
refreshes results.csv, and returns the current results table.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from synthetic_user.adapters.base import ProductAdapter, UnavailableAdapter
from synthetic_user.adapters.contract import ProductContract, request_contract
from synthetic_user.adapters.http_json import (
    HttpJsonAdapter,
    Transport,
    urllib_transport,
)
from synthetic_user.adapters.http_sse import (
    HttpSseAdapter,
    HttpSseOptions,
    SseTransport,
    aiohttp_sse_transport,
)
from synthetic_user.adapters.mock_llm import MockLLMAdapter
from synthetic_user.adapters.registry import create_adapter
from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import (
    ArtifactNotFoundError,
    ConfigError,
    SyntheticUserError,
)
from synthetic_user.core.identifiers import RUN_ID_PATTERN
from synthetic_user.core.logging import get_run_logger
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.factory import create_llm_client
from synthetic_user.llm.tracing import TracingLLMClient
from synthetic_user.metrics.catalog import builtin_definitions, category_labels
from synthetic_user.models.execution import ContextSourceMode
from synthetic_user.models.metrics import EvidenceKind
from synthetic_user.models.persona import ComplexityPreference
from synthetic_user.models.question import MAX_DELIVERABLES, QuestionComplexity
from synthetic_user.models.reference import (
    DEFAULT_INSTRUCTIONS,
    ReferenceKind,
    ReferencePackDraft,
)
from synthetic_user.pipeline import RunInputs, Step, SyntheticUserPipeline
from synthetic_user.storage.json_store import JsonArtifactStore
from synthetic_user.web.workspace import (
    CUSTOM_METRICS,
    GUIDE,
    REFERENCE,
    RESULTS_CSV,
    ParsedHttpConnection,
    ProductConnection,
    Upload,
    Workspace,
    custom_input_limits,
    parse_upload,
)

STATIC_DIR = Path(__file__).parent / "static"
CONTRACT_TEMPLATE = STATIC_DIR / "product_contract_template.yaml"
# UI steps run in order (D115); scenarios runs the reference check itself.
_PREREQUISITE = {
    Step.QUESTIONS: Step.SCENARIOS,
    Step.EXECUTE: Step.QUESTIONS,
    Step.EVALUATE: Step.EXECUTE,
}


# Evidence kinds that need retrieved context: selecting such a metric shows the context-source
# selector in the product connection UI.
CONTEXT_EVIDENCE = frozenset(
    {
        EvidenceKind.RETRIEVED_CONTEXT,
        EvidenceKind.RANKED_CONTEXT,
        EvidenceKind.CITATIONS,
        EvidenceKind.GROUNDING_EVIDENCE,
    }
)
LEGACY_RUN_FIELDS = (
    "Evaluation profiles were removed: choose metrics in the Evaluation tab "
    "(selected_metrics / custom metrics); evaluation_profile, agent_metrics and golden_set "
    "are no longer accepted."
)


class StepRequest(BaseModel):
    """Counts are run selections (Evaluation tab), not per-step overrides."""

    model_config = ConfigDict(extra="forbid")

    start_from_question: int | None = Field(default=None, ge=1)


class ContractRequest(BaseModel):
    url: str = Field(min_length=1)


def create_app(
    settings: AppConfig,
    *,
    env: Mapping[str, str] | None = None,
    transport: Transport = urllib_transport,
    sse_transport: SseTransport = aiohttp_sse_transport,
) -> FastAPI:
    """``env`` fills ``${VAR}`` contract headers (default: the process environment);
    ``transport`` is replaceable for tests."""
    environ: Mapping[str, str] = os.environ if env is None else env
    store = JsonArtifactStore(settings.paths.artifacts_dir)
    workspace = Workspace(settings, store)
    app = FastAPI(title="Synthetic User", docs_url="/api/docs")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(SyntheticUserError)
    async def _user_error(request: Request, exc: SyntheticUserError) -> JSONResponse:
        status = 404 if isinstance(exc, ArtifactNotFoundError) else 400
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    def check_run_id(run_id: str) -> str:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ConfigError(f"invalid run id {run_id!r}")
        return run_id

    # --- pages / info --------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        return {
            "llm_provider": settings.llm.provider,
            "llm_model": settings.llm.model,
            "product_adapter": settings.product_adapter.type,
            "reference_max_chars": settings.reference.max_chars,
            "scenario_count": settings.generation.scenario.scenario_count,
            "question_count": settings.generation.question.question_count,
            "execution_resume": True,
            "max_deliverables": settings.generation.question.max_deliverables,
            "web": settings.web.model_dump(mode="json"),
            "question_complexities": [item.value for item in ComplexityPreference],
            "metric_categories": {k.value: v for k, v in category_labels().items()},
            "metric_catalog": [
                {
                    "id": d.metric_id,
                    "display_name": d.display_name,
                    "category": d.category.value,
                    "description": d.description,
                    "judge_group": d.judge_group.value,
                    "needs_context": bool(CONTEXT_EVIDENCE & set(d.required_execution_evidence)),
                    "needs_expectations": d.needs_expectations,
                    "counts_toward_verdict": d.counts_toward_verdict,
                }
                for d in builtin_definitions()
            ],
            "default_metrics": list(settings.evaluation.default_metrics),
            "custom_metric_limits": {
                "max_metrics": settings.evaluation.max_custom_metrics,
                "max_chars_per_metric": settings.evaluation.max_custom_metric_chars,
            },
            "context_sources": [m.value for m in ContextSourceMode],
            "steps": [s.value for s in Step],
            # Used when no usage instructions are given (shown as the UI's placeholder).
            "default_instructions": {k.value: text for k, text in DEFAULT_INSTRUCTIONS.items()},
        }

    @app.get("/api/samples")
    async def samples() -> dict[str, Any]:
        """Configured persona, product description and reference defaults for the setup UI."""
        result: dict[str, Any] = dict.fromkeys(("persona", "product", REFERENCE, GUIDE))
        for kind in ("persona", "product", REFERENCE):
            sample = workspace.sample(kind)
            if sample is not None:
                text, model = sample
                result[kind] = {"raw": text, "parsed": model.model_dump(mode="json")}
        return result

    @app.post("/api/preview/{kind}")
    async def preview(kind: str, file: Annotated[UploadFile, File()]) -> dict[str, Any]:
        upload = Upload(file.filename or kind, await file.read())
        model = parse_upload(
            kind,
            upload,
            max_chars=settings.reference.max_chars,
            max_upload_bytes=settings.web.max_upload_bytes,
            limits=custom_input_limits(settings),
        )
        parsed = model.model_dump(mode="json")
        raw = None if kind == CUSTOM_METRICS else upload.content.decode("utf-8-sig")
        return {"raw": raw, "parsed": parsed}

    @app.get("/api/contract-template", response_class=PlainTextResponse)
    async def contract_template() -> PlainTextResponse:
        return PlainTextResponse(
            CONTRACT_TEMPLATE.read_text(encoding="utf-8"),
            media_type="application/x-yaml",
            headers={"Content-Disposition": 'attachment; filename="product_contract.yaml"'},
        )

    # --- runs ----------------------------------------------------------------------------

    @app.get("/api/product-template", response_class=PlainTextResponse)
    async def product_template() -> PlainTextResponse:
        return PlainTextResponse(
            (STATIC_DIR / "product_template.yaml").read_text(encoding="utf-8"),
            media_type="application/x-yaml",
            headers={"Content-Disposition": 'attachment; filename="product_description.yaml"'},
        )

    @app.get("/api/reference-template/{kind}", response_class=PlainTextResponse)
    async def reference_template(kind: ReferenceKind) -> PlainTextResponse:
        path = STATIC_DIR / f"reference_template_{kind.value}.yaml"
        return PlainTextResponse(
            path.read_text(encoding="utf-8"),
            media_type="application/x-yaml",
            headers={"Content-Disposition": f'attachment; filename="{kind.value}.yaml"'},
        )

    @app.post("/api/runs")
    async def create_run(
        persona: Annotated[UploadFile | None, File()] = None,
        product: Annotated[UploadFile | None, File()] = None,
        reference: Annotated[UploadFile | None, File()] = None,
        reference_guide: Annotated[UploadFile | None, File()] = None,
        custom_metrics: Annotated[UploadFile | None, File()] = None,
        reference_instructions: Annotated[str, Form()] = "",
        persona_queries: Annotated[str, Form()] = "",
        question_complexity: Annotated[ComplexityPreference | None, Form()] = None,
        selected_metrics: Annotated[list[str] | None, Form()] = None,
        custom_metrics_text: Annotated[str, Form()] = "",
        scenario_count: Annotated[int | None, Form()] = None,
        questions_per_scenario: Annotated[int | None, Form()] = None,
        use_reference: Annotated[bool, Form()] = False,
        # Profile-era fields: refused with a migration message (never a hidden fallback).
        evaluation_profile: Annotated[str | None, Form()] = None,
        agent_metrics: Annotated[list[str] | None, Form()] = None,
        golden_set: Annotated[UploadFile | None, File()] = None,
    ) -> dict[str, Any]:
        if evaluation_profile is not None or agent_metrics is not None or golden_set is not None:
            raise ConfigError(LEGACY_RUN_FIELDS)
        if len(reference_instructions) > settings.web.max_reference_instructions_chars:
            raise ConfigError(
                "Reference instructions are longer than "
                f"{settings.web.max_reference_instructions_chars:,} characters"
            )
        uploads: dict[str, Upload | None] = {}
        for kind, file in (
            ("persona", persona),
            ("product", product),
            (REFERENCE, reference),
            (GUIDE, reference_guide),
            (CUSTOM_METRICS, custom_metrics),
        ):
            uploads[kind] = Upload(file.filename or kind, await file.read()) if file else None
        state = workspace.create_run(
            uploads,
            persona_queries,
            question_complexity=question_complexity,
            selected_metrics=selected_metrics,
            custom_metrics_text=custom_metrics_text,
            scenario_count=scenario_count,
            question_count=questions_per_scenario,
            use_reference=use_reference,
            reference_instructions=reference_instructions,
        )
        return {"state": state.model_dump(mode="json"), **workspace.results(state.run_id)}

    @app.get("/api/runs")
    async def list_runs() -> dict[str, Any]:
        return {"runs": workspace.recent_runs()}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        state = workspace.state(check_run_id(run_id))
        return {"state": state.model_dump(mode="json"), **workspace.results(run_id)}

    @app.post("/api/runs/{run_id}/evaluation")
    async def save_evaluation(
        run_id: str,
        custom_metrics: Annotated[UploadFile | None, File()] = None,
        selected_metrics: Annotated[list[str] | None, Form()] = None,
        custom_metrics_text: Annotated[str, Form()] = "",
        remove_custom_metrics: Annotated[bool, Form()] = False,
        question_complexity: Annotated[ComplexityPreference | None, Form()] = None,
        scenario_count: Annotated[int | None, Form()] = None,
        questions_per_scenario: Annotated[int | None, Form()] = None,
        lock: Annotated[bool, Form()] = True,
    ) -> dict[str, Any]:
        """Save the run's Evaluation tab (and lock it: scenarios need a locked evaluation)."""
        state = workspace.state(check_run_id(run_id))
        workbook = (
            Upload(custom_metrics.filename or CUSTOM_METRICS, await custom_metrics.read())
            if custom_metrics
            else None
        )
        state = workspace.update_evaluation(
            state,
            selected_metrics=selected_metrics,
            custom_metrics_text=custom_metrics_text,
            workbook=workbook,
            remove_workbook=remove_custom_metrics,
            scenario_count=scenario_count,
            question_count=questions_per_scenario,
            question_complexity=question_complexity,
            lock=lock,
        )
        return {"state": state.model_dump(mode="json"), **workspace.results(run_id)}

    @app.delete("/api/runs/{run_id}/evaluation/lock")
    async def unlock_evaluation(run_id: str) -> dict[str, Any]:
        """Edit the evaluation again (only until scenarios froze the metric configuration)."""
        state = workspace.unlock_evaluation(workspace.state(check_run_id(run_id)))
        return {"state": state.model_dump(mode="json"), **workspace.results(run_id)}

    @app.post("/api/runs/{run_id}/connection")
    async def connect_product(
        run_id: str,
        contract: Annotated[UploadFile, File()],
        url: Annotated[str, Form()] = "",
        context_source: Annotated[ContextSourceMode | None, Form()] = None,
    ) -> dict[str, Any]:
        """Real product: product URL + uploaded http_json/http_sse contract for execute.
        ``context_source`` = where retrieved context comes from (must match the contract)."""
        if len(url) > settings.web.max_product_url_chars:
            raise ConfigError(
                f"Product URL is longer than {settings.web.max_product_url_chars:,} characters"
            )
        state = workspace.state(check_run_id(run_id))
        upload = Upload(contract.filename or "contract.yaml", await contract.read())
        state = workspace.connect(state, url, upload, context_source)
        return {"state": state.model_dump(mode="json")}

    @app.delete("/api/runs/{run_id}/connection")
    async def disconnect_product(run_id: str) -> dict[str, Any]:
        state = workspace.state(check_run_id(run_id))
        state = state.model_copy(update={"connection": None, "contract": None})
        workspace.save_state(state)
        return {"state": state.model_dump(mode="json")}

    @app.post("/api/runs/{run_id}/contract")
    async def mock_contract(run_id: str, body: ContractRequest) -> dict[str, Any]:
        """MOCK contract from a URL (D113): questions go to the LLM mock product."""
        if len(body.url) > settings.web.max_product_url_chars:
            raise ConfigError(
                f"Product URL is longer than {settings.web.max_product_url_chars:,} characters"
            )
        state = workspace.state(check_run_id(run_id))
        inputs = workspace.load_inputs(run_id)
        contract = request_contract(body.url, inputs.product.product_id)
        state = state.model_copy(update={"contract": contract, "connection": None})
        workspace.save_state(state)
        return {
            "state": state.model_dump(mode="json"),
            "contract": contract.model_dump(mode="json"),
        }

    @app.post("/api/runs/{run_id}/steps/{step}")
    async def run_step(run_id: str, step: Step, body: StepRequest | None = None) -> dict[str, Any]:
        state = workspace.state(check_run_id(run_id))
        inputs = workspace.load_inputs(run_id)
        needed = _PREREQUISITE.get(step)
        if needed is not None and needed not in state.steps_done:
            raise ConfigError(f"Run the '{needed.value}' step before '{step.value}'")
        if step is Step.SCENARIOS and not state.evaluation_locked:
            raise ConfigError(
                "Lock the evaluation in the Evaluation tab before generating scenarios"
            )
        # The user's run selections win; server defaults apply when the run left them out.
        count = (
            state.scenario_count
            if step is Step.SCENARIOS
            else state.question_count
            if step is Step.QUESTIONS
            else None
        )
        run_settings = _settings_for_question_complexity(settings, state.question_complexity)
        llm = TracingLLMClient.wrap(
            create_llm_client(run_settings)
        )  # mock product calls traced too
        product = (
            _product_adapter(
                run_settings,
                llm,
                state.contract,
                workspace.connection_options(state),
                inputs,
                env=environ,
                transport=transport,
                sse_transport=sse_transport,
            )
            if step is Step.EXECUTE
            else UnavailableAdapter()
        )
        pipeline = SyntheticUserPipeline(
            run_settings,
            llm,
            product,
            store,
            logger=get_run_logger("synthetic_user.web", run_id),
        )
        detail: dict[str, Any] = {}

        if step in (Step.REFERENCE, Step.SCENARIOS):
            # The reference check (no LLM) runs automatically before scenarios.
            reference_step = pipeline.check_reference(inputs)
            reference = reference_step.reference
            state = workspace.mark_done(
                state.model_copy(update={"reference_id": inputs.reference_id}), Step.REFERENCE
            )
            detail = {
                "mode": reference_step.mode.value,
                "reference_id": inputs.reference_id,
                "reference_chars": reference.rendered_chars() if reference else 0,
                "warnings": reference_step.warnings,
            }
        if step is Step.SCENARIOS:
            scenario_step = await pipeline.generate_scenarios(
                run_id, inputs, count=count, force=True
            )
            config = scenario_step.metric_config
            detail = {
                **detail,
                "accepted": len(scenario_step.accepted),
                "rejected": len(scenario_step.rejected),
                "reference_packs": len(scenario_step.packs),
                "metric_config_id": config.config_id,
                "metrics": list(config.selected_metric_ids),
                "metric_compiler_called": config.compiler_called,
                "unavailable_metrics": [d.metric_id for d in config.unavailable()],
            }
        elif step is Step.QUESTIONS:
            question_step = await pipeline.generate_questions(
                run_id, inputs, count=count, force=True
            )
            detail = {
                "accepted": len(question_step.questions),
                "rejected": len(question_step.rejected),
            }
        elif step is Step.EXECUTE:
            try:
                execute_step = await pipeline.execute_questions(
                    run_id,
                    inputs,
                    force=True,
                    start_from_question=body.start_from_question if body else None,
                )
            finally:
                await pipeline.aclose()
            detail = {
                "product": _product_label(state.contract, state.connection, product.name),
                "executions": len(execute_step.executions),
                "resumed_from_question": body.start_from_question if body else None,
            }
        elif step is Step.EVALUATE:
            evaluate_step = await pipeline.evaluate_executions(run_id, inputs, force=True)
            detail = {
                "evaluations": len(evaluate_step.evaluations),
                "bug_candidates": len(evaluate_step.bug_candidates),
                "accuracy": evaluate_step.kpis.accuracy,
                "task_success_rate": evaluate_step.kpis.task_success_rate,
            }

        state = workspace.mark_done(state, step)
        return {
            "state": state.model_dump(mode="json"),
            "step": step.value,
            "detail": detail,
            **workspace.results(run_id),
        }

    @app.put("/api/runs/{run_id}/reference_packs/{pack_id}")
    async def edit_reference_pack(
        run_id: str, pack_id: str, body: ReferencePackDraft
    ) -> dict[str, Any]:
        """A person's edit of one Reference Pack (quotes re-checked against the reference).
        Saving makes questions, execute and evaluate stale: they must be run again."""
        state = workspace.state(check_run_id(run_id))
        if Step.SCENARIOS not in state.steps_done:
            raise ConfigError("Run the 'scenarios' step before editing reference packs")
        inputs = workspace.load_inputs(run_id)
        run_settings = _settings_for_question_complexity(settings, state.question_complexity)
        pipeline = SyntheticUserPipeline(
            run_settings,
            create_llm_client(run_settings),
            UnavailableAdapter(),
            store,
            logger=get_run_logger("synthetic_user.web", run_id),
        )
        pack = pipeline.update_reference_pack(run_id, inputs, pack_id, body)
        state = workspace.mark_done(state, Step.SCENARIOS)
        return {
            "state": state.model_dump(mode="json"),
            "pack": pack.model_dump(mode="json"),
            **workspace.results(run_id),
        }

    # --- downloads -----------------------------------------------------------------------

    @app.get("/api/runs/{run_id}/results.csv")
    async def results_csv(run_id: str) -> FileResponse:
        path = workspace.artifact_path(check_run_id(run_id), RESULTS_CSV)
        return FileResponse(path, media_type="text/csv", filename=f"{run_id}_results.csv")

    @app.get("/api/runs/{run_id}/artifacts/{name}")
    async def artifact(run_id: str, name: str) -> FileResponse:
        path = workspace.artifact_path(check_run_id(run_id), name)
        return FileResponse(path, filename=f"{run_id}_{name}")

    return app


def _product_adapter(
    settings: AppConfig,
    llm: LLMClient,
    contract: ProductContract | None,
    connection: ParsedHttpConnection | None,
    inputs: RunInputs,
    *,
    env: Mapping[str, str],
    transport: Transport,
    sse_transport: SseTransport,
) -> ProductAdapter:
    """Uploaded HTTP contract -> real product; mock contract -> mock; else configured adapter."""
    if connection is not None:
        ssl_verify = settings.product.ssl_verify
        if isinstance(connection.options, HttpSseOptions):
            return HttpSseAdapter(
                connection.options, env=env, ssl_verify=ssl_verify, transport=sse_transport
            )
        return HttpJsonAdapter(
            connection.options, env=env, ssl_verify=ssl_verify, transport=transport
        )
    if contract is not None:
        return MockLLMAdapter(llm, contract, inputs.product, inputs.reference)
    return create_adapter(settings, env=env, transport=transport, sse_transport=sse_transport)


def _settings_for_question_complexity(
    settings: AppConfig, complexity: ComplexityPreference | None
) -> AppConfig:
    """Apply a UI complexity choice as the hard policy for one run.

    The persisted persona file remains untouched. ``load_inputs`` applies the same choice to
    the in-memory persona so scenario and question prompts agree with this policy.
    """
    if complexity is None:
        return settings
    if complexity is ComplexityPreference.SIMPLE:
        allowed = [QuestionComplexity.SIMPLE]
        max_deliverables = 1
        max_sources = 1
        allow_cross_source_comparison = False
    elif complexity is ComplexityPreference.MEDIUM:
        allowed = [QuestionComplexity.SIMPLE, QuestionComplexity.MEDIUM]
        max_deliverables = 2
        max_sources = 1
        allow_cross_source_comparison = False
    elif complexity is ComplexityPreference.COMPLEX:
        allowed = [QuestionComplexity.COMPLEX]
        max_deliverables = MAX_DELIVERABLES
        max_sources = 2
        allow_cross_source_comparison = True
    else:
        allowed = list(QuestionComplexity)
        max_deliverables = MAX_DELIVERABLES
        max_sources = 2
        allow_cross_source_comparison = True
    question = settings.generation.question.model_copy(
        update={
            "allowed_complexities": allowed,
            "max_deliverables": max_deliverables,
            "max_sources": max_sources,
            "allow_cross_source_comparison": allow_cross_source_comparison,
        }
    )
    generation = settings.generation.model_copy(update={"question": question})
    return settings.model_copy(update={"generation": generation})


def _product_label(
    contract: ProductContract | None, connection: ProductConnection | None, adapter_name: str
) -> str:
    if connection is not None:
        return f"{adapter_name} ({connection.product_url})"
    return "mock (contract)" if contract is not None else adapter_name
