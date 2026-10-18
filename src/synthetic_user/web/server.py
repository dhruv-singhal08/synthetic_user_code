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
from pydantic import BaseModel, Field

from synthetic_user.adapters.base import ProductAdapter, UnavailableAdapter
from synthetic_user.adapters.contract import ProductContract, request_contract
from synthetic_user.adapters.http_json import (
    HttpJsonAdapter,
    HttpJsonOptions,
    Transport,
    urllib_transport,
)
from synthetic_user.adapters.mock_llm import MockLLMAdapter
from synthetic_user.adapters.registry import create_adapter
from synthetic_user.core.config import AppConfig, WebUi
from synthetic_user.core.exceptions import (
    ArtifactNotFoundError,
    ConfigError,
    SyntheticUserError,
)
from synthetic_user.core.identifiers import RUN_ID_PATTERN
from synthetic_user.core.logging import get_run_logger
from synthetic_user.llm.base import LLMClient
from synthetic_user.llm.factory import create_llm_client
from synthetic_user.models.reference import ReferencePackDraft
from synthetic_user.pipeline import RunInputs, Step, SyntheticUserPipeline
from synthetic_user.storage.json_store import JsonArtifactStore
from synthetic_user.web.workspace import (
    GUIDE,
    REFERENCE,
    RESULTS_CSV,
    ProductConnection,
    Upload,
    Workspace,
    parse_upload,
)

STATIC_DIR = Path(__file__).parent / "static"
# Two UIs over the same API (D199): web.ui picks which one "/" serves.
UI_PAGES = {
    WebUi.MODERN: STATIC_DIR / "modern" / "index.html",
    WebUi.CLASSIC: STATIC_DIR / "index.html",
}
CONTRACT_TEMPLATE = STATIC_DIR / "product_contract_template.yaml"
# UI steps run in order (D115); scenarios runs the reference check itself.
_PREREQUISITE = {
    Step.QUESTIONS: Step.SCENARIOS,
    Step.EXECUTE: Step.QUESTIONS,
    Step.EVALUATE: Step.EXECUTE,
}


class StepRequest(BaseModel):
    count: int | None = Field(default=None, ge=1, le=50)


class ContractRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2000)


def create_app(
    settings: AppConfig,
    *,
    env: Mapping[str, str] | None = None,
    transport: Transport = urllib_transport,
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
        return FileResponse(UI_PAGES[settings.web.ui])

    @app.get("/modern", include_in_schema=False)
    async def modern_ui() -> FileResponse:
        return FileResponse(UI_PAGES[WebUi.MODERN])

    @app.get("/classic", include_in_schema=False)
    async def classic_ui() -> FileResponse:
        return FileResponse(UI_PAGES[WebUi.CLASSIC])

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        return {
            "llm_provider": settings.llm.provider,
            "llm_model": settings.llm.model,
            "product_adapter": settings.product_adapter.type,
            "reference_max_chars": settings.reference.max_chars,
            "scenario_count": settings.generation.scenario.scenario_count,
            "question_count": settings.generation.question.question_count,
            "steps": [s.value for s in Step],
            "ui": settings.web.ui.value,
        }

    @app.get("/api/samples")
    async def samples() -> dict[str, Any]:
        """Only the persona has a sample; product description and reference are upload-only."""
        result: dict[str, Any] = dict.fromkeys(("persona", "product", REFERENCE, GUIDE))
        sample = workspace.sample("persona")
        if sample is not None:
            text, model = sample
            result["persona"] = {"raw": text, "parsed": model.model_dump(mode="json")}
        return result

    @app.post("/api/preview/{kind}")
    async def preview(kind: str, file: Annotated[UploadFile, File()]) -> dict[str, Any]:
        upload = Upload(file.filename or kind, await file.read())
        model = parse_upload(kind, upload, max_chars=settings.reference.max_chars)
        parsed = model.model_dump(mode="json", exclude={"text"} if kind == REFERENCE else None)
        return {"raw": upload.content.decode("utf-8-sig"), "parsed": parsed}

    @app.get("/api/contract-template", response_class=PlainTextResponse)
    async def contract_template() -> PlainTextResponse:
        return PlainTextResponse(
            CONTRACT_TEMPLATE.read_text(encoding="utf-8"),
            media_type="application/x-yaml",
            headers={"Content-Disposition": 'attachment; filename="product_contract.yaml"'},
        )

    # --- runs ----------------------------------------------------------------------------

    @app.post("/api/runs")
    async def create_run(
        persona: Annotated[UploadFile | None, File()] = None,
        product: Annotated[UploadFile | None, File()] = None,
        reference: Annotated[UploadFile | None, File()] = None,
        reference_guide: Annotated[UploadFile | None, File()] = None,
        persona_queries: Annotated[str, Form()] = "",
        use_reference: Annotated[bool, Form()] = False,
    ) -> dict[str, Any]:
        uploads: dict[str, Upload | None] = {}
        for kind, file in (
            ("persona", persona),
            ("product", product),
            (REFERENCE, reference),
            (GUIDE, reference_guide),
        ):
            uploads[kind] = Upload(file.filename or kind, await file.read()) if file else None
        state = workspace.create_run(uploads, persona_queries, use_reference=use_reference)
        return {"state": state.model_dump(mode="json"), **workspace.results(state.run_id)}

    @app.get("/api/runs")
    async def list_runs() -> dict[str, Any]:
        return {"runs": workspace.recent_runs()}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        state = workspace.state(check_run_id(run_id))
        return {"state": state.model_dump(mode="json"), **workspace.results(run_id)}

    @app.post("/api/runs/{run_id}/connection")
    async def connect_product(
        run_id: str,
        contract: Annotated[UploadFile, File()],
        url: Annotated[str, Form(max_length=2000)] = "",
    ) -> dict[str, Any]:
        """Real product: product URL + uploaded contract -> http_json adapter for execute."""
        state = workspace.state(check_run_id(run_id))
        upload = Upload(contract.filename or "contract.yaml", await contract.read())
        state = workspace.connect(state, url, upload)
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
        needed = _PREREQUISITE.get(step)
        if needed is not None and needed not in state.steps_done:
            raise ConfigError(f"Run the '{needed.value}' step before '{step.value}'")
        inputs = workspace.load_inputs(run_id)
        count = body.count if body else None
        llm = create_llm_client(settings)
        product = (
            _product_adapter(
                settings,
                llm,
                state.contract,
                workspace.connection_options(state),
                inputs,
                env=environ,
                transport=transport,
            )
            if step is Step.EXECUTE
            else UnavailableAdapter()
        )
        pipeline = SyntheticUserPipeline(
            settings, llm, product, store, logger=get_run_logger("synthetic_user.web", run_id)
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
                "reference_chars": len(reference.text) if reference else 0,
                "warnings": reference_step.warnings,
            }
        if step is Step.SCENARIOS:
            scenario_step = await pipeline.generate_scenarios(
                run_id, inputs, count=count, force=True
            )
            detail = {
                **detail,
                "accepted": len(scenario_step.accepted),
                "rejected": len(scenario_step.rejected),
                "reference_packs": len(scenario_step.packs),
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
                execute_step = await pipeline.execute_questions(run_id, inputs, force=True)
            finally:
                await pipeline.aclose()
            detail = {
                "product": _product_label(state.contract, state.connection, product.name),
                "executions": len(execute_step.executions),
            }
        elif step is Step.EVALUATE:
            evaluate_step = await pipeline.evaluate_executions(run_id, inputs, force=True)
            detail = {
                "evaluations": len(evaluate_step.evaluations),
                "bug_candidates": len(evaluate_step.bug_candidates),
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
        pipeline = SyntheticUserPipeline(
            settings,
            create_llm_client(settings),
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
    connection: HttpJsonOptions | None,
    inputs: RunInputs,
    *,
    env: Mapping[str, str],
    transport: Transport,
) -> ProductAdapter:
    """URL + uploaded contract -> the real product (http_json); mock contract -> mock product
    behind it; else the configured product adapter."""
    if connection is not None:
        return HttpJsonAdapter(connection, env=env, transport=transport)
    if contract is not None:
        return MockLLMAdapter(llm, contract, inputs.product, inputs.reference)
    return create_adapter(settings)


def _product_label(
    contract: ProductContract | None, connection: ProductConnection | None, adapter_name: str
) -> str:
    if connection is not None:
        return f"{adapter_name} ({connection.product_url})"
    return "mock (contract)" if contract is not None else adapter_name
