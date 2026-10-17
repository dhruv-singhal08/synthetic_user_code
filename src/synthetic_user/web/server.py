"""FastAPI server: JSON endpoints for the pipeline + the single-page UI in ./static.

Local tool (binds 127.0.0.1 by default). Each step endpoint runs one pipeline step,
refreshes results.csv, and returns the current results table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from synthetic_user.adapters.base import ProductAdapter, UnavailableAdapter
from synthetic_user.adapters.contract import ProductContract, request_contract
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
from synthetic_user.pipeline import RunInputs, Step, SyntheticUserPipeline
from synthetic_user.storage.json_store import JsonArtifactStore
from synthetic_user.web.workspace import INPUT_KINDS, RESULTS_CSV, Upload, Workspace, parse_upload

STATIC_DIR = Path(__file__).parent / "static"


class StepRequest(BaseModel):
    count: int | None = Field(default=None, ge=1, le=50)


class ContractRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2000)


def create_app(settings: AppConfig) -> FastAPI:
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
            "sql_dialect": settings.sql.dialect,
            "scenario_count": settings.generation.scenario.scenario_count,
            "question_count": settings.generation.question.question_count,
            "steps": [s.value for s in Step],
        }

    @app.get("/api/samples")
    async def samples() -> dict[str, Any]:
        """Only the persona has a sample; product description and reference index are
        upload-only."""
        result: dict[str, Any] = dict.fromkeys(INPUT_KINDS)
        sample = workspace.sample("persona")
        if sample is not None:
            text, model = sample
            result["persona"] = {"raw": text, "parsed": model.model_dump(mode="json")}
        return result

    @app.post("/api/preview/{kind}")
    async def preview(kind: str, file: Annotated[UploadFile, File()]) -> dict[str, Any]:
        upload = Upload(file.filename or kind, await file.read())
        model = parse_upload(kind, upload)
        return {"raw": upload.content.decode("utf-8"), "parsed": model.model_dump(mode="json")}

    # --- runs ----------------------------------------------------------------------------

    @app.post("/api/runs")
    async def create_run(
        persona: Annotated[UploadFile | None, File()] = None,
        product: Annotated[UploadFile | None, File()] = None,
        reference_index: Annotated[UploadFile | None, File()] = None,
        persona_queries: Annotated[str, Form()] = "",
        use_reference_index: Annotated[bool, Form()] = False,
    ) -> dict[str, Any]:
        uploads: dict[str, Upload | None] = {}
        for kind, file in (
            ("persona", persona),
            ("product", product),
            ("reference_index", reference_index),
        ):
            uploads[kind] = Upload(file.filename or kind, await file.read()) if file else None
        state = workspace.create_run(
            uploads, persona_queries, use_reference_index=use_reference_index
        )
        return {"state": state.model_dump(mode="json"), **workspace.results(state.run_id)}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        state = workspace.state(check_run_id(run_id))
        return {"state": state.model_dump(mode="json"), **workspace.results(run_id)}

    @app.post("/api/runs/{run_id}/contract")
    async def connect_product(run_id: str, body: ContractRequest) -> dict[str, Any]:
        state = workspace.state(check_run_id(run_id))
        inputs = workspace.load_inputs(run_id)
        contract = request_contract(body.url, inputs.product.product_id, settings.sql.dialect)
        state = state.model_copy(update={"contract": contract})
        workspace.save_state(state)
        return {
            "state": state.model_dump(mode="json"),
            "contract": contract.model_dump(mode="json"),
        }

    @app.post("/api/runs/{run_id}/steps/{step}")
    async def run_step(run_id: str, step: Step, body: StepRequest | None = None) -> dict[str, Any]:
        state = workspace.state(check_run_id(run_id))
        inputs = workspace.load_inputs(run_id)
        count = body.count if body else None
        llm = create_llm_client(settings)
        product = (
            _product_adapter(settings, llm, state.contract, inputs)
            if step is Step.EXECUTE
            else UnavailableAdapter()
        )
        pipeline = SyntheticUserPipeline(
            settings, llm, product, store, logger=get_run_logger("synthetic_user.web", run_id)
        )
        detail: dict[str, Any] = {}

        if step in (Step.INDEX, Step.SCENARIOS):
            # The index check (no LLM) runs automatically before scenarios.
            index_step = pipeline.check_reference_index(inputs)
            index_id = index_step.index.versioned_id if index_step.index else None
            state = workspace.mark_done(
                state.model_copy(update={"reference_index_id": index_id}), Step.INDEX
            )
            detail = {
                "mode": index_step.mode.value,
                "reference_index_id": index_id,
                "tables": len(index_step.index.tables) if index_step.index else 0,
                "warnings": index_step.warnings,
            }
        if step is Step.SCENARIOS:
            scenario_step = await pipeline.generate_scenarios(
                run_id, inputs, count=count, force=True
            )
            detail = {
                **detail,
                "accepted": len(scenario_step.accepted),
                "rejected": len(scenario_step.rejected),
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
                "product": "mock (contract)" if state.contract else product.name,
                "executions": len(execute_step.executions),
            }
        else:
            evaluate_step = await pipeline.evaluate_executions(run_id, inputs, force=True)
            detail = {
                "evaluations": len(evaluate_step.evaluations),
                "bug_candidates": len(evaluate_step.bug_candidates),
            }

        state = workspace.mark_done(state, step)
        return {
            "state": state.model_dump(mode="json"),
            "step": step.value,
            "detail": detail,
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
    settings: AppConfig, llm: LLMClient, contract: ProductContract | None, inputs: RunInputs
) -> ProductAdapter:
    """Contract requested from a URL -> mock product behind that contract; else the
    configured product adapter."""
    if contract is not None:
        if inputs.reference_index is None:
            raise ConfigError("The mock product needs a reference index (its database schema)")
        return MockLLMAdapter(llm, contract, inputs.reference_index, inputs.product)
    return create_adapter(settings)
