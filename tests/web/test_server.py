from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from synthetic_user.core.config import load_config
from synthetic_user.web.results_csv import COLUMNS
from synthetic_user.web.server import create_app
from synthetic_user.web.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parents[2]
PERSONA_YAML = (REPO_ROOT / "data" / "personas" / "finance_controller.yaml").read_bytes()


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    data = yaml.safe_load((REPO_ROOT / "config" / "app.fake.yaml").read_text(encoding="utf-8"))
    data["llm"]["fake_script_path"] = str(REPO_ROOT / data["llm"]["fake_script_path"])
    data["product"]["fixture_path"] = str(REPO_ROOT / data["product"]["fixture_path"])
    data["inputs"] = {k: str(REPO_ROOT / v) for k, v in data["inputs"].items()}
    data["paths"]["artifacts_dir"] = str(tmp_path / "artifacts")
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def client(config_path: Path) -> TestClient:
    return TestClient(create_app(load_config(config_path, env={})))


def _run(client: TestClient, **files: Any) -> str:
    response = client.post("/api/runs", files=files or None, data={"persona_queries": ""})
    assert response.status_code == 200, response.text
    return str(response.json()["state"]["run_id"])


def _steps(client: TestClient, run_id: str, *steps: str) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for step in steps:
        response = client.post(f"/api/runs/{run_id}/steps/{step}", json={})
        assert response.status_code == 200, (step, response.text)
        body = response.json()
    return body


# --- pages, config, samples, previews --------------------------------------------------------


def test_index_page_and_static(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200 and "Synthetic User" in page.text
    assert client.get("/static/app.js").status_code == 200


def test_config_and_samples(client: TestClient) -> None:
    config = client.get("/api/config").json()
    assert config["llm_provider"] == "fake"
    assert config["steps"] == ["index", "scenarios", "questions", "execute", "evaluate"]
    samples = client.get("/api/samples").json()
    assert set(samples) == {"persona", "product", "catalog"}
    assert samples["persona"]["parsed"]["id"] == "finance_controller"
    assert "finance_controller" in samples["persona"]["raw"]


def test_preview_valid_and_invalid_upload(client: TestClient) -> None:
    ok = client.post("/api/preview/persona", files={"file": ("p.yaml", PERSONA_YAML)})
    assert ok.status_code == 200 and ok.json()["parsed"]["id"] == "finance_controller"
    bad = client.post("/api/preview/persona", files={"file": ("p.yaml", b"id: x\nversion: 1\n")})
    assert bad.status_code == 400
    assert "Persona" in bad.json()["detail"] and "identity" in bad.json()["detail"]
    wrong_kind = client.post("/api/preview/nope", files={"file": ("p.yaml", PERSONA_YAML)})
    assert wrong_kind.status_code == 400


# --- runs and inputs -------------------------------------------------------------------------


def test_create_run_with_upload_and_queries(client: TestClient, config_path: Path) -> None:
    queries = "How much did we spend on travel?\n\nHow much did we spend on travel?\nTop vendors?"
    response = client.post(
        "/api/runs",
        files={"persona": ("my_persona.yaml", PERSONA_YAML)},
        data={"persona_queries": queries},
    )
    assert response.status_code == 200, response.text
    state = response.json()["state"]
    assert state["sources"] == {
        "persona": "uploaded: my_persona.yaml",
        "product": "sample",
        "catalog": "sample",
    }
    assert state["persona_queries"] == ["How much did we spend on travel?", "Top vendors?"]

    settings = load_config(config_path, env={})
    from synthetic_user.storage.json_store import JsonArtifactStore

    inputs = Workspace(settings, JsonArtifactStore(settings.paths.artifacts_dir)).load_inputs(
        state["run_id"]
    )
    assert inputs.persona.example_utterances[-2:] == [
        "How much did we spend on travel?",
        "Top vendors?",
    ]


def test_create_run_rejects_bad_upload_and_too_many_queries(client: TestClient) -> None:
    bad = client.post("/api/runs", files={"catalog": ("c.yaml", b"tables: [")}, data={})
    assert bad.status_code == 400 and "not valid YAML" in bad.json()["detail"]
    many = client.post("/api/runs", data={"persona_queries": "\n".join(f"q{i}" for i in range(60))})
    assert many.status_code == 400 and "At most 50" in many.json()["detail"]


# --- contract ---------------------------------------------------------------------------------


def test_contract_request(client: TestClient) -> None:
    run_id = _run(client)
    bad = client.post(f"/api/runs/{run_id}/contract", json={"url": "ftp://x"})
    assert bad.status_code == 400 and "http(s)" in bad.json()["detail"]
    ok = client.post(f"/api/runs/{run_id}/contract", json={"url": "https://nl2sql.example.com"})
    assert ok.status_code == 200
    contract = ok.json()["contract"]
    assert contract["mock"] is True
    assert contract["product_id"] == "sample_nl2sql"
    assert contract["ask"]["path"] == "/v1/ask"
    assert ok.json()["state"]["contract"]["product_url"] == "https://nl2sql.example.com"


# --- pipeline through the API -------------------------------------------------------------------


def test_full_flow_with_mock_product_fills_one_csv(client: TestClient) -> None:
    run_id = _run(client)
    client.post(f"/api/runs/{run_id}/contract", json={"url": "https://nl2sql.example.com"})

    after_scenarios = _steps(client, run_id, "index", "scenarios")
    assert len(after_scenarios["rows"]) == 3
    assert all(r["scenario_id"] and not r["question_id"] for r in after_scenarios["rows"])
    assert after_scenarios["rows"][0]["cheat_joins"].endswith("(verified)")

    after_questions = _steps(client, run_id, "questions")
    assert all(r["question"] for r in after_questions["rows"])

    after_execute = _steps(client, run_id, "execute")
    by_outcome = {r["product_outcome"]: r for r in after_execute["rows"]}
    assert set(by_outcome) == {"sql_generated", "clarification", "denied"}
    assert "JOIN employees" in by_outcome["sql_generated"]["generated_sql"]
    assert by_outcome["denied"]["product_message"] == "Forecasting is not supported yet."
    assert after_execute["detail"]["product"] == "mock (contract)"

    final = _steps(client, run_id, "evaluate")
    verdicts = sorted(r["final_verdict"] or "-" for r in final["rows"])
    assert verdicts == ["-", "fail", "pass"]
    failed = next(r for r in final["rows"] if r["final_verdict"] == "fail")
    assert failed["bug_ids"] == "BUG_001" and "join_correctness" in failed["failed_checks"]
    assert final["state"]["steps_done"] == [
        "index",
        "scenarios",
        "questions",
        "execute",
        "evaluate",
    ]

    download = client.get(f"/api/runs/{run_id}/results.csv")
    assert download.status_code == 200
    rows = list(csv.DictReader(io.StringIO(download.text)))
    assert list(rows[0]) == COLUMNS
    assert [r["question_id"] for r in rows] == [r["question_id"] for r in final["rows"]]
    assert {"results.csv", "evaluations.json", "ui_state.json"} <= set(final["artifacts"])
    assert client.get(f"/api/runs/{run_id}/artifacts/evaluations.json").status_code == 200


def test_rerunning_a_step_resets_later_steps(client: TestClient) -> None:
    run_id = _run(client)
    _steps(client, run_id, "index", "scenarios", "questions", "execute", "evaluate")
    body = _steps(client, run_id, "questions")
    assert body["state"]["steps_done"] == ["index", "scenarios", "questions"]
    assert all(not r["product_outcome"] for r in body["rows"])


def test_execute_without_contract_uses_fixture_product(client: TestClient) -> None:
    run_id = _run(client)
    body = _steps(client, run_id, "index", "scenarios", "questions", "execute")
    assert body["detail"]["product"] == "fixture"
    assert {r["product_outcome"] for r in body["rows"]} == {
        "sql_generated",
        "clarification",
        "denied",
    }


def test_step_order_and_bad_ids(client: TestClient) -> None:
    run_id = _run(client)
    early = client.post(f"/api/runs/{run_id}/steps/scenarios", json={})
    assert early.status_code == 400 and "reference index first" in early.json()["detail"]
    assert client.post(f"/api/runs/{run_id}/steps/bogus", json={}).status_code == 422
    assert client.get("/api/runs/not-a-run").status_code == 400
    assert client.get("/api/runs/RUN_20200101_000000_abcd").status_code == 404
    assert client.get(f"/api/runs/{run_id}/artifacts/..%2Fsecret").status_code in (400, 404)
    assert client.get(f"/api/runs/{run_id}/artifacts/nope.json").status_code == 400
