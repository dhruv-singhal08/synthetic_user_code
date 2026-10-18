from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from synthetic_user.adapters.http_json import HttpRequest, HttpResult
from synthetic_user.core.config import load_config
from synthetic_user.web.results_csv import COLUMNS
from synthetic_user.web.server import create_app
from synthetic_user.web.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parents[2]
PERSONA_YAML = (REPO_ROOT / "data" / "personas" / "finance_controller.yaml").read_bytes()
PRODUCT_YAML = (REPO_ROOT / "data" / "products" / "sample_product.yaml").read_bytes()
CHATBOT_YAML = (REPO_ROOT / "data" / "products" / "sample_chatbot.yaml").read_bytes()
INDEX_JSON = (REPO_ROOT / "data" / "references" / "sample_card_spend_index.json").read_bytes()
CONTRACT_YAML = b"""
method: POST
headers: {Authorization: "Bearer ${PRODUCT_API_TOKEN}"}
request_template: {question: "{utterance}"}
response_mapping:
  text: data.answer
  artifacts: [{kind: sql, path: data.sql}]
  denied_when: "data.status == unsupported"
"""
TOKEN = "tok-very-secret-123"


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    data = yaml.safe_load((REPO_ROOT / "config" / "app.fake.yaml").read_text(encoding="utf-8"))
    data["llm"]["fake_script_path"] = str(REPO_ROOT / data["llm"]["fake_script_path"])
    data["product_adapter"] = {
        "type": "fixture",
        "options": {
            "path": str(REPO_ROOT / "data/fixtures/product_responses/sample_responses.json")
        },
    }
    data["inputs"] = {
        k: ([str(REPO_ROOT / x) for x in v] if isinstance(v, list) else str(REPO_ROOT / v))
        for k, v in data["inputs"].items()
    }
    data["paths"]["artifacts_dir"] = str(tmp_path / "artifacts")
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def client(config_path: Path) -> TestClient:
    return TestClient(create_app(load_config(config_path, env={})))


def _run(client: TestClient, *, grounded: bool = True, product: bytes = PRODUCT_YAML) -> str:
    """New run: sample persona, uploaded product, index uploaded and switched on (grounded)."""
    files: dict[str, Any] = {"product": ("product.yaml", product)}
    if grounded:
        files["reference_index"] = ("index.json", INDEX_JSON)
    response = client.post(
        "/api/runs",
        files=files,
        data={"persona_queries": "", "use_reference_index": "true" if grounded else "false"},
    )
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
    # Only the persona has a sample; product description and index are upload-only.
    assert samples["product"] is None and samples["reference_index"] is None
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
        files={
            "persona": ("my_persona.yaml", PERSONA_YAML),
            "product": ("my_product.yaml", PRODUCT_YAML),
            "reference_index": ("ignored.json", INDEX_JSON),
        },
        data={"persona_queries": queries},  # use_reference_index defaults to off
    )
    assert response.status_code == 200, response.text
    state = response.json()["state"]
    assert state["sources"] == {
        "persona": "uploaded: my_persona.yaml",
        "product": "uploaded: my_product.yaml",
        "reference_index": "off",
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
    assert inputs.reference_index is None  # switched off -> ungrounded


def test_create_run_rejects_bad_upload_and_too_many_queries(client: TestClient) -> None:
    product = {"product": ("p.yaml", PRODUCT_YAML)}
    bad = client.post(
        "/api/runs",
        files={**product, "reference_index": ("i.json", b"{oops")},
        data={"use_reference_index": "true"},
    )
    assert bad.status_code == 400 and "not valid JSON" in bad.json()["detail"]
    many = client.post(
        "/api/runs",
        files=product,
        data={"persona_queries": "\n".join(f"q{i}" for i in range(60))},
    )
    assert many.status_code == 400 and "At most 50" in many.json()["detail"]


def test_create_run_needs_uploaded_product_and_index_when_switched_on(
    client: TestClient,
) -> None:
    no_product = client.post("/api/runs", data={})
    assert no_product.status_code == 400
    assert "Upload a product description" in no_product.json()["detail"]
    no_index = client.post(
        "/api/runs",
        files={"product": ("p.yaml", PRODUCT_YAML)},
        data={"use_reference_index": "true"},
    )
    assert no_index.status_code == 400 and "switch it off" in no_index.json()["detail"]


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
    assert ok.json()["state"]["connection"] is None


# --- real product connection: URL + uploaded contract ----------------------------------------


def _connect(client: TestClient, run_id: str, contract: bytes, url: str) -> Any:
    return client.post(
        f"/api/runs/{run_id}/connection",
        files={"contract": ("contract.yaml", contract)},
        data={"url": url},
    )


def test_contract_template_is_a_valid_contract(client: TestClient) -> None:
    template = client.get("/api/contract-template")
    assert template.status_code == 200 and "response_mapping" in template.text
    run_id = _run(client)
    ok = _connect(client, run_id, template.content, "https://product.example.com/v1/ask")
    assert ok.status_code == 200, ok.text


def test_connection_validates_and_hides_header_values(client: TestClient) -> None:
    run_id = _run(client)
    client.post(f"/api/runs/{run_id}/contract", json={"url": "https://mock.example.com"})

    no_url = _connect(client, run_id, CONTRACT_YAML, "")
    assert no_url.status_code == 400 and "product URL" in no_url.json()["detail"]
    bad_url = _connect(client, run_id, CONTRACT_YAML, "ftp://x")
    assert bad_url.status_code == 400 and "http(s)" in bad_url.json()["detail"]
    no_mapping = _connect(client, run_id, b"method: POST\n", "https://p.example.com")
    assert no_mapping.status_code == 400 and "response_mapping" in no_mapping.json()["detail"]

    literal = CONTRACT_YAML.replace(b"${PRODUCT_API_TOKEN}", TOKEN.encode())
    ok = _connect(client, run_id, literal, "https://p.example.com/ask")
    assert ok.status_code == 200, ok.text
    state = ok.json()["state"]
    assert state["contract"] is None  # the mock contract is replaced
    assert state["connection"]["product_url"] == "https://p.example.com/ask"
    assert state["connection"]["header_names"] == ["Authorization"]
    assert TOKEN not in ok.text and TOKEN not in client.get(f"/api/runs/{run_id}").text

    off = client.delete(f"/api/runs/{run_id}/connection")
    assert off.status_code == 200 and off.json()["state"]["connection"] is None


def test_execute_through_connected_product(config_path: Path) -> None:
    sent: list[HttpRequest] = []

    async def transport(request: HttpRequest) -> HttpResult:
        sent.append(request)
        body = {"data": {"answer": "Travel spend was 1.2M USD.", "sql": "SELECT 1"}}
        return HttpResult(200, json.dumps(body).encode())

    app = create_app(
        load_config(config_path, env={}), env={"PRODUCT_API_TOKEN": TOKEN}, transport=transport
    )
    client = TestClient(app)
    run_id = _run(client)
    assert _connect(client, run_id, CONTRACT_YAML, "https://p.example.com/ask").status_code == 200
    body = _steps(client, run_id, "index", "scenarios", "questions", "execute")

    assert body["detail"]["product"] == "http_json (https://p.example.com/ask)"
    assert sent and all(r.url == "https://p.example.com/ask" for r in sent)
    assert sent[0].headers["Authorization"] == f"Bearer {TOKEN}"
    asked = {json.loads(r.body or b"{}")["question"] for r in sent}
    rows = [r for r in body["rows"] if r["question_id"]]
    assert asked == {r["question"] for r in rows}
    assert all(r["product_outcome"] == "answered" and r["adapter"] == "http_json" for r in rows)
    assert all(r["generated_sql"] == "SELECT 1" for r in rows)
    assert TOKEN not in json.dumps(body)


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
    assert set(by_outcome) == {"answered", "clarification", "denied"}
    assert "JOIN employees" in by_outcome["answered"]["generated_sql"]
    assert by_outcome["denied"]["product_message"] == "Forecasting is not supported yet."
    assert after_execute["detail"]["product"] == "mock (contract)"

    final = _steps(client, run_id, "evaluate")
    verdicts = sorted(r["final_verdict"] or "-" for r in final["rows"])
    assert verdicts == ["fail", "pass", "pass"]  # the justified denial passes (C10)
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
        "answered",
        "clarification",
        "denied",
    }


def test_ungrounded_run_has_no_cheat_sheet(client: TestClient) -> None:
    run_id = _run(client, grounded=False, product=CHATBOT_YAML)
    state = client.get(f"/api/runs/{run_id}").json()["state"]
    assert state["sources"]["reference_index"] == "off"
    body = _steps(client, run_id, "scenarios", "questions")
    assert body["rows"] and all(r["mode"] == "ungrounded" for r in body["rows"])
    assert all(not r["cheat_tables"] for r in body["rows"])
    assert all(r["question"] for r in body["rows"])


def test_step_order_and_bad_ids(client: TestClient) -> None:
    run_id = _run(client)
    early = client.post(f"/api/runs/{run_id}/steps/questions", json={})
    assert early.status_code == 400
    assert early.json()["detail"] == "Run the 'scenarios' step before 'questions'"
    assert client.post(f"/api/runs/{run_id}/steps/bogus", json={}).status_code == 422
    assert client.get("/api/runs/not-a-run").status_code == 400
    assert client.get("/api/runs/RUN_20200101_000000_abcd").status_code == 404
    assert client.get(f"/api/runs/{run_id}/artifacts/..%2Fsecret").status_code in (400, 404)
    assert client.get(f"/api/runs/{run_id}/artifacts/nope.json").status_code == 400
