"""Clarification policy (ask / assume), conversation transcripts and run.log redaction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from synthetic_user.adapters.base import ProductRequest
from synthetic_user.adapters.http_common import HttpRequest, HttpResult
from synthetic_user.adapters.http_json import HttpJsonAdapter
from synthetic_user.core.config import AppConfig, ClarificationPolicy
from synthetic_user.core.trace import record_product_http, trace_redactions, trace_step
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.models.execution import (
    AnsweredResponse,
    ClarificationResponse,
    ProductResponse,
)
from synthetic_user.models.persona import Persona
from synthetic_user.models.question import QuestionCase
from synthetic_user.models.scenario import Scenario
from synthetic_user.nodes.executor.executor import (
    SINGLE_TURN_ASSUMPTION_INSTRUCTION,
    ProductExecutor,
)


class ConversationAdapter:
    """Asks one clarification, then answers; records every request it receives."""

    name = "conversation"

    def __init__(self, *, echo: bool = False) -> None:
        self.requests: list[ProductRequest] = []
        self.echo = echo

    async def send(self, req: ProductRequest) -> ProductResponse:
        self.requests.append(req)
        if self.echo:  # a product that repeats the whole prompt back
            return AnsweredResponse(text=f"You asked: {req.utterance}")
        if len(self.requests) == 1:
            return ClarificationResponse(message="Which quarter?")
        return AnsweredResponse(text=f"Answer for {req.utterance}")

    async def aclose(self) -> None:
        return None


def _question(scenario: Scenario) -> QuestionCase:
    return QuestionCase.model_validate(
        {
            "question_id": "Q_001",
            "scenario_id": scenario.id,
            "persona_id": scenario.persona_id,
            "mode": "grounded",
            "reference_pack_id": scenario.reference_pack_id,
            "utterance": "Who spent most?",
            "intended_goal": scenario.business_goal,
            "intent": {"task_type": "ranking"},
            "question_characteristics": {
                "style": "precise",
                "ambiguity_level": "low",
                "complexity": "simple",
            },
            "expectations": {"expected_behavior": "ranks", "success_criteria": ["ranks"]},
        }
    )


def _policy(settings: AppConfig, policy: ClarificationPolicy) -> AppConfig:
    product = settings.product.model_copy(update={"clarification_policy": policy})
    return settings.model_copy(update={"product": product})


async def _execute(
    adapter: Any,
    settings: AppConfig,
    scenario: Scenario,
    persona: Persona,
    llm: FakeLLMClient | None = None,
) -> Any:
    executor = ProductExecutor(adapter, settings, logging.getLogger("t"), llm=llm)
    (record,) = await executor.execute_many(
        [_question(scenario)],
        product_id="p",
        persona=persona,
        scenarios={scenario.id: scenario},
    )
    return record


async def test_ask_policy_answers_clarification_as_persona_in_same_session(
    app_config: AppConfig, sample_scenarios: list[Scenario], sample_persona: Persona
) -> None:
    adapter = ConversationAdapter()
    llm = FakeLLMClient([{"reply": "Last quarter, please."}])
    record = await _execute(
        adapter,
        _policy(app_config, ClarificationPolicy.ASK),
        sample_scenarios[0],
        sample_persona,
        llm,
    )
    assert record.outcome == "answered"
    assert [r.utterance for r in adapter.requests] == ["Who spent most?", "Last quarter, please."]
    assert len({r.session_id for r in adapter.requests}) == 1  # one stable session
    assert record.session_id == adapter.requests[0].session_id
    assert [(t.role, t.kind) for t in record.turns] == [
        ("user", "question"),
        ("product", "clarification"),
        ("user", "clarification_answer"),
        ("product", "answer"),
    ]
    assert "Which quarter?" in llm.calls[0].user and "synthetic user" in llm.calls[0].system


INSIGHT_ADAPTER = Path(__file__).resolve().parents[2] / "config/products/generate_sql.adapter.yaml"
INSIGHT_CLARIFICATION = {
    "type": "clarification",
    "result": None,
    "message": "How many airline routes would you like to see in the ranking?",
    "options": ["Top 10 routes", "Top 25 routes", "Top 50 routes", "All routes"],
    "session_id": "echoed",
}
INSIGHT_SQL = {
    "type": "sql",
    "result": "SELECT a.route FROM t a ORDER BY 1 LIMIT 10",
    "widget_title": "Top 10 routes",
    "message": "Here is the SQL query",
    "session_id": "echoed",
}


class ScriptedTransport:
    """Replies with the next canned body; records every request."""

    def __init__(self, bodies: list[dict[str, Any]]) -> None:
        self.bodies = bodies
        self.requests: list[HttpRequest] = []

    async def __call__(self, request: HttpRequest) -> HttpResult:
        self.requests.append(request)
        body = self.bodies[min(len(self.requests), len(self.bodies)) - 1]
        return HttpResult(200, json.dumps(body).encode())


def _insight(transport: ScriptedTransport) -> HttpJsonAdapter:
    """The shipped Insight Agent (generate_sql) contract, unchanged, over a stub transport."""
    options = yaml.safe_load(INSIGHT_ADAPTER.read_text(encoding="utf-8"))["options"]
    return HttpJsonAdapter.from_options(options, env={}, transport=transport)  # token in file


async def test_insight_clarification_options_are_answered_on_the_same_session(
    app_config: AppConfig, sample_scenarios: list[Scenario], sample_persona: Persona
) -> None:
    transport = ScriptedTransport([INSIGHT_CLARIFICATION, INSIGHT_SQL])
    llm = FakeLLMClient([{"reply": "Top 10 routes"}])
    record = await _execute(
        _insight(transport),
        _policy(app_config, ClarificationPolicy.ASK),
        sample_scenarios[0],
        sample_persona,
        llm,
    )
    bodies = [json.loads(r.body or b"{}") for r in transport.requests]
    assert [b["input"] for b in bodies] == ["Who spent most?", "Top 10 routes"]
    assert bodies[0]["session_id"] == bodies[1]["session_id"] == record.session_id
    # SSL off and the token written in the contract itself (no environment variable)
    assert all(not r.ssl_verify for r in transport.requests)
    assert transport.requests[0].headers["Authorization"] == "Bearer PASTE_YOUR_BLUE_TOKEN_HERE"
    assert record.outcome == "answered"
    assert record.response.first_artifact("sql").content == INSIGHT_SQL["result"]
    clarification = record.turns[1]
    assert clarification.kind == "clarification"
    assert clarification.options == INSIGHT_CLARIFICATION["options"]
    # the persona sees the offered options and is told to pick one of them verbatim
    assert '"Top 25 routes"' in llm.calls[0].user
    assert "reply with exactly one of them" in llm.calls[0].user


async def test_clarification_turns_are_capped(
    app_config: AppConfig, sample_scenarios: list[Scenario], sample_persona: Persona
) -> None:
    transport = ScriptedTransport([INSIGHT_CLARIFICATION])  # the product never stops asking
    settings = _policy(app_config, ClarificationPolicy.ASK)
    settings = settings.model_copy(
        update={"product": settings.product.model_copy(update={"max_clarification_turns": 2})}
    )
    llm = FakeLLMClient([{"reply": "Top 10 routes"}, {"reply": "All routes"}])
    record = await _execute(_insight(transport), settings, sample_scenarios[0], sample_persona, llm)
    assert len(transport.requests) == 3  # question + 2 answers, then it stops
    assert len(llm.calls) == 2
    assert record.outcome == "clarification"  # unresolved: recorded only, never judged
    assert [t.kind for t in record.turns].count("clarification_answer") == 2


async def test_ask_policy_without_llm_records_the_clarification(
    app_config: AppConfig, sample_scenarios: list[Scenario], sample_persona: Persona
) -> None:
    record = await _execute(
        ConversationAdapter(),
        _policy(app_config, ClarificationPolicy.ASK),
        sample_scenarios[0],
        sample_persona,
    )
    assert record.outcome == "clarification" and len(record.turns) == 2


async def test_assume_policy_sends_hidden_instruction_but_stores_the_question(
    app_config: AppConfig, sample_scenarios: list[Scenario], sample_persona: Persona
) -> None:
    adapter = ConversationAdapter(echo=True)
    record = await _execute(
        adapter,
        _policy(app_config, ClarificationPolicy.ASSUME),
        sample_scenarios[0],
        sample_persona,
    )
    (request,) = adapter.requests
    assert request.utterance.endswith(SINGLE_TURN_ASSUMPTION_INSTRUCTION)
    assert request.report_utterance == "Who spent most?"
    assert "canonical_utterance" not in request.model_dump()  # never sent as a field
    # An echoed instruction never reaches the stored answer or the transcript.
    assert record.submitted_question == "Who spent most?"
    assert isinstance(record.response, AnsweredResponse)
    assert SINGLE_TURN_ASSUMPTION_INSTRUCTION not in (record.response.text or "")
    assert all(SINGLE_TURN_ASSUMPTION_INSTRUCTION not in t.message for t in record.turns)


def test_run_log_redactions_and_field_limit(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    with (
        trace_step(log, "RUN_1", "execute", max_field_chars=1_000),
        trace_redactions(("question + hidden instruction", "question")),
    ):
        record_product_http(
            request={
                "method": "POST",
                "url": "https://p.example",
                "header_names": ["Authorization"],
                "body": {"q": "question + hidden instruction"},
            },
            response={"status": 200, "body": "x" * 5_000},
            error=None,
        )
    text = log.read_text(encoding="utf-8")
    assert "hidden instruction" not in text and '"q": "question"' in text
    assert "[... cut: 5,000 chars in total]" in text
