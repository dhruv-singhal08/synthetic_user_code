"""Clarification policy (ask / assume), conversation transcripts and run.log redaction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from synthetic_user.adapters.base import ProductRequest
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
from synthetic_user.nodes.product_executor import (
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
