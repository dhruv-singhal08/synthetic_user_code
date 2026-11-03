"""Metric telemetry capture: context in the response or from a context API (same execution),
citations, tool trace, typed response kind - normalized into one model by the executor."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import pytest

from synthetic_user.adapters.base import ProductRequest
from synthetic_user.adapters.http_common import HttpRequest, HttpResult, ResponseMapping
from synthetic_user.adapters.http_json import HttpJsonAdapter
from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import ConfigError
from synthetic_user.models.execution import AnsweredResponse
from synthetic_user.models.question import QuestionCase
from synthetic_user.nodes.executor.executor import ProductExecutor

CHUNKS = [
    {"id": "k2", "body": "Feb: 90 complaints", "doc": "report.pdf", "score_rank": 2},
    {"id": "k1", "body": "Jan: 120 complaints", "doc": "report.pdf", "score_rank": 1},
]
BODY = {
    "answer": "January 120, February 90 [1].",
    "kind": "answer",
    "request_id": "req-42",
    "chunks": CHUNKS,
    "citations": [{"chunk": "k1", "text": "January 120"}, {"chunk": "zz", "text": "made up"}],
    "tool_calls": [
        {"name": "aggregate_feedback", "input": {"month": "jan"}, "output": {"count": 120}},
        {"input": {"orphan": True}},  # no name: not an observable call
    ],
}
MAPPING: dict[str, Any] = {
    "text": "answer",
    "correlation_id": "request_id",
    "unable_to_answer_when": "kind == cannot_answer",
    "context": {
        "path": "chunks",
        "chunk_id": "id",
        "text": "body",
        "source_id": "doc",
        "rank": "score_rank",
    },
    "citations": {"path": "citations", "chunk_id": "chunk", "claim": "text"},
    "tool_trace": {"path": "tool_calls", "name": "name", "arguments": "input", "result": "output"},
}


class Transport:
    """Stub JSON transport: replies by URL; records every request."""

    def __init__(self, replies: dict[str, Callable[[HttpRequest], HttpResult]]) -> None:
        self.replies = replies
        self.requests: list[HttpRequest] = []

    async def __call__(self, request: HttpRequest) -> HttpResult:
        self.requests.append(request)
        for prefix, reply in self.replies.items():
            if request.url.startswith(prefix):
                return reply(request)
        raise AssertionError(f"unexpected request {request.url}")


def ok(body: Any) -> Callable[[HttpRequest], HttpResult]:
    return lambda _r: HttpResult(200, json.dumps(body).encode())


def adapter(transport: Transport, mapping: dict[str, Any], **options: Any) -> HttpJsonAdapter:
    return HttpJsonAdapter.from_options(
        {
            "url": "https://p.example.com/ask",
            "request_template": {"q": "{utterance}", "trace": "{uuid}"},
            "response_mapping": mapping,
            **options,
        },
        env={},
        transport=transport,
    )


def request() -> ProductRequest:
    return ProductRequest(question_id="Q_001", utterance="Compare Jan and Feb", session_id="S1")


async def test_context_citations_trace_and_kind_from_the_response() -> None:
    response = await adapter(Transport({"https://p": ok(BODY)}), MAPPING).send(request())
    assert isinstance(response, AnsweredResponse)
    assert response.response_classification == "answered"  # typed, not inferred by an LLM
    assert response.correlation_id == "req-42"
    context = response.retrieved_context
    assert context is not None and context.availability == "available"
    assert [(c.chunk_id, c.rank) for c in context.chunks] == [("k1", 1), ("k2", 2)]  # by rank
    assert context.rank_exposed and context.citations_exposed
    assert [c.chunk_id for c in context.citations] == ["k1", "zz"]
    assert response.tool_trace is not None
    assert [(t.order, t.name, t.result) for t in response.tool_trace] == [
        (1, "aggregate_feedback", {"count": 120})
    ]


async def test_typed_inability_inside_an_answered_payload() -> None:
    body = {**BODY, "kind": "cannot_answer"}
    response = await adapter(Transport({"https://p": ok(body)}), MAPPING).send(request())
    assert response.outcome == "answered"
    assert isinstance(response, AnsweredResponse)
    assert response.response_classification == "unable_to_answer"


@pytest.mark.parametrize(
    ("chunks", "availability", "error"),
    [
        (None, "unavailable", "no retrieved context at 'chunks'"),
        ("oops", "failed", "is not a list"),
        ([], "available", None),  # zero chunks retrieved: different from missing telemetry
        ([{"id": "a"}], "failed", "no text at 'body'"),
        ([{"id": "a", "body": "x", "score_rank": "first"}], "failed", "not a positive int"),
        (
            [{"id": "a", "body": "x", "score_rank": 1}, {"id": "a", "body": "y", "score_rank": 2}],
            "failed",
            "duplicate chunk ids",
        ),
    ],
)
async def test_runtime_content_validation(
    chunks: Any, availability: str, error: str | None
) -> None:
    body = {key: value for key, value in BODY.items() if key != "chunks"}
    if chunks is not None:
        body["chunks"] = chunks
    response = await adapter(Transport({"https://p": ok(body)}), MAPPING).send(request())
    assert isinstance(response, AnsweredResponse)
    context = response.retrieved_context
    assert context is not None and context.availability == availability
    assert (context.error is None) if error is None else (error in (context.error or ""))
    if availability == "available":
        assert context.chunks == []


def test_mapping_is_validated_at_configuration_time() -> None:
    with pytest.raises(ValueError, match="rank path OR order_is_rank"):
        ResponseMapping.model_validate(
            {**MAPPING, "context": {**MAPPING["context"], "order_is_rank": True}}
        )
    with pytest.raises(ValueError, match="map chunk_id or source_id"):
        ResponseMapping.model_validate({**MAPPING, "citations": {"path": "c"}})
    with pytest.raises(ConfigError, match="choose ONE context source"):
        adapter(
            Transport({}),
            MAPPING,
            context_api={"url": "https://p.example.com/ctx", "context": {"path": "c", "text": "t"}},
        )


async def test_context_api_is_correlated_to_the_same_execution(
    app_config: AppConfig, sample_scenarios: Any
) -> None:
    answer_only = {k: v for k, v in MAPPING.items() if k not in ("context", "citations")}
    context_body = {"chunks": [{"id": "k1", "body": "Jan: 120", "pos": 1}]}
    transport = Transport(
        {"https://p.example.com/ask": ok(BODY), "https://p.example.com/ctx/": ok(context_body)}
    )
    product = adapter(
        transport,
        answer_only,
        context_api={
            "url": "https://p.example.com/ctx/{correlation_id}",
            "context": {"path": "chunks", "chunk_id": "id", "text": "body", "rank": "pos"},
        },
    )
    record = (await _execute(app_config, product, sample_scenarios))[0]
    assert transport.requests[1].url == "https://p.example.com/ctx/req-42"  # product's own id
    assert record.outcome == "answered"
    assert record.context.availability == "available"
    assert record.context.source_mode == "context_api"
    assert record.context.correlation_id == "req-42"
    assert [c.chunk_id for c in record.context.chunks] == ["k1"]
    # without a product id, the request {uuid} sent with the question is the correlation id
    no_id = {k: v for k, v in BODY.items() if k != "request_id"}
    transport = Transport(
        {"https://p.example.com/ask": ok(no_id), "https://p.example.com/ctx/": ok(context_body)}
    )
    product = adapter(
        transport,
        answer_only,
        context_api={
            "url": "https://p.example.com/ctx/{correlation_id}",
            "context": {"path": "chunks", "text": "body"},
        },
    )
    record = (await _execute(app_config, product, sample_scenarios))[0]
    sent_uuid = json.loads(transport.requests[0].body or b"{}")["trace"]
    assert transport.requests[1].url == f"https://p.example.com/ctx/{sent_uuid}"
    assert record.context.correlation_id == sent_uuid


async def test_context_api_failure_never_changes_the_answer(
    app_config: AppConfig, sample_scenarios: Any
) -> None:
    answer_only = {k: v for k, v in MAPPING.items() if k not in ("context", "citations")}
    transport = Transport(
        {
            "https://p.example.com/ask": ok(BODY),
            "https://p.example.com/ctx": lambda _r: HttpResult(503, b"down"),
        }
    )
    product = adapter(
        transport,
        answer_only,
        context_api={"url": "https://p.example.com/ctx", "context": {"path": "c", "text": "t"}},
    )
    record = (await _execute(app_config, product, sample_scenarios))[0]
    assert record.outcome == "answered"
    assert record.context.availability == "failed"
    assert record.context.error == "context API HTTP 503: down"


async def test_executor_moves_bounds_and_resolves_response_context(
    app_config: AppConfig, sample_scenarios: Any
) -> None:
    many = [{"id": f"k{i}", "body": "x" * 200, "score_rank": i} for i in range(1, 6)]
    body = {**BODY, "chunks": many, "citations": [{"chunk": "k1"}, {"chunk": "k5"}]}
    settings = app_config.model_copy(
        update={
            "product": app_config.product.model_copy(
                update={"context_max_chunks": 3, "context_max_chunk_chars": 100}
            )
        }
    )
    product = adapter(Transport({"https://p": ok(body)}), MAPPING)
    record = (await _execute(settings, product, sample_scenarios))[0]
    assert isinstance(record.response, AnsweredResponse)
    assert record.response.retrieved_context is None  # moved to the execution record
    context = record.context
    assert context.source_mode == "response_field" and context.truncated
    assert [c.chunk_id for c in context.chunks] == ["k1", "k2", "k3"]
    assert all(len(c.text) == 100 for c in context.chunks)
    assert [c.resolved for c in context.citations] == [True, False]  # k5 was cut


async def test_non_answers_have_no_context(app_config: AppConfig, sample_scenarios: Any) -> None:
    body = {"answer": "Which month?", "kind": "clarify"}
    mapping = {**MAPPING, "clarification_when": "kind == clarify"}
    record = (
        await _execute(
            app_config, adapter(Transport({"https://p": ok(body)}), mapping), sample_scenarios
        )
    )[0]
    assert record.outcome == "clarification"
    assert record.context.availability == "not_requested"


async def _execute(settings: AppConfig, product: HttpJsonAdapter, scenarios: Any) -> Any:
    scenario = scenarios[0]
    question = QuestionCase.model_validate(
        {
            "question_id": "Q_001",
            "scenario_id": scenario.id,
            "persona_id": scenario.persona_id,
            "mode": "grounded",
            "reference_pack_id": scenario.reference_pack_id,
            "utterance": "Compare Jan and Feb",
            "intended_goal": scenario.business_goal,
            "intent": {"task_type": "comparison"},
            "question_characteristics": {
                "style": "precise",
                "ambiguity_level": "low",
                "complexity": "simple",
            },
            "expectations": {"expected_behavior": "b", "success_criteria": {"general": ["c"]}},
        }
    )
    executor = ProductExecutor(product, settings, logging.getLogger("t"))
    return await executor.execute_many([question], product_id="p")
