"""Run log (D202): one readable file per run with every LLM / product call, in and out."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import LLMOutputValidationError
from synthetic_user.core.logging import get_run_logger
from synthetic_user.core.trace import (
    MAX_FIELD_CHARS,
    record_product_call,
    trace_step,
    trace_subject,
)
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.llm.json_utils import extract_json_object
from synthetic_user.llm.structured import generate_structured
from synthetic_user.llm.tracing import TracingLLMClient
from synthetic_user.models.base import StrictModel

RUN = "RUN_20261019_120000_abcd"


class Answer(StrictModel):
    value: int


async def test_prompts_outputs_retries_and_raw_text_are_logged(tmp_path: Path) -> None:
    llm = TracingLLMClient(FakeLLMClient(["not json at all", {"value": "x"}, {"value": 3}]))
    path = tmp_path / "run.log"
    with trace_step(path, RUN, "questions"), trace_subject("SCN_001"):
        result = await generate_structured(llm, "THE SYSTEM", "THE USER", Answer, max_retries=2)
    assert result.value == 3
    log = path.read_text(encoding="utf-8")
    assert f"run {RUN}  step questions STARTED" in log and "step questions FINISHED" in log
    assert log.count("LLM CALL  SCN_001  Answer attempt") == 3
    assert "Answer attempt 1 · fake/fake" in log and "Answer attempt 3" in log
    assert log.count(">>> SYSTEM PROMPT\nTHE SYSTEM") == 3
    assert ">>> USER PROMPT\nTHE USER\n" in log
    assert "previous output invalid" in log  # the retry prompt exactly as sent
    assert ">>> RAW OUTPUT (could not be used)\nnot json at all" in log
    assert "[questions #1] VALIDATION Answer attempt 1: REJECTED -> no JSON object found" in log
    assert "[questions #2] VALIDATION Answer attempt 2: REJECTED -> " in log
    assert "[questions #3] VALIDATION Answer attempt 3: ACCEPTED" in log
    assert '>>> OUTPUT\n{\n  "value": 3\n}' in log


async def test_steps_append_to_one_file_with_the_run_log_lines(tmp_path: Path) -> None:
    path = tmp_path / "run.log"
    logger = get_run_logger("synthetic_user.test", RUN)
    other = get_run_logger("synthetic_user.test", "RUN_20261019_120000_ffff")
    logging.getLogger("synthetic_user").setLevel(logging.INFO)
    try:
        with trace_step(path, RUN, "execute"), trace_subject("Q_001"):
            record_product_call(
                adapter="fixture",
                question="Who spent most?",
                outcome="answered",
                answer={"text": "Ann"},
                latency_ms=12.3,
            )
            logger.info("Executed 1 question(s)")
            other.info("another run's message")
        with trace_step(path, RUN, "evaluate"):
            pass
    finally:
        logging.getLogger("synthetic_user").setLevel(logging.NOTSET)
    log = path.read_text(encoding="utf-8")
    assert (
        "[execute #1]" in log and "PRODUCT CALL  Q_001  fixture adapter · answered · 12 ms" in log
    )
    assert ">>> QUESTION SENT\nWho spent most?" in log and '"text": "Ann"' in log
    assert "INFO Executed 1 question(s)" in log
    assert "another run's message" not in log  # only this run's log lines
    assert log.index("step execute FINISHED") < log.index("step evaluate STARTED")


async def test_nothing_is_written_without_an_active_run_log(tmp_path: Path) -> None:
    llm = TracingLLMClient(FakeLLMClient([{"value": 1}]))
    assert (await generate_structured(llm, "s", "u", Answer, max_retries=0)).value == 1
    record_product_call(adapter="a", question="q", outcome="answered", answer=None, latency_ms=1)
    assert not list(tmp_path.iterdir())


async def test_wrap_is_idempotent_and_passes_names_through() -> None:
    inner = FakeLLMClient(model_name="m1")
    traced = TracingLLMClient.wrap(inner)
    assert TracingLLMClient.wrap(traced) is traced
    assert (traced.provider_name, traced.model_name) == ("fake", "m1")


async def test_provider_error_is_logged_and_raised(tmp_path: Path) -> None:
    llm = TracingLLMClient(FakeLLMClient([]))  # empty queue -> provider error
    path = tmp_path / "run.log"
    with trace_step(path, RUN, "evaluate"), pytest.raises(Exception, match="queue is empty"):
        await llm.generate_text("s", "u")
    log = path.read_text(encoding="utf-8")
    assert ">>> ERROR\n" in log and "queue is empty" in log


def test_long_fields_are_cut(tmp_path: Path) -> None:
    path = tmp_path / "run.log"
    with trace_step(path, RUN, "execute"):
        record_product_call(
            adapter="a",
            question="x" * (MAX_FIELD_CHARS + 10),
            outcome="answered",
            answer=None,
            latency_ms=None,
        )
    assert f"[... cut: {MAX_FIELD_CHARS + 10:,} chars in total]" in path.read_text("utf-8")


def test_parse_error_keeps_the_raw_text() -> None:
    with pytest.raises(LLMOutputValidationError) as info:
        extract_json_object("Sure! here you go: {broken")
    assert info.value.raw_output == "Sure! here you go: {broken"
    assert "broken" not in str(info.value)  # the message itself stays short


def test_run_log_can_be_switched_off(app_config: AppConfig) -> None:
    data = app_config.model_dump()
    data["trace"] = {"enabled": False}
    assert AppConfig.model_validate(data).trace.enabled is False
    assert app_config.trace.enabled is True  # default on
