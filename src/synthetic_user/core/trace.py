"""Per-run log of every LLM and product call (D202): what each node sent and got back.

One readable text file per run, ``runs/<run_id>/run.log``, appended as each call finishes
(``tail -f`` works during a live run). Each pipeline step adds a "step started / finished"
banner; re-running a step appends again, so the file keeps the run's history. Blocks:

- LLM CALL: step, subject (scenario / question id), what was asked for + attempt, model,
  latency, then the SYSTEM PROMPT, USER PROMPT and OUTPUT (or the error + raw text).
- VALIDATION: whether that output was accepted, else the errors that led to a retry/reject.
- PRODUCT CALL: the question sent to the product, the outcome and the recorded answer.
- PRODUCT HTTP: the request body sent and the reply (http_json adapter).
- The pipeline's own log lines for the run (counts, warnings), in time order.

Never contains secrets: HTTP header values are not written (names only).

The step and subject of a block come from context variables set by the pipeline and the
nodes, so concurrent calls are labelled correctly.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import IO, Any

RUN_LOG = "run.log"
# Longest text kept per field (a whole 200k-char reference is still kept).
MAX_FIELD_CHARS = 250_000
_RULE = "-" * 100


class RunLog:
    """Appends readable blocks for one step of one run to ``path``."""

    def __init__(self, path: Path, run_id: str, step: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.run_id = run_id
        self.step = step
        self._file: IO[str] = path.open("a", encoding="utf-8")
        self._seq = 0
        self._started = time.perf_counter()
        self.write_text(
            f"\n{'=' * 100}\n{_now()}  run {run_id}  step {step} STARTED\n{'=' * 100}\n"
        )

    def write_text(self, text: str) -> None:
        self._file.write(text)
        self._file.flush()

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def close(self) -> None:
        took = time.perf_counter() - self._started
        self.write_text(
            f"{_now()}  step {self.step} FINISHED ({took:.1f} s, {self._seq} call(s))\n"
        )
        self._file.close()


class _RunLogHandler(logging.Handler):
    """Copies this run's log records (pipeline / node messages) into the run log."""

    def __init__(self, log: RunLog) -> None:
        super().__init__(logging.INFO)
        self._log = log

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "run_id", None) != self._log.run_id:
            return
        self._log.write_text(f"{_now()}  {record.levelname} {record.getMessage()}\n")


_log: ContextVar[RunLog | None] = ContextVar("run_log", default=None)
_subject: ContextVar[str | None] = ContextVar("run_log_subject", default=None)
_purpose: ContextVar[tuple[str, int] | None] = ContextVar("run_log_purpose", default=None)
_last_call: ContextVar[int | None] = ContextVar("run_log_last_call", default=None)


@contextmanager
def trace_step(path: Path, run_id: str, step: str) -> Iterator[RunLog]:
    """Append every traced call (and the run's log lines) made inside this block to ``path``."""
    log = RunLog(path, run_id, step)
    handler = _RunLogHandler(log)
    package_logger = logging.getLogger("synthetic_user")
    package_logger.addHandler(handler)
    token = _log.set(log)
    try:
        yield log
    finally:
        _log.reset(token)
        package_logger.removeHandler(handler)
        log.close()


@contextmanager
def trace_subject(subject: str) -> Iterator[None]:
    """Label the blocks written inside this block (a scenario or question id)."""
    token = _subject.set(subject)
    try:
        yield
    finally:
        _subject.reset(token)


@contextmanager
def trace_attempt(purpose: str, attempt: int) -> Iterator[None]:
    """Label the LLM call made inside this block: the output model asked for + attempt no."""
    token = _purpose.set((purpose, attempt))
    try:
        yield
    finally:
        _purpose.reset(token)


def tracing() -> bool:
    return _log.get() is not None


def record_llm_call(
    *,
    method: str,
    model: str,
    system: str,
    user: str,
    output: Any,
    error: str | None,
    latency_ms: float,
) -> None:
    log = _log.get()
    if log is None:
        return
    seq = log.next_seq()
    purpose = _purpose.get()
    what = f"{purpose[0]} attempt {purpose[1]}" if purpose else method
    head = _head(log, seq, "LLM CALL", f"{what} · {model} · {latency_ms:.0f} ms")
    parts = [head, _section("SYSTEM PROMPT", system), _section("USER PROMPT", user)]
    if error:
        parts.append(_section("ERROR", error))
        parts.append(_section("RAW OUTPUT (could not be used)", output))
    else:
        parts.append(_section("OUTPUT", output))
    log.write_text("".join(parts))
    _last_call.set(seq)


def record_validation(purpose: str, attempt: int, errors: list[str]) -> None:
    log = _log.get()
    if log is None:
        return
    call = _last_call.get()
    verdict = "ACCEPTED" if not errors else "REJECTED -> " + "; ".join(errors)
    log.write_text(
        f"[{log.step} #{call}] VALIDATION {purpose} attempt {attempt}: {_clip(verdict)}\n"
    )


def record_product_call(
    *, adapter: str, question: str, outcome: str, answer: Any, latency_ms: float | None
) -> None:
    log = _log.get()
    if log is None:
        return
    took = f"{latency_ms:.0f} ms" if latency_ms is not None else "no latency"
    head = _head(log, log.next_seq(), "PRODUCT CALL", f"{adapter} adapter · {outcome} · {took}")
    log.write_text(head + _section("QUESTION SENT", question) + _section("ANSWER", answer))


def record_product_http(*, request: dict[str, Any], response: Any, error: str | None) -> None:
    log = _log.get()
    if log is None:
        return
    headers = ", ".join(request.get("header_names", [])) or "none"
    detail = f"{request['method']} {request['url']} · headers: {headers} (values hidden)"
    parts = [_head(log, log.next_seq(), "PRODUCT HTTP", detail)]
    parts.append(_section("REQUEST BODY", request.get("body")))
    if error:
        parts.append(_section("ERROR", error))
    else:
        parts.append(_section(f"RESPONSE (HTTP {response['status']})", response["body"]))
    log.write_text("".join(parts))


def _head(log: RunLog, seq: int, kind: str, detail: str) -> str:
    subject = _subject.get() or "-"
    return f"\n{_RULE}\n[{log.step} #{seq}] {_now()}  {kind}  {subject}  {detail}\n{_RULE}\n"


def _section(title: str, value: Any) -> str:
    if value is None:
        text = "(none)"
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    return f">>> {title}\n{_clip(text).rstrip()}\n"


def _clip(text: str) -> str:
    if len(text) > MAX_FIELD_CHARS:
        return text[:MAX_FIELD_CHARS] + f"\n[... cut: {len(text):,} chars in total]"
    return text


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
