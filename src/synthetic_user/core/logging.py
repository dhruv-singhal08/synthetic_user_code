"""Logging setup and a logger adapter that stamps every record with the run id."""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any

from synthetic_user.core.exceptions import ConfigError

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(run_id)s] %(message)s"
_NO_RUN_ID = "-"


class _DefaultRunIdFilter(logging.Filter):
    """Give records from non-run loggers a placeholder run_id so the format never fails."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = _NO_RUN_ID
        return True


class RunLoggerAdapter(logging.LoggerAdapter[logging.Logger]):
    """Attach ``run_id`` to every record logged through this adapter."""

    def __init__(self, logger: logging.Logger, run_id: str) -> None:
        super().__init__(logger, {"run_id": run_id})
        self.run_id = run_id

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        extra = dict(kwargs.get("extra") or {})
        extra.setdefault("run_id", self.run_id)
        kwargs["extra"] = extra
        return msg, kwargs


def parse_log_level(level: str) -> int:
    """Map a level name (e.g. ``INFO``) to its numeric value. Raises ConfigError if unknown."""
    mapping = logging.getLevelNamesMapping()
    key = level.strip().upper()
    if key not in mapping:
        raise ConfigError(f"Unknown log level: {level!r}")
    return mapping[key]


def setup_logging(level: str) -> None:
    """Configure the root logger. Call once, from the CLI entry point."""
    logging.basicConfig(level=parse_log_level(level), format=LOG_FORMAT, force=True)
    for handler in logging.getLogger().handlers:
        handler.addFilter(_DefaultRunIdFilter())


def get_run_logger(name: str, run_id: str) -> RunLoggerAdapter:
    return RunLoggerAdapter(logging.getLogger(name), run_id)
