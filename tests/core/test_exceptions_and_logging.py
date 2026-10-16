from __future__ import annotations

import logging

import pytest

from synthetic_user.core import exceptions
from synthetic_user.core.exceptions import ConfigError, LLMOutputValidationError, SyntheticUserError
from synthetic_user.core.logging import get_run_logger, parse_log_level


@pytest.mark.parametrize(
    "error_type",
    [
        exceptions.ConfigError,
        exceptions.InputAssetError,
        exceptions.LLMProviderError,
        exceptions.LLMOutputValidationError,
        exceptions.ReferenceIntegrityError,
        exceptions.ArtifactNotFoundError,
        exceptions.ArtifactIntegrityError,
        exceptions.ProductClientError,
        exceptions.SQLAnalysisError,
    ],
)
def test_all_errors_derive_from_base(error_type: type[Exception]) -> None:
    assert issubclass(error_type, SyntheticUserError)


def test_llm_output_validation_error_carries_attempts_and_errors() -> None:
    err = LLMOutputValidationError("bad output", attempts=3, errors=["e1", "e2"])
    assert err.attempts == 3
    assert err.errors == ("e1", "e2")
    assert str(err) == "bad output (attempts=3, errors=2)"


def test_run_logger_stamps_run_id(caplog: pytest.LogCaptureFixture) -> None:
    log = get_run_logger("synthetic_user.test", "RUN_20261013_090507_a1b2")
    with caplog.at_level(logging.INFO, logger="synthetic_user.test"):
        log.info("hello %s", "world", extra={"other": 1})
    record = caplog.records[-1]
    assert record.getMessage() == "hello world"
    assert getattr(record, "run_id", None) == "RUN_20261013_090507_a1b2"
    assert getattr(record, "other", None) == 1


def test_parse_log_level() -> None:
    assert parse_log_level("info") == logging.INFO
    assert parse_log_level(" DEBUG ") == logging.DEBUG
    with pytest.raises(ConfigError, match="Unknown log level"):
        parse_log_level("chatty")
