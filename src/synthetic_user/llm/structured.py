"""The single path for turning LLM output into a validated model: bounded retry, typed error."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from synthetic_user.core.exceptions import LLMOutputValidationError, format_validation_error
from synthetic_user.core.trace import record_validation, trace_attempt
from synthetic_user.llm.base import LLMClient
from synthetic_user.prompts.structured import with_retry_feedback

ModelT = TypeVar("ModelT", bound=BaseModel)

# Returns a list of problems; empty means the object is acceptable.
PostValidator = Callable[[ModelT], Sequence[str]]

_log = logging.getLogger(__name__)


async def generate_structured(
    llm: LLMClient,
    system: str,
    user: str,
    model_cls: type[ModelT],
    max_retries: int,
    post_validate: PostValidator[ModelT] | None = None,
) -> ModelT:
    """Call ``llm.generate_json`` and validate into ``model_cls``.

    On JSON, schema, or ``post_validate`` errors, retry with the same system prompt and the
    user prompt plus "previous output invalid: <errors>". At most ``max_retries`` retries.
    Provider errors (LLMProviderError) are not retried here and propagate unchanged.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    attempts = max_retries + 1
    all_errors: list[str] = []
    prompt = user
    purpose = model_cls.__name__
    for attempt in range(1, attempts + 1):
        with trace_attempt(purpose, attempt):
            result = await _attempt(llm, system, prompt, model_cls, post_validate)
        record_validation(purpose, attempt, [] if isinstance(result, BaseModel) else result)
        if isinstance(result, BaseModel):
            return result
        errors = result
        all_errors.extend(f"attempt {attempt}: {e}" for e in errors)
        _log.warning(
            "%s output invalid (attempt %d/%d): %s",
            model_cls.__name__,
            attempt,
            attempts,
            "; ".join(errors),
        )
        prompt = with_retry_feedback(user, errors)

    raise LLMOutputValidationError(
        f"{model_cls.__name__}: LLM output invalid after {attempts} attempt(s)",
        attempts=attempts,
        errors=all_errors,
    )


async def _attempt(
    llm: LLMClient,
    system: str,
    user: str,
    model_cls: type[ModelT],
    post_validate: PostValidator[ModelT] | None,
) -> ModelT | list[str]:
    try:
        raw = await llm.generate_json(system, user)
    except LLMOutputValidationError as exc:
        return list(exc.errors) or [str(exc)]
    try:
        obj = model_cls.model_validate(raw)
    except ValidationError as exc:
        return [format_validation_error(exc)]
    problems = list(post_validate(obj)) if post_validate is not None else []
    return problems or obj
