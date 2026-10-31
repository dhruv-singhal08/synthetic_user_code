"""Artifact plugins by name (DISCONNECTED in design v3: empty, not called by the evaluator)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import ConfigError
from synthetic_user.llm.base import LLMClient
from synthetic_user.nodes.evaluators.plugins.base import ArtifactPlugin

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]
PluginFactory = Callable[[LLMClient, AppConfig, Logger], ArtifactPlugin]

PLUGINS: dict[str, PluginFactory] = {}


def create_plugins(
    names: Sequence[str], llm: LLMClient, settings: AppConfig, logger: Logger
) -> list[ArtifactPlugin]:
    """Plugins in the given order. An unknown or repeated name is a ConfigError."""
    unknown = [n for n in names if n not in PLUGINS]
    if unknown:
        raise ConfigError(f"Unknown artifact plugins {unknown}; known: {sorted(PLUGINS)}")
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ConfigError(f"Artifact plugins {repeated} listed more than once")
    return [PLUGINS[name](llm, settings, logger) for name in names]
