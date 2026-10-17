"""Artifact plugins by name, as listed in config ``evaluation.plugins`` (e.g. ["sql"])."""

from __future__ import annotations

import logging
from collections.abc import Callable

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import ConfigError
from synthetic_user.llm.base import LLMClient
from synthetic_user.nodes.evaluators.plugins.base import ArtifactPlugin
from synthetic_user.nodes.evaluators.plugins.sql.plugin import SqlPlugin

Logger = logging.Logger | logging.LoggerAdapter[logging.Logger]
PluginFactory = Callable[[LLMClient, AppConfig, Logger], ArtifactPlugin]

PLUGINS: dict[str, PluginFactory] = {
    "sql": SqlPlugin,
}


def create_plugins(llm: LLMClient, settings: AppConfig, logger: Logger) -> list[ArtifactPlugin]:
    """Plugins in config order. An unknown or repeated name is a ConfigError."""
    names = settings.evaluation.plugins
    unknown = [n for n in names if n not in PLUGINS]
    if unknown:
        raise ConfigError(f"Unknown evaluation.plugins {unknown}; known: {sorted(PLUGINS)}")
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ConfigError(f"evaluation.plugins lists {repeated} more than once")
    return [PLUGINS[name](llm, settings, logger) for name in names]
