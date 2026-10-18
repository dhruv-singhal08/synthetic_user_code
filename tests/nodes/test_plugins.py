"""The artifact plugin hook is kept but disconnected (design v3)."""

from __future__ import annotations

import logging

import pytest

from synthetic_user.core.config import AppConfig
from synthetic_user.core.exceptions import ConfigError
from synthetic_user.llm.fake import FakeLLMClient
from synthetic_user.nodes.evaluators.plugins.registry import PLUGINS, create_plugins


def test_registry_is_empty_and_rejects_unknown_names(app_config: AppConfig) -> None:
    llm, log = FakeLLMClient(), logging.getLogger("t")
    assert PLUGINS == {}
    assert create_plugins([], llm, app_config, log) == []
    with pytest.raises(ConfigError, match=r"Unknown artifact plugins \['sql'\]"):
        create_plugins(["sql"], llm, app_config, log)
