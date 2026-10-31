"""Prompt text lives in YAML files next to the code that uses it, never in Python strings.

A prompt file is a YAML mapping of named entries:

- a text entry is a string; ``${name}`` marks a value that code fills in (``string.Template``);
- a data entry (a mapping or list) is an output example; ``${name}`` works in its strings too.

Every placeholder must be filled and every value given must be used, so a typo in a prompt
file or in the calling code fails loudly instead of silently changing the prompt.
"""

from __future__ import annotations

import copy
from functools import cache
from pathlib import Path
from string import Template
from typing import Any

import yaml

from synthetic_user.core.exceptions import PromptTemplateError


class PromptFile:
    """The named prompt entries of one ``*prompt.yaml`` file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise PromptTemplateError(f"cannot read prompt file {path}: {exc}") from exc
        if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
            raise PromptTemplateError(f"prompt file {path} must be a mapping of named entries")
        self._entries: dict[str, Any] = loaded

    def names(self) -> list[str]:
        return list(self._entries)

    def raw(self, key: str) -> Any:
        """The entry ``key`` as written in the file (placeholders unfilled)."""
        return copy.deepcopy(self._entry(key))

    def text(self, key: str, **values: object) -> str:
        """The text entry ``key`` with its ``${...}`` placeholders filled from ``values``."""
        entry = self._entry(key)
        if not isinstance(entry, str):
            raise PromptTemplateError(f"{self.path}: {key!r} is not a text entry")
        return self._fill(key, entry, values)

    def data(self, key: str, **values: object) -> Any:
        """A copy of the data entry ``key``, placeholders in its strings filled from ``values``."""
        entry = self._entry(key)
        if isinstance(entry, str):
            raise PromptTemplateError(f"{self.path}: {key!r} is a text entry, not data")
        used: set[str] = set()
        filled = self._fill_data(key, copy.deepcopy(entry), values, used)
        self._check_unused(key, values, used)
        return filled

    def _entry(self, key: str) -> Any:
        if key not in self._entries:
            raise PromptTemplateError(f"{self.path}: no prompt entry {key!r}")
        return self._entries[key]

    def _fill(self, key: str, text: str, values: dict[str, object]) -> str:
        used: set[str] = set()
        filled = self._substitute(key, text, values, used)
        self._check_unused(key, values, used)
        return filled

    def _fill_data(self, key: str, value: Any, values: dict[str, object], used: set[str]) -> Any:
        if isinstance(value, str):
            return self._substitute(key, value, values, used)
        if isinstance(value, dict):
            return {name: self._fill_data(key, item, values, used) for name, item in value.items()}
        if isinstance(value, list):
            return [self._fill_data(key, item, values, used) for item in value]
        return value

    def _substitute(self, key: str, text: str, values: dict[str, object], used: set[str]) -> str:
        template = Template(text)
        if not template.is_valid():
            raise PromptTemplateError(
                f"{self.path}: {key!r} has an invalid placeholder (write a literal $ as $$)"
            )
        names = set(template.get_identifiers())
        missing = sorted(names - values.keys())
        if missing:
            raise PromptTemplateError(f"{self.path}: {key!r} needs values for {missing}")
        used.update(names)
        return template.substitute(values)

    def _check_unused(self, key: str, values: dict[str, object], used: set[str]) -> None:
        unused = sorted(values.keys() - used)
        if unused:
            raise PromptTemplateError(f"{self.path}: {key!r} has no placeholder for {unused}")


@cache
def load_prompt_file(path: Path) -> PromptFile:
    """Read a prompt file once per process (the files ship with the package)."""
    return PromptFile(path)


def prompts_beside(module_file: str, name: str = "prompt.yaml") -> PromptFile:
    """The prompt file ``name`` in the same folder as ``module_file`` (pass ``__file__``)."""
    return load_prompt_file(Path(module_file).resolve().parent / name)
