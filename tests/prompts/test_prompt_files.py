"""Prompt text lives in ``*.yaml`` files beside the code (prompts/templates.py)."""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any

import pytest

import synthetic_user
from synthetic_user.core.exceptions import PromptTemplateError
from synthetic_user.prompts.templates import PromptFile, load_prompt_file

PACKAGE = Path(synthetic_user.__file__).parent
PROMPT_FILES = sorted(
    path
    for path in PACKAGE.rglob("*.yaml")
    if path.name.endswith("prompt.yaml") or path.parent.name == "prompts"
)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return []


def test_every_node_has_its_prompt_file() -> None:
    names = {path.relative_to(PACKAGE).as_posix() for path in PROMPT_FILES}
    assert names == {
        "adapters/mock_llm_prompt.yaml",
        "metrics/compiler_prompt.yaml",
        "nodes/evaluator/general_prompt.yaml",
        "nodes/evaluator/judge_common_prompt.yaml",
        "nodes/evaluator/retrieval_prompt.yaml",
        "nodes/executor/prompt.yaml",
        "nodes/question/prompt.yaml",
        "nodes/scenario/prompt.yaml",
        "prompts/common.yaml",
        "prompts/persona.yaml",
    }


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda path: path.parent.name + "/" + path.name)
def test_prompt_file_loads_and_every_placeholder_is_valid(path: Path) -> None:
    prompts = load_prompt_file(path)
    assert prompts.names()
    for key in prompts.names():
        for text in _strings(prompts.raw(key)):
            assert Template(text).is_valid(), f"{path.name}:{key} has an invalid placeholder"


def test_prompt_python_holds_no_prompt_text() -> None:
    """Prompt builders read their text from YAML: no multi-line prompt string literals left."""
    builders = [
        *PACKAGE.glob("nodes/*/*prompt.py"),
        PACKAGE / "adapters" / "mock_llm_prompt.py",
        PACKAGE / "metrics" / "compiler_prompt.py",
    ]
    for path in builders:
        source = path.read_text(encoding="utf-8")
        assert 'f"""' not in source, path
        assert "## TASK" not in source, path


def _file(tmp_path: Path, text: str) -> PromptFile:
    path = tmp_path / "prompt.yaml"
    path.write_text(text, encoding="utf-8")
    return PromptFile(path)


def test_text_fills_placeholders_and_keeps_a_literal_dollar(tmp_path: Path) -> None:
    prompts = _file(tmp_path, 'greet: "Hello ${name}, that costs $$5."\n')
    assert prompts.text("greet", name="Ana") == "Hello Ana, that costs $5."


def test_data_fills_placeholders_in_nested_strings(tmp_path: Path) -> None:
    prompts = _file(
        tmp_path, 'example:\n  kind: "${kinds}"\n  items: ["<${n} items>"]\n  ok: false\n'
    )
    assert prompts.data("example", kinds="<a | b>", n=3) == {
        "kind": "<a | b>",
        "items": ["<3 items>"],
        "ok": False,
    }
    first = prompts.data("example", kinds="x", n=1)
    first["items"].append("changed")
    assert prompts.data("example", kinds="x", n=1)["items"] == ["<1 items>"]  # a fresh copy


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda p: p.text("greet"), r"needs values for \['name'\]"),
        (lambda p: p.text("greet", name="A", extra="B"), r"no placeholder for \['extra'\]"),
        (lambda p: p.text("missing"), "no prompt entry 'missing'"),
        (lambda p: p.text("example"), "not a text entry"),
        (lambda p: p.data("greet", name="A"), "is a text entry"),
        (lambda p: p.text("bad"), "invalid placeholder"),
    ],
)
def test_template_mistakes_fail_loudly(tmp_path: Path, call: Any, message: str) -> None:
    prompts = _file(tmp_path, 'greet: "Hi ${name}"\nexample: {a: 1}\nbad: "costs $5"\n')
    with pytest.raises(PromptTemplateError, match=message):
        call(prompts)


@pytest.mark.parametrize("text", ["- a list\n", "not: [valid", ""])
def test_malformed_prompt_file_is_refused(tmp_path: Path, text: str) -> None:
    with pytest.raises(PromptTemplateError):
        _file(tmp_path, text)
