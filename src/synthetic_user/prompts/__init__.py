"""Prompt pieces shared by the nodes, and the loader for every ``*.yaml`` prompt file.

- ``templates.py``: reads a prompt file and fills its ``${placeholders}``.
- ``persona.py`` / ``persona.yaml``: how the persona is shown to scenario and question prompts.
- ``contracts.py``: the compact JSON output example (D204).
- ``structured.py``: the retry feedback; ``smoke.py``: the ``smoke-llm`` check.
- ``common.yaml``: the text of the three modules above.

Each node's own prompts live in its package: ``nodes/<node>/*prompt.yaml``.
"""
