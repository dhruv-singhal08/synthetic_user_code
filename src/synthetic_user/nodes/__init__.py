"""Pipeline nodes, one package per node; ``pipeline.py`` runs them in this order.

- ``scenario``: persona + product [+ reference] -> scenarios (+ NL2SQL reference packs).
- ``question``: one scenario -> questions with their answer keys.
- ``executor``: questions -> product responses, through a ``ProductAdapter``.
- ``evaluator``: responses -> scored evaluations, bug candidates and run KPIs.

Each node package holds its code (``generator.py`` / ``executor.py`` / ``evaluator.py``), the
code that assembles its prompts (``*prompt.py``) and the prompt text itself (``*prompt.yaml``).
"""
