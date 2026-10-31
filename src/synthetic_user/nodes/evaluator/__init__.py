"""Node 5: judge each product response (``Evaluator``), then the run KPIs (``kpis.py``).

- ``evaluator.py``: routing by outcome and profile, bug candidates.
- ``nl2sql.py`` / ``agent.py``: the two evaluation profiles (+ ``*_prompt.py`` / ``*_prompt.yaml``).
- ``trace_tags.py``: stable tags linking results to criteria and cheat-sheet items.
- ``no_sql.py``: shared detection of NL2SQL answers without SQL.
- ``kpis.py``: run KPIs and the KPI CSV; ``benchmark.py``: evaluator agreement on labelled cases.
- ``plugins/``: artifact plugin hook, kept but disconnected.
"""

from synthetic_user.nodes.evaluator.evaluator import EvaluationCase, Evaluator

__all__ = ["EvaluationCase", "Evaluator"]
