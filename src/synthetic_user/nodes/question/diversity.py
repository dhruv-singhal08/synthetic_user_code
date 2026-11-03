"""Lightweight semantic-diversity checks for the questions of ONE scenario (D239).

No LLM call and no embeddings. Two questions are near-duplicates only when BOTH hold:

- the same structure: task type, deliverable actions and limits, filters (name + value), style
  and ambiguity level - so varying a filter, a time expression, a grouping, the ambiguity or the
  number of results is always real diversity;
- nearly the same content words (utterance + deliverable subjects; Jaccard similarity of the
  content-word sets >= ``generation.question.max_question_similarity``) - so a paraphrase or a
  word substitution of the same request is caught, while the same structure asked in clearly
  different terminology is not.

A second check catches expectations copied from a sibling: different requested outputs (task
type, deliverable actions or limits) with word-for-word the same expected_behavior. Questions
that differ only in filters, style or ambiguity may legitimately share a generic expected
behaviour ("... for the requested period"), so they are not flagged. Problems feed the
existing bounded retry; whatever persists is rejected per question with the reason (the first
question of a pair is kept).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from synthetic_user.models.question import QuestionCase, QuestionDraft

_TOKEN = re.compile(r"[^\W_]+")
# Function words that carry no request content (English; other languages still compare well
# on their content words, only less strictly).
_STOPWORDS = frozenset(
    "a an and are as at be by can could did do does for from give get how i in is it list me "
    "my of on or our per please show tell than that the their them there these this those to "
    "us want was we were what when where which who whose why will with would you your".split()
)


@dataclass(frozen=True)
class QuestionShape:
    label: str  # "question 2" / "earlier question Q_004", for the feedback and rejections
    outputs: tuple[object, ...]  # the requested outputs: task type, deliverables, limits
    structure: tuple[object, ...]  # the request + how it is asked (style, ambiguity)
    words: frozenset[str]
    expected_behavior: str


def shape_of(question: QuestionDraft | QuestionCase, label: str) -> QuestionShape | None:
    """None for a legacy stored question without a plan (nothing to compare structurally)."""
    plan = question.plan
    if plan is None:
        return None
    traits = question.question_characteristics
    outputs = (
        question.intent.task_type.value,
        tuple(sorted((d.action.value, d.limit or 0) for d in plan.deliverables)),
    )
    filters = tuple(sorted((_norm(f.name), _norm(f.value)) for f in plan.filters))
    text = " ".join([question.utterance, *(d.subject for d in plan.deliverables)])
    return QuestionShape(
        label=label,
        outputs=outputs,
        structure=(
            *outputs,
            filters,
            plan.is_follow_up,
            traits.style.value,
            traits.ambiguity_level.value,
        ),
        words=_content_words(text),
        expected_behavior=_norm(question.expectations.expected_behavior),
    )


def similarity(a: QuestionShape, b: QuestionShape) -> float:
    if not a.words and not b.words:
        return 1.0
    return len(a.words & b.words) / len(a.words | b.words)


@dataclass(frozen=True)
class DiversityProblem:
    index: int  # position of the later question in the checked batch (0-based)
    message: str  # retry feedback / rejection reason


def diversity_problems(
    batch: Sequence[QuestionShape | None],
    earlier: Sequence[QuestionShape],
    *,
    threshold: float,
) -> list[DiversityProblem]:
    """Near-duplicates (vs earlier batch items and the scenario's earlier questions) and copied
    expectations, reported on the LATER question of each pair (at most one problem each)."""
    problems: list[DiversityProblem] = []
    for index, shape in enumerate(batch):
        if shape is None:
            continue
        others = [*earlier, *(s for s in batch[:index] if s is not None)]
        message = diversity_problem(shape, others, threshold=threshold)
        if message is not None:
            problems.append(DiversityProblem(index, message))
    return problems


def diversity_problem(
    shape: QuestionShape, others: Sequence[QuestionShape], *, threshold: float
) -> str | None:
    """The first near-duplicate or copied-expectation problem of ``shape`` vs ``others``."""
    return _near_duplicate(shape, others, threshold) or _copied_expectation(shape, others)


def _near_duplicate(
    shape: QuestionShape, others: Sequence[QuestionShape], threshold: float
) -> str | None:
    for other in others:
        if other.structure != shape.structure:
            continue
        score = similarity(shape, other)
        if score >= threshold:
            return (
                f"near-duplicate of {other.label} (same deliverables, filters, limits, style "
                f"and ambiguity; word similarity {score:.2f}): ask a different angle of the "
                "same goal by varying ambiguity, terminology, filters, time expression, "
                "grouping or phrasing, without changing the goal, capability or hard policy"
            )
    return None


def _copied_expectation(shape: QuestionShape, others: Sequence[QuestionShape]) -> str | None:
    for other in others:
        if other.outputs != shape.outputs and other.expected_behavior == (shape.expected_behavior):
            return (
                f"expected_behavior is copied word for word from {other.label} although the "
                "requested outputs differ: derive it from this question's own utterance"
            )
    return None


def _content_words(text: str) -> frozenset[str]:
    return frozenset(
        w for w in _TOKEN.findall(text.casefold()) if len(w) > 2 and w not in _STOPWORDS
    )


def _norm(text: str) -> str:
    return " ".join(_TOKEN.findall(text.casefold()))
