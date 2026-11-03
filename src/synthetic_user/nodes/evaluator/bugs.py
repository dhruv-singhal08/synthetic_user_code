"""Bug Reporter (D244): one bug candidate per distinct root cause, from validated failures only.

Sources: only the findings that decided a FAIL verdict (supported evidence, settled, not
exempted, at or above ``evaluation.fail_severity``, in a metric scored below its best or the goal
check). A passing or inconclusive evaluation, an exempted difference, a lower-severity issue on
its own, a finding on a metric the judge scored perfect and an infrastructure failure (error /
timeout: recorded only, never judged) create no bug. Another finding of the same failed
evaluation joins a bug only as supporting evidence of the SAME defect (supported, settled, not
exempted, medium or above).

Grouping is deterministic, from what the judges already return (no extra LLM call). Live
judges label one defect differently per metric (wrong_source in one, rule_violation in
another), so the mechanism alone never splits a defect:

- within a question, findings are ONE defect when they quote overlapping evidence (one quote
  inside the other, of comparable length - a whole-artifact quote links nothing), name the same
  verified subject, or have the same description; the goal-completion finding is the outcome
  of the question, so it joins the main defect and is a bug of its own only when nothing else
  failed;
- across questions, two defects are ONE bug when a finding of each has the same mechanism AND
  the same canonical subject (verified in the response, the question or the cheat sheet; a
  trace tag is read as the item it tags; catalog names canonicalised);
- findings with neither evidence, subject nor shared description stay separate defects.

In each evaluation a bug is that question's occurrence (the same ``bug_id`` in every affected
evaluation, so the evaluation contract and its finding ids are unchanged); ``consolidate_bugs``
merges the occurrences into the run-level list written to bug_candidates.json.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from synthetic_user.core.identifiers import IdKind, SequentialIdGenerator
from synthetic_user.models.evaluation import (
    BugCandidate,
    BugOccurrence,
    EvaluationResult,
    EvidenceStatus,
    Finding,
    RootCause,
    Severity,
)
from synthetic_user.models.metrics import GOAL_COMPLETION_CATEGORY
from synthetic_user.models.question import QuestionCase
from synthetic_user.nodes.evaluator.judge_common import normalise, quoted
from synthetic_user.nodes.evaluator.verdict import SEVERITY_RANK

MAX_BUG_TITLE_CHARS = 200
# Two quotes are the same evidence when the shorter is inside the longer and the longer is at
# most this many times as long (a quote of the whole artifact contains every other quote).
MAX_QUOTE_RATIO = 3
# Description lines (shown in bug_candidates.json, the CSV and the UI).
EXPECTED = "Expected"
ACTUAL = "Actual"
HYPOTHESIS = "Root-cause hypothesis"
IMPACT = "Impact"
EVIDENCE = "Evidence"

# canonical subject of a root cause, or None when it cannot be verified (per evaluation)
SubjectResolver = Callable[[str], str | None]


@dataclass
class BugGroup:
    """The findings of ONE defect in one failed evaluation (before bug ids exist). ``keys``:
    mechanism|canonical-subject of its findings - what can match another question's defect."""

    findings: list[Finding] = field(default_factory=list)
    keys: set[str] = field(default_factory=set)


def bug_groups(
    findings: Sequence[Finding],
    failing_ids: Iterable[str],
    confirmed_ids: Iterable[str],
    resolve_subject: SubjectResolver,
) -> list[BugGroup]:
    """The defects of one failed evaluation; empty when nothing failed. ``failing_ids``: the
    findings that decided the FAIL; ``confirmed_ids``: the confirmed problems that may support
    them (the verdict's, so a finding on a metric scored perfect never enters a bug)."""
    failing_set = set(failing_ids)
    confirmed = set(confirmed_ids)
    failing = [f for f in findings if f.finding_id in failing_set]
    if not failing:
        return []
    goals = [f for f in failing if f.category == GOAL_COMPLETION_CATEGORY]
    sources = [f for f in failing if f.category != GOAL_COMPLETION_CATEGORY] or goals
    supporting = [
        f
        for f in findings
        if f.finding_id not in failing_set
        and f.finding_id in confirmed
        and f.category != GOAL_COMPLETION_CATEGORY
        and _supporting(f)
    ]
    members = [*sources, *supporting]
    subjects = {f.finding_id: _subject(f, resolve_subject) for f in members}
    parent = list(range(len(members)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(members):
        for j in range(i):
            if _same_defect(a, members[j], subjects):
                parent[root(i)] = root(j)
    groups: dict[int, BugGroup] = {}
    source_ids = {f.finding_id for f in sources}
    for i, finding in enumerate(members):
        if finding.finding_id in source_ids:
            groups.setdefault(root(i), BugGroup())
    for i, finding in enumerate(members):
        group = groups.get(root(i))
        if group is None:
            continue  # supporting finding of no failing defect
        group.findings.append(finding)
        if (subject := subjects[finding.finding_id]) and finding.root_cause is not None:
            group.keys.add(f"{finding.root_cause.mechanism.value}|{subject}")
    ordered = list(groups.values())
    if goals and goals is not sources:  # the outcome joins the main (most severe) defect
        lead = max(ordered, key=lambda g: max(SEVERITY_RANK[f.severity] for f in g.findings))
        lead.findings.extend(goals)
    return ordered


def _subject(finding: Finding, resolve_subject: SubjectResolver) -> str | None:
    cause = finding.root_cause
    return resolve_subject(cause.subject) if cause is not None and cause.subject else None


def _same_defect(a: Finding, b: Finding, subjects: dict[str, str | None]) -> bool:
    subject = subjects[a.finding_id]
    if subject is not None and subject == subjects[b.finding_id]:
        return True
    if normalise(a.description) == normalise(b.description):
        return True
    if a.evidence and b.evidence:
        short, long = sorted((a.evidence, b.evidence), key=len)
        return quoted(short, long) and len(long) <= MAX_QUOTE_RATIO * max(len(short), 1)
    return False


def _supporting(finding: Finding) -> bool:
    return (
        finding.evidence_status is EvidenceStatus.SUPPORTED
        and not finding.unresolved
        and finding.exempted_by is None
        and SEVERITY_RANK[finding.severity] >= SEVERITY_RANK[Severity.MEDIUM]
    )


def assign_bugs(
    items: Sequence[tuple[EvaluationResult, QuestionCase, list[BugGroup]]],
    product_id: str,
    ids: SequentialIdGenerator,
) -> list[EvaluationResult]:
    """One bug id per defect of the batch: defects of different questions sharing a
    mechanism|subject key are the same bug. Ids follow the order of first appearance."""
    flat = [(index, group) for index, (_, _, groups) in enumerate(items) for group in groups]
    parent = list(range(len(flat)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[str, int] = {}
    for i, (_, group) in enumerate(flat):
        for key in sorted(group.keys):
            if key in owner:
                parent[root(i)] = root(owner[key])
            else:
                owner[key] = i
    bug_ids: dict[int, str] = {}
    keys: dict[int, set[str]] = {}
    for i, (_, group) in enumerate(flat):
        keys.setdefault(root(i), set()).update(group.keys)
    results: list[EvaluationResult] = []
    position = 0
    for result, question, groups in items:
        bugs = []
        for group in groups:
            bug_root = root(position)
            position += 1
            if bug_root not in bug_ids:
                bug_ids[bug_root] = ids.next(IdKind.BUG)
            bug_key = min(keys[bug_root]) if keys[bug_root] else None
            bugs.append(
                _occurrence_bug(bug_ids[bug_root], bug_key, group, result, question, product_id)
            )
        results.append(result.model_copy(update={"bug_candidates": bugs}) if bugs else result)
    return results


def _occurrence_bug(
    bug_id: str,
    key: str | None,
    group: BugGroup,
    result: EvaluationResult,
    question: QuestionCase,
    product_id: str,
) -> BugCandidate:
    findings = _ranked(group.findings)
    occurrence = BugOccurrence(
        question_id=result.question_id,
        scenario_id=result.scenario_id,
        execution_id=result.execution_id,
        reproduction_question=question.utterance,
        finding_ids=[f.finding_id for f in findings],
        metric_ids=list(dict.fromkeys(f.category for f in findings)),
        evidence=list(dict.fromkeys(f.evidence for f in findings if f.evidence)),
    )
    return _bug(bug_id, key, findings, [occurrence], product_id)


def consolidate_bugs(evaluations: Sequence[EvaluationResult]) -> list[BugCandidate]:
    """The run-level bug list: one entry per bug id, with every occurrence. Bugs written
    before D244 (one per evaluation, no occurrences) are returned unchanged."""
    by_id: dict[str, list[BugCandidate]] = {}
    findings = {f.finding_id: f for ev in evaluations for f in ev.findings}
    for evaluation in evaluations:
        for bug in evaluation.bug_candidates:
            by_id.setdefault(bug.bug_id, []).append(bug)
    out: list[BugCandidate] = []
    for bug_id, copies in by_id.items():
        if len(copies) == 1 and not copies[0].occurrences:
            out.append(copies[0])
            continue
        occurrences = [o for bug in copies for o in bug.occurrences]
        found = [findings[i] for o in occurrences for i in o.finding_ids if i in findings]
        ranked = _ranked(found)
        if not ranked:  # findings not available: keep the first copy as it is
            out.append(copies[0])
            continue
        out.append(
            _bug(bug_id, copies[0].root_cause_key, ranked, occurrences, copies[0].product_id)
        )
    return out


def _ranked(findings: Iterable[Finding]) -> list[Finding]:
    """Most severe first; a metric finding names the bug before the goal-completion outcome."""
    return sorted(
        findings,
        key=lambda f: (f.category == GOAL_COMPLETION_CATEGORY, -SEVERITY_RANK[f.severity]),
    )


def _bug(
    bug_id: str,
    key: str | None,
    findings: Sequence[Finding],
    occurrences: Sequence[BugOccurrence],
    product_id: str,
) -> BugCandidate:
    """``findings`` ranked (``_ranked``); the lead finding names the bug, the most severe one
    sets its severity."""
    lead = findings[0]
    severity = max((f.severity for f in findings), key=lambda s: SEVERITY_RANK[s])
    cause: RootCause | None = next((f.root_cause for f in findings if f.root_cause), None)
    first = occurrences[0]
    metrics = list(dict.fromkeys(m for o in occurrences for m in o.metric_ids))
    evidence = list(
        dict.fromkeys(f'{o.question_id}: "{quote}"' for o in occurrences for quote in o.evidence)
    )
    questions = list(dict.fromkeys(o.question_id for o in occurrences))
    impact = (
        f"{severity.value} severity; {len(questions)} affected question(s) "
        f"({', '.join(questions)}); metric(s): {', '.join(metrics)}"
    )
    lines = [f"{EXPECTED}: {lead.expected_behavior}", f"{ACTUAL}: {lead.observed_behavior}"]
    if cause is not None:
        lines.append(f"{HYPOTHESIS}: {cause.hypothesis}")
    lines.append(f"{IMPACT}: {impact}")
    if evidence:
        lines.append(f"{EVIDENCE}: " + "; ".join(evidence))
    return BugCandidate(
        bug_id=bug_id,
        title=_title(lead, cause)[:MAX_BUG_TITLE_CHARS],
        description="\n".join(lines),
        reproduction_question=first.reproduction_question,
        product_id=product_id,
        question_id=first.question_id,
        scenario_id=first.scenario_id,
        execution_id=first.execution_id,
        finding_ids=[i for o in occurrences for i in o.finding_ids],
        severity=severity,
        root_cause=cause,
        root_cause_key=key,
        metric_ids=metrics,
        expected_behavior=lead.expected_behavior,
        actual_behavior=lead.observed_behavior,
        impact=impact,
        evidence=evidence,
        occurrences=list(occurrences),
    )


def _title(lead: Finding, cause: RootCause | None) -> str:
    if cause is None:
        return f"{lead.category}: {lead.description}"
    label = cause.mechanism.value.replace("_", " ")
    subject = f" ({cause.subject})" if cause.subject else ""
    return f"{label}{subject}: {lead.description}"
