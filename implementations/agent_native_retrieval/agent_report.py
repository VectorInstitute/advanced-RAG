"""Per-hop summaries of RI #5's arms, and the table joined to RI #4's rows.

Everything here is a pure function of per-question records, so it is tested
without a model. :func:`public_row` keeps identifiers and numbers and drops
every piece of question, answer, and passage text, which is what lets the
results file be committed.

All-supporting recall is the share of questions with every supporting passage
in hand, in two readings. "Read" means shown to the model in full: for RI #4's
rows, the top 5 passages its answer prompt receives; for RI #5's rows, the
passages opened with ``open``, returned whole by ``search``, or, for
``grep-iterative``, shown in a step or in the answer prompt. "Seen" adds what a
result named without showing in full: for RI #4, the top 10 ranked; for RI #5,
every passage a ``grep`` preview or a truncated ``search`` hit named. An agent
can read a bridge entity off a ``grep`` preview without opening the passage,
which is why both readings are reported.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from aieng.advanced_rag.evaluation.multihop import paired_bootstrap


#: Fields of a record that hold no text and go into the committed file.
PUBLIC_FIELDS = (
    "question_id",
    "hops",
    "em",
    "f1",
    "model_calls",
    "tool_calls",
    "prompt_tokens",
    "completion_tokens",
    "seconds",
    "exhausted",
    "nudged",
    "read_ids",
    "seen_ids",
    "ranked",
)

RI4_ARMS = ("dense", "dense-iterative", "hipporag2")


def public_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a record's identifiers and numbers, with tool counts, and no text."""
    row = {name: record[name] for name in PUBLIC_FIELDS if name in record}
    trace = record.get("trace") or []
    if trace:
        counts: dict[str, int] = {}
        for event in trace:
            counts[event["name"]] = counts.get(event["name"], 0) + 1
        row["tools"] = counts
        row["zero_hit_greps"] = sum(1 for e in trace if e["name"] == "grep" and e.get("total_hits") == 0)
    return row


def _share(values: Sequence[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _avg(values: Sequence[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def summarise(rows: Sequence[Mapping[str, Any]], gold: Mapping[str, Sequence[str]]) -> dict[str, dict[str, Any]]:
    """Summarise one arm's rows by hop count and over all questions.

    Parameters
    ----------
    rows : sequence of mapping
        Records from :func:`public_row` or the full runs.
    gold : mapping
        Question identifier to its supporting passage identifiers.

    Returns
    -------
    dict
        Per group (``2hop``, ``3hop``, ``4hop``, ``all``): question count, EM,
        F1, all-supporting recall over passages read and over passages seen,
        mean supporting recall over passages read, the share of questions with
        a supporting passage seen in a preview and never read, calls, tool
        calls, tokens, and seconds per question, and the exhaustion rate. Rows
        with a ranking add all-supporting recall at 5 and 10.
    """
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(f"{row['hops']}hop", []).append(row)
    groups = dict(sorted(groups.items()))
    groups["all"] = list(rows)
    summary: dict[str, dict[str, Any]] = {}
    for name, members in groups.items():
        golds = [set(gold[r["question_id"]]) for r in members]
        read = [set(r.get("read_ids") or ()) for r in members]
        seen = [set(r.get("seen_ids") or ()) | set(r.get("read_ids") or ()) for r in members]
        entry: dict[str, Any] = {
            "questions": len(members),
            "exact_match": _avg([r["em"] for r in members]),
            "f1": _avg([r["f1"] for r in members]),
            "all_supporting_recall_read": _share([g <= rd for g, rd in zip(golds, read)]),
            "mean_supporting_recall_read": _avg([len(g & rd) / len(g) for g, rd in zip(golds, read)]),
            "all_supporting_recall_seen": _share([g <= s for g, s in zip(golds, seen)]),
            "seen_not_read_rate": _share([bool((g & s) - rd) for g, s, rd in zip(golds, seen, read)]),
            "calls_per_question": _avg([r.get("model_calls", 0) for r in members]),
            "tool_calls_per_question": _avg([r.get("tool_calls", 0) for r in members]),
            "tokens_per_question": round(
                mean([r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in members]), 1
            )
            if members
            else 0.0,
            "seconds_per_question": round(mean([r.get("seconds", 0.0) for r in members]), 2) if members else 0.0,
            "exhaustion_rate": _share([bool(r.get("exhausted")) for r in members]),
        }
        if members and all("ranked" in r for r in members):
            for depth in (5, 10):
                entry[f"all_supporting_recall_at_{depth}"] = _share(
                    [g <= set(r["ranked"][:depth]) for g, r in zip(golds, members)]
                )
        summary[name] = entry
    return summary


def compare_f1(
    first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]], hops: int | None = None
) -> dict[str, Any]:
    """Return the paired F1 difference of two arms with a bootstrap interval."""
    a = {r["question_id"]: r["f1"] for r in first if hops is None or r["hops"] == hops}
    b = {r["question_id"]: r["f1"] for r in second if hops is None or r["hops"] == hops}
    shared = sorted(set(a) & set(b))
    result = paired_bootstrap([a[q] for q in shared], [b[q] for q in shared])
    return {
        "questions": len(shared),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in vars(result).items()},
    }


def ri4_rows(path: Path) -> dict[str, Any]:
    """Return RI #4's committed summaries and costs for the joined arms."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    joined = {}
    for arm in RI4_ARMS:
        summary = data["arms"][arm]["summary"]
        cost = data["arms"][arm]["cost"]
        questions = data["test"]["questions"]
        joined[arm] = {
            "source": "RI #4",
            "model": data["test"]["model"],
            "summary": summary,
            "calls_per_question": cost["calls_per_question"],
            "tokens_per_question": cost["tokens_per_question"],
            "questions": questions,
        }
    return {"test": data["test"], "arms": joined}


def joined_table(ri5: Mapping[str, Any], ri4: Mapping[str, Any], group: str = "all") -> list[dict[str, Any]]:
    """Return one row per arm, RI #4's then RI #5's, for one hop group.

    ``recall_read`` and ``recall_seen`` are all-supporting recall in the two
    readings the module docstring defines.
    """
    rows = []
    for arm, entry in ri4["arms"].items():
        if group not in entry["summary"]:
            continue
        s = entry["summary"][group]
        rows.append(
            {
                "arm": arm,
                "source": "RI #4",
                "model": entry["model"],
                "questions": s["questions"],
                "exact_match": s["exact_match"],
                "f1": s["f1"],
                "recall_read": s["all_supporting_recall_at_5"],
                "recall_seen": s["all_supporting_recall_at_10"],
                "calls_per_question": entry["calls_per_question"] if group == "all" else None,
                "tokens_per_question": entry["tokens_per_question"] if group == "all" else None,
                "seconds_per_question": None,
                "exhaustion_rate": None,
            }
        )
    for arm, entry in ri5["arms"].items():
        if group not in entry["summary"]:
            continue
        s = entry["summary"][group]
        rows.append(
            {
                "arm": arm,
                "source": "RI #5",
                "model": entry["model"],
                "questions": s["questions"],
                "exact_match": s["exact_match"],
                "f1": s["f1"],
                "recall_read": s["all_supporting_recall_read"],
                "recall_seen": s["all_supporting_recall_seen"],
                "calls_per_question": s["calls_per_question"],
                "tokens_per_question": s["tokens_per_question"],
                "seconds_per_question": s["seconds_per_question"],
                "exhaustion_rate": s["exhaustion_rate"],
            }
        )
    return rows


def format_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return the joined table as fixed-width text, one model column per row."""

    def cell(value: Any, digits: int = 3) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    header = (
        f"{'arm':<16}{'source':<7}{'model':<17}{'n':>4}{'EM':>7}{'F1':>7}{'read':>7}{'seen':>7}"
        f"{'calls/q':>8}{'tokens/q':>10}{'sec/q':>7}{'exhaust':>8}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['arm']:<16}{r['source']:<7}{r['model']:<17}{r['questions']:>4}{cell(r['exact_match']):>7}"
            f"{cell(r['f1']):>7}{cell(r['recall_read']):>7}{cell(r['recall_seen']):>7}"
            f"{cell(r['calls_per_question'], 2):>8}{cell(r['tokens_per_question'], 0):>10}"
            f"{cell(r['seconds_per_question'], 1):>7}{cell(r['exhaustion_rate'], 2):>8}"
        )
    return "\n".join(lines)


def compare_models(results: Mapping[str, Mapping[str, Any]], group: str = "all") -> list[dict[str, Any]]:
    """Return one row per arm and per result file, for two models' runs side by side.

    Parameters
    ----------
    results : mapping
        Label to a committed result file's content (``musique_agent_arms*.json``),
        in the order the rows should read.
    group : str, optional
        ``all``, ``2hop``, ``3hop``, or ``4hop``.

    Returns
    -------
    list of dict
        Rows with ``arm``, ``label``, ``model``, ``produced_by`` (the file's
        ``produced_by.label``), ``f1``, ``exact_match``, ``judged`` (None when the
        file has no verdicts), and the cost columns for the ``all`` group.
    """
    arms: list[str] = []
    for content in results.values():
        arms.extend(a for a in content["arms"] if a not in arms)
    rows = []
    for arm in arms:
        for label, content in results.items():
            entry = content["arms"].get(arm)
            if entry is None or group not in entry["summary"]:
                continue
            s = entry["summary"][group]
            rows.append(
                {
                    "arm": arm,
                    "label": label,
                    "model": entry["model"],
                    "produced_by": content["produced_by"]["label"],
                    "f1": s["f1"],
                    "exact_match": s["exact_match"],
                    "judged": (entry.get("judged") or {}).get(group),
                    "calls_per_question": s["calls_per_question"],
                    "tokens_per_question": s["tokens_per_question"],
                    "exhaustion_rate": s["exhaustion_rate"],
                }
            )
    return rows


def format_comparison(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return :func:`compare_models` rows as a fixed-width table."""

    def cell(value: Any, digits: int = 3) -> str:
        return "-" if value is None else f"{value:.{digits}f}"

    header = (
        f"{'arm':<16}{'model':<24}{'produced_by':<24}{'EM':>7}{'F1':>7}{'judged':>8}"
        f"{'calls/q':>9}{'tokens/q':>10}{'exhaust':>9}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['arm']:<16}{r['model']:<24}{r['produced_by']:<24}{cell(r['exact_match']):>7}{cell(r['f1']):>7}"
            f"{cell(r['judged']):>8}{cell(r['calls_per_question'], 2):>9}{cell(r['tokens_per_question'], 0):>10}"
            f"{cell(r['exhaustion_rate'], 2):>9}"
        )
    return "\n".join(lines)


__all__ = [
    "PUBLIC_FIELDS",
    "RI4_ARMS",
    "compare_f1",
    "compare_models",
    "format_comparison",
    "format_table",
    "joined_table",
    "public_row",
    "ri4_rows",
    "summarise",
]
