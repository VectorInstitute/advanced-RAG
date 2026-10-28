"""How far literal grep can reach on MuSiQue, with no model call.

For each question the probe asks two things of a case-insensitive literal grep
over the corpus. First hop: does some capitalized span of the question match
the first supporting passage? Each later hop: does the previous hop's gold
answer (the bridge entity an agent would read off the passage it opened) match
the next supporting passage? A question's whole chain is reachable when every
hop is.

It is an upper bound on what ``C5`` can do, never an accuracy result: it hands
the agent the right bridge answers and the best first span. What it shows is
where the mechanism has room to win.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from statistics import median
from typing import Any

from agent_arms import spans
from aieng.advanced_rag.agents.corpus_tools import Corpus, RegexGrep
from aieng.advanced_rag.corpora.musique import MusiqueQuestion


def _hits(grep: RegexGrep, text: str) -> list[str] | None:
    text = text.strip()
    if len(text) < 3:
        return None
    found = grep.grep(re.escape(text), max_results=10_000)
    return [m.doc_id for m in found.matches]


def reachability(questions: Sequence[MusiqueQuestion], corpus: Corpus) -> dict[str, Any]:
    """Return per-hop reachability under literal grep, and the hit-list sizes.

    Returns
    -------
    dict
        ``by_hop``: for each hop count, the question count, the share whose
        first supporting passage some question span hits, the share of bridge
        greps that hit the next supporting passage, and the share whose whole
        chain is reachable. ``rows``: one record per question with identifiers
        and booleans only. ``hit_sizes``: median and 90th percentile hits.
    """
    grep = RegexGrep(corpus, scan_cap=10_000)
    rows = []
    first_sizes: list[int] = []
    bridge_sizes: list[int] = []
    for question in questions:
        gold = question.supporting_ids
        best = None
        for span in spans(question.question):
            hits = _hits(grep, span)
            if hits and gold[0] in hits and (best is None or len(hits) < best):
                best = len(hits)
        if best is not None:
            first_sizes.append(best)
        bridges = []
        for step, nxt in zip(question.decomposition, gold[1:]):
            hits = _hits(grep, step.answer)
            ok = hits is not None and nxt in hits
            bridges.append(ok)
            if ok and hits:
                bridge_sizes.append(len(hits))
        rows.append(
            {
                "question_id": question.question_id,
                "hops": question.hops,
                "first_hop": best is not None,
                "bridges": bridges,
                "chain": best is not None and all(bridges),
            }
        )
    by_hop = {}
    for hops in sorted({r["hops"] for r in rows}):
        group = [r for r in rows if r["hops"] == hops]
        bridge_flags = [b for r in group for b in r["bridges"]]
        by_hop[hops] = {
            "questions": len(group),
            "first_hop": round(sum(r["first_hop"] for r in group) / len(group), 3),
            "bridge": round(sum(bridge_flags) / len(bridge_flags), 3) if bridge_flags else 0.0,
            "chain": round(sum(r["chain"] for r in group) / len(group), 3),
        }

    def p90(values: list[int]) -> int:
        return sorted(values)[int(0.9 * len(values))] if values else 0

    return {
        "by_hop": by_hop,
        "rows": rows,
        "hit_sizes": {
            "first_hop_median": median(first_sizes) if first_sizes else 0,
            "first_hop_p90": p90(first_sizes),
            "bridge_median": median(bridge_sizes) if bridge_sizes else 0,
            "bridge_p90": p90(bridge_sizes),
        },
    }


def main() -> int:
    """Write the probe for the shared 300 and the ladder subset at both corpus sizes."""
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from agent_arms import ladder_questions, musique_corpus  # noqa: PLC0415
    from aieng.advanced_rag.corpora.musique import load_musique, stratified_test_set  # noqa: PLC0415

    shared = stratified_test_set(load_musique()[1])
    small = musique_corpus(11_656)
    out = {
        "shared-11656": reachability(shared, small),
        "ladder-11656": reachability(ladder_questions(shared), small),
        "ladder-139416": reachability(ladder_questions(shared), musique_corpus(139_416)),
    }
    for result in out.values():
        result["by_hop"] = {str(k): v for k, v in result["by_hop"].items()}
    # No chat model is called, so the model is null with zero calls.
    out["produced_by"] = {"model": None, "model_calls": 0, "producer": "grep_probe.py"}
    path = Path(__file__).resolve().parent / "outputs" / "musique_grep_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for name, result in out.items():
        if name != "produced_by":
            print(name, result["by_hop"], result["hit_sizes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["reachability"]
