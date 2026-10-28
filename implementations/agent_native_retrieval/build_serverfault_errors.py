"""Build the ServerFault error-string set: questions keyed on an exact string in a post.

Candidates are extracted, not generated. A candidate key is an error code (such
as ``AH00558`` or ``0x80070005``), an upper-case identifier (such as
``NT_STATUS_ACCESS_DENIED``), or an error line from a question's title or body,
cut at the clause where the error starts; in a title, only a quoted error
counts. A key is kept only when it occurs, as
an exact case-sensitive substring, in the title or question of every document of
the 10,000-question slice that contains it, and in at most one document for an
error line or three for a code or identifier. Those documents are the gold
labels, and :func:`question_sets.load_serverfault_questions` re-checks them on
every load.

The checker then reads each candidate's post and keeps the key only when it is
verbatim program output the question is about and the accepted answer resolves.
The question wording comes from :data:`question_sets.TEMPLATES`, so the
committed rows carry no post text: the key is stored as offsets and a digest.

    python build_serverfault_errors.py --target 30
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import build_common
import question_sets as qs
from aieng.advanced_rag.corpora import serverfault
from aieng.advanced_rag.evaluation.synthetic import Model, parse_json


OUT = qs.SERVERFAULT_ERRORS

_CODE = re.compile(r"(?<![\w-])(?:[A-Z]{2,6}-\d{3,6}|AH\d{5}|0x[0-9A-Fa-f]{8}|[A-Z]{2,}\d{3,5})(?![\w-])")
_IDENTIFIER = re.compile(r"(?<!\w)(?:[A-Z][A-Z0-9]*(?:_[A-Z0-9]+){2,}|E[A-Z]{4,})(?!\w)")
_MARKER = re.compile(
    r"\b(error|failed|failure|denied|refused|fatal|cannot|can't|could not|"
    r"unable to|no such|not permitted|not found|invalid|timed out)\b",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"[\"“]([^\"”]{20,100})[\"”]")
_PROSE = re.compile(r"\b(I|I'm|I've|my|we|our|you|your|but|when I|tried)\b", re.IGNORECASE)

#: Codes and identifiers that name a standard or a product rather than an error.
_GENERIC = frozenset(["SHA256", "SHA512", "SHA-256", "SHA-512", "RFC1918", "ECDSA", "ESTABLISHED", "ERROR", "EXPORT"])

#: Most documents an error line, and a code or identifier, may occur in.
MAX_HOLDERS = {"message": 1, "code": 3, "identifier": 3}

CHECK_SYSTEM = "You review evaluation questions for a search benchmark. Reply with one JSON object only."

CHECK_PROMPT = """A benchmark question asks how to fix the problem behind this exact string:

KEY: {key}

It was taken from the post below. Answer three things:
1. "verbatim": is KEY program output (an error message, error code, status
   name, or log line) copied into the post, rather than the asker's own prose?
2. "about": is the post's question about the problem KEY reports?
3. "resolved": does the accepted answer explain or fix that problem?

Title: {title}

Question:
{question}

Accepted answer:
{answer}

Return JSON: {{"verbatim": true/false, "about": true/false, "resolved": true/false, "reason": "one sentence"}}"""


@dataclass(frozen=True)
class Candidate:
    """One key and where it was found."""

    key: str
    kind: str
    doc_id: str
    field: str
    start: int


def error_line_key(line: str, *, quoted_only: bool = False) -> tuple[str, int] | None:
    """Return the error clause of a line and its offset in the line, or ``None``.

    A quoted string that holds an error marker is the key when the line has
    one. Otherwise the key runs from the start of the clause the marker is in
    to the end of the line, unless ``quoted_only`` is set: a title is the
    asker's summary, and only a string quoted in it is program output.
    """
    for match in _QUOTED.finditer(line):
        if _MARKER.search(match.group(1)) and _usable(match.group(1)):
            return match.group(1), match.start(1)
    if quoted_only:
        return None
    stripped = line.strip()
    marker = _MARKER.search(stripped)
    if marker is None:
        return None
    before = stripped[: marker.start()]
    cut = max(before.rfind("] "), before.rfind(": "), before.rfind("> "))
    key = (stripped[cut + 2 :] if cut >= 0 else stripped).strip().rstrip(".:")
    if not _usable(key):
        return None
    return key, line.index(key)


def _usable(key: str) -> bool:
    if not (20 <= len(key) <= 100) or len(key.split()) < 3:
        return False
    return not (key.endswith("?") or _PROSE.search(key))


def extract(documents: Sequence[serverfault.Document]) -> list[Candidate]:
    """Return every candidate key, in slice order."""
    found: list[Candidate] = []
    for document in documents:
        for field in ("title", "question"):
            text = document.title if field == "title" else document.question_text
            for pattern, kind in ((_CODE, "code"), (_IDENTIFIER, "identifier")):
                for match in pattern.finditer(text):
                    if match.group(0) not in _GENERIC:
                        found.append(Candidate(match.group(0), kind, document.doc_id, field, match.start()))
            offset = 0
            for line in text.splitlines(keepends=True):
                hit = error_line_key(line, quoted_only=field == "title")
                if hit is not None:
                    found.append(Candidate(hit[0], "message", document.doc_id, field, offset + hit[1]))
                offset += len(line)
    return found


def holders(key: str, documents: Sequence[serverfault.Document]) -> tuple[list[str], bool]:
    """Return the documents holding ``key``, and whether each asks about it.

    A document asks about a key when its title or question contains it.
    """
    ids, asked = [], True
    for document in documents:
        if key in document.text:
            ids.append(document.doc_id)
            asked = asked and (key in document.title or key in document.question_text)
    return ids, asked


def check(checker: Model, key: str, document: serverfault.Document) -> dict[str, Any]:
    """Ask whether the key is program output that the post asks about and resolves."""
    payload = parse_json(
        checker(
            CHECK_SYSTEM,
            CHECK_PROMPT.format(
                key=key,
                title=document.title,
                question=document.question_text[:3000],
                answer=document.answer_text[:3000],
            ),
        )
    )
    verdict: dict[str, Any] = {k: bool(payload.get(k)) for k in ("verbatim", "about", "resolved")}
    verdict["reason"] = str(payload.get("reason") or "")[:300]
    verdict["passed"] = all(verdict[k] for k in ("verbatim", "about", "resolved"))
    return verdict


def ordered_candidates(documents: Sequence[serverfault.Document], seed: int) -> list[Candidate]:
    """Return one candidate per distinct key, shuffled by ``seed``, title keys first.

    A title names what the whole post is about, so its keys go first.
    """
    unique: dict[str, Candidate] = {}
    for candidate in extract(documents):
        unique.setdefault(candidate.key, candidate)
    shuffled = list(unique.values())
    random.Random(seed).shuffle(shuffled)
    shuffled.sort(key=lambda c: c.field != "title")
    return shuffled


def eligible(
    candidates: Sequence[Candidate], documents: Sequence[serverfault.Document]
) -> list[tuple[Candidate, list[str]]]:
    """Return the candidates whose holders all ask about them, with those holders."""
    queue = []
    for candidate in candidates:
        ids, asked = holders(candidate.key, documents)
        if asked and 1 <= len(ids) <= MAX_HOLDERS[candidate.kind]:
            queue.append((candidate, ids))
    return queue


def to_row(n: int, record: dict[str, Any], by_id: dict[str, serverfault.Document], checker: str) -> dict[str, Any]:
    """Return the committed row for an accepted record: no post text, only pointers."""
    candidate: Candidate = record["candidate"]
    gold = [candidate.doc_id] + [i for i in record["gold_doc_ids"] if i != candidate.doc_id]
    return {
        "query_id": f"sf-err-{n:03d}",
        "kind": candidate.kind,
        "template": qs.TEMPLATES[candidate.kind],
        "key_sha256": qs.sha256_text(candidate.key),
        "key_length": len(candidate.key),
        "key_location": {
            "doc_id": candidate.doc_id,
            "field": candidate.field,
            "start": candidate.start,
            "end": candidate.start + len(candidate.key),
        },
        "gold_doc_ids": gold,
        "gold_links": [by_id[i].link for i in gold],
        "gold_licences": [by_id[i].licence for i in gold],
        "hits_in_slice": len(gold),
        "is_answerable": True,
        "checker": checker,
        # The checker's reasons quote the key, so only its verdicts are kept.
        "checks": [{k: v[k] for k in ("verbatim", "about", "resolved")} for v in record["checks"]],
    }


def main() -> None:
    """Build the set and its summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--messages", type=int, default=18, help="error lines among the target")
    parser.add_argument("--checker", default=build_common.DEFAULT_CHECKER)
    parser.add_argument("--checker-per-minute", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--trace", type=Path, default=None, help="also write every tried record")
    args = parser.parse_args()

    clock = build_common.Clock()
    documents = serverfault.documents(serverfault.fetch_slice())
    by_id = {d.doc_id: d for d in documents}
    checker = build_common.model(args.checker, args.checker_per_minute)

    shuffled = ordered_candidates(documents, args.seed)

    quotas = {"message": args.messages, "code_or_identifier": args.target - args.messages}
    records: list[dict[str, Any]] = []
    kept: Counter[str] = Counter()
    used_docs: set[str] = set()

    def group(kind: str) -> str:
        return "message" if kind == "message" else "code_or_identifier"

    queue = eligible(shuffled, documents)
    eligible_counts = Counter(candidate.kind for candidate, _ in queue)

    def run(item: tuple[Candidate, list[str]]) -> dict[str, Any]:
        candidate, ids = item
        verdicts = [check(checker, candidate.key, by_id[i]) for i in ids]
        return {
            "candidate": candidate,
            "gold_doc_ids": sorted(ids),
            "checks": verdicts,
            "accepted": all(v["passed"] for v in verdicts),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while queue and any(kept[g] < quotas[g] for g in quotas):
            batch: list[tuple[Candidate, list[str]]] = []
            while queue and len(batch) < 12:
                candidate, ids = queue.pop(0)
                if kept[group(candidate.kind)] >= quotas[group(candidate.kind)]:
                    continue
                if used_docs & set(ids):
                    continue
                used_docs.update(ids)
                batch.append((candidate, ids))
            for record in pool.map(run, batch):
                g = group(record["candidate"].kind)
                if kept[g] >= quotas[g]:
                    continue
                records.append(record)
                kept[g] += record["accepted"]
            print(f"checked {len(records)}, kept {dict(kept)}", flush=True)

    rows = [
        to_row(n, record, by_id, checker.name)
        for n, record in enumerate((r for r in records if r["accepted"]), start=1)
    ]
    qs.write_jsonl(args.out, rows)
    if args.trace is not None:
        qs.write_jsonl(
            args.trace,
            [
                {
                    "key": r["candidate"].key,
                    "kind": r["candidate"].kind,
                    "doc_id": r["candidate"].doc_id,
                    "accepted": r["accepted"],
                    "checks": r["checks"],
                }
                for r in records
            ],
        )

    failed = Counter(
        k
        for r in records
        if not r["accepted"]
        for v in r["checks"]
        for k in ("verbatim", "about", "resolved")
        if not v[k]
    )
    build_common.write_summary(
        args.out.with_suffix(".summary.json"),
        {
            "set": args.out.name,
            "generator": None,
            "question_source": "extracted keys in fixed templates",
            "checker": checker.name,
            "slice": "serverfault 10,000-question slice, 9,999 documents",
            "candidates_extracted": len(shuffled),
            "candidates_eligible_by_kind": dict(eligible_counts),
            "checked": len(records),
            "accepted": len(rows),
            "accepted_by_kind": dict(Counter(r["kind"] for r in rows)),
            "acceptance_rate": round(len(rows) / len(records), 4) if records else 0.0,
            "rejected_check_failures": dict(failed),
            "seconds": clock.seconds(),
        },
        checker,
    )
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
