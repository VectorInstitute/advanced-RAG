#!/usr/bin/env python3
"""Describe three public corpora in profiles, generate tools, and compare arms.

The "describe your data" extension of RI #5. A team writes a short profile of
its corpus (file kinds, identifier patterns, tables, sections, code) and
:mod:`aieng.advanced_rag.agents.profile` turns it into tools. This module runs
that idea on three public corpora and compares four arms on one model:

* ``G``: the generic ``C7`` arm (``search``, ``grep`` inside what search
  returned, ``open``).
* ``D``: ``C7`` with the profile's description in the system prompt and no new
  tools, which separates describing the data from building tools for it.
* ``P``: ``C7`` plus every tool the profile generates, and the description.
* ``Q``: ``open`` plus the generated tools only, with no ``search`` or ``grep``.

Every arm runs as an ADK agent through
:func:`~aieng.advanced_rag.agents.session_loop.run_session`, with each model
response cached by ADK's response-cache plugin.

The corpora are the ServerFault 10,000-question slice, the 84 FinanceBench
filings as RI #3 ingested them, and the ``click`` library at a pinned commit.
Each is written to a folder in the cache as a team would hold it, and the
profile in ``profiles/`` describes that folder. Nothing written here is
committed: ServerFault posts are CC BY-SA, FinanceBench is CC BY-NC, and the
committed outputs hold scores, counts, and question identifiers only.

Examples
--------
Write the three folders, build the question sets, then run and judge every arm
on the on-prem model at 20 requests a minute::

    $ python describe_your_data.py prepare
    $ python describe_your_data.py questions
    $ python describe_your_data.py run --per-minute 20
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import html
import json
import random
import re
import sys
import tarfile
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aieng.advanced_rag.agents.corpus_tools import (
    STANDARD_ARMS,
    ArmConfig,
    Corpus,
    CorpusToolServer,
    DenseSearch,
)
from aieng.advanced_rag.agents.profile import (
    DataProfile,
    ProfileTools,
    index_python,
    load_profile,
    module_name,
)
from aieng.advanced_rag.agents.records import AgentRun, MeasuredLimiter
from aieng.advanced_rag.cache.paths import root as cache_root
from aieng.advanced_rag.models import DEFAULT_MODELS, ONPREM_MODELS, ModelChoice, ModelSet, request_params
from aieng.advanced_rag.runs import Step


HERE = Path(__file__).resolve().parent
PROFILES = HERE / "profiles"
OUTPUTS = HERE / "outputs" / "describe_your_data"
RI_ID = "ri5"
PRODUCER = "04_describe_your_data"
CORPUS_NAMES = ("serverfault", "financebench", "click")

#: The pinned repository: click 8.1.8, BSD-3-Clause (``LICENSE.txt``).
CLICK_COMMIT = "d4ecb8a89a52ad3faacf3f1c65065d08c1d11c14"
CLICK_URL = f"https://github.com/pallets/click/archive/{CLICK_COMMIT}.tar.gz"
CLICK_SHA256 = "187d5ca45dec237364dff89c829c8faececafead4d5b599f008b07cb70c86a10"
CLICK_BYTES = 343_452

#: Questions per corpus. Small on purpose: a single run looking for a large effect.
QUESTIONS_PER_CORPUS = 24

#: Questions a live run on a mid or frontier model is capped at, per corpus.
LIVE_CAP_ABOVE_CHEAP = 10

ARM_NAMES = ("G", "D", "P", "Q")

#: Names the agent loop in the runs' cache key. Runs from the earlier
#: chat-completions loop sent other tool declarations.
AGENT_LOOP = "adk"

ANSWER_INSTRUCTION = """You answer questions from {corpus} using the tools.
A question can need several steps: find the right document, read the part that answers it, and
check exact values in the text or tables. Read the documents your answer relies on. When you know
the answer, reply with one line:
Answer: <the answer, short: a number, a name, a title, or a phrase>
If the documents do not contain it, reply: Answer: unknown"""


def work_root() -> Path:
    """Return where the corpus folders, question sets, and indexes go."""
    return Path(cache_root()) / "ri5" / "describe-your-data"


def corpus_folder(name: str) -> Path:
    """Return one corpus's folder, as its profile describes it."""
    return work_root() / "corpora" / name


def profile_path(name: str) -> Path:
    """Return the committed profile for one corpus."""
    return PROFILES / f"{name}.yaml"


# ---------------------------------------------------------------------------
# Corpus folders


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def prepare_serverfault(folder: Path | None = None) -> Path:
    """Write the ServerFault slice as one Markdown file per post plus two tables.

    ``posts/q<id>.md`` holds the question and its accepted answer under ``##``
    headings. ``posts.csv`` has one row per post and ``post_tags.csv`` one row
    per post and tag.
    """
    from aieng.advanced_rag.corpora import serverfault  # noqa: PLC0415

    folder = folder or corpus_folder("serverfault")
    payload = serverfault.fetch_slice()
    by_id = {str(q["question_id"]): q for q in payload["questions"]}
    documents = serverfault.documents(payload)
    posts = folder / "posts"
    posts.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[Any, ...]] = []
    tag_rows: list[tuple[str, str]] = []
    for doc in documents:
        rel = f"posts/{doc.doc_id}.md"
        (folder / rel).write_text(doc.text, encoding="utf-8")
        raw = by_id[str(doc.question_id)]
        accepted: Mapping[str, Any] = next(
            (a for a in raw.get("answers", []) if str(a["answer_id"]) == str(raw.get("accepted_answer_id"))),
            {},
        )
        rows.append(
            (
                rel,
                doc.question_id,
                doc.title,
                "|".join(doc.tags),
                doc.score,
                int(accepted.get("score", 0) or 0),
                int(raw.get("view_count", 0) or 0),
                int(raw.get("answer_count", 0) or 0),
                _year(raw.get("creation_date")),
                doc.link,
                doc.licence,
            )
        )
        tag_rows.extend((rel, tag) for tag in doc.tags)
    _write_csv(
        folder / "posts.csv",
        (
            "doc_id",
            "question_id",
            "title",
            "tags",
            "score",
            "answer_score",
            "views",
            "answers",
            "year",
            "link",
            "licence",
        ),
        rows,
    )
    _write_csv(folder / "post_tags.csv", ("doc_id", "tag"), tag_rows)
    return folder


def _year(epoch: Any) -> int | None:
    import datetime  # noqa: PLC0415

    try:
        return datetime.datetime.fromtimestamp(int(epoch), tz=datetime.UTC).year
    except (TypeError, ValueError):
        return None


_NUMBER = re.compile(r"^\(?-?\$?\s*\(?([0-9][0-9,]*\.?[0-9]*)\)?\s*%?\)?$")


def parse_number(text: str) -> float | None:
    """Return a table cell's number, with parentheses read as negative.

    ``$ 1,577``, ``(370)``, and ``3.2 %`` give 1577.0, -370.0, and 3.2. Text
    that is not a single number gives ``None``.
    """
    clean = text.strip().replace("—", "").replace("$ ", "$")
    if not clean:
        return None
    match = _NUMBER.match(clean.replace(" ", ""))
    if match is None:
        return None
    try:
        value = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    negative = clean.startswith("(") or clean.startswith("-") or clean.startswith("$(")
    return -value if negative else value


_YEAR = re.compile(r"^(?:FY\s*)?(?:19|20)\d\d$")


def header_row_indexes(grid: Sequence[Sequence[Mapping[str, Any]]]) -> list[int]:
    """Return the leading rows that are column headers.

    Docling marks some header cells and misses others, so a leading row also
    counts when every filled cell after the first column is text or a year.
    The run of header rows stops at the first row holding a number, and at
    four rows.
    """
    found: list[int] = []
    for index, row in enumerate(grid[:4]):
        filled = [c for c in row[1:] if str(c.get("text", "")).strip()]
        marked = bool(filled) and all(c.get("column_header") for c in filled)
        texts = [str(c.get("text", "")).strip() for c in filled]
        textual = bool(texts) and all(parse_number(t) is None or _YEAR.match(t) for t in texts)
        if not (marked or textual):
            break
        found.append(index)
    return found


def table_rows(filing: str, index: int, table: Mapping[str, Any]) -> tuple[dict[str, Any], list[tuple[Any, ...]]]:
    """Return one Docling table's summary row and its cells in long form.

    Each cell row carries the table, its position, its text and number, the
    row's label (the first non-header cell of the row), and the column's label
    (the header cells above it, joined).
    """
    grid = table.get("data", {}).get("grid", [])
    prov = table.get("prov") or [{}]
    page = int(prov[0].get("page_no", 0) or 0)
    table_id = f"{filing}#t{index:03d}"
    header_rows = header_row_indexes(grid)
    column_labels: list[str] = []
    width = max((len(row) for row in grid), default=0)
    for col in range(width):
        parts = []
        for r in header_rows:
            if col < len(grid[r]):
                text = grid[r][col].get("text", "").strip()
                if text and text not in parts:
                    parts.append(text)
        column_labels.append(" ".join(parts))
    cells: list[tuple[Any, ...]] = []
    for r, row in enumerate(grid):
        if r in header_rows:
            continue
        label = next((c.get("text", "").strip() for c in row if c.get("text", "").strip()), "")
        for c, cell in enumerate(row):
            text = cell.get("text", "").strip()
            if not text or c == 0:
                continue
            cells.append(
                (
                    table_id,
                    filing,
                    page,
                    r,
                    c,
                    label[:300],
                    column_labels[c][:200] if c < len(column_labels) else "",
                    text[:300],
                    parse_number(text),
                )
            )
    first_label = next((row[0].get("text", "").strip() for row in grid if row and row[0].get("text", "").strip()), "")
    summary = {
        "table_id": table_id,
        "doc_id": f"filings/{filing}.md",
        "filing": filing,
        "page": page,
        "n_rows": len(grid),
        "n_cols": width,
        "columns": " | ".join(label for label in column_labels if label)[:300],
        "first_row_label": first_label[:200],
    }
    return summary, cells


def prepare_financebench(folder: Path | None = None, ingested: Path | None = None) -> Path:
    """Write each ingested filing as one Markdown file plus three tables.

    ``filings/<doc_name>.md`` has a ``# Page N`` heading per page and a ``##``
    heading for each line Docling labelled a section header. ``filings.csv``
    lists the filings, ``tables.csv`` every table Docling found, and
    ``cells.csv`` every table cell with its row and column labels. Needs RI
    #3's section folders (``ingest_financebench.py``).
    """
    folder = folder or corpus_folder("financebench")
    ingested = ingested or cache_root() / "ingestion" / "financebench"
    from aieng.advanced_rag.corpora.sources import cache_dir  # noqa: PLC0415

    info_path = cache_dir() / "financebench-document-information.jsonl"
    info = {
        r["doc_name"]: r
        for r in (json.loads(line) for line in info_path.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    filings_dir = folder / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)
    filing_rows, table_summaries, all_cells = [], [], []
    for source in sorted(p for p in ingested.iterdir() if (p / "document.json").is_file()):
        filing = source.name
        document = json.loads((source / "document.json").read_text(encoding="utf-8"))
        headers: dict[int, set[str]] = {}
        for item in document.get("texts", []):
            if item.get("label") == "section_header" and item.get("prov"):
                headers.setdefault(int(item["prov"][0]["page_no"]), set()).add(item.get("text", "").strip())
        parts: list[str] = []
        pages = sorted((source / "pages").glob("p*.md"))
        for page_path in pages:
            number = int(page_path.stem[1:]) + 1
            parts.append(f"# Page {number}")
            wanted = headers.get(number, set())
            for line in page_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and stripped in wanted and not stripped.startswith("|"):
                    parts.append(f"## {stripped}")
                else:
                    parts.append(line)
            parts.append("")
        (filings_dir / f"{filing}.md").write_text("\n".join(parts), encoding="utf-8")
        meta = info.get(filing, {})
        filing_rows.append(
            (
                f"filings/{filing}.md",
                filing,
                meta.get("company", ""),
                meta.get("gics_sector", ""),
                meta.get("doc_type", ""),
                meta.get("doc_period", ""),
                len(pages),
            )
        )
        for index, table in enumerate(document.get("tables", [])):
            summary, cells = table_rows(filing, index, table)
            table_summaries.append(summary)
            all_cells.extend(cells)
    _write_csv(
        folder / "filings.csv",
        ("doc_id", "filing", "company", "sector", "doc_type", "period", "pages"),
        filing_rows,
    )
    _write_csv(
        folder / "tables.csv",
        ("table_id", "doc_id", "filing", "page", "n_rows", "n_cols", "columns", "first_row_label"),
        ([s[k] for k in s] for s in table_summaries),
    )
    _write_csv(
        folder / "cells.csv",
        ("table_id", "filing", "page", "row", "col", "row_label", "column_label", "text", "value"),
        all_cells,
    )
    return folder


def fetch_click(downloads: Path | None = None) -> Path:
    """Download the pinned click archive once and verify its size and digest.

    Raises
    ------
    ValueError
        If the archive differs from its pin.
    """
    downloads = downloads or work_root() / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    target = downloads / f"click-{CLICK_COMMIT[:8]}.tar.gz"
    if not target.is_file():
        with urllib.request.urlopen(CLICK_URL, timeout=60) as response:  # noqa: S310 - a pinned https URL
            target.write_bytes(response.read())
    payload = target.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != CLICK_BYTES or digest != CLICK_SHA256:
        raise ValueError(
            f"{target.name} is {len(payload):,} bytes with sha256 {digest}; the pin is "
            f"{CLICK_BYTES:,} bytes with {CLICK_SHA256}"
        )
    return target


def prepare_click(folder: Path | None = None) -> Path:
    """Extract the pinned click source, docs, and changelog into the folder.

    The archive is read as data: members are filtered to plain files under the
    archive's top folder, and nothing in it is run.
    """
    folder = folder or corpus_folder("click")
    archive = fetch_click()
    prefix = f"click-{CLICK_COMMIT}/"
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.startswith(prefix)]
        for member in members:
            rel = member.name[len(prefix) :]
            if ".." in Path(rel).parts:
                continue
            target = folder / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tar.extractfile(member)
            if extracted is not None:
                target.write_bytes(extracted.read())
    return folder


PREPARE: Mapping[str, Callable[[], Path]] = {
    "serverfault": prepare_serverfault,
    "financebench": prepare_financebench,
    "click": prepare_click,
}


def load(name: str) -> tuple[DataProfile, Corpus, ProfileTools]:
    """Return a corpus's profile, documents, and generated tools.

    Raises
    ------
    FileNotFoundError
        If the corpus folder has not been written; run ``prepare`` first.
    """
    folder = corpus_folder(name)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is missing; run `python describe_your_data.py prepare {name}`")
    profile = load_profile(profile_path(name), root=folder)
    corpus = profile.corpus()
    return profile, corpus, ProfileTools(profile, corpus)


def dense_search(name: str, corpus: Corpus, embedder: Any = None) -> DenseSearch:
    """Return local ``bge-small-en-v1.5`` search over a corpus, cached on disk."""
    if embedder is None:
        from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder  # noqa: PLC0415

        embedder = SentenceTransformerEmbedder()
    marker = hashlib.sha256("\n".join(d.doc_id for d in corpus.documents).encode()).hexdigest()[:12]
    return DenseSearch(corpus, embedder, cache=work_root() / "embeddings" / f"{name}-{marker}.npy")


# ---------------------------------------------------------------------------
# Question sets


@dataclass(frozen=True)
class Question:
    """One question with its reference answer.

    Parameters
    ----------
    qid : str
        Stable identifier, safe to commit.
    corpus : str
        Which corpus it is asked of.
    question : str
        The text the agent receives.
    reference : str
        The reference answer the judge compares against.
    qtype : str
        The question's kind, for the per-type table.
    source : str
        ``synthetic`` (the synthetic Q&A path), ``structured`` (computed from
        the corpus's own tables or code), or ``gold`` (the dataset's own).
    gold_docs : tuple of str
        Documents that hold the answer, when known.
    """

    qid: str
    corpus: str
    question: str
    reference: str
    qtype: str
    source: str
    gold_docs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        record = asdict(self)
        record["gold_docs"] = list(self.gold_docs)
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Question:
        """Read a record written by :meth:`to_dict`."""
        fields = dict(record)
        fields["gold_docs"] = tuple(fields.get("gold_docs") or ())
        return cls(**fields)


def _qid(corpus: str, *parts: Any) -> str:
    return f"{corpus}-" + hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()[:10]


def serverfault_structured(folder: Path, tools: ProfileTools, *, per_type: int = 3, seed: int = 0) -> list[Question]:
    """Return count, top-score, two-tag, and identifier questions with computed answers.

    Each answer is computed from ``posts.csv`` and ``post_tags.csv`` or from
    the identifier index, so it is exact. Every question can also be answered
    with ``grep``: tags are on each post's ``Tags:`` line, and grep reports how
    many documents match.
    """
    rng = random.Random(seed)
    with (folder / "posts.csv").open(encoding="utf-8", newline="") as handle:
        posts = {r["doc_id"]: r for r in csv.DictReader(handle)}
    by_tag: dict[str, list[str]] = {}
    for doc_id, row in posts.items():
        for tag in row["tags"].split("|"):
            if tag:
                by_tag.setdefault(tag, []).append(doc_id)
    questions: list[Question] = []
    mid = sorted(t for t, d in by_tag.items() if 8 <= len(d) <= 40)
    for tag in rng.sample(mid, per_type):
        questions.append(
            Question(
                _qid("serverfault", "count", tag),
                "serverfault",
                f"How many questions in this ServerFault collection are tagged `{tag}`?",
                str(len(by_tag[tag])),
                "count",
                "structured",
            )
        )
    pairs: Counter[tuple[str, str]] = Counter()
    for row in posts.values():
        tags = sorted(t for t in row["tags"].split("|") if t)
        for i, a in enumerate(tags):
            for b in tags[i + 1 :]:
                pairs[(a, b)] += 1
    pair_pool = sorted(p for p, n in pairs.items() if 4 <= n <= 15)
    for a, b in rng.sample(pair_pool, per_type):
        questions.append(
            Question(
                _qid("serverfault", "pair", a, b),
                "serverfault",
                f"How many questions in this ServerFault collection are tagged both `{a}` and `{b}`?",
                str(pairs[(a, b)]),
                "count_two_tags",
                "structured",
            )
        )
    top_pool = []
    for tag in sorted(t for t, d in by_tag.items() if 5 <= len(d) <= 30):
        scores = sorted((int(posts[d]["score"]), d) for d in by_tag[tag])
        if len(scores) > 1 and scores[-1][0] > scores[-2][0]:
            top_pool.append((tag, scores[-1][1]))
    for tag, doc_id in rng.sample(top_pool, per_type):
        questions.append(
            Question(
                _qid("serverfault", "top", tag),
                "serverfault",
                f"Of the questions in this ServerFault collection tagged `{tag}`, which one has the "
                "highest score? Give its title.",
                html.unescape(posts[doc_id]["title"]),
                "top_score",
                "structured",
                (doc_id,),
            )
        )
    unique = sorted(
        (value, next(iter(docs)))
        for value, docs in tools.identifier_index.get("unix_path", {}).items()
        if len(docs) == 1 and value.count("/") >= 3 and len(value) >= 18
    )
    for value, doc_id in rng.sample(unique, per_type):
        questions.append(
            Question(
                _qid("serverfault", "path", value),
                "serverfault",
                f"Which question in this ServerFault collection mentions the path `{value}`? Give its title.",
                html.unescape(posts[doc_id]["title"]),
                "identifier",
                "structured",
                (doc_id,),
            )
        )
    return questions


SERVERFAULT_SYNTHETIC_CONFIG: Mapping[str, Any] = {
    "domain": "ServerFault questions about system and network administration, each with its accepted answer",
    "users": [
        {
            "name": "sysadmin",
            "description": "An administrator who hit a problem and searches for the fix, using exact names, "
            "commands, and settings.",
            "weight": 1,
        }
    ],
    "questions": [
        {
            "name": "factoid",
            "description": "A short answer from the accepted answer: a command, a setting, a file, a value.",
            "weight": 2,
        },
        {"name": "procedure", "description": "What to run or change to fix the problem.", "weight": 1},
    ],
    "phrasings": [{"name": "natural", "description": "A complete, natural question.", "weight": 1}],
    "passages_per_document": 1,
    "statements_per_passage": 1,
    "null_share": 0.0,
    "held_out_share": 0.0,
    "held_out_minimum": 0,
    "paraphrase": False,
    "workers": 4,
    "seed": 0,
}

CLICK_SYNTHETIC_CONFIG: Mapping[str, Any] = {
    **SERVERFAULT_SYNTHETIC_CONFIG,
    "domain": "the documentation of click, a Python library for building command-line interfaces",
    "users": [
        {
            "name": "developer",
            "description": "A Python developer writing a command-line tool, who knows Python and not click.",
            "weight": 1,
        }
    ],
    "questions": [
        {
            "name": "factoid",
            "description": "A short answer: a parameter, a class, a default, a behaviour.",
            "weight": 2,
        },
        {"name": "how_to", "description": "How to do one thing with click.", "weight": 1},
    ],
}


def synthetic_questions(
    name: str,
    corpus: Corpus,
    doc_ids: Sequence[str],
    config: Mapping[str, Any],
    *,
    limit: int,
    generator: Any,
    checker: Any,
    judge: Any,
    embedder: Any,
) -> list[Question]:
    """Return single-hop questions from the synthetic Q&A path over some documents.

    The builder's uniqueness check sees only ``doc_ids``, so a question can
    have a second answer elsewhere in the corpus. The judge reads the
    reference answer, so a different correct answer from another document
    counts as wrong.
    """
    from aieng.advanced_rag.evaluation.synthetic import (  # noqa: PLC0415
        Document,
        build_single_hop_set,
        config_from_dict,
    )

    documents = [Document(d, corpus.get(d).text, d) for d in doc_ids if corpus.get(d) is not None]
    built = build_single_hop_set(
        documents,
        config_from_dict(dict(config)),
        generator=generator,
        checker=checker,
        judge=judge,
        embedder=embedder,
    )
    chosen = [q for q in built.answerable if q.variant == "plain"][:limit]
    return [
        Question(
            _qid(name, "synthetic", q.query_id),
            name,
            q.question,
            q.reference,
            str(q.labels.get("category") or q.kind),
            "synthetic",
            (q.doc_id,) if q.doc_id else (),
        )
        for q in chosen
    ]


def financebench_questions(per_type: int = 8, seed: int = 0) -> list[Question]:
    """Return a stratified draw of FinanceBench's own questions and answers."""
    from aieng.advanced_rag.corpora.sources import cache_dir  # noqa: PLC0415

    records = [
        json.loads(line)
        for line in (cache_dir() / "financebench-open-source.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng = random.Random(seed)
    chosen: list[Question] = []
    for qtype in ("metrics-generated", "domain-relevant", "novel-generated"):
        pool = sorted((r for r in records if r["question_type"] == qtype), key=lambda r: r["financebench_id"])
        for r in rng.sample(pool, per_type):
            chosen.append(
                Question(
                    r["financebench_id"],
                    "financebench",
                    r["question"],
                    r["answer"],
                    qtype,
                    "gold",
                    (f"filings/{r['doc_name']}.md",),
                )
            )
    return chosen


def _literal(node: Any) -> str | None:
    import ast  # noqa: PLC0415

    if isinstance(node, ast.Constant) and node.value is not None and not isinstance(node.value, bytes):
        return repr(node.value)
    return None


def _method_defaults(cls: Any, rel: str) -> list[tuple[str, str, str, str]]:
    import ast  # noqa: PLC0415

    found = []
    for item in cls.body:
        if not isinstance(item, ast.FunctionDef) or (item.name.startswith("_") and item.name != "__init__"):
            continue
        positional = item.args.args[-len(item.args.defaults) :] if item.args.defaults else []
        pairs = list(zip(positional, item.args.defaults, strict=False))
        pairs += [(a, d) for a, d in zip(item.args.kwonlyargs, item.args.kw_defaults, strict=False) if d is not None]
        for arg, default in pairs:
            value = _literal(default)
            if value is not None and not arg.arg.startswith("_"):
                found.append((f"{cls.name}.{item.name}", arg.arg, value, rel))
    return found


def click_code_questions(folder: Path, *, seed: int = 0) -> list[Question]:
    """Return default-value, base-class, and defining-file questions from the source.

    Answers come from :mod:`ast` over ``src/click``, so they are exact. A
    parameter whose default differs between overloads is left out.
    """
    import ast  # noqa: PLC0415

    rng = random.Random(seed)
    raw_defaults: list[tuple[str, str, str, str]] = []
    bases: list[tuple[str, str, str]] = []
    symbols = []
    for path in sorted((folder / "src" / "click").glob("*.py")):
        rel = path.relative_to(folder).as_posix()
        text = path.read_text(encoding="utf-8")
        symbols.extend(index_python(rel, text, module_name(rel, "src")))
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                names = [ast.unparse(b) for b in node.bases]
                if names and names != ["object"]:
                    bases.append((node.name, ", ".join(names), rel))
                raw_defaults.extend(_method_defaults(node, rel))
    values: dict[tuple[str, str], set[str]] = {}
    for owner, arg, value, _ in raw_defaults:
        values.setdefault((owner, arg), set()).add(value)
    defaults = sorted({d for d in raw_defaults if len(values[(d[0], d[1])]) == 1})
    questions: list[Question] = []
    for owner, arg, value, rel in rng.sample(defaults, 6):
        questions.append(
            Question(
                _qid("click", "default", owner, arg),
                "click",
                f"In the click source, what is the default value of the `{arg}` parameter of `{owner}`?",
                value,
                "default_value",
                "structured",
                (rel,),
            )
        )
    for cls, base, rel in rng.sample(sorted(set(bases)), 5):
        questions.append(
            Question(
                _qid("click", "base", cls),
                "click",
                f"In the click source, which class or classes does `{cls}` directly inherit from?",
                base,
                "base_class",
                "structured",
                (rel,),
            )
        )
    counts = Counter(s.name for s in symbols)
    single = sorted(
        (s.name, s.doc_id)
        for s in symbols
        if s.kind in ("function", "class") and counts[s.name] == 1 and not s.name.startswith("_")
    )
    for name, rel in rng.sample(single, 5):
        questions.append(
            Question(
                _qid("click", "file", name),
                "click",
                f"Which file in the click repository defines `{name}`? Give its path.",
                rel,
                "defining_file",
                "structured",
                (rel,),
            )
        )
    return questions


def questions_path(name: str) -> Path:
    """Return where a corpus's question set is kept (local, never committed)."""
    return work_root() / "questions" / f"{name}.json"


def save_questions(name: str, questions: Sequence[Question]) -> Path:
    """Write a question set to the local cache."""
    path = questions_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([q.to_dict() for q in questions], indent=1), encoding="utf-8")
    return path


def load_questions(name: str) -> list[Question]:
    """Read a question set from the local cache."""
    return [Question.from_dict(r) for r in json.loads(questions_path(name).read_text(encoding="utf-8"))]


def question_digest(questions: Sequence[Question]) -> str:
    """Return a digest of a question set, for cache keys."""
    blob = json.dumps([q.to_dict() for q in questions], sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Arms and runs


@dataclass(frozen=True)
class ArmPlan:
    """One arm as run: its tools and the system prompt it gets."""

    arm: ArmConfig
    instruction: str
    described: bool


def arm_plans(tools: ProfileTools) -> dict[str, ArmPlan]:
    """Return the four arms for one corpus. See the module docstring."""
    base = STANDARD_ARMS["C7"]
    return {
        "G": ArmPlan(base.with_(name="G"), ANSWER_INSTRUCTION, False),
        "D": ArmPlan(base.with_(name="D"), tools.instruction(ANSWER_INSTRUCTION, tools=False), True),
        "P": ArmPlan(tools.arm(base, name="P"), tools.instruction(ANSWER_INSTRUCTION), True),
        "Q": ArmPlan(tools.arm(base, name="Q", generic=False), tools.instruction(ANSWER_INSTRUCTION), True),
    }


def is_cheap(model: str) -> bool:
    """Return whether a model is a family's default cheap tier or an on-prem model."""
    cheap = {roles["generation"] for roles in DEFAULT_MODELS.values()} | set(ONPREM_MODELS)
    return model in cheap


def live_cap(model: str) -> int | None:
    """Return how many questions per corpus a live run may use on this model."""
    return None if is_cheap(model) else LIVE_CAP_ABOVE_CHEAP


def env_file() -> Path | None:
    """Return the repository's ``.env``, or the main checkout's from a git worktree.

    The resolver looks for ``.env`` upwards from the working directory, which
    misses the main checkout's file when the code runs from a worktree.
    """
    import subprocess  # noqa: PLC0415

    local = HERE.parents[1] / ".env"
    if local.is_file():
        return local
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=HERE,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    shared = Path(common).parent / ".env"
    return shared if shared.is_file() else None


_CACHES: dict[str, Any] = {}


def response_cache(cache_name: str = "ri5-describe-your-data") -> Any:
    """Return the one response cache for ``cache_name``, shared by judge and agents.

    ADK's keys are tagged, so an agent's responses never collide with the
    judge's in the same file.
    """
    from aieng.advanced_rag.graphrag.proxy import ResponseCache  # noqa: PLC0415

    if cache_name not in _CACHES:
        _CACHES[cache_name] = ResponseCache(cache_root() / "responses" / f"{cache_name}.sqlite")
    return _CACHES[cache_name]


def governed_client(per_minute: int = 20, *, cache_name: str = "ri5-describe-your-data") -> tuple[Any, MeasuredLimiter]:
    """Return a cached, throttled, metered client on the proxy, and its limiter.

    The client is for the judge; the agent arms share its limiter and its
    :func:`response_cache`. The key is read with
    :func:`aieng.advanced_rag.proxy.resolve_api_key` and never printed.
    """
    from aieng.advanced_rag.graphrag.proxy import govern  # noqa: PLC0415
    from aieng.advanced_rag.proxy import resolve_api_key, resolve_base_url  # noqa: PLC0415
    from openai import OpenAI  # noqa: PLC0415

    limiter = MeasuredLimiter(per_minute=per_minute)
    client = govern(
        OpenAI(
            base_url=resolve_base_url(env_file()),
            api_key=resolve_api_key(env_file()),
            max_retries=0,
            timeout=600.0,
        ),
        cache=response_cache(cache_name),
        limiter=limiter,
    )
    return client, limiter


def run_record(question: Question, arm: str, run: AgentRun) -> dict[str, Any]:
    """Return what a run keeps: costs, tool counts, documents, and the answer."""
    counts = Counter(event["name"] for event in run.trace)
    gold = set(question.gold_docs)
    return {
        "qid": question.qid,
        "corpus": question.corpus,
        "qtype": question.qtype,
        "source": question.source,
        "arm": arm,
        "model": run.model,
        "answer": run.answer,
        "model_calls": run.model_calls,
        "tool_calls": run.tool_calls,
        "prompt_tokens": run.prompt_tokens,
        "completion_tokens": run.completion_tokens,
        "seconds": run.seconds,
        "exhausted": run.exhausted,
        "tool_counts": dict(counts),
        "gold_read": bool(gold) and bool(gold & set(run.read_ids)),
        "gold_seen": bool(gold) and bool(gold & set(run.seen_ids)),
        "errors": sum(1 for e in run.trace if e.get("error")),
    }


def _parallel(function: Callable[[Any], Any], items: Sequence[Any], workers: int) -> list[Any]:
    """Map over threads, each in a copy of the caller's context.

    The usage meter that :func:`aieng.advanced_rag.runs.load_or_run` binds is
    a context variable, and a worker thread starts with an empty context, so
    without the copy every call a worker makes goes unmetered.
    """
    import contextvars  # noqa: PLC0415

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(contextvars.copy_context().run, function, item) for item in items]
        return [future.result() for future in futures]


def run_arms(
    name: str,
    questions: Sequence[Question],
    choice: ModelChoice,
    *,
    client: Any,
    limiter: MeasuredLimiter | None,
    arms: Sequence[str] = ARM_NAMES,
    workers: int = 4,
    search: Any = None,
    cache: Any = None,
) -> list[dict[str, Any]]:
    """Run every arm on every question of one corpus and return the run records.

    Each arm is an ADK agent run by
    :func:`~aieng.advanced_rag.agents.session_loop.run_session`, throttled by
    ``limiter`` and cached in ``cache`` (:func:`response_cache` when omitted).
    ``client`` is unused and kept so callers written for the earlier loop
    still work.
    """
    from aieng.advanced_rag.agents.session_loop import run_session  # noqa: PLC0415

    cache = cache if cache is not None else response_cache()

    _, corpus, tools = load(name)
    server = CorpusToolServer(corpus, search=search or dense_search(name, corpus))
    tools.register(server)
    plans = arm_plans(tools)
    label = f"the {name} corpus"
    jobs = [(q, a) for a in arms for q in questions]

    def one(job: tuple[Question, str]) -> dict[str, Any]:
        question, arm = job
        plan = plans[arm]
        session = server.session(plan.arm)
        run = run_session(
            question.question,
            session,
            model=choice.model,
            corpus_description=label,
            instruction=plan.instruction,
            cache=cache,
            limiter=limiter,
            env_file=env_file(),
        )
        return run_record(question, arm, run)

    return _parallel(one, jobs, workers)


def judge_records(
    records: Sequence[Mapping[str, Any]],
    questions: Sequence[Question],
    choice: ModelChoice,
    *,
    client: Any,
    workers: int = 4,
) -> list[dict[str, Any]]:
    """Judge every run's answer against its reference with the shared judge prompt."""
    from aieng.advanced_rag.evaluation.answers import judge_answer  # noqa: PLC0415

    by_id = {q.qid: q for q in questions}
    params = request_params(choice.model, max_tokens=400)
    params.pop("extra_body", None)

    def complete(prompt: str) -> str:
        response = client.chat.completions.create(
            model=choice.model, messages=[{"role": "user", "content": prompt}], **params
        )
        return response.choices[0].message.content or ""

    def one(record: Mapping[str, Any]) -> dict[str, Any]:
        question = by_id[record["qid"]]
        answer = str(record.get("answer") or "")
        if not answer or answer.lower() == "unknown":
            verdict = {"correct": False, "abstained": True, "reason": "no answer"}
        else:
            try:
                judged = judge_answer(record["qid"], question.question, answer, question.reference, [], complete)
                verdict = {"correct": judged.correct, "abstained": judged.abstained, "reason": judged.reason}
            except ValueError:
                verdict = {"correct": False, "abstained": False, "reason": "judge returned no JSON"}
        return {"qid": record["qid"], "arm": record["arm"], **verdict, "judge": choice.model}

    return _parallel(one, records, workers)


def runs_step(name: str, questions: Sequence[Question], profile: DataProfile, arms: Sequence[str] = ARM_NAMES) -> Step:
    """Return the cached step for one corpus's runs."""
    return Step(
        RI_ID,
        "describe-your-data-runs",
        PRODUCER,
        {
            "corpus": name,
            "profile": hashlib.sha256(json.dumps(profile.to_dict(), sort_keys=True).encode()).hexdigest()[:16],
            "questions": question_digest(questions),
            "arms": list(arms),
            "instruction": hashlib.sha256(ANSWER_INSTRUCTION.encode()).hexdigest()[:12],
            "arm_configs": {a: plan_record(a) for a in arms},
            "loop": AGENT_LOOP,
        },
        role="generation",
    )


def plan_record(arm: str) -> str:
    """Return a short label of what an arm offers, for the cache key."""
    return {
        "G": "C7",
        "D": "C7+description",
        "P": "C7+profile-tools+description",
        "Q": "open+profile-tools+description",
    }[arm]


def judge_step(name: str, runs_digest: str) -> Step:
    """Return the cached step for judging one corpus's runs."""
    return Step(
        RI_ID,
        "describe-your-data-judgements",
        PRODUCER,
        {"corpus": name, "prompt": "evaluation.answers.JUDGE_PROMPT"},
        role="judging",
        upstream={"runs": runs_digest},
    )


# ---------------------------------------------------------------------------
# Summaries


@dataclass
class ArmSummary:
    """One corpus and arm: accuracy and cost per question."""

    corpus: str
    arm: str
    model: str
    judge: str
    questions: int
    correct: int
    accuracy: float
    calls: float
    tool_calls: float
    prompt_tokens: float
    completion_tokens: float
    exhausted: int
    gold_read: float
    by_type: dict[str, float] = field(default_factory=dict)
    tool_mix: dict[str, int] = field(default_factory=dict)


def summarise(records: Sequence[Mapping[str, Any]], verdicts: Sequence[Mapping[str, Any]]) -> list[ArmSummary]:
    """Return one summary per corpus and arm."""
    verdict = {(v["qid"], v["arm"]): v for v in verdicts}
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["corpus"], r["arm"]), []).append(r)
    out = []
    for (corpus, arm), rows in groups.items():
        n = len(rows)
        correct = [bool(verdict.get((r["qid"], arm), {}).get("correct")) for r in rows]
        types: dict[str, list[bool]] = {}
        for r, ok in zip(rows, correct, strict=True):
            types.setdefault(r["qtype"], []).append(ok)
        mix: Counter[str] = Counter()
        for r in rows:
            mix.update(r["tool_counts"])
        with_gold = [
            r for r in rows if r.get("gold_read") is not None and r["qtype"] not in ("count", "count_two_tags")
        ]
        out.append(
            ArmSummary(
                corpus=corpus,
                arm=arm,
                model=rows[0]["model"],
                judge=next((verdict[(r["qid"], arm)]["judge"] for r in rows if (r["qid"], arm) in verdict), ""),
                questions=n,
                correct=sum(correct),
                accuracy=round(sum(correct) / n, 4),
                calls=round(sum(r["model_calls"] for r in rows) / n, 2),
                tool_calls=round(sum(r["tool_calls"] for r in rows) / n, 2),
                prompt_tokens=round(sum(r["prompt_tokens"] for r in rows) / n),
                completion_tokens=round(sum(r["completion_tokens"] for r in rows) / n),
                exhausted=sum(1 for r in rows if r["exhausted"]),
                gold_read=round(sum(1 for r in with_gold if r["gold_read"]) / len(with_gold), 4) if with_gold else 0.0,
                by_type={t: round(sum(v) / len(v), 3) for t, v in sorted(types.items())},
                tool_mix=dict(mix.most_common()),
            )
        )
    order = {name: i for i, name in enumerate(CORPUS_NAMES)}
    return sorted(out, key=lambda s: (order.get(s.corpus, 9), ARM_NAMES.index(s.arm)))


def committed_record(
    summaries: Sequence[ArmSummary],
    records: Sequence[Mapping[str, Any]],
    verdicts: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the record ``outputs/`` keeps: no question, answer, or corpus text."""
    verdict = {(v["qid"], v["arm"]): v for v in verdicts}
    keep = (
        "qid",
        "corpus",
        "qtype",
        "source",
        "arm",
        "model_calls",
        "tool_calls",
        "prompt_tokens",
        "completion_tokens",
        "exhausted",
        "gold_read",
        "gold_seen",
        "tool_counts",
    )
    return {
        **meta,
        "summaries": [asdict(s) for s in summaries],
        "per_question": [
            {**{k: r[k] for k in keep}, "correct": bool(verdict.get((r["qid"], r["arm"]), {}).get("correct"))}
            for r in records
        ],
    }


def capped(questions: Sequence[Question], cap: int | None) -> list[Question]:
    """Return at most ``cap`` questions, spread across the set so every type stays."""
    if cap is None or cap >= len(questions):
        return list(questions)
    step = len(questions) / cap
    return [questions[int(i * step)] for i in range(cap)]


def committed_results() -> dict[str, Any]:
    """Return the committed reference results from ``outputs/``."""
    return dict(json.loads((OUTPUTS / "results.json").read_text(encoding="utf-8")))


def results(
    models: ModelSet,
    *,
    live: bool = False,
    corpora: Sequence[str] = CORPUS_NAMES,
    per_minute: int = 20,
    workers: int = 4,
) -> tuple[list[dict[str, Any]], str]:
    """Return one summary row per corpus and arm, and where the rows came from.

    Each corpus goes through :func:`aieng.advanced_rag.runs.load_or_run` twice,
    for the runs and for their judgements. With ``live`` off you get your
    earlier result or the reference result from the cache. With ``live`` on
    and no result of yours, the arms run on ``models.generation`` and are
    judged by ``models.judging``, capped at :data:`LIVE_CAP_ABOVE_CHEAP`
    questions per corpus on a mid or frontier model. When the cache, the
    corpus folders, or the question sets are missing and ``live`` is off, the
    committed reference summary is returned instead.

    Returns
    -------
    tuple
        The rows (as :class:`ArmSummary` records) and a one-line origin note.
    """
    from aieng.advanced_rag.runs import ReferenceMissingError, load_or_run  # noqa: PLC0415

    client: Any = None
    limiter: MeasuredLimiter | None = None

    def connect() -> tuple[Any, MeasuredLimiter | None]:
        nonlocal client, limiter
        if client is None:
            client, limiter = governed_client(per_minute)
        return client, limiter

    records: list[dict[str, Any]] = []
    verdicts: list[dict[str, Any]] = []
    origins: set[str] = set()
    try:
        for name in corpora:
            profile, _, _ = load(name)
            questions = load_questions(name)

            def run(choice: ModelChoice, n: str = name, qs: list[Question] = questions) -> Any:
                c, lim = connect()
                return run_arms(n, capped(qs, live_cap(choice.model)), choice, client=c, limiter=lim, workers=workers)

            runs = load_or_run(runs_step(name, questions, profile), run, models=models, live=live)

            def judge(choice: ModelChoice, rs: Any = runs.value, qs: list[Question] = questions) -> Any:
                c, _ = connect()
                return judge_records(rs, qs, choice, client=c, workers=workers)

            judged = load_or_run(judge_step(name, runs.digest), judge, models=models, live=live)
            records.extend(runs.value)
            verdicts.extend(judged.value)
            origins.add(runs.origin)
    except (ReferenceMissingError, FileNotFoundError) as error:
        if live:
            raise
        committed = committed_results()
        note = (
            f"committed reference summary ({committed['model']}, judged by {committed['judge']}); "
            f"the local cache had none: {error}"
        )
        return [s for s in committed["summaries"] if s["corpus"] in corpora], note
    rows = [asdict(s) for s in summarise(records, verdicts)]
    return rows, f"from the step cache ({', '.join(sorted(origins))})"


# ---------------------------------------------------------------------------
# Command line


def _onprem() -> ModelSet:
    from aieng.advanced_rag.models import choose_models  # noqa: PLC0415

    return choose_models("onprem")


def command_prepare(names: Sequence[str]) -> None:
    """Write the corpus folders."""
    for name in names:
        folder = PREPARE[name]()
        _, corpus, tools = load(name)
        print(name, folder, json.dumps(tools.summary()))


def command_questions(names: Sequence[str], per_minute: int) -> None:
    """Build and save each corpus's question set."""
    from aieng.advanced_rag.evaluation.synthetic import proxy_model  # noqa: PLC0415
    from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder  # noqa: PLC0415

    cache = cache_root() / "responses" / "ri5-describe-your-data-questions.sqlite"
    generator = None
    for name in names:
        _, corpus, tools = load(name)
        if name == "financebench":
            questions = financebench_questions()
        else:
            if generator is None:
                generator = proxy_model("Qwen3_6-35B-A3B", cache_path=cache, per_minute=per_minute, env_path=env_file())
                checker = proxy_model(
                    "Qwen3_8-27B", cache_path=cache, per_minute=max(per_minute // 2, 4), env_path=env_file()
                )
                embedder = SentenceTransformerEmbedder()
            rng = random.Random(0)
            if name == "serverfault":
                structured = serverfault_structured(corpus_folder(name), tools)
                pool = sorted(d.doc_id for d in corpus.documents if 1500 <= len(d.text) <= 6000)
                docs = rng.sample(pool, 30)
                config = SERVERFAULT_SYNTHETIC_CONFIG
            else:
                structured = click_code_questions(corpus_folder(name))
                docs = sorted(
                    d.doc_id for d in corpus.documents if d.doc_id.startswith("docs/") and d.doc_id.endswith(".rst")
                )
                config = CLICK_SYNTHETIC_CONFIG
            need = QUESTIONS_PER_CORPUS - len(structured)
            synthetic = synthetic_questions(
                name,
                corpus,
                docs,
                config,
                limit=need,
                generator=generator,
                checker=checker,
                judge=generator,
                embedder=embedder,
            )
            questions = structured + synthetic
        path = save_questions(name, questions)
        print(name, len(questions), Counter(q.qtype for q in questions), path)


def command_run(names: Sequence[str], per_minute: int, workers: int) -> None:
    """Run and judge every arm on the on-prem reference model through the step cache."""
    from aieng.advanced_rag.runs import load_or_run  # noqa: PLC0415

    models = _onprem()
    client, limiter = governed_client(per_minute)
    all_records: list[dict[str, Any]] = []
    all_verdicts: list[dict[str, Any]] = []
    for name in names:
        profile, _, _ = load(name)
        questions = load_questions(name)
        step = runs_step(name, questions, profile)
        runs = load_or_run(
            step,
            lambda choice, n=name, qs=questions: run_arms(
                n, qs, choice, client=client, limiter=limiter, workers=workers
            ),
            models=models,
            live=True,
        )
        judged = load_or_run(
            judge_step(name, runs.digest),
            lambda choice, rs=runs.value, qs=questions: judge_records(rs, qs, choice, client=client, workers=workers),
            models=models,
            live=True,
        )
        all_records.extend(runs.value)
        all_verdicts.extend(judged.value)
    summaries = summarise(all_records, all_verdicts)
    for s in summaries:
        print(
            f"{s.corpus:13} {s.arm} acc {s.accuracy:.3f} ({s.correct}/{s.questions}) calls {s.calls:5.2f} "
            f"in {s.prompt_tokens:>8,.0f} out {s.completion_tokens:>6,.0f} exhausted {s.exhausted} "
            f"gold_read {s.gold_read:.2f} {s.by_type}"
        )
    meta = {
        "produced_by": {
            "model": models.generation.model,
            "judge": models.judging.model,
            "embeddings": models.embeddings,
            "endpoint": models.generation.endpoint,
        },
        "model": models.generation.model,
        "judge": models.judging.model,
        "embeddings": models.embeddings,
        "rate_per_minute": per_minute,
        "click_commit": CLICK_COMMIT,
        "arms": {a: plan_record(a) for a in ARM_NAMES},
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "results.json").write_text(
        json.dumps(committed_record(summaries, all_records, all_verdicts, meta), indent=1) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command: ``prepare``, ``questions``, or ``run``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("prepare", "questions", "run"))
    parser.add_argument("corpora", nargs="*", default=list(CORPUS_NAMES))
    parser.add_argument("--per-minute", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        command_prepare(args.corpora)
    elif args.command == "questions":
        command_questions(args.corpora, args.per_minute)
    else:
        command_run(args.corpora, args.per_minute, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
