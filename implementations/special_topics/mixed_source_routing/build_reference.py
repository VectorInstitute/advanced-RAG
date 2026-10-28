#!/usr/bin/env python3
"""Build the sources, the question set, and the reference run for mixed-source routing.

The corpus is the 84 FinanceBench filings as ``table_structured_reasoning``
ingested them with Docling. Each filing is split once with that topic's
table-aware chunker, and every chunk is either prose or one table (or one row
group of a large table, with its headers repeated). From those chunks come two
sources:

* a dense index over the prose chunks, local ``bge-small-en-v1.5``;
* the tables as SQLite (``filings``, ``tables``, ``cells``), the same tables
  the ``agent_native_retrieval`` FinanceBench profile declares, queried with
  one generated SQL statement and one retry, in the shape of TAG (query
  synthesis, execution, answer generation).

One more index holds every chunk, prose and table alike, for the arm that puts
everything in one place. The arms then differ only in who chooses the source:

* ``U5`` and ``U10``: no choice, one index over everything, top 5 or 10.
* ``RE``: the embedding router in ``routing.py``.
* ``RL``: the LLM router in ``routing.py``.
* ``B``: every question to both sources, with no router.
* ``A``: the ``agent_native_retrieval`` agent with the FinanceBench profile's
  generated tools (arm ``P`` of ``describe_your_data.py``), which picks its own
  tools.
* ``O``: the oracle router, which reads the gold label.

Every route's answer is computed for every question, so a router arm is a
lookup of its decision into those answers, and the scoring can say for each
mismatch whether the route the router chose also answered.

Nothing written here is committed except identifiers and metrics: FinanceBench
is CC BY-NC 4.0, so chunks, questions, answers, and responses stay in the local
caches. See ``outputs/ATTRIBUTION.md``.

Examples
--------
Chunk and embed the corpus, build the questions, and run every arm on the
on-prem model at 20 requests a minute::

    $ python build_reference.py prepare
    $ python build_reference.py questions --per-minute 20
    $ python build_reference.py run --per-minute 20

``followup`` re-scores the table route with other sources, and ``heldout``
tests those sources on 20 table questions over filings the reference
questions never touched.
"""

from __future__ import annotations

import argparse
import contextvars
import hashlib
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aieng.advanced_rag.cache.paths import root as cache_root
from aieng.advanced_rag.models import DEFAULT_MODELS, ONPREM_MODELS, ModelChoice, ModelSet, request_params
from aieng.advanced_rag.runs import Step


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
AGENT_DIR = HERE.parents[1] / "agent_native_retrieval"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import describe_your_data as dyd  # noqa: E402
import routing  # noqa: E402


RI_ID = "special-topics"
PRODUCER = "mixed_source_routing/01_route_across_sources"
QTYPES = routing.ROUTE_NAMES
PER_TYPE = 20

#: Chunk size for the ``table_structured_reasoning`` chunker, the size it measured.
CHUNK_CHARS = 1200

#: Rows of SQL output shown to the reader, and their character cap.
SQL_ROWS = 30
SQL_CHARS = 6000

#: Questions per type a live run on a mid or frontier model is capped at.
LIVE_CAP_PER_TYPE = 4

ARM_NAMES = ("U5", "U10", "RE", "RL", "B", "A", "O")
ARM_LABELS: Mapping[str, str] = {
    "U5": "one index over prose and tables, top 5",
    "U10": "one index over prose and tables, top 10",
    "RE": "embedding router (semantic-router style, local bge-small)",
    "RL": "LLM router (LlamaIndex LLMSingleSelector style), one call",
    "B": "no router: every question to both sources",
    "A": "agent_native_retrieval agent with the FinanceBench profile's tools (arm P)",
    "O": "oracle router: the gold label",
}


def work_root() -> Path:
    """Return where chunks, embeddings, and questions are kept, locally."""
    return Path(cache_root()) / "special_topics" / "mixed-source-routing"


def ingested_root() -> Path:
    """Return the Docling ingestion of the FinanceBench filings."""
    return Path(cache_root()) / "ingestion" / "financebench"


# ---------------------------------------------------------------------------
# Chunks


@dataclass(frozen=True)
class SourceChunk:
    """One chunk of one filing, either prose or a table.

    Parameters
    ----------
    chunk_id : str
        ``<filing>::<index>``, from the ``table_structured_reasoning`` chunker.
    filing : str
        The FinanceBench ``doc_name``.
    kind : str
        ``prose`` or ``table``.
    page : int
        Zero-based page of the chunk's first item.
    text : str
        The chunk as the chunker wrote it.
    """

    chunk_id: str
    filing: str
    kind: str
    page: int
    text: str


def filing_info() -> dict[str, dict[str, Any]]:
    """Return FinanceBench's document information, by ``doc_name``."""
    from aieng.advanced_rag.corpora.sources import cache_dir  # noqa: PLC0415

    path = cache_dir() / "financebench-document-information.jsonl"
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {r["doc_name"]: r for r in rows}


def filing_label(filing: str, info: Mapping[str, Mapping[str, Any]]) -> str:
    """Return a readable name for a filing, such as ``3M 2018 10K``."""
    meta = info.get(filing, {})
    company = str(meta.get("company") or filing.split("_", maxsplit=1)[0])
    period = str(meta.get("doc_period") or "")
    kind = str(meta.get("doc_type") or "").upper()
    return " ".join(x for x in (company, period, kind) if x)


def chunk_corpus(root: Path | None = None) -> list[SourceChunk]:
    """Chunk every ingested filing and label each chunk prose or table.

    The ``table_structured_reasoning`` table-aware chunker flushes the prose
    buffer before a table and gives a table (or each row group of it) a chunk
    whose span is the table's span, so a chunk is a table chunk exactly when its
    span is a table item's.
    """
    from aieng.advanced_rag.retrieval.filings import chunk_filing, item_pages, load_document  # noqa: PLC0415

    root = root or ingested_root()
    chunks: list[SourceChunk] = []
    for folder in sorted(p for p in root.iterdir() if (p / "document.json").is_file()):
        filing = chunk_filing(folder.name, load_document(folder), strategy="table_aware", max_chars=CHUNK_CHARS)
        pages = item_pages(filing.items)
        tables = {(i.start, i.end) for i in filing.items if i.kind == "table"}
        starts = [(i.start, p) for i, p in zip(filing.items, pages, strict=True)]
        for chunk in filing.chunks:
            page = next((p for s, p in reversed(starts) if s <= chunk.start), 0)
            kind = "table" if (chunk.start, chunk.end) in tables else "prose"
            chunks.append(SourceChunk(chunk.chunk_id, folder.name, kind, page, chunk.text))
    return chunks


def chunks_path() -> Path:
    """Return the local chunk file."""
    return work_root() / "chunks.jsonl"


def save_chunks(chunks: Sequence[SourceChunk]) -> Path:
    """Write the chunks to the local cache."""
    path = chunks_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    return path


def load_chunks() -> list[SourceChunk]:
    """Read the chunks from the local cache.

    Raises
    ------
    FileNotFoundError
        If ``prepare`` has not run.
    """
    path = chunks_path()
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run `python build_reference.py prepare`")
    return [SourceChunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def chunk_digest(chunks: Sequence[SourceChunk]) -> str:
    """Return a digest of the chunk identifiers and kinds, for cache keys."""
    blob = "\n".join(f"{c.chunk_id}|{c.kind}" for c in chunks).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def embedding_text(chunk: SourceChunk, label: str) -> str:
    """Return what is embedded for a chunk: the filing's name, the page, and the text.

    A chunk from the middle of a filing rarely names its company, so the
    header is what lets a question that names a company find it.
    """
    return f"{label} | page {chunk.page + 1}\n{chunk.text}"


class ChunkIndex:
    """Cosine search over chunk embeddings, optionally restricted to one kind.

    Parameters
    ----------
    chunks : sequence of SourceChunk
        Every chunk, in a fixed order.
    embedder : object
        Has ``encode`` and ``encode_queries``. Local ``bge-small-en-v1.5``.
    labels : mapping
        Filing to readable name, for :func:`embedding_text`.
    cache : Path or None, optional
        An ``.npy`` file read when present and written after encoding.
    """

    def __init__(
        self,
        chunks: Sequence[SourceChunk],
        embedder: Any,
        labels: Mapping[str, str],
        *,
        cache: Path | None = None,
    ) -> None:
        import numpy as np  # noqa: PLC0415

        self._np = np
        self.chunks = list(chunks)
        self.embedder = embedder
        if cache is not None and cache.is_file():
            matrix = np.load(cache)
        else:
            texts = [embedding_text(c, labels.get(c.filing, c.filing)) for c in self.chunks]
            matrix = np.asarray(embedder.encode(texts), dtype=np.float32)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, matrix)
        if matrix.shape[0] != len(self.chunks):
            raise ValueError(f"{matrix.shape[0]} embedding rows for {len(self.chunks)} chunks")
        self._matrix = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9, None)
        self._kinds = np.array([c.kind for c in self.chunks])

    def search(self, query: str, k: int, kind: str | None = None) -> list[SourceChunk]:
        """Return the ``k`` chunks closest to the query, best first."""
        np = self._np
        vector = np.asarray(self.embedder.encode_queries([query])[0], dtype=np.float32)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-9)
        scores = self._matrix @ vector
        if kind is not None:
            scores = np.where(self._kinds == kind, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:k]
        return [self.chunks[i] for i in order if np.isfinite(scores[i])]


def embeddings_path(chunks: Sequence[SourceChunk]) -> Path:
    """Return the cached embedding matrix for this chunk set."""
    return work_root() / "embeddings" / f"chunks-{chunk_digest(chunks)}.npy"


_EMBEDDER: Any = None


def embedder() -> Any:
    """Return the local ``bge-small-en-v1.5`` embedder, loaded once."""
    global _EMBEDDER  # noqa: PLW0603 - one model per process
    if _EMBEDDER is None:
        from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder  # noqa: PLC0415

        _EMBEDDER = SentenceTransformerEmbedder()
    return _EMBEDDER


def load_index(chunks: Sequence[SourceChunk] | None = None) -> ChunkIndex:
    """Return the index over every chunk, embedding it on first use.

    The first embedding of about 54,000 chunks takes about 40 minutes on CPU.
    """
    chunks = chunks if chunks is not None else load_chunks()
    info = filing_info()
    labels = {c.filing: filing_label(c.filing, info) for c in chunks}
    return ChunkIndex(chunks, embedder(), labels, cache=embeddings_path(chunks))


def command_prepare() -> None:
    """Chunk the corpus and embed every chunk."""
    chunks = chunk_corpus()
    save_chunks(chunks)
    kinds = Counter(c.kind for c in chunks)
    print(f"{len(chunks):,} chunks over {len({c.filing for c in chunks})} filings: {dict(kinds)}")
    load_index(chunks)
    print("embeddings at", embeddings_path(chunks))


# ---------------------------------------------------------------------------
# Model calls


class Completer:
    """Chat completions through a governed client, returning text and tokens.

    Parameters
    ----------
    client : openai.OpenAI
        Wrapped with :func:`aieng.advanced_rag.graphrag.proxy.govern`, so each
        request is cached, throttled, and metered.
    model : str
        The bare model name.
    """

    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def __call__(self, prompt: str, max_tokens: int = 400) -> routing.Reply:
        """Send one user message and return the reply."""
        from aieng.advanced_rag.metering import usage_of  # noqa: PLC0415

        params = request_params(self.model, max_tokens=max_tokens)
        # govern adds each family's thinking switch per call; leaving it out
        # here keeps it out of the cache key, as the agent loop does.
        params.pop("extra_body", None)
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], **params
        )
        usage = usage_of(response)
        text = re.sub(r"<think>.*?</think>", "", response.choices[0].message.content or "", flags=re.DOTALL)
        return routing.Reply(text.strip(), usage.prompt_tokens, usage.completion_tokens)


def parallel(function: Callable[[Any], Any], items: Sequence[Any], workers: int = 4) -> list[Any]:
    """Map over threads, each in a copy of the caller's context so metering follows."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(contextvars.copy_context().run, function, item) for item in items]
        return [f.result() for f in futures]


def parse_json(text: str) -> dict[str, Any]:
    """Return the first JSON object in a reply, or an empty dict."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if match is None:
        return {}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# Sources and routes


READER_PROMPT = """Answer the question about company filings using only the evidence below.

{evidence}

Question: {question}

Answer every part of the question. Give figures with their units as the evidence states them.
Reply with one line: Answer: <the answer, short>
If the evidence does not contain the answer, reply: Answer: not stated"""

SQL_PROMPT = """Write one SQLite query that fetches the figures needed to answer a question
about company filings. The tables were extracted from the filings:

{schema}

How to query them:
- Find the filing in `filings` by company and period; `cells.filing` holds the same filing name.
- Row labels vary in wording, so match them with lower(row_label) LIKE '%...%' using a short
  distinctive phrase. Column labels often hold a year.
- Select filing, page, table_id, row_label, column_label, and text, so the answer can be read
  from the rows. Return at most {rows} rows.
- If the question also asks for something stated in words rather than a figure, fetch only the
  figures.
{feedback}
Question: {question}

Reply with the query only, in a ```sql block."""

#: The follow-up's SQL prompt: the filing list and the three fixes the reference
#: run's failed queries called for (see the notebook's follow-up section).
SQL_LINKED_PROMPT = """Write one SQLite query that fetches the figures needed to answer a question
about company filings. The tables were extracted from the filings:

{schema}

Every filing in the tables (filing | company | period | doc_type):
{filings}

How to query them:
- Pick the filing names from the list above and filter with cells.filing IN (...). A filing
  also reports earlier years beside its own, so a figure for one year can sit in that year's
  filing or in the next one or two; include them.
- Match a line item with ONE short distinctive word or phrase: lower(row_label) LIKE '%word%'.
  Do not AND several LIKE conditions on row_label. A segment, a period, or a unit is often in
  column_label instead, such as "Twelve Months Ended June 30, 2023 Flexibles".
- Filter the period with lower(column_label) LIKE '%2021%' when the question names a year.
- Select filing, page, table_id, row_label, column_label, and text. Return at most {rows} rows.
- If the question also asks for something stated in words rather than a figure, fetch only the
  figures.
{feedback}
Question: {question}

Reply with the query only, in a ```sql block."""

RETRY_EMPTY = """
Your previous query returned no rows:
{sql}
Broaden it: check the filing name with the filings table, use a shorter LIKE pattern, or drop
the column filter.
"""

RETRY_ERROR = """
Your previous query failed:
{sql}
The error was: {error}
"""

_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Return the SQL from a reply, from a fenced block when there is one."""
    match = _SQL_BLOCK.search(text or "")
    sql = match.group(1) if match else (text or "")
    return sql.strip().rstrip(";").strip()


@dataclass
class Cost:
    """Calls and tokens one part of a route used."""

    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, reply: routing.Reply) -> None:
        """Count one reply."""
        self.model_calls += 1
        self.prompt_tokens += reply.prompt_tokens
        self.completion_tokens += reply.completion_tokens

    def plus(self, other: Cost) -> Cost:
        """Return the sum of two."""
        return Cost(
            self.model_calls + other.model_calls,
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


def render_chunks(chunks: Sequence[SourceChunk], labels: Mapping[str, str]) -> str:
    """Return retrieved chunks as numbered evidence."""
    return "\n\n".join(
        f"[{i}] {labels.get(c.filing, c.filing)}, page {c.page + 1} ({c.kind})\n{c.text}"
        for i, c in enumerate(chunks, 1)
    )


def render_rows(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Return SQL rows as evidence text, capped at :data:`SQL_CHARS`."""
    lines = [" | ".join(columns)]
    for row in rows:
        line = " | ".join("" if v is None else str(v) for v in row)
        if sum(len(x) + 1 for x in lines) + len(line) > SQL_CHARS:
            lines.append("(more rows left out)")
            break
        lines.append(line)
    return "\n".join(lines)


class Sources:
    """The two sources, the one-index baseline, and the reader over a corpus of filings.

    Parameters
    ----------
    index : ChunkIndex
        Every chunk, prose and table.
    tables : SqlTables
        The FinanceBench profile's tables.
    complete : Completer
        The model that writes SQL and answers.
    labels : mapping
        Filing to readable name.
    """

    def __init__(self, index: ChunkIndex, tables: Any, complete: Completer, labels: Mapping[str, str]) -> None:
        self.index = index
        self.tables = tables
        self.complete = complete
        self.labels = labels

    def read(self, question: str, evidence: str, cost: Cost) -> str:
        """Answer from evidence with one call, counting it in ``cost``."""
        from aieng.advanced_rag.agents.records import parse_answer  # noqa: PLC0415

        reply = self.complete(READER_PROMPT.format(evidence=evidence, question=question), 300)
        cost.add(reply)
        return str(parse_answer(reply.text))

    def filing_list(self) -> str:
        """Return every filing's name, company, period, and type, one per line."""
        _, rows, _ = self.tables.query(
            "SELECT filing, company, period, doc_type FROM filings ORDER BY filing", max_rows=1000
        )
        return "\n".join(" | ".join(str(v) for v in row) for row in rows)

    def sql_evidence(self, question: str, *, linked: bool = False) -> dict[str, Any]:
        """Write SQL for the question, run it, and retry once on an error or no rows.

        ``linked`` uses :data:`SQL_LINKED_PROMPT`, which lists every filing and
        says how labels are split between rows and columns.
        """
        cost = Cost()
        template = SQL_LINKED_PROMPT if linked else SQL_PROMPT
        extra = {"filings": self.filing_list()} if linked else {}
        attempts: list[dict[str, Any]] = []
        feedback = ""
        evidence = ""
        for _ in range(2):
            reply = self.complete(
                template.format(
                    schema=self.tables.schema(), rows=SQL_ROWS, feedback=feedback, question=question, **extra
                ),
                500,
            )
            cost.add(reply)
            sql = extract_sql(reply.text)
            try:
                columns, rows, _ = self.tables.query(sql, max_rows=SQL_ROWS)
            except Exception as error:  # noqa: BLE001 - sqlite3 raises several types
                attempts.append({"sql": sql, "error": str(error)[:300], "rows": 0})
                feedback = RETRY_ERROR.format(sql=sql, error=str(error)[:300])
                continue
            attempts.append({"sql": sql, "error": "", "rows": len(rows)})
            if rows:
                evidence = render_rows(columns, rows)
                break
            feedback = RETRY_EMPTY.format(sql=sql)
        return {"evidence": evidence, "attempts": attempts, "cost": asdict(cost)}

    def answer_all(self, question: str) -> dict[str, Any]:
        """Return every route's and every baseline's answer and cost for one question.

        Routes share their evidence: ``both`` reads the same prose chunks as
        ``prose`` and the same SQL rows as ``table``, so its cost is the SQL
        calls plus one reading call.
        """
        prose = self.index.search(question, 5, kind="prose")
        everything = {k: self.index.search(question, k) for k in (5, 10)}
        sql = self.sql_evidence(question)
        sql_cost = Cost(**sql["cost"])
        prose_text = render_chunks(prose, self.labels)
        table_text = sql["evidence"] or "(the SQL query returned no rows)"
        out: dict[str, Any] = {"sql": sql["attempts"], "routes": {}}
        plans = {
            "prose": (prose_text, Cost(), [c.chunk_id for c in prose]),
            "table": ("Rows from the filings' tables:\n" + table_text, sql_cost, []),
            "both": (
                "Rows from the filings' tables:\n" + table_text + "\n\nPassages from the filings:\n" + prose_text,
                sql_cost,
                [c.chunk_id for c in prose],
            ),
            "U5": (render_chunks(everything[5], self.labels), Cost(), [c.chunk_id for c in everything[5]]),
            "U10": (render_chunks(everything[10], self.labels), Cost(), [c.chunk_id for c in everything[10]]),
        }
        for name, (evidence, before, chunk_ids) in plans.items():
            cost = Cost()
            answer = self.read(question, evidence, cost)
            total = before.plus(cost)
            out["routes"][name] = {"answer": answer, "chunk_ids": chunk_ids, **asdict(total)}
        return out

    def answer_followup(self, question: str) -> dict[str, Any]:
        """Return the follow-up's alternative table routes and their ``both`` versions.

        ``table_vec`` reads the top 5 table chunks by dense search, with no SQL.
        ``table_sql2`` writes SQL with :data:`SQL_LINKED_PROMPT`. Each ``both_``
        route adds the same top 5 prose chunks as the ``prose`` route.
        """
        prose = self.index.search(question, 5, kind="prose")
        tables = self.index.search(question, 5, kind="table")
        sql = self.sql_evidence(question, linked=True)
        sql_cost = Cost(**sql["cost"])
        prose_text = render_chunks(prose, self.labels)
        table_text = render_chunks(tables, self.labels)
        rows_text = sql["evidence"] or "(the SQL query returned no rows)"
        plans = {
            "table_vec": (table_text, Cost(), [c.chunk_id for c in tables]),
            "both_vec": (table_text + "\n\n" + prose_text, Cost(), [c.chunk_id for c in tables + prose]),
            "table_sql2": ("Rows from the filings' tables:\n" + rows_text, sql_cost, []),
            "both_sql2": (
                "Rows from the filings' tables:\n" + rows_text + "\n\nPassages from the filings:\n" + prose_text,
                sql_cost,
                [c.chunk_id for c in prose],
            ),
        }
        out: dict[str, Any] = {"sql": sql["attempts"], "routes": {}}
        for name, (evidence, before, chunk_ids) in plans.items():
            cost = Cost()
            answer = self.read(question, evidence, cost)
            out["routes"][name] = {"answer": answer, "chunk_ids": chunk_ids, **asdict(before.plus(cost))}
        return out


# ---------------------------------------------------------------------------
# Question set


TABLE_PROMPT = """This table is from {document}, page {page}:

{table}

One cell holds {value}. Its row label is "{row}" and its column header is "{column}".

Write one question whose answer is exactly that cell's value. The question must name the
company and the filing year, name the line item and the period in the words a financial
analyst would use, and make sense with no table in front of the reader. Do not quote the value.

Return one JSON object and nothing else:
{{"question": "...", "answer": "<the value with its unit>"}}"""

PROSE_PROMPT = """This passage is from {document}, page {page}:

{passage}

Write one question a reader of this filing might ask that the passage answers with a short
phrase copied from it: a name, a place, a date, a programme, a policy, or a reason. Do not ask
for a figure from a financial table. The question must name the company and the filing year and
make sense without the passage in front of the reader.

Return one JSON object and nothing else:
{{"question": "...", "answer": "<a phrase of at most 12 words copied from the passage>"}}"""

CHECK_PROMPT = """Answer the question using only the evidence below.

{evidence}

Question: {question}

Reply with one line: Answer: <the answer, short>
If the evidence does not contain the answer, reply: Answer: not stated"""

MERGE_PROMPT = """Two questions ask about the same filing, {document}:

1. {first}
2. {second}

Write one natural question that asks both things, as an analyst would ask it in one sentence.
Keep the company, the filing year, and every detail each question needs. Do not answer them.

Return one JSON object and nothing else:
{{"question": "..."}}"""

_YEAR = re.compile(r"(?:19|20)\d\d")
_FOLD = re.compile(r"[^a-z0-9]+")


def fold(text: str) -> str:
    """Return lower-case text with punctuation and spacing collapsed, for matching."""
    return _FOLD.sub(" ", text.lower()).strip()


@dataclass(frozen=True)
class Question:
    """One question with its gold route and reference answer.

    Parameters
    ----------
    qid : str
        Stable identifier, safe to commit.
    qtype : str
        The gold route: ``prose``, ``table``, or ``both``.
    filing : str
        The filing it asks about.
    question : str
        The text the arms receive. Local only.
    reference : str
        The reference answer. Local only.
    sources : tuple of str
        Chunk identifiers, and for a table part ``<table chunk>@r<row>c<col>``.
    """

    qid: str
    qtype: str
    filing: str
    question: str
    reference: str
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        record = asdict(self)
        record["sources"] = list(self.sources)
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Question:
        """Read a record written by :meth:`to_dict`."""
        fields = dict(record)
        fields["sources"] = tuple(fields.get("sources") or ())
        return cls(**fields)


class Funnel:
    """Counts of candidates entering and passing each filter, by question kind."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], list[int]] = {}

    def check(self, kind: str, stage: str, ok: bool) -> bool:
        """Count one candidate at one stage and return ``ok``."""
        row = self.rows.setdefault((kind, stage), [0, 0])
        row[0] += 1
        row[1] += int(ok)
        return ok

    def to_rows(self) -> list[dict[str, Any]]:
        """Return the funnel as records."""
        return [{"kind": k, "stage": s, "entered": e, "passed": p} for (k, s), (e, p) in self.rows.items()]


def table_facts(filing: str, prose_texts: Sequence[str], cell_texts: set[str]) -> list[dict[str, Any]]:
    """Return candidate table cells of one filing that only a table holds.

    A cell qualifies when it is a number with at least two digits, has a row
    label and a column header naming a year, is the only value its row label
    and column header have in the filing, appears in the SQL ``cells`` table,
    and is not written as the same figure in any prose chunk of the filing.
    """
    from aieng.advanced_rag.retrieval.chunkers import serialise  # noqa: PLC0415
    from aieng.advanced_rag.retrieval.filings import item_pages, load_document  # noqa: PLC0415
    from aieng.advanced_rag.retrieval.tables import contains_value  # noqa: PLC0415

    text, items = serialise(load_document(ingested_root() / filing))
    pages = item_pages(items)
    found: list[dict[str, Any]] = []
    values: dict[tuple[str, str], set[str]] = {}
    for item in items:
        if item.grid is None:
            continue
        for fact in item.grid.facts():
            values.setdefault((fact.row_label.lower(), fact.column_header), set()).add(fact.value)
    for number, (item, page) in enumerate(zip(items, pages, strict=True)):
        if item.grid is None:
            continue
        for fact in item.grid.facts():
            label = fact.row_label.strip()
            if not (
                fact.is_answerable
                and len(re.sub(r"\D", "", fact.value)) >= 2
                and 4 <= len(label) <= 100
                and re.search(r"[A-Za-z]{3}", label)
                and _YEAR.search(fact.column_header)
                and len(values[(label.lower(), fact.column_header)]) == 1
                and fact.value.strip() in cell_texts
            ):
                continue
            if any(contains_value(p, fact.value) for p in prose_texts):
                continue
            found.append(
                {
                    "item": number,
                    "page": page,
                    "table": item.grid.to_markdown(),
                    "value": fact.value.strip(),
                    "row": label,
                    "column": fact.column_header,
                    "cell": f"r{fact.row}c{fact.column}",
                    "span": [item.start, item.end],
                }
            )
    return found


def prose_candidates(chunks: Sequence[SourceChunk]) -> list[SourceChunk]:
    """Return prose chunks with enough running text to ask about.

    Cover pages and the first three pages are left out, as are chunks that are
    mostly figures.
    """
    keep = []
    for chunk in chunks:
        text = chunk.text
        letters = sum(ch.isalpha() for ch in text)
        digits = sum(ch.isdigit() for ch in text)
        running = len(text) >= 500 and letters > 0.7 * len(text) and digits < 0.05 * len(text)
        if chunk.kind == "prose" and chunk.page >= 3 and running:
            keep.append(chunk)
    return keep


def _qid(*parts: Any) -> str:
    return "msr-" + hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()[:10]


def build_questions(  # noqa: PLR0915 - one pass through the funnel reads best in one place
    *,
    writer: Completer,
    checker: Completer,
    judge: Completer,
    per_type: int = PER_TYPE,
    filings: int = 36,
    seed: int = 0,
    workers: int = 4,
    exclude: Collection[str] = (),
    kinds: Collection[str] = ("table", "prose"),
) -> tuple[list[Question], Funnel]:
    """Write table, prose, and two-source questions over sampled filings, checking each.

    ``exclude`` names filings to leave out, and ``kinds`` limits the candidates
    to ``table``, ``prose``, or both. With only ``table``, no two-source
    question is merged.

    A table candidate is checked by answering it from its table alone, and a
    prose candidate from its passage alone, with the judge comparing the
    checker's answer with the reference. A ``both`` question merges one
    accepted table question and one accepted prose question on the same
    filing, so it needs a figure only a table holds and a statement only the
    text holds.
    """
    from aieng.advanced_rag.agents.records import parse_answer  # noqa: PLC0415
    from aieng.advanced_rag.evaluation.answers import judge_answer  # noqa: PLC0415

    chunks = load_chunks()
    info = filing_info()
    by_filing: dict[str, list[SourceChunk]] = {}
    for chunk in chunks:
        by_filing.setdefault(chunk.filing, []).append(chunk)
    cells = cell_texts()
    rng = random.Random(seed)
    chosen = sorted(set(by_filing) - set(exclude))
    rng.shuffle(chosen)
    chosen = chosen[:filings]
    funnel = Funnel()

    def correct(qid: str, question: str, reference: str, evidence: str) -> bool:
        reply = checker(CHECK_PROMPT.format(evidence=evidence, question=question), 200)
        answer = parse_answer(reply.text)
        if not answer or "not stated" in answer.lower():
            return False
        try:
            verdict = judge_answer(qid, question, answer, reference, [], lambda p: judge(p, 300).text)
        except ValueError:
            return False
        return bool(verdict.correct)

    jobs: list[tuple[str, str, Any]] = []
    for filing in chosen:
        own = by_filing[filing]
        prose_texts = [c.text for c in own if c.kind == "prose"]
        facts = table_facts(filing, prose_texts, cells.get(filing, set()))
        funnel.check("table", "filing has a table-only cell", bool(facts))
        for fact in rng.sample(facts, min(2, len(facts))):
            jobs.append(("table", filing, fact))
        if "prose" not in kinds:
            continue
        passages = prose_candidates(own)
        funnel.check("prose", "filing has running text", bool(passages))
        for chunk in rng.sample(passages, min(2, len(passages))):
            jobs.append(("prose", filing, chunk))

    def one(job: tuple[str, str, Any]) -> dict[str, Any] | None:
        kind, filing, source = job
        document = filing_label(filing, info)
        table_texts = [c.text for c in by_filing[filing] if c.kind == "table"]
        if kind == "table":
            prompt = TABLE_PROMPT.format(
                document=document,
                page=source["page"] + 1,
                table=source["table"][:6000],
                value=source["value"],
                row=source["row"],
                column=source["column"],
            )
            evidence = source["table"][:6000]
            anchor = f"{filing}#i{source['item']:04d}@{source['cell']}"
        else:
            prompt = PROSE_PROMPT.format(document=document, page=source.page + 1, passage=source.text)
            evidence = source.text
            anchor = source.chunk_id
        written = parse_json(writer(prompt, 300).text)
        question = str(written.get("question", "")).strip()
        answer = str(written.get("answer", "")).strip()
        if not funnel.check(kind, "question written", bool(question and answer)):
            return None
        if kind == "table":
            years = _YEAR.findall(source["column"])
            if not funnel.check(kind, "question names the column's year", any(y in question for y in years)):
                return None
            reference = answer if re.sub(r"\D", "", source["value"]) in re.sub(r"\D", "", answer) else source["value"]
        else:
            if not funnel.check(kind, "answer copied from the passage", fold(answer) in fold(source.text)):
                return None
            if not funnel.check(
                kind, "answer absent from the filing's tables", not any(fold(answer) in fold(t) for t in table_texts)
            ):
                return None
            reference = answer
        if not funnel.check(
            kind, "checker answers from the source alone", correct(anchor, question, reference, evidence)
        ):
            return None
        return {"kind": kind, "filing": filing, "question": question, "reference": reference, "anchor": anchor}

    accepted = [r for r in parallel(one, jobs, workers) if r is not None]
    pools: dict[str, dict[str, list[dict[str, Any]]]] = {"table": {}, "prose": {}}
    for record in accepted:
        pools[record["kind"]].setdefault(record["filing"], []).append(record)

    # Pair first, so every filing that can give a two-source question gives one,
    # then fill the single-source types from what is left.
    pairs = []
    for filing in sorted(set(pools["table"]) & set(pools["prose"])):
        pairs.append((pools["table"][filing].pop(0), pools["prose"][filing].pop(0)))
    rng.shuffle(pairs)
    pairs = pairs[:per_type]

    def merge(pair: tuple[dict[str, Any], dict[str, Any]]) -> Question | None:
        table, prose = pair
        written = parse_json(
            writer(
                MERGE_PROMPT.format(
                    document=filing_label(table["filing"], info), first=table["question"], second=prose["question"]
                ),
                300,
            ).text
        )
        question = str(written.get("question", "")).strip()
        kept = all(y in question for y in _YEAR.findall(table["question"]))
        if not funnel.check("both", "questions merged, years kept", bool(question) and kept):
            return None
        return Question(
            _qid("both", table["anchor"], prose["anchor"]),
            "both",
            table["filing"],
            question,
            f"(1) {table['reference']}; (2) {prose['reference']}",
            (table["anchor"], prose["anchor"]),
        )

    both = [q for q in parallel(merge, pairs, workers) if q is not None]
    singles: list[Question] = []
    for kind in ("prose", "table"):
        if kind not in kinds:
            continue
        rest = [r for records in pools[kind].values() for r in records]
        rng.shuffle(rest)
        for record in rest[:per_type]:
            singles.append(
                Question(
                    _qid(kind, record["anchor"]),
                    kind,
                    record["filing"],
                    record["question"],
                    record["reference"],
                    (record["anchor"],),
                )
            )
    order = {name: i for i, name in enumerate(QTYPES)}
    questions = sorted(singles + both, key=lambda q: (order[q.qtype], q.qid))
    return questions, funnel


def cell_texts() -> dict[str, set[str]]:
    """Return, per filing, every cell text in the profile's ``cells`` table."""
    import csv  # noqa: PLC0415

    found: dict[str, set[str]] = {}
    with (dyd.corpus_folder("financebench") / "cells.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            found.setdefault(row["filing"], set()).add(row["text"].strip())
    return found


def questions_path() -> Path:
    """Return the local question file (never committed)."""
    return work_root() / "questions.json"


def save_questions(questions: Sequence[Question], funnel: Funnel, meta: Mapping[str, Any]) -> Path:
    """Write the question set and its funnel to the local cache."""
    path = questions_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**meta, "funnel": funnel.to_rows(), "questions": [q.to_dict() for q in questions]}, indent=1),
        encoding="utf-8",
    )
    return path


def load_questions() -> tuple[list[Question], dict[str, Any]]:
    """Read the question set and its metadata from the local cache.

    Raises
    ------
    FileNotFoundError
        If ``questions`` has not run.
    """
    path = questions_path()
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run `python build_reference.py questions`")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Question.from_dict(q) for q in payload.pop("questions")], payload


def question_digest(questions: Sequence[Question]) -> str:
    """Return a digest of a question set, for cache keys."""
    blob = json.dumps([q.to_dict() for q in questions], sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Runs


def governed_client(per_minute: int = 20) -> tuple[Any, Any]:
    """Return a cached, throttled, metered client on the proxy, and its limiter."""
    client, limiter = dyd.governed_client(per_minute, cache_name="special-topics-mixed-source-routing")
    return client, limiter


def sql_tables() -> Any:
    """Return the FinanceBench profile's tables, loaded into read-only SQLite."""
    _, _, tools = dyd.load("financebench")
    return tools.tables


def is_cheap(model: str) -> bool:
    """Return whether a model is a family's cheap tier or an on-prem model."""
    return model in {roles["generation"] for roles in DEFAULT_MODELS.values()} | set(ONPREM_MODELS)


def capped(questions: Sequence[Question], model: str) -> list[Question]:
    """Return all questions on a cheap model, else a few per type."""
    if is_cheap(model):
        return list(questions)
    out: list[Question] = []
    for qtype in QTYPES:
        out += [q for q in questions if q.qtype == qtype][:LIVE_CAP_PER_TYPE]
    return out


def run_pipelines(
    questions: Sequence[Question], choice: ModelChoice, *, client: Any, workers: int = 4
) -> list[dict[str, Any]]:
    """Run every route, both one-index arms, and the LLM router on each question."""
    complete = Completer(client, choice.model)
    chunks = load_chunks()
    info = filing_info()
    labels = {c.filing: filing_label(c.filing, info) for c in chunks}
    sources = Sources(load_index(chunks), sql_tables(), complete, labels)
    router = routing.LLMRouter(routing.FILING_ROUTES, complete)

    def one(question: Question) -> dict[str, Any]:
        record = sources.answer_all(question.question)
        record["llm_route"] = router.route(question.question).to_dict()
        return {"qid": question.qid, "qtype": question.qtype, "model": choice.model, **record}

    return parallel(one, list(questions), workers)


def agent_cache_path() -> Path:
    """Return the response cache of the agent arm's ADK runs, kept locally."""
    return Path(cache_root()) / "responses" / "special-topics-mixed-source-routing-adk.sqlite"


def run_agent(
    questions: Sequence[Question], choice: ModelChoice, *, limiter: Any, workers: int = 4
) -> list[dict[str, Any]]:
    """Run the ``agent_native_retrieval`` agent with the profile's tools (arm ``P``).

    Each question is one Google ADK run through
    :func:`aieng.advanced_rag.agents.session_loop.run_session`, with the arm's
    call budget, forced final answer, and read-first guard. Every model
    response is cached in :func:`agent_cache_path`, so a second run of the
    same questions makes no model call.
    """
    from aieng.advanced_rag.agents.corpus_tools import CorpusToolServer  # noqa: PLC0415
    from aieng.advanced_rag.agents.session_loop import run_session  # noqa: PLC0415
    from aieng.advanced_rag.graphrag.proxy import ResponseCache  # noqa: PLC0415

    _, corpus, tools = dyd.load("financebench")
    server = CorpusToolServer(corpus, search=dyd.dense_search("financebench", corpus, embedder()))
    tools.register(server)
    plan = dyd.arm_plans(tools)["P"]
    cache = ResponseCache(agent_cache_path())

    def one(question: Question) -> dict[str, Any]:
        run = run_session(
            question.question,
            server.session(plan.arm),
            model=choice.model,
            corpus_description="the financebench corpus",
            instruction=plan.instruction,
            cache=cache,
            limiter=limiter,
            # From a git worktree the key lookup misses the main checkout's .env.
            env_file=dyd.env_file(),
        )
        sql_calls = sum(1 for e in run.trace if e["name"] == "sql")
        text_calls = sum(1 for e in run.trace if e["name"] in ("search", "grep", "open", "outline", "open_section"))
        return {
            "qid": question.qid,
            "qtype": question.qtype,
            "model": run.model,
            "answer": run.answer,
            "model_calls": run.model_calls,
            "tool_calls": run.tool_calls,
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
            "exhausted": run.exhausted,
            "tool_counts": dict(Counter(e["name"] for e in run.trace)),
            "route": agent_route(sql_calls, text_calls),
        }

    return parallel(one, list(questions), workers)


def agent_route(sql_calls: int, text_calls: int) -> str:
    """Return which sources an agent run touched, as a route name.

    ``sql`` is the table source; every other tool reads the text. A run that
    used neither counts as ``prose``, the agent's default.
    """
    if sql_calls and text_calls:
        return "both"
    return "table" if sql_calls else "prose"


def judge_answers(
    pipelines: Sequence[Mapping[str, Any]],
    agent: Sequence[Mapping[str, Any]],
    questions: Sequence[Question],
    choice: ModelChoice,
    *,
    client: Any,
    workers: int = 4,
) -> list[dict[str, Any]]:
    """Judge every distinct answer against its reference with the shared prompt."""
    from aieng.advanced_rag.evaluation.answers import judge_answer  # noqa: PLC0415

    complete = Completer(client, choice.model)
    by_id = {q.qid: q for q in questions}
    jobs = [(r["qid"], name, route["answer"]) for r in pipelines for name, route in r["routes"].items()]
    jobs += [(r["qid"], "A", r["answer"]) for r in agent]

    def one(job: tuple[str, str, str]) -> dict[str, Any]:
        qid, name, answer = job
        question = by_id[qid]
        if not answer or answer.strip().lower() in ("unknown", "not stated"):
            return {"qid": qid, "route": name, "correct": False, "abstained": True, "judge": choice.model}
        try:
            verdict = judge_answer(
                qid, question.question, answer, question.reference, [], lambda p: complete(p, 400).text
            )
        except ValueError:
            return {"qid": qid, "route": name, "correct": False, "abstained": False, "judge": choice.model}
        return {
            "qid": qid,
            "route": name,
            "correct": bool(verdict.correct),
            "abstained": bool(verdict.abstained),
            "judge": choice.model,
        }

    return parallel(one, jobs, workers)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def pipelines_step(questions: Sequence[Question], chunks_id: str) -> Step:
    """Return the cached step for the routes, baselines, and LLM router."""
    return Step(
        RI_ID,
        "mixed-routing-pipelines",
        PRODUCER,
        {
            "questions": question_digest(questions),
            "chunks": chunks_id,
            "reader": _digest(READER_PROMPT),
            "sql": _digest(SQL_PROMPT + RETRY_EMPTY + RETRY_ERROR),
            "router": _digest(routing.LLM_ROUTER_PROMPT + json.dumps([asdict(r) for r in routing.FILING_ROUTES])),
            "k": [5, 10],
            "embeddings": "BAAI/bge-small-en-v1.5",
        },
        role="generation",
    )


def agent_step(questions: Sequence[Question]) -> Step:
    """Return the cached step for the agent arm."""
    return Step(
        RI_ID,
        "mixed-routing-agent",
        PRODUCER,
        {
            "questions": question_digest(questions),
            "arm": "describe_your_data P on financebench",
            "instruction": _digest(dyd.ANSWER_INSTRUCTION),
            "loop": "google-adk session_loop",
        },
        role="generation",
    )


def judge_step(pipelines_digest: str, agent_digest: str) -> Step:
    """Return the cached step for judging both."""
    return Step(
        RI_ID,
        "mixed-routing-judgements",
        PRODUCER,
        {"prompt": "evaluation.answers.JUDGE_PROMPT"},
        role="judging",
        upstream={"pipelines": pipelines_digest, "agent": agent_digest},
    )


# ---------------------------------------------------------------------------
# Arms and summaries


def embedding_routes(questions: Sequence[Question]) -> dict[str, dict[str, Any]]:
    """Return the embedding router's decision for every question. No model call."""
    router = routing.EmbeddingRouter(routing.FILING_ROUTES, embedder())
    return {q.qid: router.route(q.question).to_dict() for q in questions}


def arm_records(
    questions: Sequence[Question],
    pipelines: Sequence[Mapping[str, Any]],
    agent: Sequence[Mapping[str, Any]],
    verdicts: Sequence[Mapping[str, Any]],
    embedded: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return one record per question and arm: route, correctness, calls, and tokens.

    A router arm's cost is its own decision's cost plus the chosen route's.
    """
    correct = {(v["qid"], v["route"]): bool(v["correct"]) for v in verdicts}
    by_q = {q.qid: q for q in questions}
    records: list[dict[str, Any]] = []
    for row in pipelines:
        qid = row["qid"]
        gold = by_q[qid].qtype
        routes = row["routes"]
        decisions = {
            "U5": ("U5", None),
            "U10": ("U10", None),
            "RE": (embedded[qid]["route"], embedded[qid]),
            "RL": (row["llm_route"]["route"], row["llm_route"]),
            "B": ("both", None),
            "O": (gold, None),
        }
        for arm, (route, decision) in decisions.items():
            took = routes[route]
            records.append(
                {
                    "qid": qid,
                    "qtype": gold,
                    "arm": arm,
                    "model": row["model"],
                    "route": route if route in routing.ROUTE_NAMES else "",
                    "correct": correct.get((qid, route), False),
                    "model_calls": took["model_calls"] + (decision or {}).get("model_calls", 0),
                    "prompt_tokens": took["prompt_tokens"] + (decision or {}).get("prompt_tokens", 0),
                    "completion_tokens": took["completion_tokens"] + (decision or {}).get("completion_tokens", 0),
                }
            )
    for row in agent:
        records.append(
            {
                "qid": row["qid"],
                "qtype": row["qtype"],
                "arm": "A",
                "model": row["model"],
                "route": row["route"],
                "correct": correct.get((row["qid"], "A"), False),
                "model_calls": row["model_calls"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "exhausted": row["exhausted"],
                "tool_counts": row["tool_counts"],
            }
        )
    return records


def summaries(
    records: Sequence[Mapping[str, Any]], verdicts: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return one summary per arm and the per-route outcome table.

    The second value holds, per gold type, how often each fixed route answered
    correctly, and the hindsight labels: the cheapest route that answered.
    """
    gold = {r["qid"]: r["qtype"] for r in records}
    by_route: dict[str, dict[str, bool]] = {}
    for v in verdicts:
        by_route.setdefault(v["qid"], {})[v["route"]] = bool(v["correct"])
    out = []
    for arm in ARM_NAMES:
        rows = [r for r in records if r["arm"] == arm]
        if not rows:
            continue
        report = None
        if arm in ("RE", "RL", "A"):
            report = routing.routing_report({r["qid"]: r["route"] for r in rows}, gold, by_route)
        summary = asdict(routing.summarise_arm(arm, rows, routing=report))
        summary["label"] = ARM_LABELS[arm]
        if arm == "A":
            summary["exhausted"] = sum(1 for r in rows if r.get("exhausted"))
            mix: Counter[str] = Counter()
            for r in rows:
                mix.update(r.get("tool_counts", {}))
            summary["tool_mix"] = dict(mix.most_common())
        out.append(summary)
    outcome: dict[str, Any] = {}
    for qtype in QTYPES:
        qids = [q for q, t in gold.items() if t == qtype]
        outcome[qtype] = {
            "questions": len(qids),
            "route_accuracy": {
                route: round(sum(by_route.get(q, {}).get(route, False) for q in qids) / len(qids), 4) if qids else 0.0
                for route in (*routing.ROUTE_NAMES, "U5", "U10", "A")
            },
            "hindsight": dict(
                Counter(
                    str(routing.hindsight_route({k: by_route.get(q, {}).get(k, False) for k in routing.ROUTE_NAMES}))
                    for q in qids
                )
            ),
            "any_route_correct": sum(
                any(by_route.get(q, {}).get(k, False) for k in (*routing.ROUTE_NAMES, "U5", "U10", "A")) for q in qids
            ),
        }
    return out, outcome


def followup_results() -> dict[str, Any]:
    """Return the committed follow-up: every router under every table source."""
    return dict(json.loads((OUTPUTS / "followup_sources.json").read_text(encoding="utf-8")))


def committed_results() -> dict[str, Any]:
    """Return the committed reference results."""
    return dict(json.loads((OUTPUTS / "results.json").read_text(encoding="utf-8")))


def results(
    models: ModelSet, *, live: bool = False, per_minute: int = 20, workers: int = 4
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Return the arm summaries, the per-route outcome table, and where they came from.

    Each of the three steps goes through :func:`aieng.advanced_rag.runs.load_or_run`.
    With ``live`` off you get your earlier result or the reference from the
    cache; with ``live`` on and no result of yours, the steps run on
    ``models.generation`` and ``models.judging``, capped at
    :data:`LIVE_CAP_PER_TYPE` questions per type on a mid or frontier model.
    When the caches or the question set are missing and ``live`` is off, the
    committed reference summary is returned.
    """
    from aieng.advanced_rag.runs import ReferenceMissingError, load_or_run  # noqa: PLC0415

    client: Any = None
    limiter: Any = None

    def connect() -> tuple[Any, Any]:
        nonlocal client, limiter
        if client is None:
            client, limiter = governed_client(per_minute)
        return client, limiter

    try:
        questions, _ = load_questions()
        chunks_id = chunk_digest(load_chunks())
        pipes = load_or_run(
            pipelines_step(questions, chunks_id),
            lambda c: run_pipelines(capped(questions, c.model), c, client=connect()[0], workers=workers),
            models=models,
            live=live,
        )
        agent = load_or_run(
            agent_step(questions),
            lambda c: run_agent(capped(questions, c.model), c, limiter=connect()[1], workers=workers),
            models=models,
            live=live,
        )
        judged = load_or_run(
            judge_step(pipes.digest, agent.digest),
            lambda c: judge_answers(pipes.value, agent.value, questions, c, client=connect()[0], workers=workers),
            models=models,
            live=live,
        )
    except (ReferenceMissingError, FileNotFoundError) as error:
        if live:
            raise
        committed = committed_results()
        note = (
            f"committed reference summary ({committed['produced_by']['model']}, judged by "
            f"{committed['produced_by']['judge']}); the local cache had none: {error}"
        )
        return committed["summaries"], committed["outcome"], note
    asked = {r["qid"] for r in pipes.value}
    shown = [q for q in questions if q.qid in asked]
    records = arm_records(shown, pipes.value, agent.value, judged.value, embedding_routes(shown))
    rows, outcome = summaries(records, judged.value)
    return rows, outcome, f"from the step cache ({pipes.origin}, {agent.origin}, {judged.origin})"


# ---------------------------------------------------------------------------
# Follow-up: other table sources, and embedding routers with other examples

#: The table source behind each route set. ``sql_plain`` is the reference run's;
#: ``table_chunks`` reads table chunks by dense search; ``sql_linked`` writes SQL
#: from :data:`SQL_LINKED_PROMPT`.
ROUTE_SETS: Mapping[str, Mapping[str, str]] = {
    "sql_plain": {"prose": "prose", "table": "table", "both": "both"},
    "table_chunks": {"prose": "prose", "table": "table_vec", "both": "both_vec"},
    "sql_linked": {"prose": "prose", "table": "table_sql2", "both": "both_sql2"},
}

#: Router arms re-scored under every route set.
FOLLOWUP_ARMS: Mapping[str, str] = {
    "RE": "embedding router, the hand-written examples",
    "REm": "embedding router, company names, years, and form types masked",
    "REl": "embedding router, leave-one-out over the labelled questions",
    "RL": "LLM router",
    "B": "no router: both",
    "O": "oracle router",
}


def run_followup(
    questions: Sequence[Question], choice: ModelChoice, *, client: Any, workers: int = 4
) -> list[dict[str, Any]]:
    """Run the follow-up's table routes on every question."""
    complete = Completer(client, choice.model)
    chunks = load_chunks()
    info = filing_info()
    labels = {c.filing: filing_label(c.filing, info) for c in chunks}
    sources = Sources(load_index(chunks), sql_tables(), complete, labels)

    def one(question: Question) -> dict[str, Any]:
        return {
            "qid": question.qid,
            "qtype": question.qtype,
            "model": choice.model,
            **sources.answer_followup(question.question),
        }

    return parallel(one, list(questions), workers)


def followup_step(questions: Sequence[Question], chunks_id: str) -> Step:
    """Return the cached step for the follow-up's routes."""
    return Step(
        RI_ID,
        "mixed-routing-followup",
        PRODUCER,
        {
            "questions": question_digest(questions),
            "chunks": chunks_id,
            "reader": _digest(READER_PROMPT),
            "sql": _digest(SQL_LINKED_PROMPT + RETRY_EMPTY + RETRY_ERROR),
            "k": 5,
        },
        role="generation",
    )


def followup_judge_step(followup_digest: str) -> Step:
    """Return the cached step for judging the follow-up's routes."""
    return Step(
        RI_ID,
        "mixed-routing-followup-judgements",
        PRODUCER,
        {"prompt": "evaluation.answers.JUDGE_PROMPT"},
        role="judging",
        upstream={"followup": followup_digest},
    )


_FORM = re.compile(r"\b(?:10-?K|10-?Q|8-?K|annual report|earnings (?:release|filing|report))\b", re.IGNORECASE)


def mask_entities(text: str, companies: Iterable[str]) -> str:
    """Replace company names, years, and form types, leaving the information need.

    Parameters
    ----------
    text : str
        A question or an example.
    companies : iterable of str
        Company names to mask, longest first so a name's prefix does not win.
    """
    masked = text
    for name in sorted({c for c in companies if c}, key=len, reverse=True):
        masked = re.sub(re.escape(name) + r"(?:'s|\u2019s)?", "the company", masked, flags=re.IGNORECASE)
    masked = _YEAR.sub("<year>", masked)
    return _FORM.sub("the filing", masked)


def embedding_variants(questions: Sequence[Question]) -> dict[str, dict[str, str]]:
    """Return three embedding routers' decisions per question. No model call.

    ``RE`` uses the hand-written examples, ``REm`` the same with company
    names, years, and form types masked in examples and questions, and ``REl``
    uses the other labelled questions as the examples (leave one out), which is
    what a team gets after labelling a few dozen of its own questions.
    """
    import numpy as np  # noqa: PLC0415

    companies = [str(m.get("company", "")) for m in filing_info().values()]
    companies += [c.split()[0] for c in companies if c]
    plain = routing.EmbeddingRouter(routing.FILING_ROUTES, embedder())
    masked_routes = tuple(
        routing.Route(r.name, r.description, tuple(mask_entities(u, companies) for u in r.utterances))
        for r in routing.FILING_ROUTES
    )
    masked = routing.EmbeddingRouter(masked_routes, embedder())
    vectors = np.asarray(embedder().encode_queries([q.question for q in questions]), dtype=np.float32)
    vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)
    sims = vectors @ vectors.T
    out: dict[str, dict[str, str]] = {"RE": {}, "REm": {}, "REl": {}}
    for i, q in enumerate(questions):
        out["RE"][q.qid] = plain.route(q.question).route
        out["REm"][q.qid] = masked.route(mask_entities(q.question, companies)).route
        scores = {}
        for name in routing.ROUTE_NAMES:
            own = sorted(
                (float(sims[i, j]) for j, o in enumerate(questions) if j != i and o.qtype == name), reverse=True
            )
            scores[name] = sum(own[:3]) / len(own[:3]) if own else -1.0
        out["REl"][q.qid] = max(scores, key=lambda n: (scores[n], -routing.ROUTE_ORDER[n]))
    return out


def followup_summary(
    questions: Sequence[Question],
    pipelines: Sequence[Mapping[str, Any]],
    followup: Sequence[Mapping[str, Any]],
    verdicts: Sequence[Mapping[str, Any]],
    embedded: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Return every router arm under every route set, with routing reports.

    A router's decision is a route name; the route set says which source
    answers it. Costs add the LLM router's own call for ``RL``.
    """
    gold = {q.qid: q.qtype for q in questions}
    rows = {r["qid"]: {**r["routes"]} for r in pipelines}
    for r in followup:
        rows[r["qid"]].update(r["routes"])
    llm = {r["qid"]: r["llm_route"] for r in pipelines}
    correct = {(v["qid"], v["route"]): bool(v["correct"]) for v in verdicts}
    decisions: dict[str, dict[str, str]] = {
        **{k: dict(v) for k, v in embedded.items()},
        "RL": {q: d["route"] for q, d in llm.items()},
        "B": dict.fromkeys(gold, "both"),
        "O": dict(gold),
    }
    out: dict[str, Any] = {}
    for set_name, mapping in ROUTE_SETS.items():
        by_route = {q: {r: correct.get((q, mapping[r]), False) for r in routing.ROUTE_NAMES} for q in gold}
        arms = []
        for arm in FOLLOWUP_ARMS:
            records = []
            for qid, route in decisions[arm].items():
                took = rows[qid][mapping[route]]
                extra = llm[qid] if arm == "RL" else {}
                records.append(
                    {
                        "qtype": gold[qid],
                        "route": route,
                        "model": "",
                        "correct": correct.get((qid, mapping[route]), False),
                        "model_calls": took["model_calls"] + extra.get("model_calls", 0),
                        "prompt_tokens": took["prompt_tokens"] + extra.get("prompt_tokens", 0),
                        "completion_tokens": took["completion_tokens"] + extra.get("completion_tokens", 0),
                    }
                )
            report = routing.routing_report(decisions[arm], gold, by_route) if arm not in ("B", "O") else None
            summary = asdict(routing.summarise_arm(arm, records, routing=report))
            summary.pop("model")
            summary["label"] = FOLLOWUP_ARMS[arm]
            arms.append(summary)
        route_accuracy = {
            qtype: {
                r: round(
                    sum(by_route[q][r] for q in gold if gold[q] == qtype) / sum(1 for q in gold if gold[q] == qtype), 4
                )
                for r in routing.ROUTE_NAMES
            }
            for qtype in QTYPES
        }
        out[set_name] = {"arms": arms, "route_accuracy": route_accuracy}
    return out


def prose_recall(questions: Sequence[Question], index: ChunkIndex | None = None) -> dict[str, int]:
    """Return how many prose questions find their source chunk in each search.

    No model call: the ``prose`` route's top 5 over prose chunks, and the
    one-index arms' top 5 and top 10 over every chunk.
    """
    index = index or load_index()
    prose = [q for q in questions if q.qtype == "prose"]
    found = {"questions": len(prose), "prose_top5": 0, "all_top5": 0, "all_top10": 0}
    for q in prose:
        gold = q.sources[0]
        found["prose_top5"] += gold in {c.chunk_id for c in index.search(q.question, 5, kind="prose")}
        found["all_top5"] += gold in {c.chunk_id for c in index.search(q.question, 5)}
        found["all_top10"] += gold in {c.chunk_id for c in index.search(q.question, 10)}
    return found


def command_followup(per_minute: int, workers: int) -> None:
    """Run the follow-up routes on the on-prem model and write their results."""
    from aieng.advanced_rag.runs import load_or_run  # noqa: PLC0415

    models = _onprem()
    client, _ = governed_client(per_minute)
    questions, _ = load_questions()
    chunks = load_chunks()
    chunks_id = chunk_digest(chunks)
    pipes = _cached(pipelines_step(questions, chunks_id), models)
    agent = _cached(agent_step(questions), models)
    judged = _cached(judge_step(pipes.digest, agent.digest), models)
    follow = load_or_run(
        followup_step(questions, chunks_id),
        lambda c: run_followup(questions, c, client=client, workers=workers),
        models=models,
        live=True,
    )
    follow_judged = load_or_run(
        followup_judge_step(follow.digest),
        lambda c: judge_answers(follow.value, [], questions, c, client=client, workers=workers),
        models=models,
        live=True,
    )
    embedded = embedding_variants(questions)
    summary = followup_summary(questions, pipes.value, follow.value, [*judged.value, *follow_judged.value], embedded)
    for set_name, block in summary.items():
        for arm in block["arms"]:
            types = "  ".join(f"{t} {v['accuracy']:.2f}" for t, v in arm["by_type"].items())
            agree = f"agree {arm['routing']['agreement']:.2f}" if arm["routing"] else ""
            print(f"{set_name:12} {arm['arm']:4} acc {arm['accuracy']:.3f} calls {arm['calls']:5.2f} | {types} {agree}")
    sql = [
        {"qid": r["qid"], "attempts": len(r["sql"]), "rows": r["sql"][-1]["rows"] if r["sql"] else 0}
        for r in follow.value
    ]
    payload = {
        "produced_by": produced_by(models, rate_per_minute=per_minute),
        "route_sets": {k: dict(v) for k, v in ROUTE_SETS.items()},
        "arms": dict(FOLLOWUP_ARMS),
        "results": summary,
        "sql_linked_attempts": sql,
        "prose_gold_chunk_recall": prose_recall(questions),
    }
    (OUTPUTS / "followup_sources.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Held-out table questions

#: The table sources compared on the held-out questions: the reference run's
#: SQL prompt, the follow-up's SQL prompt, and dense search over table chunks.
HELDOUT_ROUTES: Mapping[str, str] = {
    "sql_plain": "SQL from the reference prompt",
    "sql_linked": "SQL from the follow-up prompt (filing list and matching rules)",
    "table_chunks": "top 5 table chunks by dense search",
    "table_hybrid": "follow-up SQL rows and the top 5 table chunks, one reading call",
}

#: The three sources that existed before the held-out run. ``table_hybrid`` was
#: designed after seeing their held-out scores, so its score is itself in-sample.
SINGLE_SOURCES = ("sql_plain", "sql_linked", "table_chunks")

HELDOUT_SEED = 1


def heldout_path() -> Path:
    """Return the local held-out question file (never committed)."""
    return work_root() / "heldout_questions.json"


def load_heldout() -> list[Question]:
    """Read the held-out table questions from the local cache.

    Raises
    ------
    FileNotFoundError
        If ``heldout`` has not built them.
    """
    path = heldout_path()
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run `python build_reference.py heldout`")
    return [Question.from_dict(q) for q in json.loads(path.read_text(encoding="utf-8"))["questions"]]


def answer_heldout(sources: Sources, question: str) -> dict[str, Any]:
    """Return the three table sources' answers and costs for one question."""
    tables = sources.index.search(question, 5, kind="table")
    plain = sources.sql_evidence(question)
    linked = sources.sql_evidence(question, linked=True)
    plans = {
        "sql_plain": (plain, Cost(**plain["cost"])),
        "sql_linked": (linked, Cost(**linked["cost"])),
    }
    out: dict[str, Any] = {"sql": {name: sql["attempts"] for name, (sql, _) in plans.items()}, "routes": {}}
    evidence = {
        "sql_plain": "Rows from the filings' tables:\n" + (plain["evidence"] or "(the SQL query returned no rows)"),
        "sql_linked": "Rows from the filings' tables:\n" + (linked["evidence"] or "(the SQL query returned no rows)"),
        "table_chunks": render_chunks(tables, sources.labels),
    }
    evidence["table_hybrid"] = evidence["sql_linked"] + "\n\nTable chunks:\n" + evidence["table_chunks"]
    for name, text in evidence.items():
        cost = Cost()
        answer = sources.read(question, text, cost)
        before = {**{n: c for n, (_, c) in plans.items()}, "table_hybrid": plans["sql_linked"][1]}.get(name, Cost())
        out["routes"][name] = {"answer": answer, **asdict(before.plus(cost))}
    return out


def run_heldout(
    questions: Sequence[Question], choice: ModelChoice, *, client: Any, workers: int = 4
) -> list[dict[str, Any]]:
    """Run the three table sources on every held-out question."""
    complete = Completer(client, choice.model)
    chunks = load_chunks()
    info = filing_info()
    labels = {c.filing: filing_label(c.filing, info) for c in chunks}
    sources = Sources(load_index(chunks), sql_tables(), complete, labels)

    def one(question: Question) -> dict[str, Any]:
        return {
            "qid": question.qid,
            "qtype": question.qtype,
            "model": choice.model,
            **answer_heldout(sources, question.question),
        }

    return parallel(one, list(questions), workers)


def heldout_step(questions: Sequence[Question], chunks_id: str) -> Step:
    """Return the cached step for the held-out table sources."""
    return Step(
        RI_ID,
        "mixed-routing-heldout",
        PRODUCER,
        {
            "questions": question_digest(questions),
            "chunks": chunks_id,
            "reader": _digest(READER_PROMPT),
            "sql": _digest(SQL_PROMPT + SQL_LINKED_PROMPT + RETRY_EMPTY + RETRY_ERROR),
            "k": 5,
            "routes": list(HELDOUT_ROUTES),
        },
        role="generation",
    )


def heldout_judge_step(heldout_digest: str) -> Step:
    """Return the cached step for judging the held-out answers."""
    return Step(
        RI_ID,
        "mixed-routing-heldout-judgements",
        PRODUCER,
        {"prompt": "evaluation.answers.JUDGE_PROMPT"},
        role="judging",
        upstream={"heldout": heldout_digest},
    )


def heldout_summary(
    questions: Sequence[Question], runs: Sequence[Mapping[str, Any]], verdicts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Return, per table source, the questions answered, SQL rows, calls, and tokens."""
    correct = {(v["qid"], v["route"]): bool(v["correct"]) for v in verdicts}
    by_id = {r["qid"]: r for r in runs}
    out: dict[str, Any] = {}
    for name in HELDOUT_ROUTES:
        rows = [by_id[q.qid] for q in questions]
        entry: dict[str, Any] = {
            "label": HELDOUT_ROUTES[name],
            "questions": len(rows),
            "correct": sum(correct.get((r["qid"], name), False) for r in rows),
            "calls": round(sum(r["routes"][name]["model_calls"] for r in rows) / len(rows), 2),
            "prompt_tokens": round(sum(r["routes"][name]["prompt_tokens"] for r in rows) / len(rows)),
        }
        if name in rows[0]["sql"]:
            last = [(r["sql"][name][-1]["rows"] if r["sql"][name] else 0) for r in rows]
            entry["sql_no_rows"] = sum(1 for n in last if n == 0)
            entry["sql_errors"] = sum(1 for r in rows for a in r["sql"][name] if a["error"])
        out[name] = entry
    wins = {
        "linked_only": sum(
            correct.get((q.qid, "sql_linked"), False) and not correct.get((q.qid, "sql_plain"), False)
            for q in questions
        ),
        "plain_only": sum(
            correct.get((q.qid, "sql_plain"), False) and not correct.get((q.qid, "sql_linked"), False)
            for q in questions
        ),
        "any_source": sum(any(correct.get((q.qid, n), False) for n in SINGLE_SOURCES) for q in questions),
    }
    out["paired"] = wins
    return out


def reference_in_rows(tables: Any, attempts: Sequence[Mapping[str, Any]], reference: str) -> bool:
    """Return whether the last SQL attempt's rows hold the reference figure's digits.

    A digit match is a loose check: it can pass on an unrelated row, so read it
    as an upper bound on what the reader could have found.
    """
    digits = re.sub(r"\D", "", reference)
    if not attempts or attempts[-1]["error"] or not attempts[-1]["sql"] or not digits:
        return False
    try:
        _, rows, _ = tables.query(attempts[-1]["sql"], max_rows=SQL_ROWS)
    except Exception:  # noqa: BLE001 - sqlite3 raises several types
        return False
    return any(digits in re.sub(r"\D", "", " ".join(str(v) for v in row)) for row in rows)


def command_heldout(per_minute: int, workers: int) -> None:
    """Build fresh table questions, run three table sources, and write results."""
    from aieng.advanced_rag.runs import load_or_run  # noqa: PLC0415

    models = _onprem()
    client, _ = governed_client(per_minute)
    seen_questions, _ = load_questions()
    seen = {q.filing for q in seen_questions}
    path = heldout_path()
    if path.is_file():
        questions = load_heldout()
        saved = json.loads(path.read_text(encoding="utf-8"))
        funnel_rows, meta = saved["funnel"], saved["produced_by"]
    else:
        writer = Completer(client, "Qwen3_6-35B-A3B")
        checker = Completer(client, "Qwen3_8-27B")
        questions, funnel = build_questions(
            writer=writer,
            checker=checker,
            judge=writer,
            seed=HELDOUT_SEED,
            workers=workers,
            exclude=seen,
            kinds=("table",),
        )
        funnel_rows = funnel.to_rows()
        meta = produced_by(
            models,
            model="Qwen3_6-35B-A3B",
            question_writer="Qwen3_6-35B-A3B",
            question_checker="Qwen3_8-27B",
            rate_per_minute=per_minute,
            mode=f"explore: one run, sampling seed {HELDOUT_SEED}, filings outside the reference question set",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"produced_by": meta, "funnel": funnel_rows, "questions": [q.to_dict() for q in questions]}, indent=1
            ),
            encoding="utf-8",
        )
    overlap = {q.filing for q in questions} & seen
    if overlap or any(q.qtype != "table" for q in questions):
        raise RuntimeError("held-out questions must be table questions on filings outside the reference set")
    chunks_id = chunk_digest(load_chunks())
    run = load_or_run(
        heldout_step(questions, chunks_id),
        lambda c: run_heldout(questions, c, client=client, workers=workers),
        models=models,
        live=True,
    )
    judged = load_or_run(
        heldout_judge_step(run.digest),
        lambda c: judge_answers(run.value, [], questions, c, client=client, workers=workers),
        models=models,
        live=True,
    )
    summary = heldout_summary(questions, run.value, judged.value)
    gold = {q.qid: q for q in questions}
    tables = sql_tables()
    verdict = {(v["qid"], v["route"]): bool(v["correct"]) for v in judged.value}
    per_question = [
        {
            "qid": r["qid"],
            "filing": gold[r["qid"]].filing,
            "sources": list(gold[r["qid"]].sources),
            "correct": {n: verdict[(r["qid"], n)] for n in HELDOUT_ROUTES},
            "sql_rows": {n: (a[-1]["rows"] if a else 0) for n, a in r["sql"].items()},
            "reference_in_rows": {
                n: reference_in_rows(tables, a, gold[r["qid"]].reference) for n, a in r["sql"].items()
            },
        }
        for r in run.value
    ]
    for name in ("sql_plain", "sql_linked"):
        summary[name]["sql_reference_in_rows"] = sum(q["reference_in_rows"][name] for q in per_question)
    for name, entry in summary.items():
        print(name, entry)
    payload = {
        "produced_by": {**meta, "model": models.generation.model, "judge": models.judging.model},
        "filings": len({q.filing for q in questions}),
        "filings_overlapping_reference_set": 0,
        "funnel": funnel_rows,
        "routes": dict(HELDOUT_ROUTES),
        "results": summary,
        "per_question": per_question,
    }
    (OUTPUTS / "heldout_tables.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def heldout_results() -> dict[str, Any]:
    """Return the committed held-out table results."""
    return dict(json.loads((OUTPUTS / "heldout_tables.json").read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Command line


def _onprem() -> ModelSet:
    from aieng.advanced_rag.models import choose_models  # noqa: PLC0415

    return choose_models("onprem")


def produced_by(models: ModelSet, **extra: Any) -> dict[str, Any]:
    """Return the model record every committed result carries."""
    return {
        "model": models.generation.model,
        "judge": models.judging.model,
        "embeddings": f"{models.embeddings} (local)",
        "endpoint": models.generation.endpoint,
        "producer": "implementations/special_topics/mixed_source_routing/build_reference.py",
        "mode": "explore: one run, sampling seed 0",
        **extra,
    }


def command_questions(per_minute: int, workers: int) -> None:
    """Build the question set on the on-prem models and write its committed summary."""
    models = _onprem()
    client, _ = governed_client(per_minute)
    writer = Completer(client, "Qwen3_6-35B-A3B")
    checker = Completer(client, "Qwen3_8-27B")
    questions, funnel = build_questions(writer=writer, checker=checker, judge=writer, workers=workers)
    meta = {
        "produced_by": produced_by(
            models, model="Qwen3_6-35B-A3B", question_writer="Qwen3_6-35B-A3B", question_checker="Qwen3_8-27B"
        )
    }
    save_questions(questions, funnel, meta)
    counts = Counter(q.qtype for q in questions)
    print(dict(counts), questions_path())
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "question_set.json").write_text(
        json.dumps(
            {
                **meta,
                "counts": {t: counts.get(t, 0) for t in QTYPES},
                "filings": len({q.filing for q in questions}),
                "funnel": funnel.to_rows(),
                "questions": [
                    {"qid": q.qid, "qtype": q.qtype, "filing": q.filing, "sources": list(q.sources)} for q in questions
                ],
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )


def command_run(per_minute: int, workers: int) -> None:
    """Run and judge every arm on the on-prem model, then write the results."""
    models = _onprem()
    rows, outcome, origin = results(models, live=True, per_minute=per_minute, workers=workers)
    print(origin)
    for r in rows:
        types = "  ".join(f"{t} {v['accuracy']:.2f}" for t, v in r["by_type"].items())
        print(
            f"{r['arm']:4} acc {r['accuracy']:.3f} ({r['correct']}/{r['questions']}) calls {r['calls']:5.2f} "
            f"in {r['prompt_tokens']:>8,.0f} | {types}"
        )
    print(json.dumps(outcome, indent=1))
    questions, _ = load_questions()
    chunks = load_chunks()
    pipes = _cached(pipelines_step(questions, chunk_digest(chunks)), models)
    agent = _cached(agent_step(questions), models)
    judged = _cached(judge_step(pipes.digest, agent.digest), models)
    records = arm_records(questions, pipes.value, agent.value, judged.value, embedding_routes(questions))
    keep = ("qid", "qtype", "arm", "route", "correct", "model_calls", "prompt_tokens", "completion_tokens")
    sql = [
        {
            "qid": r["qid"],
            "attempts": len(r["sql"]),
            "rows": r["sql"][-1]["rows"] if r["sql"] else 0,
            "errors": sum(1 for a in r["sql"] if a["error"]),
        }
        for r in pipes.value
    ]
    payload = {
        "produced_by": produced_by(models, rate_per_minute=per_minute),
        "arms": dict(ARM_LABELS),
        "chunks": {"digest": chunk_digest(chunks), **Counter(c.kind for c in chunks)},
        "summaries": rows,
        "outcome": outcome,
        "sql_attempts": sql,
        "per_question": [{k: r[k] for k in keep} for r in records],
    }
    (OUTPUTS / "results.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def _cached(step: Step, models: ModelSet) -> Any:
    from aieng.advanced_rag.runs import load_or_run  # noqa: PLC0415

    def missing(choice: ModelChoice) -> Any:
        raise RuntimeError(f"{step.artifact_type} should be cached after the run")

    return load_or_run(step, missing, models=models, live=False, show=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command named on the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("command", choices=("prepare", "questions", "run", "followup", "heldout"))
    parser.add_argument("--per-minute", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.per_minute > 20:
        parser.error("the reference runs at 20 requests a minute or less")
    if args.command == "prepare":
        command_prepare()
    elif args.command == "questions":
        command_questions(args.per_minute, args.workers)
    elif args.command == "run":
        command_run(args.per_minute, args.workers)
    elif args.command == "heldout":
        command_heldout(args.per_minute, args.workers)
    else:
        command_followup(args.per_minute, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
