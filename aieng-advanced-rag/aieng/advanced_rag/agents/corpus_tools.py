"""A tool server over a corpus: ``search``, ``grep``, and ``open``, every call logged.

An agent that retrieves with tools needs three things from the corpus: a ranked
search, a literal or regular-expression match, and a way to read a document by
line. This module builds those over one :class:`Corpus`, behind backends that can
be swapped without touching the tools:

* :class:`RegexGrep` matches Python regular expressions over the documents held
  in memory. It needs no binary and runs in every workspace, so the MuSiQue
  notebooks use it.
* :class:`RipgrepGrep` runs ``rg`` over files on disk, which is what a coding
  agent does over a repository. The starter notebook and code corpora use it,
  and :func:`materialize` writes an in-memory corpus out as files for it.
* :class:`DenseSearch` ranks documents by cosine similarity of local embeddings.
  Any callable from a query and ``k`` to ``(doc_id, score)`` pairs also works.

An arm is an :class:`ArmConfig`: which tools the model sees, their limits, the
model-call budget, and the guards. Switching arms changes configuration only.

:meth:`CorpusToolServer.session` returns a :class:`ToolSession` for one question.
Its ``tools`` are plain functions, so they go to the ADK
:class:`~aieng.advanced_rag.agents.loop.AgentLoop` unchanged, as
:func:`~aieng.advanced_rag.agents.session_loop.run_session` passes them. Every call is
recorded in the session's :class:`ToolTrace` with its arguments, a digest of its
result, the documents the result named, and the documents whose full text it
showed. That separation is what the "seen in a preview but never opened"
diagnostic reads: a gold document the agent saw in a ``grep`` preview and never
opened is a navigation failure, not a search failure.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol


try:  # The third-party engine can stop a runaway pattern; ``re`` cannot.
    import regex as _regex_engine
except ImportError:  # pragma: no cover - depends on the environment
    _regex_engine = None


#: Every tool name an arm can offer.
TOOL_NAMES: tuple[str, ...] = ("search", "grep", "open")

#: Characters per token used for the output cap. A rough rule for English prose,
#: chosen so the cap never needs a tokenizer.
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# The corpus


@dataclass(frozen=True)
class CorpusDocument:
    """One document an agent can grep and open.

    Parameters
    ----------
    doc_id : str
        Stable identifier the tools return and ``open`` accepts.
    title : str
        A short label shown beside each hit.
    text : str
        The full text. Lines are what ``open`` addresses.
    """

    doc_id: str
    title: str
    text: str


class Corpus:
    """Documents in a fixed order, addressable by identifier.

    Parameters
    ----------
    documents : iterable of CorpusDocument
        The documents. Identifiers must be unique.
    name : str, optional
        A label for reports and cache keys.

    Raises
    ------
    ValueError
        If two documents share an identifier.
    """

    def __init__(
        self, documents: Iterable[CorpusDocument], name: str = "corpus"
    ) -> None:
        self.name = name
        self.documents: list[CorpusDocument] = list(documents)
        self._by_id = {doc.doc_id: doc for doc in self.documents}
        if len(self._by_id) != len(self.documents):
            raise ValueError("two documents share an identifier")

    def __len__(self) -> int:
        """Return the number of documents."""
        return len(self.documents)

    def __contains__(self, doc_id: object) -> bool:
        """Return whether the corpus holds that identifier."""
        return doc_id in self._by_id

    def get(self, doc_id: str) -> CorpusDocument | None:
        """Return one document, or ``None`` for an unknown identifier."""
        return self._by_id.get(doc_id)

    @property
    def characters(self) -> int:
        """Return the total characters of text, titles excluded."""
        return sum(len(doc.text) for doc in self.documents)

    @classmethod
    def from_passages(
        cls, passages: Iterable[Any], name: str = "corpus", *, title_line: bool = True
    ) -> Corpus:
        """Build a corpus from records with ``passage_id``, ``title``, and ``text``.

        Parameters
        ----------
        passages : iterable
            Such as :class:`aieng.advanced_rag.corpora.musique.Passage`.
        name : str, optional
            The corpus label.
        title_line : bool, optional
            Put the title on line 1 of the text, so ``grep`` matches titles and
            ``open`` shows them. MuSiQue's bridge entities are often article
            titles, so the MuSiQue arms keep this on.

        Returns
        -------
        Corpus
            One document per passage, in the order given.
        """
        return cls(
            (
                CorpusDocument(
                    p.passage_id,
                    p.title,
                    f"{p.title}\n{p.text}" if title_line else p.text,
                )
                for p in passages
            ),
            name,
        )

    @classmethod
    def from_chunk_store(cls, store: Any, name: str = "corpus") -> Corpus:
        """Build a corpus from a chunk store, one document per ``doc_id``.

        The store is a :class:`~aieng.advanced_rag.retrieval.interface.ChunkStore`.

        Each document is rebuilt from its chunks' spans, so a grep over the
        corpus reports a match in an overlap once.

        Parameters
        ----------
        store : ChunkStore
            The chunk index a retriever already holds.
        name : str, optional
            The corpus label.

        Returns
        -------
        Corpus
            One document per ``doc_id`` in the store.
        """
        return cls(
            (CorpusDocument(d, d, store.document(d)) for d in store.document_ids()),
            name,
        )

    @classmethod
    def from_folder(
        cls,
        folder: Path | str,
        *,
        patterns: Sequence[str] = ("**/*.md", "**/*.txt", "**/*.py", "**/*.rst"),
        max_bytes: int = 2_000_000,
        name: str | None = None,
    ) -> Corpus:
        """Read the text files under a folder, one document per file.

        Parameters
        ----------
        folder : Path or str
            The root. Document identifiers are paths relative to it.
        patterns : sequence of str, optional
            Glob patterns to include.
        max_bytes : int, optional
            Files larger than this are skipped.
        name : str or None, optional
            The corpus label. The folder name when omitted.

        Returns
        -------
        Corpus
            Files in sorted path order. Undecodable bytes are replaced.
        """
        root = Path(folder)
        found: dict[str, Path] = {}
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_file() and path.stat().st_size <= max_bytes:
                    found[path.relative_to(root).as_posix()] = path
        return cls(
            (
                CorpusDocument(
                    rel, rel, found[rel].read_text(encoding="utf-8", errors="replace")
                )
                for rel in sorted(found)
            ),
            name or root.name,
        )


def materialize(corpus: Corpus, folder: Path | str) -> dict[str, str]:
    """Write a corpus as one file per document, for :class:`RipgrepGrep`.

    Identifiers that are not safe file names are encoded. The mapping returned
    turns a relative path back into its identifier.

    Parameters
    ----------
    corpus : Corpus
        The documents.
    folder : Path or str
        Where to write. Created when missing.

    Returns
    -------
    dict
        Relative path to document identifier.
    """
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for doc in corpus.documents:
        safe = re.sub(r"[^A-Za-z0-9._/-]", "_", doc.doc_id).strip("/") or "doc"
        if ".." in safe.split("/"):
            safe = safe.replace("..", "__")
        rel = safe if safe not in mapping else f"{safe}.{len(mapping)}"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(doc.text, encoding="utf-8")
        mapping[rel] = doc.doc_id
    return mapping


# ---------------------------------------------------------------------------
# Grep backends


@dataclass(frozen=True)
class GrepMatch:
    """One document that matched, with its first matching line.

    Parameters
    ----------
    doc_id : str
        The document.
    title : str
        Its title.
    line : int
        Line number of the first match, counting from 1.
    preview : str
        Text around the first match.
    count : int
        Matches in this document.
    """

    doc_id: str
    title: str
    line: int
    preview: str
    count: int


@dataclass(frozen=True)
class GrepResult:
    """What a grep backend found.

    Parameters
    ----------
    matches : tuple of GrepMatch
        Up to the requested number of documents, in corpus order.
    total_hits : int
        Documents that matched, counted up to the backend's scan cap.
    total_matches : int
        Matches over those documents.
    capped : bool
        Whether the scan stopped at the cap, so the totals are lower bounds.
    note : str
        Anything the model should know, such as a pattern read literally.
    """

    matches: tuple[GrepMatch, ...]
    total_hits: int
    total_matches: int
    capped: bool = False
    note: str = ""


class GrepBackend(Protocol):
    """Anything that greps a corpus."""

    def grep(
        self,
        pattern: str,
        *,
        max_results: int,
        ignore_case: bool = True,
        preview_chars: int = 200,
        within: Iterable[str] | None = None,
    ) -> GrepResult:
        """Return the documents matching ``pattern``."""
        ...


def _preview(text: str, start: int, end: int, width: int) -> str:
    """Return about ``width`` characters centred on a match, on one line."""
    half = max((width - (end - start)) // 2, 20)
    left = max(start - half, 0)
    right = min(end + half, len(text))
    body = text[left:right].replace("\n", " ")
    return ("..." if left else "") + body + ("..." if right < len(text) else "")


class RegexGrep:
    """Regular-expression grep over the documents held in memory.

    The documents are joined once into one string with a separator line, so a
    grep is one scan of the corpus rather than one per document, which keeps a
    grep over 139,416 passages under a second.

    Parameters
    ----------
    corpus : Corpus
        The documents.
    scan_cap : int, optional
        Stop counting after this many matching documents. A pattern such as
        ``the`` would otherwise count every document.
    timeout : float, optional
        Seconds a pattern may run, when the ``regex`` engine is installed. A
        catastrophic pattern from a model then returns a note instead of
        hanging the run.
    """

    SEPARATOR = "\n\x1e\n"

    def __init__(
        self, corpus: Corpus, *, scan_cap: int = 5000, timeout: float = 5.0
    ) -> None:
        self.corpus = corpus
        self.scan_cap = scan_cap
        self.timeout = timeout
        parts: list[str] = []
        self._starts: list[int] = []
        cursor = 0
        for doc in corpus.documents:
            self._starts.append(cursor)
            parts.append(doc.text)
            cursor += len(doc.text) + len(self.SEPARATOR)
        self._blob = self.SEPARATOR.join(parts)

    def _compile(self, pattern: str, ignore_case: bool) -> tuple[Any, str]:
        engine: Any = _regex_engine or re
        flags = engine.IGNORECASE if ignore_case else 0
        try:
            return engine.compile(pattern, flags), ""
        except Exception:  # noqa: BLE001 - both engines raise their own error type
            note = "the pattern is not a valid regular expression, so it was matched literally"
            return engine.compile(re.escape(pattern), flags), note

    def _finditer(self, compiled: Any, text: str) -> Iterable[Any]:
        found: Iterable[Any]
        if _regex_engine is not None:
            found = compiled.finditer(text, timeout=self.timeout)
        else:
            found = compiled.finditer(text)
        return found

    def grep(
        self,
        pattern: str,
        *,
        max_results: int = 20,
        ignore_case: bool = True,
        preview_chars: int = 200,
        within: Iterable[str] | None = None,
    ) -> GrepResult:
        """Return the documents matching ``pattern``, in corpus order.

        Parameters
        ----------
        pattern : str
            A Python regular expression. An invalid one is matched literally.
        max_results : int, optional
            Documents to return with previews.
        ignore_case : bool, optional
            Match without regard to case.
        preview_chars : int, optional
            Characters of context per document.
        within : iterable of str or None, optional
            Restrict the match to these documents.

        Returns
        -------
        GrepResult
            The first ``max_results`` matching documents and the totals.
        """
        if not pattern.strip():
            return GrepResult((), 0, 0, note="empty pattern")
        compiled, note = self._compile(pattern, ignore_case)
        if within is not None:
            return self._grep_within(
                compiled, set(within), max_results, preview_chars, note
            )
        counts: dict[int, int] = {}
        first: dict[int, tuple[int, int]] = {}
        capped = False
        try:
            for match in self._finditer(compiled, self._blob):
                if match.end() == match.start():
                    continue
                index = bisect.bisect_right(self._starts, match.start()) - 1
                if index not in counts:
                    if len(counts) >= self.scan_cap:
                        capped = True
                        break
                    first[index] = (match.start(), match.end())
                counts[index] = counts.get(index, 0) + 1
        except TimeoutError:
            capped = True
            note = (
                note + "; " if note else ""
            ) + "the pattern ran too long and was stopped"
        matches = []
        for index in sorted(counts)[:max_results]:
            doc = self.corpus.documents[index]
            start, end = (offset - self._starts[index] for offset in first[index])
            matches.append(
                GrepMatch(
                    doc.doc_id,
                    doc.title,
                    doc.text.count("\n", 0, start) + 1,
                    _preview(doc.text, start, end, preview_chars),
                    counts[index],
                )
            )
        return GrepResult(
            tuple(matches), len(counts), sum(counts.values()), capped, note
        )

    def _grep_within(
        self,
        compiled: Any,
        allowed: set[str],
        max_results: int,
        preview_chars: int,
        note: str,
    ) -> GrepResult:
        matches: list[GrepMatch] = []
        hits = total = 0
        for doc in self.corpus.documents:
            if doc.doc_id not in allowed:
                continue
            found = [
                m for m in self._finditer(compiled, doc.text) if m.end() > m.start()
            ]
            if not found:
                continue
            hits += 1
            total += len(found)
            if len(matches) < max_results:
                start, end = found[0].start(), found[0].end()
                matches.append(
                    GrepMatch(
                        doc.doc_id,
                        doc.title,
                        doc.text.count("\n", 0, start) + 1,
                        _preview(doc.text, start, end, preview_chars),
                        len(found),
                    )
                )
        return GrepResult(tuple(matches), hits, total, False, note)


class RipgrepUnavailableError(RuntimeError):
    """``rg`` is not on the path."""


class RipgrepGrep:
    """Grep with ripgrep over files on disk.

    Ripgrep reads the files each call, so a change on disk shows up in the next
    grep with no index to rebuild, the property coding agents rely on. Its
    regular-expression syntax is Rust's, close to Python's for the patterns a
    model writes, without look-around or back-references.

    Parameters
    ----------
    folder : Path or str
        The root to search.
    doc_ids : mapping or None, optional
        Relative path to document identifier, as :func:`materialize` returns.
        Paths are the identifiers when omitted.
    corpus : Corpus or None, optional
        Supplies titles for the hits. The path is the title when omitted.
    rg : str, optional
        The ripgrep executable.
    scan_cap : int, optional
        Stop reading results after this many matching files.
    timeout : float, optional
        Seconds before the ``rg`` process is stopped.

    Raises
    ------
    RipgrepUnavailableError
        If ``rg`` is not on the path. Install ripgrep, or use
        :class:`RegexGrep` over :meth:`Corpus.from_folder`.
    """

    def __init__(
        self,
        folder: Path | str,
        *,
        doc_ids: Mapping[str, str] | None = None,
        corpus: Corpus | None = None,
        rg: str = "rg",
        scan_cap: int = 5000,
        timeout: float = 30.0,
    ) -> None:
        executable = shutil.which(rg)
        if executable is None:
            raise RipgrepUnavailableError(
                f"{rg!r} is not on the path; install ripgrep, or grep with RegexGrep "
                "over Corpus.from_folder"
            )
        self.rg = executable
        self.folder = Path(folder)
        self.doc_ids = dict(doc_ids or {})
        self._paths = {doc_id: rel for rel, doc_id in self.doc_ids.items()}
        self.corpus = corpus
        self.scan_cap = scan_cap
        self.timeout = timeout

    def _doc_id(self, path: str) -> str:
        rel = Path(path).resolve().relative_to(self.folder.resolve()).as_posix()
        return self.doc_ids.get(rel, rel)

    def _title(self, doc_id: str) -> str:
        found = self.corpus.get(doc_id) if self.corpus is not None else None
        return found.title if found is not None else doc_id

    def grep(
        self,
        pattern: str,
        *,
        max_results: int = 20,
        ignore_case: bool = True,
        preview_chars: int = 200,
        within: Iterable[str] | None = None,
    ) -> GrepResult:
        """Return the files matching ``pattern``, sorted by path.

        Parameters are those of :meth:`RegexGrep.grep`. An invalid pattern is
        retried as a fixed string.
        """
        if not pattern.strip():
            return GrepResult((), 0, 0, note="empty pattern")
        targets: list[str] = [str(self.folder)]
        if within is not None:
            targets = [str(self.folder / self._paths.get(d, d)) for d in within]
            if not targets:
                return GrepResult((), 0, 0)
        note = ""
        events = self._run(pattern, ignore_case, targets, fixed=False)
        if events is None:
            note = "the pattern is not a valid regular expression, so it was matched literally"
            events = self._run(pattern, ignore_case, targets, fixed=True) or []
        counts: dict[str, int] = {}
        first: dict[str, tuple[int, str, int, int]] = {}
        capped = False
        for event in events:
            if event.get("type") != "match":
                continue
            data = event["data"]
            doc_id = self._doc_id(data["path"]["text"])
            submatches = data.get("submatches") or [{"start": 0, "end": 0}]
            if doc_id not in counts:
                if len(counts) >= self.scan_cap:
                    capped = True
                    break
                line_text = data["lines"].get("text", "")
                first[doc_id] = (
                    int(data["line_number"]),
                    line_text,
                    int(submatches[0]["start"]),
                    int(submatches[0]["end"]),
                )
            counts[doc_id] = counts.get(doc_id, 0) + len(submatches)
        ordered = sorted(counts, key=lambda d: self._paths.get(d, d))
        matches = tuple(
            GrepMatch(
                d,
                self._title(d),
                first[d][0],
                _preview(
                    first[d][1].rstrip("\n"), first[d][2], first[d][3], preview_chars
                ),
                counts[d],
            )
            for d in ordered[:max_results]
        )
        return GrepResult(matches, len(counts), sum(counts.values()), capped, note)

    def _run(
        self, pattern: str, ignore_case: bool, targets: Sequence[str], *, fixed: bool
    ) -> list[dict[str, Any]] | None:
        command = [self.rg, "--json", "--no-config", "--max-columns", "2000"]
        if ignore_case:
            command.append("--ignore-case")
        if fixed:
            command.append("--fixed-strings")
        command += ["-e", pattern, "--", *targets]
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=self.timeout, check=False
        )
        # ripgrep exits 1 for no match and 2 for an error such as a bad pattern.
        if completed.returncode == 2 and not completed.stdout:
            return None
        return [
            json.loads(line) for line in completed.stdout.splitlines() if line.strip()
        ]


# ---------------------------------------------------------------------------
# Search backends


SearchFunction = Callable[[str, int], Sequence[tuple[str, float]]]


class DenseSearch:
    """Cosine-similarity search over local embeddings of every document.

    Parameters
    ----------
    corpus : Corpus
        The documents.
    embedder : object
        Has ``encode(texts)`` and ``encode_queries(texts)``, as
        :class:`~aieng.advanced_rag.retrieval.embedders.SentenceTransformerEmbedder`
        does. Local ``BAAI/bge-small-en-v1.5`` in this project.
    matrix : array or None, optional
        Precomputed document embeddings, one row per document in corpus order.
    cache : Path or None, optional
        An ``.npy`` file read when present and written after encoding.
    """

    def __init__(
        self,
        corpus: Corpus,
        embedder: Any,
        *,
        matrix: Any = None,
        cache: Path | None = None,
    ) -> None:
        import numpy as np  # noqa: PLC0415 - numpy is optional for grep-only arms

        self._np = np
        self.corpus = corpus
        self.embedder = embedder
        if matrix is None and cache is not None and Path(cache).is_file():
            matrix = np.load(cache)
        if matrix is None:
            matrix = np.asarray(
                embedder.encode([doc.text for doc in corpus.documents]),
                dtype=np.float32,
            )
            if cache is not None:
                Path(cache).parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, matrix)
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape[0] != len(corpus):
            raise ValueError(
                f"{matrix.shape[0]} embedding rows for {len(corpus)} documents"
            )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._matrix = matrix / np.clip(norms, 1e-9, None)

    def __call__(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return the ``k`` documents most similar to ``query``, best first."""
        np = self._np
        vector = np.asarray(self.embedder.encode_queries([query])[0], dtype=np.float32)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-9)
        scores = self._matrix @ vector
        order = np.lexsort((np.arange(len(scores)), -scores))[:k]
        return [(self.corpus.documents[i].doc_id, float(scores[i])) for i in order]


# ---------------------------------------------------------------------------
# Arms


@dataclass(frozen=True)
class ArmConfig:
    """One arm: the tools a model sees, their limits, the budget, and the guards.

    Parameters
    ----------
    name : str
        The arm's label, such as ``C5``.
    tools : tuple of str
        A subset of :data:`TOOL_NAMES`.
    description : str, optional
        One line for the arm table.
    search_k : int, optional
        Documents per ``search``. Fixed, not a tool argument, so ``C0`` matches
        a top-5 retriever.
    search_preview_tokens : int, optional
        Tokens of each search hit shown. A hit shorter than this is shown whole
        and counts as read.
    grep_max_results : int, optional
        Documents per ``grep`` shown with previews.
    grep_preview_chars : int, optional
        Characters of context per grep hit.
    grep_scope : str, optional
        ``corpus`` greps everything. ``searched`` greps only documents an
        earlier ``search`` in the same question returned, so search bounds the
        workspace and grep reads inside it.
    max_model_calls : int, optional
        The call budget. The last call is made with tools disabled, so a run
        that reaches the cap still answers, and the run is marked exhausted.
    output_cap_tokens : int, optional
        Longest tool result, in tokens. ``open`` stops at a line boundary under
        the cap and says which line to continue from.
    narrow_above : int or None, optional
        When a grep matches more documents than this, return the count and ask
        for a narrower pattern instead of the previews. ``None`` turns it off.
    already_read_notice : bool, optional
        Answer a second ``open`` of the same range with a short notice instead
        of the text again.
    read_before_answer : bool, optional
        If the model answers without opening anything, ask it once to open its
        sources first. Applies only to arms that offer ``open`` or an extra
        tool whose :class:`ToolKind` reads.
    ignore_case : bool, optional
        Default case handling for ``grep``.
    extra_tools : tuple of str, optional
        Tools beyond :data:`TOOL_NAMES`, by the names a :class:`ToolKind`
        registered on the server, such as the tools a data profile generates.
        The server checks them when it opens a session.
    """

    name: str
    tools: tuple[str, ...]
    description: str = ""
    search_k: int = 5
    search_preview_tokens: int = 512
    grep_max_results: int = 20
    grep_preview_chars: int = 200
    grep_scope: Literal["corpus", "searched"] = "corpus"
    max_model_calls: int = 30
    output_cap_tokens: int = 4000
    narrow_above: int | None = 200
    already_read_notice: bool = True
    read_before_answer: bool = True
    ignore_case: bool = True
    extra_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject an unknown tool or a budget below two calls."""
        unknown = set(self.tools) - set(TOOL_NAMES)
        if unknown:
            raise ValueError(f"unknown tools {sorted(unknown)}; expected {TOOL_NAMES}")
        if self.max_model_calls < 2:
            raise ValueError("max_model_calls must allow a tool call and an answer")

    def with_(self, **changes: Any) -> ArmConfig:
        """Return a copy with some fields changed."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record, for cache keys and reports."""
        record = asdict(self)
        record["tools"] = list(self.tools)
        # Left out when empty, so an arm defined before extra tools existed
        # keeps the record, and so the cache key, it had.
        if self.extra_tools:
            record["extra_tools"] = list(self.extra_tools)
        else:
            record.pop("extra_tools")
        return record


#: The arms built for RI #5. ``C0`` is the closest to classic retrieval, ``C5``
#: has no vector index in the loop, and ``C7`` bounds grep with search.
STANDARD_ARMS: Mapping[str, ArmConfig] = {
    "C0": ArmConfig(
        "C0",
        ("search",),
        "search with k fixed at 5 and 512-token previews",
        read_before_answer=False,
    ),
    "C5": ArmConfig("C5", ("grep", "open"), "grep and open only"),
    "C7": ArmConfig(
        "C7",
        ("search", "grep", "open"),
        "search at k 20, then grep inside what search returned, then open",
        search_k=20,
        search_preview_tokens=40,
        grep_scope="searched",
    ),
}


# ---------------------------------------------------------------------------
# The trace


def digest(value: Any) -> str:
    """Return a short, stable digest of a JSON value."""
    blob = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class ToolEvent:
    """One tool call and what it returned.

    Parameters
    ----------
    index : int
        Position in the question's calls, from 0.
    name : str
        The tool.
    arguments : dict
        The arguments as the model sent them.
    result_digest : str
        Digest of the result the model received.
    returned_ids : tuple of str
        Documents the result named, in order.
    read_ids : tuple of str
        Documents whose full text the result showed.
    total_hits : int or None
        For ``grep``, documents that matched.
    result_chars : int
        Length of the result as the model received it.
    seconds : float
        Time the tool took.
    error : str
        Why the call failed, or ``""``.
    """

    index: int
    name: str
    arguments: Mapping[str, Any]
    result_digest: str
    returned_ids: tuple[str, ...] = ()
    read_ids: tuple[str, ...] = ()
    total_hits: int | None = None
    result_chars: int = 0
    seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        record = asdict(self)
        record["arguments"] = dict(self.arguments)
        record["returned_ids"] = list(self.returned_ids)
        record["read_ids"] = list(self.read_ids)
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> ToolEvent:
        """Read a record written by :meth:`to_dict`."""
        fields = dict(record)
        fields["returned_ids"] = tuple(fields.get("returned_ids") or ())
        fields["read_ids"] = tuple(fields.get("read_ids") or ())
        return cls(**fields)


@dataclass
class ToolTrace:
    """Every tool call one question made, in order."""

    events: list[ToolEvent] = field(default_factory=list)

    def add(self, event: ToolEvent) -> None:
        """Record one call."""
        self.events.append(event)

    @property
    def seen_ids(self) -> list[str]:
        """Return every document any result named, first-seen order."""
        return list(dict.fromkeys(i for e in self.events for i in e.returned_ids))

    @property
    def read_ids(self) -> list[str]:
        """Return every document whose full text a result showed, first-read order."""
        return list(dict.fromkeys(i for e in self.events for i in e.read_ids))

    def seen_not_read(self, gold: Iterable[str]) -> list[str]:
        """Return the gold documents the agent saw in a preview and never read."""
        read = set(self.read_ids)
        seen = set(self.seen_ids)
        return [g for g in gold if g in seen and g not in read]

    def counts(self) -> dict[str, int]:
        """Return calls per tool name."""
        found: dict[str, int] = {}
        for event in self.events:
            found[event.name] = found.get(event.name, 0) + 1
        return found

    def to_list(self) -> list[dict[str, Any]]:
        """Return a JSON-serialisable record of every call."""
        return [event.to_dict() for event in self.events]


# ---------------------------------------------------------------------------
# The server


@dataclass(frozen=True)
class ToolKind:
    """A tool the server can offer beyond ``search``, ``grep``, and ``open``.

    Parameters
    ----------
    name : str
        The tool's name, which an arm lists in ``extra_tools``. It must equal
        the ``__name__`` of the function ``build`` returns.
    build : callable
        Called with a :class:`ToolSession`; returns the plain tool function for
        that session. The function records its own calls with
        :meth:`ToolSession.record`, as the built-in tools do.
    reads : bool, optional
        Whether the tool shows a document's full text, so the
        read-before-answer guard counts it as reading.
    """

    name: str
    build: Callable[[ToolSession], Callable[..., dict[str, Any]]]
    reads: bool = False


class CorpusToolServer:
    """The tools of every arm over one corpus.

    Parameters
    ----------
    corpus : Corpus
        The documents ``open`` reads.
    grep : GrepBackend or None, optional
        The grep backend. A :class:`RegexGrep` over ``corpus`` when omitted.
    search : callable or None, optional
        ``search(query, k)`` returning ``(doc_id, score)`` pairs, best first,
        such as :class:`DenseSearch`. Required by arms that offer ``search``.
    tool_kinds : iterable of ToolKind, optional
        Further tools an arm can name in ``extra_tools``. More can be added
        with :meth:`register`.

    Examples
    --------
    >>> server = CorpusToolServer(
    ...     corpus, search=DenseSearch(corpus, embedder)
    ... )  # doctest: +SKIP
    >>> session = server.session(STANDARD_ARMS["C5"])  # doctest: +SKIP
    >>> session.call("grep", {"pattern": "Lothair II"})  # doctest: +SKIP
    """

    def __init__(
        self,
        corpus: Corpus,
        *,
        grep: GrepBackend | None = None,
        search: SearchFunction | None = None,
        tool_kinds: Iterable[ToolKind] = (),
    ) -> None:
        self.corpus = corpus
        self.grep_backend: GrepBackend = grep if grep is not None else RegexGrep(corpus)
        self.search_backend = search
        self.tool_kinds: dict[str, ToolKind] = {}
        for kind in tool_kinds:
            self.register(kind)

    def register(self, kind: ToolKind) -> None:
        """Add a tool kind that arms can name in ``extra_tools``.

        Raises
        ------
        ValueError
            If the name is a built-in tool's or is already registered.
        """
        if kind.name in TOOL_NAMES or kind.name in self.tool_kinds:
            raise ValueError(f"a tool named {kind.name!r} is already on the server")
        self.tool_kinds[kind.name] = kind

    def session(self, arm: ArmConfig) -> ToolSession:
        """Return the tools for one question under one arm, with a fresh trace.

        Raises
        ------
        ValueError
            If the arm offers ``search`` and the server has no search backend,
            or names an extra tool the server has not registered.
        """
        if "search" in arm.tools and self.search_backend is None:
            raise ValueError(
                f"arm {arm.name} offers search and the server has no search backend"
            )
        missing = [name for name in arm.extra_tools if name not in self.tool_kinds]
        if missing:
            raise ValueError(
                f"arm {arm.name} names tools the server does not have: {missing}"
            )
        return ToolSession(self, arm)


class ToolSession:
    """The tools one question sees, and the trace of what they returned.

    Build it with :meth:`CorpusToolServer.session`. ``tools`` holds plain
    functions named ``search``, ``grep``, and ``open`` as the arm allows.
    """

    def __init__(self, server: CorpusToolServer, arm: ArmConfig) -> None:
        self.server = server
        self.arm = arm
        self.trace = ToolTrace()
        self._opened: set[tuple[str, int, int]] = set()
        self._searched: list[str] = []
        builders = {
            "search": self._search_tool,
            "grep": self._grep_tool,
            "open": self._open_tool,
        }
        self.tools: list[Callable[..., dict[str, Any]]] = [
            builders[n]() for n in arm.tools
        ]
        kinds = server.tool_kinds
        self.tools += [kinds[n].build(self) for n in arm.extra_tools]
        self._by_name = {tool.__name__: tool for tool in self.tools}

    @property
    def opened_any(self) -> bool:
        """Return whether the agent has opened any document."""
        return bool(self._opened)

    @property
    def offers_reading(self) -> bool:
        """Return whether any tool in this session shows a document in full."""
        kinds = self.server.tool_kinds
        return "open" in self.arm.tools or any(
            kinds[n].reads for n in self.arm.extra_tools
        )

    def mark_read(self, doc_id: str, start_line: int = 1, end_line: int = 0) -> None:
        """Note that an extra tool showed a document, for the read guards."""
        self._opened.add((doc_id, start_line, end_line))

    def record(
        self,
        name: str,
        arguments: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        returned: Sequence[str] = (),
        read: Sequence[str] = (),
        total_hits: int | None = None,
        started: float | None = None,
        error: str = "",
    ) -> None:
        """Add one call to the trace, for a tool from a :class:`ToolKind`.

        Parameters are those of a :class:`ToolEvent`; ``started`` is a
        :func:`time.perf_counter` reading taken when the call began.
        """
        self._record(
            name,
            arguments,
            result,
            returned=returned,
            read=read,
            total_hits=total_hits,
            started=started,
            error=error,
        )

    def call(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Run one tool by name, as a model's tool call does.

        An unknown tool or bad arguments come back as an ``error`` field the
        model can read, and are recorded in the trace.

        Parameters
        ----------
        name : str
            The tool.
        arguments : mapping
            Its arguments.

        Returns
        -------
        dict
            What the model receives.
        """
        tool = self._by_name.get(name)
        if tool is None:
            result = {
                "error": f"no tool named {name!r}; available: {sorted(self._by_name)}"
            }
            self._record(name, arguments, result, error=result["error"])
            return result
        try:
            return tool(**dict(arguments))
        except TypeError as error:
            result = {"error": f"bad arguments for {name}: {error}"}
            self._record(name, arguments, result, error=result["error"])
            return result

    # The tools record themselves, so a direct call from a framework that never
    # goes through :meth:`call` still lands in the trace.

    def _record(
        self,
        name: str,
        arguments: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        returned: Sequence[str] = (),
        read: Sequence[str] = (),
        total_hits: int | None = None,
        started: float | None = None,
        error: str = "",
    ) -> None:
        self.trace.add(
            ToolEvent(
                index=len(self.trace.events),
                name=name,
                arguments=dict(arguments),
                result_digest=digest(result),
                returned_ids=tuple(returned),
                read_ids=tuple(read),
                total_hits=total_hits,
                result_chars=len(json.dumps(result, default=str)),
                seconds=round(time.perf_counter() - started, 4) if started else 0.0,
                error=error,
            )
        )

    def _search_tool(self) -> Callable[..., dict[str, Any]]:
        session = self

        def search(query: str) -> dict[str, Any]:
            """Search the corpus by meaning and return the closest documents.

            Parameters
            ----------
            query : str
                What to look for, in natural language.

            Returns
            -------
            dict
                ``results``: each with ``doc_id``, ``title``, ``score``, and
                ``text`` (the start of the document).
            """
            started = time.perf_counter()
            arm = session.arm
            cap = arm.search_preview_tokens * CHARS_PER_TOKEN
            backend = session.server.search_backend
            # session() refuses this; a hand-built session might not.
            if backend is None:
                raise ValueError("this server has no search backend")
            ranked = backend(query, arm.search_k)
            results: list[dict[str, Any]] = []
            read: list[str] = []
            for doc_id, score in ranked:
                doc = session.server.corpus.get(doc_id)
                if doc is None:
                    continue
                whole = len(doc.text) <= cap
                if whole:
                    read.append(doc_id)
                results.append(
                    {
                        "doc_id": doc_id,
                        "title": doc.title,
                        "score": round(score, 4),
                        "text": doc.text if whole else doc.text[:cap] + "...",
                    }
                )
            returned = [str(r["doc_id"]) for r in results]
            session._searched.extend(returned)
            result = {"results": results}
            session._record(
                "search",
                {"query": query},
                result,
                returned=returned,
                read=read,
                started=started,
            )
            return result

        return search

    def _grep_tool(self) -> Callable[..., dict[str, Any]]:
        session = self
        scoped = self.arm.grep_scope == "searched"

        def grep(pattern: str, ignore_case: bool = True) -> dict[str, Any]:
            """Find the documents whose text matches a regular expression.

            Matching is over each document's title and text, and it is exact:
            the pattern must appear in the document as written. Grep for a
            name, title, or identifier on its own ("Ralph Rapson"), not a
            sentence about it ("Ralph Rapson died"). A regular expression such
            as ``Rapson.*(born|died)`` matches within one line. Results are in
            corpus order, not ranked.

            Parameters
            ----------
            pattern : str
                A regular expression. Escape ``.``, ``(``, and ``?`` to match
                them literally.
            ignore_case : bool
                Match without regard to case. True by default.

            Returns
            -------
            dict
                ``total_hits`` (documents matched) and ``matches``, each with
                ``doc_id``, ``title``, ``line``, ``count``, and a ``preview``
                around the first match. Open a document to read it in full.
            """
            started = time.perf_counter()
            arm = session.arm
            arguments = {"pattern": pattern, "ignore_case": ignore_case}
            within = list(dict.fromkeys(session._searched)) if scoped else None
            if scoped and not within:
                result: dict[str, Any] = {
                    "total_hits": 0,
                    "matches": [],
                    "note": "grep looks only inside documents a search returned; search first",
                }
                session._record(
                    "grep", arguments, result, total_hits=0, started=started
                )
                return result
            found = session.server.grep_backend.grep(
                pattern,
                max_results=arm.grep_max_results,
                ignore_case=ignore_case,
                preview_chars=arm.grep_preview_chars,
                within=within,
            )
            notes = [found.note] if found.note else []
            if arm.narrow_above is not None and found.total_hits > arm.narrow_above:
                result = {
                    "total_hits": found.total_hits,
                    "matches": [],
                    "note": "; ".join(
                        [
                            *notes,
                            f"{'at least ' if found.capped else ''}{found.total_hits} documents "
                            "match; narrow the pattern with a more specific name or phrase",
                        ]
                    ),
                }
                session._record(
                    "grep",
                    arguments,
                    result,
                    total_hits=found.total_hits,
                    started=started,
                )
                return result
            if found.total_hits > len(found.matches):
                notes.append(
                    f"showing the first {len(found.matches)} of {found.total_hits} documents; "
                    "narrow the pattern to see others"
                )
            if found.total_hits == 0:
                notes.append("no document matches")
            result = {
                "total_hits": found.total_hits,
                "matches": [
                    {
                        "doc_id": m.doc_id,
                        "title": m.title,
                        "line": m.line,
                        "count": m.count,
                        "preview": m.preview,
                    }
                    for m in found.matches
                ],
            }
            if notes:
                result["note"] = "; ".join(notes)
            session._record(
                "grep",
                arguments,
                result,
                returned=[m.doc_id for m in found.matches],
                total_hits=found.total_hits,
                started=started,
            )
            return result

        return grep

    def _open_tool(self) -> Callable[..., dict[str, Any]]:
        session = self

        def open(doc_id: str, start_line: int = 1, end_line: int = 0) -> dict[str, Any]:  # noqa: A001 - the tool's name
            """Read a document, or a range of its lines, in full.

            Parameters
            ----------
            doc_id : str
                A ``doc_id`` from an earlier result.
            start_line : int
                First line to read, counting from 1.
            end_line : int
                Last line to read, inclusive. 0 reads to the end.

            Returns
            -------
            dict
                ``doc_id``, ``title``, ``total_lines``, and ``text`` with each
                line prefixed by its number. A long document stops early with a
                ``note`` naming the line to continue from.
            """
            started = time.perf_counter()
            arguments = {
                "doc_id": doc_id,
                "start_line": start_line,
                "end_line": end_line,
            }
            doc = session.server.corpus.get(doc_id)
            if doc is None:
                result: dict[str, Any] = {
                    "error": f"no document {doc_id!r}; use a doc_id from a result"
                }
                session._record(
                    "open", arguments, result, error=result["error"], started=started
                )
                return result
            lines = doc.text.splitlines() or [""]
            first = max(int(start_line or 1), 1)
            last = (
                len(lines)
                if not end_line or end_line <= 0
                else min(int(end_line), len(lines))
            )
            key = (doc_id, first, last)
            if session.arm.already_read_notice and key in session._opened:
                result = {
                    "doc_id": doc_id,
                    "note": "you already opened these lines earlier in this question; "
                    "read them above",
                }
                session._record(
                    "open", arguments, result, returned=[doc_id], started=started
                )
                return result
            cap = session.arm.output_cap_tokens * CHARS_PER_TOKEN
            shown: list[str] = []
            used = 0
            stop = last
            for number in range(first, last + 1):
                line = f"{number}: {lines[number - 1]}"
                if shown and used + len(line) > cap:
                    stop = number - 1
                    break
                shown.append(line[:cap])
                used += len(line) + 1
            result = {
                "doc_id": doc_id,
                "title": doc.title,
                "total_lines": len(lines),
                "text": "\n".join(shown),
            }
            if stop < last:
                result["note"] = (
                    f"output cap reached; continue with open(doc_id={doc_id!r}, "
                    f"start_line={stop + 1})"
                )
            session._opened.add(key)
            # Any range the agent opened counts as read: a line range is how a
            # long document is read, and the scorer asks whether the agent
            # looked at the passage, not whether it read every line.
            session._record(
                "open",
                arguments,
                result,
                returned=[doc_id],
                read=[doc_id],
                started=started,
            )
            return result

        return open


__all__ = [
    "CHARS_PER_TOKEN",
    "STANDARD_ARMS",
    "TOOL_NAMES",
    "ArmConfig",
    "Corpus",
    "CorpusDocument",
    "CorpusToolServer",
    "DenseSearch",
    "GrepBackend",
    "GrepMatch",
    "GrepResult",
    "RegexGrep",
    "RipgrepGrep",
    "RipgrepUnavailableError",
    "SearchFunction",
    "ToolEvent",
    "ToolKind",
    "ToolSession",
    "ToolTrace",
    "digest",
    "materialize",
]
