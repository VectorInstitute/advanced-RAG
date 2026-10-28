"""One retrieval interface, with the controls attached as capabilities.

A name, :meth:`Retriever.retrieve`, and :meth:`Retriever.aretrieve` are the
required surface. Everything else is an optional capability an implementation
offers by defining its methods, so a method never has to claim a control it
cannot honour: three of the four graph adapters have no ``index`` at all, and
none of them can filter by document.

Ask :func:`declares` which capabilities a retriever offers. The capability
protocols type an argument and no longer answer ``isinstance``, because
``index``, ``fetch``, ``find``, and ``read`` are names unrelated objects carry:
a list has ``index`` and a string has ``find``, and neither retrieves.

The capabilities are adjectives, from :class:`Indexable` to :class:`Greppable`.
Every ``Supports*`` protocol in ``typing.__all__`` wraps one dunder with a
no-argument signature, and ``index(chunks, batch_size)`` is a domain method, so
the prefix said the wrong thing and ``SupportsIndex`` shadowed the standard
library's own name.

Text reaches this boundary at index time rather than at call time. A retriever
composes a :class:`ChunkStore` built from the chunks ``index`` was already
handed, which is what lets ``retrieve`` return :class:`Evidence` without a
caller passing a chunk mapping back in. The store costs 1.14 MB on the committed
20-contract CUAD slice, measured over 954 chunks.

No ordering promise is made here, and that is deliberate. The negative control
scores every hit 0.0 and returns its pool in shuffle order, and two graph
adapters report no score at all, so a rule that results arrive in descending
score would either exclude them or be false. Passages arrive in the order the
method ranked them, and a conformance check asserts shape and provenance rather
than order.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..provenance import Provenance


if TYPE_CHECKING:
    # Reaching ``graphrag.evidence`` runs the ``graphrag`` package, which imports
    # the adapters, and the adapters import this module back. The two private
    # methods that build evidence import it inside their own bodies instead.
    from ..graphrag.evidence import Evidence


# The shared empty view every unfilled ``Hints`` starts from.
_NO_EXTRA: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True)
class Hints:
    """Typed, advisory steering for one request.

    Advisory is the contract. An implementation may ignore any field, and one
    that ignores every field must return exactly the ranking it would have
    returned for ``hints=None``. That is what keeps a hint from moving a
    measured number.

    Parameters
    ----------
    keywords : tuple of str, optional
        Query terms to prioritise. The LightRAG adapter maps these onto
        ``hl_keywords``.
    entities : tuple of str, optional
        Seed entities, which replace a graph method's own linking step.
    categories : tuple of str, optional
        Corpus-defined labels, such as a CUAD clause category.
    extra : mapping, optional
        Method-specific steering nothing shared reads. Copied behind a read-only
        view, so the record cannot be filled in after it is built.
    """

    keywords: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    extra: Mapping[str, Any] = _NO_EXTRA

    def __post_init__(self) -> None:
        """Copy ``extra`` behind a read-only view, so frozen means frozen.

        ``extra`` as a plain dict left one field of a frozen record writable:
        ``hints.extra["x"] = 1`` flipped :meth:`is_empty` after the fact, and
        that is the invariant the conformance kit asserts under "an empty
        ``Hints()`` changed the ranking". The copy also stops a caller's own dict
        moving a ranking after the request was built.
        """
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    def __hash__(self) -> int:
        """Return a hash over the four fields, so a request can key a cache.

        A mapping is unhashable, which left the generated ``__hash__`` raising
        ``TypeError`` on a frozen record. Hashing the items instead keeps the
        promise the dataclass makes, and raises only when a value in ``extra`` is
        itself unhashable.

        Returns
        -------
        int
            A hash over ``keywords``, ``entities``, ``categories``, and the
            items of ``extra``.
        """
        return hash(
            (
                self.keywords,
                self.entities,
                self.categories,
                tuple(sorted(self.extra.items())),
            )
        )

    def is_empty(self) -> bool:
        """Return whether there is nothing here for an implementation to honour.

        Returns
        -------
        bool
            ``True`` when every field is empty, which must rank exactly as
            ``hints=None`` does.
        """
        return not (self.keywords or self.entities or self.categories or self.extra)


@dataclass(frozen=True)
class Candidate:
    """One hit, listed cheaply so a caller can choose what to read in full.

    Parameters
    ----------
    chunk_id : str
        Identifier to pass to :meth:`Fetchable.fetch`.
    doc_id : str
        Source document.
    score : float or None, optional
        ``None`` for a method that reports no score.
    preview : str, optional
        The passage's leading characters, never the whole of it.
    retriever : str, optional
        Name of the method that produced this.
    """

    chunk_id: str
    doc_id: str
    score: float | None = None
    preview: str = ""
    retriever: str = ""


class ScopeUnsupported(ValueError):  # noqa: N818 - the documented name
    """Raised when a non-``None`` scope reaches a method that cannot filter.

    Ignoring a scope is not an option this interface offers, and neither is
    filtering the results after ranking: dropping passages from a ranking of
    ``k`` returns fewer than ``k`` and quietly stops ``k`` meaning what it means
    everywhere else. The measured reason to name the error instead is that an
    unscoped dense arm answers 75.8% of ContractNLI's unanswerable questions
    against 13.0% scoped, so a dropped scope reports a different method's
    number.
    """


@runtime_checkable
class Retriever(Protocol):
    """Anything that turns a question into ranked, cited, text-carrying passages.

    ``scope`` is the corpus's own filter, with the meaning it has on
    ``pipeline.spine.Retrieve``: on CUAD it is the contract identifier, and
    ``None`` means the whole corpus. A method that cannot filter raises
    :class:`ScopeUnsupported` rather than ignoring the argument.

    Passages arrive in the order the method ranked them. This interface makes no
    ordering promise beyond that, and in particular does not promise that scores
    descend.
    """

    name: str

    def retrieve(
        self,
        question: str,
        k: int = 10,
        *,
        scope: str | None = None,
        hints: Hints | None = None,
    ) -> list[Evidence]:
        """Return up to ``k`` passages, in the order this method ranked them.

        Parameters
        ----------
        question : str
            The question, unmodified.
        k : int, optional
            Maximum passages to return.
        scope : str or None, optional
            The corpus's own filter. ``None`` means the whole corpus.
        hints : Hints or None, optional
            Advisory steering, which an implementation may ignore.

        Returns
        -------
        list of Evidence
            Ranked passages, each carrying text and provenance.

        Raises
        ------
        ScopeUnsupported
            If ``scope`` is not ``None`` and this method cannot filter.
        """
        ...

    async def aretrieve(
        self,
        question: str,
        k: int = 10,
        *,
        scope: str | None = None,
        hints: Hints | None = None,
    ) -> list[Evidence]:
        """Return what :meth:`retrieve` returns, for an asynchronous caller.

        Parameters
        ----------
        question : str
            The question, unmodified.
        k : int, optional
            Maximum passages to return.
        scope : str or None, optional
            The corpus's own filter. ``None`` means the whole corpus.
        hints : Hints or None, optional
            Advisory steering, which an implementation may ignore.

        Returns
        -------
        list of Evidence
            Ranked passages, each carrying text and provenance.

        Raises
        ------
        ScopeUnsupported
            If ``scope`` is not ``None`` and this method cannot filter.
        """
        ...


class RetrieverBase:
    """The base a retrieval method subclasses, defining ``retrieve`` itself.

    Named a base rather than a mixin because it supplies the primary method. A
    scikit-learn mixin adds derived behaviour on top of a method the class
    already has: ``TransformerMixin`` contributes ``fit_transform`` while the
    subclass writes ``fit`` and ``transform``. This class is the other shape, and
    ``torch.nn.Module`` is the precedent for the abstract-by-exception mechanism.

    :meth:`__call__` is ``pipeline.spine.Retrieve``, so an implementation drops
    into ``AdvancedRagSpine`` and ``harvest_traces`` with no call-site edit. It
    forwards the ``k`` it is given to :meth:`retrieve`, and defaults to 10, the
    depth the retired ``harness_retrieve`` bound. The spine passes
    ``SpineConfig.top_k``, so a configured depth above that default reaches the
    method rather than being truncated away afterwards.

    :meth:`aretrieve` awaits the synchronous path without yielding to the loop.
    An asynchronous-native method overrides it rather than inheriting a thread
    pool it did not ask for, because a shared sentence-transformers encoder is
    not re-entrant.
    """

    name: str

    def retrieve(
        self,
        question: str,
        k: int = 10,
        *,
        scope: str | None = None,
        hints: Hints | None = None,
    ) -> list[Evidence]:
        """Return up to ``k`` ranked passages. Subclasses implement this.

        Parameters
        ----------
        question : str
            The question, unmodified.
        k : int, optional
            Maximum passages to return.
        scope : str or None, optional
            The corpus's own filter. ``None`` means the whole corpus.
        hints : Hints or None, optional
            Advisory steering, which an implementation may ignore.

        Returns
        -------
        list of Evidence
            Ranked passages, each carrying text and provenance.

        Raises
        ------
        NotImplementedError
            Always, on the mixin itself.
        """
        raise NotImplementedError(
            f"Retriever [{type(self).__name__}] is missing the required "
            '"retrieve" method'
        )

    def __call__(
        self, question: str, scope: str | None = None, k: int = 10
    ) -> Sequence[Evidence]:
        """Retrieve at the depth the caller asks for.

        Parameters
        ----------
        question : str
            The question, unmodified.
        scope : str or None, optional
            The corpus's own filter. ``None`` means the whole corpus.
        k : int, optional
            Maximum passages to return. ``AdvancedRagSpine`` passes
            ``SpineConfig.top_k`` here.

        Returns
        -------
        sequence of Evidence
            Ranked passages, as ``pipeline.spine.Retrieve`` expects them.
        """
        return self.retrieve(question, k, scope=scope)

    async def aretrieve(
        self,
        question: str,
        k: int = 10,
        *,
        scope: str | None = None,
        hints: Hints | None = None,
    ) -> list[Evidence]:
        """Await the synchronous path, which does not yield to the loop.

        Parameters
        ----------
        question : str
            The question, unmodified.
        k : int, optional
            Maximum passages to return.
        scope : str or None, optional
            The corpus's own filter. ``None`` means the whole corpus.
        hints : Hints or None, optional
            Advisory steering, which an implementation may ignore.

        Returns
        -------
        list of Evidence
            Exactly what :meth:`retrieve` returns for the same arguments.
        """
        return self.retrieve(question, k, scope=scope, hints=hints)


class Indexable(Protocol):
    """A method that builds its own store from chunks.

    For annotation. Ask :func:`declares` at runtime rather than ``isinstance``:
    one method name is not evidence of a capability.
    """

    def index(self, chunks: Sequence[Any], batch_size: int = 64) -> None:
        """Build whatever this method searches over.

        Parameters
        ----------
        chunks : sequence
            Objects carrying ``chunk_id``, ``doc_id``, ``text``, ``start``, and
            ``end``.
        batch_size : int, optional
            Items per batch, for a method that encodes in batches.
        """
        ...


class Findable(Protocol):
    """Find-then-fetch, step one: rank identifiers and show a preview.

    For annotation. Ask :func:`declares` at runtime rather than ``isinstance``:
    one method name is not evidence of a capability.
    """

    def find(
        self,
        question: str,
        k: int = 10,
        *,
        scope: str | None = None,
        preview: int = 200,
        hints: Hints | None = None,
    ) -> list[Candidate]:
        """Return up to ``k`` candidates, in rank order.

        Parameters
        ----------
        question : str
            The question, unmodified.
        k : int, optional
            Maximum candidates to return.
        scope : str or None, optional
            The corpus's own filter. ``None`` means the whole corpus.
        preview : int, optional
            Leading characters of the passage to include.
        hints : Hints or None, optional
            Advisory steering, which an implementation may ignore.

        Returns
        -------
        list of Candidate
            The ranking, each hit carrying a preview rather than the whole
            passage. Whether listing costs less than :meth:`Retriever.retrieve`
            is the implementation's business: the harness retrievers rank first
            and truncate afterwards, so for them it costs the same.
        """
        ...


class Fetchable(Protocol):
    """Find-then-fetch, step two: whole passages for chosen identifiers.

    For annotation. Ask :func:`declares` at runtime rather than ``isinstance``:
    one method name is not evidence of a capability.
    """

    def fetch(self, chunk_ids: Sequence[str]) -> list[Evidence]:
        """Return the named passages in full.

        Parameters
        ----------
        chunk_ids : sequence of str
            Identifiers, normally from :meth:`Findable.find`.

        Returns
        -------
        list of Evidence
            One record per identifier this method holds, in the order asked.
        """
        ...


class Expandable(Protocol):
    """Widen a hit to the unit a reader wants: a parent section, or neighbours.

    For annotation. Ask :func:`declares` at runtime rather than ``isinstance``:
    one method name is not evidence of a capability.
    """

    def expand(
        self, chunk_ids: Sequence[str], *, limit: int | None = None
    ) -> list[Evidence]:
        """Return the wider passage each identifier sits in.

        Parameters
        ----------
        chunk_ids : sequence of str
            Identifiers to widen.
        limit : int or None, optional
            Maximum passages to return. ``None`` returns one per identifier.

        Returns
        -------
        list of Evidence
            The widened passages, de-duplicated.
        """
        ...


class Readable(Protocol):
    """Read a range out of a document, the way an agent reads a file.

    Two methods, which is why ``read`` is the one capability spanning a pair.
    The bounds are keyword-only because the two disagree on purpose:
    ``read`` takes 0-based half-open character offsets and ``read_lines`` takes
    1-based inclusive line numbers, so a positional pair of numbers would mean
    two different ranges depending on which method it reached.

    For annotation. Ask :func:`declares` at runtime rather than ``isinstance``:
    one method name is not evidence of a capability.
    """

    def read(self, doc_id: str, *, start: int = 0, end: int | None = None) -> Evidence:
        """Return a character range of one document.

        Parameters
        ----------
        doc_id : str
            The document to read.
        start : int, optional
            First character offset, counted from the document start.
        end : int or None, optional
            One past the last character. ``None`` reads to the end.

        Returns
        -------
        Evidence
            The range, with provenance carrying the span read.
        """
        ...

    def read_lines(
        self, doc_id: str, *, first: int = 1, last: int | None = None
    ) -> Evidence:
        """Return a line range of one document.

        Parameters
        ----------
        doc_id : str
            The document to read.
        first : int, optional
            First line, counting from 1.
        last : int or None, optional
            Last line, inclusive. ``None`` reads to the end.

        Returns
        -------
        Evidence
            The range, with provenance carrying the character span read.
        """
        ...


class Greppable(Protocol):
    """Literal or regular-expression match over passage text.

    For annotation. Ask :func:`declares` at runtime rather than ``isinstance``:
    one method name is not evidence of a capability.
    """

    def grep(
        self,
        pattern: str,
        k: int = 10,
        *,
        scope: str | None = None,
        preview: int = 200,
    ) -> list[Candidate]:
        """Return up to ``k`` matches, in document order.

        Parameters
        ----------
        pattern : str
            A Python regular expression.
        k : int, optional
            Maximum matches to return.
        scope : str or None, optional
            The corpus's own filter. ``None`` searches the whole corpus.
        preview : int, optional
            Characters of matched text to include.

        Returns
        -------
        list of Candidate
            Matches in document order, which is not a ranking.
        """
        ...


#: Each optional capability and the methods an implementation defines to offer it.
#: ``read`` needs two, which is why six capabilities span seven methods.
CAPABILITY_METHODS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "index": ("index",),
        "find": ("find",),
        "fetch": ("fetch",),
        "expand": ("expand",),
        "read": ("read", "read_lines"),
        "grep": ("grep",),
    }
)

#: The optional capabilities, in the order a report lists them.
CAPABILITIES: tuple[str, ...] = tuple(CAPABILITY_METHODS)


def is_retriever(candidate: Any) -> bool:
    """Return whether this object is a retrieval method at all.

    The qualifier every capability check runs first. ``index``, ``fetch``,
    ``find``, and ``read`` are names unrelated objects carry, so one of them is
    not evidence of a capability until the object it sits on retrieves.

    Parameters
    ----------
    candidate : Any
        The object to look at.

    Returns
    -------
    bool
        ``True`` when the object names itself and defines ``retrieve``, which is
        the required surface of :class:`Retriever`.

    Examples
    --------
    >>> is_retriever([])
    False
    >>> is_retriever("abc")
    False
    """
    return isinstance(getattr(candidate, "name", None), str) and callable(
        getattr(candidate, "retrieve", None)
    )


def declares(candidate: Any, capability: str) -> bool:
    """Return whether a retriever offers one optional capability.

    This is the gate to write, rather than ``isinstance`` against a capability
    protocol or ``hasattr`` on a method name. A list has an ``index`` method and
    a string has ``find``; neither retrieves, so neither declares anything here.

    Parameters
    ----------
    candidate : Any
        The object to check, normally a retriever.
    capability : str
        One of :data:`CAPABILITIES`.

    Returns
    -------
    bool
        ``True`` when the object retrieves and defines every method the
        capability needs.

    Raises
    ------
    ValueError
        If ``capability`` is not one this interface defines.

    Examples
    --------
    >>> declares([], "index")
    False
    """
    try:
        methods = CAPABILITY_METHODS[capability]
    except KeyError:
        raise ValueError(
            f"unknown capability {capability!r}: this interface defines "
            f"{', '.join(CAPABILITIES)}"
        ) from None
    if not is_retriever(candidate):
        return False
    return all(callable(getattr(candidate, method, None)) for method in methods)


def capabilities_of(candidate: Any) -> tuple[str, ...]:
    """Return every capability a retriever declares, in :data:`CAPABILITIES` order.

    Parameters
    ----------
    candidate : Any
        The object to check, normally a retriever.

    Returns
    -------
    tuple of str
        The declared capabilities, empty for anything that does not retrieve.
    """
    return tuple(name for name in CAPABILITIES if declares(candidate, name))


class ChunkStore:
    """The corpus text a method already holds, addressable five ways.

    Satisfies :class:`Fetchable`, :class:`Readable`,
    :class:`Greppable`, and the offset half of :class:`Expandable`. A
    retriever composes one instead of growing five methods of its own, and it is
    built from the chunks ``index`` was already handed, so nothing re-reads the
    corpus.

    :meth:`document` rebuilds a document's text from its chunks' half-open
    spans, which is exact for the overlapping fixed-width windows
    ``retrieval.corpus.chunk_contract`` produces: across the committed
    20-contract slice, no document has a coverage hole and no adjacent pair
    disagrees on its overlap. That is what gives a character range, a line
    range, and a grep a meaning over an in-memory chunk index. The chunker drops
    a window whose text is only whitespace, so a gap is filled with spaces,
    which keeps every retained character at its own offset.

    Parameters
    ----------
    chunks : sequence, optional
        Objects carrying ``chunk_id``, ``text``, ``start``, ``end``, and one of
        ``doc_id`` or ``contract_id``.
    retriever : str, optional
        Label recorded on every record this store builds.
    """

    def __init__(self, chunks: Sequence[Any] = (), retriever: str = "") -> None:
        self.retriever = retriever
        self._chunks: dict[str, Any] = {}
        self._order: dict[str, list[str]] = {}
        self.add(chunks)

    def __len__(self) -> int:
        """Return how many distinct chunks this store holds."""
        return len(self._chunks)

    def __contains__(self, chunk_id: object) -> bool:
        """Return whether this store holds that chunk identifier."""
        return chunk_id in self._chunks

    def add(self, chunks: Sequence[Any]) -> None:
        """Retain more chunks, keeping each document in offset order.

        Re-indexing the same chunk replaces it, so indexing twice over one
        corpus does not double the store.

        Parameters
        ----------
        chunks : sequence
            The chunks to retain.
        """
        touched: set[str] = set()
        for chunk in chunks:
            provenance = Provenance.from_chunk(chunk)
            key = provenance.chunk_id or provenance.doc_id
            if key not in self._chunks:
                self._order.setdefault(provenance.doc_id, []).append(key)
            self._chunks[key] = chunk
            touched.add(provenance.doc_id)
        for doc_id in touched:
            self._order[doc_id].sort(key=lambda key: (self._start(key), key))

    def document_ids(self) -> list[str]:
        """Return the documents this store holds, in first-seen order.

        Returns
        -------
        list of str
            One identifier per document.
        """
        return list(self._order)

    def fetch(self, chunk_ids: Sequence[str]) -> list[Evidence]:
        """Return the named passages in full, skipping identifiers it lacks.

        Parameters
        ----------
        chunk_ids : sequence of str
            Identifiers to read.

        Returns
        -------
        list of Evidence
            One record per identifier this store holds, in the order asked.
        """
        return self.rank([(chunk_id, None) for chunk_id in chunk_ids])

    def rank(self, hits: Sequence[tuple[str, float | None]]) -> list[Evidence]:
        """Turn ranked ``(chunk_id, score)`` pairs into evidence, in that order.

        This is the one call a retriever needs to put text and provenance on its
        own ranking.

        Parameters
        ----------
        hits : sequence of tuple
            ``(chunk_id, score)`` in rank order. A score of ``None`` records a
            method that reports none.

        Returns
        -------
        list of Evidence
            The same order, with any identifier this store lacks skipped.
        """
        built = [self._evidence(chunk_id, score) for chunk_id, score in hits]
        return [item for item in built if item is not None]

    def expand(
        self, chunk_ids: Sequence[str], *, limit: int | None = None
    ) -> list[Evidence]:
        """Return each hit with its offset neighbours, de-duplicated.

        A flat chunk index has no hierarchy, so widening a hit means the windows
        either side of it. ``pipeline.hierarchical.small_to_big`` is the other
        meaning of the word, and it needs a hierarchy this store does not have.

        Parameters
        ----------
        chunk_ids : sequence of str
            Identifiers to widen.
        limit : int or None, optional
            Maximum passages to return. ``None`` returns every neighbour found.

        Returns
        -------
        list of Evidence
            The hit and its neighbours, in offset order per hit.
        """
        widened: dict[str, None] = {}
        for chunk_id in chunk_ids:
            for neighbour in self._neighbour_ids(chunk_id, 1, 1):
                widened.setdefault(neighbour, None)
        ids = list(widened)
        return self.fetch(ids[:limit] if limit else ids)

    def neighbours(
        self, chunk_id: str, *, before: int = 1, after: int = 1
    ) -> list[Evidence]:
        """Return one hit with the windows either side of it, in offset order.

        Parameters
        ----------
        chunk_id : str
            The hit to look around.
        before : int, optional
            Windows to include ahead of the hit.
        after : int, optional
            Windows to include after the hit.

        Returns
        -------
        list of Evidence
            The hit and its neighbours, or an empty list for an unknown hit.
        """
        return self.fetch(self._neighbour_ids(chunk_id, before, after))

    def document(self, doc_id: str) -> str:
        """Rebuild one document's text from its chunks' spans.

        Parameters
        ----------
        doc_id : str
            The document to rebuild.

        Returns
        -------
        str
            The text, with a dropped whitespace-only window filled with spaces
            so every retained character keeps its own offset. Empty for a
            document this store does not hold.
        """
        parts: list[str] = []
        cursor = 0
        for chunk_id in self._order.get(doc_id, ()):
            chunk = self._chunks[chunk_id]
            if chunk.start > cursor:
                parts.append(" " * (chunk.start - cursor))
                cursor = chunk.start
            if chunk.end > cursor:
                parts.append(chunk.text[cursor - chunk.start :])
                cursor = chunk.end
        return "".join(parts)

    def read(self, doc_id: str, *, start: int = 0, end: int | None = None) -> Evidence:
        """Return a character range of one document.

        Parameters
        ----------
        doc_id : str
            The document to read.
        start : int, optional
            First character offset.
        end : int or None, optional
            One past the last character. ``None`` reads to the end.

        Returns
        -------
        Evidence
            The range, with provenance carrying the span actually read.
        """
        text = self.document(doc_id)
        stop = len(text) if end is None else min(end, len(text))
        return self._range(doc_id, text[start:stop], start, stop)

    def read_lines(
        self, doc_id: str, *, first: int = 1, last: int | None = None
    ) -> Evidence:
        """Return a line range of one document, counting lines from 1.

        Lines are derived rather than stored, because every offset in this
        package is a character offset.

        Parameters
        ----------
        doc_id : str
            The document to read.
        first : int, optional
            First line, counting from 1.
        last : int or None, optional
            Last line, inclusive. ``None`` reads to the end.

        Returns
        -------
        Evidence
            The range, with provenance carrying the character span read and the
            line numbers in ``extra``.
        """
        text = self.document(doc_id)
        lines = text.splitlines(keepends=True)
        stop = len(lines) if last is None else min(last, len(lines))
        start_offset = sum(len(line) for line in lines[: max(first - 1, 0)])
        body = "".join(lines[max(first - 1, 0) : stop])
        return self._range(
            doc_id,
            body,
            start_offset,
            start_offset + len(body),
            extra={"lines": (first, stop)},
        )

    def grep(
        self,
        pattern: str,
        k: int = 10,
        *,
        scope: str | None = None,
        preview: int = 200,
    ) -> list[Candidate]:
        """Return up to ``k`` pattern matches, in document order.

        Matching runs over each document's rebuilt text rather than over the
        windows, so an overlap does not report one match twice.

        Parameters
        ----------
        pattern : str
            A Python regular expression.
        k : int, optional
            Maximum matches to return.
        scope : str or None, optional
            One document, or ``None`` for every document this store holds.
        preview : int, optional
            Characters of matched text to include.

        Returns
        -------
        list of Candidate
            Matches in document order, which is not a ranking.
        """
        compiled = re.compile(pattern)
        found: list[Candidate] = []
        documents = [scope] if scope is not None else self.document_ids()
        for doc_id in documents:
            text = self.document(doc_id)
            for match in compiled.finditer(text):
                found.append(
                    Candidate(
                        chunk_id=self._covering(doc_id, match.start()),
                        doc_id=doc_id,
                        preview=text[match.start() : match.start() + preview],
                        retriever=self.retriever,
                    )
                )
                if len(found) >= k:
                    return found
        return found

    def _start(self, chunk_id: str) -> int:
        return int(getattr(self._chunks[chunk_id], "start", 0) or 0)

    def _neighbour_ids(self, chunk_id: str, before: int, after: int) -> list[str]:
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            return []
        order = self._order[Provenance.from_chunk(chunk).doc_id]
        at = order.index(chunk_id)
        return order[max(at - before, 0) : at + after + 1]

    def _covering(self, doc_id: str, offset: int) -> str:
        """Return the chunk holding a character offset, or the document id."""
        for chunk_id in self._order.get(doc_id, ()):
            chunk = self._chunks[chunk_id]
            if chunk.start <= offset < chunk.end:
                return chunk_id
        return doc_id

    def _evidence(self, chunk_id: str, score: float | None) -> Evidence | None:
        # Deferred for the reason the TYPE_CHECKING block above gives: at module
        # scope this would pull in the graph adapters, which import this module.
        from ..graphrag.evidence import Evidence

        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            return None
        return Evidence(
            text=chunk.text,
            provenance=Provenance.from_chunk(chunk),
            score=score,
            retriever=self.retriever,
        )

    def _range(
        self,
        doc_id: str,
        text: str,
        start: int,
        end: int,
        extra: dict[str, Any] | None = None,
    ) -> Evidence:
        from ..graphrag.evidence import Evidence

        return Evidence(
            text=text,
            provenance=Provenance(
                doc_id=doc_id, start=start, end=end, extra=extra or {}
            ),
            retriever=self.retriever,
        )


__all__ = [
    "CAPABILITIES",
    "CAPABILITY_METHODS",
    "Candidate",
    "ChunkStore",
    "Expandable",
    "Fetchable",
    "Findable",
    "Greppable",
    "Hints",
    "Indexable",
    "Readable",
    "Retriever",
    "RetrieverBase",
    "ScopeUnsupported",
    "capabilities_of",
    "declares",
    "is_retriever",
]

#: Deprecated alias for :class:`Retriever`, kept because the reference
#: implementations import it under the old name. Deprecated in 0.1.1; use
#: ``Retriever``.
Retrieval = Retriever

#: Deprecated alias for :class:`RetrieverBase`, kept because the reference
#: implementations subclass it under the old name. Deprecated in 0.1.1; use
#: ``RetrieverBase``.
RetrievalMixin = RetrieverBase

#: Deprecated aliases for the capability protocols, kept because the reference
#: implementations import them under the ``Supports*`` names. Deprecated in
#: 0.1.1; use the adjectives. They answer no ``isinstance``: ask
#: :func:`declares`.
SupportsIndex = Indexable
SupportsFind = Findable
SupportsFetch = Fetchable
SupportsExpand = Expandable
SupportsRead = Readable
SupportsGrep = Greppable
