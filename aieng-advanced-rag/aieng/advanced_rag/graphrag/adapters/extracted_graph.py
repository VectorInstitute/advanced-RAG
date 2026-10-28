"""Retrieval over a graph built by a direct extraction pass.

This is the method the GraphRAG comparison recommends teaching on: one
extraction pass produces entities and relations, each carrying the chunks it
came from, and retrieval walks that structure instead of embedding the query.

It is here rather than in the experiment because it is the arm every other
method is compared against, and because it needs nothing installed. It also
makes the shape of a graph retriever legible: link, expand, collect, rank. A
participant can read all four steps.

Deliberately no dense fallback. A graph arm that quietly answers with a chunk
index when the graph finds nothing reports the chunk index's score as its own.
Coverage is reported instead, so "the graph had nothing to say" is visible.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Iterable, Sequence

from ...provenance import Provenance
from ...retrieval.interface import Hints, RetrieverBase, ScopeUnsupported
from ..evidence import Evidence


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")

# Function words. An entity whose whole name is made of these ("a", "with",
# "how to") is never linked, because on a 63,436-entity ServerFault graph the
# extractor produced entities such as "a", and any-word linking reached them
# from almost every question.
STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "done",
        "down",
        "during",
        "each",
        "either",
        "else",
        "etc",
        "even",
        "ever",
        "every",
        "few",
        "for",
        "from",
        "further",
        "get",
        "gets",
        "got",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "one",
        "only",
        "or",
        "other",
        "our",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "upon",
        "us",
        "very",
        "via",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "whether",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "yes",
        "yet",
        "you",
        "your",
    ]
)


def _token_list(text: str) -> list[str]:
    """Return the lowercased tokens of ``text`` in order.

    Trailing ``.``, ``/``, ``-`` and ``_`` are stripped, so "file." at the end of
    a sentence is the token "file".
    """
    found = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0).lower().rstrip("._/-")
        if token:
            found.append(token)
    return found


def _tokens(text: str) -> set[str]:
    return set(_token_list(text))


class ExtractedGraphRetriever(RetrieverBase):
    """Entity-linked retrieval over an extracted graph.

    Linking matches an entity when its whole name, or one of its aliases, occurs
    in the question as a run of whole tokens. Names made only of stopwords are
    never linked. Each linked entity weighs the inverse document frequency of the
    documents that mention it, so a rare entity such as ``pkcs12`` outweighs
    ``linux``. Expansion then walks one hop, and each neighbour receives
    :attr:`NEIGHBOUR_SHARE` of its seed's weight divided by the square root of
    the seed's degree.

    Parameters
    ----------
    entities : iterable
        Objects with ``name``, ``chunks`` and ``documents``. An optional
        ``aliases`` attribute lists other surface forms to link by.
    relations : iterable
        Objects with ``source``, ``target``, ``type`` and ``chunks``.
    chunk_texts : dict
        Chunk id to text, so retrieved chunks can be returned as evidence.
    chunk_documents : dict, optional
        Chunk id to document id. Falls back to the prefix of the chunk id.
    expand_hops : int, optional
        How far to walk from a linked entity. One hop is the setting that makes
        this different from a keyword index; zero reduces it to entity lookup.
    name : str, optional
        Label used in reports.
    """

    #: Share of a seed's weight a one-hop neighbour receives, before the division
    #: by the square root of the seed's degree. A hub such as ``ssh`` has hundreds
    #: of neighbours, and without the division they outrank the post the question
    #: names. 0.25 ranked best of the one-hop shares tried on the ServerFault graph.
    NEIGHBOUR_SHARE = 0.25

    def __init__(
        self,
        entities: Iterable[Any],
        relations: Iterable[Any],
        chunk_texts: dict[str, str],
        chunk_documents: dict[str, str] | None = None,
        expand_hops: int = 1,
        name: str = "extracted graph",
    ) -> None:
        self.name = name
        self.expand_hops = expand_hops
        self._chunk_texts = chunk_texts
        self._chunk_documents = chunk_documents or {}
        self._entity_chunks: dict[str, list[str]] = {}
        self._entity_documents: dict[str, set[str]] = {}
        self._neighbours: dict[str, set[str]] = defaultdict(set)
        # A name is indexed as its token sequence. Spellings that tokenize alike
        # ("~/.ssh/config" and ".ssh/config") share a key and link together.
        self._by_phrase: dict[tuple[str, ...], set[str]] = defaultdict(set)
        self._phrase_lengths: dict[str, set[int]] = defaultdict(set)

        for entity in entities:
            key = entity.name.lower()
            self._entity_chunks[key] = list(entity.chunks)
            documents = set(getattr(entity, "documents", ()) or ())
            if not documents:
                documents = {self._document_of(c) for c in entity.chunks}
            self._entity_documents.setdefault(key, set()).update(documents)
            for surface in (entity.name, *(getattr(entity, "aliases", ()) or ())):
                phrase = tuple(_token_list(surface))
                if not phrase or all(t in STOPWORDS for t in phrase):
                    continue
                self._by_phrase[phrase].add(key)
                self._phrase_lengths[phrase[0]].add(len(phrase))

        total = max(len({d for ds in self._entity_documents.values() for d in ds}), 1)
        self._weight = {
            key: math.log(1 + total / max(len(documents), 1))
            for key, documents in self._entity_documents.items()
        }

        for relation in relations:
            source, target = relation.source.lower(), relation.target.lower()
            if source in self._entity_chunks and target in self._entity_chunks:
                self._neighbours[source].add(target)
                self._neighbours[target].add(source)

    def _document_of(self, chunk_id: str) -> str:
        return self._chunk_documents.get(chunk_id, chunk_id.split("::", maxsplit=1)[0])

    def link(self, query: str) -> list[str]:
        """Return entities whose whole name is in the query, most specific first.

        Specificity is the entity's weight (rarer first), then the number of
        tokens its name spans, so ``nginx worker_connections`` precedes
        ``nginx`` when both are equally rare.
        """
        tokens = _token_list(query)
        span: dict[str, int] = {}
        for start, token in enumerate(tokens):
            for length in self._phrase_lengths.get(token, ()):
                phrase = tuple(tokens[start : start + length])
                for entity in self._by_phrase.get(phrase, ()):
                    span[entity] = max(span.get(entity, 0), length)
        return sorted(span, key=lambda e: (-self._weight[e], -span[e], e))

    def expand(self, seeds: Sequence[str]) -> dict[str, float]:
        """Walk out from the linked entities, weighting seeds above neighbours."""
        weights: dict[str, float] = {}
        for seed in seeds:
            weights[seed] = weights.get(seed, 0.0) + self._weight.get(seed, 1.0)
        if self.expand_hops >= 1:
            for seed in seeds:
                neighbours = self._neighbours.get(seed, ())
                share = (
                    self.NEIGHBOUR_SHARE
                    * self._weight.get(seed, 1.0)
                    / math.sqrt(max(len(neighbours), 1))
                )
                for neighbour in neighbours:  # one hop
                    weights[neighbour] = weights.get(neighbour, 0.0) + share
        return weights

    def retrieve(
        self,
        question: str,
        k: int = 10,
        *,
        scope: str | None = None,
        hints: Hints | None = None,
    ) -> list[Evidence]:
        """Return up to ``k`` chunks reached through the graph.

        An empty result means the question linked to nothing, which is a real
        outcome of this method and is not papered over.

        ``hints.entities`` is honoured, because it names the seeds :meth:`link`
        exists to find, so a caller that already knows them replaces the linking
        step rather than steering it. Every other field is ignored.

        Parameters
        ----------
        question : str
            The question, tokenised for entity linking.
        k : int, optional
            Maximum passages to return.
        scope : str or None, optional
            Only ``None`` is accepted.
        hints : Hints or None, optional
            ``hints.entities`` replaces the linking step when it is set.

        Returns
        -------
        list of Evidence
            Ranked passages carrying text, provenance, and the entities they
            were reached through.

        Raises
        ------
        ScopeUnsupported
            If ``scope`` is not ``None``. The walk runs over the whole graph,
            and filtering a finished ranking would return fewer than ``k``.
        """
        if scope is not None:
            raise ScopeUnsupported(
                f"{self.name} walks the whole graph and cannot filter to {scope!r}",
            )
        seeded = [name.lower() for name in hints.entities] if hints else []
        seeds = [name for name in seeded if name in self._entity_chunks] or self.link(
            question
        )
        if not seeds:
            return []
        weights = self.expand(seeds)

        scores: dict[str, float] = defaultdict(float)
        for entity, weight in weights.items():
            for chunk_id in self._entity_chunks.get(entity, ()):  # provenance
                if chunk_id in self._chunk_texts:
                    scores[chunk_id] += weight

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [
            Evidence(
                text=self._chunk_texts[chunk_id],
                provenance=Provenance(
                    doc_id=self._document_of(chunk_id),
                    chunk_id=chunk_id,
                ),
                score=float(score),
                retriever=self.name,
                extra={"linked_entities": seeds[:5]},
            )
            for chunk_id, score in ranked
        ]
