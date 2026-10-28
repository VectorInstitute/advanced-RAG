"""Parent-child chunking: retrieve on small units, read from big ones.

The chunking spike measured flat strategies, where one window is both the unit
that gets ranked and the unit that reaches the reader. Those two jobs want
different sizes. A short window embeds to a sharp vector and ranks well; a short
window handed to a generator drops the sentence before it that said which party
the clause binds.

Small-to-big keeps both. Children are the indexed units and stay small. Parents
are the read units and stay whole. :func:`small_to_big` runs between retrieval
and generation: it swaps each retrieved child's text for its parent's and keeps
the child's identifier, so the retrieval metrics still score the unit the
retriever actually ranked.

Contextual chunking is the other half. :func:`contextualize` prefixes each child
with where it sits before the child is indexed, which recovers the document and
section a bare window loses. The prefix is index-time only: character offsets
keep pointing into the source document, so span-derived relevance judgments stay
valid.

One integration note. Relevance judgments are derived from the chunks you index.
Swapping flat chunks for hierarchical children changes the identifiers, so the
judgment set has to be rebuilt from the children before any number from the two
setups is compared.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from ..graphrag.evidence import Evidence
from ..provenance import Provenance


if TYPE_CHECKING:  # pragma: no cover - the retrieval half is being consolidated
    from aieng.advanced_rag.retrieval.corpus import Chunk


@dataclass(frozen=True)
class Node:
    """One unit of a chunk hierarchy, parent or child.

    Parameters
    ----------
    node_id : str
        Stable identifier. Child identifiers are what a retriever indexes and
        what the retrieval metrics score, so they must not move between runs.
    doc_id : str
        Identifier of the source document.
    text : str
        The unit's text.
    start : int
        Character offset into the source document.
    end : int
        Exclusive end offset.
    parent_id : str or None, optional
        The containing parent, or ``None`` for a parent.
    heading : str or None, optional
        Nearest preceding heading, when one was detected.
    """

    node_id: str
    doc_id: str
    text: str
    start: int
    end: int
    parent_id: str | None = None
    heading: str | None = None

    @property
    def is_parent(self) -> bool:
        """Whether this node is a read unit rather than an indexed one."""
        return self.parent_id is None

    def provenance(self) -> Provenance:
        """Return the citable location of this node."""
        return Provenance(
            doc_id=self.doc_id,
            chunk_id=self.node_id,
            section=self.heading,
            start=self.start,
            end=self.end,
        )


# Contract headings come in three shapes: a named division, a decimal outline
# number, or a short line in capitals. Body text in capitals is common in
# contracts, so the length bound and the spacing filter in detect_headings
# together stop the third arm from matching whole paragraphs.
_HEADING_RE = re.compile(
    r"^[ \t]*("
    r"(?:ARTICLE|SECTION|EXHIBIT|SCHEDULE|APPENDIX|ANNEX)\s+[0-9IVXLC]+[A-Za-z]?[^\n]{0,80}"
    r"|\d+(?:\.\d+)*\.?[ \t]+[A-Z][^\n]{0,80}"
    r"|[A-Z][A-Z0-9 ,'&/()\-.]{3,60}"
    r")[ \t]*$",
    re.MULTILINE,
)


def detect_headings(text: str, min_spacing: int = 200) -> list[tuple[int, str]]:
    """Find heading offsets in a document.

    Parameters
    ----------
    text : str
        The document text.
    min_spacing : int, optional
        Minimum characters between two accepted headings. A match closer than
        this to the previous one is dropped, which is what keeps a run of
        capitalised body lines from becoming a hundred one-line sections.

    Returns
    -------
    list of (int, str)
        Character offset and heading text, in document order.
    """
    headings: list[tuple[int, str]] = []
    for match in _HEADING_RE.finditer(text):
        offset = match.start()
        if headings and offset - headings[-1][0] < min_spacing:
            continue
        headings.append((offset, match.group(1).strip()))
    return headings


# A window shorter than this fraction of the target is folded into the one
# before it. Without the fold, a section a few characters longer than the window
# produces a chunk holding one line, which is noise in the index and useless as
# context.
_MIN_TAIL_RATIO = 0.25


def _windows(start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Return character windows over ``[start, end)``, folding a short tail back."""
    stride = size - overlap
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + size, end)
        spans.append((cursor, stop))
        if stop >= end:
            break
        cursor += stride
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < size * _MIN_TAIL_RATIO:
        spans[-2] = (spans[-2][0], spans[-1][1])
        spans.pop()
    return spans or [(start, end)]


def _parent_spans(
    text: str, parent_chars: int, headings: Sequence[tuple[int, str]]
) -> list[tuple[int, int, str | None]]:
    """Cut a document into non-overlapping parent spans.

    Parents do not overlap. Two overlapping parents would put the same sentences
    in the prompt twice whenever both were selected, and the reader pays for
    every repeated token.
    """
    if not headings:
        return [(a, b, None) for a, b in _windows(0, len(text), parent_chars, 0)]

    sections: list[tuple[int, int, str | None]] = []
    if headings[0][0] > 0:
        sections.append((0, headings[0][0], None))
    for index, (offset, title) in enumerate(headings):
        stop = headings[index + 1][0] if index + 1 < len(headings) else len(text)
        # A section longer than one parent window is split, and every piece
        # keeps the section title so a citation still names where it came from.
        sections.extend(
            (a, b, title) for a, b in _windows(offset, stop, parent_chars, 0)
        )
    return sections


@dataclass
class Hierarchy:
    """Parents and children for one or more documents.

    Parameters
    ----------
    parents : tuple of Node
        Read units, in document order.
    children : tuple of Node
        Indexed units, in document order.
    """

    parents: tuple[Node, ...] = ()
    children: tuple[Node, ...] = ()
    _by_id: dict[str, Node] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Build the identifier lookup the expansion step reads."""
        self._by_id = {node.node_id: node for node in (*self.parents, *self.children)}

    def parent_of(self, node_id: str) -> Node | None:
        """Return the parent of a child identifier, or ``None`` if unknown."""
        child = self._by_id.get(node_id)
        if child is None or child.parent_id is None:
            return None
        return self._by_id.get(child.parent_id)

    def node(self, node_id: str) -> Node | None:
        """Return any node by identifier, parent or child."""
        return self._by_id.get(node_id)

    def expand(self, node_ids: Sequence[str], limit: int | None = None) -> list[Node]:
        """Map ranked child identifiers to their parents, de-duplicated.

        Two children of one parent collapse to a single read unit, which is the
        point: a hit cluster inside one section costs one section of
        context, not three overlapping copies of it.

        Parameters
        ----------
        node_ids : sequence of str
            Child identifiers in rank order.
        limit : int or None, optional
            Maximum parents to return.

        Returns
        -------
        list of Node
            Parents in first-hit order. Unknown identifiers are skipped.
        """
        seen: dict[str, Node] = {}
        for node_id in node_ids:
            parent = self.parent_of(node_id)
            if parent is not None:
                seen.setdefault(parent.node_id, parent)
            if limit is not None and len(seen) >= limit:
                break
        return list(seen.values())

    def summary(self) -> dict[str, float | int]:
        """Return counts a participant can sanity-check a split against."""
        parent_chars = sum(len(p.text) for p in self.parents)
        child_chars = sum(len(c.text) for c in self.children)
        return {
            "documents": len({p.doc_id for p in self.parents}),
            "parents": len(self.parents),
            "children": len(self.children),
            "mean_parent_chars": round(parent_chars / max(len(self.parents), 1), 1),
            "mean_child_chars": round(child_chars / max(len(self.children), 1), 1),
            "children_per_parent": round(
                len(self.children) / max(len(self.parents), 1), 2
            ),
        }

    @classmethod
    def combine(cls, parts: Iterable["Hierarchy"]) -> "Hierarchy":
        """Merge per-document hierarchies into one index."""
        parents: list[Node] = []
        children: list[Node] = []
        for part in parts:
            parents.extend(part.parents)
            children.extend(part.children)
        return cls(parents=tuple(parents), children=tuple(children))


def split_document(
    doc_id: str,
    text: str,
    *,
    parent_chars: int = 3000,
    child_chars: int = 600,
    overlap_chars: int = 100,
    use_headings: bool = True,
) -> Hierarchy:
    """Split one document into parent read units and child indexed units.

    A child never straddles two parents. That is what makes the parent lookup
    single-valued, and it is why children are cut inside each parent rather than
    over the whole document.

    Parameters
    ----------
    doc_id : str
        Identifier of the source document.
    text : str
        The document text.
    parent_chars : int, optional
        Maximum characters in a read unit.
    child_chars : int, optional
        Characters in an indexed unit.
    overlap_chars : int, optional
        Overlap between neighbouring children, to keep a sentence cut in half
        from being lost by both windows.
    use_headings : bool, optional
        Align parents to detected headings. Turn this off for documents with no
        heading structure, which falls back to fixed windows.

    Returns
    -------
    Hierarchy
        The parents and children for this document.

    Raises
    ------
    ValueError
        If the sizes cannot produce forward progress.
    """
    if child_chars <= 0 or parent_chars <= 0:
        raise ValueError("parent_chars and child_chars must be positive")
    if not 0 <= overlap_chars < child_chars:
        raise ValueError("overlap_chars must be in [0, child_chars)")
    if child_chars > parent_chars:
        raise ValueError("child_chars must not exceed parent_chars")

    headings = detect_headings(text) if use_headings else []
    parents: list[Node] = []
    children: list[Node] = []
    for index, (start, end, heading) in enumerate(
        _parent_spans(text, parent_chars, headings)
    ):
        window = text[start:end]
        if not window.strip():
            continue
        parent_id = f"{doc_id}::p{index:03d}"
        parents.append(
            Node(
                node_id=parent_id,
                doc_id=doc_id,
                text=window,
                start=start,
                end=end,
                heading=heading,
            )
        )
        for child_index, (child_start, child_end) in enumerate(
            _windows(start, end, child_chars, overlap_chars)
        ):
            child_text = text[child_start:child_end]
            if not child_text.strip():
                continue
            children.append(
                Node(
                    node_id=f"{parent_id}::c{child_index:03d}",
                    doc_id=doc_id,
                    text=child_text,
                    start=child_start,
                    end=child_end,
                    parent_id=parent_id,
                    heading=heading,
                )
            )
    return Hierarchy(parents=tuple(parents), children=tuple(children))


def split_documents(
    documents: Mapping[str, str],
    **options: Any,
) -> Hierarchy:
    """Split several documents and combine them into one hierarchy."""
    return Hierarchy.combine(
        split_document(doc_id, text, **options) for doc_id, text in documents.items()
    )


CONTEXT_TEMPLATE = "{header}\n\n{text}"


def contextualize(
    hierarchy: Hierarchy,
    *,
    template: str = CONTEXT_TEMPLATE,
    describe: Callable[[str, str], str] | None = None,
) -> tuple[Node, ...]:
    """Prefix each child with where it sits, for indexing.

    A 600-character window pulled out of a contract does not say which contract
    or which clause it belongs to, so a query naming either has nothing to match
    on. The prefix puts that back.

    Offsets are left alone. They keep pointing into the source document, so
    span-derived relevance judgments and any citation built from them stay
    correct even though ``text`` no longer starts at ``start``.

    Parameters
    ----------
    hierarchy : Hierarchy
        The split to contextualise.
    template : str, optional
        Template with ``{header}`` and ``{text}``.
    describe : callable or None, optional
        Takes ``(parent_text, child_text)`` and returns a sentence situating the
        child. This is one model call per child, which on a full corpus is the
        most expensive thing in the pipeline; leave it ``None`` for the
        deterministic header alone.

    Returns
    -------
    tuple of Node
        Children with prefixed text, in the input order.
    """
    out: list[Node] = []
    for child in hierarchy.children:
        parent = hierarchy.parent_of(child.node_id)
        parts = [child.doc_id]
        if child.heading:
            parts.append(child.heading)
        if describe is not None and parent is not None:
            situated = describe(parent.text, child.text).strip()
            if situated:
                parts.append(situated)
        header = " > ".join(parts)
        out.append(replace(child, text=template.format(header=header, text=child.text)))
    return tuple(out)


def small_to_big(
    hierarchy: Hierarchy,
    evidence: Sequence[Evidence],
    limit: int | None = None,
) -> list[Evidence]:
    """Replace retrieved child passages with their parents' text.

    The returned provenance keeps the child's ``chunk_id`` because that is the
    unit the retriever ranked and the unit the retrieval metrics score. The
    offsets and section move to the parent because that is the text now being
    read. ``extra`` records both identifiers so the swap is auditable.

    Passages whose identifier is not in the hierarchy pass through untouched, so
    this is safe to run over a mixed result set.

    Parameters
    ----------
    hierarchy : Hierarchy
        The split the children came from.
    evidence : sequence of Evidence
        Retrieved passages in rank order.
    limit : int or None, optional
        Maximum passages to return after de-duplication.

    Returns
    -------
    list of Evidence
        One passage per distinct parent, in first-hit order.
    """
    out: list[Evidence] = []
    seen: set[str] = set()
    for item in evidence:
        child_id = item.chunk_id or ""
        parent = hierarchy.parent_of(child_id)
        if parent is None:
            out.append(item)
        else:
            if parent.node_id in seen:
                continue
            seen.add(parent.node_id)
            out.append(
                replace(
                    item,
                    text=parent.text,
                    provenance=replace(
                        item.provenance,
                        section=parent.heading,
                        start=parent.start,
                        end=parent.end,
                        extra={
                            **item.provenance.extra,
                            "parent_id": parent.node_id,
                            "child_span": (item.provenance.start, item.provenance.end),
                        },
                    ),
                )
            )
        if limit is not None and len(out) >= limit:
            break
    return out


def to_evidence(
    nodes: Sequence[Node], retriever: str = "hierarchical"
) -> list[Evidence]:
    """Wrap nodes as evidence, for handing a parent lookup straight to a reader."""
    return [
        Evidence(text=node.text, provenance=node.provenance(), retriever=retriever)
        for node in nodes
    ]


def to_chunks(nodes: Sequence[Node]) -> list["Chunk"]:
    """Convert nodes to the harness chunk type so existing retrievers index them.

    The harness names the document field ``contract_id`` because it was built on
    CUAD; the mapping happens here so nothing else has to know.

    Parameters
    ----------
    nodes : sequence of Node
        Usually ``hierarchy.children``, or the output of :func:`contextualize`.

    Returns
    -------
    list of Chunk
        The same units in the harness's shape.
    """
    # Imported inside the function: the retrieval half of the harness is being
    # consolidated onto this path, and a module-level import would make the
    # whole pipeline unimportable until that lands.
    from aieng.advanced_rag.retrieval.corpus import Chunk as HarnessChunk

    return [
        HarnessChunk(
            chunk_id=node.node_id,
            doc_id=node.doc_id,
            text=node.text,
            start=node.start,
            end=node.end,
        )
        for node in nodes
    ]
