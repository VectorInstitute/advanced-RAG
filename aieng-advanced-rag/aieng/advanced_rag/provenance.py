"""Where a piece of retrieved text came from.

Every RI has to answer "why do you believe that", and RI #4 requires citations
outright. Without one shared record, each experiment invents its own and the
citation surface differs per method, which is what happened before this module
existed.

The model follows a Document / Section / Block / Atom hierarchy, flattened to
the part that survives every graph store tested. NetworkX, Kuzu, and Neo4j all
hold list and scalar properties but none of them hold nested objects, so
provenance also has a flat string form (:func:`encode`, :func:`decode`) for
round-tripping through a graph property.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_FIELD_SEPARATOR = "|"
_PAIR_SEPARATOR = ":"


def _escape(value: str) -> str:
    """Escape separator characters so a value cannot split its own record."""
    return (
        value.replace("\\", "\\\\")
        .replace(_FIELD_SEPARATOR, "\\p")
        .replace(_PAIR_SEPARATOR, "\\c")
    )


def _unescape(value: str) -> str:
    """Reverse :func:`_escape`."""
    out: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character == "\\" and index + 1 < len(value):
            following = value[index + 1]
            out.append(
                {"p": _FIELD_SEPARATOR, "c": _PAIR_SEPARATOR, "\\": "\\"}.get(
                    following, following
                )
            )
            index += 2
            continue
        out.append(character)
        index += 1
    return "".join(out)


@dataclass(frozen=True)
class Provenance:
    """A citable location inside a corpus.

    Parameters
    ----------
    doc_id : str
        Stable identifier for the source document.
    chunk_id : str or None, optional
        Identifier for the retrieved unit within the document.
    section : str or None, optional
        Human-readable heading or section path, for display in a citation.
    start : int or None, optional
        Character offset of the span within the document text.
    end : int or None, optional
        Exclusive end offset of the span.
    page : int or None, optional
        Page number, for paginated sources such as Docling output.
    extra : dict, optional
        Anything source-specific that should not force a schema change.
    """

    doc_id: str
    chunk_id: str | None = None
    section: str | None = None
    start: int | None = None
    end: int | None = None
    page: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def span(self) -> tuple[int, int] | None:
        """Return the character span, or ``None`` when offsets are absent."""
        if self.start is None or self.end is None:
            return None
        return (self.start, self.end)

    def citation(self) -> str:
        """Render a short human-readable citation.

        Returns
        -------
        str
            Something like ``"q266232 > Accepted answer (p. 3)"``, degrading
            gracefully when section or page are missing.
        """
        parts = [self.doc_id]
        if self.section:
            parts.append(self.section)
        rendered = " > ".join(parts)
        if self.page is not None:
            rendered = f"{rendered} (p. {self.page})"
        return rendered

    def encode(self) -> str:
        """Flatten to a single string for storage in a graph property.

        Returns
        -------
        str
            A pipe-delimited ``key:value`` record, omitting absent fields.
        """
        pairs: list[tuple[str, str]] = [("doc", self.doc_id)]
        for key, value in (
            ("chunk", self.chunk_id),
            ("section", self.section),
            ("start", self.start),
            ("end", self.end),
            ("page", self.page),
        ):
            if value is not None:
                pairs.append((key, str(value)))
        return _FIELD_SEPARATOR.join(
            f"{key}{_PAIR_SEPARATOR}{_escape(value)}" for key, value in pairs
        )

    @classmethod
    def decode(cls, encoded: str) -> "Provenance":
        """Rebuild from :meth:`encode` output.

        Parameters
        ----------
        encoded : str
            A record produced by :meth:`encode`.

        Returns
        -------
        Provenance
            The reconstructed record.

        Raises
        ------
        ValueError
            If the record carries no document identifier.
        """
        fields: dict[str, str] = {}
        for segment in encoded.split(_FIELD_SEPARATOR):
            if _PAIR_SEPARATOR not in segment:
                continue
            key, value = segment.split(_PAIR_SEPARATOR, 1)
            fields[key] = _unescape(value)

        if "doc" not in fields:
            raise ValueError(f"provenance record has no doc id: {encoded!r}")

        def as_int(key: str) -> int | None:
            raw = fields.get(key)
            return int(raw) if raw is not None and raw.lstrip("-").isdigit() else None

        return cls(
            doc_id=fields["doc"],
            chunk_id=fields.get("chunk"),
            section=fields.get("section"),
            start=as_int("start"),
            end=as_int("end"),
            page=as_int("page"),
        )

    @classmethod
    def from_chunk(cls, chunk: Any) -> "Provenance":
        """Build from a chunk object carrying offsets.

        Accepts anything with ``chunk_id`` and a document attribute, which
        covers the chunk types used by the evaluation harness and the graph
        experiments without importing either.

        Parameters
        ----------
        chunk : Any
            An object with ``chunk_id`` and one of ``doc_id`` or
            ``contract_id``, optionally ``start`` and ``end``.

        Returns
        -------
        Provenance
            The corresponding record.
        """
        doc_id = getattr(chunk, "doc_id", None) or getattr(chunk, "contract_id", None)
        if doc_id is None:
            raise ValueError(f"cannot find a document id on {chunk!r}")
        return cls(
            doc_id=str(doc_id),
            chunk_id=getattr(chunk, "chunk_id", None),
            start=getattr(chunk, "start", None),
            end=getattr(chunk, "end", None),
        )

    @classmethod
    def from_docling(cls, doc_id: str, item: dict[str, Any]) -> "Provenance":
        """Build from a Docling item's ``prov`` block.

        Docling records page numbers under ``prov[0].page_no``.

        Parameters
        ----------
        doc_id : str
            Identifier for the source document.
        item : dict
            A Docling element, optionally carrying ``prov`` and ``label``.

        Returns
        -------
        Provenance
            The corresponding record.
        """
        prov = (item.get("prov") or [{}])[0]
        return cls(
            doc_id=doc_id,
            chunk_id=item.get("self_ref"),
            section=item.get("label"),
            page=prov.get("page_no"),
            extra={"docling_label": item.get("label")} if item.get("label") else {},
        )
