"""Direct graph extraction with an enforced schema and cited provenance.

One pass over a corpus turns each document into entities and relations. Three
properties make the result usable for support work.

**The schema is enforced after the call.** A model asked for seven entity types
and seven relation types returns near-misses: ``resolves`` for ``fixes``, ``tool``
for ``Software``. :data:`ENTITY_ALIASES` and :data:`RELATION_ALIASES` map those
onto the schema, a relation that maps to nothing is dropped, and an entity type
that maps to nothing becomes ``Concept``. Every label in the finished graph is on
the schema, and :class:`SchemaReport` counts what was mapped, coerced, or dropped
so the rate is visible.

**Every fact cites where it came from.** The model quotes the words that support
each entity and relation, and the quote is located in the document text. The
location is a :class:`~aieng.advanced_rag.provenance.Provenance`: the document, the
section (the question or the accepted answer), the block within it, and the
character span of the quote. A quote that cannot be found is kept with its section
and marked unlocated, instead of dropping the fact or inventing a span.

**The graph is deterministic.** Calls run concurrently, but results are folded in
document order, so the graph does not depend on worker count or completion order.

The model call is any ``complete(prompt) -> str``, so extraction is tested without
a network. A ``complete`` with a ``batch`` method, such as
:meth:`aieng.advanced_rag.batch.ProxyBatch.completer`, receives every window's
prompt in one batch instead of one call per thread.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aieng.advanced_rag.provenance import Provenance


ENTITY_TYPES = (
    "Software",
    "Hardware",
    "Protocol",
    "ConfigItem",
    "Command",
    "ErrorOrSymptom",
    "Concept",
)

RELATION_TYPES = (
    "causes",
    "fixes",
    "configures",
    "depends_on",
    "part_of",
    "alternative_to",
    "diagnoses",
)

#: Surface forms a model returns for each entity type, lowercased with spaces and
#: punctuation removed. A type not found here becomes ``Concept``.
ENTITY_ALIASES: dict[str, str] = {
    "software": "Software",
    "tool": "Software",
    "service": "Software",
    "application": "Software",
    "program": "Software",
    "package": "Software",
    "library": "Software",
    "daemon": "Software",
    "operatingsystem": "Software",
    "os": "Software",
    "hardware": "Hardware",
    "device": "Hardware",
    "disk": "Hardware",
    "server": "Hardware",
    "networkdevice": "Hardware",
    "protocol": "Protocol",
    "standard": "Protocol",
    "networkprotocol": "Protocol",
    "configitem": "ConfigItem",
    "config": "ConfigItem",
    "configuration": "ConfigItem",
    "setting": "ConfigItem",
    "option": "ConfigItem",
    "directive": "ConfigItem",
    "flag": "ConfigItem",
    "parameter": "ConfigItem",
    "command": "Command",
    "cmd": "Command",
    "commandline": "Command",
    "cli": "Command",
    "errororsymptom": "ErrorOrSymptom",
    "error": "ErrorOrSymptom",
    "symptom": "ErrorOrSymptom",
    "failure": "ErrorOrSymptom",
    "issue": "ErrorOrSymptom",
    "problem": "ErrorOrSymptom",
    "errormessage": "ErrorOrSymptom",
    "concept": "Concept",
    "technique": "Concept",
    "idea": "Concept",
    "other": "Concept",
}

#: Surface forms a model returns for each relation type, normalised the same way. A
#: relation type not found here is dropped.
RELATION_ALIASES: dict[str, str] = {
    "causes": "causes",
    "cause": "causes",
    "causedby": "causes",
    "leadsto": "causes",
    "resultsin": "causes",
    "triggers": "causes",
    "fixes": "fixes",
    "fix": "fixes",
    "resolves": "fixes",
    "solves": "fixes",
    "mitigates": "fixes",
    "repairs": "fixes",
    "configures": "configures",
    "configure": "configures",
    "sets": "configures",
    "enables": "configures",
    "disables": "configures",
    "controls": "configures",
    "dependson": "depends_on",
    "requires": "depends_on",
    "needs": "depends_on",
    "uses": "depends_on",
    "relieson": "depends_on",
    "partof": "part_of",
    "componentof": "part_of",
    "belongsto": "part_of",
    "includedin": "part_of",
    "contains": "part_of",
    "alternativeto": "alternative_to",
    "replaces": "alternative_to",
    "insteadof": "alternative_to",
    "equivalentto": "alternative_to",
    "diagnoses": "diagnoses",
    "diagnose": "diagnoses",
    "detects": "diagnoses",
    "indicates": "diagnoses",
    "shows": "diagnoses",
    "checks": "diagnoses",
    "reveals": "diagnoses",
}

SYSTEM = (
    "You extract knowledge graphs from technical support content. "
    "You output only JSON. No prose, no markdown fences."
)

PROMPT = """Extract entities and relations from this system-administration text.

Entity types: {entity_types}
Relation types: {relation_types}

Rules:
- Use the exact surface form from the text as the entity name, lowercased.
- Only extract relations where both endpoints are entities you listed.
- Prefer specific entities ("nginx worker_connections") over vague ones ("server").
- For each entity and each relation, give an "evidence" field: the shortest exact
  quote from the text that supports it, copied word for word.
- Skip anything you are not confident about. A small precise graph beats a large
  noisy one.
- At most 12 entities and 12 relations.

Return exactly this JSON shape:
{{"entities": [{{"name": "...", "type": "...", "description": "...", "evidence": "..."}}],
  "relations": [{{"source": "...", "target": "...", "type": "...", "description": "...",
  "evidence": "..."}}]}}

TEXT:
{text}"""

# A call over more text than this is split at block boundaries.
MAX_WINDOW_CHARS = 6000


def _label_key(label: str) -> str:
    """Normalise a label for alias lookup: lowercase letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", label.lower())


def _name_key(name: str) -> str:
    return " ".join(name.lower().split())


def canonical_entity_type(label: str) -> tuple[str, bool]:
    """Return the schema entity type for a model's label, and whether it was coerced.

    Parameters
    ----------
    label : str
        What the model wrote.

    Returns
    -------
    tuple of (str, bool)
        The schema type, and True when the label matched nothing and became
        ``Concept``.
    """
    key = _label_key(label)
    if key in ENTITY_ALIASES:
        return ENTITY_ALIASES[key], False
    return "Concept", True


def canonical_relation_type(label: str) -> str | None:
    """Return the schema relation type for a model's label, or None when it has none."""
    return RELATION_ALIASES.get(_label_key(label))


@dataclass(frozen=True)
class Block:
    """One run of text inside a section.

    Parameters
    ----------
    doc_id : str
        The document the block belongs to.
    section : str
        ``Title``, ``Question``, or ``Accepted answer``.
    index : int
        The block's position within its section, from zero.
    text : str
        The block text.
    start : int
        Character offset of the block in the document text.
    end : int
        Exclusive end offset.
    """

    doc_id: str
    section: str
    index: int
    text: str
    start: int
    end: int

    @property
    def block_id(self) -> str:
        """Return the identifier a citation names: document, section, and block."""
        return f"{self.doc_id}::{self.section.lower().replace(' ', '-')}::b{self.index:03d}"


_SECTION_MARKERS = (
    ("## Question\n", "Question"),
    ("## Accepted answer\n", "Accepted answer"),
)


def sections(doc_id: str, text: str) -> list[tuple[str, list[Block]]]:
    """Split a support document into sections of blocks, with document offsets.

    The document is a title line, a tags line, a ``## Question`` section, and a
    ``## Accepted answer`` section, which is the text
    :class:`~aieng.advanced_rag.corpora.serverfault.Document` renders. Blocks are the
    paragraphs of a section, separated by blank lines.

    Parameters
    ----------
    doc_id : str
        The document identifier.
    text : str
        The document text.

    Returns
    -------
    list of (str, list of Block)
        Sections in document order. A section with no text is omitted.
    """
    marks: list[tuple[int, int, str]] = []
    for marker, name in _SECTION_MARKERS:
        at = text.find(marker)
        if at != -1:
            marks.append((at, at + len(marker), name))
    marks.sort()
    spans: list[tuple[str, int, int]] = []
    head_end = marks[0][0] if marks else len(text)
    spans.append(("Title", 0, head_end))
    for i, (_, body_start, name) in enumerate(marks):
        body_end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        spans.append((name, body_start, body_end))
    found: list[tuple[str, list[Block]]] = []
    for name, start, end in spans:
        blocks: list[Block] = []
        cursor = start
        for piece in re.split(r"(\n\s*\n)", text[start:end]):
            stripped = piece.strip()
            if stripped and not re.fullmatch(r"\n\s*\n", piece):
                offset = text.index(stripped, cursor)
                blocks.append(
                    Block(
                        doc_id,
                        name,
                        len(blocks),
                        stripped,
                        offset,
                        offset + len(stripped),
                    )
                )
                cursor = offset + len(stripped)
            else:
                cursor += len(piece)
        if blocks:
            found.append((name, blocks))
    return found


def windows(
    blocks: Sequence[Block], limit: int = MAX_WINDOW_CHARS
) -> list[list[Block]]:
    """Group a section's blocks into runs that fit one call, splitting at blocks."""
    runs: list[list[Block]] = []
    current: list[Block] = []
    size = 0
    for block in blocks:
        if current and size + len(block.text) > limit:
            runs.append(current)
            current, size = [], 0
        current.append(block)
        size += len(block.text) + 2
    if current:
        runs.append(current)
    return runs


def locate(quote: str, blocks: Sequence[Block]) -> Provenance | None:
    """Find a quote in a run of blocks and cite its span.

    An exact match is tried first, then a match that ignores case and runs of
    whitespace, because a model reflows a quote more often than it rewords one.

    Parameters
    ----------
    quote : str
        The words the model gave as evidence.
    blocks : sequence of Block
        The blocks the model was shown.

    Returns
    -------
    Provenance or None
        The document, section, block, and character span, or None when the quote is
        not in the text.
    """
    needle = quote.strip()
    if len(needle) < 3:
        return None
    for block in blocks:
        at = block.text.find(needle)
        if at != -1:
            return _cite(block, at, at + len(needle))
    squeezed = _squeeze(needle)
    for block in blocks:
        flat, index = _squeeze_with_index(block.text)
        at = flat.find(squeezed)
        if at != -1:
            return _cite(block, index[at], index[at + len(squeezed) - 1] + 1)
    return None


def _squeeze(text: str) -> str:
    return " ".join(text.lower().split())


def _squeeze_with_index(text: str) -> tuple[str, list[int]]:
    """Return lowercased text with whitespace collapsed, and each character's index."""
    flat: list[str] = []
    index: list[int] = []
    previous_space = True
    for position, char in enumerate(text):
        if char.isspace():
            if not previous_space:
                flat.append(" ")
                index.append(position)
            previous_space = True
        else:
            flat.append(char.lower())
            index.append(position)
            previous_space = False
    if flat and flat[-1] == " ":
        flat.pop()
        index.pop()
    return "".join(flat), index


def _cite(block: Block, start: int, end: int) -> Provenance:
    return Provenance(
        doc_id=block.doc_id,
        chunk_id=block.block_id,
        section=block.section,
        start=block.start + start,
        end=block.start + end,
    )


@dataclass(frozen=True)
class Mention:
    """One place a fact was found.

    Parameters
    ----------
    provenance : Provenance
        The document, section, block, and span. ``start`` and ``end`` are None when
        the quote could not be located, and ``chunk_id`` is then the section.
    quote : str
        The words the model gave as evidence.
    located : bool
        Whether the quote was found in the text.
    """

    provenance: Provenance
    quote: str
    located: bool


@dataclass
class Entity:
    """A graph node.

    Parameters
    ----------
    name : str
        The lowercased surface form.
    type : str
        One of :data:`ENTITY_TYPES`.
    description : str
        The longest description any mention gave.
    mentions : list of Mention
        Every place the entity was found.
    """

    name: str
    type: str
    description: str = ""
    mentions: list[Mention] = field(default_factory=list)

    @property
    def chunks(self) -> list[str]:
        """Return the block identifiers that mention the entity."""
        return list(dict.fromkeys(m.provenance.chunk_id or "" for m in self.mentions))

    @property
    def documents(self) -> list[str]:
        """Return the documents that mention the entity."""
        return list(dict.fromkeys(m.provenance.doc_id for m in self.mentions))


@dataclass
class Relation:
    """A graph edge between two entities."""

    source: str
    target: str
    type: str
    description: str = ""
    mentions: list[Mention] = field(default_factory=list)

    @property
    def chunks(self) -> list[str]:
        """Return the block identifiers that state the relation."""
        return list(dict.fromkeys(m.provenance.chunk_id or "" for m in self.mentions))

    @property
    def documents(self) -> list[str]:
        """Return the documents that state the relation."""
        return list(dict.fromkeys(m.provenance.doc_id for m in self.mentions))


@dataclass
class SchemaReport:
    """What the schema enforcement did, so its rate is visible.

    Attributes
    ----------
    entity_types_mapped : int
        Entities whose type was an alias for a schema type.
    entity_types_coerced : int
        Entities whose type matched nothing and became ``Concept``.
    relations_mapped : int
        Relations whose type was an alias for a schema type.
    relations_rejected : int
        Relations dropped because their type matched nothing.
    relations_dangling : int
        Relations dropped because an endpoint was not an extracted entity.
    quotes_located : int
        Facts whose quote was found in the text.
    quotes_unlocated : int
        Facts kept with section-level provenance because the quote was not found.
    calls_failed : int
        Windows whose call raised or returned nothing parseable.
    """

    entity_types_mapped: int = 0
    entity_types_coerced: int = 0
    relations_mapped: int = 0
    relations_rejected: int = 0
    relations_dangling: int = 0
    quotes_located: int = 0
    quotes_unlocated: int = 0
    calls_failed: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return the counts as a plain dict."""
        return dict(self.__dict__)


@dataclass
class KnowledgeGraph:
    """Entities and relations with their mentions.

    Entities are keyed by lowercased name and relations by source, target, and type.
    ``ExtractedGraphRetriever`` (in ``graphrag.adapters.extracted_graph``) reads
    ``entities`` and ``relations`` directly.
    """

    entities: dict[str, Entity] = field(default_factory=dict)
    relations: dict[tuple[str, str, str], Relation] = field(default_factory=dict)
    report: SchemaReport = field(default_factory=SchemaReport)

    def stats(self) -> dict[str, Any]:
        """Return counts that say what the graph holds and how well it is cited."""
        degrees: Counter[str] = Counter()
        for relation in self.relations.values():
            degrees[relation.source] += 1
            degrees[relation.target] += 1
        mentions = [m for e in self.entities.values() for m in e.mentions]
        mentions += [m for r in self.relations.values() for m in r.mentions]
        cross = sum(1 for e in self.entities.values() if len(e.documents) > 1)
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "entity_types": dict(Counter(e.type for e in self.entities.values())),
            "relation_types": dict(Counter(r.type for r in self.relations.values())),
            "isolated_entities": len(self.entities) - len(degrees),
            "cross_document_entities": cross,
            "mentions": len(mentions),
            "located_mentions": sum(1 for m in mentions if m.located),
            "schema_report": self.report.to_dict(),
        }

    def validate(self) -> list[str]:
        """Return every way the graph breaks its own rules, empty when it holds.

        Checks that every label is on the schema, every relation's endpoints are
        entities, and every fact has at least one mention that names a document.
        """
        problems: list[str] = []
        for entity in self.entities.values():
            if entity.type not in ENTITY_TYPES:
                problems.append(
                    f"entity {entity.name!r} has off-schema type {entity.type!r}"
                )
            if not entity.mentions:
                problems.append(f"entity {entity.name!r} cites nothing")
        for relation in self.relations.values():
            label = f"{relation.source} -{relation.type}-> {relation.target}"
            if relation.type not in RELATION_TYPES:
                problems.append(f"relation {label} has an off-schema type")
            if (
                relation.source not in self.entities
                or relation.target not in self.entities
            ):
                problems.append(f"relation {label} has a dangling endpoint")
            if not relation.mentions:
                problems.append(f"relation {label} cites nothing")
        return problems

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable form that :meth:`from_dict` reads back."""

        def mention(m: Mention) -> dict[str, Any]:
            return {
                "provenance": m.provenance.encode(),
                "quote": m.quote,
                "located": m.located,
            }

        return {
            "entities": [
                {
                    "name": e.name,
                    "type": e.type,
                    "description": e.description,
                    "mentions": [mention(m) for m in e.mentions],
                }
                for e in self.entities.values()
            ],
            "relations": [
                {
                    "source": r.source,
                    "target": r.target,
                    "type": r.type,
                    "description": r.description,
                    "mentions": [mention(m) for m in r.mentions],
                }
                for r in self.relations.values()
            ],
            "stats": self.stats(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> KnowledgeGraph:
        """Rebuild a graph from :meth:`to_dict`."""

        def mentions(items: Iterable[Mapping[str, Any]]) -> list[Mention]:
            return [
                Mention(
                    Provenance.decode(i["provenance"]), i["quote"], bool(i["located"])
                )
                for i in items
            ]

        graph = cls()
        for item in payload["entities"]:
            graph.entities[item["name"]] = Entity(
                item["name"],
                item["type"],
                item["description"],
                mentions(item["mentions"]),
            )
        for item in payload["relations"]:
            relation = Relation(
                item["source"],
                item["target"],
                item["type"],
                item["description"],
                mentions(item["mentions"]),
            )
            graph.relations[(relation.source, relation.target, relation.type)] = (
                relation
            )
        report = payload.get("stats", {}).get("schema_report", {})
        graph.report = SchemaReport(**{k: int(v) for k, v in report.items()})
        return graph

    def save(self, directory: Path) -> None:
        """Write ``graph.json`` and ``graph.graphml`` into ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "graph.json").write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        self.write_graphml(directory / "graph.graphml")

    @classmethod
    def load(cls, directory: Path) -> KnowledgeGraph:
        """Read a graph written by :meth:`save`."""
        payload = json.loads(
            (Path(directory) / "graph.json").read_text(encoding="utf-8")
        )
        return cls.from_dict(payload)

    def write_graphml(self, path: Path) -> None:
        """Write the graph as GraphML, with each node's and edge's citations as text."""
        import networkx as nx  # noqa: PLC0415

        graph = nx.MultiDiGraph()
        for entity in self.entities.values():
            graph.add_node(
                entity.name,
                type=entity.type,
                description=entity.description,
                documents="|".join(entity.documents),
                provenance="\n".join(m.provenance.encode() for m in entity.mentions),
            )
        for relation in self.relations.values():
            graph.add_edge(
                relation.source,
                relation.target,
                key=relation.type,
                type=relation.type,
                description=relation.description,
                documents="|".join(relation.documents),
                provenance="\n".join(m.provenance.encode() for m in relation.mentions),
            )
        nx.write_graphml(graph, path)


def parse_json_object(raw: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response, tolerating fences and extra prose."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    for candidate in (text, text[text.find("{") : text.rfind("}") + 1]):
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    return {}


def _fold(
    graph: KnowledgeGraph,
    payload: Mapping[str, Any],
    run: Sequence[Block],
    section: str,
    doc_id: str,
) -> None:
    """Fold one window's extraction into the graph, enforcing the schema."""
    report = graph.report
    section_cite = Provenance(
        doc_id=doc_id, chunk_id=f"{doc_id}::{section}", section=section
    )

    def mention(quote: str) -> Mention:
        found = locate(quote, run) if quote else None
        if found is None:
            report.quotes_unlocated += 1
            return Mention(section_cite, quote, False)
        report.quotes_located += 1
        return Mention(found, quote, True)

    _fold_entities(graph, payload.get("entities") or [], mention)
    _fold_relations(graph, payload.get("relations") or [], mention)


def _fold_entities(
    graph: KnowledgeGraph, items: Iterable[Any], mention: Callable[[str], Mention]
) -> None:
    report = graph.report
    for item in items:
        if not isinstance(item, dict) or not str(item.get("name", "")).strip():
            continue
        name = _name_key(str(item["name"]))
        kind, coerced = canonical_entity_type(str(item.get("type", "")))
        if coerced:
            report.entity_types_coerced += 1
        else:
            report.entity_types_mapped += 1
        description = str(item.get("description", ""))
        entity = graph.entities.setdefault(name, Entity(name, kind, description))
        if len(description) > len(entity.description):
            entity.description = description
        entity.mentions.append(mention(str(item.get("evidence", ""))))


def _fold_relations(
    graph: KnowledgeGraph, items: Iterable[Any], mention: Callable[[str], Mention]
) -> None:
    report = graph.report
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = canonical_relation_type(str(item.get("type", "")))
        if kind is None:
            report.relations_rejected += 1
            continue
        source = _name_key(str(item.get("source", "")))
        target = _name_key(str(item.get("target", "")))
        if not source or not target or source == target:
            continue
        if source not in graph.entities or target not in graph.entities:
            report.relations_dangling += 1
            continue
        report.relations_mapped += 1
        description = str(item.get("description", ""))
        relation = graph.relations.setdefault(
            (source, target, kind), Relation(source, target, kind, description)
        )
        if len(description) > len(relation.description):
            relation.description = description
        relation.mentions.append(mention(str(item.get("evidence", ""))))


def build_graph(
    documents: Iterable[tuple[str, str]],
    complete: Callable[[str], str],
    *,
    max_workers: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> KnowledgeGraph:
    """Extract every document into one graph.

    Parameters
    ----------
    documents : iterable of (str, str)
        ``(doc_id, text)`` pairs, with the text shaped as :func:`sections` reads.
    complete : callable
        Takes a prompt, returns the model's text. It is called from several
        threads, so it must be safe to share. When it has a
        ``batch(prompts, on_progress=None)`` method returning one reply or one
        exception per prompt, every prompt goes to that method at once and
        ``max_workers`` is not used; the batch sets its own concurrency.
    max_workers : int, optional
        Concurrent calls.
    on_progress : callable, optional
        Called with ``(finished, total)`` as calls complete.

    Returns
    -------
    KnowledgeGraph
        The graph, folded in document order so it does not depend on completion
        order. A call that raises or returns nothing parseable adds nothing and is
        counted in ``report.calls_failed``.
    """
    jobs: list[tuple[str, str, list[Block]]] = []
    for doc_id, text in documents:
        found = sections(doc_id, text)
        # The title and tags are a line or two, so they share the question's call.
        # That makes two calls per document instead of three, and the title's words
        # are context the question's extraction needs.
        head = [block for name, blocks in found if name == "Title" for block in blocks]
        for name, blocks in found:
            if name == "Title" and any(n == "Question" for n, _ in found):
                continue
            body = head + blocks if name == "Question" else blocks
            for run in windows(body):
                jobs.append((doc_id, name, run))

    def prompt_for(job: tuple[str, str, list[Block]]) -> str:
        _, _, run = job
        body = "\n\n".join(block.text for block in run)
        prompt = PROMPT.format(
            entity_types=", ".join(ENTITY_TYPES),
            relation_types=", ".join(RELATION_TYPES),
            text=body,
        )
        return f"{SYSTEM}\n\n{prompt}"

    def call(job: tuple[str, str, list[Block]]) -> dict[str, Any] | None:
        try:
            return parse_json_object(complete(prompt_for(job))) or None
        except Exception:  # noqa: BLE001 - any failure is one failed window, counted below
            return None

    results: list[dict[str, Any] | None] = [None] * len(jobs)
    batch = getattr(complete, "batch", None)
    if callable(batch):
        replies = batch([prompt_for(job) for job in jobs], on_progress=on_progress)
        results = [
            None
            if isinstance(reply, BaseException)
            else parse_json_object(str(reply)) or None
            for reply in replies
        ]
    else:
        finished = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(call, job): i for i, job in enumerate(jobs)}
            from concurrent.futures import as_completed  # noqa: PLC0415

            for future in as_completed(futures):
                results[futures[future]] = future.result()
                finished += 1
                if on_progress is not None:
                    on_progress(finished, len(jobs))

    graph = KnowledgeGraph()
    for (doc_id, section, run), payload in zip(jobs, results, strict=True):
        if payload is None:
            graph.report.calls_failed += 1
            continue
        _fold(graph, payload, run, section, doc_id)
    return graph


__all__ = [
    "ENTITY_ALIASES",
    "ENTITY_TYPES",
    "RELATION_ALIASES",
    "RELATION_TYPES",
    "Block",
    "Entity",
    "KnowledgeGraph",
    "Mention",
    "Relation",
    "SchemaReport",
    "build_graph",
    "canonical_entity_type",
    "canonical_relation_type",
    "locate",
    "parse_json_object",
    "sections",
    "windows",
]
