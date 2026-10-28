"""Maintain an extracted knowledge graph as its corpus changes.

A graph built by :func:`~aieng.advanced_rag.graphrag.extraction.build_graph` is a
fold of per-document extractions: every mention names the document it came from.
That makes four maintenance operations possible without a rebuild.

* **Retraction by provenance.** :func:`retract` removes every mention that cites a
  document, drops the entities and relations left with no support, drops relations
  whose endpoint went with them, and reports what changed around them.
* **Revision.** :func:`revise` retracts a document and folds its new extraction
  in, and :func:`edit_alignment` checks the changed facts against the changed text.
* **Deduplication.** :func:`duplicate_groups` proposes near-duplicate entity names
  and :func:`merge_entities` merges a group, unioning provenance and rewiring
  relations.
* **A gate.** :func:`maintenance_gate` runs schema checks, compares topology before
  and after, and asks smoke questions that must not cite a retracted document.

Entity and relation descriptions in a :class:`KnowledgeGraph` carry no provenance:
each is the longest description any mention gave. The per-document extractions
(:func:`extract_units`) do carry it, so :func:`retract` repairs a description that
came from the retracted document when it is given them, and otherwise reports the
description as unverified.

The module also reads Stack Exchange post history (:func:`read_post_history`) so a
revision can be replayed from the dump's real earlier versions.
"""

from __future__ import annotations

import difflib
import math
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any
from xml.etree import ElementTree as ET

from .extraction import (
    Entity,
    KnowledgeGraph,
    Mention,
    Relation,
    SchemaReport,
    build_graph,
)


RelationKey = tuple[str, str, str]

#: Stack Exchange ``PostHistoryTypeId`` values that carry a full body.
BODY_REVISION_TYPES = (2, 5, 8)
#: ``PostHistoryTypeId`` values that carry a full title.
TITLE_REVISION_TYPES = (1, 4, 7)


# ---------------------------------------------------------------------------
# Per-document extraction units
# ---------------------------------------------------------------------------


def extract_units(
    documents: Iterable[tuple[str, str]],
    complete: Callable[[str], str],
    *,
    max_workers: int = 4,
) -> dict[str, KnowledgeGraph]:
    """Extract each document into its own graph, keyed by document id.

    Each unit is what :func:`~aieng.advanced_rag.graphrag.extraction.build_graph`
    returns for one document, so its descriptions are that document's own. With a
    ``complete`` that replays a response cache, this rebuilds the units of a
    finished graph at no cost.

    Parameters
    ----------
    documents : iterable of (str, str)
        ``(doc_id, text)`` pairs.
    complete : callable
        ``complete(prompt) -> str``, shared across threads.
    max_workers : int, optional
        Documents extracted at once.

    Returns
    -------
    dict of str to KnowledgeGraph
        One unit per document, in input order.
    """
    pairs = list(documents)

    def one(pair: tuple[str, str]) -> KnowledgeGraph:
        return build_graph([pair], complete, max_workers=2)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        graphs = list(pool.map(one, pairs))
    return {doc_id: graph for (doc_id, _), graph in zip(pairs, graphs, strict=True)}


def fold_units(
    units: Mapping[str, KnowledgeGraph], order: Sequence[str] | None = None
) -> KnowledgeGraph:
    """Fold per-document units into one graph, in document order.

    An entity keeps the type of its first unit and the longest description, with
    ties going to the earlier unit, which is the rule ``build_graph`` applies to
    mentions. A unit only holds relations whose endpoints it extracted itself, so a
    folded graph can hold fewer relations than a single ``build_graph`` pass over
    the same documents, which accepts an endpoint extracted from an earlier
    document.

    Parameters
    ----------
    units : mapping of str to KnowledgeGraph
        Per-document graphs.
    order : sequence of str, optional
        Document order. Defaults to the mapping's order.

    Returns
    -------
    KnowledgeGraph
        The folded graph, with the units' schema reports summed.
    """
    graph = KnowledgeGraph()
    for doc_id in order if order is not None else list(units):
        add_unit(graph, units[doc_id])
    return graph


def add_unit(graph: KnowledgeGraph, unit: KnowledgeGraph) -> None:
    """Fold one document's unit into ``graph`` in place."""
    for entity in unit.entities.values():
        held = graph.entities.get(entity.name)
        if held is None:
            graph.entities[entity.name] = Entity(
                entity.name, entity.type, entity.description, list(entity.mentions)
            )
            continue
        if len(entity.description) > len(held.description):
            held.description = entity.description
        held.mentions.extend(entity.mentions)
    for key, relation in unit.relations.items():
        held_relation = graph.relations.get(key)
        if held_relation is None:
            graph.relations[key] = Relation(
                relation.source,
                relation.target,
                relation.type,
                relation.description,
                list(relation.mentions),
            )
            continue
        if len(relation.description) > len(held_relation.description):
            held_relation.description = relation.description
        held_relation.mentions.extend(relation.mentions)
    for name, value in unit.report.to_dict().items():
        setattr(graph.report, name, getattr(graph.report, name) + value)


# ---------------------------------------------------------------------------
# Retraction
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceIndex:
    """Which facts each document supports, and each entity's neighbours.

    Build it once with :meth:`of` and reuse it to measure many retractions
    without copying the graph.
    """

    entities_by_doc: dict[str, set[str]] = field(default_factory=dict)
    relations_by_doc: dict[str, set[RelationKey]] = field(default_factory=dict)
    neighbours: dict[str, set[str]] = field(default_factory=dict)
    relations_by_entity: dict[str, set[RelationKey]] = field(default_factory=dict)

    @classmethod
    def of(cls, graph: KnowledgeGraph) -> ProvenanceIndex:
        """Index a graph's provenance and adjacency."""
        index = cls(
            defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
        )
        for entity in graph.entities.values():
            for doc_id in entity.documents:
                index.entities_by_doc[doc_id].add(entity.name)
        for key, relation in graph.relations.items():
            for doc_id in relation.documents:
                index.relations_by_doc[doc_id].add(key)
            index.neighbours[relation.source].add(relation.target)
            index.neighbours[relation.target].add(relation.source)
            index.relations_by_entity[relation.source].add(key)
            index.relations_by_entity[relation.target].add(key)
        return index


@dataclass
class RetractionReport:
    """What retracting one or more documents changed.

    Attributes
    ----------
    doc_ids : list of str
        The retracted documents.
    entities_removed : list of str
        Entities whose every mention cited a retracted document.
    entities_touched : list of str
        Entities that lost some mentions and survive.
    relations_removed : list of tuple
        Relations whose every mention cited a retracted document.
    relations_dangling : list of tuple
        Relations that kept support but lost an endpoint, and were dropped.
    relations_touched : list of tuple
        Relations that lost some mentions and survive.
    neighbourhood : int
        Entities one hop from a removed or touched entity, before the change, that
        are not themselves removed or touched. Their retrieval context changes.
    descriptions_repaired : int
        Surviving facts whose description came from a retracted document and was
        replaced from the remaining units.
    descriptions_unverified : int
        Surviving touched facts whose description could not be checked, because no
        units were given.
    types_changed : int
        Surviving entities whose type came from a retracted document and now
        comes from the first remaining one.
    remaining_citations : int
        Mentions that still cite a retracted document afterwards. Zero when the
        retraction is complete.
    """

    doc_ids: list[str]
    entities_removed: list[str] = field(default_factory=list)
    entities_touched: list[str] = field(default_factory=list)
    relations_removed: list[RelationKey] = field(default_factory=list)
    relations_dangling: list[RelationKey] = field(default_factory=list)
    relations_touched: list[RelationKey] = field(default_factory=list)
    neighbourhood: int = 0
    descriptions_repaired: int = 0
    descriptions_unverified: int = 0
    types_changed: int = 0
    remaining_citations: int = 0

    def counts(self) -> dict[str, int]:
        """Return the report as counts, for tables and JSON."""
        return {
            "entities_removed": len(self.entities_removed),
            "entities_touched": len(self.entities_touched),
            "relations_removed": len(self.relations_removed),
            "relations_dangling": len(self.relations_dangling),
            "relations_touched": len(self.relations_touched),
            "neighbourhood": self.neighbourhood,
            "descriptions_repaired": self.descriptions_repaired,
            "descriptions_unverified": self.descriptions_unverified,
            "types_changed": self.types_changed,
            "remaining_citations": self.remaining_citations,
        }


def retraction_impact(
    graph: KnowledgeGraph, doc_ids: Iterable[str], index: ProvenanceIndex
) -> RetractionReport:
    """Measure what :func:`retract` would change, without changing the graph.

    Parameters
    ----------
    graph : KnowledgeGraph
        The graph, left unchanged.
    doc_ids : iterable of str
        Documents to retract.
    index : ProvenanceIndex
        ``ProvenanceIndex.of(graph)``.

    Returns
    -------
    RetractionReport
        The counts :func:`retract` would report, with ``remaining_citations`` and
        the description counts left at zero.
    """
    gone = set(doc_ids)
    report = RetractionReport(sorted(gone))
    entity_names = sorted(
        {name for doc in gone for name in index.entities_by_doc.get(doc, ())}
    )
    removed: set[str] = set()
    for name in entity_names:
        if all(m.provenance.doc_id in gone for m in graph.entities[name].mentions):
            removed.add(name)
            report.entities_removed.append(name)
        else:
            report.entities_touched.append(name)
    keys = sorted({key for doc in gone for key in index.relations_by_doc.get(doc, ())})
    survivors: set[RelationKey] = set()
    for key in keys:
        if all(m.provenance.doc_id in gone for m in graph.relations[key].mentions):
            report.relations_removed.append(key)
        else:
            survivors.add(key)
    for name in sorted(removed):
        for key in index.relations_by_entity.get(name, ()):
            if (
                key not in report.relations_removed
                and key not in report.relations_dangling
                and (key in survivors or key not in keys)
            ):
                report.relations_dangling.append(key)
    dangling = set(report.relations_dangling)
    report.relations_touched = sorted(survivors - dangling)
    report.relations_dangling.sort()
    changed = removed | set(report.entities_touched)
    around = {n for name in changed for n in index.neighbours.get(name, ())}
    report.neighbourhood = len(around - changed)
    return report


def retract(
    graph: KnowledgeGraph,
    doc_ids: Iterable[str],
    *,
    units: Mapping[str, KnowledgeGraph] | None = None,
    index: ProvenanceIndex | None = None,
) -> RetractionReport:
    """Remove every fact a set of documents supports, in place.

    Mentions citing a retracted document are removed. An entity or relation left
    with no mentions is dropped, and so is a relation whose endpoint was dropped.

    Parameters
    ----------
    graph : KnowledgeGraph
        Changed in place.
    doc_ids : iterable of str
        Documents to retract.
    units : mapping of str to KnowledgeGraph, optional
        Per-document extractions, from :func:`extract_units`. When given, each
        surviving fact that lost mentions takes the type and description that
        folding its remaining documents' units gives, so a retraction from a
        folded graph equals a fold of the remaining units. Without them, the type
        and description stay as they were and are counted as unverified.
    index : ProvenanceIndex, optional
        A prebuilt index of ``graph``. It is stale after this call.

    Returns
    -------
    RetractionReport
        What changed. ``remaining_citations`` is zero for a complete retraction.
    """
    gone = set(doc_ids)
    index = index or ProvenanceIndex.of(graph)
    report = retraction_impact(graph, gone, index)
    for name in report.entities_removed:
        del graph.entities[name]
    for key in report.relations_removed + report.relations_dangling:
        graph.relations.pop(key, None)
    for name in report.entities_touched:
        entity = graph.entities[name]
        entity.mentions = [
            m for m in entity.mentions if m.provenance.doc_id not in gone
        ]
        _refresh(entity, units, report, name)
    for key in report.relations_touched:
        relation = graph.relations[key]
        relation.mentions = [
            m for m in relation.mentions if m.provenance.doc_id not in gone
        ]
        _refresh(relation, units, report, key)
    report.remaining_citations = len(facts_citing(graph, gone))
    return report


def _refresh(
    fact: Entity | Relation,
    units: Mapping[str, KnowledgeGraph] | None,
    report: RetractionReport,
    key: str | RelationKey,
) -> None:
    """Recompute a surviving fact's type and description from its remaining units."""
    if units is None:
        report.descriptions_unverified += 1
        return
    found: list[Entity | Relation] = []
    for doc in fact.documents:
        unit = units.get(doc)
        if unit is None:
            continue
        hit: Entity | Relation | None = (
            unit.entities.get(key) if isinstance(key, str) else unit.relations.get(key)
        )
        if hit is not None:
            found.append(hit)
    if not found:
        report.descriptions_unverified += 1
        return
    best = found[0].description
    for hit in found[1:]:
        if len(hit.description) > len(best):
            best = hit.description
    if best != fact.description:
        fact.description = best
        report.descriptions_repaired += 1
    if (
        isinstance(fact, Entity)
        and isinstance(found[0], Entity)
        and found[0].type != fact.type
    ):
        fact.type = found[0].type
        report.types_changed += 1


def facts_citing(graph: KnowledgeGraph, doc_ids: Iterable[str]) -> list[str]:
    """Return a label for every entity or relation that still cites a document."""
    wanted = set(doc_ids)
    found = [
        f"entity {e.name!r}"
        for e in graph.entities.values()
        if wanted & set(e.documents)
    ]
    found += [
        f"relation {r.source} -{r.type}-> {r.target}"
        for r in graph.relations.values()
        if wanted & set(r.documents)
    ]
    return found


def copy_graph(graph: KnowledgeGraph) -> KnowledgeGraph:
    """Return a copy whose entities, relations, and mention lists are independent."""
    out = KnowledgeGraph(report=SchemaReport(**graph.report.to_dict()))
    for name, e in graph.entities.items():
        out.entities[name] = Entity(e.name, e.type, e.description, list(e.mentions))
    for key, r in graph.relations.items():
        out.relations[key] = Relation(
            r.source, r.target, r.type, r.description, list(r.mentions)
        )
    return out


# ---------------------------------------------------------------------------
# Revision and comparison
# ---------------------------------------------------------------------------


@dataclass
class GraphDiff:
    """How two graphs over the same documents differ, by entity and relation key.

    Attributes
    ----------
    entities_added, entities_removed, entities_kept : list of str
        Entity names only in ``after``, only in ``before``, and in both.
    relations_added, relations_removed, relations_kept : list of tuple
        Relation keys, the same way.
    type_changes : list of tuple
        ``(name, before_type, after_type)`` for kept entities whose type differs.
    """

    entities_added: list[str]
    entities_removed: list[str]
    entities_kept: list[str]
    relations_added: list[RelationKey]
    relations_removed: list[RelationKey]
    relations_kept: list[RelationKey]
    type_changes: list[tuple[str, str, str]]

    @property
    def entity_jaccard(self) -> float:
        """Return shared entities over all entities, 1.0 for two empty graphs."""
        union = (
            len(self.entities_added)
            + len(self.entities_removed)
            + len(self.entities_kept)
        )
        return len(self.entities_kept) / union if union else 1.0

    @property
    def relation_jaccard(self) -> float:
        """Return shared relations over all relations, 1.0 for two empty graphs."""
        union = (
            len(self.relations_added)
            + len(self.relations_removed)
            + len(self.relations_kept)
        )
        return len(self.relations_kept) / union if union else 1.0

    def counts(self) -> dict[str, Any]:
        """Return the diff as counts and the two Jaccard figures."""
        return {
            "entities_added": len(self.entities_added),
            "entities_removed": len(self.entities_removed),
            "entities_kept": len(self.entities_kept),
            "relations_added": len(self.relations_added),
            "relations_removed": len(self.relations_removed),
            "relations_kept": len(self.relations_kept),
            "type_changes": len(self.type_changes),
            "entity_jaccard": round(self.entity_jaccard, 4),
            "relation_jaccard": round(self.relation_jaccard, 4),
        }


def graph_diff(before: KnowledgeGraph, after: KnowledgeGraph) -> GraphDiff:
    """Compare two graphs by entity name and by relation source, target, and type."""
    b, a = set(before.entities), set(after.entities)
    rb, ra = set(before.relations), set(after.relations)
    kept = sorted(a & b)
    return GraphDiff(
        sorted(a - b),
        sorted(b - a),
        kept,
        sorted(ra - rb),
        sorted(rb - ra),
        sorted(ra & rb),
        [
            (n, before.entities[n].type, after.entities[n].type)
            for n in kept
            if before.entities[n].type != after.entities[n].type
        ],
    )


def revise(
    graph: KnowledgeGraph,
    doc_id: str,
    new_unit: KnowledgeGraph,
    *,
    units: dict[str, KnowledgeGraph] | None = None,
) -> RetractionReport:
    """Replace one document's facts with a fresh extraction of its new version.

    The document is retracted and ``new_unit`` folded in. A fact the new version
    repeats is re-added with its new mentions, so the retraction report
    overstates what the revision removed; compare the units with
    :func:`graph_diff` for the change itself.

    Parameters
    ----------
    graph : KnowledgeGraph
        Changed in place.
    doc_id : str
        The revised document.
    new_unit : KnowledgeGraph
        The new version's extraction.
    units : dict of str to KnowledgeGraph, optional
        Per-document units. Used to repair descriptions, and updated with the new
        unit.

    Returns
    -------
    RetractionReport
        What the retraction step changed.
    """
    report = retract(graph, [doc_id], units=units)
    add_unit(graph, new_unit)
    if units is not None:
        units[doc_id] = new_unit
    return report


_WORD = re.compile(r"\S+")


def changed_spans(
    old: str, new: str
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return the character spans a word-level diff marks as changed.

    Parameters
    ----------
    old, new : str
        The two versions of a document.

    Returns
    -------
    tuple of (list, list)
        Changed ``(start, end)`` spans in ``old`` and in ``new``.
    """
    old_words = [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(old)]
    new_words = [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(new)]
    matcher = difflib.SequenceMatcher(
        None, [w for w, _, _ in old_words], [w for w, _, _ in new_words], autojunk=False
    )
    in_old: list[tuple[int, int]] = []
    in_new: list[tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:
            in_old.append((old_words[i1][1], old_words[i2 - 1][2]))
        if j2 > j1:
            in_new.append((new_words[j1][1], new_words[j2 - 1][2]))
    return in_old, in_new


def _touches(
    mentions: Sequence[Mention], spans: Sequence[tuple[int, int]], pad: int
) -> bool | None:
    """Return whether a located mention overlaps a span, or None if none is located."""
    located = [
        m.provenance for m in mentions if m.located and m.provenance.start is not None
    ]
    if not located:
        return None
    for p in located:
        start, end = int(p.start or 0), int(p.end or 0)
        if any(start < e + pad and end > s - pad for s, e in spans):
            return True
    return False


def edit_alignment(
    old_text: str,
    new_text: str,
    old_unit: KnowledgeGraph,
    new_unit: KnowledgeGraph,
    *,
    pad: int = 40,
) -> dict[str, Any]:
    """Check whether the facts a revision changed sit where the text changed.

    An added fact is explained by the edit when one of its located mentions in the
    new text falls within ``pad`` characters of a changed span, and a removed fact
    when its mention in the old text does. A fact whose quotes were never located
    cannot be checked and is counted separately.

    Parameters
    ----------
    old_text, new_text : str
        The two versions of the document.
    old_unit, new_unit : KnowledgeGraph
        Each version's extraction.
    pad : int, optional
        Characters of slack around a changed span.

    Returns
    -------
    dict
        Counts of added and removed facts that the edit explains, does not explain,
        or cannot be checked against, plus the changed character share of each
        version.
    """
    in_old, in_new = changed_spans(old_text, new_text)
    diff = graph_diff(old_unit, new_unit)
    tally: Counter[str] = Counter()
    groups: list[tuple[str, list[Sequence[Mention]], list[tuple[int, int]]]] = [
        (
            "added_entities",
            [new_unit.entities[n].mentions for n in diff.entities_added],
            in_new,
        ),
        (
            "removed_entities",
            [old_unit.entities[n].mentions for n in diff.entities_removed],
            in_old,
        ),
        (
            "added_relations",
            [new_unit.relations[k].mentions for k in diff.relations_added],
            in_new,
        ),
        (
            "removed_relations",
            [old_unit.relations[k].mentions for k in diff.relations_removed],
            in_old,
        ),
    ]
    for label, mention_lists, spans in groups:
        for mentions in mention_lists:
            verdict = _touches(mentions, spans, pad)
            tally[
                f"{label}_{'unlocated' if verdict is None else 'at_edit' if verdict else 'elsewhere'}"
            ] += 1
    return {
        **diff.counts(),
        "changed_chars_old": sum(e - s for s, e in in_old),
        "changed_chars_new": sum(e - s for s, e in in_new),
        "chars_old": len(old_text),
        "chars_new": len(new_text),
        **{
            f"{label}_{verdict}": tally[f"{label}_{verdict}"]
            for label, _, _ in groups
            for verdict in ("at_edit", "elsewhere", "unlocated")
        },
    }


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

#: Words whose trailing ``s`` is not a plural on this corpus: secure protocol
#: variants and names that are distinct from their stem.
PLURAL_EXCEPTIONS = frozenset(
    {
        "https",
        "ftps",
        "ldaps",
        "smtps",
        "imaps",
        "pop3s",
        "windows",
        "wins",
        "iops",
        "aws",
        "ntfs",
        "cifs",
        "btrfs",
    }
)

# A name folds only when every character is a letter, a digit, a space, a hyphen,
# an underscore, or a dot inside a word. Anything else (a leading dash, a sigil, a
# slash, a glob) marks a command flag, a path, or a pattern, where one character
# changes the meaning: ``rm -rf /*`` and ``rm -rf .*`` must never merge.
_FOLDABLE = re.compile(r"^[a-z0-9](?:[a-z0-9 _-]|(?<=[a-z0-9])\.(?=[a-z0-9]))*$")
_SEPARATORS = re.compile(r"[\s_-]+")


def _singular(word: str) -> str:
    if (
        len(word) > 3
        and word.endswith("s")
        and not word.endswith(("ss", "us", "is", "os"))
        and word not in PLURAL_EXCEPTIONS
    ):
        return word[:-1]
    return word


def fold_name(name: str) -> str | None:
    """Return the deduplication key for an entity name, or None to leave it alone.

    Case, separators (space, hyphen, underscore), dots inside a word, a trailing
    ``()``, and a plural ``s`` on each word are folded, so ``x.509`` meets
    ``x509`` and ``remote-server`` meets ``remote servers``. Names with any other
    punctuation are never folded.

    Parameters
    ----------
    name : str
        The entity name, lowercased as the graph stores it.

    Returns
    -------
    str or None
        The key, or None when the name holds characters that are not folded.
    """
    text = name.strip().lower()
    text = text.removesuffix("()")
    if not text or not _FOLDABLE.match(text):
        return None
    words = [_singular(w.replace(".", "")) for w in _SEPARATORS.split(text) if w]
    return "".join(words) or None


def duplicate_groups(
    graph: KnowledgeGraph, key: Callable[[str], str | None] = fold_name
) -> list[list[str]]:
    """Group entity names that share a deduplication key.

    Parameters
    ----------
    graph : KnowledgeGraph
        The graph to scan.
    key : callable, optional
        Name to key, or None to skip the name. Defaults to :func:`fold_name`.

    Returns
    -------
    list of list of str
        Groups of two or more names, largest first, each ordered so its first
        name is the one :func:`merge_entities` keeps.
    """
    found: dict[str, list[str]] = defaultdict(list)
    for name in graph.entities:
        folded = key(name)
        if folded is not None:
            found[folded].append(name)
    groups = [
        sorted(names, key=lambda n: _keep_order(graph, n))
        for names in found.values()
        if len(names) > 1
    ]
    return sorted(groups, key=lambda g: (-len(g), g[0]))


def _keep_order(graph: KnowledgeGraph, name: str) -> tuple[int, int, str]:
    return (-len(graph.entities[name].mentions), len(name), name)


@dataclass
class MergeReport:
    """What one merge changed.

    Attributes
    ----------
    kept : str
        The surviving name.
    merged : list of str
        Names folded into it.
    relations_rewired : int
        Relations whose endpoint was renamed to ``kept``.
    relations_collapsed : int
        Rewired relations that matched an existing one and were combined.
    self_loops_dropped : int
        Relations between two merged names, which became a loop and were dropped.
    type_changed : bool
        Whether the surviving entity's type changed.
    """

    kept: str
    merged: list[str]
    relations_rewired: int = 0
    relations_collapsed: int = 0
    self_loops_dropped: int = 0
    type_changed: bool = False


def merge_entities(
    graph: KnowledgeGraph, names: Sequence[str], into: str | None = None
) -> MergeReport:
    """Merge entities into one, in place, keeping every mention.

    The survivor takes the type most of the merged mentions carry (the survivor's
    own type on a tie) and the longest description. Relations are rewired to the
    survivor, combined when they now share a key, and dropped when both ends were
    merged.

    Parameters
    ----------
    graph : KnowledgeGraph
        Changed in place.
    names : sequence of str
        The entities to merge. Names not in the graph are ignored.
    into : str, optional
        The survivor. Defaults to the name with the most mentions.

    Returns
    -------
    MergeReport
        What changed.

    Raises
    ------
    ValueError
        If ``into`` is given and is not one of ``names``.
    """
    present = [n for n in dict.fromkeys(names) if n in graph.entities]
    if into is not None and into not in present:
        raise ValueError(f"{into!r} is not one of the entities being merged")
    if not present:
        return MergeReport(into or "", [])
    keep = into or min(present, key=lambda n: _keep_order(graph, n))
    others = [n for n in present if n != keep]
    report = MergeReport(keep, others)
    if not others:
        return report
    survivor = graph.entities[keep]
    weights: Counter[str] = Counter()
    for name in present:
        entity = graph.entities[name]
        weights[entity.type] += len(entity.mentions)
    best = max(weights.values())
    new_type = (
        survivor.type
        if weights[survivor.type] == best
        else sorted(t for t, w in weights.items() if w == best)[0]
    )
    report.type_changed = new_type != survivor.type
    survivor.type = new_type
    for name in others:
        entity = graph.entities.pop(name)
        survivor.mentions.extend(entity.mentions)
        if len(entity.description) > len(survivor.description):
            survivor.description = entity.description
    merged = set(others)
    for key in [k for k in graph.relations if k[0] in merged or k[1] in merged]:
        relation = graph.relations.pop(key)
        source = keep if relation.source in merged else relation.source
        target = keep if relation.target in merged else relation.target
        if source == target:
            report.self_loops_dropped += 1
            continue
        report.relations_rewired += 1
        new_key = (source, target, relation.type)
        held = graph.relations.get(new_key)
        if held is None:
            graph.relations[new_key] = Relation(
                source, target, relation.type, relation.description, relation.mentions
            )
            continue
        report.relations_collapsed += 1
        held.mentions.extend(relation.mentions)
        if len(relation.description) > len(held.description):
            held.description = relation.description
    return report


# ---------------------------------------------------------------------------
# Topology and the gate
# ---------------------------------------------------------------------------


def topology(graph: KnowledgeGraph, *, hubs: int = 10) -> dict[str, Any]:
    """Return structural statistics a maintenance change is checked against.

    Degree counts relations at either end, so it is the number of edges an
    entity's one-hop expansion walks.

    Parameters
    ----------
    graph : KnowledgeGraph
        The graph.
    hubs : int, optional
        How many highest-degree entities to list.

    Returns
    -------
    dict
        Entity and relation counts, isolated entities, single-document entities,
        connected components and the largest one's share, degree summary, the top
        hubs, and the type shares.
    """
    degree: Counter[str] = Counter()
    parent: dict[str, str] = {name: name for name in graph.entities}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for relation in graph.relations.values():
        degree[relation.source] += 1
        degree[relation.target] += 1
        if relation.source in parent and relation.target in parent:
            a, b = find(relation.source), find(relation.target)
            if a != b:
                parent[a] = b
    sizes = Counter(find(name) for name in graph.entities)
    degrees = [degree[name] for name in graph.entities]
    total = max(len(graph.entities), 1)
    entity_types = Counter(e.type for e in graph.entities.values())
    relation_types = Counter(r.type for r in graph.relations.values())
    return {
        "entities": len(graph.entities),
        "relations": len(graph.relations),
        "isolated_entities": sum(1 for d in degrees if d == 0),
        "single_document_entities": sum(
            1 for e in graph.entities.values() if len(e.documents) == 1
        ),
        "components": len(sizes),
        "largest_component_share": round(max(sizes.values(), default=0) / total, 4),
        "degree_mean": round(sum(degrees) / total, 3),
        "degree_median": median(degrees) if degrees else 0,
        "degree_max": max(degrees, default=0),
        "hubs": [[name, d] for name, d in degree.most_common(hubs)],
        "entity_type_shares": {
            t: round(c / total, 4) for t, c in sorted(entity_types.items())
        },
        "relation_type_shares": {
            t: round(c / max(len(graph.relations), 1), 4)
            for t, c in sorted(relation_types.items())
        },
    }


def topology_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Return ``after - before`` for each numeric field of two topology results."""
    delta: dict[str, Any] = {}
    for key, value in before.items():
        if isinstance(value, (int, float)) and isinstance(after.get(key), (int, float)):
            delta[key] = round(after[key] - value, 4)
    for key in ("entity_type_shares", "relation_type_shares"):
        names = set(before.get(key, {})) | set(after.get(key, {}))
        delta[key] = {
            n: round(
                after.get(key, {}).get(n, 0.0) - before.get(key, {}).get(n, 0.0), 4
            )
            for n in sorted(names)
        }
    return delta


@dataclass
class GateLimits:
    """Bounds a maintenance change must stay inside to pass the gate.

    Attributes
    ----------
    max_entity_loss_share : float
        Largest share of entities the change may remove.
    max_isolated_increase_share : float
        Largest rise in isolated entities, as a share of entities.
    max_largest_component_drop : float
        Largest fall in the largest component's share.
    max_type_share_shift : float
        Largest change in any entity or relation type's share.
    """

    max_entity_loss_share: float = 0.01
    max_isolated_increase_share: float = 0.005
    max_largest_component_drop: float = 0.01
    max_type_share_shift: float = 0.01


@dataclass
class GateCheck:
    """One gate check: its name, whether it passed, and what it found."""

    name: str
    passed: bool
    detail: str


@dataclass
class GateResult:
    """Every gate check, and whether all of them passed."""

    checks: list[GateCheck]

    @property
    def passed(self) -> bool:
        """Return True when every check passed."""
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {"passed": self.passed, "checks": [c.__dict__ for c in self.checks]}


class EntityLookup:
    """Cite documents for a question through the entities it names, for smoke tests.

    An entity is linked when its whole name occurs in the question, and each
    block that mentions a linked entity scores the entity's inverse document
    frequency, so a rare entity outweighs ``linux``. The lookup reads the graph it
    was built from, so a lookup built before a change and used after it is a
    stale serving index, which the gate's smoke questions catch.

    Parameters
    ----------
    graph : KnowledgeGraph
        The graph to read. Only its state at construction is used.
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        documents = {d for e in graph.entities.values() for d in e.documents}
        total = max(len(documents), 1)
        self._names = sorted(
            (n for n in graph.entities if len(n) >= 2), key=lambda n: -len(n)
        )
        self._weights = {
            n: math.log(total / max(len(graph.entities[n].documents), 1))
            for n in self._names
        }
        self._blocks = {
            n: sorted(
                {
                    (m.provenance.chunk_id or m.provenance.doc_id, m.provenance.doc_id)
                    for m in graph.entities[n].mentions
                }
            )
            for n in self._names
        }

    def link(self, question: str) -> list[str]:
        """Return the entities whose whole name is in the question, longest first."""
        text = " " + " ".join(question.lower().split()) + " "
        return [
            n
            for n in self._names
            if n in text and re.search(rf"(?<![\w.~/-]){re.escape(n)}(?![\w~/-])", text)
        ]

    def cite(self, question: str, k: int = 10) -> list[str]:
        """Return the documents of the ``k`` highest-scoring blocks, in rank order."""
        scores: dict[tuple[str, str], float] = defaultdict(float)
        for name in self.link(question):
            for block in self._blocks[name]:
                scores[block] += self._weights[name]
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [doc_id for (_, doc_id), _ in ranked]


def maintenance_gate(
    after: KnowledgeGraph,
    *,
    before_topology: Mapping[str, Any],
    retracted: Iterable[str] = (),
    smoke: Mapping[str, Callable[[], Sequence[str]]] | None = None,
    smoke_before: Mapping[str, Sequence[str]] | None = None,
    expected_entity_removals: int = 0,
    limits: GateLimits | None = None,
    after_topology: Mapping[str, Any] | None = None,
) -> GateResult:
    """Check a maintained graph before it replaces the one in service.

    Parameters
    ----------
    after : KnowledgeGraph
        The graph after the change.
    before_topology : mapping
        :func:`topology` of the graph before the change.
    retracted : iterable of str
        Documents the change retracted. No fact may cite one.
    smoke : mapping of str to callable, optional
        Question to a function returning the document ids its answer cites after
        the change. No cited document may be a retracted one, and each question
        must cite something.
    smoke_before : mapping of str to sequence of str, optional
        What each smoke question cited before the change. A question that cited
        no retracted document then cannot detect a failed retraction, and fails
        as not armed.
    expected_entity_removals : int, optional
        Entities the change is meant to remove, such as the names a merge folds
        away. Only removals beyond these count against the entity-loss limit.
    limits : GateLimits, optional
        Topology bounds.
    after_topology : mapping, optional
        :func:`topology` of ``after``, when already computed.

    Returns
    -------
    GateResult
        Every check with its finding.
    """
    limits = limits or GateLimits()
    gone = set(retracted)
    checks: list[GateCheck] = []
    problems = after.validate()
    first = f", first: {problems[0]}" if problems else ""
    checks.append(
        GateCheck(
            "schema: validate()", not problems, f"{len(problems)} problems{first}"
        )
    )
    citing = facts_citing(after, gone)
    first = f", first: {citing[0]}" if citing else ""
    checks.append(
        GateCheck(
            "provenance: no fact cites a retracted document",
            not citing,
            f"{len(citing)} facts{first}",
        )
    )
    now = after_topology or topology(after)
    delta = topology_delta(before_topology, now)
    entities = max(int(before_topology["entities"]), 1)
    loss = (-delta["entities"] - expected_entity_removals) / entities
    checks.append(
        GateCheck(
            "topology: entity loss",
            loss <= limits.max_entity_loss_share,
            f"{-int(delta['entities']):,} removed, {expected_entity_removals:,} expected, {loss:.3%} unexpected",
        )
    )
    isolated = delta["isolated_entities"] / entities
    checks.append(
        GateCheck(
            "topology: isolated entities",
            isolated <= limits.max_isolated_increase_share,
            f"{int(delta['isolated_entities']):+,} isolated",
        )
    )
    drop = -delta["largest_component_share"]
    checks.append(
        GateCheck(
            "topology: largest component",
            drop <= limits.max_largest_component_drop,
            f"share {before_topology['largest_component_share']} to {now['largest_component_share']}",
        )
    )
    shift = max(
        [
            abs(v)
            for key in ("entity_type_shares", "relation_type_shares")
            for v in delta[key].values()
        ]
        or [0.0]
    )
    checks.append(
        GateCheck(
            "topology: type shares",
            shift <= limits.max_type_share_shift,
            f"largest shift {shift:.4f}",
        )
    )
    for question, cite in (smoke or {}).items():
        cited = list(cite())
        bad = sorted(set(cited) & gone)
        armed = smoke_before is None or bool(set(smoke_before.get(question, ())) & gone)
        notes = [f"cites {len(set(cited))} documents"]
        if bad:
            notes.append(f"retracted: {', '.join(bad)}")
        if not cited:
            notes.append("none")
        if not armed:
            notes.append("not armed: cited no retracted document before the change")
        checks.append(
            GateCheck(
                f"smoke: {question}",
                bool(cited) and not bad and armed,
                ", ".join(notes),
            )
        )
    return GateResult(checks)


# ---------------------------------------------------------------------------
# Stack Exchange post history
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Revision:
    """One full-text revision of a post's body or title.

    Parameters
    ----------
    post_id : int
        The question or answer identifier.
    kind : int
        ``PostHistoryTypeId``. 2, 5, and 8 are bodies, 1, 4, and 7 titles.
    created : str
        When the revision was made, as the dump records it.
    guid : str
        ``RevisionGUID``, shared by the rows of one edit.
    text : str
        The full Markdown source (body) or title text.
    """

    post_id: int
    kind: int
    created: str
    guid: str
    text: str


def extract_member(
    dump: Path, dest: Path, member: str, *, program: str | None = None
) -> Path:
    """Extract one file from a Stack Exchange ``.7z`` dump, once.

    Parameters
    ----------
    dump : Path
        The archive.
    dest : Path
        Directory to extract into.
    member : str
        The archive member, for example ``PostHistory.xml``.
    program : str, optional
        Path to a ``7z`` binary. Defaults to ``7z`` or ``7zr`` on the path.

    Returns
    -------
    Path
        The extracted file.

    Raises
    ------
    RuntimeError
        If no ``7z`` program is found.
    """
    target = dest / member
    if target.exists():
        return target
    binary = program or shutil.which("7z") or shutil.which("7zr")
    if binary is None:
        raise RuntimeError("extracting the dump needs the 7z program (p7zip or 7zip)")
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [binary, "x", "-y", f"-o{dest}", str(dump), member],
        check=True,
        stdout=subprocess.DEVNULL,
    )  # noqa: S603
    return target


def read_post_history(path: Path, post_ids: Iterable[int]) -> dict[int, list[Revision]]:
    """Read the body and title revisions of the given posts from ``PostHistory.xml``.

    The file is streamed, so memory stays flat on the 2 GB ServerFault history.

    Parameters
    ----------
    path : Path
        The extracted ``PostHistory.xml``.
    post_ids : iterable of int
        Questions and answers to keep.

    Returns
    -------
    dict of int to list of Revision
        Each post's revisions in file order, which is chronological.
    """
    wanted = set(post_ids)
    kinds = set(BODY_REVISION_TYPES) | set(TITLE_REVISION_TYPES)
    found: dict[int, list[Revision]] = defaultdict(list)
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != "row":
            continue
        attrib = element.attrib
        post_id = int(attrib.get("PostId", 0))
        kind = int(attrib.get("PostHistoryTypeId", 0))
        if post_id in wanted and kind in kinds:
            found[post_id].append(
                Revision(
                    post_id,
                    kind,
                    attrib.get("CreationDate", ""),
                    attrib.get("RevisionGUID", ""),
                    attrib.get("Text", ""),
                )
            )
        element.clear()
    return dict(found)


def body_revisions(revisions: Sequence[Revision]) -> list[Revision]:
    """Return a post's body revisions, oldest first."""
    return [r for r in revisions if r.kind in BODY_REVISION_TYPES]


def render_markdown(source: str) -> str:
    """Render a revision's Markdown to the plain text the corpus loader produces.

    ``Posts.xml`` holds rendered HTML and ``PostHistory.xml`` holds Markdown, so a
    revision is rendered with CommonMark and passed through the loader's own
    ``strip_html``. On 300 ServerFault posts the rendered latest revision matched
    the loader's text exactly for 67% and at a similarity above 0.98 for 97%.

    Raises
    ------
    ImportError
        If ``markdown-it-py`` is not installed.
    """
    from ..corpora.serverfault import strip_html  # noqa: PLC0415

    return strip_html(_markdown().render(source))


@lru_cache(maxsize=1)
def _markdown() -> Any:
    from markdown_it import MarkdownIt  # noqa: PLC0415

    return MarkdownIt("commonmark")


__all__ = [
    "BODY_REVISION_TYPES",
    "PLURAL_EXCEPTIONS",
    "TITLE_REVISION_TYPES",
    "EntityLookup",
    "GateCheck",
    "GateLimits",
    "GateResult",
    "GraphDiff",
    "MergeReport",
    "ProvenanceIndex",
    "RetractionReport",
    "Revision",
    "add_unit",
    "body_revisions",
    "changed_spans",
    "copy_graph",
    "duplicate_groups",
    "edit_alignment",
    "extract_member",
    "extract_units",
    "facts_citing",
    "fold_name",
    "fold_units",
    "graph_diff",
    "maintenance_gate",
    "merge_entities",
    "read_post_history",
    "render_markdown",
    "retract",
    "retraction_impact",
    "revise",
    "topology",
    "topology_delta",
]
