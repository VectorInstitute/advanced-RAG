"""Measure how much an LLM-built knowledge graph changes between builds of one corpus.

The functions here need no model and no network. They turn a build's raw
extractions into comparable sets, compare two builds, and derive graphs that need
no model at all (a title-mention graph, a union of builds, a graph with triples
removed), so the runner and the tests share one definition of every measure.

A build is a list of :class:`Extraction` records, one per document, each holding the
triples the model returned for that document. Three builders produce them:

* ``atlas``: the triple prompt atlas-rag's HippoRAG 2 pipeline sends, with no
  schema. Each triple is a head, a free-text relation, and a tail.
* ``schema``: the same prompt with a closed list of relation types, and a fold that
  maps the model's labels onto that list and drops any relation that maps to none.
* ``lightrag``: whatever LightRAG's graph store holds after an insert.

Every comparison normalises names first (:func:`normalise_name`), so two builds that
differ only in case, punctuation, or a leading article count as agreeing.
"""

from __future__ import annotations

import random
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any


#: The relation types the closed MuSiQue schema allows. They cover the relations
#: MuSiQue's single-hop sub-questions ask about (place of birth, performer,
#: country, member of, and so on), merged into twelve.
MUSIQUE_RELATIONS = (
    "located_in",
    "part_of",
    "born_in",
    "died_in",
    "citizen_of",
    "family_of",
    "member_of",
    "employed_by",
    "educated_at",
    "created_by",
    "owned_by",
    "happened_on",
)

#: The entity kinds the closed schema names in its prompt. Atlas-rag's triple format
#: has no type field, so these guide the model and are not stored.
MUSIQUE_ENTITY_KINDS = ("Person", "Organization", "Place", "CreativeWork", "Event", "Date", "Other")

#: Surface forms a model writes for each schema relation, after :func:`label_key`.
MUSIQUE_RELATION_ALIASES: dict[str, str] = {
    **{label_name.replace("_", ""): label_name for label_name in MUSIQUE_RELATIONS},
    "locatedin": "located_in",
    "isin": "located_in",
    "country": "located_in",
    "partof": "part_of",
    "memberof": "member_of",
    "bornin": "born_in",
    "placeofbirth": "born_in",
    "diedin": "died_in",
    "placeofdeath": "died_in",
    "citizenof": "citizen_of",
    "nationality": "citizen_of",
    "spouseof": "family_of",
    "spouse": "family_of",
    "childof": "family_of",
    "parentof": "family_of",
    "familyof": "family_of",
    "employedby": "employed_by",
    "worksfor": "employed_by",
    "playsfor": "member_of",
    "educatedat": "educated_at",
    "createdby": "created_by",
    "performedby": "created_by",
    "writtenby": "created_by",
    "directedby": "created_by",
    "producedby": "created_by",
    "ownedby": "owned_by",
    "publishedby": "owned_by",
    "happenedon": "happened_on",
    "occurredon": "happened_on",
    "date": "happened_on",
}

_ARTICLES = re.compile(r"^(the|a|an)\s+")


def label_key(label: str) -> str:
    """Return a label as lowercase letters and digits only, for alias lookup."""
    return re.sub(r"[^a-z0-9]", "", label.lower())


def normalise_name(name: str) -> str:
    """Return an entity name in the form two builds are compared in.

    Lowercases, strips accents, turns punctuation into spaces, collapses runs of
    whitespace, and drops a leading article. ``"The Beatles"`` and ``"beatles"``
    compare equal; ``"Paris, France"`` and ``"Paris"`` do not.

    Parameters
    ----------
    name : str
        The name as the model wrote it.

    Returns
    -------
    str
        The normalised name, empty when nothing alphanumeric is left.
    """
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return _ARTICLES.sub("", text)


def normalise_relation(label: str) -> str:
    """Return a relation label in comparable form: lowercase words, single spaces."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(label).lower()).split())


def fold_relation(label: str, aliases: Mapping[str, str] = MUSIQUE_RELATION_ALIASES) -> str | None:
    """Return the schema relation for a model's label, or None when it maps to none."""
    return aliases.get(label_key(label))


@dataclass(frozen=True)
class Triple:
    """One normalised fact: head, relation, and tail."""

    head: str
    relation: str
    tail: str


@dataclass
class Extraction:
    """What one build extracted from one document.

    Parameters
    ----------
    doc_id : str
        The document's identifier.
    triples : list of Triple
        Normalised triples. A triple whose head or tail normalises to nothing, or
        whose head equals its tail, is left out.
    entities : dict
        Normalised entity name to the type the model gave, for builders that type
        entities. Every head and tail is also an entity, with type ``""`` when none
        was given.
    failed : bool
        Whether the document's call failed or returned nothing parseable.
    """

    doc_id: str
    triples: list[Triple] = field(default_factory=list)
    entities: dict[str, str] = field(default_factory=dict)
    failed: bool = False

    @classmethod
    def from_raw(
        cls,
        doc_id: str,
        triples: Iterable[Mapping[str, Any]],
        *,
        entities: Iterable[Mapping[str, Any]] = (),
        failed: bool = False,
        head: str = "Head",
        relation: str = "Relation",
        tail: str = "Tail",
    ) -> Extraction:
        """Build an extraction from raw dictionaries, normalising every name."""
        found = cls(doc_id, failed=failed)
        for item in entities:
            name = normalise_name(str(item.get("name", "")))
            if name:
                found.entities.setdefault(name, str(item.get("type", "")))
        for item in triples:
            if not isinstance(item, Mapping):
                continue
            h = normalise_name(str(item.get(head, "")))
            t = normalise_name(str(item.get(tail, "")))
            r = normalise_relation(str(item.get(relation, "")))
            if not h or not t or h == t:
                continue
            found.triples.append(Triple(h, r, t))
            found.entities.setdefault(h, "")
            found.entities.setdefault(t, "")
        return found

    def nodes(self) -> set[str]:
        """Return the normalised entity names."""
        return set(self.entities)

    def labelled_edges(self) -> set[tuple[str, str, str]]:
        """Return the triples as ``(head, relation, tail)``."""
        return {(t.head, t.relation, t.tail) for t in self.triples}

    def pairs(self) -> set[frozenset[str]]:
        """Return the edges with direction and label dropped."""
        return {frozenset((t.head, t.tail)) for t in self.triples}


Build = Sequence[Extraction]


def jaccard(a: set[Any], b: set[Any]) -> float:
    """Return the Jaccard index of two sets, 1.0 when both are empty."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def graph_nodes(build: Build) -> set[str]:
    """Return every normalised entity in a build, merged across documents by name."""
    return {n for e in build for n in e.nodes()}


def graph_edges(build: Build, labelled: bool = True) -> set[Any]:
    """Return every edge in a build, labelled triples or unlabelled pairs."""
    if labelled:
        return {x for e in build for x in e.labelled_edges()}
    return {x for e in build for x in e.pairs()}


def describe(values: Sequence[float]) -> dict[str, float]:
    """Return the mean, median, minimum, and maximum of a list, rounded to 4 places."""
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def compare_builds(a: Build, b: Build) -> dict[str, Any]:
    """Compare two builds of the same documents.

    Parameters
    ----------
    a, b : sequence of Extraction
        Two builds. Documents are matched by identifier, and only documents both
        builds hold are compared per document.

    Returns
    -------
    dict
        ``graph``: Jaccard of the merged node set, labelled edge set, and
        unlabelled edge set. ``per_document``: the same three, described across
        documents. ``type_agreement``: the share of entities both builds hold whose
        types agree, when both builds typed them.
    """
    by_a = {e.doc_id: e for e in a}
    by_b = {e.doc_id: e for e in b}
    shared = [d for d in by_a if d in by_b]
    per_doc: dict[str, list[float]] = {"nodes": [], "labelled_edges": [], "edges": []}
    agree = total = 0
    for doc in shared:
        x, y = by_a[doc], by_b[doc]
        per_doc["nodes"].append(jaccard(x.nodes(), y.nodes()))
        per_doc["labelled_edges"].append(jaccard(x.labelled_edges(), y.labelled_edges()))
        per_doc["edges"].append(jaccard(x.pairs(), y.pairs()))
        for name in x.entities.keys() & y.entities.keys():
            tx, ty = x.entities[name], y.entities[name]
            if tx and ty:
                total += 1
                agree += tx == ty
    return {
        "documents": len(shared),
        "graph": {
            "nodes": round(jaccard(graph_nodes(a), graph_nodes(b)), 4),
            "labelled_edges": round(jaccard(graph_edges(a), graph_edges(b)), 4),
            "edges": round(jaccard(graph_edges(a, False), graph_edges(b, False)), 4),
        },
        "per_document": {k: describe(v) for k, v in per_doc.items()},
        "type_agreement": round(agree / total, 4) if total else None,
    }


def compare_many(builds: Mapping[str, Build]) -> dict[str, Any]:
    """Compare every pair of builds and average the headline figures.

    Returns
    -------
    dict
        ``pairs``: each pair's :func:`compare_builds` result, keyed ``"a|b"``.
        ``mean``: the mean over pairs of each graph-level Jaccard and of each
        per-document mean.
    """
    pairs = {f"{x}|{y}": compare_builds(builds[x], builds[y]) for x, y in combinations(sorted(builds), 2)}
    if not pairs:
        return {"pairs": {}, "mean": {}}
    mean: dict[str, float] = {}
    for key in ("nodes", "labelled_edges", "edges"):
        mean[f"graph_{key}"] = round(statistics.fmean(p["graph"][key] for p in pairs.values()), 4)
        mean[f"per_document_{key}"] = round(statistics.fmean(p["per_document"][key]["mean"] for p in pairs.values()), 4)
    agreements = [p["type_agreement"] for p in pairs.values() if p["type_agreement"] is not None]
    if agreements:
        mean["type_agreement"] = round(statistics.fmean(agreements), 4)
    return {"pairs": pairs, "mean": mean}


def structure(build: Build) -> dict[str, Any]:
    """Return the shape of a build's merged graph: size, degrees, and hubs.

    Degree counts unlabelled edges, so two relations between the same pair count
    once. ``top_hubs`` lists the ten highest-degree entity names, which lets two
    builds' hubs be compared by overlap.
    """
    nodes = graph_nodes(build)
    pairs = graph_edges(build, labelled=False)
    degree: Counter[str] = Counter()
    for pair in pairs:
        for name in pair:
            degree[name] += 1
    degrees = [degree.get(n, 0) for n in nodes]
    relations = Counter(t.relation for e in build for t in e.triples)
    return {
        "documents": len(build),
        "failed_documents": sum(1 for e in build if e.failed),
        "empty_documents": sum(1 for e in build if not e.triples),
        "nodes": len(nodes),
        "labelled_edges": len(graph_edges(build)),
        "edges": len(pairs),
        "triples_per_document": describe([len(e.triples) for e in build]),
        "degree": describe([float(d) for d in degrees]),
        "isolated_share": round(sum(1 for d in degrees if d == 0) / len(degrees), 4) if degrees else 0.0,
        "distinct_relation_labels": len(relations),
        # Ties broken by name: most_common keeps insertion order, which follows set
        # iteration and so changed with the hash seed between runs.
        "top_hubs": [n for n, _ in sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))[:10]],
        "cross_document_nodes": _cross_document(build),
    }


def _cross_document(build: Build) -> int:
    seen: Counter[str] = Counter()
    for e in build:
        for name in e.nodes():
            seen[name] += 1
    return sum(1 for c in seen.values() if c > 1)


def degree_distance(a: Build, b: Build) -> float:
    """Return the two-sample Kolmogorov-Smirnov statistic between two degree lists."""
    da = sorted(_degrees(a))
    db = sorted(_degrees(b))
    if not da or not db:
        return 0.0
    points = sorted(set(da) | set(db))
    best = 0.0
    for x in points:
        fa = _count_le(da, x) / len(da)
        fb = _count_le(db, x) / len(db)
        best = max(best, abs(fa - fb))
    return round(best, 4)


def _degrees(build: Build) -> list[int]:
    degree: Counter[str] = Counter()
    for pair in graph_edges(build, labelled=False):
        for name in pair:
            degree[name] += 1
    return [degree.get(n, 0) for n in graph_nodes(build)]


def _count_le(values: Sequence[int], x: int) -> int:
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def hub_overlap(a: Build, b: Build) -> float:
    """Return the Jaccard of two builds' ten highest-degree entities."""
    return round(jaccard(set(structure(a)["top_hubs"]), set(structure(b)["top_hubs"])), 4)


# ---------------------------------------------------------------------------
# Graphs that need no model
# ---------------------------------------------------------------------------


def title_mention_records(passages: Sequence[tuple[str, str, str]]) -> list[dict[str, Any]]:
    """Return atlas-rag records for a graph built from titles alone, with no model.

    Each passage's article title is an entity. A passage whose text contains another
    passage's title, as a whole word and ignoring case, adds the triple
    ``(own title, "mentions", other title)``. A title shorter than four characters
    is never a mention target, because it matches inside unrelated text, but its
    passage still links to the longer titles it mentions.

    Parameters
    ----------
    passages : sequence of (str, str, str)
        ``(passage_id, title, text)``.

    Returns
    -------
    list of dict
        One record per passage, in atlas-rag's extraction format.
    """
    titles = sorted({t for _, t, _ in passages if len(t) >= 4}, key=len, reverse=True)
    patterns = {t: re.compile(r"(?<!\w)" + re.escape(t.lower()) + r"(?!\w)") for t in titles}
    records = []
    for pid, title, text in passages:
        lowered = text.lower()
        triples = [
            {"Head": title, "Relation": "mentions", "Tail": other}
            for other in titles
            if other != title and patterns[other].search(lowered)
        ]
        records.append(atlas_record(pid, f"{title}\n{text}", triples, "title-mention graph, no model"))
    return records


def atlas_record(doc_id: str, text: str, triples: Sequence[Mapping[str, str]], raw: str) -> dict[str, Any]:
    """Return one record in the format atlas-rag's extraction stage writes."""
    return {
        "id": doc_id,
        "original_text": text,
        "metadata": {"lang": "en"},
        "entity_relation_dict": [dict(t) for t in triples],
        "entity_relation_output": raw,
    }


def union_records(builds: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Merge several builds' atlas records into one, keeping every distinct triple.

    Triples are deduplicated on their exact head, relation, and tail strings, and
    records keep the first build's order.
    """
    merged: dict[str, dict[str, Any]] = {}
    for build in builds:
        for record in build:
            target = merged.setdefault(
                record["id"], atlas_record(record["id"], record["original_text"], [], "union of builds")
            )
            seen = {(t["Head"], t["Relation"], t["Tail"]) for t in target["entity_relation_dict"]}
            for triple in record["entity_relation_dict"]:
                key = (triple["Head"], triple["Relation"], triple["Tail"])
                if key not in seen:
                    seen.add(key)
                    target["entity_relation_dict"].append(dict(triple))
    return list(merged.values())


def drop_triples(records: Sequence[Mapping[str, Any]], fraction: float, seed: int = 0) -> list[dict[str, Any]]:
    """Return a copy of a build's records with a random share of triples removed.

    Each triple is removed independently with probability ``fraction``, so the
    expected labelled-edge Jaccard between the result and the input is
    ``1 - fraction``.
    """
    rng = random.Random(seed)
    out = []
    for record in records:
        kept = [dict(t) for t in record["entity_relation_dict"] if rng.random() >= fraction]
        out.append(atlas_record(record["id"], record["original_text"], kept, f"{fraction:.0%} of triples dropped"))
    return out


def extraction_from_record(record: Mapping[str, Any], failed: bool = False) -> Extraction:
    """Return the :class:`Extraction` for one atlas-rag record."""
    triples = [t for t in record.get("entity_relation_dict") or [] if isinstance(t, Mapping)]
    return Extraction.from_raw(str(record["id"]), triples, failed=failed)


_SYLLABLES = ("ka", "lo", "mi", "ren", "tu", "sha", "vel", "dor", "ny", "qua", "bi", "zor", "fen", "ga", "pi", "wen")


def invent_names(titles: Iterable[str], seed: int = 0) -> dict[str, str]:
    """Map each title to a distinct invented name with the same word count.

    The names are built from syllables no English word list holds as a whole word,
    so a model cannot recognise the entity and has to read the passage. The map
    depends only on the titles and the seed.
    """
    rng = random.Random(seed)
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for title in sorted(set(titles)):
        words = max(1, len(title.split()))
        while True:
            name = " ".join(
                "".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 3))).capitalize() for _ in range(words)
            )
            if name not in used:
                break
        used.add(name)
        mapping[title] = name
    return mapping


def rename_text(text: str, mapping: Mapping[str, str]) -> str:
    """Replace each mapped title wherever it stands as whole words, longest first."""
    for title in sorted(mapping, key=len, reverse=True):
        text = re.sub(r"(?<!\w)" + re.escape(title) + r"(?!\w)", mapping[title], text)
    return text


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SCHEMA_ADDENDUM = """

Use only these relation types, written exactly as shown: {relations}.
Entities must be one of these kinds: {kinds}. Name each entity as the text names it.
Leave out any fact that fits none of the relation types."""


def schema_instruction(base_instruction: str) -> str:
    """Return atlas-rag's triple instruction with the closed MuSiQue schema added."""
    return base_instruction.rstrip() + SCHEMA_ADDENDUM.format(
        relations=", ".join(MUSIQUE_RELATIONS), kinds=", ".join(MUSIQUE_ENTITY_KINDS)
    )


FREE_TYPES_LINES = (
    "Entity types: any short type label you choose for each entity\n"
    "Relation types: any short verb phrase you choose for each relation"
)


def free_type_prompt(schema_prompt: str, entity_types: Sequence[str], relation_types: Sequence[str]) -> str:
    """Return the schema prompt with its two type lists replaced by free choice.

    Everything else in the prompt stays: the rules, the evidence quotes, the
    twelve-item cap, and the output shape. Comparing the two isolates the schema.

    Raises
    ------
    ValueError
        If the prompt does not contain the two type lines.
    """
    lines = f"Entity types: {', '.join(entity_types)}\nRelation types: {', '.join(relation_types)}"
    if lines not in schema_prompt:
        raise ValueError("the schema prompt does not hold the type lines")
    return schema_prompt.replace(lines, FREE_TYPES_LINES)


def response_identity(a: Mapping[str, str], b: Mapping[str, str]) -> dict[str, Any]:
    """Compare two builds' raw responses to the same prompts, document by document.

    At temperature 0 two builds send byte-identical requests, so any difference
    between their responses comes from the server (batch composition, kernel
    order), not from sampling. The shared prefix shows where a pair diverges.

    Parameters
    ----------
    a, b : mapping
        Document identifier to the raw response text.

    Returns
    -------
    dict
        ``documents``: documents both builds hold. ``identical``: the share whose
        responses match exactly. ``divergence_point``: for the documents that
        differ, the length of the shared prefix as a share of the shorter
        response, described.
    """
    shared = [d for d in a if d in b]
    points: list[float] = []
    same = 0
    for doc in shared:
        x, y = a[doc], b[doc]
        if x == y:
            same += 1
            continue
        n = 0
        limit = min(len(x), len(y))
        while n < limit and x[n] == y[n]:
            n += 1
        points.append(n / limit if limit else 0.0)
    return {
        "documents": len(shared),
        "identical": round(same / len(shared), 4) if shared else None,
        "divergence_point": describe(points),
    }


# ---------------------------------------------------------------------------
# Retrieval agreement
# ---------------------------------------------------------------------------


def ranking_agreement(a: Mapping[str, Sequence[str]], b: Mapping[str, Sequence[str]], k: int = 5) -> dict[str, Any]:
    """Compare two arms' rankings question by question.

    Parameters
    ----------
    a, b : mapping
        Question identifier to ranked passage identifiers.
    k : int, optional
        Depth compared.

    Returns
    -------
    dict
        ``top_k_jaccard``: the per-question Jaccard of the top ``k`` sets, described.
        ``identical_top_k``: the share of questions whose top ``k`` sets match.
    """
    shared = [q for q in a if q in b]
    values = [jaccard(set(a[q][:k]), set(b[q][:k])) for q in shared]
    return {
        "questions": len(shared),
        "k": k,
        "top_k_jaccard": describe(values),
        "identical_top_k": round(sum(1 for v in values if v == 1.0) / len(values), 4) if values else None,
    }


__all__ = [
    "MUSIQUE_ENTITY_KINDS",
    "MUSIQUE_RELATIONS",
    "MUSIQUE_RELATION_ALIASES",
    "Extraction",
    "Triple",
    "atlas_record",
    "compare_builds",
    "compare_many",
    "degree_distance",
    "describe",
    "drop_triples",
    "extraction_from_record",
    "fold_relation",
    "free_type_prompt",
    "graph_edges",
    "graph_nodes",
    "hub_overlap",
    "jaccard",
    "label_key",
    "normalise_name",
    "normalise_relation",
    "ranking_agreement",
    "response_identity",
    "schema_instruction",
    "structure",
    "title_mention_records",
    "union_records",
]
