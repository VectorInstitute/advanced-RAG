"""X2: a graph-readiness check a team runs on its own corpus before building a graph.

:func:`graph_readiness` takes documents, and optionally questions, gold
documents, a search function, and entity names from any extractor. It makes no
model call. It reports the statistics `scan.md`'s decision guide asks about:

* ``recurring_name_share``: of the candidate entity names, the share that occur in
  two or more documents. A graph adds links only where names recur.
* ``linked_document_share``: the share of documents that share at least one
  recurring name with another document.
* ``relational_question_share``: the share of questions asking for a count, a
  complete list, a comparison, or a path, by a pattern classifier. Top-k
  retrieval cannot guarantee completeness on these.
* ``bridge_question_share``: with gold documents, the share of questions where
  some gold document contains none of the names the question mentions, so it is
  reachable only through another document.
* ``dense_all_supporting_at_k``: with gold documents and a search function, the
  share of questions whose top ``k`` holds every gold document. High values leave
  a graph little headroom.

Recurrence is counted lexically: a name occurs in a document when its tokens
appear there in order, ignoring case. Names that occur in more than 5% of
documents are hubs ("server", "error") and are left out, because they link
everything and nothing.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


HUB_SHARE = 0.05
MAX_NAME_TOKENS = 5
_WORD = re.compile(r"[a-z0-9][a-z0-9._+#/-]*[a-z0-9+#]|[a-z0-9]")
_CAPITALISED = re.compile(r"\b[A-Z][\w'&.-]*(?:\s+(?:of|the|de|and|[A-Z][\w'&.-]*))*")
_CODE_LIKE = re.compile(r"(?<![\w/.-])(?=[\w./-]*[A-Za-z])[\w-]*[._/][\w./-]*\w")
_SENTENCE_STARTERS = frozenset(
    [
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "in",
        "on",
        "at",
        "for",
        "to",
        "of",
        "and",
        "or",
        "but",
        "if",
        "when",
        "while",
        "with",
        "is",
        "are",
        "was",
        "were",
        "i",
        "we",
        "you",
        "he",
        "she",
        "they",
        "my",
        "our",
        "your",
        "his",
        "her",
        "their",
        "there",
        "here",
        "what",
        "which",
        "who",
        "how",
        "why",
        "where",
        "also",
        "however",
        "after",
        "before",
        "during",
        "since",
        "because",
        "although",
    ]
)
RELATIONAL_PATTERNS = (
    r"\bhow many\b",
    r"\bnumber of\b",
    r"\bcount\b",
    r"\blist (all|every|the)\b",
    r"\b(all|every) (the )?\w+ (that|which|who|where)\b",
    r"\bwhich (posts|documents|contracts|filings|companies|servers|services|errors)\b",
    r"\b(compare|comparison|more than|less than|fewer than|most|least|highest|lowest)\b",
    r"\b(between|link|connect|path from|related to) .+ and\b",
    r"\bacross (all|the)\b",
)
_RELATIONAL = [re.compile(p, re.IGNORECASE) for p in RELATIONAL_PATTERNS]


def tokens(text: str) -> list[str]:
    """Return lower-case word tokens, keeping dots, slashes, and dashes inside one."""
    return _WORD.findall(text.lower())


def candidate_names(text: str) -> set[str]:
    """Return entity-like names from a text with no model call.

    Capitalised spans skip a leading sentence starter ("The", "When"); code-like
    tokens hold a dot, slash, or underscore ("sysctl.conf", "/etc/hosts").
    """
    found: set[str] = set()
    for match in _CAPITALISED.finditer(text):
        words = match.group(0).split()
        while words and words[0].lower() in _SENTENCE_STARTERS:
            words = words[1:]
        while words and words[-1].lower() in {"of", "the", "de", "and"}:
            words = words[:-1]
        if words and len(words) <= MAX_NAME_TOKENS and len(" ".join(words)) >= 3:
            found.add(" ".join(words))
    found.update(m.group(0) for m in _CODE_LIKE.finditer(text) if len(m.group(0)) >= 4)
    return found


def is_relational(question: str) -> bool:
    """Return whether a question asks for a count, a list, a comparison, or a path."""
    return any(p.search(question) for p in _RELATIONAL)


class NameIndex:
    """Document frequency of names, by token n-grams of each document.

    Parameters
    ----------
    documents : mapping
        Document identifier to text.
    names : iterable of str
        The names to look up; each is matched as its token sequence.
    """

    def __init__(self, documents: Mapping[str, str], names: Iterable[str]) -> None:
        wanted: dict[tuple[str, ...], str] = {}
        for name in names:
            key = tuple(tokens(name))
            if 0 < len(key) <= MAX_NAME_TOKENS:
                wanted.setdefault(key, name)
        lengths = sorted({len(k) for k in wanted})
        self.docs_of: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for doc_id, text in documents.items():
            words = tokens(text)
            for n in lengths:
                for i in range(len(words) - n + 1):
                    gram = tuple(words[i : i + n])
                    if gram in wanted:
                        self.docs_of[gram].add(doc_id)
        self.names = wanted
        self.documents = len(documents)

    def in_text(self, text: str) -> set[tuple[str, ...]]:
        """Return the indexed names whose tokens occur in a text."""
        words = tokens(text)
        found = set()
        for n in sorted({len(k) for k in self.names}):
            for i in range(len(words) - n + 1):
                gram = tuple(words[i : i + n])
                if gram in self.names:
                    found.add(gram)
        return found


@dataclass
class ReadinessReport:
    """The readiness statistics for one corpus and question set."""

    documents: int
    names: int
    hub_names: int
    recurring_name_share: float
    linked_document_share: float
    questions: int | None = None
    relational_question_share: float | None = None
    question_name_share: float | None = None
    bridge_question_share: float | None = None
    dense_all_supporting_at_k: float | None = None
    k: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)

    def reading(self) -> list[str]:
        """Return one line per statistic, read against the decision guide."""
        lines = [
            f"{self.recurring_name_share:.0%} of names recur across documents; "
            f"{self.linked_document_share:.0%} of documents share a name with another.",
        ]
        if self.relational_question_share is not None:
            lines.append(
                f"{self.relational_question_share:.0%} of questions ask for counts, lists, comparisons, or paths "
                "(top-k retrieval cannot guarantee these are complete)."
            )
        if self.bridge_question_share is not None:
            lines.append(
                f"{self.bridge_question_share:.0%} of questions need a document that names nothing in the question."
            )
        if self.dense_all_supporting_at_k is not None:
            lines.append(
                f"dense retrieval holds every gold document in its top {self.k} for "
                f"{self.dense_all_supporting_at_k:.0%} of questions (the headroom a graph can take)."
            )
        return lines


def graph_readiness(
    documents: Mapping[str, str],
    *,
    names: Iterable[str] | None = None,
    questions: Mapping[str, str] | None = None,
    gold: Mapping[str, Sequence[str]] | None = None,
    search: Callable[[str, int], Sequence[str]] | None = None,
    k: int = 10,
    hub_share: float = HUB_SHARE,
) -> ReadinessReport:
    """Return the graph-readiness statistics for a corpus, with no model call.

    Parameters
    ----------
    documents : mapping
        Document identifier to text.
    names : iterable of str, optional
        Entity names from any extractor (a graph build, an LLM pass over a
        sample). Defaults to :func:`candidate_names` over every document.
    questions : mapping, optional
        Question identifier to question text.
    gold : mapping, optional
        Question identifier to the documents that answer it.
    search : callable, optional
        ``search(question, k)`` returning ranked document identifiers.
    k : int, optional
        Depth for the dense statistic.
    hub_share : float, optional
        Names in more than this share of documents are hubs and left out.

    Returns
    -------
    ReadinessReport
    """
    if names is None:
        names = set().union(*(candidate_names(t) for t in documents.values())) if documents else set()
    index = NameIndex(documents, names)
    limit = max(2, int(hub_share * len(documents)))
    frequencies = {key: len(index.docs_of.get(key, ())) for key in index.names}
    seen = {key: f for key, f in frequencies.items() if f >= 1}
    hubs = {key for key, f in seen.items() if f > limit}
    kept = {key: f for key, f in seen.items() if key not in hubs}
    recurring = {key for key, f in kept.items() if f >= 2}
    linked: set[str] = set()
    for key in recurring:
        linked.update(index.docs_of[key])
    report = ReadinessReport(
        documents=len(documents),
        names=len(kept),
        hub_names=len(hubs),
        recurring_name_share=round(len(recurring) / len(kept), 4) if kept else 0.0,
        linked_document_share=round(len(linked) / len(documents), 4) if documents else 0.0,
    )
    if not questions:
        return report
    report.questions = len(questions)
    report.relational_question_share = round(sum(is_relational(q) for q in questions.values()) / len(questions), 4)
    named = {qid: index.in_text(text) - hubs for qid, text in questions.items()}
    report.question_name_share = round(sum(bool(v) for v in named.values()) / len(questions), 4)
    if gold:
        scored = [qid for qid in questions if gold.get(qid)]
        texts_lower = {d: set(_ngrams(tokens(documents[d]))) for qid in scored for d in gold[qid] if d in documents}
        bridges = 0
        for qid in scored:
            need = named[qid]
            if any(d in texts_lower and not (need & texts_lower[d]) for d in gold[qid]):
                bridges += 1
        report.bridge_question_share = round(bridges / len(scored), 4) if scored else None
        if search is not None and scored:
            hits = sum(set(gold[qid]) <= set(search(questions[qid], k)) for qid in scored)
            report.dense_all_supporting_at_k = round(hits / len(scored), 4)
            report.k = k
    return report


def _ngrams(words: Sequence[str]) -> Iterable[tuple[str, ...]]:
    for n in range(1, MAX_NAME_TOKENS + 1):
        for i in range(len(words) - n + 1):
            yield tuple(words[i : i + n])


def name_frequencies(documents: Mapping[str, str], names: Iterable[str]) -> Counter[str]:
    """Return how many documents each name occurs in."""
    index = NameIndex(documents, names)
    return Counter({index.names[key]: len(docs) for key, docs in index.docs_of.items()})


# ---------------------------------------------------------------- reference table


def reference_table(args: Any, *, work: Any, graph_file: Any, texts: Callable[[], dict[str, str]]) -> dict[str, Any]:
    """Compute the readiness rows for the corpora in this repository."""
    import json  # noqa: PLC0415
    import sys  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from aieng.advanced_rag.corpora.musique import load_musique, stratified_test_set  # noqa: PLC0415

    here = Path(__file__).resolve().parents[1]
    rows: dict[str, Any] = {}

    passages, all_questions = load_musique()
    m_docs = {p.passage_id: f"{p.title}\n{p.text}" for p in passages}
    m_questions = stratified_test_set(all_questions)
    committed = json.loads(
        (here.parents[1] / "graphrag_technical_support" / "outputs" / "musique_graph_test.json").read_text()
    )
    dense_rank = {r["question_id"]: r["ranked"] for r in committed["rows"]["dense"]}
    m_names = _hipporag_names(
        Path(work).parents[1] / "graphrag" / "musique" / "hipporag2" / "kg_graphml" / "musique_graph.graphml"
    )
    for label, names in (("graph names", m_names), ("regex names", None)):
        report = graph_readiness(
            m_docs,
            names=names,
            questions={q.question_id: q.question for q in m_questions},
            gold={q.question_id: q.supporting_ids for q in m_questions},
            search=_replay(dense_rank, {x.question: x.question_id for x in m_questions}),
        )
        rows[f"musique / multi-hop questions / {label}"] = report.to_dict()

    sf_texts = texts()
    sf_graph = json.loads(Path(graph_file).read_text())
    sf_names = [e["name"] for e in sf_graph["entities"]]
    sys.path.insert(0, str(here.parents[0].parent / "agent_native_retrieval"))
    from aieng.advanced_rag.corpora import serverfault  # noqa: PLC0415
    from question_sets import load_serverfault_questions  # noqa: PLC0415

    sf_questions = load_serverfault_questions(serverfault.documents(serverfault.fetch_slice()))
    search = _post_search(sf_texts, work)
    x3 = json.loads((Path(work) / "x3-full.json").read_text()) if (Path(work) / "x3-full.json").is_file() else None
    for label, names in (("graph names", sf_names), ("regex names", None)):
        report = graph_readiness(
            sf_texts,
            names=names,
            questions={q.query_id: q.question for q in sf_questions},
            gold={q.query_id: list(q.gold_doc_ids) for q in sf_questions},
            search=search,
        )
        rows[f"serverfault / support questions / {label}"] = report.to_dict()
    if x3:
        report = graph_readiness(
            sf_texts,
            names=sf_names,
            questions={q["qid"]: q["question"] for q in x3["questions"]},
        )
        rows["serverfault / X3 relational questions / graph names"] = report.to_dict()

    contracts = json.loads((Path(work).parents[1] / "corpora" / "contract-nli-test.json").read_text())
    c_docs = {str(d.get("id", i)): d.get("text", "") for i, d in enumerate(contracts["documents"])}
    hypotheses = {
        str(k): (v.get("hypothesis", "") if isinstance(v, dict) else str(v)) for k, v in contracts["labels"].items()
    }
    rows["contractnli / hypotheses / regex names"] = graph_readiness(c_docs, questions=hypotheses).to_dict()
    return {"rows": rows, "hub_share": HUB_SHARE, "k": 10}


def _replay(rankings: Mapping[str, Sequence[str]], ids: Mapping[str, str]) -> Callable[[str, int], list[str]]:
    """Return a search function that replays committed rankings by question text."""

    def search(question: str, k: int) -> list[str]:
        return list(rankings[ids[question]][:k])

    return search


def _hipporag_names(path: Any) -> list[str]:
    """Return the entity names in an atlas-rag graphml file."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    names = []
    for _, element in ET.iterparse(str(path)):
        if element.tag.endswith("node"):
            data = {x.get("key"): x.text for x in element}
            if data.get("d1") == "entity" and data.get("d0"):
                names.append(str(data["d0"]))
            element.clear()
    return names


def _post_search(texts: Mapping[str, str], work: Any) -> Callable[[str, int], list[str]]:
    import numpy as np  # noqa: PLC0415
    from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder  # noqa: PLC0415

    ids = sorted(texts)
    matrix = np.load(work / f"serverfault-posts-bge-small-{len(ids)}.npy")
    matrix = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9, None)
    embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")

    def search(query: str, k: int) -> list[str]:
        q = np.asarray(embedder.encode_queries([query])[0], dtype=np.float32)
        order = np.argsort(-(matrix @ (q / max(float(np.linalg.norm(q)), 1e-9))), kind="stable")[:k]
        return [ids[i] for i in order]

    return search
