"""The five retrieval arms of the MuSiQue graph test, behind one small interface.

Every arm ranks the same 11,656 passages for a question and returns passage
identifiers, so one scorer reads all five and the answer is generated from the
same top passages whichever arm found them.

* ``dense``: ``BAAI/bge-small-en-v1.5`` embeddings, cosine similarity, no model call.
* ``dense-iterative``: a model splits the question into single-hop steps, answers
  each from dense retrieval, and substitutes the answer into the next, so a later
  hop can name an entity only an earlier one found. The ranking interleaves the steps.
* ``hipporag2``: atlas-rag's HippoRAG 2 over the graph built from the passages.
* ``lightrag``: LightRAG in ``mix`` mode at its default ``top_k`` of 40.
* ``hybrid``: weighted reciprocal rank fusion of dense and HippoRAG 2.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from aieng.advanced_rag.corpora.musique import Passage
from aieng.advanced_rag.graphrag.evidence import to_runs
from aieng.advanced_rag.retrieval.sparse import weighted_rrf


class Arm(Protocol):
    """Anything that ranks passage identifiers for a question."""

    name: str

    async def rank(self, question: str, k: int) -> list[str]:
        """Return up to ``k`` passage identifiers, best first."""
        ...


class DenseArm:
    """Dense retrieval over every passage, in a Qdrant collection.

    The passage matrix is cached as an ``.npy`` file, which is also the
    ``vectors.npy`` layout of an exported collection, so a cached matrix loads
    into the store without re-embedding. Ties break on passage identifier.

    Parameters
    ----------
    passages : sequence of Passage
        The corpus.
    embedder : object
        Has ``encode(texts)`` for passages and ``encode_queries(texts)`` for
        questions, returning one vector per text, as the package's embedders do.
    cache : Path or None, optional
        An ``.npy`` file for the passage matrix, read when present.
    store : QdrantStore, str, or None, optional
        Where the vectors go. The default is an in-process collection, which
        scans exactly; 11,656 passages are under qdrant-client's 20,000-point
        guidance for that mode.
    """

    name = "dense"

    def __init__(
        self,
        passages: Sequence[Passage],
        embedder: Any,
        cache: Path | None = None,
        store: Any = ":memory:",
    ) -> None:
        from aieng.advanced_rag.retrieval.stores import QdrantStore  # noqa: PLC0415

        self._embedder = embedder
        if cache is not None and cache.is_file():
            matrix = np.load(cache)
        else:
            texts = [f"{p.title}\n{p.text}" for p in passages]
            matrix = np.asarray(embedder.encode(texts), dtype=np.float32)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, matrix)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.store = store if isinstance(store, QdrantStore) else QdrantStore(store, collection="musique-dense")
        self.store.add([p.passage_id for p in passages], matrix / np.clip(norms, 1e-9, None))

    async def rank(self, question: str, k: int) -> list[str]:
        """Return the ``k`` passages most similar to the question."""
        query = np.asarray(self._embedder.encode_queries([question])[0], dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-9)
        return [passage_id for passage_id, _ in self.store.search(query, k)]


DECOMPOSE = """Break this multi-hop question into the single-hop questions needed to answer it, in order.
A later question may refer to an earlier answer as #1, #2, and so on. Use as few steps as the
question needs, at most four.

Return only JSON: {{"steps": ["...", "..."]}}

Question: {question}"""

STEP = """Answer the question using only the passages. Reply with a short answer of a few words,
or "unknown" if the passages do not say.

Passages:
{passages}

Question: {question}
Answer:"""


def parse_steps(raw: str, question: str) -> list[str]:
    """Return the decomposition steps, or the question itself when none parse."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    for candidate in (text, text[text.find("{") : text.rfind("}") + 1]):
        try:
            steps = json.loads(candidate).get("steps")
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(steps, list):
            cleaned = [str(s).strip() for s in steps if str(s).strip()]
            if cleaned:
                return cleaned[:4]
    return [question]


def substitute(step: str, answers: Sequence[str]) -> str:
    """Replace ``#n`` in a step with the n-th earlier answer when there is one."""

    def fill(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        return answers[index] if 0 <= index < len(answers) else match.group(0)

    return re.sub(r"#(\d+)", fill, step)


def interleave(rankings: Sequence[Sequence[str]], k: int) -> list[str]:
    """Merge ranked lists round-robin, skipping repeats, until ``k`` are chosen."""
    merged: list[str] = []
    seen: set[str] = set()
    for depth in range(max((len(r) for r in rankings), default=0)):
        for ranking in rankings:
            if depth < len(ranking) and ranking[depth] not in seen:
                seen.add(ranking[depth])
                merged.append(ranking[depth])
                if len(merged) == k:
                    return merged
    return merged


class IterativeDenseArm:
    """Dense retrieval driven by a decomposition that uses each step's answer.

    Parameters
    ----------
    dense : Arm
        The dense arm each step retrieves with.
    complete : callable
        Takes a prompt, returns the model's text. Called from worker threads.
    text_of : callable
        Passage identifier to passage text, to show the model a step's passages.
    step_k : int, optional
        Passages retrieved per step.
    shown : int, optional
        Passages of a step shown to the model when it answers the step.
    """

    name = "dense-iterative"

    def __init__(
        self,
        dense: Arm,
        complete: Callable[[str], str],
        text_of: Callable[[str], str],
        *,
        step_k: int = 5,
        shown: int = 3,
    ) -> None:
        self._dense = dense
        self._complete = complete
        self._text_of = text_of
        self._step_k = step_k
        self._shown = shown

    async def rank(self, question: str, k: int) -> list[str]:
        """Return up to ``k`` passages, interleaved across the steps' retrievals."""
        raw = await asyncio.to_thread(self._complete, DECOMPOSE.format(question=question))
        steps = parse_steps(raw, question)
        answers: list[str] = []
        rankings: list[list[str]] = []
        for step in steps:
            asked = substitute(step, answers)
            ids = await self._dense.rank(asked, self._step_k)
            rankings.append(ids)
            shown = "\n\n".join(self._text_of(i) for i in ids[: self._shown])
            reply = await asyncio.to_thread(self._complete, STEP.format(passages=shown, question=asked))
            answers.append(reply.strip().splitlines()[0].strip() if reply.strip() else "unknown")
        return interleave(rankings, k)


def passage_id_map(graph_file: Path, passages: Sequence[Passage]) -> dict[str, str]:
    """Map atlas-rag's passage node keys back to the corpus's passage identifiers.

    Atlas-rag keys each passage node by a hash of its own and keeps the passage text
    in the node's ``id`` attribute, so the text is what identifies the passage. Every
    passage node must map, because an unmapped retrieval would score as a miss and
    read as a method failure.

    Parameters
    ----------
    graph_file : Path
        The GraphML file atlas-rag wrote, ``kg_graphml/<keyword>_graph.graphml``.
    passages : sequence of Passage
        The corpus the graph was built from, indexed as ``title`` then ``text``.

    Returns
    -------
    dict
        Node key to passage identifier.

    Raises
    ------
    ValueError
        If a passage node's text matches no passage.
    """
    import networkx as nx  # noqa: PLC0415

    by_text = {f"{p.title}\n{p.text}": p.passage_id for p in passages}
    graph = nx.read_graphml(graph_file)
    mapping: dict[str, str] = {}
    missing = 0
    for node, data in graph.nodes(data=True):
        if data.get("type") != "passage":
            continue
        found = by_text.get(str(data.get("id", "")))
        if found is None:
            missing += 1
        else:
            mapping[str(node)] = found
    if missing:
        raise ValueError(f"{missing} passage nodes in {graph_file} match no passage text")
    return mapping


class ReplayArm:
    """Return the ranking another run stored for each question, with no retrieval.

    Answering a stored ranking on a second model separates the answering step
    from the retriever: both models read the same five passages.

    Parameters
    ----------
    name : str
        The arm's name, usually the stored arm's.
    rankings : dict
        Question text to the stored ranked passage identifiers.
    """

    def __init__(self, name: str, rankings: Mapping[str, Sequence[str]]) -> None:
        self.name = name
        self._rankings = rankings

    async def rank(self, question: str, k: int) -> list[str]:
        """Return the stored ranking for the question.

        Raises
        ------
        KeyError
            If the stored run did not rank this question.
        """
        return list(self._rankings[question])[:k]


class WithholdArm:
    """Drop chosen passages from another arm's ranking, to make a question unanswerable.

    Removing a question's supporting passages from what the reader sees turns
    an answerable question into one the passages cannot answer, while keeping
    the passages the retriever ranked next, which are on the question's topic.

    Parameters
    ----------
    arm : Arm
        The arm whose ranking is filtered.
    withheld : mapping
        Question text to the passage identifiers to drop.
    everything : bool, optional
        Drop every passage, so the reader answers from none.
    """

    def __init__(self, arm: Arm, withheld: Mapping[str, Collection[str]], *, everything: bool = False) -> None:
        self.name = getattr(arm, "name", "withheld")
        self._arm = arm
        self._withheld = withheld
        self._everything = everything

    async def rank(self, question: str, k: int) -> list[str]:
        """Return the arm's ranking without the withheld passages, ``k`` at most."""
        if self._everything:
            return []
        drop = set(self._withheld.get(question, ()))
        ranked = await self._arm.rank(question, k + len(drop))
        return [p for p in ranked if p not in drop][:k]


class SyncEvidenceArm:
    """Adapt a synchronous retriever that returns evidence into an arm.

    Parameters
    ----------
    name : str
        The arm's name.
    retriever : object
        Has ``retrieve(question, k)`` returning evidence.
    id_map : dict or None, optional
        Maps the retriever's identifiers to passage identifiers. An identifier it
        does not hold is an error, not a silent miss.
    """

    def __init__(self, name: str, retriever: Any, id_map: Mapping[str, str] | None = None) -> None:
        self.name = name
        self._retriever = retriever
        self._id_map = id_map

    async def rank(self, question: str, k: int) -> list[str]:
        """Return the retriever's passage identifiers for the question."""
        evidence = await asyncio.to_thread(self._retriever.retrieve, question, k)
        ids = to_runs(evidence)[:k]
        if self._id_map is None:
            return list(ids)
        return [self._id_map[i] for i in ids]


class LightRagArm:
    """LightRAG in ``mix`` mode at its default ``top_k`` of 40.

    The adapter sets both its entity and chunk budgets to the depth it is asked
    for, so asking for 40 retrieves at LightRAG's default and the ranking is cut
    to ``k`` afterwards.
    """

    name = "lightrag"
    DEFAULT_TOP_K = 40

    def __init__(self, retriever: Any) -> None:
        self._retriever = retriever

    async def rank(self, question: str, k: int) -> list[str]:
        """Return the first ``k`` of the passages LightRAG retrieves at ``top_k=40``."""
        evidence = await self._retriever.aretrieve(question, self.DEFAULT_TOP_K)
        return list(to_runs(evidence)[:k])


class HybridArm:
    """Weighted reciprocal rank fusion of two arms.

    Parameters
    ----------
    first, second : Arm
        The arms to fuse.
    weights : tuple of float, optional
        One weight per arm. Equal weights are plain reciprocal rank fusion.
    depth : int, optional
        How deep each arm's ranking is read before fusing.
    """

    name = "hybrid"

    def __init__(self, first: Arm, second: Arm, weights: tuple[float, float] = (1.0, 1.0), depth: int = 20) -> None:
        self._arms = (first, second)
        self._weights = weights
        self._depth = depth

    async def rank(self, question: str, k: int) -> list[str]:
        """Return the fused ranking of both arms."""
        rankings = await asyncio.gather(*(a.rank(question, self._depth) for a in self._arms))
        return [i for i, _ in weighted_rrf(rankings, self._weights, k)]


ANSWER = """Answer the question using only the passages. Reply with the answer only, a short phrase
of a few words, and no explanation. If the passages do not contain it, reply "unknown".

Passages:
{passages}

Question: {question}
Answer:"""

#: The same instruction without the abstention option, for testing whether a
#: model's gap on fixed passages is abstention rather than reading.
ANSWER_BEST_GUESS = """Answer the question using the passages. Reply with the answer only, a short phrase
of a few words, and no explanation. If the passages do not state it directly, give the most
likely answer they support.

Passages:
{passages}

Question: {question}
Answer:"""

#: Marks an answer the flagged prompt gives without direct support.
FLAG = "(unsure)"

#: A middle instruction: always answer, but mark a guess, so a caller can choose
#: afterwards whether to show it.
ANSWER_FLAGGED = """Answer the question using the passages. Reply with the answer only, a short phrase
of a few words, and no explanation. If the passages do not state it directly, give the most
likely answer they support and add "(unsure)" after it.

Passages:
{passages}

Question: {question}
Answer:"""

ANSWER_PROMPTS: Mapping[str, str] = {
    "standard": ANSWER,
    "best-guess": ANSWER_BEST_GUESS,
    "flagged": ANSWER_FLAGGED,
}

_FLAG_PATTERN = re.compile(r"\s*\(\s*unsure\s*\)\.?", re.IGNORECASE)


def split_flag(reply: str) -> tuple[str, bool]:
    """Return the answer without the unsure mark, and whether the mark was there."""
    stripped, count = _FLAG_PATTERN.subn("", reply)
    return stripped.strip(), count > 0


async def answer(
    complete: Callable[[str], str],
    question: str,
    passage_ids: Sequence[str],
    text_of: Callable[[str], str],
    template: str = ANSWER,
) -> str:
    """Generate one short answer from the given passages, the same way for every arm."""
    passages = "\n\n".join(f"[{n}] {text_of(i)}" for n, i in enumerate(passage_ids, start=1))
    reply: Awaitable[str] = asyncio.to_thread(complete, template.format(passages=passages, question=question))
    text = (await reply).strip()
    return text.splitlines()[0].strip() if text else ""
