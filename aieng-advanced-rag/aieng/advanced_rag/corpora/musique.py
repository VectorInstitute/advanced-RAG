"""MuSiQue as HippoRAG 2 packages it: 1,000 multi-hop questions over 11,656 passages.

The packaging on the Hugging Face hub (``osunlp/HippoRAG_2``) holds a question
file and a corpus file. Every question lists about twenty candidate paragraphs
and marks the ones that support the answer, and the corpus is exactly the union
of those paragraphs, so a question's supporting passages are corpus passages and
retrieval over the corpus can be judged against them.

Questions come in three hop counts: 518 with two supporting paragraphs, 316 with
three, and 166 with four. The hop count is the number of supporting paragraphs
and also the prefix of the question's identifier.

Both files are addressed by the hub revision they were read at and by a SHA-256
digest, so a changed upstream file fails here instead of moving a figure.

The pool is the larger corpus of the scale ladder: every distinct paragraph in
the six question files of the authors' MuSiQue v1.0 release, 139,416 in all,
which is the corpus IRCoT (Trivedi et al., ACL 2023) indexes for MuSiQue. The
11,656 packaged passages are a subset of it and keep their identifiers in the
pool, so a question's supporting passages are the same identifiers at both
sizes. :func:`fetch_pool` builds it from the release archive, which is pinned by
SHA-256 together with each of the six members it reads.

The dataset's own repository states CC BY 4.0. The packaging card states no
licence, so this module is not registered in ``corpora.SOURCES``. The files are fetched
into the local corpus cache and never committed.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sources import (
    ChecksumMismatchError,
    Licence,
    PinnedSource,
    cache_dir,
    fetch,
    verify,
)


#: The revision of ``osunlp/HippoRAG_2`` both files were read at.
MUSIQUE_REVISION = "5ec05b38deecc3318bb432c69865959c56058990"

_RESOLVE = (
    f"https://huggingface.co/datasets/osunlp/HippoRAG_2/resolve/{MUSIQUE_REVISION}"
)

MUSIQUE_LICENCE = Licence(
    name="CC BY 4.0",
    url="https://creativecommons.org/licenses/by/4.0/",
    permits_commercial_use=True,
    attribution=(
        "MuSiQue: Multihop Questions via Single-hop Question Composition, Trivedi, "
        "Balasubramanian, Khot, and Sabharwal, TACL 2022. "
        "https://github.com/StonyBrookNLP/musique"
    ),
    notes=(
        "The MuSiQue repository's README states 'MuSiQue is distributed under a CC "
        "BY 4.0 License'. The osunlp/HippoRAG_2 dataset card, which repackages "
        "the questions and passages, states no licence."
    ),
)

#: The 1,000 questions, each with its candidate paragraphs and decomposition.
MUSIQUE_QUESTIONS = PinnedSource(
    name="musique-questions",
    url=f"{_RESOLVE}/musique.json",
    sha256="98ed4e21d3076532f6388d42320fb809599c63a0d8dffca8ece5e41922be6b46",
    licence=MUSIQUE_LICENCE,
)

#: The 11,656 distinct passages, in the order the packaging lists them.
MUSIQUE_CORPUS = PinnedSource(
    name="musique-corpus",
    url=f"{_RESOLVE}/musique_corpus.json",
    sha256="73157a03ce3f0b1a5673dd5dc12bb970c24976dbffc688af9eecdd758c97ffcb",
    licence=MUSIQUE_LICENCE,
)

#: Google Drive file of the authors' v1.0 release, as the MuSiQue README at
#: commit ``922ac98f`` links it. Drive keeps no revisions a URL can name, so
#: the archive's SHA-256 is the pin.
MUSIQUE_RELEASE_FILE_ID = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"

#: The release archive, ``musique_v1.0.zip``: 272,049,578 bytes.
MUSIQUE_RELEASE = PinnedSource(
    name="musique-v1.0.zip",
    url=(
        "https://drive.usercontent.google.com/download"
        f"?id={MUSIQUE_RELEASE_FILE_ID}&export=download&confirm=t"
    ),
    sha256="98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd",
    licence=MUSIQUE_LICENCE,
)

#: The six question files the pool reads, in the order IRCoT reads them, each
#: with the SHA-256 of the member's bytes.
POOL_MEMBERS: Mapping[str, str] = {
    "data/musique_ans_v1.0_dev.jsonl": (
        "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"
    ),
    "data/musique_ans_v1.0_test.jsonl": (
        "92ee3067957b8ebce885baa71156a7c0c29f3bc72cca04510aef506911dd768f"
    ),
    "data/musique_ans_v1.0_train.jsonl": (
        "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a"
    ),
    "data/musique_full_v1.0_dev.jsonl": (
        "8cab31d56a3a1c4ef491b205a8dab3f1ac9c66e472098c6cf1de4e20294f7a4a"
    ),
    "data/musique_full_v1.0_test.jsonl": (
        "01d7b0f752bdc1fc98856d605560e021a89be903baeb9daff11adeb7fac61328"
    ),
    "data/musique_full_v1.0_train.jsonl": (
        "b1cd998f7e0e2838d6fda024e4ad1eb0e7fc3edefdadb0bd9b5b10b0907f2034"
    ),
}

#: Distinct paragraphs across the six files.
POOL_SIZE = 139_416

#: Passages in the packaged corpus, which open the pool in their own order.
SUBSET_SIZE = 11_656

#: SHA-256 of the built pool as :func:`pool_bytes` serialises it.
POOL_SHA256 = "777bd36d26d28b232c0bdc5e6587c93551d1b101447babd192c2127138fa959d"

#: Where :func:`fetch_pool` caches the built pool.
POOL_FILENAME = "musique-pool-139416.json"

#: Questions per hop count in the packaging.
HOP_COUNTS = {2: 518, 3: 316, 4: 166}

#: The stratified test: questions drawn per hop count, 300 in all.
TEST_PER_HOP = {2: 150, 3: 75, 4: 75}

#: Seed for the stratified draw. Changing it changes which questions are tested.
TEST_SEED = 20261006


@dataclass(frozen=True)
class Passage:
    """One corpus passage.

    Parameters
    ----------
    passage_id : str
        ``musique::`` followed by the passage's zero-based position in the corpus
        file, padded to five digits. Stable for a given revision.
    title : str
        The Wikipedia article title.
    text : str
        The passage text.
    """

    passage_id: str
    title: str
    text: str


@dataclass(frozen=True)
class SubQuestion:
    """One step of a question's decomposition.

    Parameters
    ----------
    question : str
        The sub-question. ``#1`` and later refer to earlier answers.
    answer : str
        Its answer.
    support_passage_id : str
        The corpus passage that supports it.
    step_id : int or None, optional
        Original single-hop question identifier, not its position in the list.
    depends_on : tuple of int, optional
        Original identifiers of earlier steps referenced by this question.
    """

    question: str
    answer: str
    support_passage_id: str
    step_id: int | None = None
    depends_on: tuple[int, ...] = ()


@dataclass(frozen=True)
class HopGraph:
    """A question's gold decomposition graph, for evaluation only.

    Parameters
    ----------
    nodes : tuple of SubQuestion
        Steps in the packaging's topological order, with answers and supports.
    edges : tuple of tuple of int and int
        Directed pairs of original step identifiers, from prerequisite to user
        of its answer. ``#1`` refers to the first node, not to step ID 1.
    """

    nodes: tuple[SubQuestion, ...]
    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class MusiqueQuestion:
    """One MuSiQue question and its judgments.

    Parameters
    ----------
    question_id : str
        The dataset's identifier, such as ``2hop__13548_13529``.
    question : str
        The composed multi-hop question.
    answer : str
        The reference answer.
    answer_aliases : tuple of str
        Other accepted forms of the answer. Often empty.
    hops : int
        Supporting paragraphs the question needs: 2, 3, or 4.
    supporting_ids : tuple of str
        The corpus passages that support the answer, in decomposition order.
    decomposition : tuple of SubQuestion
        The single-hop steps the question composes.
    """

    question_id: str
    question: str
    answer: str
    answer_aliases: tuple[str, ...]
    hops: int
    supporting_ids: tuple[str, ...]
    decomposition: tuple[SubQuestion, ...]

    @property
    def hop_graph(self) -> HopGraph:
        """Return gold nodes and prerequisite-to-dependent edges."""
        return HopGraph(
            nodes=self.decomposition,
            edges=tuple(
                (dependency, step.step_id)
                for step in self.decomposition
                if step.step_id is not None
                for dependency in step.depends_on
            ),
        )

    @property
    def is_answerable(self) -> bool:
        """Return True: the packaging holds only answerable questions."""
        return True

    @property
    def kind(self) -> str:
        """Return the composition pattern, such as ``3hop1`` or ``4hop2``."""
        return self.question_id.split("__")[0]


@dataclass(frozen=True)
class MusiquePaths:
    """Where the verified files are.

    Parameters
    ----------
    questions : Path
        The question file.
    corpus : Path
        The corpus file.
    """

    questions: Path
    corpus: Path


def passage_id(position: int) -> str:
    """Return the identifier of the passage at a zero-based corpus position."""
    return f"musique::{position:05d}"


def parse_corpus(records: Iterable[Mapping[str, Any]]) -> list[Passage]:
    """Turn corpus records into passages with stable identifiers.

    Raises
    ------
    ValueError
        If two records have the same title and text, which would make a
        question's paragraphs ambiguous.
    """
    passages = [
        Passage(passage_id(i), str(r["title"]), str(r["text"]))
        for i, r in enumerate(records)
    ]
    keys = {(p.title, p.text) for p in passages}
    if len(keys) != len(passages):
        raise ValueError("the corpus holds a duplicate (title, text) passage")
    return passages


def parse_questions(
    records: Iterable[Mapping[str, Any]], passages: Sequence[Passage]
) -> list[MusiqueQuestion]:
    """Turn question records into questions judged against ``passages``.

    Parameters
    ----------
    records : iterable of mapping
        Question records as the packaging stores them.
    passages : sequence of Passage
        The corpus, from :func:`parse_corpus`.

    Returns
    -------
    list of MusiqueQuestion
        In file order.

    Raises
    ------
    ValueError
        If identifiers, supports, hop counts, or decomposition references are
        invalid. References must point to earlier steps, preventing cycles.
    """
    by_text = {(p.title, p.text): p.passage_id for p in passages}
    questions = []
    seen_ids: set[str] = set()
    for record in records:
        qid = str(record["id"])
        if qid in seen_ids:
            raise ValueError(f"{qid}: duplicate question identifier")
        seen_ids.add(qid)
        match = re.fullmatch(r"([234])hop\d*__\d+(?:_\d+)*", qid)
        if match is None:
            raise ValueError(f"{qid}: invalid MuSiQue question identifier")
        hops = int(match[1])
        if record.get("answerable") is not True:
            raise ValueError(f"{qid}: expected an answerable question")
        by_idx: dict[int, str] = {}
        supporting: list[str] = []
        for paragraph in record["paragraphs"]:
            idx = int(paragraph["idx"])
            if idx in by_idx:
                raise ValueError(f"{qid}: duplicate paragraph index {idx}")
            key = (str(paragraph["title"]), str(paragraph["paragraph_text"]))
            if key not in by_text:
                raise ValueError(
                    f"{qid}: paragraph {paragraph['idx']} is not in the corpus"
                )
            by_idx[idx] = by_text[key]
            if paragraph["is_supporting"]:
                supporting.append(by_text[key])
        if len(supporting) != hops:
            raise ValueError(
                f"{qid}: {len(supporting)} supporting paragraphs for a {hops}-hop question"
            )
        if len(set(supporting)) != hops:
            raise ValueError(f"{qid}: duplicate supporting passage")
        steps = _parse_decomposition(record, by_idx, hops)
        if {step.support_passage_id for step in steps} != set(supporting):
            raise ValueError(
                f"{qid}: the decomposition and the marked paragraphs disagree"
            )
        questions.append(
            MusiqueQuestion(
                question_id=qid,
                question=str(record["question"]),
                answer=str(record["answer"]),
                answer_aliases=tuple(
                    str(a) for a in record.get("answer_aliases") or ()
                ),
                hops=hops,
                supporting_ids=tuple(step.support_passage_id for step in steps),
                decomposition=steps,
            )
        )
    return questions


def _parse_decomposition(
    record: Mapping[str, Any], by_idx: Mapping[int, str], hops: int
) -> tuple[SubQuestion, ...]:
    qid = str(record["id"])
    raw_steps = record["question_decomposition"]
    if len(raw_steps) != hops:
        raise ValueError(f"{qid}: decomposition length does not match {hops} hops")
    steps: list[SubQuestion] = []
    seen: set[int] = set()
    for position, raw in enumerate(raw_steps, start=1):
        step_id = int(raw["id"])
        if step_id in seen:
            raise ValueError(f"{qid}: duplicate decomposition step ID {step_id}")
        seen.add(step_id)
        support_idx = int(raw["paragraph_support_idx"])
        if support_idx not in by_idx:
            raise ValueError(f"{qid}: missing supporting paragraph index {support_idx}")
        text = str(raw["question"])
        references = tuple(
            dict.fromkeys(int(ref) for ref in re.findall(r"#(\d+)\b", text))
        )
        if any(ref < 1 or ref >= position for ref in references):
            raise ValueError(f"{qid}: step {position} references a non-earlier step")
        dependencies = tuple(int(raw_steps[ref - 1]["id"]) for ref in references)
        steps.append(
            SubQuestion(
                question=text,
                answer=str(raw["answer"]),
                support_passage_id=by_idx[support_idx],
                step_id=step_id,
                depends_on=dependencies,
            )
        )
    return tuple(steps)


def stratified_test_set(
    questions: Sequence[MusiqueQuestion],
    per_hop: Mapping[int, int] = TEST_PER_HOP,
    seed: int = TEST_SEED,
) -> list[MusiqueQuestion]:
    """Draw the same number of questions per hop count, reproducibly.

    Parameters
    ----------
    questions : sequence of MusiqueQuestion
        The full question set.
    per_hop : mapping, optional
        Hop count to how many questions to draw. Defaults to 150, 75, and 75.
    seed : int, optional
        Seed for the draw.

    Returns
    -------
    list of MusiqueQuestion
        Questions grouped by ascending hop count, each group in file order.

    Raises
    ------
    ValueError
        If a hop count has fewer questions than requested.
    """
    chosen: list[MusiqueQuestion] = []
    rng = random.Random(seed)
    for hops in sorted(per_hop):
        pool = [q for q in questions if q.hops == hops]
        if len(pool) < per_hop[hops]:
            raise ValueError(
                f"only {len(pool)} {hops}-hop questions, {per_hop[hops]} requested"
            )
        picked = set(rng.sample(range(len(pool)), per_hop[hops]))
        chosen.extend(q for i, q in enumerate(pool) if i in picked)
    return chosen


def fetch_musique(*, dest: Path | None = None) -> MusiquePaths:
    """Download and verify the two files into the corpus cache.

    Raises
    ------
    ChecksumMismatchError
        If either file differs from its pin.
    """
    root = cache_dir(dest)
    return MusiquePaths(
        questions=fetch(MUSIQUE_QUESTIONS, dest=root),
        corpus=fetch(MUSIQUE_CORPUS, dest=root),
    )


def load_musique(
    paths: MusiquePaths | None = None,
) -> tuple[list[Passage], list[MusiqueQuestion]]:
    """Load the corpus and the 1,000 questions, fetching them when needed.

    Parameters
    ----------
    paths : MusiquePaths or None, optional
        Files already on disk, checked against both pinned digests before
        parsing. When omitted the pinned files are fetched into the corpus cache.

    Returns
    -------
    tuple of (list of Passage, list of MusiqueQuestion)
        The 11,656 passages and the 1,000 questions.
    """
    files = paths if paths is not None else fetch_musique()
    corpus_bytes = Path(files.corpus).read_bytes()
    question_bytes = Path(files.questions).read_bytes()
    verify(corpus_bytes, MUSIQUE_CORPUS.sha256, what=str(files.corpus))
    verify(question_bytes, MUSIQUE_QUESTIONS.sha256, what=str(files.questions))
    passages = parse_corpus(json.loads(corpus_bytes))
    questions = parse_questions(json.loads(question_bytes), passages)
    return passages, questions


def _member_paragraphs(
    archive: zipfile.ZipFile, member: str, expected_sha256: str
) -> Iterable[tuple[str, str]]:
    """Return a member's (title, text) pairs once the member's digest checks.

    The member is streamed line by line rather than read whole, because the
    largest is about 477 MB.
    """
    digest = hashlib.sha256()
    pairs: list[tuple[str, str]] = []
    with archive.open(member) as handle:
        for raw in handle:
            digest.update(raw)
            line = raw.strip()
            if not line:
                continue
            for paragraph in json.loads(line)["paragraphs"]:
                pairs.append(
                    (str(paragraph["title"]), str(paragraph["paragraph_text"]))
                )
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ChecksumMismatchError(
            f"{member}: expected sha256 {expected_sha256}, got {actual}. "
            "The release changed. Do not use it until CORPUS.md is updated."
        )
    return pairs


def build_pool(
    release: Path,
    subset: Sequence[Passage],
    *,
    members: Mapping[str, str] = POOL_MEMBERS,
    expected_size: int | None = POOL_SIZE,
) -> list[Passage]:
    """Build the pooled corpus from the release archive.

    The packaged passages come first, in their own order and with their own
    identifiers. Every other distinct paragraph follows in the order the
    members list it, numbered on from the subset's last position. Paragraphs
    are distinct by title and text, which here gives the same count as IRCoT's
    rule of hashing the title and text joined by a space.

    Parameters
    ----------
    release : Path
        The verified ``musique_v1.0.zip``.
    subset : sequence of Passage
        The packaged corpus, from :func:`load_musique`.
    members : mapping, optional
        Member path to SHA-256, in reading order.
    expected_size : int or None, optional
        Paragraph count the pool must have. ``None`` skips the check.

    Returns
    -------
    list of Passage
        The pool, ``musique::00000`` onward.

    Raises
    ------
    ChecksumMismatchError
        If a member differs from its pin.
    ValueError
        If a packaged passage is missing from the release, or the pool's size
        differs from ``expected_size``.
    """
    pool = list(subset)
    seen = {(p.title, p.text) for p in pool}
    in_release: set[tuple[str, str]] = set()
    with zipfile.ZipFile(release) as archive:
        for member, sha256 in members.items():
            for pair in _member_paragraphs(archive, member, sha256):
                in_release.add(pair)
                if pair not in seen:
                    seen.add(pair)
                    pool.append(Passage(passage_id(len(pool)), *pair))
    missing = sum(1 for p in subset if (p.title, p.text) not in in_release)
    if missing:
        raise ValueError(f"{missing} packaged passages are not in the release")
    if expected_size is not None and len(pool) != expected_size:
        raise ValueError(
            f"the pool holds {len(pool)} paragraphs, expected {expected_size}"
        )
    return pool


def pool_bytes(passages: Sequence[Passage]) -> bytes:
    """Serialise a pool the way :data:`POOL_SHA256` was computed."""
    return json.dumps(
        [{"title": p.title, "text": p.text} for p in passages],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def fetch_pool(
    *, dest: Path | None = None, subset_paths: MusiquePaths | None = None
) -> Path:
    """Return the cached pool file, building it from the release when needed.

    A cached pool is verified against :data:`POOL_SHA256` on every call, and a
    pool that fails is rebuilt.

    Parameters
    ----------
    dest : Path or None, optional
        Corpus cache directory.
    subset_paths : MusiquePaths or None, optional
        The packaged files, when already on disk.

    Returns
    -------
    Path
        The verified pool, a JSON list of ``title`` and ``text`` records.

    Raises
    ------
    ChecksumMismatchError
        If the archive, a member, or the built pool differs from its pin.
    """
    root = cache_dir(dest)
    target = root / POOL_FILENAME
    if target.is_file():
        try:
            verify(target.read_bytes(), POOL_SHA256, what=str(target))
        except ChecksumMismatchError:
            target.unlink()
        else:
            return target
    subset, _ = load_musique(subset_paths or fetch_musique(dest=root))
    release = fetch(MUSIQUE_RELEASE, dest=root)
    payload = pool_bytes(
        build_pool(release, subset, members=POOL_MEMBERS, expected_size=POOL_SIZE)
    )
    verify(payload, POOL_SHA256, what="the built MuSiQue pool")
    target.write_bytes(payload)
    return target


def load_pool(path: Path | None = None) -> list[Passage]:
    """Load the 139,416-passage pool, building it when needed.

    Parameters
    ----------
    path : Path or None, optional
        A pool file already on disk. It is checked for size and for the
        packaged passages at its head, not for its digest; :func:`fetch_pool`
        checks the digest.

    Returns
    -------
    list of Passage
        The pool. The first 11,656 are the packaged corpus.

    Raises
    ------
    ValueError
        If the file holds a number of passages other than :data:`POOL_SIZE`.
    """
    source = path if path is not None else fetch_pool()
    passages = parse_corpus(json.loads(Path(source).read_text(encoding="utf-8")))
    if len(passages) != POOL_SIZE:
        raise ValueError(
            f"{source} holds {len(passages)} passages, expected {POOL_SIZE}"
        )
    return passages


__all__ = [
    "HOP_COUNTS",
    "HopGraph",
    "MUSIQUE_CORPUS",
    "MUSIQUE_LICENCE",
    "MUSIQUE_QUESTIONS",
    "MUSIQUE_RELEASE",
    "MUSIQUE_RELEASE_FILE_ID",
    "MUSIQUE_REVISION",
    "POOL_FILENAME",
    "POOL_MEMBERS",
    "POOL_SHA256",
    "POOL_SIZE",
    "SUBSET_SIZE",
    "TEST_PER_HOP",
    "TEST_SEED",
    "MusiquePaths",
    "MusiqueQuestion",
    "Passage",
    "SubQuestion",
    "build_pool",
    "fetch_musique",
    "fetch_pool",
    "load_musique",
    "load_pool",
    "parse_corpus",
    "parse_questions",
    "passage_id",
    "pool_bytes",
    "stratified_test_set",
]
