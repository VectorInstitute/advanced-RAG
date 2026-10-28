"""The generated question record, the set it belongs to, and the funnel that built it.

Gold evidence is a list of character spans into the exact text the chunker
sees, each with the verbatim quote it covers. That one choice is what lets a
generated set be scored under any chunker: :meth:`SyntheticSet.qrels` maps the
spans onto whatever chunks it is handed with the harness's own overlap rule,
:func:`~aieng.advanced_rag.corpora.chunking.relevant_chunk_ids`, and fails
loudly when a span is covered by no chunk.

A :class:`SyntheticQuestion` satisfies the runner's ``RunnableQuery`` protocol,
so it goes into :class:`~aieng.advanced_rag.evaluation.runner.Experiment`
unchanged. A question with no evidence is a null question, its
``is_answerable`` is false, and ``score_null_detection`` reads it with no
adaptation. :meth:`SyntheticQuestion.to_query` converts a document-scoped
question into the corpus :class:`~aieng.advanced_rag.retrieval.corpus.Query`
for the per-document retrievers.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aieng.advanced_rag.corpora.chunking import relevant_chunk_ids
from aieng.advanced_rag.retrieval.corpus import AnswerSpan, Chunk, Query


@dataclass(frozen=True)
class EvidenceSpan:
    """One gold passage: where it is, and what it says.

    Parameters
    ----------
    doc_id : str
        The document the span is in.
    start : int
        Character offset of the first character, into the document text.
    end : int
        Character offset one past the last character.
    quote : str
        The document text between ``start`` and ``end``.
    role : str, optional
        ``"gold"`` for the span the question was written from, and
        ``"equivalent"`` for a second passage the uniqueness filter found to
        give the same answer.
    """

    doc_id: str
    start: int
    end: int
    quote: str
    role: str = "gold"

    def answer_span(self) -> AnswerSpan:
        """Return the span as the harness's answer record."""
        return AnswerSpan(text=self.quote, start=self.start, end=self.end)


@dataclass(frozen=True)
class SyntheticQuestion:
    """One generated question, its reference answer, and its gold spans.

    Parameters
    ----------
    query_id : str
        Unique within a set.
    question : str
        The question text.
    reference : str
        The reference answer, for ``judge_answer``. For a null question, a
        statement of what is missing.
    evidence : tuple of EvidenceSpan, optional
        Gold spans. Empty for a null question.
    kind : str, optional
        The taxonomy type, such as ``single_hop`` or ``held_out_null``.
    scope : str or None, optional
        The document retrieval is scoped to, for corpora where a question
        belongs to one document (ContractNLI, CUAD). ``None`` searches the
        whole corpus.
    group_id : str, optional
        Shared by the wording variants of one question.
    variant : str, optional
        ``"plain"`` or ``"paraphrase"``.
    labels : mapping, optional
        Recorded difficulty measures. Never used to filter.
    provenance : mapping, optional
        Models, categories, and the statement the question came from.
    """

    query_id: str
    question: str
    reference: str
    evidence: tuple[EvidenceSpan, ...] = ()
    kind: str = "single_hop"
    scope: str | None = None
    group_id: str = ""
    variant: str = "plain"
    labels: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_answerable(self) -> bool:
        """Return whether the question has gold evidence."""
        return bool(self.evidence)

    @property
    def doc_id(self) -> str | None:
        """Return the retrieval scope, under the name ``scope_of`` reads."""
        return self.scope

    @property
    def category(self) -> str:
        """Return the question category the generator was asked for."""
        return str(self.provenance.get("question_category", self.kind))

    @property
    def documents(self) -> tuple[str, ...]:
        """Return the evidence documents, in first-seen order."""
        return tuple(dict.fromkeys(span.doc_id for span in self.evidence))

    def answer_spans(self, doc_id: str | None = None) -> tuple[AnswerSpan, ...]:
        """Return the gold spans as answer records, optionally for one document."""
        return tuple(
            span.answer_span()
            for span in self.evidence
            if doc_id is None or span.doc_id == doc_id
        )

    def to_query(self) -> Query:
        """Return the corpus query record for a document-scoped question.

        Returns
        -------
        Query
            With ``doc_id`` set to the scope, or to the single evidence
            document when the question has no scope.

        Raises
        ------
        ValueError
            When the question has no scope and its evidence spans more than one
            document or none, so no single document can be named.
        """
        doc_id = self.scope
        if doc_id is None and len(self.documents) == 1:
            doc_id = self.documents[0]
        if doc_id is None:
            raise ValueError(
                f"{self.query_id} names no single document; run it corpus-wide "
                "through Experiment instead of converting it to a Query"
            )
        return Query(
            query_id=self.query_id,
            doc_id=doc_id,
            category=self.category,
            question=self.question,
            answers=self.answer_spans(doc_id),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        record = asdict(self)
        record["evidence"] = [asdict(span) for span in self.evidence]
        record["labels"] = dict(self.labels)
        record["provenance"] = dict(self.provenance)
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> SyntheticQuestion:
        """Rebuild a question from :meth:`to_dict` output."""
        return cls(
            query_id=str(record["query_id"]),
            question=str(record["question"]),
            reference=str(record.get("reference", "")),
            evidence=tuple(EvidenceSpan(**span) for span in record.get("evidence", ())),
            kind=str(record.get("kind", "single_hop")),
            scope=record.get("scope"),
            group_id=str(record.get("group_id", "")),
            variant=str(record.get("variant", "plain")),
            labels=dict(record.get("labels", {})),
            provenance=dict(record.get("provenance", {})),
        )


def question_id(*parts: object) -> str:
    """Return a stable identifier built from what makes a question unique."""
    digest = hashlib.sha1(
        "\x1f".join(str(part) for part in parts).encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return f"sq-{digest[:12]}"


#: The three ends a candidate can reach. Every candidate reaches exactly one.
OUTCOMES = ("accepted", "rejected", "filtered")


@dataclass(frozen=True)
class Outcome:
    """Where one candidate ended and why.

    Parameters
    ----------
    candidate_id : str
        The candidate's identifier.
    kind : str
        The question type it was generated as.
    outcome : str
        One of :data:`OUTCOMES`. ``rejected`` means the generator declined;
        ``filtered`` means a filter dropped it.
    stage : str
        The last stage it entered.
    reason : str
        Why it stopped, empty for an accepted candidate.
    """

    candidate_id: str
    kind: str
    outcome: str
    stage: str
    reason: str = ""


class Funnel:
    """Thread-safe stage counts and one outcome per candidate.

    Stages are counted per question kind, because a held-out null and an
    answerable question pass different checks. A stage's acceptance rate is the
    share of candidates of that kind that entered it and passed. Stages are
    listed in the order they are first entered.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entered: Counter[tuple[str, str]] = Counter()
        self._passed: Counter[tuple[str, str]] = Counter()
        self._order: list[tuple[str, str]] = []
        self._outcomes: dict[str, Outcome] = {}

    def check(self, stage: str, ok: bool, kind: str = "") -> bool:
        """Count one candidate of ``kind`` through ``stage`` and return ``ok``."""
        key = (kind, stage)
        with self._lock:
            if key not in self._entered:
                self._order.append(key)
            self._entered[key] += 1
            if ok:
                self._passed[key] += 1
        return ok

    def finish(
        self,
        candidate_id: str,
        kind: str,
        outcome: str,
        stage: str,
        reason: str = "",
    ) -> None:
        """Record where a candidate ended.

        Raises
        ------
        ValueError
            If ``outcome`` is not one of :data:`OUTCOMES`, or the candidate
            already has an outcome.
        """
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
        with self._lock:
            if candidate_id in self._outcomes:
                raise ValueError(f"{candidate_id} already ended")
            self._outcomes[candidate_id] = Outcome(
                candidate_id, kind, outcome, stage, reason
            )

    @property
    def outcomes(self) -> list[Outcome]:
        """Return every candidate's outcome, sorted by identifier."""
        with self._lock:
            return sorted(self._outcomes.values(), key=lambda o: o.candidate_id)

    def stages(self) -> list[dict[str, Any]]:
        """Return each stage's kind, entered, passed, and acceptance rate."""
        with self._lock:
            return [
                {
                    "kind": kind,
                    "stage": stage,
                    "entered": self._entered[(kind, stage)],
                    "passed": self._passed[(kind, stage)],
                    "acceptance_rate": round(
                        self._passed[(kind, stage)] / self._entered[(kind, stage)], 4
                    ),
                }
                for kind, stage in self._order
            ]

    def to_dict(self) -> dict[str, Any]:
        """Return stage counts, outcome totals by kind, and drop reasons."""
        outcomes = self.outcomes
        totals: dict[str, Counter[str]] = {}
        reasons: Counter[str] = Counter()
        for item in outcomes:
            totals.setdefault(item.kind, Counter())[item.outcome] += 1
            if item.reason:
                reasons[f"{item.stage}: {item.reason}"] += 1
        return {
            "stages": self.stages(),
            "outcomes": {kind: dict(counts) for kind, counts in sorted(totals.items())},
            "candidates": len(outcomes),
            "accepted": sum(1 for o in outcomes if o.outcome == "accepted"),
            "drop_reasons": dict(reasons.most_common()),
        }


@dataclass
class SyntheticSet:
    """A generated question set and what it took to build.

    Parameters
    ----------
    questions : list of SyntheticQuestion
        Accepted questions, both wording variants included.
    funnel : Funnel or None, optional
        The build's stage counts and per-candidate outcomes.
    meta : dict, optional
        Models, configuration, corpus, and usage.
    """

    questions: list[SyntheticQuestion]
    funnel: Funnel | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[SyntheticQuestion]:
        """Iterate over the questions."""
        return iter(self.questions)

    def __len__(self) -> int:
        """Return the number of questions."""
        return len(self.questions)

    def variant(self, name: str = "plain", *, fallback: bool = True) -> SyntheticSet:
        """Return one wording variant per question group.

        Parameters
        ----------
        name : str, optional
            ``"plain"`` or ``"paraphrase"``.
        fallback : bool, optional
            When a group has no question in ``name``, keep its plain question,
            so the set covers the same groups whichever variant is chosen.

        Returns
        -------
        SyntheticSet
            One question per group, sharing this set's funnel and meta.
        """
        groups: dict[str, dict[str, SyntheticQuestion]] = {}
        for question in self.questions:
            groups.setdefault(question.group_id or question.query_id, {})[
                question.variant
            ] = question
        chosen: list[SyntheticQuestion] = []
        for variants in groups.values():
            if name in variants:
                chosen.append(variants[name])
            elif fallback and "plain" in variants:
                chosen.append(variants["plain"])
        return SyntheticSet(chosen, self.funnel, self.meta)

    @property
    def answerable(self) -> list[SyntheticQuestion]:
        """Return the questions with gold evidence."""
        return [q for q in self.questions if q.is_answerable]

    @property
    def nulls(self) -> list[SyntheticQuestion]:
        """Return the questions with no gold evidence."""
        return [q for q in self.questions if not q.is_answerable]

    def qrels(self, chunks: Iterable[Chunk]) -> dict[str, set[str]]:
        """Map every question's gold spans onto ``chunks``.

        Parameters
        ----------
        chunks : iterable of Chunk
            Any chunking of the documents the set was generated from, with
            honest character offsets.

        Returns
        -------
        dict
            Query identifier to relevant chunk identifiers. Null questions map
            to an empty set.

        Raises
        ------
        ValueError
            If an answerable question's spans overlap no chunk, which means the
            chunks do not cover the text the generator read.
        """
        by_doc: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            by_doc.setdefault(chunk.doc_id, []).append(chunk)
        qrels: dict[str, set[str]] = {}
        for question in self.questions:
            relevant: set[str] = set()
            for doc_id in question.documents:
                relevant |= relevant_chunk_ids(
                    by_doc.get(doc_id, ()), question.answer_spans(doc_id)
                )
            qrels[question.query_id] = relevant
        unreachable = [
            q.query_id
            for q in self.questions
            if q.is_answerable and not qrels[q.query_id]
        ]
        if unreachable:
            raise ValueError(
                f"{len(unreachable)} answerable questions have no covering chunk, "
                f"e.g. {unreachable[:3]}. The chunks do not cover the text the "
                "questions were generated from."
            )
        return qrels

    def queries(self) -> list[Query]:
        """Return every question as a document-scoped corpus query.

        Raises
        ------
        ValueError
            From :meth:`SyntheticQuestion.to_query`, for a question that names
            no single document.
        """
        return [question.to_query() for question in self.questions]

    def references(self) -> dict[str, str]:
        """Return query identifier to reference answer, for ``judge_answer``."""
        return {q.query_id: q.reference for q in self.questions}

    def summary(self) -> dict[str, Any]:
        """Return counts by kind and variant."""
        kinds = Counter(q.kind for q in self.questions)
        variants = Counter(q.variant for q in self.questions)
        return {
            "questions": len(self.questions),
            "answerable": len(self.answerable),
            "null": len(self.nulls),
            "by_kind": dict(sorted(kinds.items())),
            "by_variant": dict(sorted(variants.items())),
            "documents": len({d for q in self.questions for d in q.documents}),
        }

    def to_jsonl(self, path: Path) -> None:
        """Write one question per line."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for question in self.questions:
                handle.write(json.dumps(question.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls, path: Path, meta: dict[str, Any] | None = None) -> SyntheticSet:
        """Read a set written by :meth:`to_jsonl`."""
        questions = [
            SyntheticQuestion.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(questions, None, dict(meta or {}))


__all__ = [
    "OUTCOMES",
    "EvidenceSpan",
    "Funnel",
    "Outcome",
    "SyntheticQuestion",
    "SyntheticSet",
    "question_id",
]
