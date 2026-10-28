"""Route a question to prose, to tables, or to both, and score the routing.

A corpus that holds documents and tables can be queried three ways: one index
over everything, a router that picks a source per question, or an agent that
picks its own tools. This module holds the routers and the scoring. The sources
themselves (a dense index over prose, guarded SQL over the tables) live in
``build_reference.py``, so the routers here run on any sources.

Three routers share one interface, ``route(question) -> RouteDecision``:

* :class:`EmbeddingRouter` compares the question with a few example questions
  per route, using local embeddings, and picks the route whose closest examples
  score highest. It makes no model call.
* :class:`LLMRouter` lists the routes with their descriptions and asks a model
  to pick one, in one call.
* :class:`OracleRouter` reads the gold label, which gives the upper bound a
  perfect router reaches with these sources.

Prior art and why this design
-----------------------------
The two classifier routers follow published designs and are written here
because none of those libraries is in the project's lock file, and adding one
for a few dozen lines would change the environment of every participant.

* ``EmbeddingRouter`` is the algorithm of ``semantic-router`` (Aurelio AI, MIT
  licence): a route is a name plus example utterances, a query is embedded, and
  each route scores the mean of its top-k utterance similarities. Here it runs
  on the project's local ``BAAI/bge-small-en-v1.5`` embedder.
* ``LLMRouter`` follows LlamaIndex's ``LLMSingleSelector`` behind
  ``RouterQueryEngine``: numbered choices with descriptions in a prompt, and a
  JSON reply naming one choice and a reason. LlamaIndex's multi-selector and
  ``SQLAutoVectorQueryEngine`` correspond to the ``both`` route.
* Learned routers (RouteLLM for strong-or-weak model choice, Adaptive-RAG's
  T5 query-complexity classifier, StructRAG's DPO-trained structure router)
  need training data this special topic does not have. :func:`hindsight_route`
  produces the outcome-based labels Adaptive-RAG trains on, so a learned router
  is the next step these records support.

Scoring separates two kinds of routing error. A *mismatch* is a decision that
differs from the gold label. A *costly* error is a mismatch where the chosen
route answered wrong and the gold route answered right. A mismatch that still
answers correctly costs nothing, which is common when a figure appears in both
a table and the text around it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


#: The three routes, from cheapest to most expensive.
ROUTE_NAMES: tuple[str, ...] = ("prose", "table", "both")

#: Position of each route in :data:`ROUTE_NAMES`; ties go to the cheaper route.
ROUTE_ORDER: Mapping[str, int] = {name: i for i, name in enumerate(ROUTE_NAMES)}

#: The route a router falls back to when it cannot decide: query every source.
FALLBACK_ROUTE = "both"


@dataclass(frozen=True)
class Route:
    """One route a router can choose.

    Parameters
    ----------
    name : str
        One of :data:`ROUTE_NAMES`.
    description : str
        What the route's source holds, for :class:`LLMRouter`.
    utterances : tuple of str
        Example questions the route answers, for :class:`EmbeddingRouter`.
        Write them for the kind of question, never copy evaluation questions.
    """

    name: str
    description: str
    utterances: tuple[str, ...] = ()


#: Routes for a corpus of company filings with their tables extracted. The
#: examples are written for this special topic and name no company in the
#: question set.
FILING_ROUTES: tuple[Route, ...] = (
    Route(
        "prose",
        "The text of the filings: business descriptions, risk factors, management's "
        "discussion, accounting policies, legal proceedings, and named people, places, "
        "and events. Use it for questions answered by a statement in words.",
        (
            "What does the company say is the main risk to its supply chain?",
            "Who audits the company's financial statements?",
            "Where is the company headquartered according to its annual report?",
            "How does the company describe its competitive position?",
            "What reason does management give for the change in margins?",
            "Which accounting standard did the company adopt during the year?",
            "What legal proceedings does the filing describe?",
            "How does the company recognize revenue from service contracts?",
        ),
    ),
    Route(
        "table",
        "The tables extracted from the filings, queried with SQL: every cell with its "
        "row label (a line item such as Total revenues) and its column header (often a "
        "year). Use it for questions that ask for a reported figure.",
        (
            "What was the company's total revenue in fiscal 2021?",
            "How much cash and cash equivalents did the company hold at year end 2019?",
            "What were diluted earnings per share for 2022?",
            "What amount of long-term debt did the balance sheet report in 2020?",
            "What was the company's capital expenditure in 2018?",
            "How much did the company pay in dividends in 2017?",
            "What was operating income for the most recent fiscal year?",
            "What were total current liabilities at the end of 2023?",
        ),
    ),
    Route(
        "both",
        "Both sources together. Use it for a question that asks for a reported figure "
        "and also for something the text states in words.",
        (
            "What was the company's net income in 2020, and what did management cite as the main driver of the change?",
            "How much did the company spend on research in 2019, and what areas does it say it focuses research on?",
            "What were total assets at the end of 2021, and who is the company's auditor?",
            "What was revenue in 2022, and which segment does the filing describe as growing fastest?",
            "How much inventory did the company report in 2018, and how does it value inventory?",
            "What was the effective tax rate in 2023, and what does the filing say explains it?",
        ),
    ),
)


@dataclass(frozen=True)
class RouteDecision:
    """What a router chose for one question.

    Parameters
    ----------
    route : str
        The chosen route.
    scores : dict
        Per-route scores, for routers that compute them.
    model_calls : int
        Model calls the decision took.
    prompt_tokens, completion_tokens : int
        Tokens those calls used.
    note : str
        The model's reason, or why the router fell back.
    """

    route: str
    scores: Mapping[str, float] = field(default_factory=dict)
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "route": self.route,
            "scores": {k: round(float(v), 4) for k, v in self.scores.items()},
            "model_calls": self.model_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "note": self.note,
        }


@dataclass(frozen=True)
class Reply:
    """One model reply and the tokens it used."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


#: ``complete(prompt, max_tokens)`` returning a :class:`Reply`.
Complete = Callable[[str, int], Reply]


def _check_routes(routes: Sequence[Route]) -> None:
    names = [r.name for r in routes]
    if len(set(names)) != len(names):
        raise ValueError("two routes share a name")
    if not names:
        raise ValueError("a router needs at least one route")


class EmbeddingRouter:
    """Pick the route whose example questions are closest to the question.

    Parameters
    ----------
    routes : sequence of Route
        Each needs at least one utterance.
    embedder : object
        Has ``encode(texts)`` and ``encode_queries(texts)``, as
        :class:`~aieng.advanced_rag.retrieval.embedders.SentenceTransformerEmbedder`
        does.
    top_k : int, optional
        How many of a route's closest utterances to average.
    threshold : float or None, optional
        Below this best score, return :data:`FALLBACK_ROUTE` instead. ``None``
        always takes the best route.

    Raises
    ------
    ValueError
        If two routes share a name or a route has no utterances.
    """

    def __init__(
        self,
        routes: Sequence[Route],
        embedder: Any,
        *,
        top_k: int = 3,
        threshold: float | None = None,
    ) -> None:
        import numpy as np  # noqa: PLC0415 - only this router needs numpy

        _check_routes(routes)
        if any(not r.utterances for r in routes):
            raise ValueError("every route needs at least one utterance")
        self._np = np
        self.routes = tuple(routes)
        self.embedder = embedder
        self.top_k = top_k
        self.threshold = threshold
        texts = [u for r in self.routes for u in r.utterances]
        # Utterances are questions, so they are encoded as queries: bge models
        # add an instruction prefix to queries and not to passages.
        matrix = np.asarray(embedder.encode_queries(texts), dtype=np.float32)
        self._matrix = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9, None)
        self._owner = [r.name for r in self.routes for _ in r.utterances]

    def scores(self, question: str) -> dict[str, float]:
        """Return each route's mean similarity over its ``top_k`` closest utterances."""
        np = self._np
        vector = np.asarray(self.embedder.encode_queries([question])[0], dtype=np.float32)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-9)
        sims = self._matrix @ vector
        found: dict[str, float] = {}
        for route in self.routes:
            own = sorted(
                (float(s) for s, o in zip(sims, self._owner, strict=True) if o == route.name),
                reverse=True,
            )
            found[route.name] = sum(own[: self.top_k]) / len(own[: self.top_k])
        return found

    def route(self, question: str) -> RouteDecision:
        """Return the best-scoring route, or the fallback below the threshold."""
        found = self.scores(question)
        best = max(found, key=lambda name: (found[name], -ROUTE_ORDER.get(name, 9)))
        if self.threshold is not None and found[best] < self.threshold:
            return RouteDecision(FALLBACK_ROUTE, found, note=f"best score {found[best]:.3f} under threshold")
        return RouteDecision(best, found)


LLM_ROUTER_PROMPT = """Some choices are given below. Each is a source that can answer
questions about {corpus}. Pick the one choice that best answers the question.

{choices}

Question: {question}

Return one JSON object and nothing else:
{{"choice": <the choice number>, "reason": "<one short sentence>"}}"""

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse_choice(text: str, count: int) -> tuple[int | None, str]:
    """Return the 1-based choice and reason from a router reply.

    A reply with no JSON object, or a choice outside ``1..count``, gives
    ``None``, and the caller falls back.
    """
    clean = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    match = _JSON.search(clean)
    if match is None:
        return None, "no JSON in the reply"
    try:
        record = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, "the reply's JSON does not parse"
    try:
        choice = int(record.get("choice"))
    except (TypeError, ValueError):
        return None, "no numeric choice"
    if not 1 <= choice <= count:
        return None, f"choice {choice} is outside 1 to {count}"
    return choice, str(record.get("reason", ""))[:300]


class LLMRouter:
    """Ask a model to pick one route from numbered descriptions.

    Parameters
    ----------
    routes : sequence of Route
        The choices, in the order shown.
    complete : callable
        ``complete(prompt, max_tokens)`` returning a :class:`Reply`. Supply a
        cached, metered client.
    corpus : str, optional
        What the prompt calls the corpus.
    max_tokens : int, optional
        Completion cap for the call.
    """

    def __init__(
        self,
        routes: Sequence[Route],
        complete: Complete,
        *,
        corpus: str = "a collection of company filings",
        max_tokens: int = 200,
    ) -> None:
        _check_routes(routes)
        self.routes = tuple(routes)
        self.complete = complete
        self.corpus = corpus
        self.max_tokens = max_tokens

    def prompt(self, question: str) -> str:
        """Return the prompt sent for one question."""
        choices = "\n".join(f"({i}) {route.description}" for i, route in enumerate(self.routes, 1))
        return LLM_ROUTER_PROMPT.format(corpus=self.corpus, choices=choices, question=question)

    def route(self, question: str) -> RouteDecision:
        """Return the route the model picked, or the fallback for an unusable reply."""
        reply = self.complete(self.prompt(question), self.max_tokens)
        choice, reason = parse_choice(reply.text, len(self.routes))
        names = [r.name for r in self.routes]
        fallback = FALLBACK_ROUTE if FALLBACK_ROUTE in names else names[-1]
        return RouteDecision(
            names[choice - 1] if choice else fallback,
            {},
            model_calls=1,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            note=reason if choice else f"fell back: {reason}",
        )


class OracleRouter:
    """Route by the gold label: the ceiling for a router over these sources.

    Parameters
    ----------
    labels : mapping
        Question text to its gold route.
    """

    def __init__(self, labels: Mapping[str, str]) -> None:
        self.labels = dict(labels)

    def route(self, question: str) -> RouteDecision:
        """Return the gold route.

        Raises
        ------
        KeyError
            If the question has no label.
        """
        return RouteDecision(self.labels[question], {}, note="gold label")


class FixedRouter:
    """Send every question to one route, such as ``both`` with no router at all."""

    def __init__(self, route: str) -> None:
        self.fixed = route

    def route(self, question: str) -> RouteDecision:  # noqa: ARG002 - the interface
        """Return the fixed route."""
        return RouteDecision(self.fixed, {}, note="fixed")


# ---------------------------------------------------------------------------
# Scoring


def hindsight_route(correct: Mapping[str, bool], costs: Mapping[str, float] | None = None) -> str | None:
    """Return the cheapest route that answered correctly, or ``None`` if none did.

    These are the outcome-based labels Adaptive-RAG trains its classifier on:
    when several routes answer, the cheaper one wins.

    Parameters
    ----------
    correct : mapping
        Route to whether its answer was judged correct.
    costs : mapping or None, optional
        Route to its cost. :data:`ROUTE_NAMES` order when omitted.
    """
    winners = [name for name, ok in correct.items() if ok]
    if not winners:
        return None
    if costs is None:
        return min(winners, key=lambda n: ROUTE_ORDER.get(n, 9))
    return min(winners, key=lambda n: (costs.get(n, float("inf")), ROUTE_ORDER.get(n, 9)))


@dataclass(frozen=True)
class RoutingReport:
    """How one router's decisions compare with the gold labels.

    Parameters
    ----------
    confusion : dict
        Gold route to chosen route to count.
    mismatches : int
        Decisions that differ from the gold label.
    costly : int
        Mismatches where the chosen route was wrong and the gold route right.
    rescued : int
        Mismatches where the chosen route was right and the gold route wrong.
    questions : int
        Questions scored.
    """

    confusion: Mapping[str, Mapping[str, int]]
    mismatches: int
    costly: int
    rescued: int
    questions: int

    @property
    def agreement(self) -> float:
        """Return the share of decisions that match the gold label."""
        return 1 - self.mismatches / self.questions if self.questions else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "confusion": {g: dict(c) for g, c in self.confusion.items()},
            "mismatches": self.mismatches,
            "costly": self.costly,
            "rescued": self.rescued,
            "questions": self.questions,
            "agreement": round(self.agreement, 4),
        }


def routing_report(
    chosen: Mapping[str, str],
    gold: Mapping[str, str],
    correct_by_route: Mapping[str, Mapping[str, bool]],
) -> RoutingReport:
    """Score one router's decisions.

    Parameters
    ----------
    chosen : mapping
        Question id to the route the router chose.
    gold : mapping
        Question id to its gold route.
    correct_by_route : mapping
        Question id to route to whether that route's answer was correct.

    Returns
    -------
    RoutingReport
        The confusion matrix and the error counts.
    """
    confusion: dict[str, Counter[str]] = {name: Counter() for name in ROUTE_NAMES}
    mismatches = costly = rescued = 0
    for qid, route in chosen.items():
        label = gold[qid]
        confusion.setdefault(label, Counter())[route] += 1
        if route == label:
            continue
        mismatches += 1
        outcome = correct_by_route.get(qid, {})
        took, should = bool(outcome.get(route)), bool(outcome.get(label))
        if should and not took:
            costly += 1
        elif took and not should:
            rescued += 1
    return RoutingReport(
        {g: dict(c) for g, c in confusion.items()},
        mismatches,
        costly,
        rescued,
        len(chosen),
    )


@dataclass
class ArmSummary:
    """One arm's accuracy and cost per question, overall and per question type."""

    arm: str
    model: str
    questions: int
    correct: int
    accuracy: float
    calls: float
    prompt_tokens: float
    completion_tokens: float
    by_type: dict[str, dict[str, float]] = field(default_factory=dict)
    routes: dict[str, int] = field(default_factory=dict)
    routing: dict[str, Any] | None = None


def summarise_arm(
    arm: str,
    records: Iterable[Mapping[str, Any]],
    *,
    routing: RoutingReport | None = None,
) -> ArmSummary:
    """Summarise one arm's per-question records.

    Each record needs ``qtype``, ``correct``, ``model_calls``, ``prompt_tokens``,
    ``completion_tokens``, and ``model``, and may carry the ``route`` taken.
    """
    rows = list(records)
    if not rows:
        raise ValueError(f"arm {arm} has no records")
    n = len(rows)
    types: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        types.setdefault(str(r["qtype"]), []).append(r)

    def mean(key: str, group: Sequence[Mapping[str, Any]]) -> float:
        return round(sum(float(r[key]) for r in group) / len(group), 2)

    return ArmSummary(
        arm=arm,
        model=str(rows[0]["model"]),
        questions=n,
        correct=sum(bool(r["correct"]) for r in rows),
        accuracy=round(sum(bool(r["correct"]) for r in rows) / n, 4),
        calls=mean("model_calls", rows),
        prompt_tokens=round(mean("prompt_tokens", rows)),
        completion_tokens=round(mean("completion_tokens", rows)),
        by_type={
            t: {
                "questions": len(g),
                "accuracy": round(sum(bool(r["correct"]) for r in g) / len(g), 4),
                "calls": mean("model_calls", g),
                "prompt_tokens": round(mean("prompt_tokens", g)),
            }
            for t, g in sorted(types.items(), key=lambda kv: ROUTE_ORDER.get(kv[0], 9))
        },
        routes=dict(Counter(str(r["route"]) for r in rows if r.get("route"))),
        routing=routing.to_dict() if routing is not None else None,
    )


__all__ = [
    "FALLBACK_ROUTE",
    "FILING_ROUTES",
    "LLM_ROUTER_PROMPT",
    "ROUTE_NAMES",
    "ArmSummary",
    "Complete",
    "EmbeddingRouter",
    "FixedRouter",
    "LLMRouter",
    "OracleRouter",
    "Reply",
    "Route",
    "RouteDecision",
    "RoutingReport",
    "hindsight_route",
    "parse_choice",
    "routing_report",
    "summarise_arm",
]
