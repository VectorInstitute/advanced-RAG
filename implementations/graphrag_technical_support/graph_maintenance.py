#!/usr/bin/env python3
"""Measure graph maintenance on the ServerFault graph and a small LightRAG index.

Each step writes one JSON file under ``outputs/maintenance/``, which
``02_graph_maintenance.ipynb`` reads. The steps that make no model call
(``retraction``, ``dedup``, ``gate``) read the ServerFault graph from the local
artifact cache and replay its extraction response cache. The others call a model:

* ``revision`` re-extracts the earlier version of edited posts with the reference
  model, from the dump's ``PostHistory.xml``.
* ``drift`` re-extracts unchanged posts with a second model, and with the reference
  model again for the noise floor.
* ``lightrag`` deletes, merges, and edits on a copy of the 5-passage LightRAG index.

Every call goes through the shared response cache, so a re-run costs nothing. The
model names are parameters and are recorded in each output.

Examples
--------
Run the model-free steps, then the model steps at 15 requests a minute::

    $ python graph_maintenance.py retraction dedup gate
    $ python graph_maintenance.py revision drift lightrag --rpm 15
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import os
import random
import shutil
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from aieng.advanced_rag.cache.paths import root as cache_root
from aieng.advanced_rag.corpora import serverfault
from aieng.advanced_rag.corpora.sources import cache_dir
from aieng.advanced_rag.graphrag import extraction
from aieng.advanced_rag.graphrag import maintenance as mt
from aieng.advanced_rag.graphrag.adapters.extracted_graph import ExtractedGraphRetriever
from aieng.advanced_rag.graphrag.proxy import CallLedger, RateLimiter, ResponseCache, async_llm, govern
from aieng.advanced_rag.runs import Provenance


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "maintenance"
BASE_URL = os.environ.get("LLM_BASE_URL", "https://proxy.vectorinstitute.ai/v1")
#: The model that built the ServerFault graph. Revisions are re-extracted with it so
#: a changed fact reflects the edit and not a model change.
REFERENCE_MODEL = os.environ.get("MAINTENANCE_REFERENCE_MODEL", "Qwen3_8-27B")
#: The second model for the drift measurement.
DRIFT_MODEL = os.environ.get("MAINTENANCE_DRIFT_MODEL", "Qwen3_8-Flash-Next")
#: The model that built the 5-passage LightRAG index.
LIGHTRAG_MODEL = os.environ.get("MAINTENANCE_LIGHTRAG_MODEL", "Qwen3_6-35B-A3B")
MAX_TOKENS = 2000
DEFAULT_RPM = 15
SEED = 0
RETRACTION_SAMPLE = 500
REVISION_SAMPLE = 25
DRIFT_SAMPLE = 50
DEDUP_LABEL_SAMPLE = 60
# Five posts retracted in the worked example and the gate. The smoke questions
# below are written for them, in words of this script's own.
DEMO_POSTS = ("q9708", "q62411", "q309357", "q264595", "q253313")
SMOKE_QUESTIONS = {
    "q9708": "What is the difference between a pem file, a .der file, and .pkcs12?",
    "q62411": "Pipe du -h into sort so the largest directories come first",
    "q309357": "Use nmap to check a tcp port when ping only speaks icmp",
    "q264595": "Can scp -r copy a sourcedirectory, or do I send a .tar file?",
    "q253313": "ssh says bad owner or permissions on ~/.ssh/config; which chmod fixes it?",
}


def work_root() -> Path:
    """Return the directory the ServerFault graph and its response cache live under."""
    return Path(cache_root()) / "graphrag" / "serverfault"


PRODUCER = "graphrag_technical_support/graph_maintenance.py"


def model_free(note: str, producer: str = PRODUCER) -> dict[str, Any]:
    """Return the ``produced_by`` block of a result that called no chat model."""
    return {
        "model": None,
        "response_models": [],
        "family": None,
        "endpoint": None,
        "producer": producer,
        "usage": {},
        "note": note,
        "model_calls": 0,
    }


def drift_produced_by(drift_model: str, reference: str, cold_calls: Mapping[str, int]) -> dict[str, Any]:
    """Return the drift result's ``produced_by`` block, named for the second model.

    The block has one ``model`` slot. The drift measured is the second model's,
    so it takes the slot, and the reference model's re-run sits in ``usage``.
    """
    block: dict[str, Any] = Provenance.for_model(
        drift_model,
        producer=PRODUCER,
        usage={model: {"calls": calls} for model, calls in cold_calls.items()},
        note=(
            f"Second model {drift_model} against reference {reference}, which built the "
            "graph and is also re-run on the same posts. Calls are the cold build; "
            "each arm's cold_tokens holds its tokens."
        ),
    ).to_dict()
    return block


def write(name: str, payload: Mapping[str, Any]) -> Path:
    """Write one output file and return its path."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def api_key() -> str:
    """Return the API key from the environment, never printing it."""
    for name in ("LLM_API_KEY", "VECTOR_PROXY_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value.strip()
    raise SystemExit("set LLM_API_KEY, VECTOR_PROXY_API_KEY, or OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# Shared loading
# ---------------------------------------------------------------------------


class Corpus:
    """The ServerFault graph, its documents, and its per-document units, loaded once."""

    def __init__(self) -> None:
        self.graph = extraction.KnowledgeGraph.load(work_root() / "graph")
        self.payload = serverfault.fetch_slice()
        self.documents = serverfault.documents(self.payload)
        self.by_id = {d.doc_id: d for d in self.documents}
        self.cache = ResponseCache(work_root() / "responses.sqlite")
        self.units = mt.extract_units(
            [(d.doc_id, d.text) for d in self.documents], replay(self.cache, REFERENCE_MODEL), max_workers=1
        )

    def chunk_texts(self) -> dict[str, str]:
        """Return every block's text by block id, for the graph retriever."""
        return {
            block.block_id: block.text
            for d in self.documents
            for _, blocks in extraction.sections(d.doc_id, d.text)
            for block in blocks
        }


def request_key(model: str, prompt: str, tag: str | None = None) -> str:
    """Return the response-cache key ``govern`` stores an extraction call under."""
    params: dict[str, Any] = {"temperature": 0.0, "max_tokens": MAX_TOKENS}
    if tag:
        params["user"] = tag
    return str(
        ResponseCache.key(
            {
                "kind": "chat-completion",
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "params": {**params, "thinking": False},
            }
        )
    )


def replay(cache: ResponseCache, model: str, tag: str | None = None) -> Callable[[str], str]:
    """Return a ``complete`` that answers from the response cache only.

    A miss raises, which ``build_graph`` counts as a failed call, as the original
    build counted the calls that failed after retries.
    """

    def complete(prompt: str) -> str:
        raw = cache.get(request_key(model, prompt, tag))
        if raw is None:
            raise KeyError("not in the response cache")
        return str(json.loads(raw)["choices"][0]["message"]["content"] or "")

    return complete


def live(
    cache: ResponseCache, model: str, limiter: RateLimiter, ledger: CallLedger, tag: str | None = None
) -> Callable[[str], str]:
    """Return a cached, throttled ``complete`` that calls the model on a miss."""
    from openai import OpenAI  # noqa: PLC0415

    client = govern(
        OpenAI(base_url=BASE_URL, api_key=api_key(), max_retries=0, timeout=600.0),
        cache=cache,
        limiter=limiter,
        ledger=ledger,
        max_retries=16,
    )

    def complete(prompt: str) -> str:
        extra: dict[str, Any] = {"user": tag} if tag else {}
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            **extra,
        )
        return str(response.choices[0].message.content or "")

    return complete


def cached_tokens(key: str) -> int:
    """Return the prompt plus completion tokens a cached ServerFault call cost."""
    import sqlite3  # noqa: PLC0415

    with sqlite3.connect(work_root() / "responses.sqlite") as db:
        row = db.execute("SELECT prompt_tokens, completion_tokens FROM responses WHERE key = ?", (key,)).fetchone()
    return int(row[0] or 0) + int(row[1] or 0) if row else 0


def window_prompts(text: str, doc_id: str) -> list[str]:
    """Return the prompts ``build_graph`` sends for one document, in order."""
    prompts: list[str] = []

    def record(prompt: str) -> str:
        prompts.append(prompt)
        return "{}"

    extraction.build_graph([(doc_id, text)], record, max_workers=1)
    return prompts


def describe(values: Sequence[float]) -> dict[str, float]:
    """Return mean, median, and maximum."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "mean": round(statistics.mean(values), 2),
        "median": statistics.median(values),
        "max": max(values),
    }


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def step_retraction(corpus: Corpus) -> dict[str, Any]:
    """Retract posts by provenance: the five worked examples, then 500 one at a time."""
    graph, units = corpus.graph, corpus.units
    folded = mt.fold_units(units, [d.doc_id for d in corpus.documents])
    fold_diff = mt.graph_diff(graph, folded)
    index = mt.ProvenanceIndex.of(graph)

    demo: dict[str, Any] = {}
    for doc_id in DEMO_POSTS:
        working = mt.copy_graph(graph)
        report = mt.retract(working, [doc_id], units=units)
        removed = sorted(report.entities_removed)
        blocks = extraction.sections(doc_id, corpus.by_id[doc_id].text)
        demo[doc_id] = {
            # What a chunk index deletes for the same post: its blocks, keyed by
            # document, with nothing else touched.
            "blocks": sum(len(b) for _, b in blocks),
            **report.counts(),
            "entities_removed_names": removed,
            "entities_touched_names": sorted(report.entities_touched),
            "hub_degrees_touched": sorted(
                ([n, len(index.relations_by_entity.get(n, ()))] for n in report.entities_touched),
                key=lambda item: -int(item[1]),
            )[:5],
        }

    # A retraction from the folded graph with units must equal a fold of the
    # remaining units, which is what makes it safe to apply instead of a rebuild.
    working = mt.copy_graph(folded)
    mt.retract(working, DEMO_POSTS, units=units)
    rebuilt = mt.fold_units(units, [d.doc_id for d in corpus.documents if d.doc_id not in DEMO_POSTS])
    equal = (
        mt.graph_diff(working, rebuilt).counts()["entity_jaccard"] == 1.0
        and mt.graph_diff(working, rebuilt).counts()["relation_jaccard"] == 1.0
        and all(
            working.entities[n].description == rebuilt.entities[n].description
            and working.entities[n].type == rebuilt.entities[n].type
            and len(working.entities[n].mentions) == len(rebuilt.entities[n].mentions)
            for n in rebuilt.entities
        )
    )

    rng = random.Random(SEED)
    sample = rng.sample([d.doc_id for d in corpus.documents], RETRACTION_SAMPLE)
    rows: list[dict[str, Any]] = []
    for doc_id in sample:
        impact = mt.retraction_impact(graph, [doc_id], index)
        stale = 0
        for name in impact.entities_touched:
            entity = graph.entities[name]
            rest = [
                units[d].entities[name].description
                for d in entity.documents
                if d != doc_id and name in units[d].entities
            ]
            stale += entity.description not in rest
        for key in impact.relations_touched:
            relation = graph.relations[key]
            rest = [
                units[d].relations[key].description
                for d in relation.documents
                if d != doc_id and key in units[d].relations
            ]
            stale += relation.description not in rest
        rows.append({"doc_id": doc_id, **impact.counts(), "stale_descriptions": stale})
    fields = (
        "entities_removed",
        "entities_touched",
        "relations_removed",
        "relations_dangling",
        "relations_touched",
        "neighbourhood",
        "stale_descriptions",
    )
    support = [len(e.documents) for e in graph.entities.values()]
    return {
        "model": REFERENCE_MODEL,
        "graph": {
            "entities": len(graph.entities),
            "relations": len(graph.relations),
            "documents": len(corpus.documents),
        },
        "single_document_entities": sum(1 for s in support if s == 1),
        "fold_of_units_against_build": {
            **fold_diff.counts(),
            "note": "relations the single build pass accepted because an endpoint came from an earlier document",
        },
        "retract_equals_rebuild_of_remaining_units": equal,
        "demo": demo,
        "sample": {
            "seed": SEED,
            "posts": RETRACTION_SAMPLE,
            "summary": {f: describe([r[f] for r in rows]) for f in fields},
        },
        "rows": rows,
        "top_hubs": [
            [n, len(index.relations_by_entity[n])]
            for n in sorted(index.relations_by_entity, key=lambda n: -len(index.relations_by_entity[n]))[:10]
        ],
    }


def step_dedup(corpus: Corpus) -> dict[str, Any]:
    """Propose duplicate groups by name folding, sample them for labels, merge all."""
    graph = corpus.graph
    groups = mt.duplicate_groups(graph)
    rng = random.Random(SEED)
    labelled = rng.sample(groups, DEDUP_LABEL_SAMPLE)
    before = mt.topology(graph)
    working = mt.copy_graph(graph)
    reports = [mt.merge_entities(working, group) for group in groups]
    after = mt.topology(working)
    sizes = [len(g) for g in groups]
    types_disagree = sum(1 for g in groups if len({graph.entities[n].type for n in g}) > 1)
    cross_before = sum(1 for e in graph.entities.values() if len(e.documents) > 1)
    cross_after = sum(1 for e in working.entities.values() if len(e.documents) > 1)
    return {
        "model": REFERENCE_MODEL,
        "rule": "fold_name: case, separators, dots inside a word, trailing (), plural s; other punctuation never folds",
        "groups": len(groups),
        "entities_in_groups": sum(sizes),
        "group_size": describe(sizes),
        "groups_with_type_disagreement": types_disagree,
        "largest_groups": [g[:8] for g in groups[:10]],
        "label_sample": [
            {
                "group": g,
                "types": [graph.entities[n].type for n in g],
                "mentions": [len(graph.entities[n].mentions) for n in g],
            }
            for g in labelled
        ],
        "merge": {
            "entities_before": len(graph.entities),
            "entities_after": len(working.entities),
            "relations_before": len(graph.relations),
            "relations_after": len(working.relations),
            "relations_rewired": sum(r.relations_rewired for r in reports),
            "relations_collapsed": sum(r.relations_collapsed for r in reports),
            "self_loops_dropped": sum(r.self_loops_dropped for r in reports),
            "types_changed": sum(r.type_changed for r in reports),
            "cross_document_entities_before": cross_before,
            "cross_document_entities_after": cross_after,
            "validation_problems_after": len(working.validate()),
            "topology_before": before,
            "topology_after": after,
            "topology_delta": mt.topology_delta(before, after),
        },
    }


def _smoke(lookup: mt.EntityLookup) -> dict[str, Callable[[], list[str]]]:
    return {q: functools.partial(lookup.cite, q, 10) for q in SMOKE_QUESTIONS.values()}


def _rank(cited: Sequence[str], doc_id: str) -> int | None:
    return next((i + 1 for i, d in enumerate(cited) if d == doc_id), None)


def step_gate(corpus: Corpus) -> dict[str, Any]:
    """Gate five changes, from a provenance retraction to a stale serving index."""
    graph = corpus.graph
    before = mt.topology(graph)
    lookup = mt.EntityLookup(graph)
    before_smoke = {q: f() for q, f in _smoke(lookup).items()}

    # The package's one-hop graph retriever, on the same questions, for comparison.
    walker = ExtractedGraphRetriever(graph.entities.values(), graph.relations.values(), corpus.chunk_texts())
    walker_ranks = {
        q: _rank([e.provenance.doc_id for e in walker.retrieve(q, k=50)], doc) for doc, q in SMOKE_QUESTIONS.items()
    }

    def gate(changed: extraction.KnowledgeGraph, serving: mt.EntityLookup | None = None) -> mt.GateResult:
        return mt.maintenance_gate(
            changed,
            before_topology=before,
            retracted=DEMO_POSTS,
            smoke=_smoke(serving or mt.EntityLookup(changed)),
            smoke_before=before_smoke,
        )

    retracted = mt.copy_graph(graph)
    report = mt.retract(retracted, DEMO_POSTS, units=corpus.units)
    gate_retract = gate(retracted)
    # The graph is right and the lookup serving it was built before the change.
    gate_stale = gate(retracted, serving=lookup)

    # The delete a graph store makes easy: drop every node and edge the posts
    # touch. It removes shared hubs and leaves relations pointing at them.
    naive = mt.copy_graph(graph)
    hit = {n for n, e in naive.entities.items() if set(e.documents) & set(DEMO_POSTS)}
    for name in hit:
        del naive.entities[name]
    for key in [k for k, r in naive.relations.items() if set(r.documents) & set(DEMO_POSTS)]:
        del naive.relations[key]
    gate_naive = gate(naive)

    # Mention-only removal: strip the posts' mentions and keep every fact, which
    # leaves facts that cite nothing.
    stripped = mt.copy_graph(graph)
    for fact in list(stripped.entities.values()) + list(stripped.relations.values()):
        fact.mentions = [m for m in fact.mentions if m.provenance.doc_id not in DEMO_POSTS]
    gate_stripped = gate(stripped)

    merged = mt.copy_graph(graph)
    groups = mt.duplicate_groups(graph)
    for group in groups:
        mt.merge_entities(merged, group)
    gate_merge = mt.maintenance_gate(
        merged, before_topology=before, expected_entity_removals=sum(len(g) - 1 for g in groups)
    )

    return {
        "model": REFERENCE_MODEL,
        "retracted": list(DEMO_POSTS),
        "smoke_questions": SMOKE_QUESTIONS,
        "smoke_before": {
            q: {"target": doc, "rank_of_target": _rank(before_smoke[q], doc), "documents": len(set(before_smoke[q]))}
            for doc, q in SMOKE_QUESTIONS.items()
        },
        "graph_retriever_rank_of_target_at_50": walker_ranks,
        "limits": mt.GateLimits().__dict__,
        "provenance_retraction": {**gate_retract.to_dict(), "report": report.counts()},
        "stale_serving_index": gate_stale.to_dict(),
        "naive_delete": {**gate_naive.to_dict(), "entities_deleted": len(hit)},
        "mentions_only": gate_stripped.to_dict(),
        "dedup_merge": gate_merge.to_dict(),
    }


def post_history(corpus: Corpus) -> dict[int, list[mt.Revision]]:
    """Return the slice's post history, extracting and caching it on first use."""
    cached = work_root() / "post-history-slice.json"
    if cached.exists():
        raw = json.loads(cached.read_text(encoding="utf-8"))
        return {int(k): [mt.Revision(**r) for r in v] for k, v in raw.items()}
    dump = serverfault.fetch_dump()
    path = mt.extract_member(
        dump, cache_dir() / "serverfault-dump", "PostHistory.xml", program=os.environ.get("SEVENZIP")
    )
    wanted = {q["question_id"] for q in corpus.payload["questions"]}
    wanted |= {a["answer_id"] for q in corpus.payload["questions"] for a in q["answers"]}
    try:
        history: dict[int, list[mt.Revision]] = mt.read_post_history(path, wanted)
    finally:
        # 2 GB extracted; the slice's 20,000 posts are kept below.
        path.unlink(missing_ok=True)
    cached.write_text(json.dumps({k: [r.__dict__ for r in v] for k, v in history.items()}), encoding="utf-8")
    return history


def revision_candidates(corpus: Corpus, history: Mapping[int, Sequence[mt.Revision]]) -> list[dict[str, Any]]:
    """Return documents whose question or answer has a substantive last edit.

    A candidate's latest revision must render to exactly the loader's text, so the
    earlier revision differs from the graph's document by the edit and nothing else.
    """
    found: list[dict[str, Any]] = []
    for question in corpus.payload["questions"]:
        doc = corpus.by_id.get(f"q{question['question_id']}")
        if doc is None:
            continue
        accepted = question["accepted_answer_id"]
        for post_id, section in ((question["question_id"], "question_text"), (accepted, "answer_text")):
            bodies = mt.body_revisions(history.get(post_id, []))
            if len(bodies) < 2:
                continue
            current = getattr(doc, section)
            if mt.render_markdown(bodies[-1].text) != current:
                continue
            earlier = mt.render_markdown(bodies[-2].text)
            in_old, in_new = mt.changed_spans(earlier, current)
            changed = sum(e - s for s, e in in_old) + sum(e - s for s, e in in_new)
            if changed >= 40 and earlier != current:
                found.append(
                    {
                        "doc_id": doc.doc_id,
                        "post_id": post_id,
                        "section": section,
                        "earlier": earlier,
                        "changed_chars": changed,
                    }
                )
    return found


def step_revision(corpus: Corpus, rpm: int, model: str = REFERENCE_MODEL) -> dict[str, Any]:
    """Re-extract earlier versions of edited posts; align fact changes to the edit."""
    history = post_history(corpus)
    candidates = revision_candidates(corpus, history)
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for c in candidates:
        by_doc.setdefault(c["doc_id"], []).append(c)
    rng = random.Random(SEED)
    chosen_docs = rng.sample(sorted(by_doc), REVISION_SAMPLE)
    chosen = [rng.choice(by_doc[d]) for d in chosen_docs]
    ledger = CallLedger("revision")
    complete = live(corpus.cache, model, RateLimiter(per_minute=rpm), ledger)
    earlier_docs = {c["doc_id"]: replace(corpus.by_id[c["doc_id"]], **{c["section"]: c["earlier"]}) for c in chosen}
    earlier_units = mt.extract_units([(d, doc.text) for d, doc in earlier_docs.items()], complete)

    # A window whose text the edit did not touch has the same prompt as the
    # current version's, which the graph build already paid for. Only the edited
    # windows cost a call.
    new_calls, new_tokens, reused = 0, 0, 0
    for doc_id, doc in earlier_docs.items():
        current = set(window_prompts(corpus.by_id[doc_id].text, doc_id))
        for prompt in window_prompts(doc.text, doc_id):
            if prompt in current:
                reused += 1
                continue
            new_calls += 1
            new_tokens += cached_tokens(request_key(model, prompt))
    rows = []
    for c in chosen:
        doc_id = c["doc_id"]
        alignment = mt.edit_alignment(
            earlier_docs[doc_id].text, corpus.by_id[doc_id].text, earlier_units[doc_id], corpus.units[doc_id]
        )
        rows.append(
            {"doc_id": doc_id, "edited": "question" if c["section"] == "question_text" else "answer", **alignment}
        )

    # The graph as it stood before the edits: the earlier units in place of the
    # current ones. Applying the revisions must leave no fact citing an earlier
    # version that the current one dropped.
    units_before = dict(corpus.units)
    units_before.update(earlier_units)
    order = [d.doc_id for d in corpus.documents]
    graph = mt.fold_units(units_before, order)
    before = mt.topology(graph)
    reports = [mt.revise(graph, c["doc_id"], corpus.units[c["doc_id"]], units=units_before) for c in chosen]
    target = mt.fold_units(corpus.units, order)
    final = mt.graph_diff(graph, target)
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    totals = {k: sum(r[k] for r in rows) for k in keys if not k.endswith("jaccard")}
    return {
        "model": model,
        "source": "PostHistory.xml in the pinned dump; the earlier revision is the one before each post's last body edit",
        "candidates": {"posts": len(candidates), "documents": len(by_doc)},
        "seed": SEED,
        "documents": REVISION_SAMPLE,
        "cost": ledger.to_dict(),
        "edited_window_calls": new_calls,
        "edited_window_tokens": new_tokens,
        "unchanged_windows_reused": reused,
        "rows": rows,
        "totals": totals,
        "entity_jaccard": describe([r["entity_jaccard"] for r in rows]),
        "relation_jaccard": describe([r["relation_jaccard"] for r in rows]),
        "revise_reports": describe([r.counts()["entities_removed"] for r in reports]),
        "after_revisions_against_current_fold": final.counts(),
        "after_revisions_validation_problems": len(graph.validate()),
        "topology_before_revisions": {k: before[k] for k in ("entities", "relations", "isolated_entities")},
    }


def step_drift(
    corpus: Corpus, rpm: int, drift_model: str = DRIFT_MODEL, reference: str = REFERENCE_MODEL
) -> dict[str, Any]:
    """Re-extract unchanged posts with a second model and with the reference model."""
    rng = random.Random(SEED + 1)
    pool = [d.doc_id for d in corpus.documents if d.doc_id not in DEMO_POSTS]
    chosen = rng.sample(pool, DRIFT_SAMPLE)
    limiter = RateLimiter(per_minute=rpm)
    pairs = [(d, corpus.by_id[d].text) for d in chosen]
    results: dict[str, Any] = {}
    for label, model, tag in (("rerun_reference", reference, "maintenance-rerun"), ("second_model", drift_model, None)):
        ledger = CallLedger(label)
        units = mt.extract_units(pairs, live(corpus.cache, model, limiter, ledger, tag))
        rows = []
        for doc_id in chosen:
            diff = mt.graph_diff(corpus.units[doc_id], units[doc_id])
            new = units[doc_id]
            rows.append(
                {
                    "doc_id": doc_id,
                    **diff.counts(),
                    "entities": len(new.entities),
                    "relations": len(new.relations),
                    "located_share": round(
                        sum(m.located for e in new.entities.values() for m in e.mentions)
                        / max(sum(len(e.mentions) for e in new.entities.values()), 1),
                        4,
                    ),
                    "failed_calls": new.report.calls_failed,
                }
            )
        # What swapping these posts' units into the graph would do to the whole.
        graph = mt.fold_units({**corpus.units, **units}, [d.doc_id for d in corpus.documents])
        # Cold cost, read from the cache so a replayed run reports what the first
        # run paid.
        prompts = [p for doc_id, text in pairs for p in window_prompts(text, doc_id)]
        results[label] = {
            "model": model,
            "cost": ledger.to_dict(),
            "cold_calls": len(prompts),
            "cold_tokens": sum(cached_tokens(request_key(model, p, tag)) for p in prompts),
            "entity_jaccard": describe([r["entity_jaccard"] for r in rows]),
            "relation_jaccard": describe([r["relation_jaccard"] for r in rows]),
            "type_changes_on_shared_entities": sum(r["type_changes"] for r in rows),
            "shared_entities": sum(r["entities_kept"] for r in rows),
            "entities_per_post": describe([r["entities"] for r in rows]),
            "relations_per_post": describe([r["relations"] for r in rows]),
            "located_share": describe([r["located_share"] for r in rows]),
            "failed_calls": sum(r["failed_calls"] for r in rows),
            "whole_graph_if_swapped": mt.graph_diff(
                mt.fold_units(corpus.units, [d.doc_id for d in corpus.documents]), graph
            ).counts(),
            "rows": rows,
        }
    cold_calls: dict[str, int] = {}
    for arm in results.values():
        cold_calls[arm["model"]] = cold_calls.get(arm["model"], 0) + arm["cold_calls"]
    reference_counts = [
        {"entities": len(corpus.units[d].entities), "relations": len(corpus.units[d].relations)} for d in chosen
    ]
    return {
        "produced_by": drift_produced_by(drift_model, reference, cold_calls),
        "reference_model": reference,
        "drift_model": drift_model,
        "seed": SEED + 1,
        "documents": DRIFT_SAMPLE,
        "reference_entities_per_post": describe([r["entities"] for r in reference_counts]),
        "reference_relations_per_post": describe([r["relations"] for r in reference_counts]),
        **results,
    }


# ---------------------------------------------------------------------------
# LightRAG and HippoRAG 2
# ---------------------------------------------------------------------------

LIGHTRAG_DELETE = "musique::00004"
LIGHTRAG_EDIT = "musique::00002"
LIGHTRAG_MERGE = (["Barcelona", "FC Barcelona"], "FC Barcelona")
# The edit is this script's own change to one sentence of the passage, so the
# re-extraction has something to find.
LIGHTRAG_EDIT_FIND = "Maradona"
LIGHTRAG_EDIT_REPLACE = "Diego Armando Maradona"


def _lightrag_snapshot(directory: Path) -> dict[str, Any]:
    import networkx as nx  # noqa: PLC0415

    graph = nx.read_graphml(directory / "graph_chunk_entity_relation.graphml")
    status = json.loads((directory / "kv_store_doc_status.json").read_text())
    return {
        "nodes": sorted(graph.nodes()),
        "edges": sorted(sorted([str(a), str(b)]) for a, b in graph.edges()),
        "descriptions": {str(n): str(d.get("description", "")) for n, d in graph.nodes(data=True)},
        "documents": sorted(status),
        "chunks": sum(int(v.get("chunks_count", 0)) for v in status.values()),
    }


def _lightrag_change(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    nodes_b, nodes_a = set(before["nodes"]), set(after["nodes"])
    edges_b = {tuple(e) for e in before["edges"]}
    edges_a = {tuple(e) for e in after["edges"]}
    rewritten = sorted(n for n in nodes_a & nodes_b if before["descriptions"][n] != after["descriptions"][n])
    return {
        "nodes": [len(nodes_b), len(nodes_a)],
        "edges": [len(edges_b), len(edges_a)],
        "chunks": [before["chunks"], after["chunks"]],
        "documents": [len(before["documents"]), len(after["documents"])],
        "nodes_removed": sorted(nodes_b - nodes_a),
        "nodes_added": sorted(nodes_a - nodes_b),
        "descriptions_rewritten": rewritten,
        "edges_removed": len(edges_b - edges_a),
        "edges_added": len(edges_a - edges_b),
    }


async def _lightrag(rpm: int, model: str) -> dict[str, Any]:
    import lightrag  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder  # noqa: PLC0415
    from lightrag import LightRAG, QueryParam  # noqa: PLC0415
    from lightrag.utils import EmbeddingFunc  # noqa: PLC0415
    from openai import AsyncOpenAI  # noqa: PLC0415

    source = Path(cache_root()) / "graphrag" / "musique" / "lightrag-5"
    directory = Path(cache_root()) / "graphrag" / "musique" / "lightrag-5-maintenance"
    shutil.rmtree(directory, ignore_errors=True)
    shutil.copytree(source, directory)
    ledger = CallLedger("lightrag-maintenance")
    llm = async_llm(
        AsyncOpenAI(base_url=BASE_URL, api_key=api_key(), max_retries=0, timeout=600.0),
        model,
        cache=ResponseCache(Path(cache_root()) / "graphrag" / "musique" / "responses-maintenance.sqlite"),
        limiter=RateLimiter(per_minute=rpm),
        ledger=ledger,
    )
    encoder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")

    async def embed(texts: Sequence[str]) -> Any:
        return np.array(encoder.encode(list(texts)), dtype=np.float32)

    rag = LightRAG(
        working_dir=str(directory),
        llm_model_func=llm,
        llm_model_name=model,
        embedding_func=EmbeddingFunc(embedding_dim=encoder.dimension(), max_token_size=512, func=embed),
        chunk_token_size=300,
        chunk_overlap_token_size=50,
    )
    await rag.initialize_storages()

    async def measured(label: str, action: Callable[[], Any]) -> dict[str, Any]:
        before = _lightrag_snapshot(directory)
        calls, tokens = ledger.calls + ledger.cache_hits, ledger.total_tokens
        began = time.perf_counter()
        result = await action()
        seconds = time.perf_counter() - began
        await rag.finalize_storages()
        await rag.initialize_storages()
        after = _lightrag_snapshot(directory)
        return {
            "operation": label,
            "result": str(getattr(result, "status", "")) or ("ok" if result is not None else ""),
            "model_calls": ledger.calls + ledger.cache_hits - calls,
            "tokens": ledger.total_tokens - tokens,
            "seconds": round(seconds, 1),
            **_lightrag_change(before, after),
        }

    steps = [await measured(f"adelete_by_doc_id({LIGHTRAG_DELETE})", lambda: rag.adelete_by_doc_id(LIGHTRAG_DELETE))]
    sources, target = LIGHTRAG_MERGE
    steps.append(
        await measured(f"amerge_entities({sources} -> {target})", lambda: rag.amerge_entities(sources, target))
    )

    status = json.loads((directory / "kv_store_full_docs.json").read_text())
    original = str(status[LIGHTRAG_EDIT]["content"])
    edited = original.replace(LIGHTRAG_EDIT_FIND, LIGHTRAG_EDIT_REPLACE, 1)

    async def edit() -> Any:
        await rag.adelete_by_doc_id(LIGHTRAG_EDIT)
        return await rag.ainsert(edited, ids=[LIGHTRAG_EDIT], file_paths=[LIGHTRAG_EDIT])

    steps.append(await measured(f"edit {LIGHTRAG_EDIT}: adelete_by_doc_id then ainsert", edit))

    question = "Which club did Jesús Aranguren spend his whole professional career with?"
    data = await rag.aquery_data(question, param=QueryParam(mode="mix", top_k=10, chunk_top_k=10))
    cited = sorted({str(c.get("file_path")) for c in (data.get("data") or {}).get("chunks") or []})
    await rag.finalize_storages()
    return {
        "lightrag_version": getattr(lightrag, "__version__", "unknown"),
        "model": model,
        "index": "graphrag/musique/lightrag-5, copied before any change",
        "steps": steps,
        "smoke": {"question": question, "cites": cited, "cites_deleted": LIGHTRAG_DELETE in cited},
        "cost": ledger.to_dict(),
    }


def step_lightrag(rpm: int, model: str = LIGHTRAG_MODEL) -> dict[str, Any]:
    """Delete, merge, and edit on a copy of the 5-passage LightRAG index."""
    return asyncio.run(_lightrag(rpm, model))


HIPPORAG_MODULES = (
    "atlas_rag.kg_construction.triple_extraction",
    "atlas_rag.kg_construction.triple_config",
    "atlas_rag.vectorstore.create_graph_index",
    "atlas_rag.vectorstore.embedding_model",
    "atlas_rag.retriever.hipporag2",
    "atlas_rag.retriever.base",
)


def step_hipporag() -> dict[str, Any]:
    """List what atlas-rag exposes for changing an index, and quote its build cost."""
    import importlib  # noqa: PLC0415
    import inspect  # noqa: PLC0415
    from importlib.metadata import version  # noqa: PLC0415

    words = ("delete", "remove", "update", "upsert", "incremental", "merge", "insert")
    found: list[str] = []
    # The modules a HippoRAG 2 build and query go through. Some other atlas-rag
    # modules parse the command line when imported, so the package is not walked.
    for module_name in HIPPORAG_MODULES:
        loaded = importlib.import_module(module_name)
        for name, member in inspect.getmembers(loaded):
            if not (getattr(member, "__module__", "") or "").startswith("atlas_rag"):
                continue
            candidates = [name]
            if inspect.isclass(member):
                candidates += [f"{name}.{m}" for m in vars(member) if not m.startswith("_")]
            found += [f"{module_name}:{c}" for c in candidates if any(w in c.lower().rsplit(".", 1)[-1] for w in words)]
    cost = json.loads((HERE / "outputs" / "musique_hipporag2_index_cost.json").read_text())
    return {
        "produced_by": model_free(
            "Lists the atlas-rag methods that could change an index; no model call. "
            "rebuild_cost quotes the HippoRAG 2 MuSiQue index build, which ran on "
            f"{cost.get('model')}, from musique_hipporag2_index_cost.json."
        ),
        "atlas_rag_version": version("atlas-rag"),
        "modules": list(HIPPORAG_MODULES),
        "searched_for": list(words),
        "matches": sorted(set(found)),
        "rebuild_cost": {k: cost.get(k) for k in ("model", "passages", "cost", "calls_per_passage", "wall_seconds")},
    }


STEPS = ("retraction", "dedup", "gate", "revision", "drift", "lightrag", "hipporag")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the named steps and write their outputs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("steps", nargs="+", choices=STEPS)
    parser.add_argument("--rpm", type=int, default=DEFAULT_RPM)
    args = parser.parse_args(argv)
    corpus: Corpus | None = None
    for step in args.steps:
        began = time.perf_counter()
        if step in ("retraction", "dedup", "gate", "revision", "drift") and corpus is None:
            corpus = Corpus()
        if step == "retraction" and corpus:
            payload = step_retraction(corpus)
        elif step == "dedup" and corpus:
            payload = step_dedup(corpus)
        elif step == "gate" and corpus:
            payload = step_gate(corpus)
        elif step == "revision" and corpus:
            payload = step_revision(corpus, args.rpm)
        elif step == "drift" and corpus:
            payload = step_drift(corpus, args.rpm)
        elif step == "lightrag":
            payload = step_lightrag(args.rpm)
        else:
            payload = step_hipporag()
        path = write(f"{step}.json", payload)
        print(f"{step}: {path.name} in {time.perf_counter() - began:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
