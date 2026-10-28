#!/usr/bin/env python3
r"""Build the same corpus several times and measure how far the graphs move.

Every build here sends every call to the model: no response cache sits in front of
an extraction call, so a repeated build is a fresh generation and never a replay.
Raw responses are written to ``records.jsonl`` as they arrive, and an interrupted
build resumes by skipping the documents already written. LightRAG's builds go
through the shared proxy client, which needs a response cache, so each LightRAG
build gets its own empty cache file and its cache hits are reported.

Evaluation is different: it answers fixed questions over a finished graph, so its
calls share one cache, ``eval-responses.sqlite``, and the reader is always
``Qwen3_6-35B-A3B`` at temperature 0 whichever model built the graph.

Subcommands
-----------
``sf-build``
    Extract 100 ServerFault posts with the schema prompt, the same prompt with free
    types, or atlas-rag's triple prompt.
``mq-build`` and ``mq-index``
    Extract the MuSiQue sub-corpus with atlas-rag's prompt or the closed schema,
    then index the finished build for HippoRAG 2.
``mq-greedy``
    Send the same extraction prompts several times at temperature 0, plain, with a
    fixed seed, and with ``top_k`` 1, to see whether any setting makes the server
    repeat itself.
``mq-derive``
    Write and index a graph that needs no model: the reference build's triples for
    the sub-corpus, a title-mention graph, a union of builds, or a build with a
    share of its triples removed.
``mq-eval``
    Rank and answer the 100 questions over one graph with HippoRAG 2, or over the
    sub-corpus with dense retrieval.
``lr-build`` and ``lr-eval``
    Build LightRAG over the 60-question sub-corpus in one insert or five, and
    evaluate it.
``report``
    Gather every build and evaluation into ``outputs/graph_build_stability.json``.

Examples
--------
::

    $ python run_stability.py mq-build --builder atlas \
        --model Qwen3_6-35B-A3B --temperature 0.7 --rep 1
    $ python run_stability.py mq-eval --build atlas-Qwen3_6-35B-A3B-t0.7-r1
    $ python run_stability.py report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RI4 = HERE.parents[1] / "graphrag_technical_support"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RI4))

import stability as st  # noqa: E402


#: The reader for every evaluation, so only the graph changes between rows.
READER = "Qwen3_6-35B-A3B"
#: Questions per hop count in the MuSiQue sub-corpus, taken in order from the
#: front of RI #4's stratified test set.
SUBSET_PER_HOP = {2: 50, 3: 25, 4: 25}
#: The smaller sub-corpus LightRAG is built over, the first questions of the above.
LIGHTRAG_PER_HOP = {2: 30, 3: 15, 4: 15}
SERVERFAULT_POSTS = 100
SERVERFAULT_SEED = 33
KEYWORD = "musique"
EXTRACTION_MAX_TOKENS = 4096
SERVERFAULT_MAX_TOKENS = 2000
VALIDATION_ATTEMPTS = 3
ANSWER_PASSAGES = 5
RANK_DEPTH = 10
OUT = HERE / "outputs" / "graph_build_stability.json"


def work_root() -> Path:
    """Return where builds, indexes, and evaluations live, outside the repository."""
    from aieng.advanced_rag.cache.paths import root  # noqa: PLC0415

    return Path(root()) / "special_topics" / "graph_build_stability"


def build_name(builder: str, model: str, temperature: float | None, rep: int) -> str:
    """Return a build's directory name, which names everything that varies."""
    temp = "default" if temperature is None else f"{temperature:g}"
    return f"{builder}-{model}-t{temp}-r{rep}"


# ---------------------------------------------------------------------------
# One uncached, throttled, metered call
# ---------------------------------------------------------------------------


class Caller:
    """Send chat requests to one model with no cache, under a request budget.

    Every call reaches the model. Calls go through
    :class:`aieng.advanced_rag.batch.ProxyBatch`, which retries only transient
    errors and reports each completed call to the usage meter, which the spend
    ceiling reads.

    Parameters
    ----------
    model : str
        The model name.
    rpm : int
        Requests allowed per minute.
    """

    def __init__(self, model: str, rpm: int) -> None:
        from aieng.advanced_rag.batch import ProxyBatch  # noqa: PLC0415
        from build_indexes import BASE_URL, api_key  # noqa: PLC0415

        self.model = model
        self.batch = ProxyBatch(model, rpm=rpm, timeout=600.0, max_retries=12, base_url=BASE_URL, api_key=api_key())
        self.ledger = self.batch.ledger

    def chat(self, messages: list[dict[str, str]], *, temperature: float | None, max_tokens: int) -> str:
        """Return the model's text for one request.

        Raises
        ------
        RuntimeError
            When the request fails, after any retries of a transient error.
        """
        from aieng.advanced_rag.spend import SpendLimitReached  # noqa: PLC0415

        params: dict[str, Any] = {"max_tokens": max_tokens}
        if temperature is not None:
            params["temperature"] = temperature
        try:
            outcome = self.batch.create(messages, **params)
        except SpendLimitReached:
            raise
        except Exception as error:
            raise RuntimeError(f"{self.model}: call failed: {error}") from error
        return outcome.text or ""


def temperature_sent(model: str, temperature: float | None) -> float | None:
    """Return the temperature a request carries, or None when the family drops it."""
    from aieng.advanced_rag.models import request_params  # noqa: PLC0415

    sent = request_params(model, max_tokens=16, temperature=temperature).get("temperature")
    return None if sent is None else float(sent)


def run_jobs(
    jobs: Sequence[Any],
    work: Callable[[Any], dict[str, Any]],
    out: Path,
    *,
    workers: int,
    key: Callable[[Any], str],
) -> None:
    """Run jobs concurrently, appending each result to ``out`` as one JSON line.

    A job whose key is already in ``out`` is skipped, so a stopped build resumes.
    """
    done: set[str] = set()
    if out.is_file():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["key"])
    todo = [j for j in jobs if key(j) not in done]
    lock = threading.Lock()
    finished = [0]
    print(f"{len(todo)} of {len(jobs)} jobs to run", flush=True)
    with out.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(workers) as pool:

        def one(job: Any) -> None:
            result = {"key": key(job), **work(job)}
            with lock:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                finished[0] += 1
                if finished[0] % 100 == 0:
                    print(f"  {finished[0]}/{len(todo)}", flush=True)

        list(pool.map(one, todo))


def write_meta(directory: Path, caller: Caller, extra: Mapping[str, Any], seconds: float) -> None:
    """Write a build's cost and settings, added to an earlier run's figures."""
    path = directory / "meta.json"
    old = json.loads(path.read_text()) if path.is_file() else {}
    cost = caller.ledger.to_dict()
    previous = old.get("cost", {})
    for k in ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "failed_attempts", "empty_responses"):
        cost[k] = cost.get(k, 0) + previous.get(k, 0)
    meta = {
        **old,
        **extra,
        "model": caller.model,
        "cost": cost,
        "wall_seconds": round(old.get("wall_seconds", 0) + seconds, 1),
        "response_cache": "none: every call reached the model",
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(meta, indent=1) + "\n")


# ---------------------------------------------------------------------------
# ServerFault
# ---------------------------------------------------------------------------


def serverfault_jobs() -> list[tuple[str, str, int, list[Any]]]:
    """Return the windows of the 100 sampled posts, as ``build_graph`` cuts them."""
    from aieng.advanced_rag.corpora import serverfault  # noqa: PLC0415
    from aieng.advanced_rag.graphrag import extraction  # noqa: PLC0415

    docs = serverfault.documents(serverfault.fetch_slice(serverfault.SLICE_10K_QUESTIONS))
    chosen = random.Random(SERVERFAULT_SEED).sample(docs, SERVERFAULT_POSTS)
    jobs = []
    for doc in chosen:
        found = extraction.sections(doc.doc_id, doc.text)
        head = [b for name, blocks in found if name == "Title" for b in blocks]
        for name, blocks in found:
            if name == "Title" and any(n == "Question" for n, _ in found):
                continue
            body = head + blocks if name == "Question" else blocks
            for index, run in enumerate(extraction.windows(body)):
                jobs.append((doc.doc_id, name, index, run))
    return jobs


def serverfault_messages(arm: str, body: str) -> list[dict[str, str]]:
    """Return the request for one window under one arm."""
    from aieng.advanced_rag.graphrag import extraction as ex  # noqa: PLC0415
    from atlas_rag.llm_generator.prompt.triple_extraction_prompt import TRIPLE_INSTRUCTIONS  # noqa: PLC0415

    schema = ex.PROMPT.format(
        entity_types=", ".join(ex.ENTITY_TYPES), relation_types=", ".join(ex.RELATION_TYPES), text=body
    )
    if arm == "schema":
        return [{"role": "user", "content": f"{ex.SYSTEM}\n\n{schema}"}]
    if arm == "free":
        prompt = st.free_type_prompt(schema, ex.ENTITY_TYPES, ex.RELATION_TYPES)
        return [{"role": "user", "content": f"{ex.SYSTEM}\n\n{prompt}"}]
    if arm == "atlas":
        instr = TRIPLE_INSTRUCTIONS["en"]
        return [
            {"role": "system", "content": instr["system"]},
            {"role": "user", "content": instr["entity_relation"] + "\n" + body},
        ]
    raise ValueError(arm)


def cmd_sf_build(args: argparse.Namespace) -> None:
    """Extract the ServerFault sample once under one arm."""
    name = build_name(f"sf-{args.arm}", args.model, args.temperature, args.rep)
    directory = work_root() / "serverfault" / name
    directory.mkdir(parents=True, exist_ok=True)
    caller = Caller(args.model, args.rpm)
    jobs = serverfault_jobs()

    def work(job: tuple[str, str, int, list[Any]]) -> dict[str, Any]:
        doc_id, section, index, run = job
        body = "\n\n".join(b.text for b in run)
        try:
            raw = caller.chat(
                serverfault_messages(args.arm, body), temperature=args.temperature, max_tokens=SERVERFAULT_MAX_TOKENS
            )
            return {"doc_id": doc_id, "section": section, "window": index, "raw": raw, "failed": False}
        except RuntimeError:
            return {"doc_id": doc_id, "section": section, "window": index, "raw": "", "failed": True}

    began = time.perf_counter()
    run_jobs(jobs, work, directory / "records.jsonl", workers=args.workers, key=lambda j: f"{j[0]}|{j[1]}|{j[2]}")
    write_meta(
        directory,
        caller,
        {
            "arm": args.arm,
            "temperature_requested": args.temperature,
            "temperature_sent": temperature_sent(args.model, args.temperature),
            "posts": SERVERFAULT_POSTS,
            "windows": len(jobs),
        },
        time.perf_counter() - began,
    )


def serverfault_views(directory: Path, jobs: Sequence[tuple[str, str, int, list[Any]]]) -> dict[str, list[Any]]:
    """Return a ServerFault build's extractions, per post, under each way of reading it.

    ``raw`` keeps the model's own labels. ``folded`` applies the RI #4 schema fold:
    entity types through the alias map (unknown types become ``Concept``),
    relation types through the alias map, and a relation that maps to nothing or
    has an endpoint the model did not list is dropped. Both views of one build come
    from the same responses.
    """
    from aieng.advanced_rag.graphrag import extraction as ex  # noqa: PLC0415

    blocks = {(d, s, i): run for d, s, i, run in jobs}
    raw_by_doc: dict[str, dict[str, list[Any]]] = {}
    folded_by_doc: dict[str, dict[str, list[Any]]] = {}
    failed: dict[str, bool] = {}
    for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        doc = rec["doc_id"]
        raw_by_doc.setdefault(doc, {"entities": [], "triples": []})
        folded_by_doc.setdefault(doc, {"entities": [], "triples": []})
        failed.setdefault(doc, False)
        if rec["failed"]:
            failed[doc] = True
            continue
        if directory.name.startswith("sf-atlas"):
            parsed = atlas_parse(rec["raw"])
            if parsed is None:
                failed[doc] = True
                continue
            raw_by_doc[doc]["triples"] += [
                {"Head": t.get("Head", ""), "Relation": t.get("Relation", ""), "Tail": t.get("Tail", "")}
                for t in parsed
            ]
            continue
        payload = ex.parse_json_object(rec["raw"])
        if not payload:
            failed[doc] = True
            continue
        ents = [e for e in payload.get("entities") or [] if isinstance(e, dict)]
        rels = [r for r in payload.get("relations") or [] if isinstance(r, dict)]
        raw_by_doc[doc]["entities"] += [{"name": e.get("name", ""), "type": str(e.get("type", ""))} for e in ents]
        raw_by_doc[doc]["triples"] += [
            {"Head": r.get("source", ""), "Relation": str(r.get("type", "")), "Tail": r.get("target", "")} for r in rels
        ]
        graph = ex.KnowledgeGraph()
        ex._fold(graph, payload, blocks[(doc, rec["section"], rec["window"])], rec["section"], doc)  # noqa: SLF001
        folded_by_doc[doc]["entities"] += [{"name": e.name, "type": e.type} for e in graph.entities.values()]
        folded_by_doc[doc]["triples"] += [
            {"Head": r.source, "Relation": r.type, "Tail": r.target} for r in graph.relations.values()
        ]
    views: dict[str, list[Any]] = {"raw": [], "folded": []}
    for doc in raw_by_doc:
        views["raw"].append(
            st.Extraction.from_raw(
                doc, raw_by_doc[doc]["triples"], entities=raw_by_doc[doc]["entities"], failed=failed[doc]
            )
        )
        views["folded"].append(
            st.Extraction.from_raw(
                doc, folded_by_doc[doc]["triples"], entities=folded_by_doc[doc]["entities"], failed=failed[doc]
            )
        )
    return views


def schema_conformance(directory: Path) -> dict[str, Any]:
    """Return the share of a ServerFault build's raw labels on the RI #4 schema."""
    from aieng.advanced_rag.graphrag import extraction as ex  # noqa: PLC0415

    ent_total = ent_exact = ent_alias = rel_total = rel_exact = rel_alias = 0
    for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        payload = ex.parse_json_object(rec["raw"]) if rec["raw"] else {}
        for e in payload.get("entities") or []:
            if isinstance(e, dict):
                label = str(e.get("type", ""))
                ent_total += 1
                ent_exact += label in ex.ENTITY_TYPES
                ent_alias += not ex.canonical_entity_type(label)[1]
        for r in payload.get("relations") or []:
            if isinstance(r, dict):
                label = str(r.get("type", ""))
                rel_total += 1
                rel_exact += label in ex.RELATION_TYPES
                rel_alias += ex.canonical_relation_type(label) is not None
    return {
        "entity_labels": ent_total,
        "entity_on_schema": round(ent_exact / ent_total, 4) if ent_total else None,
        "entity_on_schema_after_alias": round(ent_alias / ent_total, 4) if ent_total else None,
        "relation_labels": rel_total,
        "relation_on_schema": round(rel_exact / rel_total, 4) if rel_total else None,
        "relation_on_schema_after_alias": round(rel_alias / rel_total, 4) if rel_total else None,
    }


# ---------------------------------------------------------------------------
# MuSiQue sub-corpus
# ---------------------------------------------------------------------------


def musique_subset(per_hop: Mapping[int, int] = SUBSET_PER_HOP) -> tuple[list[Any], list[Any]]:
    """Return the sub-corpus questions and every candidate paragraph they list.

    Questions come from the front of RI #4's stratified test set, by hop count. The
    passages are the union of the roughly twenty candidate paragraphs each question
    lists in the packaging, supporting and distracting, in corpus order.
    """
    from aieng.advanced_rag.corpora.musique import fetch_musique, load_musique, stratified_test_set  # noqa: PLC0415

    passages, questions = load_musique()
    test = stratified_test_set(questions)
    chosen = [q for hops, n in sorted(per_hop.items()) for q in [x for x in test if x.hops == hops][:n]]
    raw = {r["id"]: r for r in json.loads(fetch_musique().questions.read_text(encoding="utf-8"))}
    by_text = {(p.title, p.text): p.passage_id for p in passages}
    wanted: set[str] = set()
    for q in chosen:
        for para in raw[q.question_id]["paragraphs"]:
            wanted.add(by_text[(str(para["title"]), str(para["paragraph_text"]))])
    return chosen, [p for p in passages if p.passage_id in wanted]


def passage_text(passage: Any) -> str:
    """Return the text atlas-rag indexes: title, newline, passage."""
    return f"{passage.title}\n{passage.text}"


def atlas_parse(raw: str) -> list[dict[str, Any]] | None:
    """Parse and validate one triple response as atlas-rag does, or None on failure."""
    import json_repair  # noqa: PLC0415
    from atlas_rag.llm_generator.format.validate_json_output import (  # noqa: PLC0415
        fix_triple_extraction_response,
        validate_output,
    )
    from atlas_rag.llm_generator.format.validate_json_schema import ATLAS_SCHEMA  # noqa: PLC0415

    if "</think>" in raw:
        raw = raw.rsplit("</think>", maxsplit=1)[-1].strip()
    try:
        text = validate_output(
            raw,
            schema=ATLAS_SCHEMA["entity_relation"],
            fix_function=fix_triple_extraction_response,
            allow_empty=True,
        )
    except Exception:  # noqa: BLE001 - atlas-rag treats any validation failure as a retry
        return None
    loaded = json_repair.loads(text)
    return [t for t in loaded if isinstance(t, dict)] if isinstance(loaded, list) else None


def musique_messages(builder: str, text: str) -> list[dict[str, str]]:
    """Return atlas-rag's request for one passage, with the schema added when asked."""
    from atlas_rag.llm_generator.prompt.triple_extraction_prompt import TRIPLE_INSTRUCTIONS  # noqa: PLC0415

    instr = TRIPLE_INSTRUCTIONS["en"]
    body = instr["entity_relation"] if builder == "atlas" else st.schema_instruction(instr["entity_relation"])
    return [{"role": "system", "content": instr["system"]}, {"role": "user", "content": body + "\n" + text}]


def cmd_mq_build(args: argparse.Namespace) -> None:
    """Extract the sub-corpus once and index it."""
    from aieng.advanced_rag import spend  # noqa: PLC0415
    from aieng.advanced_rag.models import endpoint_of  # noqa: PLC0415

    paid = endpoint_of(args.model) != "vector-cluster"
    if paid:
        spend.session()
    prefix = "renamed-" if args.renamed else ""
    name = build_name(prefix + args.builder, args.model, args.temperature, args.rep)
    directory = work_root() / "musique" / ("renamed" if args.renamed else "builds") / name
    directory.mkdir(parents=True, exist_ok=True)
    _, passages = musique_subset()
    if args.limit:
        passages = passages[: args.limit]
    caller = Caller(args.model, args.rpm)
    mapping = st.invent_names([p.title for p in passages if len(p.title) >= 4]) if args.renamed else {}

    def work(passage: Any) -> dict[str, Any]:
        text = st.rename_text(passage_text(passage), mapping) if mapping else passage_text(passage)
        raws: list[str] = []
        for _ in range(VALIDATION_ATTEMPTS):
            try:
                raw = caller.chat(
                    musique_messages(args.builder, text),
                    temperature=args.temperature,
                    max_tokens=EXTRACTION_MAX_TOKENS,
                )
            except RuntimeError:
                raws.append("")
                continue
            raws.append(raw)
            parsed = atlas_parse(raw)
            if parsed is not None:
                triples = [
                    {
                        "Head": str(t.get("Head", "")),
                        "Relation": str(t.get("Relation", "")),
                        "Tail": str(t.get("Tail", "")),
                    }
                    for t in parsed
                ]
                if args.builder == "schema":
                    folded = []
                    for t in triples:
                        kind = st.fold_relation(t["Relation"])
                        if kind is not None:
                            folded.append({**t, "Relation": kind})
                    return {
                        **st.atlas_record(passage.passage_id, text, folded, raw),
                        "raw_triples": triples,
                        "attempts": len(raws),
                        "failed": False,
                    }
                return {
                    **st.atlas_record(passage.passage_id, text, triples, raw),
                    "attempts": len(raws),
                    "failed": False,
                }
        return {
            **st.atlas_record(passage.passage_id, text, [], raws[-1] if raws else ""),
            "attempts": len(raws),
            "failed": True,
        }

    began = time.perf_counter()
    run_jobs(passages, work, directory / "records.jsonl", workers=args.workers, key=lambda p: p.passage_id)
    write_meta(
        directory,
        caller,
        {
            "builder": args.builder,
            "temperature_requested": args.temperature,
            "temperature_sent": temperature_sent(args.model, args.temperature),
            "passages": len(passages),
            "endpoint": endpoint_of(args.model),
        },
        time.perf_counter() - began,
    )
    if paid:
        print(spend.session().report())


#: Request settings the greedy probe compares, each sent at temperature 0 except
#: the last, which shows what a seed does when sampling is on.
GREEDY_ARMS: dict[str, tuple[float, dict[str, Any]]] = {
    "t0": (0.0, {}),
    "t0_seed7": (0.0, {"seed": 7}),
    "t0_topk1": (0.0, {"top_k": 1}),
    "t0.7_seed7": (0.7, {"seed": 7}),
}


def cmd_mq_greedy(args: argparse.Namespace) -> None:
    """Repeat a few extraction prompts under each greedy setting and save them."""
    from aieng.advanced_rag.batch import ProxyBatch  # noqa: PLC0415
    from build_indexes import BASE_URL, api_key  # noqa: PLC0415

    _, passages = musique_subset()
    chosen = [passages[i] for i in args.passages]
    # No cache: the probe measures what repeated identical requests return.
    batch = ProxyBatch(READER, timeout=900.0, max_retries=12, base_url=BASE_URL, api_key=api_key())

    def one(job: tuple[str, Any, int]) -> dict[str, Any]:
        arm, passage, repeat = job
        temperature, extra = GREEDY_ARMS[arm]
        outcome = batch.create(
            musique_messages("atlas", passage_text(passage)),
            max_tokens=EXTRACTION_MAX_TOKENS,
            temperature=temperature,
            **({"extra_body": extra} if extra else {}),
        )
        text = outcome.text or ""
        return {"arm": arm, "passage_id": passage.passage_id, "repeat": repeat, "response": text}

    jobs = [(arm, p, r) for arm in GREEDY_ARMS for p in chosen for r in range(args.repeats)]
    with batch, ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(one, jobs))
    target = work_root() / "musique" / "greedy" / "responses.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    print(f"wrote {len(results)} responses to {target}")


def cmd_mq_index(args: argparse.Namespace) -> None:
    """Index a finished build for HippoRAG 2."""
    print(index_records(work_root() / "musique" / "builds" / args.build))


def _device() -> str:
    """Return ``cuda`` when the local torch build sees a GPU, else ``cpu``."""
    import torch  # noqa: PLC0415

    return "cuda" if torch.cuda.is_available() else "cpu"


def read_records(directory: Path) -> list[dict[str, Any]]:
    """Return a build's records in corpus order."""
    records = [json.loads(x) for x in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines() if x]
    return sorted(records, key=lambda r: r["id"])


def is_indexed(directory: Path) -> bool:
    """Return whether a build's index finished: the edge index is written last."""
    return any((directory / "index" / "precompute").glob("*edge_faiss.index"))


def index_records(directory: Path) -> Path:
    """Turn a build's atlas-rag records into a HippoRAG 2 index under ``index/``.

    The records go through atlas-rag's own CSV, GraphML, and embedding steps with
    the reference build's prompt and schema files, so the index has the reference
    layout and the RI #4 adapter reads it unchanged.
    """
    from atlas_rag.kg_construction.triple_config import ProcessingConfig  # noqa: PLC0415
    from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor  # noqa: PLC0415
    from atlas_rag.llm_generator.format.validate_json_schema import ATLAS_SCHEMA  # noqa: PLC0415
    from atlas_rag.llm_generator.prompt.triple_extraction_prompt import TRIPLE_INSTRUCTIONS  # noqa: PLC0415
    from atlas_rag.vectorstore import create_embeddings_and_index  # noqa: PLC0415
    from atlas_rag.vectorstore.embedding_model import SentenceEmbedding  # noqa: PLC0415
    from build_indexes import ENCODER  # noqa: PLC0415
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    index = directory / "index"
    if is_indexed(directory):
        return index
    (index / "kg_extraction").mkdir(parents=True, exist_ok=True)
    with (index / "kg_extraction" / f"stability_{KEYWORD}_output_1_in_1.json").open("w", encoding="utf-8") as fh:
        for rec in read_records(directory):
            keep = {
                k: rec[k] for k in ("id", "original_text", "metadata", "entity_relation_dict", "entity_relation_output")
            }
            fh.write(json.dumps(keep, ensure_ascii=False) + "\n")
    prompt = index / "prompt.json"
    schema = index / "schema.json"
    prompt.write_text(json.dumps({"en": {k: TRIPLE_INSTRUCTIONS["en"][k] for k in ("system", "entity_relation")}}))
    schema.write_text(json.dumps({"entity_relation": ATLAS_SCHEMA["entity_relation"]}))

    class _Stub:
        model_name = "stability"

    config = ProcessingConfig(
        model_path="stability",
        data_directory=str(index),
        filename_pattern=KEYWORD,
        output_directory=str(index),
        include_concept=False,
        triple_extraction_prompt_path=str(prompt),
        triple_extraction_schema_path=str(schema),
    )
    extractor = KnowledgeGraphExtractor(model=_Stub(), config=config)
    extractor.convert_json_to_csv()
    extractor.convert_to_graphml()
    # The encoder is local either way. A GPU makes the 50,000 node and edge
    # embeddings of one index take seconds instead of most of an hour.
    encoder = SentenceEmbedding(SentenceTransformer(ENCODER, device=_device()))
    create_embeddings_and_index(
        sentence_encoder=encoder,
        model_name=ENCODER,
        working_directory=str(index),
        keyword=KEYWORD,
        include_concept=False,
        include_events=False,
        normalize_embeddings=True,
    )
    return index


def reference_records(passage_ids: set[str]) -> list[dict[str, Any]]:
    """Return the reference HippoRAG 2 build's records for the sub-corpus passages.

    The reference build extracted every passage of the full corpus on 2026-10-06,
    one call per passage, so its records for these passages are a build of this
    sub-corpus made sixteen days before the others. It is read, never written.
    """
    from build_indexes import work_root as ri4_root  # noqa: PLC0415

    found: dict[str, dict[str, Any]] = {}
    for path in sorted((ri4_root() / "hipporag2" / "kg_extraction").glob("*.json")):
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["id"] in passage_ids:
                found[rec["id"]] = rec
    return [found[k] for k in sorted(found)]


def cmd_mq_derive(args: argparse.Namespace) -> None:
    """Write and index a graph that needs no model calls."""
    _, passages = musique_subset()
    ids = {p.passage_id for p in passages}
    root = work_root() / "musique" / "builds"
    if args.kind == "reference":
        name, records = "atlas-Qwen3_6-35B-A3B-t0.7-r0", reference_records(ids)
    elif args.kind == "titles":
        name = "titles"
        records = st.title_mention_records([(p.passage_id, p.title, p.text) for p in passages])
    elif args.kind == "union":
        name = "union-" + "+".join(b.split("-")[-1] for b in args.builds)
        records = st.union_records([read_records(root / b) for b in args.builds])
    elif args.kind == "drop":
        name = f"drop{int(args.fraction * 100)}-{args.builds[0]}"
        records = st.drop_triples(read_records(root / args.builds[0]), args.fraction, seed=args.seed)
    else:
        raise ValueError(args.kind)
    if len(records) != len(ids):
        raise SystemExit(f"{name}: {len(records)} records for {len(ids)} passages")
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "records.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps({"key": rec["id"], **rec, "failed": False}, ensure_ascii=False) + "\n")
    (directory / "meta.json").write_text(
        json.dumps({"derived": args.kind, "from": args.builds, "fraction": args.fraction, "model_calls": 0}, indent=1)
        + "\n"
    )
    index_records(directory)
    print(f"wrote {directory}")


def eval_client(rpm: int) -> tuple[Any, Callable[[str], str], Any]:
    """Return the governed reader client, its answer function, and its ledger."""
    from aieng.advanced_rag.graphrag.proxy import CallLedger, RateLimiter, ResponseCache, govern  # noqa: PLC0415
    from aieng.advanced_rag.models import request_params  # noqa: PLC0415
    from build_indexes import BASE_URL, api_key  # noqa: PLC0415
    from openai import OpenAI  # noqa: PLC0415

    ledger = CallLedger("eval")
    client = govern(
        OpenAI(base_url=BASE_URL, api_key=api_key(), max_retries=0, timeout=600.0),
        cache=ResponseCache(work_root() / "musique" / "eval-responses.sqlite"),
        limiter=RateLimiter(per_minute=rpm),
        ledger=ledger,
        max_retries=16,
    )
    params = request_params(READER, max_tokens=256, temperature=0.0)
    params.pop("extra_body", None)

    def complete(prompt: str) -> str:
        response = client.chat.completions.create(
            model=READER, messages=[{"role": "user", "content": prompt}], **params
        )
        return str(response.choices[0].message.content or "")

    return client, complete, ledger


async def score_arm(
    arm: Any, questions: Sequence[Any], complete: Callable[[str], str], text_of: Callable[[str], str], workers: int
) -> list[Any]:
    """Rank and answer every question with one arm, returning RI #4's rows."""
    from aieng.advanced_rag.evaluation.multihop import exact_match, token_f1  # noqa: PLC0415
    from musique_arms import answer  # noqa: PLC0415
    from musique_report import Row  # noqa: PLC0415

    gate = asyncio.Semaphore(workers)

    async def one(q: Any) -> Any:
        async with gate:
            ranked = await arm.rank(q.question, RANK_DEPTH)
            reply = await answer(complete, q.question, ranked[:ANSWER_PASSAGES], text_of)
        golds = (q.answer, *q.answer_aliases)
        return Row(q.question_id, q.hops, tuple(ranked), exact_match(reply, golds), token_f1(reply, golds))

    return list(await asyncio.gather(*(one(q) for q in questions)))


class ExactDenseArm:
    """Cosine ranking over the sub-corpus by exact scan, ties broken by identifier.

    RI #4's dense arm keeps its vectors in an in-process Qdrant collection, which
    also scans exactly; this arm needs only numpy, so it runs in any environment
    that has the encoder.
    """

    name = "dense"

    def __init__(self, passages: Sequence[Any], embedder: Any) -> None:
        import numpy as np  # noqa: PLC0415

        self._embedder = embedder
        self._ids = [p.passage_id for p in passages]
        matrix = np.asarray(embedder.encode([passage_text(p) for p in passages]), dtype=np.float32)
        self._matrix = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9, None)

    async def rank(self, question: str, k: int) -> list[str]:
        """Return the ``k`` passages most similar to the question."""
        import numpy as np  # noqa: PLC0415

        query = np.asarray(self._embedder.encode_queries([question])[0], dtype=np.float32)
        scores = self._matrix @ (query / max(float(np.linalg.norm(query)), 1e-9))
        order = sorted(range(len(self._ids)), key=lambda i: (-float(scores[i]), self._ids[i]))
        return [self._ids[i] for i in order[:k]]


def cmd_mq_eval(args: argparse.Namespace) -> None:
    """Evaluate one graph, or dense retrieval, on the sub-corpus questions."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from aieng.advanced_rag.graphrag.adapters.hipporag2 import HippoRAG2Retriever  # noqa: PLC0415
    from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder  # noqa: PLC0415
    from build_indexes import ENCODER  # noqa: PLC0415
    from musique_arms import SyncEvidenceArm, passage_id_map  # noqa: PLC0415
    from musique_report import summarise  # noqa: PLC0415

    per_hop = LIGHTRAG_PER_HOP if args.small else SUBSET_PER_HOP
    questions, passages = musique_subset(per_hop)
    by_id = {p.passage_id: p for p in passages}
    client, complete, ledger = eval_client(args.rpm)
    if args.build == "dense":
        arm: Any = ExactDenseArm(passages, SentenceTransformerEmbedder(ENCODER, device=_device()))
    else:
        index = work_root() / "musique" / "builds" / args.build / "index"
        retriever = HippoRAG2Retriever.build(
            kg_dir=str(index),
            llm_client=client,
            llm_model=READER,
            encoder_model=ENCODER,
            keyword=KEYWORD,
            include_concept=False,
            include_events=False,
        )
        all_passages, _ = __import__("aieng.advanced_rag.corpora.musique", fromlist=["load_musique"]).load_musique()
        arm = SyncEvidenceArm(
            "hipporag2",
            retriever,
            id_map=passage_id_map(index / "kg_graphml" / f"{KEYWORD}_graph.graphml", all_passages),
        )
    began = time.perf_counter()
    rows = asyncio.run(score_arm(arm, questions, complete, lambda i: passage_text(by_id[i]), args.workers))
    gold = {q.question_id: q.supporting_ids for q in questions}
    result = {
        "build": args.build,
        "reader": READER,
        "questions": len(questions),
        "passages": len(passages),
        "summary": summarise(rows, gold),
        "cost": {**ledger.to_dict(), "wall_seconds": round(time.perf_counter() - began, 1)},
        "rows": [r.to_dict() for r in rows],
    }
    target = work_root() / "musique" / "eval" / f"{args.build}{'-small' if args.small else ''}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=1) + "\n")
    s = result["summary"]["all"]
    print(
        f"{args.build}: all@5 {s['all_supporting_recall_at_5']:.3f} all@10 {s['all_supporting_recall_at_10']:.3f} "
        f"F1 {s['f1']:.3f} calls {ledger.calls} hits {ledger.cache_hits}"
    )


# ---------------------------------------------------------------------------
# LightRAG
# ---------------------------------------------------------------------------

LIGHTRAG_BATCHES = 5


async def build_lightrag(
    directory: Path, passages: Sequence[Any], mode: str, model: str, rpm: int, workers: int
) -> Any:
    """Insert the passages into a fresh LightRAG store, in one insert or several."""
    from aieng.advanced_rag.graphrag.proxy import CallLedger, RateLimiter, ResponseCache, async_llm  # noqa: PLC0415
    from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder  # noqa: PLC0415
    from build_indexes import (  # noqa: PLC0415
        BASE_URL,
        ENCODER,
        LIGHTRAG_CHUNK_OVERLAP,
        LIGHTRAG_CHUNK_TOKENS,
        api_key,
        embedder_function,
        throttle_flushes,
    )
    from lightrag import LightRAG  # noqa: PLC0415
    from openai import AsyncOpenAI  # noqa: PLC0415

    cache_file = directory / "responses.sqlite"
    ledger = CallLedger("lightrag")
    llm = async_llm(
        AsyncOpenAI(base_url=BASE_URL, api_key=api_key(), max_retries=0, timeout=600.0),
        model,
        cache=ResponseCache(cache_file),
        limiter=RateLimiter(per_minute=rpm),
        ledger=ledger,
        max_retries=16,
    )
    store = directory / "store"
    store.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(store),
        llm_model_func=llm,
        llm_model_name=model,
        embedding_func=embedder_function(SentenceTransformerEmbedder(ENCODER)),
        chunk_token_size=LIGHTRAG_CHUNK_TOKENS,
        chunk_overlap_token_size=LIGHTRAG_CHUNK_OVERLAP,
        llm_model_max_async=workers,
        max_parallel_insert=max(2, workers // 2),
        default_embedding_timeout=600,
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
    )
    await rag.initialize_storages()
    flush_all = throttle_flushes(rag, 180.0)
    batches = [list(passages)]
    if mode == "incremental":
        size = -(-len(passages) // LIGHTRAG_BATCHES)
        batches = [list(passages[i : i + size]) for i in range(0, len(passages), size)]
    try:
        for n, batch in enumerate(batches, start=1):
            ids = [p.passage_id for p in batch]
            await rag.ainsert([passage_text(p) for p in batch], ids=ids, file_paths=ids)
            await rag.apipeline_process_enqueue_documents()
            await flush_all()
            print(f"batch {n}/{len(batches)} inserted: {ledger.calls} calls", flush=True)
        # LightRAG marks a document failed when one extraction call fails and
        # leaves it out of the graph; one more pass re-processes those documents.
        await rag.apipeline_process_enqueue_documents()
    finally:
        await flush_all()
        await rag.finalize_storages()
    return ledger


def cmd_lr_build(args: argparse.Namespace) -> None:
    """Build LightRAG once over the 60-question sub-corpus."""
    name = build_name(f"lightrag-{args.mode}", args.model, 0.0, args.rep)
    directory = work_root() / "lightrag" / name
    if (directory / "meta.json").is_file():
        raise SystemExit(f"{directory} is built; a LightRAG rebuild needs a fresh directory")
    directory.mkdir(parents=True, exist_ok=True)
    _, passages = musique_subset(LIGHTRAG_PER_HOP)
    began = time.perf_counter()
    ledger = asyncio.run(build_lightrag(directory, passages, args.mode, args.model, args.rpm, args.workers))
    from build_indexes import document_statuses  # noqa: PLC0415

    meta = {
        "mode": args.mode,
        "model": args.model,
        "temperature_sent": 0.0,
        "passages": len(passages),
        "batches": 1 if args.mode == "once" else LIGHTRAG_BATCHES,
        "cost": ledger.to_dict(),
        "wall_seconds": round(time.perf_counter() - began, 1),
        "response_cache": "a fresh file per build; hits are repeats of a request within the build",
        "document_statuses": document_statuses(directory / "store"),
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


def lightrag_build(directory: Path) -> list[Any]:
    """Return a LightRAG store's graph as per-passage extractions.

    Each node and edge is attributed to every passage its ``file_path`` lists.
    LightRAG's edges are undirected and carry keywords rather than one label, so
    the head and tail are sorted and the relation is left empty.
    """
    import networkx as nx  # noqa: PLC0415

    graph = nx.read_graphml(directory / "store" / "graph_chunk_entity_relation.graphml")
    per_doc: dict[str, dict[str, list[Any]]] = {}

    def docs(data: Mapping[str, Any]) -> list[str]:
        return [d for d in str(data.get("file_path", "")).split("<SEP>") if d]

    for node, data in graph.nodes(data=True):
        for d in docs(data):
            per_doc.setdefault(d, {"entities": [], "triples": []})["entities"].append(
                {"name": node, "type": str(data.get("entity_type", ""))}
            )
    for a, b, data in graph.edges(data=True):
        head, tail = sorted((a, b))
        for d in docs(data):
            per_doc.setdefault(d, {"entities": [], "triples": []})["triples"].append(
                {"Head": head, "Relation": "", "Tail": tail}
            )
    return [st.Extraction.from_raw(d, v["triples"], entities=v["entities"]) for d, v in sorted(per_doc.items())]


def cmd_lr_eval(args: argparse.Namespace) -> None:
    """Evaluate one LightRAG build on the 60 questions."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from aieng.advanced_rag.graphrag.adapters.lightrag import LightRAGRetriever  # noqa: PLC0415
    from aieng.advanced_rag.graphrag.proxy import async_llm  # noqa: PLC0415
    from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder  # noqa: PLC0415
    from build_indexes import BASE_URL, ENCODER, api_key, embedder_function  # noqa: PLC0415
    from musique_arms import LightRagArm  # noqa: PLC0415
    from musique_report import summarise  # noqa: PLC0415
    from openai import AsyncOpenAI  # noqa: PLC0415

    questions, passages = musique_subset(LIGHTRAG_PER_HOP)
    by_id = {p.passage_id: p for p in passages}
    client, complete, ledger = eval_client(args.rpm)
    directory = work_root() / "lightrag" / args.build

    async def go() -> list[Any]:
        from aieng.advanced_rag.graphrag.proxy import RateLimiter, ResponseCache  # noqa: PLC0415

        llm = async_llm(
            AsyncOpenAI(base_url=BASE_URL, api_key=api_key(), max_retries=0, timeout=600.0),
            READER,
            cache=ResponseCache(work_root() / "musique" / "eval-responses.sqlite"),
            limiter=RateLimiter(per_minute=args.rpm),
            ledger=ledger,
            max_retries=16,
        )
        light = await LightRAGRetriever.build(
            str(directory / "store"), llm, embedder_function(SentenceTransformerEmbedder(ENCODER)), mode="mix"
        )
        return await score_arm(LightRagArm(light), questions, complete, lambda i: passage_text(by_id[i]), args.workers)

    began = time.perf_counter()
    rows = asyncio.run(go())
    gold = {q.question_id: q.supporting_ids for q in questions}
    result = {
        "build": args.build,
        "reader": READER,
        "questions": len(questions),
        "passages": len(passages),
        "summary": summarise(rows, gold),
        "cost": {**ledger.to_dict(), "wall_seconds": round(time.perf_counter() - began, 1)},
        "rows": [r.to_dict() for r in rows],
    }
    target = work_root() / "lightrag" / "eval" / f"{args.build}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=1) + "\n")
    s = result["summary"]["all"]
    print(f"{args.build}: all@5 {s['all_supporting_recall_at_5']:.3f} F1 {s['f1']:.3f}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0915
    """Parse the subcommand and run it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, rpm: int, workers: int) -> None:
        p.add_argument("--rpm", type=int, default=rpm)
        p.add_argument("--workers", type=int, default=workers)

    p = sub.add_parser("sf-build")
    p.add_argument("--arm", choices=("schema", "free", "atlas"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--rep", type=int, required=True)
    common(p, 50, 32)
    p.set_defaults(func=cmd_sf_build)

    p = sub.add_parser("mq-build")
    p.add_argument("--builder", choices=("atlas", "schema"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--limit", type=int, default=None, help="first N passages only, for a pilot")
    p.add_argument(
        "--renamed", action="store_true", help="replace every article title with an invented name; not indexed"
    )
    common(p, 60, 48)
    p.set_defaults(func=cmd_mq_build)

    p = sub.add_parser("mq-greedy")
    p.add_argument("--passages", type=int, nargs="+", default=[3, 10, 20, 40, 80, 160])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--workers", type=int, default=6)
    p.set_defaults(func=cmd_mq_greedy)

    p = sub.add_parser("mq-index")
    p.add_argument("--build", required=True)
    p.set_defaults(func=cmd_mq_index)

    p = sub.add_parser("mq-derive")
    p.add_argument("--kind", choices=("reference", "titles", "union", "drop"), required=True)
    p.add_argument("--builds", nargs="*", default=[])
    p.add_argument("--fraction", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_mq_derive)

    p = sub.add_parser("mq-eval")
    p.add_argument("--build", required=True, help="a build directory name, or 'dense'")
    p.add_argument("--small", action="store_true", help="the 60-question sub-corpus")
    common(p, 25, 8)
    p.set_defaults(func=cmd_mq_eval)

    p = sub.add_parser("lr-build")
    p.add_argument("--mode", choices=("once", "incremental"), required=True)
    p.add_argument("--model", default=READER)
    p.add_argument("--rep", type=int, required=True)
    common(p, 60, 32)
    p.set_defaults(func=cmd_lr_build)

    p = sub.add_parser("lr-eval")
    p.add_argument("--build", required=True)
    common(p, 25, 8)
    p.set_defaults(func=cmd_lr_eval)

    p = sub.add_parser("report")
    p.set_defaults(func=lambda a: __import__("stability_report").write_report(OUT))

    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
