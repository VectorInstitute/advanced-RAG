#!/usr/bin/env python3
r"""Test the carried-over diagram claim on ViDoRe v3 physics slides; write the result.

The claim: OCR returned almost nothing useful on diagrams, while a vision-model
description recovered sequence and connection information. Each stage resumes
from the local work folder:

``images``     pick one gold page per English query whose judged region is a
               picture, plus a seeded control of text-only pages, and write
               their images
``ocr``        RapidOCR on every picked page, on the CPU
``select``     ``Qwen3_8-27B`` (on-prem) marks the picture pages where the
               answer needs a relation the drawing shows (arrow, flow, order,
               connection, arrangement)
``describe``   describe every picked page on the participant vision models:
               the generic and relations prompts on ``gemini-3.5-flash-lite``,
               the relations prompt on ``claude-haiku-5-5``, and the relations
               prompt on a copy with everything but the OCR'd text painted out
``answer``     ``gemini-3.5-flash-lite`` answers each query from each
               representation of its gold page, and from the page image
``judge``      ``gemini-3.8-flash`` splits each reference answer into facts,
               marks which facts each representation states, and grades
               each answer
``relations``  ``claude-sonnet-5-5`` lists the relations each diagram page draws,
               and ``gemini-3.8-flash`` marks which ones each representation states
``retrieval``  describe all 1,674 pages and rank them for every English query with
               the text layer alone and with the descriptions added
``report``     write ``outputs/diagrams.json``, identifiers and counts only

Examples
--------
Everything, under a ceiling, from a local snapshot of the dataset::

    $ ADVANCED_RAG_SPEND_LIMIT_USD=8 python diagram_claim.py \\
        --data <snapshot_dir> images ocr select describe answer judge report

where ``<snapshot_dir>`` is a ``huggingface_hub.snapshot_download`` of
``vidore/vidore_v3_physics``. Without ``--data`` the script downloads it into the
artifact cache.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from aieng.advanced_rag import spend  # noqa: E402
from aieng.advanced_rag.cache.paths import root as cache_root  # noqa: E402
from vdocs import diagrams, enrich  # noqa: E402
from vdocs.pages import data_uri  # noqa: E402
from vdocs.vlm import ProxyModels, parse_json  # noqa: E402


OUT = HERE / "outputs" / "diagrams.json"
WORK = Path(cache_root()) / "special_topics" / "visual_documents_diagrams"
SELECTOR = "Qwen3_8-27B"
DESCRIBER = "gemini-3.5-flash-lite"
SECOND_DESCRIBER = "claude-haiku-5-5"
READER = "gemini-3.5-flash-lite"
JUDGE = "gemini-3.8-flash"
#: Lists the relations each diagram page draws, from a provider family other
#: than the main describer's.
GOLD = "claude-sonnet-5-5"
#: Other-picture pages added to the relation test as a control, and the seed.
RELATION_CONTROLS = 20
#: Text-only control pages, and the seed that samples them.
CONTROL_PAGES = 40
SEED = 0
#: Text representations each query is read from, in report order.
REPRESENTATIONS = (
    "text-layer",
    "ocr",
    "ocr-boxes",
    "vlm-generic",
    "vlm-relations",
    "vlm-relations-haiku",
    "vlm-text-only",
)
#: The vision calls behind each description: (model, prompt, text-only image).
DESCRIPTIONS = {
    "vlm-generic": (DESCRIBER, enrich.GENERIC_PICTURE_PROMPT, False),
    "vlm-relations": (DESCRIBER, diagrams.RELATIONS_PROMPT, False),
    "vlm-relations-haiku": (SECOND_DESCRIBER, diagrams.RELATIONS_PROMPT, False),
    "vlm-text-only": (DESCRIBER, diagrams.RELATIONS_PROMPT, True),
}


def _save(name: str, value: Any) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / f"{name}.json").write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _load(name: str) -> Any:
    return json.loads((WORK / f"{name}.json").read_text(encoding="utf-8"))


#: Requests per minute for each paid model; ``--rpm`` overrides it.
RPM = [60]


def _models(rpm: int | None = None) -> ProxyModels:
    return ProxyModels.from_environment(
        rpm=rpm or RPM[0], cache_path=Path(cache_root()) / "responses" / "visual_documents_diagrams.sqlite"
    )


def _image(corpus_id: int, text_only: bool = False) -> Any:
    from PIL import Image  # noqa: PLC0415

    picture = Image.open(WORK / "images" / f"{corpus_id}.jpg").convert("RGB")
    if text_only:
        return diagrams.text_only_image(picture, _ocr_lines(corpus_id))
    return picture


def _ocr_lines(corpus_id: int) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = json.loads((WORK / "ocr" / f"{corpus_id}.json").read_text(encoding="utf-8"))
    return lines


def _usage(reply: Any) -> dict[str, int]:
    return {"prompt": reply.prompt_tokens, "completion": reply.completion_tokens}


def stage_images(data: diagrams.Physics, args: argparse.Namespace) -> None:
    """Pick the pages and write their images."""
    english = [int(q["query_id"]) for q in data.english()]
    visual = diagrams.best_page_per_query(diagrams.visual_pairs(data.qrels, english))
    text = diagrams.best_page_per_query(diagrams.text_pairs(data.qrels, english))
    spare = sorted(set(text) - set(visual))
    control = sorted(random.Random(SEED).sample(spare, min(CONTROL_PAGES, len(spare))))
    picked = {
        "visual": [{"query_id": q, "corpus_id": c} for q, c in visual.items()],
        "text": [{"query_id": q, "corpus_id": text[q]} for q in control],
    }
    _save("picked", picked)
    ids = {p["corpus_id"] for group in picked.values() for p in group}
    data.images(ids, WORK / "images")
    print(len(picked["visual"]), "visual pairs,", len(picked["text"]), "text pairs,", len(ids), "pages", flush=True)


_ENGINE: list[Any] = []


def _ocr_one(corpus_id: int) -> str:
    target = WORK / "ocr" / f"{corpus_id}.json"
    if target.is_file():
        return f"{corpus_id} cached"
    if not _ENGINE:
        from rapidocr import RapidOCR  # noqa: PLC0415

        _ENGINE.append(RapidOCR())
    lines = diagrams.ocr_lines(_image(corpus_id), _ENGINE[0])
    target.write_text(json.dumps(lines, ensure_ascii=False) + "\n", encoding="utf-8")
    return f"{corpus_id} {len(lines)} lines"


def stage_ocr(data: diagrams.Physics, args: argparse.Namespace) -> None:
    """Run RapidOCR on every picked page, one process per worker."""
    picked = _load("picked")
    ids = sorted({p["corpus_id"] for group in picked.values() for p in group})
    (WORK / "ocr").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        for line in pool.map(_ocr_one, ids):
            print(line, flush=True)


def stage_select(data: diagrams.Physics, args: argparse.Namespace) -> None:
    """Mark the picture pages whose answer needs a relation the drawing shows."""
    picked = _load("picked")
    llm = _models(rpm=20)

    def one(pair: dict[str, int]) -> tuple[str, dict[str, Any]]:
        query = data.query(pair["query_id"])
        reply = llm.complete(
            SELECTOR,
            diagrams.SELECT_PROMPT.format(query=query["query"], answer=query["answer"]),
            [data_uri(_image(pair["corpus_id"]))],
            max_tokens=200,
        )
        try:
            found = parse_json(reply.text)
        except (ValueError, json.JSONDecodeError):
            found = {}
        return str(pair["query_id"]), {
            "diagram_relation": found.get("diagram_relation") is True,
            "kind": str(found.get("kind", "")),
            "parsed": bool(found),
        }

    _save("selection", dict(enrich.parallel_map(one, picked["visual"], workers=6)))
    chosen = sum(v["diagram_relation"] for v in _load("selection").values())
    print(chosen, "of", len(picked["visual"]), "picture pairs need a drawn relation", flush=True)


def _pages() -> list[int]:
    picked = _load("picked")
    return sorted({p["corpus_id"] for group in picked.values() for p in group})


def stage_describe(data: diagrams.Physics, args: argparse.Namespace) -> None:
    """Describe every picked page with each description's model and prompt."""
    spend.session()
    llm = _models()
    out: dict[str, dict[str, Any]] = {}
    for name, (model, prompt, text_only) in DESCRIPTIONS.items():

        def one(
            corpus_id: int, model: str = model, prompt: str = prompt, text_only: bool = text_only
        ) -> tuple[str, Any]:
            reply = llm.complete(model, prompt, [data_uri(_image(corpus_id, text_only))], max_tokens=1200)
            return str(corpus_id), {"text": reply.text, **_usage(reply)}

        out[name] = dict(enrich.parallel_map(one, _pages(), workers=8))
        _save("descriptions", out)
        print(name, "described", len(out[name]), "pages", flush=True)
    print(spend.report())


def _representations(data: diagrams.Physics) -> dict[str, dict[str, str]]:
    described = _load("descriptions")
    out: dict[str, dict[str, str]] = {}
    for corpus_id in _pages():
        lines = _ocr_lines(corpus_id)
        texts = {
            "text-layer": data.text_layer(corpus_id),
            "ocr": diagrams.ocr_text(lines),
            "ocr-boxes": diagrams.ocr_with_boxes(lines),
        }
        texts.update({name: str(described[name][str(corpus_id)]["text"]) for name in DESCRIPTIONS})
        out[str(corpus_id)] = texts
    return out


def _pairs() -> list[dict[str, int]]:
    picked = _load("picked")
    return [*picked["visual"], *picked["text"]]


def stage_answer(data: diagrams.Physics, args: argparse.Namespace) -> None:
    """Answer each query from each representation of its gold page and its image."""
    spend.session()
    llm = _models()
    texts = _representations(data)
    jobs = [(p, r) for p in _pairs() for r in (*REPRESENTATIONS, "page-image")]

    def one(job: tuple[dict[str, int], str]) -> tuple[str, Any]:
        pair, representation = job
        query = str(data.query(pair["query_id"])["query"])
        if representation == "page-image":
            reply = llm.complete(
                READER,
                diagrams.IMAGE_READER_PROMPT.format(query=query),
                [data_uri(_image(pair["corpus_id"]))],
                max_tokens=300,
            )
        else:
            content = texts[str(pair["corpus_id"])][representation][:12000]
            reply = llm.complete(READER, diagrams.READER_PROMPT.format(content=content, query=query), max_tokens=300)
        return f"{representation}|{pair['query_id']}", {"answer": reply.text, **_usage(reply)}

    _save("answers", dict(enrich.parallel_map(one, jobs, workers=8)))
    print("answered", len(jobs), flush=True)
    print(spend.report())


def stage_judge(data: diagrams.Physics, args: argparse.Namespace) -> None:
    """List each reference answer's facts, mark coverage, and grade the answers."""
    spend.session()
    llm = _models()
    pairs = _pairs()

    def facts_of(pair: dict[str, int]) -> tuple[str, Any]:
        query = data.query(pair["query_id"])
        reply = llm.complete(
            JUDGE, diagrams.FACTS_PROMPT.format(query=query["query"], answer=query["answer"]), max_tokens=800
        )
        try:
            found = diagrams.parse_facts(parse_json(reply.text))
        except (ValueError, json.JSONDecodeError):
            found = []
        return str(pair["query_id"]), {"facts": found, **_usage(reply)}

    facts = dict(enrich.parallel_map(facts_of, pairs, workers=8))
    _save("facts", facts)
    texts = _representations(data)

    def cover(job: tuple[dict[str, int], str]) -> tuple[str, Any]:
        pair, representation = job
        items = facts[str(pair["query_id"])]["facts"]
        if not items:
            return f"{representation}|{pair['query_id']}", {"stated": [], "prompt": 0, "completion": 0}
        content = texts[str(pair["corpus_id"])][representation][:12000]
        reply = llm.complete(
            JUDGE, diagrams.COVERAGE_PROMPT.format(facts=diagrams.numbered(items), content=content), max_tokens=300
        )
        try:
            stated = sorted(diagrams.stated_indices(parse_json(reply.text), len(items)))
        except (ValueError, json.JSONDecodeError):
            stated = []
        return f"{representation}|{pair['query_id']}", {"stated": stated, **_usage(reply)}

    _save("coverage", dict(enrich.parallel_map(cover, [(p, r) for p in pairs for r in REPRESENTATIONS], workers=8)))
    answers = _load("answers")

    def grade(key: str) -> tuple[str, Any]:
        query = data.query(int(key.split("|")[1]))
        reply = llm.complete(
            JUDGE,
            diagrams.ANSWER_JUDGE_PROMPT.format(
                query=query["query"], answer=query["answer"], candidate=answers[key]["answer"]
            ),
            max_tokens=400,
        )
        return key, {"correct": diagrams.parse_flag(reply.text, "correct"), **_usage(reply)}

    _save("grades", dict(enrich.parallel_map(grade, sorted(answers), workers=8)))
    print(spend.report())


def _groups() -> dict[str, list[dict[str, int]]]:
    picked = _load("picked")
    selection = _load("selection")
    return {
        "diagram": [p for p in picked["visual"] if selection[str(p["query_id"])]["diagram_relation"]],
        "other-picture": [p for p in picked["visual"] if not selection[str(p["query_id"])]["diagram_relation"]],
        "text": picked["text"],
    }


def _cost(*stages: str) -> dict[str, Any]:
    tokens: dict[str, list[int]] = {}
    model_of = {
        "descriptions": lambda key: DESCRIPTIONS[key][0],
        "answers": lambda key: READER,
        "facts": lambda key: JUDGE,
        "coverage": lambda key: JUDGE,
        "grades": lambda key: JUDGE,
        "relations": lambda key: GOLD,
        "relation-coverage": lambda key: JUDGE,
    }
    for stage in stages:
        found = _load(stage)
        rows = (
            [(name, row) for name, block in found.items() for row in block.values()]
            if stage == "descriptions"
            else [(key, row) for key, row in found.items()]
        )
        for key, row in rows:
            model = model_of[stage](key)
            total = tokens.setdefault(model, [0, 0, 0])
            total[0] += 1
            total[1] += int(row.get("prompt", 0))
            total[2] += int(row.get("completion", 0))
    out = {}
    for model, (calls, prompt, completion) in sorted(tokens.items()):
        rate = spend.price_of(model, spend.OFFLINE_PRICES)
        usd = None if rate is None else prompt * rate.input_per_mtok / 1e6 + completion * rate.output_per_mtok / 1e6
        out[model] = {
            "calls": calls,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "usd_list_price": round(usd, 3) if usd is not None else None,
        }
    return out


def stage_report(data: diagrams.Physics, args: argparse.Namespace) -> None:
    """Write the committed result: identifiers, counts, and rates, no dataset text."""
    from aieng.advanced_rag.evaluation.multihop import mcnemar_exact  # noqa: PLC0415

    selection = _load("selection")
    facts = {k: v["facts"] for k, v in _load("facts").items()}
    covered = _load("coverage")
    grades = _load("grades")
    groups = _groups()
    result: dict[str, Any] = {
        "produced_by": {
            "model": DESCRIBER,
            "producer": "implementations/special_topics/visual_documents/diagram_claim.py",
            "mode": "explore: one run, control sample seed 0",
            "dataset": f"{diagrams.DATASET} (CC BY 4.0; French slides, English queries)",
            "selector": f"{SELECTOR} (vector-cluster)",
            "relation_lister": GOLD,
            "describers": {name: model for name, (model, _, _) in DESCRIPTIONS.items()},
            "ocr": "RapidOCR (rapidocr package defaults) on 1600-pixel page renders",
            "text_layer": "pypdfium2 get_text_range on the source PDF page",
            "reader": READER,
            "judge": JUDGE,
            "endpoint": "vector-proxy-paid (describers, reader, judge); vector-cluster (selector)",
        },
        "selection": {
            "picture_pairs": len(selection),
            "diagram_relation": sum(v["diagram_relation"] for v in selection.values()),
            "kinds": {
                kind: sum(1 for v in selection.values() if v["diagram_relation"] and v["kind"] == kind)
                for kind in ("sequence", "connection", "arrangement")
            },
            "unparsed": sum(1 for v in selection.values() if not v["parsed"]),
        },
        "groups": {},
    }
    for group, pairs in groups.items():
        keys = [str(p["query_id"]) for p in pairs]
        block: dict[str, Any] = {
            "pairs": len(pairs),
            "query_ids": sorted(int(k) for k in keys),
            "pages": sorted({p["corpus_id"] for p in pairs}),
            "facts_per_pair": round(sum(len(facts[k]) for k in keys) / len(keys), 2) if keys else None,
            "relational_share": round(
                sum(f["kind"] == "relational" for k in keys for f in facts[k])
                / max(1, sum(len(facts[k]) for k in keys)),
                4,
            ),
            "coverage": {},
            "relational_coverage": {},
            "correct": {},
        }
        group_facts = {k: facts[k] for k in keys}
        for representation in REPRESENTATIONS:
            stated = {k: set(covered[f"{representation}|{k}"]["stated"]) for k in keys}
            block["coverage"][representation] = diagrams.coverage(group_facts, stated)
            block["relational_coverage"][representation] = diagrams.coverage(group_facts, stated, "relational")
        for representation in (*REPRESENTATIONS, "page-image"):
            block["correct"][representation] = sum(bool(grades[f"{representation}|{k}"]["correct"]) for k in keys)
        block["ungraded"] = sum(
            1 for r in (*REPRESENTATIONS, "page-image") for k in keys if grades[f"{r}|{k}"]["correct"] is None
        )
        paired = {}
        for base in ("text-layer", "ocr"):
            for other in ("vlm-relations", "vlm-text-only", "page-image"):
                test = mcnemar_exact(
                    [bool(grades[f"{base}|{k}"]["correct"]) for k in keys],
                    [bool(grades[f"{other}|{k}"]["correct"]) for k in keys],
                )
                paired[f"{base}_vs_{other}"] = {
                    "base_only": test.only_a,
                    "other_only": test.only_b,
                    "p": round(test.p_value, 4),
                }
        block["paired_answers"] = paired
        result["groups"][group] = block
    relations = {k: v["relations"] for k, v in _load("relations").items()}
    covered_relations = _load("relation-coverage")
    result["relations"] = {}
    for group, ids in _relation_pages().items():
        keys = [str(c) for c in ids]
        block = {
            "pages": ids,
            "pages_with_relations": sum(1 for k in keys if relations[k]),
            "relations_per_page": round(sum(len(relations[k]) for k in keys) / len(keys), 2),
            "recall": {},
        }
        for representation in REPRESENTATIONS:
            stated = {k: set(covered_relations[f"{representation}|{k}"]["stated"]) for k in keys}
            block["recall"][representation] = diagrams.coverage({k: relations[k] for k in keys}, stated)
        result["relations"][group] = block
    result["cost"] = _cost("descriptions", "answers", "facts", "coverage", "grades", "relations", "relation-coverage")
    retrieval = WORK / "retrieval.json"
    if retrieval.is_file():
        result["retrieval"] = json.loads(retrieval.read_text(encoding="utf-8"))
    hand = WORK / "hand-check.json"
    if hand.is_file():
        result["hand_check"] = json.loads(hand.read_text(encoding="utf-8"))["summary"]
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {g: {k: v for k, v in b.items() if k not in ("query_ids", "pages")} for g, b in result["groups"].items()},
            indent=1,
        )
    )


def _relation_pages() -> dict[str, list[int]]:
    groups = _groups()
    other = sorted({p["corpus_id"] for p in groups["other-picture"]} - {p["corpus_id"] for p in groups["diagram"]})
    return {
        "diagram": sorted({p["corpus_id"] for p in groups["diagram"]}),
        "other-picture": sorted(random.Random(SEED).sample(other, min(RELATION_CONTROLS, len(other)))),
    }


def stage_relations(data: diagrams.Physics, args: argparse.Namespace) -> None:
    """List each test page's drawn relations; mark the ones each representation states.

    The queries' reference answers span several pages, so they cannot score
    one page. This stage scores the page against itself: ``claude-sonnet-5-5``
    lists the relations the drawing shows, and ``gemini-3.8-flash`` marks
    which of them each text representation states.
    """
    spend.session()
    llm = _models()
    pages = _relation_pages()
    ids = sorted({c for group in pages.values() for c in group})

    def gold(corpus_id: int) -> tuple[str, Any]:
        reply = llm.complete(GOLD, diagrams.GOLD_RELATIONS_PROMPT, [data_uri(_image(corpus_id))], max_tokens=1500)
        try:
            found = diagrams.parse_relations(parse_json(reply.text))
        except (ValueError, json.JSONDecodeError):
            found = []
        return str(corpus_id), {"relations": found, **_usage(reply)}

    relations = dict(enrich.parallel_map(gold, ids, workers=8))
    _save("relations", relations)
    texts = _representations(data)

    def cover(job: tuple[int, str]) -> tuple[str, Any]:
        corpus_id, representation = job
        items = relations[str(corpus_id)]["relations"]
        if not items:
            return f"{representation}|{corpus_id}", {"stated": [], "prompt": 0, "completion": 0}
        content = texts[str(corpus_id)][representation][:12000]
        reply = llm.complete(
            JUDGE, diagrams.COVERAGE_PROMPT.format(facts=diagrams.numbered(items), content=content), max_tokens=400
        )
        try:
            stated = sorted(diagrams.stated_indices(parse_json(reply.text), len(items)))
        except (ValueError, json.JSONDecodeError):
            stated = []
        return f"{representation}|{corpus_id}", {"stated": stated, **_usage(reply)}

    jobs = [(c, r) for c in ids for r in REPRESENTATIONS]
    _save("relation-coverage", dict(enrich.parallel_map(cover, jobs, workers=8)))
    print("relations listed for", len(ids), "pages", flush=True)
    print(spend.report())


#: Characters per description piece in the retrieval index; bge-small reads 512 tokens.
PIECE_CHARS = 1500


def _pieces(text: str, size: int = PIECE_CHARS) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def stage_retrieval(data: diagrams.Physics, args: argparse.Namespace) -> None:  # noqa: PLR0915
    """Rank all 1,674 pages for each English query, with and without descriptions.

    Every page is described with the relations prompt on the describer. Each
    page's text layer is one entry, each 1,500-character piece of its
    description another, and a ranking collapses to pages. ``bge-small-en-v1.5``
    searches with the English queries and BM25 with the French ones, the
    slides' language.
    """
    import numpy as np  # noqa: PLC0415
    from aieng.advanced_rag.retrieval.corpus import Chunk, Query  # noqa: PLC0415
    from aieng.advanced_rag.retrieval.index import DenseIndex  # noqa: PLC0415
    from aieng.advanced_rag.retrieval.sparse import BM25Retriever  # noqa: PLC0415
    from vdocs.pipeline import _embedder  # noqa: PLC0415

    spend.session()
    pages = sorted(data.pages)
    data.images(pages, WORK / "images")
    llm = _models()
    model, prompt, _ = DESCRIPTIONS["vlm-relations"]

    def one(corpus_id: int) -> tuple[str, Any]:
        reply = llm.complete(model, prompt, [data_uri(_image(corpus_id))], max_tokens=1200)
        return str(corpus_id), {"text": reply.text, **_usage(reply)}

    described = dict(enrich.parallel_map(one, pages, workers=8))
    _save("all-descriptions", described)
    print("described", len(described), "pages", flush=True)
    print(spend.report())

    entries: dict[str, list[tuple[str, int, str]]] = {"text-layer": [], "text-layer+vlm-relations": []}
    for corpus_id in pages:
        layer = (f"{corpus_id}|t", corpus_id, data.text_layer(corpus_id))
        entries["text-layer"].append(layer)
        entries["text-layer+vlm-relations"].append(layer)
        for i, piece in enumerate(_pieces(str(described[str(corpus_id)]["text"]))):
            entries["text-layer+vlm-relations"].append((f"{corpus_id}|d{i}", corpus_id, piece))
    english = data.english()
    ids = [int(q["query_id"]) for q in english]
    questions = {
        "english": [str(q["query"]) for q in english],
        "french": [str(data.query(i, "french")["query"]) for i in ids],
    }
    embedder = _embedder()
    rankings: dict[str, dict[str, list[int]]] = {}

    def collapse(chunk_ids: Sequence[str]) -> list[int]:
        out: list[int] = []
        for chunk_id in chunk_ids:
            page = int(chunk_id.split("|")[0])
            if page not in out:
                out.append(page)
            if len(out) == 10:
                break
        return out

    for arm, rows in entries.items():
        vectors = np.asarray(embedder.encode([text for _, _, text in rows]), dtype=np.float32)
        index = DenseIndex([cid for cid, _, _ in rows], vectors)
        encode = getattr(embedder, "encode_queries", embedder.encode)
        found = index.search(np.asarray(encode(questions["english"]), dtype=np.float32), None, 60)
        rankings[f"dense|{arm}"] = {str(q): collapse(r) for q, r in zip(ids, found, strict=True)}
        bm25 = BM25Retriever()
        bm25.index([Chunk(cid, enrich.UNSCOPED, text, 0, len(text)) for cid, _, text in rows])
        rankings[f"bm25|{arm}"] = {
            str(q): collapse([h.chunk_id for h in bm25.search(Query(str(q), enrich.UNSCOPED, "", text), 60)])
            for q, text in zip(ids, questions["french"], strict=True)
        }
        print(arm, "ranked", flush=True)
    _save("rankings", rankings)

    gold: dict[int, set[int]] = {}
    for row in data.qrels.itertuples(index=False):
        if int(row.query_id) in set(ids):
            gold.setdefault(int(row.query_id), set()).add(int(row.corpus_id))
    groups = {g: [p["query_id"] for p in pairs] for g, pairs in _groups().items()}
    text_ids = set(diagrams.best_page_per_query(diagrams.text_pairs(data.qrels, ids)))
    groups["text"] = sorted(text_ids - set(diagrams.best_page_per_query(diagrams.visual_pairs(data.qrels, ids))))
    groups["all"] = ids
    out: dict[str, Any] = {"pages": len(pages), "groups": {g: len(q) for g, q in groups.items()}, "recall": {}}
    for name, ranked in rankings.items():
        out["recall"][name] = {}
        for group, members in groups.items():
            for k in (5, 10):
                hits = [bool(gold[q] & set(ranked[str(q)][:k])) for q in members]
                out["recall"][name][f"{group}@{k}"] = round(sum(hits) / len(hits), 4) if hits else None
    out["cost"] = {
        model: {
            "calls": len(described),
            "prompt_tokens": sum(int(v["prompt"]) for v in described.values()),
            "completion_tokens": sum(int(v["completion"]) for v in described.values()),
        }
    }
    (WORK / "retrieval.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["recall"], indent=1))


STAGES = {
    "images": stage_images,
    "ocr": stage_ocr,
    "select": stage_select,
    "describe": stage_describe,
    "answer": stage_answer,
    "judge": stage_judge,
    "relations": stage_relations,
    "retrieval": stage_retrieval,
    "report": stage_report,
}


def _snapshot(path: str | None) -> Path:
    if path:
        return Path(path)
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    target = Path(cache_root()) / "datasets" / "vidore_v3_physics"
    return Path(snapshot_download(diagrams.DATASET, repo_type="dataset", revision=diagrams.REVISION, local_dir=target))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the named stages in order."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("stages", nargs="+", choices=list(STAGES))
    parser.add_argument("--data", help="local snapshot of vidore/vidore_v3_physics; downloaded when omitted")
    parser.add_argument("--processes", type=int, default=4, help="parallel OCR processes")
    parser.add_argument("--rpm", type=int, default=60, help="requests per minute for each paid model")
    args = parser.parse_args(argv)
    RPM[0] = args.rpm
    data = diagrams.Physics(_snapshot(args.data))
    for name in args.stages:
        STAGES[name](data, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
