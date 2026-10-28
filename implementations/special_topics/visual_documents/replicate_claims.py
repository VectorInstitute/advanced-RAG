#!/usr/bin/env python3
"""Measure the two carried-over claims notebook 01 states, and write the results.

Stages, each resumable from the local caches:

``parse``      parse pages on the CPU: ``ocr-forced`` on a 300-page universe (every
               question's gold page plus seeded distractors), ``ocr-auto`` on the
               figure pages, and ``flag-forced`` on a 40-page sample
``fidelity``   compare each condition's page text and tables with the text layer
``retrieval``  rank chunks with ``bge-small-en-v1.5`` over the 300-page universe,
               for the text layer and ``ocr-forced`` (arm ``T``)
``answers``    read each condition's top five chunks with the reference reader and
               score the answers with the reference judge (on-prem, no spend)
``crops``      describe a crop-size sample of pictures with three prompts on the
               reference vision model, then judge each description for claims the
               crop does not support with ``gemini-3.8-flash`` (paid)
``report``     write ``outputs/replication.json``, identifiers and metrics only

Examples
--------
Everything, with a ceiling on the paid judge::

    $ ADVANCED_RAG_SPEND_LIMIT_USD=3 python replicate_claims.py \
        parse fidelity retrieval answers crops report
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
from vdocs import arms, enrich, replication  # noqa: E402
from vdocs.pages import elements, page_id  # noqa: E402
from vdocs.pipeline import REFERENCE_JUDGE, REFERENCE_VLM, Pipeline, _embedder, reference_models  # noqa: E402
from vdocs.scoring import score_answer  # noqa: E402
from vdocs.vlm import ProxyModels, parse_json  # noqa: E402


OUT = HERE / "outputs" / "replication.json"
WORK = Path(cache_root()) / "special_topics" / "visual_documents_replication"
JUDGE = "gemini-3.8-flash"
#: Pictures sampled per crop-size bin, and the seed.
PER_BIN = 30
SEED = 0
#: Pages in the ``flag-forced`` sample: two pages from each of 20 filings.
FLAG_DOCS = 20
FLAG_PAGES = 2
#: Size of the reduced universe that ``ocr-forced`` parses and both text arms
#: index. Parsing all 1,000 pages with OCR did not fit the shared machine.
REDUCED_PAGES = 300
RANKED = ("text-layer", "ocr-forced")
PROMPTS = {
    "generic": enrich.GENERIC_PICTURE_PROMPT,
    "domain": enrich.DOMAIN_PICTURE_PROMPT,
    "grounded-generic": replication.GROUNDED_GENERIC_PROMPT,
}


def _save(name: str, value: Any) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / f"{name}.json").write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")


def _load(name: str) -> Any:
    return json.loads((WORK / f"{name}.json").read_text(encoding="utf-8"))


def reduced_universe(pipe: Pipeline) -> list[str]:
    """Return every question's gold or rewritten-from page, plus seeded distractors."""
    universe = pipe.universe()
    gold: set[str] = set()
    for q in pipe.questions():
        gold |= set(q.get("gold_pages") or [])
        if q.get("rewritten_from"):
            gold.add(str(q["rewritten_from"]))
    gold &= set(universe.pages)
    spare = sorted(set(universe.distractor) - gold)
    extra = random.Random(SEED).sample(spare, REDUCED_PAGES - len(gold))
    return sorted(gold | set(extra))


def _groups(pipe: Pipeline, condition: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]]
    if condition == replication.REFERENCE:
        pages = set(reduced_universe(pipe)) | set(pipe.universe().figure)
        for group in _groups(pipe, "flag-forced").items():
            pages |= {page_id(group[0], p) for p in group[1]}
        groups = replication.pages_by_doc(pages)
        return groups
    if condition == "ocr-forced":
        groups = replication.pages_by_doc(reduced_universe(pipe))
        return groups
    if condition == "ocr-auto":
        groups = replication.pages_by_doc(pipe.universe().figure)
        return groups
    groups = replication.pages_by_doc(pipe.universe().pages)
    docs = sorted(random.Random(SEED).sample(sorted(groups), FLAG_DOCS))
    return {d: groups[d][:FLAG_PAGES] for d in docs}


def _parse_one(job: tuple[str, str, list[int]]) -> str:
    condition, doc_id, pages = job
    pipe = Pipeline(reference_models(), show=False)
    replication.parse_condition(pipe.corpus, doc_id, pages, condition, WORK / "staging")
    return f"{condition} {doc_id} {len(pages)} pages"


def stage_parse(pipe: Pipeline, args: argparse.Namespace) -> None:
    """Parse the pages under every condition, one process per document."""
    order = (replication.REFERENCE, "ocr-forced", "flag-forced", "ocr-auto")
    jobs = [(c, d, p) for c in order for d, p in _groups(pipe, c).items()]
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        for line in pool.map(_parse_one, jobs):
            print(line, flush=True)


def stage_fidelity(pipe: Pipeline, args: argparse.Namespace) -> None:
    """Compare each condition's page text and table cells with the text layer."""
    out: dict[str, Any] = {}
    reference = replication.condition_corpus(pipe.corpus, replication.REFERENCE)
    for condition in replication.CONDITIONS:
        corpus = replication.condition_corpus(pipe.corpus, condition)
        rows = {}
        for doc_id, pages in _groups(pipe, condition).items():
            ref_export = replication.load_export(reference.folder(doc_id))
            export = replication.load_export(corpus.folder(doc_id))
            for page in pages:
                identifier = page_id(doc_id, page)
                row = replication.page_fidelity(reference.page_text(identifier), corpus.page_text(identifier))
                ref_cells = replication.table_cells(ref_export, page + 1)
                cells = replication.table_cells(export, page + 1)
                kept, total = replication.multiset_recall(ref_cells, cells)
                row.update(
                    {
                        "tables_ref": replication.table_count(ref_export, page + 1),
                        "tables": replication.table_count(export, page + 1),
                        "cells_ref": total,
                        "cells_kept": kept,
                    }
                )
                rows[identifier] = row
        out[condition] = rows
        print(condition, len(rows), "pages compared", flush=True)
    _save("fidelity", out)


def _rank(
    pipe: Pipeline, corpus: Any
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, str], dict[str, str]]:
    found = pipe.questions()
    chunks, first_page = enrich.build_chunks(corpus, reduced_universe(pipe))
    retriever = arms.DenseChunks(chunks, _embedder())
    ranked = retriever.search([str(q["question"]) for q in found], 50)
    ids = [str(q["query_id"]) for q in found]
    chunk_rank = {qid: r[:20] for qid, r in zip(ids, ranked, strict=True)}
    page_rank = {qid: arms.collapse_to_pages(r, first_page) for qid, r in zip(ids, ranked, strict=True)}
    text_of = {c.chunk_id: c.text for c in chunks}
    used = {c for r in ranked for c in r[: arms.CHUNKS_TO_READER]}
    return chunk_rank, page_rank, {c: text_of[c] for c in used}, {c: first_page[c] for c in used}


def stage_retrieval(pipe: Pipeline, args: argparse.Namespace) -> None:
    """Rank arm ``T`` chunks over the reduced universe for both parses."""
    for condition in RANKED:
        chunk_rank, page_rank, texts, pages = _rank(pipe, replication.condition_corpus(pipe.corpus, condition))
        _save(f"rankings-{condition}", {"chunks": chunk_rank, "pages": page_rank, "text": texts, "page": pages})
        print(condition, "ranked", len(page_rank), "questions", flush=True)


def stage_answers(pipe: Pipeline, args: argparse.Namespace) -> None:
    """Read each condition's top five chunks with the reference reader; score them."""
    llm = ProxyModels.from_environment()
    found = pipe.questions()
    by_id = {str(q["query_id"]): q for q in found}
    for condition in RANKED:
        ranked = _load(f"rankings-{condition}")
        inputs = arms.ArmInputs(
            chunk_text={"T": ranked["text"]},
            chunk_page={"T": ranked["page"]},
            chunk_rankings={"T": ranked["chunks"]},
            page_rankings={"T": ranked["pages"]},
            page_text=lambda page, enriched: "",
        )
        answers = arms.run_reader(llm, model=REFERENCE_VLM, corpus=pipe.corpus, row="T", questions=found, inputs=inputs)

        def judge(prompt: str) -> str:
            return str(llm.complete(REFERENCE_JUDGE, prompt, max_tokens=300).text)

        def one(item: tuple[str, Any]) -> tuple[str, dict[str, Any]]:
            qid, record = item
            verdict = score_answer(by_id[qid], str(record["answer"]), judge)
            return qid, {"correct": verdict.correct, "method": verdict.method}

        verdicts = dict(enrich.parallel_map(one, sorted(answers.items())))
        _save(f"verdicts-{condition}", verdicts)
        print(condition, sum(v["correct"] for v in verdicts.values()), "of", len(verdicts), "correct", flush=True)


def crop_sample(pipe: Pipeline) -> list[Any]:
    """Return up to :data:`PER_BIN` pictures per crop-size bin from the universe."""
    pictures = [e for d in pipe.universe().doc_ids for e in elements(pipe.corpus, d) if e.kind == "picture"]
    rng = random.Random(SEED)
    chosen = []
    for name, _, _ in replication.AREA_BINS:
        in_bin = sorted((e for e in pictures if replication.area_bin(e.area) == name), key=lambda e: e.element_id)
        chosen.extend(sorted(rng.sample(in_bin, min(PER_BIN, len(in_bin))), key=lambda e: e.element_id))
    return chosen


def stage_crops(pipe: Pipeline, args: argparse.Namespace) -> None:
    """Describe the crop sample with three prompts, then judge each description."""
    from vdocs.pages import crop_element, data_uri  # noqa: PLC0415

    sample = crop_sample(pipe)
    onprem = ProxyModels.from_environment()
    descriptions = {
        name: enrich.describe_pictures(onprem, REFERENCE_VLM, pipe.corpus, sample, prompt=prompt)
        for name, prompt in PROMPTS.items()
    }
    _save("crop-descriptions", descriptions)
    spend.session()
    paid = ProxyModels.from_environment(
        rpm=60, cache_path=Path(cache_root()) / "responses" / "replication-crops.sqlite"
    )

    def one(item: tuple[str, Any]) -> tuple[str, dict[str, Any]]:
        name, element = item
        text = str(descriptions[name][element.element_id]["text"])
        reply = paid.complete(
            JUDGE,
            replication.UNSUPPORTED_PROMPT.format(description=text[:4000]),
            [data_uri(crop_element(pipe.corpus, element))],
            max_tokens=1500,
        )
        try:
            found = parse_json(reply.text)
        except (ValueError, json.JSONDecodeError):
            found = {}
        return f"{name}|{element.element_id}", {**replication.count_verdicts(found), "judged": found}

    judged = dict(enrich.parallel_map(one, [(n, e) for n in PROMPTS for e in sample]))
    _save("crop-verdicts", judged)
    _save("crop-sample", [{"element_id": e.element_id, "area": e.area, "bbox": e.bbox} for e in sample])
    print(spend.report())


def _share(values: Sequence[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _fidelity_summary(rows: dict[str, Any]) -> dict[str, Any]:
    with_text = [r for r in rows.values() if r["ref_chars"] >= 200]
    no_layer = [r for r in rows.values() if r["ref_chars"] < 50]
    numbers_ref = sum(r["numbers_ref"] for r in with_text)
    cells_ref = sum(r["cells_ref"] for r in rows.values())
    return {
        "pages": len(rows),
        "identical_pages": sum(r["identical"] for r in rows.values()),
        "pages_with_text_layer": len(with_text),
        "mean_word_accuracy": round(sum(r["word_accuracy"] for r in with_text) / len(with_text), 4)
        if with_text
        else None,
        "numbers_kept": round(sum(r["numbers_kept"] for r in with_text) / numbers_ref, 4) if numbers_ref else None,
        "numbers_extra_per_page": round(sum(r["numbers_extra"] for r in with_text) / len(with_text), 2)
        if with_text
        else None,
        "pages_without_text_layer": len(no_layer),
        "chars_added_on_pages_without_text_layer": sum(r["chars"] for r in no_layer),
        "pages_without_text_layer_gaining_200_chars": sum(r["chars"] >= 200 for r in no_layer),
        "tables_ref": sum(r["tables_ref"] for r in rows.values()),
        "tables": sum(r["tables"] for r in rows.values()),
        "table_cells_kept": round(sum(r["cells_kept"] for r in rows.values()) / cells_ref, 4) if cells_ref else None,
    }


def stage_report(pipe: Pipeline, args: argparse.Namespace) -> None:
    """Write the committed results: counts and rates, no corpus text."""
    from aieng.advanced_rag.evaluation.multihop import mcnemar_exact  # noqa: PLC0415
    from vdocs.report import group_of  # noqa: PLC0415

    found = pipe.questions()
    gold = {str(q["query_id"]): set(q.get("gold_pages") or []) for q in found}
    groups = sorted({group_of(q) for q in found})
    fidelity = _load("fidelity")
    claim1: dict[str, Any] = {"fidelity": {c: _fidelity_summary(fidelity[c]) for c in replication.CONDITIONS}}
    retrieval: dict[str, Any] = {}
    accuracy: dict[str, Any] = {}
    verdicts = {}
    for condition in RANKED:
        pages = _load(f"rankings-{condition}")["pages"]
        verdicts[condition] = _load(f"verdicts-{condition}")
        retrieval[condition] = {}
        accuracy[condition] = {}
        for group in [*groups, "all"]:
            picked = [q for q in found if group in ("all", group_of(q)) and gold[str(q["query_id"])]]
            hits = [bool(gold[str(q["query_id"])] & set(pages[str(q["query_id"])][:5])) for q in picked]
            retrieval[condition][group] = {"n": len(hits), "recall_at_5": _share(hits)}
            answered = [q for q in found if group in ("all", group_of(q))]
            accuracy[condition][group] = {
                "n": len(answered),
                "correct": sum(bool(verdicts[condition][str(q["query_id"])]["correct"]) for q in answered),
            }
    paired = {}
    for condition in RANKED[1:]:
        ids = sorted(verdicts["text-layer"])
        test = mcnemar_exact(
            [bool(verdicts["text-layer"][i]["correct"]) for i in ids],
            [bool(verdicts[condition][i]["correct"]) for i in ids],
        )
        paired[f"text-layer_vs_{condition}"] = {
            "text_layer_only": test.only_a,
            "condition_only": test.only_b,
            "p": round(test.p_value, 4),
        }
    claim1.update({"page_recall_at_5": retrieval, "answers": accuracy, "paired": paired})

    sample = _load("crop-sample")
    judged = _load("crop-verdicts")
    area_of = {s["element_id"]: s["area"] for s in sample}
    claim2: dict[str, Any] = {"per_bin": {}, "bins": {}}
    for name, low, high in replication.AREA_BINS:
        ids = [i for i, a in area_of.items() if low <= a < high]
        claim2["bins"][name] = len(ids)
        claim2["per_bin"][name] = {
            prompt: replication.summarise_unsupported(judged[f"{prompt}|{i}"] for i in ids) for prompt in PROMPTS
        }
    claim2["all"] = {
        prompt: replication.summarise_unsupported(judged[f"{prompt}|{i}"] for i in area_of) for prompt in PROMPTS
    }
    claim2["unparsed_judgements"] = sum(1 for v in judged.values() if v.get("claims") is None)
    hand = WORK / "hand-check.json"
    if hand.is_file():
        claim2["hand_check"] = json.loads(hand.read_text(encoding="utf-8"))["summary"]
    result = {
        "produced_by": {
            "model": REFERENCE_VLM,
            "producer": "implementations/special_topics/visual_documents/replicate_claims.py",
            "mode": "explore: one run, fixed sampling seed 0",
            "parser": "docling (table structure on); RapidOCR PP-OCRv6 small, English, for OCR conditions",
            "raster_dpi": replication.RASTER_DPI,
            "embeddings": "BAAI/bge-small-en-v1.5 (local)",
            "reader": REFERENCE_VLM,
            "answer_judge": REFERENCE_JUDGE,
            "describer": REFERENCE_VLM,
            "description_judge": JUDGE,
            "endpoint": "vector-cluster (reader, describer, answer judge); vector-proxy-paid (description judge)",
        },
        "claim_1_text_layer_vs_ocr": claim1,
        "claim_2_small_crops": claim2,
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"fidelity": claim1["fidelity"], "paired": paired}, indent=1))


STAGES = {
    "parse": stage_parse,
    "fidelity": stage_fidelity,
    "retrieval": stage_retrieval,
    "answers": stage_answers,
    "crops": stage_crops,
    "report": stage_report,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the named stages in order."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stages", nargs="+", choices=list(STAGES))
    parser.add_argument("--processes", type=int, default=7, help="parallel Docling processes for parse")
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    pipe = Pipeline(reference_models(), show=False)
    for name in args.stages:
        STAGES[name](pipe, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
