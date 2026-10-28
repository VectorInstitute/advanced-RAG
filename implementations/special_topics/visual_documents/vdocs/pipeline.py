"""Every step of the comparison, so a cached reference and a live run read as one path.

:class:`Pipeline` is what both notebooks and ``build_reference.py`` call. Each
model-produced step goes through :func:`aieng.advanced_rag.runs.load_or_run`,
so with ``live`` off a reader gets the reference result built on the on-prem
models, labelled with its model and the cost of rebuilding it on theirs, and
with ``live`` on the step runs on the reader's own models. Steps that call no
model (the page universe, the text indexes, the page-image index, and the
rankings) go through :class:`aieng.advanced_rag.cache.TieredResultCache` with
every input named in the key.

The reference run used three on-prem models: ``Qwen3_8-27B`` read pictures,
wrote questions, and answered for every row; ``Qwen3_8-Flash-Next`` checked the
questions; ``Qwen3_6-35B-A3B`` judged answers no number settled.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from aieng.advanced_rag.cache import CacheKey, JsonValue, TieredResultCache, fetch_directory, store_directory
from aieng.advanced_rag.corpora import financebench
from aieng.advanced_rag.models import ModelChoice, ModelSet, choose_models
from aieng.advanced_rag.runs import Step, StepResult, load_or_run

from . import arms, enrich, questions, visual
from .pages import (
    PAGE_SCALE,
    Corpus,
    Element,
    Universe,
    elements,
    financebench_corpus,
    page_id,
    select_universe,
    split_page_id,
)
from .scoring import score_answer
from .vlm import ProxyModels


RI_ID = "visual-documents"
#: Reads pictures, writes questions, and answers for every row in the reference run.
REFERENCE_VLM = "Qwen3_8-27B"
#: Checks the generated questions, a different model from the generator.
REFERENCE_CHECKER = "Qwen3_8-Flash-Next"
#: Judges answers that no number settles.
REFERENCE_JUDGE = "Qwen3_6-35B-A3B"
#: The live default: a stratified subset, about a sixth of the reference set.
LIVE_SUBSET = 60
UNIVERSE_SIZE = 1000
SEED = 0
EMBEDDER = "BAAI/bge-small-en-v1.5"


def reference_models() -> ModelSet:
    """Return the model set the reference run used, one per role."""
    return choose_models("onprem", generation=REFERENCE_VLM, extraction=REFERENCE_VLM, judging=REFERENCE_JUDGE)


def digest_of(*parts: Any) -> str:
    """Return a short digest of prompts or settings, for cache keys."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class NotCachedError(LookupError):
    """A step that calls no model has no cached result, and expensive work is off."""


@dataclass
class Pipeline:
    """The comparison, step by step, for one model set.

    Parameters
    ----------
    models : ModelSet
        Whose models run the live steps: ``extraction`` describes pictures and
        tables, ``generation`` reads for every row, ``judging`` judges.
    live : bool, optional
        Run a model step when no result for these models is cached. Pass the
        notebook's ``RUN_EXPENSIVE``.
    expensive : bool or None, optional
        Build a missing index or ranking. Defaults to ``live``.
    rebuild_enrichment : bool, optional
        Describe pictures and summarise tables with ``models.extraction`` instead
        of reading the reference descriptions.
    corpus : Corpus or None, optional
        The filings. FinanceBench's when omitted.
    cache : TieredResultCache or None, optional
        The artifact cache.
    workdir : Path or None, optional
        Where directory artifacts are unpacked.
    confirm : bool, optional
        Passed to ``load_or_run``.
    """

    models: ModelSet
    live: bool = False
    expensive: bool | None = None
    rebuild_enrichment: bool = False
    corpus: Corpus = field(default_factory=financebench_corpus)
    cache: TieredResultCache = field(default_factory=TieredResultCache)
    workdir: Path | None = None
    confirm: bool = False
    show: bool = True
    env_path: Path | None = None
    _llm: ProxyModels | None = None
    _memo: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Default ``expensive`` to ``live`` and pick a working directory."""
        if self.expensive is None:
            self.expensive = self.live
        if self.workdir is None:
            self.workdir = Path(tempfile.gettempdir()) / "aieng-visual-documents"

    @property
    def llm(self) -> ProxyModels:
        """Return the proxy client, built on first use so a cached run needs no key."""
        if self._llm is None:
            self._llm = ProxyModels.from_environment(env_path=self.env_path)
        return self._llm

    # ------------------------------------------------------------------ cache helpers

    def _cached(self, key: CacheKey, compute: Callable[[], JsonValue], what: str) -> JsonValue:
        found = self.cache.get(key)
        if found is not None:
            return found.value
        if not self.expensive:
            raise NotCachedError(
                f"no cached {what}; fetch the reference artifacts or set RUN_EXPENSIVE = True to build it",
            )
        return self.cache.load_or_compute(key, compute).value

    def _step(self, step: Step, compute: Callable[[ModelChoice], Any], models: ModelSet | None = None) -> StepResult:
        return load_or_run(
            step,
            compute,
            models=models or self.models,
            live=self.live,
            confirm=self.confirm,
            cache=self.cache,
            workdir=self.workdir,
            show=self.show,
        )

    # ------------------------------------------------------------------ universe

    def universe_key(self) -> CacheKey:
        """Return the key of the page universe."""
        return CacheKey(
            RI_ID,
            "page-universe",
            "select_universe",
            {
                "financebench_commit": financebench.FINANCEBENCH_COMMIT,
                "size": UNIVERSE_SIZE,
                "seed": SEED,
                "version": 1,
            },
        )

    def universe(self) -> Universe:
        """Return the page universe, choosing it on first use."""
        if "universe" not in self._memo:

            def compute() -> JsonValue:
                figure_docs = _ingested(self.corpus)
                controls = [page_id(q.doc_id, p) for q in financebench.load_queries() for p in q.gold_pages]
                chosen = select_universe(self.corpus, figure_docs, controls, size=UNIVERSE_SIZE, seed=SEED)
                return cast(JsonValue, chosen.to_dict())

            value = self._cached(self.universe_key(), compute, "page universe")
            self._memo["universe"] = Universe.from_dict(cast(Mapping[str, Sequence[str]], value))
        return cast(Universe, self._memo["universe"])

    def elements(self) -> list[Any]:
        """Return every picture and table on the universe's pages."""
        if "elements" not in self._memo:
            key = CacheKey(RI_ID, "page-elements", "docling", {"universe": self.universe_key().digest, "version": 1})

            def compute() -> JsonValue:
                pages = set(self.universe().pages)
                found = [e for doc in self.universe().doc_ids for e in elements(self.corpus, doc) if e.page_id in pages]
                return [dataclasses.asdict(e) for e in found]

            stored = self.cache.load_or_compute(key, compute).value
            self._memo["elements"] = [
                Element(**{**row, "bbox": tuple(row["bbox"])}) for row in cast(list[dict[str, Any]], stored)
            ]
        return cast(list[Any], self._memo["elements"])

    # ------------------------------------------------------------------ enrichment

    def _enrichment_models(self) -> ModelSet:
        return self.models if self.rebuild_enrichment else reference_models()

    def enrichment_step(self, name: str) -> Step:
        """Return the step for ``e1-pictures``, ``e2-pictures``, or ``e2-tables``."""
        prompts = {
            "e1-pictures": enrich.GENERIC_PICTURE_PROMPT,
            "e2-pictures": enrich.DOMAIN_PICTURE_PROMPT,
            "e2-tables": enrich.TABLE_SUMMARY_PROMPT,
        }
        artifact = "table-summaries" if name == "e2-tables" else "picture-descriptions"
        return Step(
            RI_ID,
            artifact,
            name,
            {"universe": self.universe_key().digest, "prompt": digest_of(prompts[name]), "scale": PAGE_SCALE},
            role="extraction",
            reference_model=REFERENCE_VLM,
        )

    def enrichment(self, name: str) -> StepResult:
        """Return the descriptions or summaries for one enrichment pass."""
        memo = f"enrichment:{name}"
        if memo not in self._memo:

            def compute(choice: ModelChoice) -> Any:
                pages = self.universe().pages
                found = self.elements()
                if name == "e2-tables":
                    return enrich.summarise_tables(
                        self.llm, choice.model, self.corpus, enrich.summarisable_tables(found, pages)
                    )
                prompt = enrich.GENERIC_PICTURE_PROMPT if name == "e1-pictures" else enrich.DOMAIN_PICTURE_PROMPT
                return enrich.describe_pictures(
                    self.llm, choice.model, self.corpus, enrich.describable_pictures(found, pages), prompt=prompt
                )

            self._memo[memo] = self._step(self.enrichment_step(name), compute, self._enrichment_models())
        return cast(StepResult, self._memo[memo])

    def enrichment_text(self, arm: str) -> tuple[dict[str, str], dict[str, str]]:
        """Return picture and table text by element for arm ``T``, ``E1``, or ``E2``."""
        if arm == "T":
            return {}, {}
        if arm == "E1":
            return _texts(self.enrichment("e1-pictures").value), {}
        return _texts(self.enrichment("e2-pictures").value), _texts(self.enrichment("e2-tables").value)

    def enrichment_digests(self) -> dict[str, str]:
        """Return the digests of the three enrichment results, for downstream keys."""
        return {name: self.enrichment(name).digest for name in ("e1-pictures", "e2-pictures", "e2-tables")}

    # ------------------------------------------------------- chunks and page text

    def chunks(self, arm: str) -> tuple[list[Any], dict[str, str]]:
        """Return the chunks and each one's first page for ``T``, ``E1``, or ``E2``."""
        memo = f"chunks:{arm}"
        if memo not in self._memo:
            pictures, tables = self.enrichment_text(arm)
            self._memo[memo] = enrich.build_chunks(self.corpus, self.universe().pages, pictures=pictures, tables=tables)
        return cast(tuple[list[Any], dict[str, str]], self._memo[memo])

    def page_texts_key(self) -> CacheKey:
        """Return the key of every page's text, plain and with E2 enrichment."""
        return CacheKey(RI_ID, "page-texts", "page_items", {"universe": self.universe_key().digest, **self._e2()})

    def _e2(self) -> dict[str, str]:
        digests = self.enrichment_digests()
        return {"e2-pictures": digests["e2-pictures"], "e2-tables": digests["e2-tables"]}

    def page_texts(self) -> dict[str, dict[str, str]]:
        """Return ``{"plain": {page: text}, "enriched": {page: text}}``."""
        if "page_texts" not in self._memo:
            pictures, tables = self.enrichment_text("E2")

            def compute() -> JsonValue:
                by_doc: dict[str, list[int]] = {}
                for identifier in self.universe().pages:
                    doc_id, page = split_page_id(identifier)
                    by_doc.setdefault(doc_id, []).append(page)
                plain: dict[str, JsonValue] = {}
                enriched: dict[str, JsonValue] = {}
                for doc_id, pages in sorted(by_doc.items()):
                    for page in pages:
                        key = page_id(doc_id, page)
                        plain[key] = enrich.page_text_with_enrichment(self.corpus, key)
                        enriched[key] = enrich.page_text_with_enrichment(
                            self.corpus, key, pictures=pictures, tables=tables
                        )
                return {"plain": plain, "enriched": enriched}

            self._memo["page_texts"] = self._cached(self.page_texts_key(), compute, "page texts")
        return cast(dict[str, dict[str, str]], self._memo["page_texts"])

    # ------------------------------------------------------------------ questions

    def question_step(self) -> Step:
        """Return the step that builds the question set."""
        return Step(
            RI_ID,
            "question-set",
            "figure-table-null",
            {
                "universe": self.universe_key().digest,
                "prompts": digest_of(
                    questions.FIGURE_PROMPT, questions.TABLE_PROMPT, questions.NEAR_MISS_PROMPT, questions.ANSWER_PROMPT
                ),
                "checker": REFERENCE_CHECKER,
                "judge": REFERENCE_JUDGE,
                "seed": SEED,
                "version": 1,
            },
            role="extraction",
            reference_model=REFERENCE_VLM,
            upstream=self._e2(),
        )

    def question_set(self) -> StepResult:
        """Return the question set: questions, the funnel, and the live subset."""
        if "questions" not in self._memo:

            def compute(choice: ModelChoice) -> Any:
                return self._build_questions(choice.model)

            # The question set is a reference artifact whatever the reader's
            # models are: questions written by a participant's model would make
            # their scores incomparable with the reference rows.
            self._memo["questions"] = self._step(self.question_step(), compute, reference_models())
        return cast(StepResult, self._memo["questions"])

    def questions(self, which: str = "all") -> list[dict[str, Any]]:
        """Return every question, or the live subset with ``which="subset"``."""
        value = self.question_set().value
        found = list(value["questions"])
        if which == "subset":
            keep = set(value["subset"])
            found = [q for q in found if q["query_id"] in keep]
        return found

    def _build_questions(self, generator: str) -> dict[str, Any]:
        universe = self.universe()
        found = self.elements()
        funnel = questions.Funnel()
        descriptions = _texts(self.enrichment("e2-pictures").value)
        candidates = questions.figure_candidates(
            self.llm,
            model=generator,
            corpus=self.corpus,
            figure_pages=universe.figure,
            pictures=[e for e in found if e.kind == "picture"],
            funnel=funnel,
        )
        figures = questions.filter_figure_questions(
            self.llm,
            checker=REFERENCE_CHECKER,
            judge=REFERENCE_JUDGE,
            corpus=self.corpus,
            candidates=candidates,
            descriptions=descriptions,
            funnel=funnel,
        )
        figures = _cap(figures, 120, "subgroup")
        complex_tables = [e for e in found if e.page_id in set(universe.table) and e.complex_table]
        tables = questions.table_questions(
            self.llm,
            generator=generator,
            checker=REFERENCE_CHECKER,
            judge=REFERENCE_JUDGE,
            corpus=self.corpus,
            tables=complex_tables,
            funnel=funnel,
            seed=SEED,
        )
        tables = _cap(tables, 60, "subgroup")
        accepted = questions.assign_ids(figures, "fig", "figure") + questions.assign_ids(tables, "tab", "table")

        controls = _controls(universe, 40)
        texts = self.page_texts()
        chunks, first_page = self.chunks("E2")
        dense = arms.DenseChunks(chunks, _embedder())
        bm25 = arms.BM25Chunks(chunks)

        def retrieve_pages(question: str) -> list[str]:
            pages: list[str] = []
            for ranking in (dense.search([question], 30)[0], bm25.search([question], 30)[0]):
                for page in arms.collapse_to_pages(ranking, first_page, 5):
                    if page not in pages:
                        pages.append(page)
            return pages

        import random  # noqa: PLC0415

        rng = random.Random(SEED)
        sources = rng.sample([q for q in accepted if q["group"] == "figure"], min(30, len(figures))) + rng.sample(
            [q for q in accepted if q["group"] == "table"], min(20, len(tables))
        )
        nulls = questions.near_miss_nulls(
            self.llm,
            generator=generator,
            checker=REFERENCE_CHECKER,
            corpus=self.corpus,
            sources=sources,
            retrieve_pages=retrieve_pages,
            page_text=lambda p: texts["enriched"].get(p, ""),
            funnel=funnel,
        )
        everything = (
            accepted + questions.assign_ids(controls, "ctl", "control") + questions.assign_ids(nulls, "nul", "null")
        )
        subset = questions.stratified_subset(everything, LIVE_SUBSET, SEED)
        return {"questions": everything, "funnel": funnel.to_rows(), "subset": subset}

    # ------------------------------------------------------------------ rankings

    def text_rankings_key(self) -> CacheKey:
        """Return the key of the text arms' rankings."""
        return CacheKey(
            RI_ID,
            "text-rankings",
            "dense-colbert",
            {
                "questions": self.question_set().digest,
                "embedder": EMBEDDER,
                "chunk_chars": 1200,
                **self.enrichment_digests(),
            },
        )

    def text_rankings(self) -> dict[str, Any]:
        """Return chunk and page rankings for T, E1, E2, and L, and the chunks read."""
        if "text_rankings" not in self._memo:

            def compute() -> JsonValue:
                return cast(JsonValue, self._rank_text())

            self._memo["text_rankings"] = self._cached(self.text_rankings_key(), compute, "text-arm rankings")
        return cast(dict[str, Any], self._memo["text_rankings"])

    def _rank_text(self) -> dict[str, Any]:
        found = self.questions()
        texts = [str(q["question"]) for q in found]
        ids = [str(q["query_id"]) for q in found]
        chunk_rankings: dict[str, dict[str, list[str]]] = {}
        page_rankings: dict[str, dict[str, list[str]]] = {}
        reader_chunks: dict[str, dict[str, str]] = {}
        chunk_page: dict[str, dict[str, str]] = {}
        seconds: dict[str, float] = {}
        sizes: dict[str, int] = {}
        embedder = _embedder()
        for arm in ("T", "E1", "E2", "L"):
            source = "E2" if arm == "L" else arm
            chunks, first_page = self.chunks(source)
            started = time.perf_counter()
            retriever: Any = arms.ColbertChunks(chunks) if arm == "L" else arms.DenseChunks(chunks, embedder)
            seconds[f"{arm}_index"] = round(time.perf_counter() - started, 1)
            started = time.perf_counter()
            ranked = retriever.search(texts, 50)
            seconds[f"{arm}_query_total"] = round(time.perf_counter() - started, 2)
            sizes[arm] = len(chunks)
            chunk_rankings[arm] = {qid: r[:20] for qid, r in zip(ids, ranked, strict=True)}
            page_rankings[arm] = {
                qid: arms.collapse_to_pages(r, first_page) for qid, r in zip(ids, ranked, strict=True)
            }
            text_of = {c.chunk_id: c.text for c in chunks}
            used = {c for r in ranked for c in r[: arms.CHUNKS_TO_READER]}
            reader_chunks.setdefault(source, {}).update({c: text_of[c] for c in used})
            chunk_page.setdefault(source, {}).update({c: first_page[c] for c in used})
        return {
            "chunk_rankings": chunk_rankings,
            "page_rankings": page_rankings,
            "reader_chunks": reader_chunks,
            "chunk_page": chunk_page,
            "seconds": seconds,
            "chunks": sizes,
        }

    def page_index_key(self) -> CacheKey:
        """Return the key of the ColModernVBERT page index."""
        return CacheKey(
            RI_ID,
            "page-index",
            "colmodernvbert",
            {"universe": self.universe_key().digest, "model": visual.MODEL, "scale": PAGE_SCALE, "dtype": "float16"},
        )

    def query_embeddings_key(self) -> CacheKey:
        """Return the key of the question set's ColModernVBERT query embeddings."""
        return CacheKey(
            RI_ID,
            "query-embeddings",
            "colmodernvbert",
            {"questions": self.question_set().digest, "model": visual.MODEL},
        )

    def page_index(self, providers: Sequence[str] | None = None) -> visual.PageIndex:
        """Return the page index from the cache, or build it if expensive work is on."""
        if "page_index" not in self._memo:
            key = self.page_index_key()
            destination = cast(Path, self.workdir) / "page-index" / key.digest[:16]
            path = fetch_directory(self.cache, key, destination, replace=True)
            if path is None:
                if not self.expensive:
                    raise NotCachedError(
                        "no cached page index; fetch the reference artifacts or set RUN_EXPENSIVE = True"
                    )
                from .pages import render_page  # noqa: PLC0415

                model = visual.load_model(providers)
                built = visual.build_index(
                    self.universe().pages,
                    lambda p: render_page(self.corpus, p),
                    model,
                    meta={"providers": list(providers or ["CPUExecutionProvider"]), "scale": PAGE_SCALE},
                )
                scratch = Path(tempfile.mkdtemp(prefix="vdocs-index-"))
                built.save(scratch)
                store_directory(self.cache, key, scratch)
                path = fetch_directory(self.cache, key, destination, replace=True)
            self._memo["page_index"] = visual.PageIndex.load(cast(Path, path))
        return cast(visual.PageIndex, self._memo["page_index"])

    def query_embeddings(self, providers: Sequence[str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the question set's query matrices and how long they took to embed."""
        if "query_embeddings" not in self._memo:
            key = self.query_embeddings_key()
            destination = cast(Path, self.workdir) / "query-embeddings" / key.digest[:16]
            path = fetch_directory(self.cache, key, destination, replace=True)
            if path is None:
                if not self.expensive:
                    raise NotCachedError("no cached query embeddings; set RUN_EXPENSIVE = True to embed them")
                found = self.questions()
                model = visual.load_model(providers)
                matrices, per_query = visual.embed_queries(model, [str(q["question"]) for q in found])
                scratch = Path(tempfile.mkdtemp(prefix="vdocs-queries-"))
                visual.save_queries(
                    scratch,
                    [str(q["query_id"]) for q in found],
                    matrices,
                    {
                        "seconds_per_query": round(per_query, 3),
                        "providers": list(providers or ["CPUExecutionProvider"]),
                    },
                )
                store_directory(self.cache, key, scratch)
                path = fetch_directory(self.cache, key, destination, replace=True)
            self._memo["query_embeddings"] = visual.load_queries(cast(Path, path))
        return cast(tuple[dict[str, Any], dict[str, Any]], self._memo["query_embeddings"])

    def cpu_probe(self, pages: int = 4) -> dict[str, Any]:
        """Return ColModernVBERT's CPU cost and its agreement with the stored index.

        The page index may have been built on a GPU. Re-embedding a few pages on
        the CPU and comparing them with the stored vectors checks that the two
        runtimes write the same vectors, so a CPU query searches a GPU-built
        index correctly.
        """
        key = CacheKey(
            RI_ID,
            "cpu-probe",
            "colmodernvbert",
            {"index": self.page_index_key().digest, "queries": self.query_embeddings_key().digest, "pages": pages},
        )

        def compute() -> JsonValue:
            import os  # noqa: PLC0415

            import numpy as np  # noqa: PLC0415

            from .pages import render_page  # noqa: PLC0415

            index = self.page_index()
            stored, _ = self.query_embeddings()
            chosen = self.universe().figure[:pages]
            model = visual.load_model(["CPUExecutionProvider"])
            matrices, per_page = visual.embed_images(model, [render_page(self.corpus, p) for p in chosen])
            agreement = []
            for identifier, matrix in zip(chosen, matrices, strict=True):
                row = index.page_ids.index(identifier)
                saved = index.vectors[index.offsets[row] : index.offsets[row + 1]].astype(np.float32)
                if saved.shape == matrix.shape:
                    agreement.append(float(np.mean(np.sum(saved * matrix, axis=1))))
            found = self.questions()[:pages]
            queries, per_query = visual.embed_queries(model, [str(q["question"]) for q in found], batch_size=1)
            query_agreement = [
                float(np.mean(np.sum(stored[str(q["query_id"])] * m, axis=1)))
                for q, m in zip(found, queries, strict=True)
                if stored[str(q["query_id"])].shape == m.shape
            ]
            return {
                "seconds_per_page": round(per_page, 2),
                "seconds_per_query": round(per_query, 2),
                "page_vector_cosine": round(min(agreement), 5) if agreement else None,
                "query_vector_cosine": round(min(query_agreement), 5) if query_agreement else None,
                "cpus": os.cpu_count(),
                "load_average": round(os.getloadavg()[0], 1),
            }

        return cast(dict[str, Any], self._cached(key, compute, "CPU probe"))

    def visual_rankings_key(self) -> CacheKey:
        """Return the key of the page-image arms' rankings."""
        return CacheKey(
            RI_ID,
            "visual-rankings",
            "maxsim",
            {
                "index": self.page_index_key().digest,
                "queries": self.query_embeddings_key().digest,
                "text": self.text_rankings_key().digest,
                "rrf_k": arms.RRF_K,
            },
        )

    def rankings(self) -> dict[str, dict[str, list[str]]]:
        """Return page rankings for all eight arms."""
        if "rankings" not in self._memo:
            text = self.text_rankings()

            def compute() -> JsonValue:
                index = self.page_index()
                matrices, _ = self.query_embeddings()
                found = self.questions()
                started = time.perf_counter()
                hits = index.search_many([matrices[str(q["query_id"])] for q in found], k=10)
                seconds = time.perf_counter() - started
                visual_pages = {str(q["query_id"]): [p for p, _ in h] for q, h in zip(found, hits, strict=True)}
                fused = {
                    qid: arms.reciprocal_rank_fusion([text["page_rankings"]["E2"][qid], pages])
                    for qid, pages in visual_pages.items()
                }
                return {
                    "VV": visual_pages,
                    "H": fused,
                    "seconds_per_query": round(seconds / max(1, len(found)), 4),
                }

            found_visual = cast(
                dict[str, Any], self._cached(self.visual_rankings_key(), compute, "page-image rankings")
            )
            pages = dict(text["page_rankings"])
            pages["VT"] = found_visual["VV"]
            pages["VV"] = found_visual["VV"]
            pages["TV"] = text["page_rankings"]["E2"]
            pages["H"] = found_visual["H"]
            self._memo["rankings"] = pages
            self._memo["maxsim_seconds"] = found_visual["seconds_per_query"]
        return cast(dict[str, dict[str, list[str]]], self._memo["rankings"])

    def arm_inputs(self) -> arms.ArmInputs:
        """Return what every row's reader needs."""
        text = self.text_rankings()
        texts = self.page_texts()
        return arms.ArmInputs(
            chunk_text=text["reader_chunks"],
            chunk_page=text["chunk_page"],
            chunk_rankings=text["chunk_rankings"],
            page_rankings=self.rankings(),
            page_text=lambda page, enriched: texts["enriched" if enriched else "plain"].get(page, ""),
        )

    # ------------------------------------------------------- answers and verdicts

    def answers_step(self, row: str, which: str) -> Step:
        """Return the step for one row's answers on the full set or the subset."""
        return Step(
            RI_ID,
            "reader-answers",
            f"row-{row}",
            {
                "questions": which,
                "prompt": digest_of(questions.ANSWER_PROMPT),
                "chunks": arms.CHUNKS_TO_READER,
                "pages": arms.PAGES_TO_READER,
                "scale": PAGE_SCALE,
            },
            role="generation",
            reference_model=REFERENCE_VLM,
            upstream={
                "questions": self.question_set().digest,
                "text": self.text_rankings_key().digest,
                "visual": self.visual_rankings_key().digest,
            },
        )

    def answers(self, row: str, which: str = "all") -> StepResult:
        """Return one row's answers."""
        memo = f"answers:{row}:{which}"
        if memo not in self._memo:
            found = self.questions(which)
            inputs = self.arm_inputs()

            def compute(choice: ModelChoice) -> Any:
                return arms.run_reader(
                    self.llm, model=choice.model, corpus=self.corpus, row=row, questions=found, inputs=inputs
                )

            self._memo[memo] = self._step(self.answers_step(row, which), compute)
        return cast(StepResult, self._memo[memo])

    def verdicts_step(self, row: str, which: str, answers_digest: str) -> Step:
        """Return the step that scores one row's answers."""
        return Step(
            RI_ID,
            "answer-verdicts",
            f"row-{row}",
            {"questions": which, "tolerance": 0.01, "judge_prompt": "evaluation.answers.JUDGE_PROMPT"},
            role="judging",
            reference_model=REFERENCE_JUDGE,
            upstream={"answers": answers_digest},
        )

    def verdicts(self, row: str, which: str = "all") -> StepResult:
        """Return one row's verdicts, each ``{"correct", "method"}``."""
        memo = f"verdicts:{row}:{which}"
        if memo not in self._memo:
            answered = self.answers(row, which)
            found = {str(q["query_id"]): q for q in self.questions(which)}

            def compute(choice: ModelChoice) -> Any:
                def judge(prompt: str) -> str:
                    return self.llm.complete(choice.model, prompt, max_tokens=300).text

                def one(item: tuple[str, Any]) -> tuple[str, dict[str, Any]]:
                    qid, record = item
                    verdict = score_answer(found[qid], str(record["answer"]), judge)
                    return qid, {"correct": verdict.correct, "method": verdict.method}

                return dict(enrich.parallel_map(one, sorted(answered.value.items())))

            self._memo[memo] = self._step(self.verdicts_step(row, which, answered.digest), compute)
        return cast(StepResult, self._memo[memo])


def _texts(value: Mapping[str, Any]) -> dict[str, str]:
    return {eid: str(record.get("text", "")) for eid, record in value.items() if record.get("text")}


def _ingested(corpus: Corpus) -> list[str]:
    return sorted(p.name for p in corpus.ingestion_dir.iterdir() if (p / "document.json").is_file())


def _embedder() -> Any:
    from aieng.advanced_rag.retrieval.embedders import SentenceTransformerEmbedder  # noqa: PLC0415

    return SentenceTransformerEmbedder(EMBEDDER)


def _cap(records: list[dict[str, Any]], size: int, key: str) -> list[dict[str, Any]]:
    """Keep at most ``size`` records, drawn evenly across ``key``."""
    if len(records) <= size:
        return records
    import random  # noqa: PLC0415

    rng = random.Random(SEED)
    shuffled = sorted(records, key=lambda r: str(r["candidate_id"]))
    rng.shuffle(shuffled)
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in shuffled:
        groups.setdefault(str(record.get(key)), []).append(record)
    kept: list[dict[str, Any]] = []
    while len(kept) < size:
        for name in sorted(groups):
            if groups[name] and len(kept) < size:
                kept.append(groups[name].pop())
    return sorted(kept, key=lambda r: str(r["candidate_id"]))


def _controls(universe: Universe, size: int) -> list[dict[str, Any]]:
    """Return FinanceBench questions whose gold pages are all in the universe."""
    import random  # noqa: PLC0415

    pages = set(universe.pages)
    found = [
        q
        for q in financebench.load_queries()
        if q.gold_pages and all(page_id(q.doc_id, p) in pages for p in q.gold_pages)
    ]
    rng = random.Random(SEED)
    chosen = sorted(rng.sample(found, min(size, len(found))), key=lambda q: q.query_id)
    return [
        {
            "candidate_id": q.query_id,
            "doc_id": q.doc_id,
            "gold_pages": [page_id(q.doc_id, p) for p in q.gold_pages],
            "element_ids": [],
            "statement": "",
            "question": q.question,
            "answer": q.answer,
            "value": None,
            "unit": "",
            "subgroup": q.category,
        }
        for q in chosen
    ]


__all__ = [
    "LIVE_SUBSET",
    "REFERENCE_CHECKER",
    "REFERENCE_JUDGE",
    "REFERENCE_VLM",
    "RI_ID",
    "NotCachedError",
    "Pipeline",
    "digest_of",
    "reference_models",
]
