"""Build question sets through ``load_or_run``, live or from the reference, as one path.

A set takes three models, so the generation role alone does not name a build.
:class:`Roles` names all three: the generator is your generation model, and the
judge defaults to your judging model, which for every paid family is the shared
judge. The checker defaults to a cheap model from a different family than the
generator (see :func:`default_checker`). :func:`set_step` puts the checker and
the judge in the cache key beside the generator, so a set checked by one model
is never returned for another.

That makes the reference, built on Vector's on-prem models, a different key
from yours. :func:`load_or_build` asks :func:`~aieng.advanced_rag.runs.load_or_run`
for your result first, and when there is none and ``live`` is off, falls back
to the reference step and prints what rebuilding it on your three models would
cost, priced per role.

:func:`load_folder` reads a folder of text, Markdown, and PDF files into
:class:`~.sampling.Document` records, and :func:`write_beir` exports a set for
one chunking in BEIR's layout, so tools that read BEIR qrels (``pytrec_eval``,
``ranx``, LlamaIndex's ``EmbeddingQAFinetuneDataset``) can score it.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from aieng.advanced_rag import spend
from aieng.advanced_rag.cache import JsonValue, TieredResultCache
from aieng.advanced_rag.models import REFERENCE_MODELS, ModelChoice, ModelSet, family_of
from aieng.advanced_rag.retrieval.corpus import Chunk
from aieng.advanced_rag.runs import (
    ConfirmationRequiredError,
    Provenance,
    ReferenceMissingError,
    Step,
    StepResult,
    _refuse_live_misses,
    _unguarded,
    confirmation_needed,
    load_or_run,
)

from . import filters, generate, nulls, rewrite, statements
from .config import GenerationConfig
from .sampling import Document
from .testset import SyntheticQuestion, SyntheticSet


#: Checker used when the generator is a Google model. The shared judge is also
#: a Google model, so judging-model reuse would check Gemini with Gemini.
CHECKER_FOR_GOOGLE_GENERATOR = "gpt-6-luna"

#: Checker used for every other paid generator.
CHECKER_DEFAULT = "gemini-3.5-flash-lite"

#: On-prem checkers, which cost nothing. The second is used when the generator
#: is the first, so the checker never equals the generator.
ONPREM_CHECKER = "Qwen3_6-35B-A3B"
ONPREM_CHECKER_ALTERNATE = "Qwen3_8-27B"


def default_checker(generator: str) -> str:
    """Return a checker model that differs from the generator.

    A paid generator gets a checker from another family, because a checker from
    the generator's own family tends to share its blind spots. An on-prem
    generator keeps an on-prem checker, so a free build stays free.

    Parameters
    ----------
    generator : str
        The generation model's name.

    Returns
    -------
    str
        ``Qwen3_8-27B`` for a ``Qwen3_6-35B-A3B`` generator and
        ``Qwen3_6-35B-A3B`` for any other on-prem generator, ``gpt-6-luna`` for a
        Google generator, and ``gemini-3.5-flash-lite`` otherwise.
    """
    family = family_of(generator)
    if family == "onprem":
        return (
            ONPREM_CHECKER_ALTERNATE if generator == ONPREM_CHECKER else ONPREM_CHECKER
        )
    if family == "google":
        return CHECKER_FOR_GOOGLE_GENERATOR
    return CHECKER_DEFAULT


@dataclass(frozen=True)
class Roles:
    """The three models one build uses.

    Parameters
    ----------
    generator : str
        Writes statements, questions, near-miss nulls, and paraphrases.
    checker : str
        Answers from the gold quotes, checks uniqueness, and verifies nulls.
    judge : str
        Compares the checker's answer with the reference answer.
    """

    generator: str
    checker: str
    judge: str

    @classmethod
    def from_models(cls, models: ModelSet, *, checker: str | None = None) -> Roles:
        """Return the generation model, a cross-family checker, and the judging model.

        Without ``checker``, :func:`default_checker` picks a cheap model from a
        different family than the generator. Pass ``checker`` to override it,
        for example with the judging model, which makes the check same-family
        for a Google generator.
        """
        generator = models.generation.model
        return cls(
            generator, checker or default_checker(generator), models.judging.model
        )

    def as_dict(self) -> dict[str, str]:
        """Return role name to model name."""
        return {
            "generator": self.generator,
            "checker": self.checker,
            "judge": self.judge,
        }

    def same_family(self) -> bool:
        """Return whether the checker is in the generator's family."""
        return bool(family_of(self.generator) == family_of(self.checker))


#: The on-prem models every committed reference set was built with.
REFERENCE_ROLES = Roles("Qwen3_8-27B", "Qwen3_6-35B-A3B", "Qwen3_8-Flash-Next")


def corpus_digest(documents: Sequence[Document]) -> str:
    """Return a digest of the documents' identifiers and text, in order."""
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.doc_id.encode("utf-8") + b"\x1f")
        digest.update(document.text.encode("utf-8") + b"\x1e")
    return digest.hexdigest()


def prompt_digest() -> str:
    """Return a digest of every prompt a single-hop build sends.

    A changed prompt changes the questions, so it is part of the cache key.
    """
    texts: list[str] = []
    for module in (statements, generate, filters, nulls, rewrite):
        for name in sorted(vars(module)):
            value = getattr(module, name)
            if name.endswith(("_PROMPT", "_SYSTEM")) and isinstance(value, str):
                texts.append(f"{module.__name__}.{name}={value}")
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]


def set_step(
    ri_id: str,
    producer_id: str,
    documents: Sequence[Document],
    config: GenerationConfig,
    roles: Roles,
    *,
    artifact_type: str = "question-set",
    extra: Mapping[str, JsonValue] | None = None,
    guarded: bool = False,
) -> Step:
    """Return the step that builds one set, keyed on everything that changes it.

    Parameters
    ----------
    ri_id, producer_id : str
        Where the step belongs, as in :class:`~aieng.advanced_rag.runs.Step`.
    documents : sequence of Document
        The corpus, which enters the key as a digest.
    config : GenerationConfig
        The build settings.
    roles : Roles
        The checker and the judge enter the config; the generator is the
        step's own model.
    artifact_type : str, optional
        What the step produces.
    extra : mapping or None, optional
        Anything else the compute reads, such as the scoring arms.
    guarded : bool, optional
        Every model call the compute makes goes through a guarded client, such
        as the :func:`~aieng.advanced_rag.evaluation.synthetic.llm.proxy_model`
        models. See :class:`~aieng.advanced_rag.runs.Step`.
    """
    key_config: dict[str, JsonValue] = {
        "corpus": corpus_digest(documents),
        "documents": len(documents),
        "config": json.loads(json.dumps(config.to_dict())),
        "prompts": prompt_digest(),
        "checker": roles.checker,
        "judge": roles.judge,
    }
    key_config.update(dict(extra or {}))
    return Step(
        ri_id=ri_id,
        artifact_type=artifact_type,
        producer_id=producer_id,
        config=key_config,
        role="generation",
        reference_model=REFERENCE_ROLES.generator,
        guarded=guarded,
    )


@dataclass(frozen=True)
class Reference:
    """The reference step to fall back to, and what scales it to your run.

    Parameters
    ----------
    step : Step
        The step the reference was stored under.
    roles : Roles
        The models that built it.
    scale : float, optional
        Your build's size over the reference's, such as a document ratio, so
        the rebuild estimate prices your configuration rather than the
        reference's.
    description : str, optional
        One line naming what the reference covers, printed under its label.
    """

    step: Step
    roles: Roles = REFERENCE_ROLES
    scale: float = 1.0
    description: str = ""


def _generation_set(models: ModelSet | None, generator: str) -> ModelSet:
    base = models or REFERENCE_MODELS
    return replace(
        base, generation=ModelChoice("generation", generator, family_of(generator))
    )


def rebuild_estimate(
    provenance: Provenance | None,
    reference_roles: Roles,
    roles: Roles,
    scale: float = 1.0,
    *,
    table: Mapping[str, spend.ModelPrice] | None = None,
) -> tuple[int, float | None]:
    """Return the calls and dollars your three models would take, priced per role.

    Each model in the reference's usage is mapped to its role, and its tokens
    are priced at your model's rate for that role. Token counts differ across
    tokenizers and completion length is a property of the model, so this is
    approximate.

    Parameters
    ----------
    provenance : Provenance or None
        The reference build's record, with per-model usage.
    reference_roles, roles : Roles
        The models that built the reference, and yours.
    scale : float, optional
        Your build's size over the reference's.
    table : mapping or None, optional
        Rates to price with. :func:`aieng.advanced_rag.spend.price_table`
        when omitted.

    Returns
    -------
    tuple
        Calls, and dollars, or None when a role's model has no known rate.
    """
    if provenance is None or not provenance.usage:
        return 0, None
    mapping = {reference_roles.generator: roles.generator}
    mapping.setdefault(reference_roles.checker, roles.checker)
    mapping.setdefault(reference_roles.judge, roles.judge)
    calls, dollars = 0, 0.0
    priced = True
    for name, row in provenance.usage.items():
        calls += int(row.get("calls", 0))
        cost = spend.estimate_tokens(
            mapping.get(name, roles.generator),
            int(int(row.get("prompt_tokens", 0)) * scale),
            int(int(row.get("completion_tokens", 0)) * scale),
            table=table,
        )
        if cost is None:
            priced = False
        else:
            dollars += cost
    return int(calls * scale), dollars if priced else None


def load_or_build(
    step: Step,
    compute: Callable[[ModelChoice], Any],
    *,
    roles: Roles,
    models: ModelSet | None = None,
    live: bool = False,
    confirm: bool = False,
    cache: TieredResultCache | None = None,
    reference: Reference | None = None,
    show: bool = True,
) -> StepResult:
    """Return your set, the labelled reference, or a live build on your models.

    Your own cached result comes first. Without one, the reference is shown
    with what your build would cost, priced per role; with ``live`` on, your
    build then runs, unless ``confirm`` is false and its estimate is over 10%
    of the key's budget left, or over $1 while the key reports no budget and
    $300 is assumed (see :func:`~aieng.advanced_rag.runs.confirmation_needed`).

    Parameters
    ----------
    step : Step
        From :func:`set_step`, for your roles and configuration.
    compute : callable
        Builds the set; called as ``compute(choice)`` by ``load_or_run``.
    roles : Roles
        Your three models.
    models : ModelSet or None, optional
        Your model set, from ``choose_models``.
    live : bool, optional
        Build on your models when no result of yours exists. Pass the
        notebook's ``RUN_EXPENSIVE``.
    confirm : bool, optional
        Build even when the estimate is over 10% of the budget left, or over
        $1 while the budget is assumed.
    cache : TieredResultCache or None, optional
        The cache, whose shipped tier holds the committed reference.
    reference : Reference or None, optional
        What to show when you have no result. None means no reference
        applies, as for your own folder.
    show : bool, optional
        Print the labels.

    Returns
    -------
    StepResult
        As :func:`~aieng.advanced_rag.runs.load_or_run` returns it.

    Raises
    ------
    ReferenceMissingError
        With ``live`` off, when you have no result and no reference applies.
    ConfirmationRequiredError
        With ``live`` on and ``confirm`` false, when the estimate is over 10%
        of the budget left, or over $1 while the budget is assumed: before the
        build, or for a step marked ``guarded``, at the build's first paid call
        the response cache cannot answer, which is not sent.
    """
    cache = cache or TieredResultCache()
    yours = _generation_set(models, roles.generator)
    own = cache.get(step.key(yours.generation)) is not None
    if (
        own
        or reference is None
        or reference.step.key(
            _generation_set(None, reference.roles.generator).generation
        )
        == step.key(yours.generation)
    ):
        if not own and not live:
            raise ReferenceMissingError(
                f"no cached {step.artifact_type} for {roles.generator} on this "
                "corpus and configuration; set RUN_EXPENSIVE = True to build it"
            )
        return load_or_run(
            step,
            compute,
            models=yours,
            live=live,
            confirm=confirm,
            cache=cache,
            show=show,
        )

    try:
        shown: StepResult | None = load_or_run(
            reference.step,
            compute,
            models=_generation_set(models, reference.roles.generator),
            live=False,
            cache=cache,
            show=show,
        )
    except ReferenceMissingError:
        # A live build does not need the reference, only its estimate.
        if not live:
            raise
        shown = None
    calls, dollars = rebuild_estimate(
        shown.provenance if shown is not None else None,
        reference.roles,
        roles,
        reference.scale,
    )
    pad = " " * (len(step.artifact_type) + 1)
    if show:
        if reference.description:
            print(pad + reference.description)
        cost = (
            f"est. ${dollars:,.2f}"
            if dollars is not None
            else "no estimate: a role's model has no price"
        )
        tail = "" if live else " (set RUN_EXPENSIVE = True to build)"
        print(
            pad + f"your build: {roles.generator} generates, {roles.checker} "
            f"checks, {roles.judge} judges; about {calls:,} calls, {cost}{tail}"
        )
    if not live and shown is not None:
        return shown
    reason = (
        confirmation_needed(dollars, spend.read_budget())
        if dollars and not confirm
        else None
    )
    if reason is not None and not step.guarded:
        raise ConfirmationRequiredError(_unguarded(reason))
    if reason is not None and show:
        print(
            pad + "over the confirmation threshold: running on cached responses "
            "and stopping before the first paid call the cache cannot answer"
        )
    # The estimate is printed and the budget checked above, so load_or_run's
    # own line, which finds no reference under your key, is left out. A build
    # over the threshold runs only as far as the response cache answers it.
    captured = io.StringIO()
    with (
        contextlib.redirect_stdout(captured),
        _refuse_live_misses(reason) if reason else contextlib.nullcontext(),
    ):
        result = load_or_run(
            step,
            compute,
            models=yours,
            live=True,
            confirm=True,
            cache=cache,
            show=show,
        )
    for line in captured.getvalue().splitlines():
        if not line.startswith(f"{step.artifact_type} running on "):
            print(line)
    return result


def set_value(generated: SyntheticSet) -> dict[str, Any]:
    """Return a set as the JSON value a step stores."""
    return {
        "questions": [q.to_dict() for q in generated],
        "funnel": generated.funnel.to_dict() if generated.funnel else {},
        "meta": generated.meta,
    }


def set_from_value(value: Mapping[str, Any]) -> SyntheticSet:
    """Rebuild a set from :func:`set_value` output, keeping the funnel in ``meta``."""
    meta = dict(value.get("meta") or {})
    meta["funnel"] = value.get("funnel") or {}
    return SyntheticSet(
        [SyntheticQuestion.from_dict(r) for r in value["questions"]], None, meta
    )


#: File types :func:`load_folder` reads.
FOLDER_SUFFIXES = (".txt", ".md", ".pdf")


def _pdf_text(path: Path) -> str:
    import pypdfium2  # noqa: PLC0415

    pdf = pypdfium2.PdfDocument(str(path))
    try:
        return "\n\n".join(page.get_textpage().get_text_range() for page in pdf)
    finally:
        pdf.close()


def load_folder(folder: Path, *, limit: int | None = None) -> list[Document]:
    """Read every text, Markdown, and PDF file under ``folder`` into a document.

    The document id is the path relative to ``folder``. PDFs are read through
    their text layer with ``pypdfium2``, so a scanned PDF with no text layer
    comes back empty and is skipped. For tables and figures, parse with Docling
    as RI #3 does and pass the parsed text instead.

    Parameters
    ----------
    folder : Path
        The corpus folder.
    limit : int or None, optional
        Read at most this many files, in sorted path order.

    Returns
    -------
    list of Document
        One per non-empty file.

    Raises
    ------
    FileNotFoundError
        When ``folder`` holds no readable file.
    """
    folder = Path(folder)
    paths = sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in FOLDER_SUFFIXES
    )
    documents: list[Document] = []
    for path in paths:
        if limit is not None and len(documents) >= limit:
            break
        text = (
            _pdf_text(path)
            if path.suffix.lower() == ".pdf"
            else path.read_text(encoding="utf-8", errors="replace")
        )
        if text.strip():
            documents.append(Document(str(path.relative_to(folder)), text))
    if not documents:
        raise FileNotFoundError(
            f"no {', '.join(FOLDER_SUFFIXES)} file with text under {folder}"
        )
    return documents


def write_beir(
    generated: SyntheticSet, chunks: Sequence[Chunk], directory: Path
) -> dict[str, int]:
    """Write a set and one chunking in BEIR's layout.

    BEIR names relevance by corpus id, so the export holds for this chunking
    only. The set's character spans stay the source of truth: export again
    for another chunker. Null questions go into ``queries.jsonl`` with no
    qrels row, which is how BEIR represents a question with no relevant
    passage.

    Parameters
    ----------
    generated : SyntheticSet
        The set, one wording variant.
    chunks : sequence of Chunk
        The chunking to name relevance in.
    directory : Path
        Receives ``corpus.jsonl``, ``queries.jsonl``, and ``qrels/test.tsv``.

    Returns
    -------
    dict
        Rows written per file.
    """
    directory = Path(directory)
    (directory / "qrels").mkdir(parents=True, exist_ok=True)
    qrels = generated.qrels(chunks)
    with (directory / "corpus.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {"_id": chunk.chunk_id, "title": chunk.doc_id, "text": chunk.text},
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (directory / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for q in generated:
            record = {
                "_id": q.query_id,
                "text": q.question,
                "metadata": {"answerable": q.is_answerable, "kind": q.kind},
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    rows = 0
    with (directory / "qrels" / "test.tsv").open("w", encoding="utf-8") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for query_id, relevant in qrels.items():
            for chunk_id in sorted(relevant):
                handle.write(f"{query_id}\t{chunk_id}\t1\n")
                rows += 1
    return {"corpus": len(chunks), "queries": len(generated), "qrels": rows}


__all__ = [
    "FOLDER_SUFFIXES",
    "REFERENCE_ROLES",
    "Reference",
    "Roles",
    "corpus_digest",
    "load_folder",
    "load_or_build",
    "prompt_digest",
    "rebuild_estimate",
    "set_from_value",
    "set_step",
    "set_value",
    "write_beir",
]
