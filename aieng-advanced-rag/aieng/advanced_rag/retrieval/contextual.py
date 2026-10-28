"""Contextual retrieval: an LLM writes a situating sentence per chunk.

Anthropic's recipe. Before embedding, ask a model what the chunk is about in the
context of its document, and prepend the answer. A chunk that reads "the Company
shall indemnify the Provider" becomes a chunk that also says which agreement and
which parties, so the query "indemnification obligations in the Acme
distribution agreement" can match it.

The cost is one model call per chunk indexed, which is why this module counts
calls and tokens rather than only wall time. On a 20-contract CUAD slice that is
954 calls, and the decision of whether to teach it rests on that number as much
as on the retrieval gain.

Two prompts are available, named by a prompt version so a stored context always
says which one wrote it:

* ``head-neighbourhood-v1`` (:func:`build_prompt`) carries the head of the
  document, which on a contract holds the title, parties and date, plus a
  neighbourhood around the chunk. It bounds the prompt when no prompt cache is
  available: a 47,000-character contract across 954 calls would otherwise be 11
  million prompt tokens.
* ``full-document-v1`` (:func:`build_full_document_prompt`) is Anthropic's
  published shape: the whole document first, then the chunk. The document is a
  prefix shared by every chunk of that document, so a provider that caches
  prompt prefixes bills it once at the write rate and then at the read rate.
* ``full-document-named-v1`` keeps that prefix and asks, as the head prompt
  does, for the document, the parties, and the section.

:func:`generate_contexts` writes contexts with any OpenAI-compatible client, or
sends them as one batch through :class:`aieng.advanced_rag.batch.ProxyBatch`, and
returns a :class:`ContextSet`: one record per chunk identifier, labelled with the
model and prompt version, carrying the tokens each call billed. The set is plain
JSON, so a context written once on one model can be replayed by anyone, and a
participant can regenerate it on their own model and compare.

:func:`metadata_header` is the free arm: a header built from document metadata
and the detected section heading, with no model call.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .. import metering
from ..metering import RunStopped
from ..models import cacheable_system, profile_for
from .corpus import Chunk, Query


SYSTEM = (
    "You situate an excerpt within its source document for a search index. "
    "Answer with one or two short sentences and nothing else."
)

PROMPT = """Here is the start of a document:
<document_head>
{head}
</document_head>

Here is the region of the document surrounding an excerpt:
<neighbourhood>
{neighbourhood}
</neighbourhood>

Here is the excerpt:
<excerpt>
{chunk}
</excerpt>

Give a short context that situates this excerpt within the document, so that a
search engine can retrieve it. Name the document, the parties or system it
concerns, and the section the excerpt belongs to, where the text supports it.
Answer with the context only."""


@dataclass
class ContextualCost:
    """Calls and tokens spent contextualising a chunk set.

    Parameters
    ----------
    chunks : int
        Chunks submitted.
    calls : int
        Calls that reached the model, as the client's usage counter reports.
    cached_calls : int
        Chunks served from the client's response cache.
    prompt_tokens : int
        Prompt tokens billed.
    completion_tokens : int
        Completion tokens billed.
    wall_seconds : float
        Wall time of the whole pass.
    failures : int
        Chunks whose call raised and were left uncontextualised.
    cache_read_tokens : int, optional
        Prompt tokens a provider cache served, already inside ``prompt_tokens``.
    cache_write_tokens : int, optional
        Prompt tokens written to a provider cache, already inside
        ``prompt_tokens``. A write bills above the input rate, so the split
        matters for the price.
    """

    chunks: int = 0
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_seconds: float = 0.0
    failures: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Return prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serialisable summary, with per-chunk rates."""
        cold = max(self.calls, 1)
        summary: dict[str, float | int] = {
            "chunks_contextualised": self.chunks,
            "cold_calls": self.calls,
            "cache_hits": self.cached_calls,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_tokens_per_chunk": round(self.prompt_tokens / cold, 1),
            "completion_tokens_per_chunk": round(self.completion_tokens / cold, 1),
            "total_tokens_per_chunk": round(self.total_tokens / cold, 1),
            "wall_seconds": round(self.wall_seconds, 1),
        }
        # Left out when zero so an on-prem summary keeps its committed shape.
        if self.cache_read_tokens:
            summary["cache_read_tokens"] = self.cache_read_tokens
        if self.cache_write_tokens:
            summary["cache_write_tokens"] = self.cache_write_tokens
        return summary


def build_prompt(
    document: str,
    chunk: Chunk,
    *,
    document_head_chars: int = 3000,
    neighbourhood_chars: int = 1500,
) -> str:
    """Render the situating prompt for one chunk.

    Parameters
    ----------
    document : str
        The whole source document.
    chunk : Chunk
        The chunk to situate.
    document_head_chars : int, optional
        Characters of the document head to include.
    neighbourhood_chars : int, optional
        Characters of context on each side of the chunk.

    Returns
    -------
    str
        The rendered prompt.
    """
    head = document[:document_head_chars]
    low = max(chunk.start - neighbourhood_chars, 0)
    high = min(chunk.end + neighbourhood_chars, len(document))
    return PROMPT.format(head=head, neighbourhood=document[low:high], chunk=chunk.text)


def contextualise(
    documents: dict[str, str],
    chunks: Sequence[Chunk],
    make_client: Callable[[], Any],
    *,
    document_head_chars: int = 3000,
    neighbourhood_chars: int = 1500,
    max_tokens: int = 160,
    workers: int = 8,
    progress_every: int = 100,
) -> tuple[list[Chunk], ContextualCost]:
    """Prepend an LLM-written context to every chunk's indexed text.

    Parameters
    ----------
    documents : dict
        Document identifier to full text.
    chunks : sequence of Chunk
        The chunks to contextualise.
    make_client : callable
        Called once per worker thread, and must return an object with a
        ``complete(prompt, system=..., max_tokens=..., kind=...)`` method and a
        ``usage`` attribute. Each client keeps its own connection to a shared
        response cache, so a re-run costs nothing and the reported cost stays
        the cost of a cold run.
    document_head_chars : int, optional
        Characters of the document head in each prompt.
    neighbourhood_chars : int, optional
        Characters of context on each side of the chunk.
    max_tokens : int, optional
        Completion budget per call.
    workers : int, optional
        Threads in the pool.
    progress_every : int, optional
        Print a progress line every N chunks. Zero silences it.

    Returns
    -------
    tuple of (list of Chunk, ContextualCost)
        Chunks whose ``embed_text`` carries the context, and what it cost.
    """
    cost = ContextualCost(chunks=len(chunks))
    lock = threading.Lock()
    local = threading.local()
    done = 0
    began = time.perf_counter()

    clients: list[Any] = []

    def client_for_thread() -> Any:
        if not hasattr(local, "client"):
            local.client = make_client()
            with lock:
                clients.append(local.client)
        return local.client

    stopped = threading.Event()

    def one(chunk: Chunk) -> Chunk:
        nonlocal done
        if stopped.is_set():
            return chunk
        client = client_for_thread()
        prompt = build_prompt(
            documents[chunk.doc_id],
            chunk,
            document_head_chars=document_head_chars,
            neighbourhood_chars=neighbourhood_chars,
        )
        try:
            context = client.complete(
                prompt,
                system=SYSTEM,
                max_tokens=max_tokens,
                kind="contextualise",
            ).strip()
        except RunStopped:
            stopped.set()
            raise
        except Exception:  # noqa: BLE001 - a failed call must not lose the chunk
            with lock:
                cost.failures += 1
            return chunk
        with lock:
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"  contextualised {done}/{len(chunks)}", flush=True)
        return chunk.with_embed_text(f"{context}\n\n{chunk.text}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        contextualised = list(pool.map(one, chunks))

    cost.wall_seconds = time.perf_counter() - began
    # Usage lives on each thread's client, so fold the pool's clients together.
    collect_usage(clients, cost)
    return contextualised, cost


def collect_usage(clients: Sequence[Any], cost: ContextualCost) -> ContextualCost:
    """Fold per-thread client usage into a cost record.

    A cached response is not counted by a metered client, so these totals are
    the cost of a cold run even when the run itself was served from cache.

    Parameters
    ----------
    clients : sequence
        The clients the pool created.
    cost : ContextualCost
        The record to update in place.

    Returns
    -------
    ContextualCost
        The same record.
    """
    for client in clients:
        usage = getattr(client, "usage", None)
        if usage is None:
            continue
        cost.calls += usage.calls
        cost.prompt_tokens += usage.prompt_tokens
        cost.completion_tokens += usage.completion_tokens
        cost.cache_read_tokens += getattr(usage, "prompt_cache_read_tokens", 0)
        cost.cache_write_tokens += getattr(usage, "prompt_cache_write_tokens", 0)
    cost.cached_calls = max(cost.chunks - cost.calls - cost.failures, 0)
    return cost


HEAD_NEIGHBOURHOOD = "head-neighbourhood-v1"
FULL_DOCUMENT = "full-document-v1"
FULL_DOCUMENT_NAMED = "full-document-named-v1"
PROMPT_VERSIONS = (HEAD_NEIGHBOURHOOD, FULL_DOCUMENT, FULL_DOCUMENT_NAMED)

# The document comes first and alone, so it is a byte-identical prefix for every
# chunk of that document. Anything that varies per chunk has to sit after it, or
# a prefix cache never hits.
FULL_DOCUMENT_PREFIX = "<document>\n{document}\n</document>"

# Anthropic's published contextual-retrieval prompt, with the courtesy word
# removed. Keep the wording stable: changing it changes ``full-document-v1``.
FULL_DOCUMENT_SUFFIX = """Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Give a short succinct context to situate this chunk within the overall document
for the purposes of improving search retrieval of the chunk. Answer only with
the succinct context and nothing else."""


# The full-document prefix with the head prompt's instruction: name the document,
# the parties, and the section. Anthropic's wording asks only to situate the
# chunk, and on contracts the model then describes the clause and leaves out the
# parties, which is the part an unscoped query needs.
FULL_DOCUMENT_NAMED_SUFFIX = """Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Give a short context that situates this chunk within the document, so that a
search engine can retrieve it. Name the document, the parties or system it
concerns, and the section the chunk belongs to, where the text supports it.
Answer with the context only."""


def build_full_document_prompt(
    document: str, chunk: Chunk, *, name_the_document: bool = False
) -> tuple[str, str]:
    """Render the full-document prompt as a cacheable prefix and a per-chunk suffix.

    Parameters
    ----------
    document : str
        The whole source document.
    chunk : Chunk
        The chunk to situate.
    name_the_document : bool, optional
        Use the instruction that asks for the document, parties, and section.

    Returns
    -------
    tuple of (str, str)
        The prefix, identical for every chunk of the document, and the suffix
        that carries the chunk.
    """
    suffix = FULL_DOCUMENT_NAMED_SUFFIX if name_the_document else FULL_DOCUMENT_SUFFIX
    return (
        FULL_DOCUMENT_PREFIX.format(document=document),
        suffix.format(chunk=chunk.text),
    )


def context_messages(
    prompt_version: str,
    document: str,
    chunk: Chunk,
    *,
    cache_control: bool = False,
    model: str | None = None,
    document_head_chars: int = 3000,
    neighbourhood_chars: int = 1500,
) -> tuple[list[dict[str, Any]], int]:
    """Build the chat messages for one chunk and the length of its shared prefix.

    Parameters
    ----------
    prompt_version : str
        One of :data:`PROMPT_VERSIONS`.
    document : str
        The whole source document.
    chunk : Chunk
        The chunk to situate.
    cache_control : bool, optional
        Mark the shared prefix with an Anthropic ``cache_control`` breakpoint, as
        a separate text part. Leave it off for providers that cache prefixes on
        their own or reject content parts.
    model : str, optional
        The model the messages are for. With ``cache_control`` set and a model
        whose family caches (see :func:`aieng.advanced_rag.models.cacheable_system`),
        the instructions and the shared prefix move into one cacheable system
        message and the user message carries only the chunk, which is the one form
        the proxy caches. A model that does not cache gets the same messages as
        without ``cache_control`` set, byte for byte.
    document_head_chars : int, optional
        Head length for the head-and-neighbourhood prompt.
    neighbourhood_chars : int, optional
        Neighbourhood on each side for the head-and-neighbourhood prompt.

    Returns
    -------
    tuple of (list of dict, int)
        The messages, and the characters of the system prompt plus the part of
        the user message that every chunk of this document shares. The price
        estimate uses that share to split billed prompt tokens into a cacheable
        part and a per-chunk part.

    Raises
    ------
    ValueError
        If ``prompt_version`` is not one of :data:`PROMPT_VERSIONS`.
    """
    if prompt_version in (FULL_DOCUMENT, FULL_DOCUMENT_NAMED):
        prefix, suffix = build_full_document_prompt(
            document, chunk, name_the_document=prompt_version == FULL_DOCUMENT_NAMED
        )
        separator = "\n\n"
    elif prompt_version == HEAD_NEIGHBOURHOOD:
        whole = build_prompt(
            document,
            chunk,
            document_head_chars=document_head_chars,
            neighbourhood_chars=neighbourhood_chars,
        )
        # The head block ends at the first closing tag, and it is the same for
        # every chunk of a document.
        cut = whole.index("</document_head>") + len("</document_head>")
        prefix, suffix = whole[:cut], whole[cut:].lstrip("\n")
        separator = "\n\n"
    else:
        raise ValueError(f"unknown prompt version {prompt_version!r}")

    if cache_control and model is not None and not profile_for(model).cache_control:
        # A family that does not mark prefixes gets the plain request, so its
        # response cache keeps hitting.
        cache_control = False
    if cache_control and model is not None:
        return (
            [
                cacheable_system(model, SYSTEM, prefix),
                {"role": "user", "content": suffix},
            ],
            len(SYSTEM) + len(prefix),
        )
    content: Any
    if cache_control:
        content = [
            {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": suffix},
        ]
    else:
        content = f"{prefix}{separator}{suffix}"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]
    return messages, len(SYSTEM) + len(prefix)


@dataclass(frozen=True)
class ContextRecord:
    """One written context and what its call billed.

    Parameters
    ----------
    doc_id : str
        The chunk's document, which is the unit a prefix cache is shared over.
    context : str
        The situating text the model wrote.
    prompt_tokens : int
        Prompt tokens the call reported.
    completion_tokens : int
        Completion tokens the call reported.
    prefix_chars : int
        Characters of the shared prefix, system prompt included.
    prompt_chars : int
        Characters of the whole prompt, system prompt included.
    cached_tokens : int, optional
        Prompt tokens the provider reported as read from its cache. The on-prem
        models report none.
    cache_write_tokens : int, optional
        Prompt tokens the provider reported as written to its cache, which bill
        above the input rate.
    """

    doc_id: str
    context: str
    prompt_tokens: int
    completion_tokens: int
    prefix_chars: int
    prompt_chars: int
    cached_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def prefix_tokens(self) -> int:
        """Return the prompt tokens of the shared prefix, by character share."""
        if self.prompt_chars <= 0:
            return 0
        share = min(self.prefix_chars / self.prompt_chars, 1.0)
        return round(self.prompt_tokens * share)


@dataclass
class ContextSet:
    """Contexts keyed by chunk identifier, labelled with the model that wrote them.

    Parameters
    ----------
    model : str
        The model that wrote every context in the set.
    prompt_version : str
        The prompt that produced them.
    records : dict
        Chunk identifier to its :class:`ContextRecord`.
    failures : list of str, optional
        Chunks whose call raised. They stay uncontextualised when applied.
    """

    model: str
    prompt_version: str
    records: dict[str, ContextRecord] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def apply(
        self, chunks: Sequence[Chunk], template: str = "{context}\n\n{text}"
    ) -> list[Chunk]:
        """Return the chunks with their context prepended to the indexed text.

        The span and ``text`` stay unchanged, so span-derived judgments still
        hold. A chunk with no record is returned as it was.

        Parameters
        ----------
        chunks : sequence of Chunk
            The chunks the contexts were written for.
        template : str, optional
            How context and chunk text combine, with ``{context}`` and ``{text}``.

        Returns
        -------
        list of Chunk
            One chunk per input, in the input order.
        """
        out: list[Chunk] = []
        for chunk in chunks:
            record = self.records.get(chunk.chunk_id)
            if record is None or not record.context:
                out.append(chunk)
            else:
                out.append(
                    chunk.with_embed_text(
                        template.format(context=record.context, text=chunk.text)
                    )
                )
        return out

    def usage(self) -> dict[str, int]:
        """Return calls and tokens summed over the set, as a cold run bills them."""
        summary = {
            "calls": len(self.records),
            "failures": len(self.failures),
            "prompt_tokens": sum(r.prompt_tokens for r in self.records.values()),
            "completion_tokens": sum(
                r.completion_tokens for r in self.records.values()
            ),
            "cached_tokens_reported": sum(
                r.cached_tokens for r in self.records.values()
            ),
        }
        written = sum(r.cache_write_tokens for r in self.records.values())
        if written:
            summary["cache_write_tokens_reported"] = written
        return summary

    def to_json(self) -> dict[str, Any]:
        """Return the set as plain JSON, for the artifact cache."""
        return {
            "model": self.model,
            "prompt_version": self.prompt_version,
            "failures": list(self.failures),
            "records": {
                chunk_id: {
                    **(
                        {"cache_write_tokens": r.cache_write_tokens}
                        if r.cache_write_tokens
                        else {}
                    ),
                    "doc_id": r.doc_id,
                    "context": r.context,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "prefix_chars": r.prefix_chars,
                    "prompt_chars": r.prompt_chars,
                    "cached_tokens": r.cached_tokens,
                }
                for chunk_id, r in sorted(self.records.items())
            },
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> ContextSet:
        """Rebuild a set from :meth:`to_json` output.

        Parameters
        ----------
        value : mapping
            What :meth:`to_json` returned.

        Returns
        -------
        ContextSet
            The set.
        """
        return cls(
            model=str(value["model"]),
            prompt_version=str(value["prompt_version"]),
            failures=[str(x) for x in value.get("failures", [])],
            records={
                str(chunk_id): ContextRecord(
                    doc_id=str(r["doc_id"]),
                    context=str(r["context"]),
                    prompt_tokens=int(r["prompt_tokens"]),
                    completion_tokens=int(r["completion_tokens"]),
                    prefix_chars=int(r["prefix_chars"]),
                    prompt_chars=int(r["prompt_chars"]),
                    cached_tokens=int(r.get("cached_tokens", 0)),
                    cache_write_tokens=int(r.get("cache_write_tokens", 0)),
                )
                for chunk_id, r in value["records"].items()
            },
        )


def _message_chars(messages: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            total += len(content)
        else:
            total += sum(len(part.get("text", "")) for part in content)
    return total


def generate_contexts(
    documents: Mapping[str, str],
    chunks: Sequence[Chunk],
    client: Any,
    *,
    model: str,
    prompt_version: str = FULL_DOCUMENT,
    cache_control: bool | None = None,
    max_tokens: int = 160,
    temperature: float = 0.0,
    workers: int = 4,
    progress_every: int = 100,
    confirm: bool = False,
) -> ContextSet:
    """Write a situating context for every chunk with an OpenAI-compatible client.

    Pass a :class:`aieng.advanced_rag.batch.ProxyBatch` to send every chunk as one
    infermesh batch, or a client wrapped by
    :func:`aieng.advanced_rag.graphrag.proxy.govern` to send them from a thread
    pool. Either way every call is throttled to the model's budget and cached,
    and a re-run costs nothing, because both key a request the same way. The
    cached response still carries the usage the cold call reported, so the set
    records the cold cost on a re-run too.

    LiteLLM removes ``cache_control`` from the messages it sends, so through a
    ``ProxyBatch`` a Claude model does not use its prompt cache. Pass a governed
    client for a full-document prompt on a family that caches.

    Chunks are submitted in document order, which keeps one document's calls
    close together in time. A prefix cache with a 5-minute lifetime then serves
    every chunk after the first.

    Parameters
    ----------
    documents : mapping
        Document identifier to full text.
    chunks : sequence of Chunk
        The chunks to situate.
    client : Any
        A :class:`~aieng.advanced_rag.batch.ProxyBatch` for ``model``, or anything
        with ``chat.completions.create(model=..., messages=..., ...)``.
    model : str
        The model to call, recorded on the set.
    prompt_version : str, optional
        One of :data:`PROMPT_VERSIONS`.
    cache_control : bool, optional
        Mark the shared prefix with an Anthropic cache breakpoint. ``None``
        decides from the model's family: on for a family that caches, off for
        the rest, so an on-prem request is the same bytes with or without this
        argument and its cached responses keep hitting.
    max_tokens : int, optional
        Completion budget per call.
    temperature : float, optional
        Sampling temperature.
    workers : int, optional
        Threads in the pool. The limiter inside the client sets the rate. A
        ``ProxyBatch`` sets its own concurrency and ignores this.
    progress_every : int, optional
        Print a progress line every N chunks. Zero silences it.
    confirm : bool, optional
        Passed to :meth:`~aieng.advanced_rag.batch.ProxyBatch.complete_batch`:
        send a paid batch whose estimate is over 10% of the key's budget left,
        or over $1 while the key reports no budget. Without it such a batch
        raises ``ConfirmationRequiredError`` before any call. Ignored for a
        client that is not a ``ProxyBatch``.

    Returns
    -------
    ContextSet
        One record per chunk whose call succeeded.
    """
    result = ContextSet(model=model, prompt_version=prompt_version)
    if cache_control is None:
        cache_control = profile_for(model).cache_control
    lock = threading.Lock()
    done = 0
    order = sorted(chunks, key=lambda c: (c.doc_id, c.start, c.chunk_id))
    requests = [
        context_messages(
            prompt_version,
            documents[chunk.doc_id],
            chunk,
            cache_control=cache_control,
            model=model,
        )
        for chunk in order
    ]

    def record(
        chunk: Chunk, request: tuple[list[dict[str, Any]], int], response: Any
    ) -> None:
        messages, prefix_chars = request
        content = (
            response.choices[0].message.content
            if response is not None and response.choices
            else None
        )
        if response is None or not content or not content.strip():
            result.failures.append(chunk.chunk_id)
            return
        # usage_of counts cache reads and writes inside prompt_tokens on every
        # family, so a cold Anthropic call is not under-reported.
        usage = metering.usage_of(response)
        result.records[chunk.chunk_id] = ContextRecord(
            doc_id=chunk.doc_id,
            context=content.strip(),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            prefix_chars=prefix_chars,
            prompt_chars=_message_chars(messages),
            cached_tokens=usage.prompt_cache_read_tokens,
            cache_write_tokens=usage.prompt_cache_write_tokens,
        )

    def progress(finished: int, total: int) -> None:
        if (
            progress_every
            and finished // progress_every > (finished - 1) // progress_every
        ):
            print(f"  contexts {finished}/{total}", flush=True)

    if hasattr(client, "complete_batch"):
        outcomes = client.complete_batch(
            [messages for messages, _ in requests],
            max_tokens=max_tokens,
            temperature=temperature,
            on_progress=progress,
            confirm=confirm,
        )
        for chunk, request, outcome in zip(order, requests, outcomes, strict=True):
            record(chunk, request, outcome.response)
        result.failures.sort()
        return result

    stopped = threading.Event()

    def one(item: tuple[Chunk, tuple[list[dict[str, Any]], int]]) -> None:
        nonlocal done
        if stopped.is_set():
            return
        chunk, request = item
        try:
            response = client.chat.completions.create(
                model=model,
                messages=request[0],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except RunStopped:
            stopped.set()
            raise
        except Exception:  # noqa: BLE001 - a failed call must not lose the run
            response = None
        with lock:
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"  contexts {done}/{len(order)}", flush=True)
            record(chunk, request, response)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, zip(order, requests, strict=True)))
    result.failures.sort()
    return result


@dataclass(frozen=True)
class ModelPrice:
    """A model's list price per million tokens, for an estimate.

    Parameters
    ----------
    model : str
        The model identifier the price belongs to.
    input_per_mtok : float
        US dollars per million uncached prompt tokens.
    output_per_mtok : float
        US dollars per million completion tokens.
    cache_write_multiplier : float or None, optional
        Price of a cache write relative to input. ``None`` means the provider
        offers no explicit prefix cache, or its rate is not recorded here, and
        the estimate bills every prompt token at the input price.
    cache_read_multiplier : float or None, optional
        Price of a cache read relative to input.
    min_cacheable_tokens : int, optional
        A prefix shorter than this is never cached.
    """

    model: str
    input_per_mtok: float
    output_per_mtok: float
    cache_write_multiplier: float | None = None
    cache_read_multiplier: float | None = None
    min_cacheable_tokens: int = 0

    @property
    def caches(self) -> bool:
        """Return whether a cached estimate is defined for this price."""
        return (
            self.cache_write_multiplier is not None
            and self.cache_read_multiplier is not None
        )


def price_contexts(
    contexts: ContextSet,
    price: ModelPrice,
    *,
    cached: bool = False,
) -> dict[str, str | bool | float | int]:
    """Estimate what writing a context set would cost at list price.

    The token counts are the ones the set's own model reported. Another model's
    tokenizer counts differently, so treat the result as an estimate of the
    order of cost.

    With ``cached`` true, the first chunk of each document pays the cache-write
    rate on the shared prefix and every later chunk of that document pays the
    read rate, which assumes the calls for one document land inside the cache
    lifetime. :func:`generate_contexts` submits them in that order.

    Parameters
    ----------
    contexts : ContextSet
        The measured set.
    price : ModelPrice
        The list price to apply.
    cached : bool, optional
        Apply prefix caching.

    Returns
    -------
    dict
        Prompt tokens split into uncached, cache-write, and cache-read, the
        completion tokens, and the dollar estimate.

    Raises
    ------
    ValueError
        If ``cached`` is true and ``price`` defines no cache rates.
    """
    if cached and not price.caches:
        raise ValueError(f"no cache rates recorded for {price.model}")
    uncached = written = read = 0
    seen: set[str] = set()
    for chunk_id in sorted(contexts.records):
        record = contexts.records[chunk_id]
        prefix = record.prefix_tokens
        if not cached or prefix < price.min_cacheable_tokens:
            uncached += record.prompt_tokens
            continue
        uncached += record.prompt_tokens - prefix
        if record.doc_id in seen:
            read += prefix
        else:
            seen.add(record.doc_id)
            written += prefix
    completion = sum(r.completion_tokens for r in contexts.records.values())
    write_rate = price.cache_write_multiplier or 1.0
    read_rate = price.cache_read_multiplier or 1.0
    dollars = (
        uncached * price.input_per_mtok
        + written * price.input_per_mtok * write_rate
        + read * price.input_per_mtok * read_rate
        + completion * price.output_per_mtok
    ) / 1_000_000
    return {
        "model": price.model,
        "cached": cached,
        "uncached_prompt_tokens": uncached,
        "cache_write_tokens": written,
        "cache_read_tokens": read,
        "completion_tokens": completion,
        "usd": round(dollars, 4),
    }


NAME_SYSTEM = "You name documents for a search index. Answer with the name only."

NAME_PROMPT = """Here is the start of an agreement:
<document_head>
{head}
</document_head>

Write the name a person would use to refer to this agreement, in at most 12
words: the parties or the organisation that issued it, and the kind of
agreement, for example "the mutual NDA between Acme Ltd and Globex". Leave out
blanks and placeholders such as [NAME]. Answer with the name only."""


def name_documents(
    documents: Mapping[str, str],
    client: Any,
    *,
    model: str,
    head_chars: int = 2000,
    max_tokens: int = 40,
    workers: int = 4,
    confirm: bool = False,
) -> tuple[dict[str, str], dict[str, int]]:
    """Ask a model for the name a person would use for each document.

    The names turn a templated question that fits every document into one that
    names its document, which is the condition under which unscoped retrieval
    has to find the right document at all.

    Parameters
    ----------
    documents : mapping
        Document identifier to full text.
    client : Any
        A :class:`~aieng.advanced_rag.batch.ProxyBatch` for ``model``, which sends
        every document as one batch, or an OpenAI-compatible client, ideally
        wrapped by ``govern``.
    model : str
        The model to call.
    head_chars : int, optional
        Characters of the document head shown to the model.
    max_tokens : int, optional
        Completion budget per call.
    workers : int, optional
        Threads in the pool.
    confirm : bool, optional
        Passed to :meth:`~aieng.advanced_rag.batch.ProxyBatch.complete_batch`,
        as in :func:`generate_contexts`.

    Returns
    -------
    tuple of (dict, dict)
        Document identifier to name, and the calls and tokens spent. A document
        whose call failed is missing from the names.
    """
    names: dict[str, str] = {}
    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    lock = threading.Lock()
    order = sorted(documents.items())

    def messages_for(text: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": NAME_SYSTEM},
            {"role": "user", "content": NAME_PROMPT.format(head=text[:head_chars])},
        ]

    def record(doc_id: str, response: Any) -> None:
        content = (
            response.choices[0].message.content
            if response is not None and response.choices
            else None
        )
        if not content:
            return
        with lock:
            names[doc_id] = clean_name(content)
            spent = getattr(response, "usage", None)
            usage["calls"] += 1
            usage["prompt_tokens"] += int(getattr(spent, "prompt_tokens", 0) or 0)
            usage["completion_tokens"] += int(
                getattr(spent, "completion_tokens", 0) or 0
            )

    if hasattr(client, "complete_batch"):
        outcomes = client.complete_batch(
            [messages_for(text) for _, text in order],
            max_tokens=max_tokens,
            temperature=0.0,
            confirm=confirm,
        )
        for (doc_id, _), outcome in zip(order, outcomes, strict=True):
            record(doc_id, outcome.response)
        return names, usage

    stopped = threading.Event()

    def one(item: tuple[str, str]) -> None:
        if stopped.is_set():
            return
        doc_id, text = item
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages_for(text),
                max_tokens=max_tokens,
                temperature=0.0,
            )
        except RunStopped:
            stopped.set()
            raise
        except Exception:  # noqa: BLE001 - one failed name must not stop the run
            return
        record(doc_id, response)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, order))
    return names, usage


def clean_name(value: str) -> str:
    """Strip quotes, a trailing full stop, and line breaks from a model-written name.

    Parameters
    ----------
    value : str
        The raw completion.

    Returns
    -------
    str
        The name on one line.
    """
    line = " ".join(value.strip().split())
    quotes = "\"'“”‘’` "
    return line.strip(quotes).rstrip(".").strip(quotes)


QUERY_TEMPLATE = "In {name}: {question}"


def naming_queries(
    queries: Sequence[Query],
    names: Mapping[str, str],
    template: str = QUERY_TEMPLATE,
) -> list[Query]:
    """Return queries that name their document, for unscoped retrieval.

    Parameters
    ----------
    queries : sequence of Query
        Templated queries, each scoped to one document.
    names : mapping
        Document identifier to the name from :func:`name_documents`.
    template : str, optional
        With ``{name}`` and ``{question}``.

    Returns
    -------
    list of Query
        The same queries, identifiers and judgments unchanged, with the question
        rewritten. A query whose document has no name keeps its question.
    """
    out: list[Query] = []
    for query in queries:
        name = names.get(query.doc_id)
        question = (
            template.format(name=name, question=query.question)
            if name
            else query.question
        )
        out.append(
            Query(
                query_id=query.query_id,
                doc_id=query.doc_id,
                category=query.category,
                question=question,
                answers=query.answers,
            )
        )
    return out


_SOURCE_SEPARATORS = re.compile(r"(%20|[_\-+]|\s)+")


def clean_source_name(source_name: str) -> str:
    """Turn a file name into words: no extension, separators as spaces.

    Parameters
    ----------
    source_name : str
        A file name such as ``"01_Acme-Mutual-NDA.pdf"``.

    Returns
    -------
    str
        For example ``"01 Acme Mutual NDA"``.
    """
    stem = source_name.strip()
    stem = re.sub(r"\.(pdf|docx?|txt|html?|md)$", "", stem, flags=re.IGNORECASE)
    return _SOURCE_SEPARATORS.sub(" ", stem).strip()


def section_at(headings: Sequence[tuple[int, str]], offset: int) -> str:
    """Return the last heading at or before ``offset``, or ``""``.

    Parameters
    ----------
    headings : sequence of (int, str)
        Heading offsets and titles in document order, as
        :func:`aieng.advanced_rag.pipeline.hierarchical.detect_headings` returns.
    offset : int
        A character offset into the document.

    Returns
    -------
    str
        The governing heading.
    """
    current = ""
    for start, title in headings:
        if start > offset:
            break
        current = title
    return current


def metadata_header(
    document: str,
    chunk: Chunk,
    *,
    source_name: str = "",
    headings: Sequence[tuple[int, str]] = (),
    head_chars: int = 200,
) -> str:
    """Build a header for one chunk from document metadata, with no model call.

    The header names the file, quotes the opening of the document, which on a
    contract carries the title and usually the parties, and names the section
    heading that governs the chunk.

    Parameters
    ----------
    document : str
        The whole source document.
    chunk : Chunk
        The chunk to label.
    source_name : str, optional
        The file name the document came from.
    headings : sequence of (int, str), optional
        Detected headings, for the section label.
    head_chars : int, optional
        Characters of the document opening to quote.

    Returns
    -------
    str
        The header, one line per field that has a value.
    """
    lines = []
    name = clean_source_name(source_name) if source_name else ""
    if name:
        lines.append(f"Document: {name}")
    opening = " ".join(document[:head_chars].split())
    if opening:
        lines.append(f"Opening: {opening}")
    section = section_at(headings, chunk.start)
    if section:
        lines.append(f"Section: {section}")
    return "\n".join(lines)


def header_chunks(
    chunks: Sequence[Chunk],
    documents: Mapping[str, str],
    *,
    source_names: Mapping[str, str] | None = None,
    headings: Mapping[str, Sequence[tuple[int, str]]] | None = None,
    head_chars: int = 200,
) -> list[Chunk]:
    """Prepend :func:`metadata_header` to every chunk's indexed text.

    Parameters
    ----------
    chunks : sequence of Chunk
        The chunks to label.
    documents : mapping
        Document identifier to full text.
    source_names : mapping or None, optional
        Document identifier to file name.
    headings : mapping or None, optional
        Document identifier to its detected headings.
    head_chars : int, optional
        Characters of the document opening to quote.

    Returns
    -------
    list of Chunk
        Chunks with the same spans and text, and the header in ``embed_text``.
    """
    source_names = source_names or {}
    headings = headings or {}
    out = []
    for chunk in chunks:
        header = metadata_header(
            documents[chunk.doc_id],
            chunk,
            source_name=source_names.get(chunk.doc_id, ""),
            headings=headings.get(chunk.doc_id, ()),
            head_chars=head_chars,
        )
        out.append(
            chunk.with_embed_text(f"{header}\n\n{chunk.text}" if header else chunk.text)
        )
    return out
