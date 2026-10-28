"""Embedders, and the query prompts the models were published with.

Three things live here: the ``Embedder`` protocol every retriever accepts, a
deterministic offline embedder for tests, and a local sentence-transformers
embedder for reported numbers.

A retrieval model trained with an asymmetric prompt scores badly without it, and
the resulting number reads as a finding about the model rather than about the
harness. BGE prepends a fixed instruction to queries only. E5 uses
``query:``/``passage:``. Nomic uses ``search_query:``/``search_document:``.
Qwen3 wraps the query in an ``Instruct:``/``Query:`` block and leaves documents
bare. Those strings live in ``MODELS`` so a run applies the prompt its model was
trained with by default, and so a deliberately prefix-free arm has to say so.

Pooling is not set here. Each repository ships its own ``modules.json`` and
sentence-transformers builds the right pooling head from it: mean for the BERT
encoders, last-token for Qwen3. ``SentenceTransformerEmbedder.pooling`` reads
back what was actually constructed, so a report records it rather than
asserting it.

Embeddings run locally. Nothing here calls a hosted embedding API.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens.

    Shared with ``sparse`` so the hashed embedder and the BM25 encoder split
    text the same way, and a difference between them cannot be tokenisation.

    Parameters
    ----------
    text : str
        The text to split.

    Returns
    -------
    list of str
        Tokens in order.
    """
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into fixed-width vectors.

    ``encode`` is the document side. A retriever calls ``encode_queries`` for
    the query side when the embedder defines it, so an asymmetric model applies
    its query prompt without the retriever knowing which model it holds.
    """

    name: str

    def dimension(self) -> int:
        """Return the vector width."""
        ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text, as documents."""
        ...


# Qwen3-Embedding is trained to condition on a task description. The model card
# uses a one-line retrieval instruction; this is that line, kept verbatim so the
# arm reproduces the published configuration.
QWEN_TASK = "Given a web search query, retrieve relevant passages that answer the query"

BGE_QUERY = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class ModelSpec:
    """One encoder and the configuration it needs to perform as published.

    Parameters
    ----------
    hf_id : str
        Hugging Face model identifier.
    label : str
        Human-readable name for a report.
    params_millions : int
        Parameter count, for the size-against-quality reading.
    query_prompt : str, optional
        The prompt the model's card prepends to a query. Empty when the model
        is symmetric.
    document_prompt : str, optional
        The prompt prepended to a passage. Empty for most models.
    trust_remote_code : bool, optional
        Whether the repository ships custom modelling code. Worth knowing
        before it runs in front of participants.
    config_kwargs : tuple of tuple, optional
        Passed to the model's config. Only needed by repositories whose custom
        modelling code defaults to a kernel that is not installed here.
    matryoshka_dims : tuple of int, optional
        Dimensions the model supports truncating to.
    max_seq_length : int or None, optional
        Override for the encoder's window.
    note : str, optional
        Why this model is in the registry.
    """

    hf_id: str
    label: str
    params_millions: int
    query_prompt: str = ""
    document_prompt: str = ""
    trust_remote_code: bool = False
    config_kwargs: tuple[tuple[str, object], ...] = ()
    matryoshka_dims: tuple[int, ...] = ()
    max_seq_length: int | None = None
    note: str = ""

    @property
    def prompt_note(self) -> str:
        """Return a human-readable record of the prompting, for a report."""
        if not self.query_prompt and not self.document_prompt:
            return "none (raw text both sides)"
        return f"query={self.query_prompt!r} document={self.document_prompt!r}"


# Keyed by Hugging Face identifier, and each entry carries the prompts that
# model was published with. An arm that deliberately runs prefix-free passes
# ``query_prompt=""`` to the embedder, so a missing prompt is a stated choice.
MODELS: dict[str, ModelSpec] = {
    "BAAI/bge-small-en-v1.5": ModelSpec(
        hf_id="BAAI/bge-small-en-v1.5",
        label="bge-small-en-v1.5",
        params_millions=33,
        query_prompt=BGE_QUERY,
        note="the bootcamp default: 384 dimensions, comfortable on CPU",
    ),
    "BAAI/bge-base-en-v1.5": ModelSpec(
        hf_id="BAAI/bge-base-en-v1.5",
        label="bge-base-en-v1.5",
        params_millions=109,
        query_prompt=BGE_QUERY,
        note="same family, one size up: separates newer from bigger",
    ),
    "BAAI/bge-large-en-v1.5": ModelSpec(
        hf_id="BAAI/bge-large-en-v1.5",
        label="bge-large-en-v1.5",
        params_millions=335,
        query_prompt=BGE_QUERY,
        note="the top of the same family, for the size curve",
    ),
    "Qwen/Qwen3-Embedding-0.6B": ModelSpec(
        hf_id="Qwen/Qwen3-Embedding-0.6B",
        label="Qwen3-Embedding-0.6B",
        params_millions=596,
        query_prompt=f"Instruct: {QWEN_TASK}\nQuery:",
        note="last-token pooling, 1024-d, 32K context; the quality ceiling on CPU",
    ),
    "ibm-granite/granite-embedding-english-r2": ModelSpec(
        hf_id="ibm-granite/granite-embedding-english-r2",
        label="granite-embedding-english-r2",
        params_millions=149,
        note="2025 encoder, Apache-2.0, symmetric (no prompt published)",
    ),
    "Snowflake/snowflake-arctic-embed-m-v2.0": ModelSpec(
        hf_id="Snowflake/snowflake-arctic-embed-m-v2.0",
        label="snowflake-arctic-embed-m-v2.0",
        params_millions=305,
        query_prompt="query: ",
        trust_remote_code=True,
        # The repository's modelling code asks for xformers' memory-efficient
        # attention by default, which is not installed and would not help on
        # CPU. Turning it off falls back to plain attention; the weights and the
        # pooling are unchanged.
        config_kwargs=(
            ("use_memory_efficient_attention", False),
            ("unpad_inputs", False),
        ),
        matryoshka_dims=(768, 512, 256, 128),
        note="supports dimension truncation",
    ),
    "nomic-ai/nomic-embed-text-v1.5": ModelSpec(
        hf_id="nomic-ai/nomic-embed-text-v1.5",
        label="nomic-embed-text-v1.5",
        params_millions=137,
        query_prompt="search_query: ",
        document_prompt="search_document: ",
        trust_remote_code=True,
        matryoshka_dims=(768, 512, 256, 128, 64),
        note="trained with Matryoshka loss",
    ),
    "sentence-transformers/all-MiniLM-L6-v2": ModelSpec(
        hf_id="sentence-transformers/all-MiniLM-L6-v2",
        label="all-MiniLM-L6-v2",
        params_millions=22,
        note="the 2021 default, as a lower reference point",
    ),
    # Gated on Hugging Face and dropped from the project's embedding comparison,
    # which never measured it. Kept in the registry so a run fails with the access
    # error rather than "unknown model", and so a participant with access can run
    # it with the prompts below.
    "google/embeddinggemma-300m": ModelSpec(
        hf_id="google/embeddinggemma-300m",
        label="embeddinggemma-300m",
        params_millions=308,
        query_prompt="task: search result | query: ",
        document_prompt="title: none | text: ",
        matryoshka_dims=(768, 512, 256, 128),
        note="gated on Hugging Face and not measured here; access is granted per account",
    ),
}


def spec_for(hf_id: str) -> ModelSpec:
    """Return the registry entry for a model.

    Parameters
    ----------
    hf_id : str
        Hugging Face model identifier.

    Returns
    -------
    ModelSpec
        The entry.

    Raises
    ------
    KeyError
        If the model is not in the registry, naming the ones that are.
    """
    try:
        return MODELS[hf_id]
    except KeyError:
        known = ", ".join(sorted(MODELS))
        raise KeyError(f"unknown model {hf_id!r}; known: {known}") from None


def apply_prompt(prompt: str, texts: Sequence[str]) -> list[str]:
    r"""Prepend a prompt, inserting the separating space the model expects.

    Model cards write these prefixes two ways: with the trailing space baked in
    (``"query: "``) or without it (Qwen3's ``"...\nQuery:"``). Normalising here
    means a spec can be copied from a model card verbatim and still encode the
    same string the card describes.

    Parameters
    ----------
    prompt : str
        The prefix. An empty string returns a copy of ``texts``.
    texts : sequence of str
        The texts to prefix.

    Returns
    -------
    list of str
        Prefixed texts.
    """
    if not prompt:
        return list(texts)
    joiner = "" if prompt.endswith((" ", "\n", "\t")) else " "
    return [f"{prompt}{joiner}{text}" for text in texts]


def _normalize(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values] if norm else [0.0] * len(values)


class HashEmbedder:
    """Deterministic hashed bag-of-words vectors.

    Offline, dependency-free, and stable across runs, which is what the tests
    need. It carries real lexical signal, so a test asserting that a working
    retriever beats a shuffled one is a genuine assertion rather than a
    tautology. It is not a serious retriever and is not used for reported
    numbers.

    Parameters
    ----------
    dimension : int, optional
        Vector width.
    """

    name = "hash-384"

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    def dimension(self) -> int:
        """Return the vector width."""
        return self._dimension

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one L2-normalised vector per text."""
        vectors = []
        for text in texts:
            counts = Counter(tokenize(text))
            vector = [0.0] * self._dimension
            for token, count in counts.items():
                digest = hashlib.blake2b(token.encode(), digest_size=4).digest()
                index = int.from_bytes(digest, "big") % self._dimension
                vector[index] += 1.0 + math.log(count)
            vectors.append(_normalize(vector))
        return vectors

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Return query vectors. Symmetric, so the same as ``encode``."""
        return self.encode(texts)


class SentenceTransformerEmbedder:
    """A local CPU embedding model. The default for reported numbers.

    Queries are encoded with the model's published query prompt and documents
    without it, which is what an asymmetric retrieval model was trained for.
    ``BAAI/bge-small-en-v1.5`` runs comfortably on a CPU container and needs no
    API key, so a baseline measured with it reproduces anywhere.

    Parameters
    ----------
    model_name : str, optional
        Hugging Face model identifier.
    device : str, optional
        Torch device. CPU is the platform the bootcamp targets.
    query_prompt : str or None, optional
        ``None`` applies the model's published query prompt from ``MODELS``,
        and no prompt for a model that is not in the registry. Pass ``""`` for
        a deliberately prefix-free arm.
    document_prompt : str or None, optional
        The same rule for the document side.
    truncate_dim : int or None, optional
        Matryoshka truncation, for models that support it.
    batch_size : int, optional
        Texts per forward pass.
    trust_remote_code : bool or None, optional
        ``None`` follows the registry entry. Some repositories need it, which
        is a supply-chain decision as much as a quality one.
    max_seq_length : int or None, optional
        Override for the encoder's window. ``None`` follows the registry.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        *,
        query_prompt: str | None = None,
        document_prompt: str | None = None,
        truncate_dim: int | None = None,
        batch_size: int = 32,
        trust_remote_code: bool | None = None,
        max_seq_length: int | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        spec = MODELS.get(model_name)
        self.name = model_name
        self.spec = spec
        self.query_prompt = (
            query_prompt
            if query_prompt is not None
            else (spec.query_prompt if spec else "")
        )
        self.document_prompt = (
            document_prompt
            if document_prompt is not None
            else (spec.document_prompt if spec else "")
        )
        self.batch_size = batch_size
        self.truncate_dim = truncate_dim
        self.encode_seconds = 0.0
        self.encoded_texts = 0

        remote = (
            trust_remote_code
            if trust_remote_code is not None
            else bool(spec and spec.trust_remote_code)
        )
        config_kwargs = (
            dict(spec.config_kwargs) if spec and spec.config_kwargs else None
        )
        self._model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=remote,
            truncate_dim=truncate_dim,
            config_kwargs=config_kwargs,
        )
        window = max_seq_length or (spec.max_seq_length if spec else None)
        if window:
            self._model.max_seq_length = window

        # Renamed in sentence-transformers 6.x; keep working on both.
        probe = getattr(
            self._model,
            "get_embedding_dimension",
            getattr(self._model, "get_sentence_embedding_dimension", None),
        )
        if probe is None:
            raise AttributeError(f"{model_name} exposes no dimension accessor")
        self._dimension = truncate_dim or int(probe())
        limit = self._model.max_seq_length
        if limit is None:
            # sentence-transformers types the window as `int | None`. A model
            # that reports none leaves token-budget chunking with nothing to
            # chunk to, and `int(None)` said so with a bare TypeError.
            raise ValueError(
                f"{model_name} reports no max_seq_length, so there is no token "
                "window to chunk to: pass max_seq_length explicitly"
            )
        self.max_seq_length = int(limit)
        self.pooling = _pooling_mode(self._model)

    @property
    def model(self) -> Any:
        """Return the underlying sentence-transformers model."""
        return self._model

    def dimension(self) -> int:
        """Return the vector width."""
        return self._dimension

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return document vectors, L2-normalised.

        Normalising here means cosine similarity is a plain dot product
        downstream, which is what every retriever in this package assumes.
        """
        return self._encode(texts, self.document_prompt)

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Return query vectors, with the model's query prompt applied."""
        return self._encode(texts, self.query_prompt)

    def _encode(self, texts: Sequence[str], prompt: str) -> list[list[float]]:
        prepared = apply_prompt(prompt, texts)
        started = time.perf_counter()
        vectors = self._model.encode(
            prepared,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.encode_seconds += time.perf_counter() - started
        self.encoded_texts += len(prepared)
        return [list(map(float, v)) for v in vectors]

    def describe(self) -> dict[str, object]:
        """Return the facts about this encoder that a report has to carry."""
        return {
            "hf_id": self.name,
            "label": self.spec.label if self.spec else self.name,
            "params_millions": self.spec.params_millions if self.spec else None,
            "dimension": self._dimension,
            "pooling": self.pooling,
            "max_seq_length": self.max_seq_length,
            "truncate_dim": self.truncate_dim,
            "query_prompt": self.query_prompt,
            "document_prompt": self.document_prompt,
            "note": self.spec.note if self.spec else "",
        }


def _pooling_mode(model: Any) -> str:
    """Read back the pooling head sentence-transformers actually built.

    Asserting the pooling in the spec would be a claim; reading it back from the
    constructed module is a measurement, and the two differ often enough (BGE is
    CLS, MiniLM is mean, Qwen3 is last-token) to be worth reporting.
    """
    for module in model.modules():
        if type(module).__name__ != "Pooling":
            continue
        mode = getattr(module, "pooling_mode", None)
        if mode:
            return str(mode)
        # sentence-transformers 3.x exposed one boolean per mode instead.
        modes = [
            attribute.removeprefix("pooling_mode_")
            for attribute in dir(module)
            if attribute.startswith("pooling_mode_")
            and getattr(module, attribute) is True
        ]
        return "+".join(modes) or "unknown"
    return "module-defined (no Pooling head)"


class CachedQueryEmbedder:
    """Wraps an embedder and remembers query vectors.

    CUAD's 41 questions repeat verbatim for every contract, so a sweep that
    re-encodes each query per contract spends most of its time on 41 distinct
    strings. Document text is never cached: it differs by chunking strategy,
    which is usually the thing being measured.

    Parameters
    ----------
    inner : Embedder
        The embedder to wrap.
    """

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.name = inner.name
        self._cache: dict[str, list[float]] = {}

    def dimension(self) -> int:
        """Return the wrapped embedder's vector width."""
        return self.inner.dimension()

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return document vectors from the wrapped embedder, uncached."""
        return self.inner.encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Return query vectors, encoding only the ones not already seen."""
        encode = getattr(self.inner, "encode_queries", self.inner.encode)
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for text, vector in zip(missing, encode(missing)):
                self._cache[text] = vector
        return [self._cache[t] for t in texts]
