"""The vector store behind every dense arm, and its Qdrant backend.

:class:`VectorStore` is the seam: add vectors under string keys with an optional
document, then search with an optional document scope. :class:`QdrantStore` is
the one implementation, and it runs in two modes.

In-process mode (``":memory:"`` or a directory path) is the default. It needs no
server, scores by an exact numpy scan, and reproduces the in-memory rankings the
published figures were measured with. It has no HNSW graph and no
quantization, so :class:`QdrantSettings` that ask for either are refused there
rather than silently ignored, which is what qdrant-client itself does.

Server mode (an ``http://`` URL, or a ``QdrantClient`` built against one) is for
quantization, oversampling and rescoring, and HNSW. ``docs/vector-store.md``
gives the measured trade-offs of each setting.

Ties are resolved on the client. Qdrant returns equal scores in no promised
order, and qdrant-client's in-process scan returns them in reverse insertion
order, while every committed figure breaks ties on chunk identifier ascending.
:meth:`QdrantStore.search` therefore over-fetches, widens the fetch until no
tied point can sit outside it, and sorts on ``(-score, key)``.

A collection ships through the artifact cache as plain files, written by
:func:`export_collection` and read back by :func:`load_collection`, and is
rebuilt in whichever store loads it. Qdrant's own storage directory is never
shipped: a local-mode directory is a SQLite file of pickled points tied to the
client version, and a server cannot read it.

This module needs the ``qdrant`` extra. Nothing in ``retrieval/__init__``
imports it, so the package still imports without the extra.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray


if TYPE_CHECKING:
    from qdrant_client import QdrantClient

    from ..cache.artifacts import CacheKey, TieredResultCache


#: The default in-process location: an exact scan held in memory.
MEMORY = ":memory:"

#: Environment variable naming a Qdrant server, read only when asked for.
URL_ENV = "QDRANT_URL"

#: Artifact type for a collection shipped through the cache.
ARTIFACT_TYPE = "vector-collection"

#: Version of the files :func:`export_collection` writes.
EXPORT_FORMAT = 1

Quantization = Literal["binary", "turbo"]

# Inputs take any float width, because the store casts to float32 itself.
_FloatArray = NDArray[np.floating[Any]]

_TURBO_BITS = {1.0: "BITS1", 1.5: "BITS1_5", 2.0: "BITS2", 4.0: "BITS4"}
_BINARY_BITS = {1.0: "ONE_BIT", 1.5: "ONE_AND_HALF_BITS", 2.0: "TWO_BITS"}

# Extra points fetched beyond k before the tie check. Sixteen covers the
# boilerplate ties on the CUAD slice in one round trip; a wider tie triggers
# another fetch rather than a wrong answer.
_TIE_MARGIN = 16
_UPSERT_BATCH = 256


class VectorStore(Protocol):
    """Where a dense arm keeps its vectors and how it searches them.

    Keys are the caller's identifiers, normally chunk identifiers. Scores are
    comparable within one search. Among equal scores, results are ordered by
    key ascending.
    """

    def __len__(self) -> int:
        """Return how many vectors the store holds."""
        ...

    def add(
        self,
        keys: Sequence[str],
        vectors: Sequence[Sequence[float]] | _FloatArray,
        doc_ids: Sequence[str] | None = None,
    ) -> None:
        """Add vectors under ``keys``, each optionally tagged with a document."""
        ...

    def search(
        self, vector: Sequence[float] | _FloatArray, k: int, *, scope: str | None = None
    ) -> list[tuple[str, float]]:
        """Return the ``k`` best ``(key, score)`` pairs, within ``scope`` if given."""
        ...


@dataclass(frozen=True)
class QdrantSettings:
    """Server-side search settings, each off unless set.

    The defaults give an exact search over the float32 vectors, which is what
    in-process mode does and what every committed figure was measured with.
    Every other setting needs a server.

    Parameters
    ----------
    quantization : {"binary", "turbo"} or None, optional
        Store a compressed copy of each vector and score against it. ``None``
        scores the float32 vectors.
    bits : float, optional
        Bits per dimension. TurboQuant takes 1, 1.5, 2, or 4; binary takes 1,
        1.5, or 2.
    asymmetric : bool, optional
        Binary only: encode the query as 8-bit scalars while the stored vectors
        stay binary.
    oversampling : float or None, optional
        Fetch ``oversampling * k`` candidates by the compressed score before
        rescoring. Without rescoring it cannot change the top ``k``.
    rescore : bool or None, optional
        Re-rank the oversampled candidates with the float32 vectors. ``None``
        leaves the server's default, which on Qdrant 1.19 is on for binary and
        for TurboQuant below 4 bits.
    hnsw_ef : int or None, optional
        Search the HNSW graph with this beam width. ``None`` keeps an exact
        scan.
    hnsw_m : int or None, optional
        Graph degree at build time. ``None`` keeps the server's default.
    ef_construct : int or None, optional
        Beam width at build time. ``None`` keeps the server's default.
    indexing_threshold_kb : int or None, optional
        Segment size above which the optimizer builds the graph and the
        quantized copy. ``None`` sets 1 KB whenever quantization or HNSW is on,
        because the server's 20 MB default applies per segment and a corpus of
        this size never crosses it.
    full_scan_threshold_kb : int or None, optional
        Segment size below which a search scans exactly whatever its graph says.
        ``None`` sets 10 KB whenever ``hnsw_ef`` is set, for the same reason.
    """

    quantization: Quantization | None = None
    bits: float = 1.0
    asymmetric: bool = False
    oversampling: float | None = None
    rescore: bool | None = None
    hnsw_ef: int | None = None
    hnsw_m: int | None = None
    ef_construct: int | None = None
    indexing_threshold_kb: int | None = None
    full_scan_threshold_kb: int | None = None

    def __post_init__(self) -> None:
        """Reject a bit width the chosen quantizer does not offer."""
        allowed = {"turbo": _TURBO_BITS, "binary": _BINARY_BITS}
        if self.quantization is not None:
            if self.quantization not in allowed:
                raise ValueError(
                    f"quantization must be 'binary', 'turbo', or None, "
                    f"not {self.quantization!r}"
                )
            if float(self.bits) not in allowed[self.quantization]:
                widths = ", ".join(f"{b:g}" for b in allowed[self.quantization])
                raise ValueError(
                    f"{self.quantization} quantization takes {widths} bits, "
                    f"not {self.bits:g}"
                )
        if self.asymmetric and self.quantization != "binary":
            raise ValueError("asymmetric applies to binary quantization only")
        if self.oversampling is not None and self.oversampling < 1.0:
            raise ValueError(
                f"oversampling must be at least 1, not {self.oversampling}"
            )
        if (
            self.oversampling is not None or self.rescore is not None
        ) and self.quantization is None:
            raise ValueError("oversampling and rescore need a quantization setting")

    @property
    def needs_server(self) -> bool:
        """Return whether any setting here does nothing in in-process mode."""
        return (
            self.quantization is not None
            or self.hnsw_ef is not None
            or self.hnsw_m is not None
            or self.ef_construct is not None
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as JSON-ready values."""
        return asdict(self)

    def collection_kwargs(self) -> dict[str, Any]:
        """Return the ``create_collection`` arguments these settings imply."""
        from qdrant_client import models

        kwargs: dict[str, Any] = {}
        if self.quantization == "turbo":
            kwargs["quantization_config"] = models.TurboQuantization(
                turbo=models.TurboQuantQuantizationConfig(
                    bits=models.TurboQuantBitSize[_TURBO_BITS[float(self.bits)]],
                    always_ram=True,
                )
            )
        elif self.quantization == "binary":
            kwargs["quantization_config"] = models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    encoding=models.BinaryQuantizationEncoding[
                        _BINARY_BITS[float(self.bits)]
                    ],
                    query_encoding=(
                        models.BinaryQuantizationQueryEncoding.SCALAR8BITS
                        if self.asymmetric
                        else None
                    ),
                    always_ram=True,
                )
            )
        if not self.needs_server:
            return kwargs
        indexing = self.indexing_threshold_kb
        kwargs["optimizers_config"] = models.OptimizersConfigDiff(
            indexing_threshold=1 if indexing is None else indexing
        )
        full_scan = self.full_scan_threshold_kb
        if full_scan is None and self.hnsw_ef is not None:
            full_scan = 10
        if full_scan is not None or self.hnsw_m is not None or self.ef_construct:
            kwargs["hnsw_config"] = models.HnswConfigDiff(
                m=self.hnsw_m,
                ef_construct=self.ef_construct,
                full_scan_threshold=full_scan,
            )
        return kwargs

    def search_params(self) -> Any:
        """Return the query-time ``SearchParams`` these settings imply."""
        from qdrant_client import models

        quantization = None
        if self.quantization is not None and (
            self.oversampling is not None or self.rescore is not None
        ):
            quantization = models.QuantizationSearchParams(
                rescore=self.rescore, oversampling=self.oversampling
            )
        # Exact only when nothing compressed or approximate was asked for. With
        # quantization alone the server picks its own path, which is how the
        # compression sweep that these settings come from was measured.
        exact = True if not self.needs_server else None
        return models.SearchParams(
            hnsw_ef=self.hnsw_ef, exact=exact, quantization=quantization
        )


def is_local(client: QdrantClient) -> bool:
    """Return whether a client runs in-process rather than against a server.

    Parameters
    ----------
    client : QdrantClient
        The client to inspect.

    Returns
    -------
    bool
        ``True`` for ``":memory:"`` and directory-path clients.
    """
    options = client.init_options
    return options.get("location") == MEMORY or options.get("path") is not None


def connect(location: str | None = MEMORY) -> QdrantClient:
    """Return a client for an in-process location or a server URL.

    Parameters
    ----------
    location : str or None, optional
        ``":memory:"``, a directory path, or an ``http://`` or ``https://`` URL.
        ``None`` reads the ``QDRANT_URL`` environment variable and falls back to
        ``":memory:"``.

    Returns
    -------
    QdrantClient
        The client. A server URL is not contacted until the first request.

    Raises
    ------
    ImportError
        If the ``qdrant`` extra is not installed.
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "the Qdrant vector store needs the qdrant extra: "
            "pip install 'aieng-advanced-rag[qdrant]'"
        ) from error
    if location is None:
        location = os.environ.get(URL_ENV) or MEMORY
    if location == MEMORY:
        return QdrantClient(location=MEMORY)
    if location.startswith(("http://", "https://")):
        return QdrantClient(url=location)
    return QdrantClient(path=location)


class QdrantStore:
    """One Qdrant collection of dense vectors, searched exactly by default.

    Point identifiers are integers in insertion order, and the store keeps the
    key of each one, so a ranking never needs the payload. The payload still
    carries ``key`` and ``doc_id``, which is what a document scope filters on
    and what someone browsing the collection on a server sees.

    Parameters
    ----------
    client : QdrantClient, str, or None, optional
        A client, or a location for :func:`connect`. ``None`` reads
        ``QDRANT_URL`` and otherwise runs in memory.
    collection : str, optional
        Collection name. An existing collection of that name is replaced on the
        first :meth:`add`, so two stores on one server need two names.
    settings : QdrantSettings or None, optional
        Quantization and HNSW settings. Anything beyond the defaults needs a
        server.

    Raises
    ------
    ValueError
        If ``settings`` ask for quantization or HNSW on an in-process client.
    """

    def __init__(
        self,
        client: QdrantClient | str | None = MEMORY,
        *,
        collection: str = "chunks",
        settings: QdrantSettings | None = None,
    ) -> None:
        self.client: QdrantClient = (
            connect(client) if client is None or isinstance(client, str) else client
        )
        self.collection = collection
        self.settings = settings or QdrantSettings()
        self.local = is_local(self.client)
        if self.local and self.settings.needs_server:
            raise ValueError(
                "quantization and HNSW settings need a Qdrant server: in-process "
                "mode scans exactly and qdrant-client ignores them without a "
                "word. Pass a server URL, or drop the settings."
            )
        self._keys: list[str] = []
        self._doc_ids: list[str | None] = []
        self._dimension: int | None = None

    def __len__(self) -> int:
        """Return how many vectors the store holds."""
        return len(self._keys)

    @property
    def dimension(self) -> int | None:
        """Return the vector width, or ``None`` before the first :meth:`add`."""
        return self._dimension

    @property
    def keys(self) -> list[str]:
        """Return the keys in point-identifier order."""
        return list(self._keys)

    @property
    def doc_ids(self) -> list[str | None]:
        """Return each point's document, in point-identifier order."""
        return list(self._doc_ids)

    def _create(self, dimension: int) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        # Dot product rather than cosine: every embedder here normalises its
        # vectors, and cosine would normalise them a second time in float32,
        # which moves scores in the last bit and can reorder a near tie.
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.DOT
            ),
            **self.settings.collection_kwargs(),
        )
        if not self.local:
            # In-process mode has no payload indexes and warns if asked.
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        self._dimension = dimension

    def add(
        self,
        keys: Sequence[str],
        vectors: Sequence[Sequence[float]] | _FloatArray,
        doc_ids: Sequence[str] | None = None,
    ) -> None:
        """Add vectors under ``keys``.

        Parameters
        ----------
        keys : sequence of str
            One identifier per vector. A key already in the store is an error,
            because two points with one key would make a ranking ambiguous.
        vectors : sequence or numpy.ndarray
            One vector per key, already normalised.
        doc_ids : sequence of str or None, optional
            The document each vector belongs to, which ``scope`` filters on.

        Raises
        ------
        ValueError
            If the counts disagree, a key repeats, or the width changes.
        """
        from qdrant_client import models

        matrix = np.asarray(vectors, dtype=np.float32)
        if len(keys) == 0:
            return
        if matrix.ndim != 2 or matrix.shape[0] != len(keys):
            raise ValueError(
                f"{len(keys)} keys against vectors of shape {matrix.shape}"
            )
        if doc_ids is not None and len(doc_ids) != len(keys):
            raise ValueError(f"{len(keys)} keys against {len(doc_ids)} document ids")
        known = set(self._keys)
        repeated = [key for key in keys if key in known]
        if repeated or len(set(keys)) != len(keys):
            raise ValueError(f"keys already in the store or repeated: {repeated[:3]}")
        if self._dimension is None:
            self._create(int(matrix.shape[1]))
        elif matrix.shape[1] != self._dimension:
            raise ValueError(
                f"vectors of width {matrix.shape[1]} added to a store of width "
                f"{self._dimension}"
            )
        first = len(self._keys)
        documents: list[str | None] = (
            [str(d) for d in doc_ids] if doc_ids is not None else [None] * len(keys)
        )
        for start in range(0, len(keys), _UPSERT_BATCH):
            stop = min(start + _UPSERT_BATCH, len(keys))
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=first + row,
                        vector=matrix[row].tolist(),
                        payload={"key": keys[row], "doc_id": documents[row]},
                    )
                    for row in range(start, stop)
                ],
                wait=True,
            )
        self._keys.extend(str(key) for key in keys)
        self._doc_ids.extend(documents)
        if not self.local and self.settings.needs_server:
            self.wait_until_indexed()

    def wait_until_indexed(self, timeout: float = 300.0) -> None:
        """Block until the server has built the graph and the quantized copy.

        A search against a half-optimized collection scores some segments
        compressed and others in float32, which gives plausible wrong numbers.

        Parameters
        ----------
        timeout : float, optional
            Seconds to wait.

        Raises
        ------
        TimeoutError
            If the collection is not indexed in time.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            info = self.client.get_collection(self.collection)
            indexed = info.indexed_vectors_count or 0
            settled = str(info.status).lower().endswith("green")
            if settled and indexed >= len(self._keys):
                return
            time.sleep(0.25)
        raise TimeoutError(
            f"collection {self.collection!r} was not indexed within {timeout:g} s"
        )

    def _request(
        self,
        vector: NDArray[np.float32],
        limit: int,
        scope: str | None,
        within: Sequence[int] | None = None,
    ) -> Any:
        from qdrant_client import models

        return models.QueryRequest(
            query=vector.tolist(),
            limit=limit,
            filter=_scope_filter(scope, within),
            params=None if self.local else self.settings.search_params(),
            with_payload=False,
        )

    def _complete(
        self,
        points: Sequence[Any],
        vector: NDArray[np.float32],
        k: int,
        limit: int,
        scope: str | None,
        within: Sequence[int] | None,
    ) -> list[tuple[str, float]]:
        """Sort one response on ``(-score, key)``, widening it past any tie."""
        while True:
            hits = sorted(
                ((self._keys[int(p.id)], float(p.score)) for p in points),
                key=lambda hit: (-hit[1], hit[0]),
            )
            exhausted = len(points) < limit
            if exhausted or len(hits) < k or hits[k - 1][1] > hits[-1][1]:
                return hits[:k]
            # The weakest point fetched ties the k-th, so an equal score may sit
            # beyond the fetch and belong in the top k on key order.
            limit *= 4
            points = self.client.query_points(
                collection_name=self.collection,
                **_as_kwargs(self._request(vector, limit, scope, within)),
            ).points

    def search(
        self,
        vector: Sequence[float] | _FloatArray,
        k: int,
        *,
        scope: str | None = None,
        within: Sequence[int] | None = None,
    ) -> list[tuple[str, float]]:
        """Return the ``k`` best ``(key, score)`` pairs, ties on key ascending.

        Parameters
        ----------
        vector : sequence of float or numpy.ndarray
            One normalised query vector.
        k : int
            Depth.
        scope : str or None, optional
            One document, or ``None`` for the whole collection.
        within : sequence of int or None, optional
            Point identifiers to restrict the search to, on top of ``scope``.
            An empty sequence matches nothing.

        Returns
        -------
        list of tuple
            ``(key, score)`` in rank order.
        """
        return self.search_batch([vector], k, scopes=[scope], within=within)[0]

    def search_batch(
        self,
        vectors: Sequence[Sequence[float] | _FloatArray] | _FloatArray,
        k: int,
        *,
        scopes: Sequence[str | None] | None = None,
        within: Sequence[int] | None = None,
    ) -> list[list[tuple[str, float]]]:
        """Search several query vectors in one round trip.

        Parameters
        ----------
        vectors : sequence or numpy.ndarray
            One normalised query vector per row.
        k : int
            Depth.
        scopes : sequence or None, optional
            One document per query, ``None`` entries searching everything.
        within : sequence of int or None, optional
            Point identifiers every query is restricted to.

        Returns
        -------
        list of list of tuple
            One ranking per query.
        """
        queries = np.asarray(vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        wanted = list(scopes) if scopes is not None else [None] * len(queries)
        if (
            k <= 0
            or not self._keys
            or len(queries) == 0
            or (within is not None and len(within) == 0)
        ):
            return [[] for _ in range(len(queries))]
        limit = k + _TIE_MARGIN
        responses = self.client.query_batch_points(
            collection_name=self.collection,
            requests=[
                self._request(vector, limit, scope, within)
                for vector, scope in zip(queries, wanted)
            ],
        )
        return [
            self._complete(response.points, vector, k, limit, scope, within)
            for response, vector, scope in zip(responses, queries, wanted)
        ]

    def vectors(self) -> NDArray[np.float32]:
        """Read every stored vector back, in point-identifier order.

        Returns
        -------
        numpy.ndarray
            A float32 matrix with one row per key.
        """
        if not self._keys:
            return np.zeros((0, self._dimension or 0), dtype=np.float32)
        records = self.client.retrieve(
            collection_name=self.collection,
            ids=list(range(len(self._keys))),
            with_vectors=True,
            with_payload=False,
        )
        rows = {int(r.id): r.vector for r in records}
        return np.asarray([rows[i] for i in range(len(self._keys))], dtype=np.float32)


def _scope_filter(scope: str | None, within: Sequence[int] | None = None) -> Any:
    if scope is None and within is None:
        return None
    from qdrant_client import models

    conditions: list[Any] = []
    if scope is not None:
        conditions.append(
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=scope))
        )
    if within is not None:
        conditions.append(models.HasIdCondition(has_id=[int(i) for i in within]))
    return models.Filter(must=conditions)


def _as_kwargs(request: Any) -> dict[str, Any]:
    return {
        "query": request.query,
        "limit": request.limit,
        "query_filter": request.filter,
        "search_params": request.params,
        "with_payload": False,
    }


def export_collection(
    store: QdrantStore,
    directory: Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    payloads: Iterable[Mapping[str, Any]] | None = None,
) -> Path:
    """Write a store's vectors and payloads as portable files.

    The directory holds ``vectors.npy`` (float32, row i is point i),
    ``points.jsonl`` (one payload per point, in the same order), and
    ``collection.json`` (the width, the distance, the format version, the
    qdrant-client version, and ``metadata``). Any store can rebuild the
    collection from these, in memory, on disk, or on a server.

    Parameters
    ----------
    store : QdrantStore
        The store to export.
    directory : Path
        Where to write. Created if missing; must be empty if it exists.
    metadata : mapping or None, optional
        What a reader needs to know the vectors are the ones it wants: the
        embedding model, the query prompt, the chunker, and a corpus digest.
    payloads : iterable of mapping or None, optional
        Extra payload per point, merged over ``key`` and ``doc_id``, such as the
        chunk offsets.

    Returns
    -------
    Path
        ``directory``.

    Raises
    ------
    FileExistsError
        If ``directory`` exists and is not empty.
    """
    from importlib.metadata import version

    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"{directory} is not empty")
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "vectors.npy", store.vectors(), allow_pickle=False)
    extra = list(payloads) if payloads is not None else [{}] * len(store)
    if len(extra) != len(store):
        raise ValueError(f"{len(extra)} payloads for {len(store)} points")
    with (directory / "points.jsonl").open("w", encoding="utf-8") as handle:
        for point, (key, doc_id, more) in enumerate(
            zip(store.keys, store.doc_ids, extra)
        ):
            record = {"id": point, "key": key, "doc_id": doc_id, **dict(more)}
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    description = {
        "format": EXPORT_FORMAT,
        "points": len(store),
        "dimension": store.dimension,
        "distance": "dot",
        "qdrant_client": version("qdrant-client"),
        "metadata": dict(metadata or {}),
    }
    (directory / "collection.json").write_text(
        json.dumps(description, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory


@dataclass(frozen=True)
class LoadedCollection:
    """What :func:`load_collection` read besides the vectors.

    Parameters
    ----------
    metadata : dict
        The ``metadata`` the export recorded.
    payloads : list of dict
        One payload per point, in point order.
    """

    metadata: dict[str, Any]
    payloads: list[dict[str, Any]]


def load_collection(directory: Path, store: QdrantStore) -> LoadedCollection:
    """Rebuild an exported collection into an empty store.

    Parameters
    ----------
    directory : Path
        A directory :func:`export_collection` wrote.
    store : QdrantStore
        An empty store, in any mode.

    Returns
    -------
    LoadedCollection
        The recorded metadata and payloads.

    Raises
    ------
    ValueError
        If the store already holds points, or the files disagree with each
        other.
    """
    directory = Path(directory)
    if len(store):
        raise ValueError("load_collection needs an empty store")
    description = json.loads((directory / "collection.json").read_text("utf-8"))
    if description.get("format") != EXPORT_FORMAT:
        raise ValueError(
            f"{directory} has export format {description.get('format')!r}; "
            f"this package reads format {EXPORT_FORMAT}"
        )
    matrix = np.load(directory / "vectors.npy", allow_pickle=False)
    with (directory / "points.jsonl").open(encoding="utf-8") as handle:
        payloads = [json.loads(line) for line in handle if line.strip()]
    if len(payloads) != matrix.shape[0] or description["points"] != len(payloads):
        raise ValueError(
            f"{directory}: {matrix.shape[0]} vectors, {len(payloads)} payloads, "
            f"{description['points']} points recorded"
        )
    if [p["id"] for p in payloads] != list(range(len(payloads))):
        raise ValueError(f"{directory}: points.jsonl is not in point order")
    doc_ids = [p.get("doc_id") for p in payloads]
    store.add(
        [str(p["key"]) for p in payloads],
        matrix,
        None if all(d is None for d in doc_ids) else [str(d) for d in doc_ids],
    )
    return LoadedCollection(metadata=dict(description["metadata"]), payloads=payloads)


def cache_collection(
    cache: TieredResultCache,
    key: CacheKey,
    store: QdrantStore,
    *,
    metadata: Mapping[str, Any] | None = None,
    payloads: Iterable[Mapping[str, Any]] | None = None,
) -> Any:
    """Export a store and keep it in the artifact cache as one archive.

    Parameters
    ----------
    cache : TieredResultCache
        The cache to write to.
    key : CacheKey
        Its ``artifact_type`` should be ``"vector-collection"``, and its
        ``config`` should name what changes the vectors (corpus digest, chunker,
        embedding model, and prompt) and never the store mode, so one artifact
        loads into memory, a path, or a server.
    store : QdrantStore
        The store to export.
    metadata, payloads : optional
        Passed to :func:`export_collection`.

    Returns
    -------
    ArtifactResult
        The stored artifact.
    """
    import tempfile

    from ..cache.directories import store_directory

    with tempfile.TemporaryDirectory() as scratch:
        exported = export_collection(
            store, Path(scratch) / "collection", metadata=metadata, payloads=payloads
        )
        return store_directory(cache, key, exported)


def fetch_collection(
    cache: TieredResultCache, key: CacheKey, store: QdrantStore
) -> LoadedCollection | None:
    """Load a cached collection into ``store``, or return ``None`` on a miss.

    Parameters
    ----------
    cache : TieredResultCache
        The cache to read.
    key : CacheKey
        The key :func:`cache_collection` stored under.
    store : QdrantStore
        An empty store, in any mode.

    Returns
    -------
    LoadedCollection or None
        What was loaded, or ``None`` when the cache holds no such artifact.
    """
    import tempfile

    from ..cache.directories import fetch_directory

    with tempfile.TemporaryDirectory() as scratch:
        found = fetch_directory(cache, key, Path(scratch) / "collection")
        if found is None:
            return None
        return load_collection(found, store)


__all__ = [
    "ARTIFACT_TYPE",
    "EXPORT_FORMAT",
    "MEMORY",
    "URL_ENV",
    "LoadedCollection",
    "QdrantSettings",
    "QdrantStore",
    "VectorStore",
    "cache_collection",
    "connect",
    "export_collection",
    "fetch_collection",
    "is_local",
    "load_collection",
]
