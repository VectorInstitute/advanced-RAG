"""X1: retrieval arms that test HippoRAG 2's MuSiQue gain against strong local models.

Every model here runs locally: ``bge-small-en-v1.5``
(dense), ``Splade_PP_en_v1`` (learned sparse), ``answerai-colbert-small-v1``
(late interaction), and the cross-encoders ``bge-reranker-v2-m3`` and
``ms-marco-MiniLM-L-6-v2``. No arm calls a language model to rank; the reader is
RI #4's, so a difference in F1 is a difference in the five passages it reads.

The arms, all over the same 11,656 passages:

* ``hybrid3``: reciprocal rank fusion of dense, SPLADE, and ColBERT.
* ``hybrid3-rerank``: ``bge-reranker-v2-m3`` over the top 100 of ``hybrid3``.
* ``chain-rerank``: ``hybrid3-rerank``, then for each of its top two passages a
  second retrieval with the question plus that passage's text, reranked against
  the same expanded query, interleaved with the first ranking. It is a bridge
  step with no graph and no model call: a hop-two passage is usually titled by
  an entity that the hop-one passage names.
* ``union-rerank``: the reranker over ``hipporag2``'s committed top 10 together
  with the top 100 of ``hybrid3``, so the graph proposes and the cross-encoder
  decides.
* ``fusion``: reciprocal rank fusion of ``hybrid3-rerank`` and ``hipporag2``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


DENSE = "BAAI/bge-small-en-v1.5"
#: The query prefix the package's registry gives bge-small, so dense queries match RI
# 4's.
BGE_QUERY = "Represent this sentence for searching relevant passages: "
COLBERT = "answerdotai/answerai-colbert-small-v1"
RERANKER = "BAAI/bge-reranker-v2-m3"
LIGHT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FIRST_STAGE_DEPTH = 100
CHAIN_SEEDS = 2
CHAIN_DEPTH = 50
#: The arms that the reader answers from, in report order.
ANSWERED_ARMS = ("hybrid3", "hybrid3-rerank", "chain-rerank", "union-rerank", "fusion")
#: Arms whose rankings are scored for retrieval only.
RETRIEVAL_ONLY_ARMS = ("splade", "colbert", "hybrid3-rerank-minilm")


def interleave(rankings: Sequence[Sequence[str]], k: int) -> list[str]:
    """Merge ranked lists round-robin, skipping repeats, until ``k`` are chosen."""
    merged: list[str] = []
    seen: set[str] = set()
    for depth in range(max((len(r) for r in rankings), default=0)):
        for ranking in rankings:
            if depth < len(ranking) and ranking[depth] not in seen:
                seen.add(ranking[depth])
                merged.append(ranking[depth])
                if len(merged) == k:
                    return merged
    return merged


def rrf(rankings: Sequence[Sequence[str]], k: int, rank_constant: int = 60) -> list[str]:
    """Return the top ``k`` of equal-weight reciprocal rank fusion.

    The same arithmetic as ``aieng.advanced_rag.retrieval.sparse.weighted_rrf`` with
    equal
    weights, repeated here so the ranking script runs without the workspace package.
    """
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, pid in enumerate(ranking, start=1):
            scores[pid] += 1.0 / (rank_constant + rank)
    return [pid for pid, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]]


class SentenceTransformersSplade:
    """SPLADE through sentence-transformers' ``SparseEncoder``, as ``{index: weight}``.

    The same ``Splade_PP_en_v1`` checkpoint as the package's fastembed encoder, run
    through PyTorch so it can use a GPU; fastembed's ONNX export runs on a CPU only.
    """

    def __init__(self, model: str = "prithivida/Splade_PP_en_v1", device: str = "cpu", batch_size: int = 16) -> None:
        import torch  # noqa: PLC0415
        from sentence_transformers import SparseEncoder  # noqa: PLC0415

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self._model = SparseEncoder(model, device=device, model_kwargs={"torch_dtype": dtype})
        self._batch = batch_size

    @staticmethod
    def _dicts(matrix: Any) -> list[dict[int, float]]:
        coo = matrix.coalesce().cpu()
        rows, cols = coo.indices().numpy()
        vals = coo.values().float().numpy()
        out: list[dict[int, float]] = [{} for _ in range(coo.shape[0])]
        for r, c, v in zip(rows, cols, vals, strict=True):
            out[int(r)][int(c)] = float(v)
        return out

    def encode_documents(self, texts: Sequence[str]) -> list[dict[int, float]]:
        """Return one sparse vector per document text."""
        return self._dicts(self._model.encode_document(list(texts), batch_size=self._batch, convert_to_tensor=True))

    def encode_query(self, text: str) -> dict[int, float]:
        """Return the sparse vector for one query."""
        return self._dicts(self._model.encode_query([text], convert_to_tensor=True))[0]


def top_k(scores: np.ndarray, ids: Sequence[str], k: int) -> list[str]:
    """Return the ``k`` highest-scoring identifiers, ties broken on identifier order."""
    k = min(k, len(ids))
    picked = np.argpartition(-scores, k - 1)[:k]
    ordered = sorted(picked.tolist(), key=lambda i: (-float(scores[i]), i))
    return [ids[i] for i in ordered]


class SpladeIndex:
    """Global SPLADE scoring over a corpus as one sparse matrix.

    Parameters
    ----------
    texts : sequence of str
        Passage texts, in corpus order.
    cache : Path
        An ``.npz`` file holding the passage matrix, read when present.
    encoder : object, optional
        Has ``encode_documents`` and ``encode_query`` returning ``{index: weight}``;
        defaults to the package's fastembed SPLADE encoder.
    """

    def __init__(self, texts: Sequence[str], cache: Path, encoder: Any | None = None) -> None:
        from scipy import sparse  # noqa: PLC0415

        if encoder is None:
            encoder = SentenceTransformersSplade()
        self._encoder = encoder
        if cache.is_file():
            self.matrix = sparse.load_npz(cache).tocsr()
            return
        rows, cols, vals = [], [], []
        width = 0
        for row, vector in enumerate(encoder.encode_documents(list(texts))):
            for col, value in vector.items():
                rows.append(row)
                cols.append(col)
                vals.append(value)
                width = max(width, col + 1)
        self.matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(len(texts), max(width, 30522)), dtype=np.float32)
        cache.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(cache, self.matrix)

    def scores(self, query: str) -> np.ndarray:
        """Return one score per passage for the query."""
        vector = self._encoder.encode_query(query)
        q = np.zeros(self.matrix.shape[1], dtype=np.float32)
        for col, value in vector.items():
            if col < q.shape[0]:
                q[col] = value
        return np.asarray(self.matrix @ q).ravel()


class FastembedColbert:
    """Wrap a fastembed late-interaction model as the package's ColBERT encoder does.

    Queries go through ``query_embed``, which applies ColBERT's ``[MASK]`` augmentation.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def encode_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Return one token matrix per document text."""
        return [np.asarray(v, dtype=np.float32) for v in self._model.embed(list(texts), batch_size=8)]

    def encode_query(self, text: str) -> np.ndarray:
        """Return the token matrix for one query."""
        return np.asarray(next(iter(self._model.query_embed([text]))), dtype=np.float32)


class TorchColbert:
    """``answerai-colbert-small-v1`` in PyTorch, with fastembed's input handling.

    The same checkpoint as :class:`FastembedColbert`, loaded from its
    ``model.safetensors``: BERT, then a 384-to-96 projection, L2-normalised per
    token. Documents get marker token 2 after ``[CLS]`` and drop punctuation and
    padding tokens; queries get marker token 1 and are padded with ``[MASK]`` to 32
    tokens, which the encoder attends to, as in the original ColBERT. Rankings
    agree closely with fastembed's but are not bit-identical.
    """

    def __init__(self, model: str = COLBERT, device: str = "cuda", batch_size: int = 64) -> None:
        import string  # noqa: PLC0415

        import torch  # noqa: PLC0415
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        from safetensors.torch import load_file  # noqa: PLC0415
        from transformers import AutoTokenizer, BertConfig, BertModel  # noqa: PLC0415

        weights = load_file(hf_hub_download(model, "model.safetensors"))
        bert: Any = BertModel(BertConfig.from_pretrained(model), add_pooling_layer=False)  # type: ignore[no-untyped-call]
        bert.load_state_dict({k[5:]: v for k, v in weights.items() if k.startswith("bert.") and "pooler" not in k})
        self._linear = torch.nn.Linear(384, 96, bias=False)
        self._linear.weight.data = weights["linear.weight"]
        bert.to(device)
        bert.eval()
        self._bert = bert
        self._linear.to(device)
        self._tok = AutoTokenizer.from_pretrained(model)
        self._skip = {self._tok.encode(s, add_special_tokens=False)[0] for s in string.punctuation}
        self._torch = torch
        self._device = device
        self._batch = batch_size

    def _encode(self, ids: list[list[int]], attend: list[list[int]]) -> Any:
        torch = self._torch
        longest = max(len(x) for x in ids)
        input_ids = torch.tensor([x + [0] * (longest - len(x)) for x in ids], device=self._device)
        mask = torch.tensor([a + [0] * (longest - len(a)) for a in attend], device=self._device)
        with torch.no_grad():
            hidden = self._bert(input_ids=input_ids, attention_mask=mask).last_hidden_state
            vectors = torch.nn.functional.normalize(self._linear(hidden), dim=-1)
        return input_ids, mask, vectors

    def encode_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Return one token matrix per document, without punctuation or padding."""
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self._batch):
            enc = self._tok(list(texts[start : start + self._batch]), truncation=True, max_length=511)["input_ids"]
            ids = [[x[0], 2, *x[1:]] for x in enc]
            input_ids, mask, vectors = self._encode(ids, [[1] * len(x) for x in ids])
            for row in range(len(ids)):
                keep = [j for j, t in enumerate(ids[row]) if t not in self._skip]
                out.append(vectors[row, keep].float().cpu().numpy())
        return out

    def encode_query(self, text: str) -> np.ndarray:
        """Return the 32-token matrix for one query."""
        enc = self._tok(text, truncation=True, max_length=511)["input_ids"]
        ids = [enc[0], 1, *enc[1:]]
        ids += [self._tok.mask_token_id] * max(0, 32 - len(ids))
        _, _, vectors = self._encode([ids], [[1] * len(ids)])
        return np.asarray(vectors[0].float().cpu().numpy(), dtype=np.float32)


class ColbertIndex:
    """Global ColBERT MaxSim scoring, with the token matrices padded on one device.

    Parameters
    ----------
    texts : sequence of str
        Passage texts, in corpus order.
    cache : Path
        An ``.npz`` file holding the concatenated token vectors and offsets.
    encoder : object, optional
        Has ``encode_documents`` and ``encode_query`` returning token matrices;
        defaults to the package's fastembed ColBERT encoder.
    device : str, optional
        Torch device for scoring; scoring on a CPU is exact and slower.
    """

    def __init__(self, texts: Sequence[str], cache: Path, encoder: Any | None = None, device: str = "cuda") -> None:
        import torch  # noqa: PLC0415

        if encoder is None:
            from fastembed import LateInteractionTextEmbedding  # noqa: PLC0415

            encoder = FastembedColbert(LateInteractionTextEmbedding(COLBERT))
        self._encoder = encoder
        if cache.is_file():
            stored = np.load(cache)
            flat, offsets = stored["flat"], stored["offsets"]
        else:
            matrices = encoder.encode_documents(list(texts))
            offsets = np.cumsum([0] + [m.shape[0] for m in matrices])
            flat = np.concatenate(matrices).astype(np.float16)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, flat=flat, offsets=offsets)
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self._torch = torch
        self._device = device
        if device == "cpu":
            # One matrix product over the flat token matrix and a segment max per
            # passage:
            # no padding, so memory stays at the size of the token vectors.
            self._flat = np.asarray(flat, dtype=np.float32)
            self._starts = np.asarray(offsets[:-1], dtype=np.int64)
            return
        lengths = np.diff(offsets)
        longest = int(lengths.max())
        width = flat.shape[1]
        padded = np.zeros((len(lengths), longest, width), dtype=np.float16)
        mask = np.zeros((len(lengths), longest), dtype=bool)
        for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            padded[i, : end - start] = flat[start:end]
            mask[i, : end - start] = True
        self._docs = torch.from_numpy(padded).to(device=device, dtype=torch.float16)
        self._mask = torch.from_numpy(mask).to(device)

    def scores(self, query: str) -> np.ndarray:
        """Return one MaxSim score per passage for the query."""
        q = np.asarray(self._encoder.encode_query(query), dtype=np.float32)
        if self._device == "cpu":
            sims = self._flat @ q.T
            return np.maximum.reduceat(sims, self._starts, axis=0).sum(axis=1)
        torch = self._torch
        qt = torch.from_numpy(q).to(self._device, torch.float16)
        out = []
        with torch.no_grad():
            for start in range(0, self._docs.shape[0], 2048):
                docs = self._docs[start : start + 2048]
                sim = torch.einsum("qd,ntd->nqt", qt, docs)
                sim = sim.masked_fill(~self._mask[start : start + 2048, None, :], float("-inf"))
                out.append(sim.max(dim=2).values.float().sum(dim=1).cpu().numpy())
        return np.concatenate(out)


class StrongStack:
    """The local-model retrieval stack and the X1 arms built from it.

    Parameters
    ----------
    ids : sequence of str
        Passage identifiers, in corpus order.
    text_of : callable
        Returns the indexed text (title, newline, passage) for an identifier.
    dense_scores, splade_scores, colbert_scores : callable
        Each maps a query string to one score per passage.
    rerank : callable
        ``rerank(query, candidate_ids, k)`` returns the top ``k`` identifiers.
    """

    def __init__(
        self,
        ids: Sequence[str],
        text_of: Callable[[str], str],
        *,
        dense_scores: Callable[[str], np.ndarray],
        splade_scores: Callable[[str], np.ndarray],
        colbert_scores: Callable[[str], np.ndarray],
        rerank: Callable[[str, Sequence[str], int], list[str]],
    ) -> None:
        self.ids = list(ids)
        self.text_of = text_of
        self._scorers = {"dense": dense_scores, "splade": splade_scores, "colbert": colbert_scores}
        self.rerank = rerank

    def first_stage(self, query: str, depth: int = FIRST_STAGE_DEPTH) -> dict[str, list[str]]:
        """Return each base retriever's ranking and their fusion, ``depth`` deep."""
        out = {name: top_k(score(query), self.ids, depth) for name, score in self._scorers.items()}
        out["hybrid3"] = rrf([out["dense"], out["splade"], out["colbert"]], depth)
        return out

    def chain(self, question: str, seeds: Sequence[str], k: int) -> list[list[str]]:
        """Return one reranked ranking per seed passage, with its text added."""
        chains = []
        for seed in seeds:
            expanded = f"{question}\n{self.text_of(seed)}"
            candidates = [p for p in self.first_stage(expanded, CHAIN_DEPTH)["hybrid3"] if p != seed]
            chains.append(self.rerank(expanded, candidates, k))
        return chains

    def arms(self, question: str, hippo: Sequence[str], k: int = 10) -> dict[str, list[str]]:
        """Return every X1 arm's top ``k`` for one question.

        ``hippo`` is ``hipporag2``'s committed ranking for the question. The
        first-stage candidates come back under ``_candidates`` for a caller that
        reranks them another way; pop it before saving.
        """
        stage = self.first_stage(question)
        reranked = self.rerank(question, stage["hybrid3"], FIRST_STAGE_DEPTH)
        union = list(dict.fromkeys([*hippo, *stage["hybrid3"]]))
        chains = self.chain(question, reranked[:CHAIN_SEEDS], k)
        return {
            "splade": stage["splade"][:k],
            "colbert": stage["colbert"][:k],
            "hybrid3": stage["hybrid3"][:k],
            "hybrid3-rerank": reranked[:k],
            "chain-rerank": interleave([reranked, *chains], k),
            "union-rerank": self.rerank(question, union, k),
            "fusion": rrf([reranked[:k], list(hippo)[:k]], k),
            "_candidates": stage["hybrid3"],
        }


def supporting_recall(ranked: Sequence[str], gold: Sequence[str], k: int) -> tuple[bool, float]:
    """Return (all supporting passages in the top ``k``, share found)."""
    top = set(ranked[:k])
    found = sum(1 for g in gold if g in top)
    return found == len(gold), found / len(gold) if gold else 0.0


def recall_table(rankings: Mapping[str, Sequence[str]], gold: Mapping[str, Sequence[str]]) -> dict[str, float]:
    """Return all-supporting and mean supporting recall at 5 and 10."""
    out: dict[str, float] = {}
    for k in (5, 10):
        pairs = [supporting_recall(rankings[q], gold[q], k) for q in gold]
        out[f"all_supporting_recall_at_{k}"] = round(sum(a for a, _ in pairs) / len(pairs), 4)
        out[f"mean_supporting_recall_at_{k}"] = round(sum(m for _, m in pairs) / len(pairs), 4)
    return out
