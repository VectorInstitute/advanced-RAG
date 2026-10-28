"""Late chunking: embed the document, then pool per chunk.

Naive chunk-wise embedding encodes each chunk in isolation, so a chunk that says
"the Company shall indemnify the Provider" carries no trace of which company or
which agreement. Late chunking runs the encoder over the whole document, then
mean-pools the token vectors belonging to each chunk. The boundaries are the
same; the vectors see the surrounding document.

Two constraints shape the implementation.

Context windows are finite. CUAD contracts run to 250,000 characters, well past
any encoder's window, so the document is processed in overlapping macro windows
and each chunk is pooled from the window that contains it with the most room on
either side. A chunk therefore sees the document up to the window size.

The encoder is not the bootcamp's encoder. Late chunking needs a long-context
model, and the default here is ``bge-small`` at 512 tokens. Reporting late
chunking on a different model against a baseline on ``bge-small`` would measure
the model swap, so ``embed_texts`` runs the same long-context model chunk by
chunk and the late-chunking arm is scored against that instead.
``Alibaba-NLP/gte-base-en-v1.5`` is the model this was measured with, at 137M
parameters and an 8,192-token window. ``jinaai/jina-embeddings-v3`` is what the
published result uses and it does not load here: its remote code fails to
resolve a relative import under current transformers.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .corpus import Chunk
from .tokens import OffsetTokenIndex


# Hidden states arrive in whatever float width the model runs in. The pooled
# vectors are cast to float32, which is what the dense index stores.
_FloatArray = NDArray[np.floating[Any]]


@dataclass
class LongContextEncoder:
    """A transformers encoder used for token-level output, not just pooling.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier for a long-context encoder.
    device : str, optional
        Torch device.
    max_tokens : int, optional
        Macro-window size, capped to the model's own position limit.
    trust_remote_code : bool, optional
        Whether the repository ships custom modelling code.
    """

    model_name: str
    device: str = "cpu"
    max_tokens: int = 8192
    trust_remote_code: bool = True

    def __post_init__(self) -> None:
        """Load the tokenizer and model, and cap the window to what it supports."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval().to(self.device)
        window = getattr(self.model.config, "max_position_embeddings", self.max_tokens)
        self.max_tokens = min(self.max_tokens, int(window) - 2)
        self.forward_tokens = 0
        self.forward_seconds = 0.0

    def _hidden(self, input_ids: Any, attention_mask: Any) -> Any:
        # Explicit position ids rather than the model's own default. Some
        # long-context encoders keep their position-id buffer non-persistent,
        # and current transformers leaves such a buffer uninitialised after a
        # meta-device load, so the model indexes its rotary table with garbage.
        # Passing the ids is cheaper than pinning an older transformers.
        positions = self.torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions.unsqueeze(0).expand(input_ids.shape[0], -1)
        began = time.perf_counter()
        with self.torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=positions,
            )
        self.forward_seconds += time.perf_counter() - began
        self.forward_tokens += int(attention_mask.sum())
        states = getattr(out, "last_hidden_state", None)
        if states is None:
            states = out[0]
        return states

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
        prefix: str = "",
    ) -> NDArray[np.float32]:
        """Return mean-pooled embeddings, one text at a time: the naive arm.

        Parameters
        ----------
        texts : sequence of str
            The chunk texts.
        batch_size : int, optional
            Texts per forward pass.
        prefix : str, optional
            The instruction some encoders are trained to expect, such as
            ``"search_document: "``. Leaving it off costs quality on those
            models, so it is applied to both arms rather than to neither.

        Returns
        -------
        numpy.ndarray
            L2-normalised float32 vectors, one row per text.
        """
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[start : start + batch_size]]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            ).to(self.device)
            states = self._hidden(encoded["input_ids"], encoded["attention_mask"])
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vectors.append(pooled.cpu().numpy())
        matrix = np.concatenate(vectors, axis=0).astype(np.float32)
        return _normalize_rows(matrix)

    def embed_late(
        self,
        text: str,
        chunks: Sequence[Chunk],
        *,
        window_overlap_tokens: int = 256,
        prefix: str = "",
    ) -> NDArray[np.float32]:
        """Embed one document, then mean-pool token vectors per chunk.

        Parameters
        ----------
        text : str
            The whole document.
        chunks : sequence of Chunk
            Chunks of that document, with offsets into it.
        window_overlap_tokens : int, optional
            Tokens each macro window repeats from the one before it.
        prefix : str, optional
            Prepended once to the document. Chunk offsets are shifted to match,
            so a chunk still pools exactly its own tokens.

        Returns
        -------
        numpy.ndarray
            L2-normalised float32 vectors, one row per chunk.
        """
        shift = len(prefix)
        text = prefix + text
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        offsets = [(int(s), int(e)) for s, e in encoded["offset_mapping"] if e > s]
        ids = [
            i
            for i, (s, e) in zip(encoded["input_ids"], encoded["offset_mapping"])
            if e > s
        ]
        index = OffsetTokenIndex(offsets)

        stride = max(self.max_tokens - window_overlap_tokens, 1)
        windows = list(range(0, max(len(ids), 1), stride))
        states_by_window: dict[int, _FloatArray] = {}
        for window_start in windows:
            window_ids = ids[window_start : window_start + self.max_tokens]
            if not window_ids:
                continue
            tensor = self.torch.tensor([window_ids], device=self.device)
            mask = self.torch.ones_like(tensor)
            states = self._hidden(tensor, mask)
            states_by_window[window_start] = states[0].cpu().numpy()

        vectors = np.zeros((len(chunks), _dim(states_by_window)), dtype=np.float32)
        for row, chunk in enumerate(chunks):
            low, high = index.token_span(chunk.start + shift, chunk.end + shift)
            high = max(high, low + 1)
            window_start = _best_window(low, high, windows, self.max_tokens)
            states = states_by_window.get(window_start)
            if states is None:
                continue
            local_low = max(low - window_start, 0)
            local_high = min(high - window_start, states.shape[0])
            if local_high <= local_low:
                local_low, local_high = 0, states.shape[0]
            vectors[row] = states[local_low:local_high].mean(axis=0)
        return _normalize_rows(vectors)


def _dim(states_by_window: dict[int, _FloatArray]) -> int:
    for states in states_by_window.values():
        return int(states.shape[1])
    return 1


def _best_window(low: int, high: int, windows: Sequence[int], size: int) -> int:
    """Return the window holding the chunk with the most context on both sides.

    A chunk that straddles a window edge would be pooled from a truncated span,
    so windows that do not contain it whole are rejected first. Among the rest,
    the one whose centre is nearest the chunk's centre gives the most
    surrounding document.
    """
    centre = (low + high) / 2
    containing = [w for w in windows if w <= low and high <= w + size]
    candidates = containing or list(windows)
    return min(candidates, key=lambda w: abs((w + size / 2) - centre))


def _normalize_rows(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Annotated so the return is an ndarray to mypy: numpy's operators are typed
    # to return Any, which `warn_return_any` rejects.
    normalised: NDArray[np.float32] = matrix / np.clip(norms, 1e-9, None)
    return normalised
