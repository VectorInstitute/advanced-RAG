"""Exact token counting over character spans, cheaply.

Recursive token splitting asks "how many tokens are in ``text[start:end]``"
thousands of times per document. Re-tokenising each candidate substring makes
the splitter cost more than the embedding step. Tokenising the document once,
keeping the offset mapping, and answering every later question with a binary
search gives the same counts for a fraction of the work.

Counts come from the tokenizer of the model that does the embedding, so "512
tokens" means 512 tokens as that encoder sees them. That matters: bge-small
truncates at 512, so a 512-token chunk exactly fills its window and anything
longer is silently cut.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from typing import Any, Protocol


class TokenCounter(Protocol):
    """Counts tokens in a character span of one fixed document."""

    def count(self, start: int, end: int) -> int:
        """Return the number of tokens overlapping ``[start, end)``."""
        ...


class OffsetTokenIndex:
    """Token offsets for one document, queried by character span.

    Parameters
    ----------
    offsets : sequence of tuple of int
        ``(start, end)`` per token, as a fast tokenizer's offset mapping
        returns them.
    """

    def __init__(self, offsets: Sequence[tuple[int, int]]) -> None:
        # Special tokens map to (0, 0); dropping them keeps the counts about
        # content rather than about the encoder's framing.
        kept = [(s, e) for s, e in offsets if e > s]
        self._starts = [s for s, _ in kept]
        self._ends = [e for _, e in kept]

    def __len__(self) -> int:
        """Return the number of content tokens in the document."""
        return len(self._starts)

    def count(self, start: int, end: int) -> int:
        """Return the tokens that overlap the half-open span ``[start, end)``."""
        if end <= start:
            return 0
        # Tokens are non-overlapping and sorted, so the overlapping run is
        # bounded by the first token ending after ``start`` and the last one
        # starting before ``end``.
        low = bisect_right(self._ends, start)
        high = bisect_left(self._starts, end)
        return max(high - low, 0)

    def token_span(self, start: int, end: int) -> tuple[int, int]:
        """Return the index range of the tokens overlapping ``[start, end)``."""
        low = bisect_right(self._ends, start)
        high = bisect_left(self._starts, end)
        return low, max(high, low)


class WhitespaceTokenIndex:
    """A tokenizer-free counter for tests and for smoke runs.

    Counts whitespace-delimited words. It is not the encoder's tokenizer and is
    never used for reported numbers.

    Parameters
    ----------
    text : str
        The document to count over.
    """

    def __init__(self, text: str) -> None:
        offsets: list[tuple[int, int]] = []
        position = 0
        for word in text.split(" "):
            if word.strip():
                offsets.append((position, position + len(word)))
            position += len(word) + 1
        self._index = OffsetTokenIndex(offsets)

    def __len__(self) -> int:
        """Return the number of words in the document."""
        return len(self._index)

    def count(self, start: int, end: int) -> int:
        """Return the words that overlap the half-open span ``[start, end)``."""
        return self._index.count(start, end)

    def token_span(self, start: int, end: int) -> tuple[int, int]:
        """Return the index range of the words overlapping ``[start, end)``."""
        return self._index.token_span(start, end)


def build_token_index(text: str, tokenizer: Any) -> OffsetTokenIndex:
    """Tokenise a whole document once and keep the character offsets.

    Parameters
    ----------
    text : str
        The document.
    tokenizer : Any
        A Hugging Face fast tokenizer, from ``load_tokenizer``.

    Returns
    -------
    OffsetTokenIndex
        A counter over that document.
    """
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )
    return OffsetTokenIndex(encoded["offset_mapping"])


def load_tokenizer(model_name: str = "BAAI/bge-small-en-v1.5") -> Any:
    """Load the fast tokenizer for an embedding model.

    Parameters
    ----------
    model_name : str, optional
        A Hugging Face model identifier. Defaults to the bootcamp's encoder, so
        a token budget means the same thing here and at index time.

    Returns
    -------
    Any
        The tokenizer.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, use_fast=True)
