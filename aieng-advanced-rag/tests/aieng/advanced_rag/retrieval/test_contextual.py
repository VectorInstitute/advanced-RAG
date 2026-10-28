"""Contextual retrieval: the prompt, the cost record, and failure handling.

No model is called. What matters here is that the situating text lands in
``embed_text`` while the span stays the unit of judgment, that a failed call
loses the context rather than the chunk, and that the reported cost is the cost
of a cold run even when the client served the answer from its cache.
"""

from __future__ import annotations

from dataclasses import dataclass

from aieng.advanced_rag.retrieval.chunkers import fixed_chars
from aieng.advanced_rag.retrieval.contextual import (
    ContextualCost,
    build_prompt,
    collect_usage,
    contextualise,
)


DOCUMENT = "ACME DISTRIBUTION AGREEMENT. " + "The parties agree as follows. " * 200


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class StubClient:
    """A client that situates every chunk and counts what it billed."""

    def __init__(self, fail: bool = False) -> None:
        self.usage = Usage()
        self.fail = fail
        self.prompts: list[str] = []

    def complete(self, prompt, system=None, max_tokens=None, kind=None):
        if self.fail:
            raise RuntimeError("the model is unavailable")
        self.prompts.append(prompt)
        self.usage.calls += 1
        self.usage.prompt_tokens += 100
        self.usage.completion_tokens += 20
        return "  From the Acme distribution agreement, section 2.  "


class TestBuildPrompt:
    def test_the_prompt_carries_the_head_the_neighbourhood_and_the_chunk(self):
        chunk = fixed_chars("acme", DOCUMENT, chunk_chars=200, overlap_chars=40)[6]
        prompt = build_prompt(
            DOCUMENT, chunk, document_head_chars=60, neighbourhood_chars=100
        )
        assert DOCUMENT[:60] in prompt
        assert chunk.text in prompt
        # The head is what tells the model which agreement this is.
        assert "ACME DISTRIBUTION AGREEMENT" in prompt

    def test_the_neighbourhood_is_clipped_to_the_document(self):
        chunk = fixed_chars("acme", DOCUMENT, chunk_chars=200, overlap_chars=40)[0]
        prompt = build_prompt(DOCUMENT, chunk, neighbourhood_chars=10**6)
        assert prompt.count(DOCUMENT) >= 1


class TestContextualise:
    def _chunks(self):
        return fixed_chars("acme", DOCUMENT, chunk_chars=400, overlap_chars=80)

    def test_the_context_is_prepended_to_the_indexed_text_only(self):
        chunks = self._chunks()
        client = StubClient()
        out, cost = contextualise(
            {"acme": DOCUMENT}, chunks, lambda: client, workers=2, progress_every=0
        )
        assert len(out) == len(chunks)
        for original, contextualised in zip(chunks, out):
            assert contextualised.indexed_text.startswith("From the Acme")
            assert contextualised.indexed_text.endswith(original.text)
            # The span is the unit of judgment and must not move.
            assert (contextualised.start, contextualised.end) == original.span
            assert contextualised.text == original.text
        assert cost.chunks == len(chunks)
        assert cost.failures == 0

    def test_a_failed_call_keeps_the_chunk_uncontextualised(self):
        chunks = self._chunks()
        client = StubClient(fail=True)
        out, cost = contextualise(
            {"acme": DOCUMENT}, chunks, lambda: client, workers=2, progress_every=0
        )
        assert [c.embed_text for c in out] == [None] * len(chunks)
        assert cost.failures == len(chunks)

    def test_the_cost_is_the_cold_cost(self):
        chunks = self._chunks()
        client = StubClient()
        _, cost = contextualise(
            {"acme": DOCUMENT}, chunks, lambda: client, workers=1, progress_every=0
        )
        assert cost.calls == len(chunks)
        assert cost.total_tokens == 120 * len(chunks)
        assert cost.to_dict()["total_tokens_per_chunk"] == 120.0


class TestCollectUsage:
    def test_it_folds_every_thread_s_client_together(self):
        cost = ContextualCost(chunks=10)
        clients = [StubClient(), StubClient()]
        for client in clients:
            client.usage.calls = 3
            client.usage.prompt_tokens = 300
            client.usage.completion_tokens = 60
        collect_usage(clients, cost)
        assert cost.calls == 6
        assert cost.total_tokens == 720

    def test_chunks_the_client_did_not_bill_for_are_cache_hits(self):
        # A re-run costs nothing, so the difference between the chunk count and
        # the billed calls is what the cache served.
        cost = ContextualCost(chunks=10)
        client = StubClient()
        client.usage.calls = 4
        collect_usage([client], cost)
        assert cost.cached_calls == 6

    def test_a_client_with_no_usage_counter_is_skipped(self):
        cost = ContextualCost(chunks=2)
        collect_usage([object()], cost)
        assert cost.calls == 0
        assert cost.cached_calls == 2
