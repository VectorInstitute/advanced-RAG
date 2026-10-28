"""The embedder protocol, the offline embedder, and the model prompt registry.

The prompt is the part of the encoder side most likely to be silently wrong: a
model run without the prefix it was trained with produces a plausible number
that misreads as a finding about the model. These tests pin the strings and the
side each one is applied to, without loading a model.
"""

from __future__ import annotations

import pytest

from aieng.advanced_rag.retrieval.embedders import (
    BGE_QUERY,
    MODELS,
    CachedQueryEmbedder,
    Embedder,
    HashEmbedder,
    apply_prompt,
    spec_for,
    tokenize,
)


class TestTokenize:
    def test_lowercases_and_drops_punctuation(self):
        assert tokenize("Acme Corp., Inc. (2019)") == ["acme", "corp", "inc", "2019"]

    def test_empty_text_has_no_tokens(self):
        assert tokenize("   ") == []


class TestRegistry:
    def test_every_key_matches_its_spec(self):
        for hf_id, spec in MODELS.items():
            assert spec.hf_id == hf_id

    def test_the_default_encoder_carries_its_published_query_prompt(self):
        # The fix this registry exists for: bge-small-en-v1.5 has a published
        # query prompt and no measurement in this repo had used it.
        spec = spec_for("BAAI/bge-small-en-v1.5")
        assert spec.query_prompt == BGE_QUERY
        assert spec.document_prompt == ""

    def test_qwen_uses_the_instruct_query_format_and_bare_documents(self):
        spec = spec_for("Qwen/Qwen3-Embedding-0.6B")
        assert spec.query_prompt.startswith("Instruct: ")
        assert spec.query_prompt.endswith("\nQuery:")
        assert spec.document_prompt == ""

    def test_asymmetric_models_prompt_both_sides(self):
        for hf_id in ("nomic-ai/nomic-embed-text-v1.5", "google/embeddinggemma-300m"):
            spec = spec_for(hf_id)
            assert spec.query_prompt and spec.document_prompt
            assert spec.query_prompt != spec.document_prompt

    def test_a_symmetric_model_says_so(self):
        spec = spec_for("sentence-transformers/all-MiniLM-L6-v2")
        assert spec.prompt_note == "none (raw text both sides)"

    def test_unknown_model_names_the_known_ones(self):
        with pytest.raises(KeyError, match="bge-small"):
            spec_for("not-a-model")


class TestApplyPrompt:
    def test_inserts_a_space_only_when_needed(self):
        assert apply_prompt("query: ", ["a"]) == ["query: a"]
        assert apply_prompt("query:", ["a"]) == ["query: a"]
        assert apply_prompt("Instruct: t\nQuery:", ["a"]) == ["Instruct: t\nQuery: a"]

    def test_an_empty_prompt_copies_the_texts(self):
        texts = ["one", "two"]
        assert apply_prompt("", texts) == texts
        assert apply_prompt("", texts) is not texts


class TestHashEmbedder:
    def test_satisfies_the_embedder_protocol(self):
        assert isinstance(HashEmbedder(dimension=16), Embedder)

    def test_deterministic(self):
        embedder = HashEmbedder(dimension=64)
        assert embedder.encode(["hello world"]) == embedder.encode(["hello world"])

    def test_vectors_are_normalized(self):
        [vector] = HashEmbedder(dimension=64).encode(["governing law delaware"])
        assert abs(sum(v * v for v in vector) - 1.0) < 1e-9

    def test_shared_vocabulary_scores_higher_than_disjoint(self):
        embedder = HashEmbedder(dimension=256)
        query, near, far = embedder.encode(
            [
                "governing law delaware",
                "the governing law is delaware",
                "shipping pallets freight",
            ]
        )
        assert sum(a * b for a, b in zip(query, near)) > sum(
            a * b for a, b in zip(query, far)
        )

    def test_empty_text_gives_a_zero_vector(self):
        [vector] = HashEmbedder(dimension=16).encode([""])
        assert vector == [0.0] * 16

    def test_the_query_side_is_symmetric(self):
        # No prompt to apply, so the two sides have to agree exactly, or the
        # offline control would stop reproducing the dense arm.
        embedder = HashEmbedder(dimension=32)
        assert embedder.encode_queries(["a b"]) == embedder.encode(["a b"])


class CountingEmbedder:
    """Counts how many texts reach each side."""

    name = "counting"

    def __init__(self) -> None:
        self.documents: list[str] = []
        self.queries: list[str] = []

    def dimension(self) -> int:
        return 2

    def encode(self, texts):
        self.documents.extend(texts)
        return [[float(len(t)), 0.0] for t in texts]

    def encode_queries(self, texts):
        self.queries.extend(texts)
        return [[0.0, float(len(t))] for t in texts]


class TestCachedQueryEmbedder:
    def test_a_repeated_query_is_encoded_once(self):
        # CUAD's questions repeat verbatim for every contract, which is the
        # whole reason this wrapper exists.
        inner = CountingEmbedder()
        cached = CachedQueryEmbedder(inner)
        cached.encode_queries(["same", "same", "other"])
        cached.encode_queries(["same"])
        assert inner.queries == ["same", "other"]

    def test_the_vectors_are_the_wrapped_ones(self):
        inner = CountingEmbedder()
        cached = CachedQueryEmbedder(inner)
        assert cached.encode_queries(["abc"]) == [[0.0, 3.0]]
        assert cached.encode_queries(["abc", "de"]) == [[0.0, 3.0], [0.0, 2.0]]

    def test_documents_are_never_cached(self):
        # Chunk text differs by strategy, which is usually what is measured.
        inner = CountingEmbedder()
        cached = CachedQueryEmbedder(inner)
        cached.encode(["doc"])
        cached.encode(["doc"])
        assert inner.documents == ["doc", "doc"]

    def test_it_forwards_the_name_and_dimension(self):
        cached = CachedQueryEmbedder(CountingEmbedder())
        assert cached.name == "counting"
        assert cached.dimension() == 2
