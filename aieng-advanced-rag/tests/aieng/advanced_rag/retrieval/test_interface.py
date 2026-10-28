"""Tests for the retrieval interface's own types, apart from any retriever.

The conformance kit covers what an implementation has to honour. What is here is
the boundary's own guarantees: an advisory record that cannot be filled in after
it is built, and capability checks that answer for the capability rather than for
a method name an unrelated object happens to carry.
"""

from __future__ import annotations

import pytest

from aieng.advanced_rag.retrieval.corpus import Chunk
from aieng.advanced_rag.retrieval.interface import (
    CAPABILITIES,
    ChunkStore,
    Expandable,
    Fetchable,
    Findable,
    Greppable,
    Hints,
    Indexable,
    Readable,
    Retriever,
    RetrieverBase,
    capabilities_of,
    declares,
    is_retriever,
)
from aieng.advanced_rag.retrieval.retrievers import (
    ChunkStoreRetriever,
    DenseRetriever,
    ShuffledRetriever,
)


def make_chunks() -> list[Chunk]:
    """Return two chunks of one document, with the spans a read walks."""
    text = "Clause one.\nClause two, on its own line."
    split = text.index("\n") + 1
    return [
        Chunk(
            chunk_id="C1::0000",
            doc_id="C1",
            text=text[:split],
            start=0,
            end=split,
        ),
        Chunk(
            chunk_id="C1::0001",
            doc_id="C1",
            text=text[split:],
            start=split,
            end=len(text),
        ),
    ]


class TestHints:
    def test_an_empty_hint_cannot_be_filled_in_after_it_is_built(self):
        hints = Hints()
        with pytest.raises(TypeError):
            hints.extra["mode"] = "local"  # type: ignore[index]
        assert hints.is_empty()

    def test_a_callers_own_dict_cannot_change_a_built_hint(self):
        extra = {"mode": "local"}
        hints = Hints(extra=extra)
        extra["mode"] = "global"
        assert hints.extra["mode"] == "local"

    def test_a_hint_is_hashable_so_a_request_can_key_a_cache(self):
        assert hash(Hints()) == hash(Hints())
        assert hash(Hints(keywords=("notice",))) != hash(Hints())
        assert len({Hints(), Hints(extra={"mode": "local"})}) == 2

    def test_hints_that_hold_the_same_steering_are_equal(self):
        assert Hints(extra={"mode": "local"}) == Hints(extra={"mode": "local"})
        assert Hints(keywords=("notice",)) != Hints(keywords=("termination",))

    def test_is_empty_reports_on_every_field(self):
        assert Hints().is_empty()
        assert not Hints(keywords=("notice",)).is_empty()
        assert not Hints(entities=("Acme",)).is_empty()
        assert not Hints(categories=("governing-law",)).is_empty()
        assert not Hints(extra={"mode": "local"}).is_empty()


class TestCapabilityChecks:
    def test_a_list_declares_nothing_although_it_has_an_index_method(self):
        assert declares([], "index") is False
        assert capabilities_of([]) == ()

    def test_a_string_declares_nothing_although_it_has_find_and_index(self):
        assert declares("abc", "find") is False
        assert declares("abc", "index") is False

    def test_an_object_carrying_the_method_name_alone_declares_nothing(self):
        class Cursor:
            def fetch(self, chunk_ids):
                return []

        assert hasattr(Cursor(), "fetch")
        assert declares(Cursor(), "fetch") is False

    def test_a_harness_retriever_declares_every_capability_it_defines(self):
        assert capabilities_of(ShuffledRetriever()) == CAPABILITIES

    def test_a_retriever_that_defines_none_of_them_declares_none(self):
        class Bare:
            name = "bare"

            def retrieve(self, question, k=10, *, scope=None, hints=None):
                return []

        assert is_retriever(Bare())
        assert capabilities_of(Bare()) == ()

    def test_read_needs_both_of_its_methods(self):
        class HalfRead:
            name = "half-read"

            def retrieve(self, question, k=10, *, scope=None, hints=None):
                return []

            def read(self, doc_id, *, start=0, end=None):
                return None

        assert declares(HalfRead(), "read") is False

    def test_a_capability_protocol_no_longer_answers_isinstance(self):
        # It used to, and `isinstance([], SupportsIndex)` was True because a
        # list has an `index` method. The name shadowed `typing.SupportsIndex`,
        # which means "converts losslessly to an int", as well.
        with pytest.raises(TypeError, match="runtime_checkable"):
            isinstance([], Indexable)

    def test_the_supports_names_still_import_as_the_adjectives(self):
        from aieng.advanced_rag.retrieval import interface

        assert interface.SupportsIndex is Indexable
        assert interface.SupportsFind is Findable
        assert interface.SupportsFetch is Fetchable
        assert interface.SupportsExpand is Expandable
        assert interface.SupportsRead is Readable
        assert interface.SupportsGrep is Greppable

    def test_an_unknown_capability_is_refused_rather_than_answered(self):
        with pytest.raises(ValueError, match="unknown capability"):
            declares(DenseRetriever, "rerank")


class TestReadBounds:
    def test_the_two_bound_pairs_cannot_be_passed_positionally(self):
        store = ChunkStore(make_chunks(), retriever="test")
        with pytest.raises(TypeError):
            store.read("C1", 0, 40)  # type: ignore[misc]
        with pytest.raises(TypeError):
            store.read_lines("C1", 1, 1)  # type: ignore[misc]

    def test_each_pair_still_reads_its_own_range_by_keyword(self):
        store = ChunkStore(make_chunks(), retriever="test")
        assert store.read("C1", start=0, end=6).text == "Clause"
        assert store.read_lines("C1", first=1, last=1).text.startswith("Clause")


class TestOneRetrieverProtocol:
    def test_the_retired_names_still_import_as_the_one_interface(self):
        from aieng.advanced_rag.graphrag.protocol import GraphRetriever
        from aieng.advanced_rag.retrieval.interface import Retrieval

        assert Retrieval is Retriever
        assert GraphRetriever is Retriever

    def test_a_harness_retriever_satisfies_it(self):
        assert isinstance(ShuffledRetriever(), Retriever)

    def test_something_that_only_ranks_identifiers_does_not(self):
        class Ranking:
            name = "ranking"

            def search(self, query, k=10):
                return []

        assert not isinstance(Ranking(), Retriever)


class TestRetrieverBase:
    def test_the_old_mixin_name_still_subclasses(self):
        from aieng.advanced_rag.retrieval.interface import RetrievalMixin

        assert RetrievalMixin is RetrieverBase

    def test_a_base_with_no_retrieve_names_itself_in_the_error(self):
        class Unfinished(RetrieverBase):
            name = "unfinished"

        with pytest.raises(NotImplementedError, match=r"Unfinished.*retrieve"):
            Unfinished().retrieve("who signs")

    def test_a_store_retriever_with_no_name_says_which_attribute_is_missing(self):
        class Nameless(ChunkStoreRetriever):
            pass

        with pytest.raises(AttributeError, match=r"Nameless.*name"):
            Nameless()
