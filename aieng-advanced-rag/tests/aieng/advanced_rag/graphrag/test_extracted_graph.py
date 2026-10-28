"""Tests for entity-linked retrieval over an extracted graph."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from aieng.advanced_rag.graphrag.adapters.extracted_graph import (
    ExtractedGraphRetriever,
    _tokens,
)


@dataclass
class FakeEntity:
    name: str
    chunks: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)


@dataclass
class FakeRelation:
    source: str
    target: str
    type: str = "depends_on"
    chunks: list[str] = field(default_factory=list)


@pytest.fixture
def retriever() -> ExtractedGraphRetriever:
    entities = [
        FakeEntity("nginx", ["d1::0"]),
        FakeEntity("nginx worker_connections", ["d1::1"]),
        FakeEntity("systemd", ["d2::0"]),
        FakeEntity("selinux", ["d3::0"]),
    ]
    relations = [
        FakeRelation("nginx", "systemd"),
        FakeRelation("systemd", "selinux"),
    ]
    texts = {
        "d1::0": "nginx basics",
        "d1::1": "tuning",
        "d2::0": "units",
        "d3::0": "contexts",
    }
    documents = {"d1::0": "d1", "d1::1": "d1", "d2::0": "d2", "d3::0": "d3"}
    return ExtractedGraphRetriever(entities, relations, texts, documents)


def test_tokenizer_keeps_paths_and_underscored_names():
    found = _tokens("set worker_connections in /etc/nginx.conf")
    # The leading slash is dropped, which is harmless: entity names and queries
    # go through the same tokenizer, so "/etc/nginx.conf" in an entity name and
    # in a query both become "etc/nginx.conf" and still link.
    assert "worker_connections" in found and "etc/nginx.conf" in found


def test_a_query_naming_no_entity_returns_nothing(retriever):
    # Deliberate: no dense fallback, so an empty graph result stays visible
    # rather than being answered by a chunk index wearing the graph's name.
    assert retriever.retrieve("how do I bake bread") == []


def test_linking_prefers_the_more_specific_entity(retriever):
    linked = retriever.link("tuning nginx worker_connections")
    assert linked[0] == "nginx worker_connections"


def test_one_hop_expansion_reaches_a_neighbour_the_query_never_named(retriever):
    hits = {evidence.doc_id for evidence in retriever.retrieve("nginx", k=10)}
    assert "d2" in hits  # systemd, reached through the relation


def test_expansion_does_not_reach_two_hops(retriever):
    hits = {evidence.doc_id for evidence in retriever.retrieve("nginx", k=10)}
    assert "d3" not in hits  # selinux is two hops from nginx


def test_seeds_outrank_neighbours(retriever):
    hits = retriever.retrieve("nginx", k=10)
    seed_score = next(e.score for e in hits if e.chunk_id == "d1::0")
    neighbour_score = next(e.score for e in hits if e.chunk_id == "d2::0")
    assert seed_score > neighbour_score


def test_zero_hops_reduces_to_entity_lookup():
    entities = [FakeEntity("nginx", ["d1::0"]), FakeEntity("systemd", ["d2::0"])]
    relations = [FakeRelation("nginx", "systemd")]
    retriever = ExtractedGraphRetriever(
        entities, relations, {"d1::0": "a", "d2::0": "b"}, expand_hops=0
    )
    assert [e.chunk_id for e in retriever.retrieve("nginx", k=5)] == ["d1::0"]


def test_evidence_carries_provenance_and_the_linked_entities(retriever):
    evidence = retriever.retrieve("nginx", k=1)[0]
    assert evidence.provenance.chunk_id == "d1::0"
    assert evidence.provenance.doc_id == "d1"
    assert "nginx" in evidence.extra["linked_entities"]
    assert evidence.retriever == "extracted graph"


def test_relations_to_unknown_entities_are_ignored():
    entities = [FakeEntity("nginx", ["d1::0"])]
    relations = [FakeRelation("nginx", "a_ghost")]
    retriever = ExtractedGraphRetriever(entities, relations, {"d1::0": "a"})
    assert [e.chunk_id for e in retriever.retrieve("nginx", k=5)] == ["d1::0"]


def test_document_id_falls_back_to_the_chunk_id_prefix():
    retriever = ExtractedGraphRetriever(
        [FakeEntity("nginx", ["d9::4"])], [], {"d9::4": "text"}
    )
    assert retriever.retrieve("nginx", k=1)[0].doc_id == "d9"


def test_k_bounds_the_result(retriever):
    assert len(retriever.retrieve("nginx systemd selinux", k=2)) == 2


@dataclass
class AliasedEntity(FakeEntity):
    aliases: list[str] = field(default_factory=list)


def _lookup(entities, relations=()):
    texts = {c: c for e in entities for c in e.chunks}
    return ExtractedGraphRetriever(entities, list(relations), texts)


def test_stopword_entities_are_never_linked():
    # The ServerFault extractor produced an entity named "a"; any-word linking
    # reached it, and every long name containing "with", from most questions.
    retriever = _lookup(
        [
            FakeEntity("a", ["d1::0"]),
            FakeEntity("with", ["d2::0"]),
            FakeEntity("connect() failed with error", ["d3::0"]),
            FakeEntity("nginx", ["d4::0"]),
        ]
    )
    assert retriever.link("nginx with a proxy") == ["nginx"]


def test_a_multi_word_name_links_only_as_a_whole_run():
    retriever = _lookup([FakeEntity("bad owner or permissions", ["d1::0"])])
    assert retriever.link("ssh says bad owner") == []
    assert retriever.link("ssh says bad owner or permissions") == [
        "bad owner or permissions"
    ]


def test_linking_respects_token_boundaries():
    retriever = _lookup([FakeEntity("ssh", ["d1::0"]), FakeEntity("tar", ["d2::0"])])
    assert retriever.link("sshd refuses the start of a tarball") == []


def test_spellings_that_tokenize_alike_link_together():
    retriever = _lookup(
        [FakeEntity("~/.ssh/config", ["d1::0"]), FakeEntity(".ssh/config", ["d2::0"])]
    )
    linked = retriever.link("which chmod fixes ~/.ssh/config?")
    assert set(linked) == {"~/.ssh/config", ".ssh/config"}


def test_a_trailing_full_stop_does_not_hide_an_entity():
    retriever = _lookup([FakeEntity("pkcs12", ["d1::0"])])
    assert retriever.link("convert it to pkcs12.") == ["pkcs12"]


def test_an_alias_links_its_entity():
    retriever = _lookup(
        [AliasedEntity("postgresql", ["d1::0"], aliases=["postgres", "pg"])]
    )
    assert retriever.link("restart postgres") == ["postgresql"]


def test_a_rare_entity_outranks_a_common_one():
    common = [FakeEntity("linux", [f"d{i}::0" for i in range(20)])]
    rare = [FakeEntity("pkcs12", ["d99::0"])]
    retriever = _lookup(common + rare)
    assert retriever.link("pkcs12 on linux") == ["pkcs12", "linux"]
    assert retriever.retrieve("pkcs12 on linux", k=1)[0].doc_id == "d99"


def test_a_hub_spreads_less_weight_to_each_neighbour_than_a_leaf():
    hub_neighbours = [FakeEntity(f"tool{i}", [f"h{i}::0"]) for i in range(16)]
    entities = [
        FakeEntity("hub", ["d1::0"]),
        FakeEntity("leaf", ["d2::0"]),
        FakeEntity("partner", ["d3::0"]),
        *hub_neighbours,
    ]
    relations = [FakeRelation("hub", e.name) for e in hub_neighbours]
    relations.append(FakeRelation("leaf", "partner"))
    retriever = _lookup(entities, relations)
    hub = retriever.expand(["hub"])
    leaf = retriever.expand(["leaf"])
    assert hub["tool0"] < leaf["partner"] < leaf["leaf"]
