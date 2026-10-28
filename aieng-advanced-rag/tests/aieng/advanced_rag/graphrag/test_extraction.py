"""Direct extraction: schema enforcement, cited provenance, and determinism."""

from __future__ import annotations

import json
import random
import threading
import time

import pytest

from aieng.advanced_rag.graphrag import extraction as ex
from aieng.advanced_rag.graphrag.adapters.extracted_graph import ExtractedGraphRetriever
from aieng.advanced_rag.graphrag.extraction import (
    ENTITY_TYPES,
    RELATION_TYPES,
    KnowledgeGraph,
    build_graph,
    canonical_entity_type,
    canonical_relation_type,
    locate,
    parse_json_object,
    sections,
    windows,
)


DOC = (
    "# Nginx returns 502\n\nTags: nginx, linux\n\n"
    "## Question\nNginx returns 502 Bad Gateway after I restart php-fpm.\n\n"
    "The error log says connect() failed.\n\n"
    "## Accepted answer\nSet the fastcgi_pass socket path in nginx.conf.\n\n"
    "Then restart php-fpm so it recreates the socket.\n"
)


def reply(entities=(), relations=()):
    return json.dumps({"entities": list(entities), "relations": list(relations)})


def ent(name, type_="Software", evidence="", description="d"):
    return {
        "name": name,
        "type": type_,
        "description": description,
        "evidence": evidence,
    }


def rel(source, target, type_="fixes", evidence="", description="d"):
    return {
        "source": source,
        "target": target,
        "type": type_,
        "description": description,
        "evidence": evidence,
    }


def test_the_schema_has_seven_entity_and_seven_relation_types():
    assert len(ENTITY_TYPES) == 7 and len(RELATION_TYPES) == 7


@pytest.mark.parametrize(
    "label, expected, coerced",
    [
        ("Software", "Software", False),
        ("tool", "Software", False),
        ("Config Item", "ConfigItem", False),
        ("error-message", "ErrorOrSymptom", False),
        ("Wizard", "Concept", True),
        ("", "Concept", True),
    ],
)
def test_entity_labels_map_onto_the_schema(label, expected, coerced):
    assert canonical_entity_type(label) == (expected, coerced)


@pytest.mark.parametrize(
    "label, expected",
    [
        ("resolves", "fixes"),
        ("Depends On", "depends_on"),
        ("part-of", "part_of"),
        ("leads to", "causes"),
        ("likes", None),
    ],
)
def test_relation_labels_map_onto_the_schema_or_drop(label, expected):
    assert canonical_relation_type(label) == expected


def test_every_alias_lands_on_a_schema_label():
    assert set(ex.ENTITY_ALIASES.values()) <= set(ENTITY_TYPES)
    assert set(ex.RELATION_ALIASES.values()) <= set(RELATION_TYPES)


def test_a_document_splits_into_sections_of_blocks_with_document_offsets():
    found = dict(sections("q1", DOC))
    assert list(found) == ["Title", "Question", "Accepted answer"]
    assert [b.index for b in found["Question"]] == [0, 1]
    for blocks in found.values():
        for block in blocks:
            assert DOC[block.start : block.end] == block.text
    assert found["Accepted answer"][0].block_id == "q1::accepted-answer::b000"


def test_blocks_are_grouped_into_windows_at_block_boundaries():
    blocks = dict(sections("q1", DOC))["Question"]
    assert windows(blocks) == [blocks]
    assert [len(w) for w in windows(blocks, limit=10)] == [1, 1]


def test_an_exact_quote_is_located_to_its_block_and_span():
    blocks = dict(sections("q1", DOC))["Accepted answer"]
    cite = locate("fastcgi_pass socket path", blocks)
    assert cite is not None
    assert (cite.doc_id, cite.section, cite.chunk_id) == (
        "q1",
        "Accepted answer",
        "q1::accepted-answer::b000",
    )
    assert DOC[cite.start : cite.end] == "fastcgi_pass socket path"


def test_a_reflowed_or_recased_quote_is_still_located():
    blocks = dict(sections("q1", DOC))["Question"]
    cite = locate("NGINX returns   502\nbad gateway", blocks)
    assert cite is not None
    assert DOC[cite.start : cite.end].lower().startswith("nginx returns 502")


def test_a_quote_that_is_not_in_the_text_is_not_located():
    blocks = dict(sections("q1", DOC))["Question"]
    assert locate("a sentence the model invented", blocks) is None
    assert locate("ab", blocks) is None


def stub(payloads):
    """Return a model that answers by which text the prompt shows."""

    def complete(prompt):
        for needle, answer in payloads.items():
            if needle in prompt:
                return answer
        return reply()

    return complete


def test_entities_and_relations_are_enforced_and_cited():
    answer = reply(
        [
            ent("php-fpm", "service", "restart php-fpm"),
            ent("nginx.conf", "Config", "nginx.conf"),
            ent("502 bad gateway", "Error", "502 Bad Gateway"),
            ent("wizard thing", "Wizard", "invented words"),
        ],
        [
            rel(
                "nginx.conf", "502 bad gateway", "resolves", "fastcgi_pass socket path"
            ),
            rel("php-fpm", "nginx.conf", "likes", "restart php-fpm"),
            rel("php-fpm", "ghost", "fixes", "restart php-fpm"),
        ],
    )
    graph = build_graph([("q1", DOC)], stub({"fastcgi_pass": answer}))
    assert graph.validate() == []
    assert graph.entities["php-fpm"].type == "Software"
    assert graph.entities["nginx.conf"].type == "ConfigItem"
    assert graph.entities["wizard thing"].type == "Concept"
    assert [(r.source, r.type) for r in graph.relations.values()] == [
        ("nginx.conf", "fixes")
    ]
    report = graph.report
    assert report.entity_types_coerced == 1 and report.entity_types_mapped == 3
    assert report.relations_rejected == 1 and report.relations_dangling == 1
    assert report.relations_mapped == 1


def test_a_located_fact_cites_the_answer_block_and_an_unlocated_one_its_section():
    answer = reply(
        [
            ent("php-fpm", "Software", "recreates the socket"),
            ent("fastcgi", "Software", "nonsense text"),
        ]
    )
    graph = build_graph([("q1", DOC)], stub({"fastcgi_pass": answer}))
    located = graph.entities["php-fpm"].mentions[0]
    assert (
        located.located and located.provenance.chunk_id == "q1::accepted-answer::b001"
    )
    assert (
        DOC[located.provenance.start : located.provenance.end] == "recreates the socket"
    )
    missing = graph.entities["fastcgi"].mentions[0]
    assert not missing.located
    assert (
        missing.provenance.section == "Accepted answer"
        and missing.provenance.start is None
    )
    assert graph.report.quotes_located == 1 and graph.report.quotes_unlocated == 1


def test_an_entity_in_two_documents_is_one_node_with_two_citations():
    second = DOC.replace("502", "504")
    answer = reply([ent("nginx", "Software", "Nginx returns")])
    graph = build_graph(
        [("q1", DOC), ("q2", second)],
        stub({"Nginx returns": answer}),
    )
    assert len(graph.entities["nginx"].mentions) >= 2
    assert set(graph.entities["nginx"].documents) == {"q1", "q2"}
    assert graph.stats()["cross_document_entities"] >= 1


def test_a_failed_or_unparseable_call_adds_nothing_and_is_counted():
    def complete(prompt):
        if "Accepted answer" in prompt or "fastcgi_pass" in prompt:
            raise RuntimeError("proxy down")
        return "no json here"

    graph = build_graph([("q1", DOC)], complete)
    assert graph.entities == {} and graph.report.calls_failed > 0


def test_the_graph_does_not_depend_on_completion_order():
    answer = reply(
        [
            ent("nginx", "Software", "Nginx returns"),
            ent("php-fpm", "Software", "php-fpm"),
        ]
    )
    docs = [(f"q{i}", DOC) for i in range(6)]
    lock = threading.Lock()

    def slow_random(prompt):
        time.sleep(random.random() * 0.01)
        with lock:
            return answer

    one = build_graph(docs, slow_random, max_workers=1).to_dict()
    many = build_graph(docs, slow_random, max_workers=8).to_dict()
    assert json.dumps(one, sort_keys=True) == json.dumps(many, sort_keys=True)


def test_progress_is_reported_as_calls_finish():
    seen = []
    build_graph(
        [("q1", DOC)],
        stub({}),
        on_progress=lambda done, total: seen.append((done, total)),
    )
    assert seen and seen[-1][0] == seen[-1][1]


def test_the_graph_round_trips_through_json_and_files(tmp_path):
    answer = reply(
        [
            ent("php-fpm", "Software", "restart php-fpm"),
            ent("nginx.conf", "ConfigItem", "nginx.conf"),
        ],
        [rel("nginx.conf", "php-fpm", "configures", "fastcgi_pass socket path")],
    )
    graph = build_graph([("q1", DOC)], stub({"fastcgi_pass": answer}))
    graph.save(tmp_path)
    again = KnowledgeGraph.load(tmp_path)
    assert json.dumps(again.to_dict(), sort_keys=True) == json.dumps(
        graph.to_dict(), sort_keys=True
    )
    assert again.validate() == []
    assert (tmp_path / "graph.graphml").is_file()


def test_the_graphml_carries_each_nodes_citations(tmp_path):
    import networkx as nx

    answer = reply([ent("php-fpm", "Software", "restart php-fpm")])
    graph = build_graph([("q1", DOC)], stub({"fastcgi_pass": answer}))
    graph.write_graphml(tmp_path / "g.graphml")
    loaded = nx.read_graphml(tmp_path / "g.graphml")
    assert "doc:q1" in loaded.nodes["php-fpm"]["provenance"]
    assert loaded.nodes["php-fpm"]["type"] == "Software"


def test_validate_names_every_broken_rule():
    graph = KnowledgeGraph()
    graph.entities["x"] = ex.Entity("x", "Gadget")
    graph.relations[("x", "y", "likes")] = ex.Relation("x", "y", "likes")
    problems = graph.validate()
    assert any("off-schema type" in p for p in problems)
    assert any("cites nothing" in p for p in problems)
    assert any("dangling" in p for p in problems)


def test_the_existing_graph_retriever_reads_the_extracted_graph():
    answer = reply(
        [
            ent("php-fpm", "Software", "restart php-fpm"),
            ent("nginx.conf", "ConfigItem", "nginx.conf"),
        ],
        [rel("nginx.conf", "php-fpm", "configures", "fastcgi_pass socket path")],
    )
    graph = build_graph([("q1", DOC)], stub({"fastcgi_pass": answer}))
    texts = {b.block_id: b.text for _, blocks in sections("q1", DOC) for b in blocks}
    retriever = ExtractedGraphRetriever(
        graph.entities.values(), graph.relations.values(), texts
    )
    hits = retriever.retrieve("how do I restart php-fpm", k=3)
    assert hits and all(h.provenance.doc_id == "q1" for h in hits)


def test_json_is_pulled_out_of_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": 1} Hope that helps.') == {"a": 1}
    assert parse_json_object("nothing") == {}
    assert parse_json_object("[1, 2]") == {}


def test_a_document_costs_two_calls_and_the_title_rides_with_the_question():
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        return reply()

    build_graph([("q1", DOC)], complete)
    assert len(prompts) == 2
    question_prompt = next(p for p in prompts if "Bad Gateway after I restart" in p)
    assert (
        "Nginx returns 502" in question_prompt
        and "Tags: nginx, linux" in question_prompt
    )
    assert not any(
        "Bad Gateway after I restart" in p and "fastcgi_pass" in p for p in prompts
    )


def test_a_title_only_document_is_still_extracted():
    seen = []
    build_graph(
        [("q1", "# Only a title\n\nTags: a\n")], lambda p: seen.append(p) or reply()
    )
    assert len(seen) == 1
