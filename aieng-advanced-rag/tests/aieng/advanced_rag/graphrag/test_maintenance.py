"""Graph maintenance: retraction, revision, deduplication, and the gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aieng.advanced_rag.graphrag import maintenance as mt
from aieng.advanced_rag.graphrag.extraction import KnowledgeGraph


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


def doc(title, question, answer):
    return f"# {title}\n\nTags: t\n\n## Question\n{question}\n\n## Accepted answer\n{answer}\n"


DOCS = {
    "q1": doc(
        "Nginx 502",
        "Nginx returns 502 after php-fpm restarts.",
        "Fix the fastcgi_pass socket in nginx.",
    ),
    "q2": doc(
        "Nginx logs",
        "Where does nginx write its error log?",
        "Look in /var/log/nginx for the error log.",
    ),
    "q3": doc(
        "Pem files",
        "What is a .pem file?",
        "A pem file holds a certificate for openssl.",
    ),
}

# Each document's two calls (question with title, then answer), keyed by a word in
# the prompt that only that call contains.
REPLIES = {
    "php-fpm restarts": reply(
        [
            ent("nginx", evidence="Nginx returns 502"),
            ent("php-fpm", evidence="php-fpm restarts"),
        ],
        [rel("php-fpm", "nginx", "causes", evidence="502 after php-fpm")],
    ),
    "fastcgi_pass socket": reply(
        [
            ent(
                "fastcgi_pass",
                "ConfigItem",
                "fastcgi_pass socket",
                "long description from q1",
            ),
            ent(
                "nginx",
                evidence="in nginx",
                description="nginx as q1 describes it at length",
            ),
        ],
        [
            rel(
                "fastcgi_pass",
                "nginx",
                "configures",
                evidence="fastcgi_pass socket in nginx",
            )
        ],
    ),
    "error log?": reply(
        [
            ent("nginx", "Concept", "nginx write"),
            ent("error log", "ConfigItem", "error log"),
        ],
        [rel("nginx", "error log", "configures", evidence="nginx write its error log")],
    ),
    "/var/log/nginx": reply(
        [
            ent(
                "error log",
                "ConfigItem",
                "the error log",
                "a longer error log description",
            )
        ]
    ),
    "a .pem file": reply([ent(".pem file", "ConfigItem", ".pem file")]),
    "certificate for": reply(
        [
            ent("pem file", "ConfigItem", "A pem file"),
            ent("openssl", evidence="openssl"),
        ],
        [
            rel(
                "openssl",
                "pem file",
                "configures",
                evidence="pem file holds a certificate for openssl",
            )
        ],
    ),
}


def complete(prompt: str) -> str:
    body = prompt.split("TEXT:", 1)[1]
    for marker, answer in REPLIES.items():
        if marker in body:
            return answer
    return reply()


@pytest.fixture
def units():
    return mt.extract_units(DOCS.items(), complete, max_workers=1)


@pytest.fixture
def graph(units):
    return mt.fold_units(units)


def test_units_fold_into_the_graph_a_single_pass_builds(units, graph):
    from aieng.advanced_rag.graphrag.extraction import build_graph

    whole = build_graph(DOCS.items(), complete, max_workers=1)
    diff = mt.graph_diff(whole, graph)
    assert diff.entity_jaccard == 1.0 and diff.relation_jaccard == 1.0
    assert graph.entities["nginx"].type == "Software"  # the first document's type
    assert graph.entities["nginx"].description == "nginx as q1 describes it at length"
    assert len(graph.entities["nginx"].mentions) == 3


def test_retraction_removes_what_only_the_post_supports(graph, units):
    report = mt.retract(graph, ["q1"], units=units)
    assert sorted(report.entities_removed) == ["fastcgi_pass", "php-fpm"]
    assert report.entities_touched == ["nginx"]
    assert len(report.relations_removed) == 2
    assert report.remaining_citations == 0
    assert mt.facts_citing(graph, ["q1"]) == []
    assert graph.validate() == []


def test_retraction_repairs_type_and_description_from_the_remaining_units(graph, units):
    report = mt.retract(graph, ["q1"], units=units)
    nginx = graph.entities["nginx"]
    assert nginx.description == "d"
    assert nginx.type == "Concept"
    assert report.descriptions_repaired == 1 and report.types_changed == 1


def test_retraction_without_units_flags_descriptions_as_unverified(graph):
    report = mt.retract(graph, ["q1"])
    assert report.descriptions_unverified == 1
    assert graph.entities["nginx"].description == "nginx as q1 describes it at length"


def test_a_retraction_with_units_equals_a_rebuild_without_the_post(graph, units):
    mt.retract(graph, ["q1"], units=units)
    rebuilt = mt.fold_units({k: v for k, v in units.items() if k != "q1"})
    assert mt.graph_diff(graph, rebuilt).entity_jaccard == 1.0
    for name, entity in rebuilt.entities.items():
        assert graph.entities[name].description == entity.description
        assert graph.entities[name].type == entity.type
        assert len(graph.entities[name].mentions) == len(entity.mentions)


def test_impact_counts_the_neighbourhood_without_changing_the_graph(graph):
    index = mt.ProvenanceIndex.of(graph)
    before = len(graph.entities)
    impact = mt.retraction_impact(graph, ["q1"], index)
    assert len(graph.entities) == before
    assert impact.neighbourhood == 1  # error log, one hop from nginx


def test_a_relation_loses_its_endpoint_and_is_dropped_as_dangling():
    graph = KnowledgeGraph.from_dict(
        {
            "entities": [
                {
                    "name": "a",
                    "type": "Software",
                    "description": "",
                    "mentions": [
                        {"provenance": "doc:x", "quote": "", "located": False}
                    ],
                },
                {
                    "name": "b",
                    "type": "Software",
                    "description": "",
                    "mentions": [
                        {"provenance": "doc:y", "quote": "", "located": False}
                    ],
                },
            ],
            "relations": [
                {
                    "source": "a",
                    "target": "b",
                    "type": "fixes",
                    "description": "",
                    "mentions": [
                        {"provenance": "doc:y", "quote": "", "located": False}
                    ],
                }
            ],
        }
    )
    report = mt.retract(graph, ["x"])
    assert report.relations_dangling == [("a", "b", "fixes")]
    assert graph.relations == {}
    assert graph.validate() == []


def test_revise_swaps_in_the_new_extraction(graph, units):
    new = mt.extract_units(
        [
            (
                "q3",
                doc(
                    "Pem files", "What is a .pem file?", "Use openssl x509 to read it."
                ),
            )
        ],
        complete,
    )["q3"]
    mt.revise(graph, "q3", new, units=units)
    assert "pem file" not in graph.entities
    assert units["q3"] is new
    assert graph.validate() == []


def test_changed_spans_mark_the_edited_words():
    old, new = "set worker_connections to 512 now", "set worker_connections to 1024 now"
    in_old, in_new = mt.changed_spans(old, new)
    assert [old[s:e] for s, e in in_old] == ["512"]
    assert [new[s:e] for s, e in in_new] == ["1024"]


def test_edit_alignment_places_added_facts_at_the_edit(units):
    old_text = DOCS["q3"]
    new_text = old_text.replace(
        "A pem file holds a certificate for openssl.", "Use openssl x509 to read it."
    )
    new_unit = mt.extract_units([("q3", new_text)], complete)["q3"]
    result = mt.edit_alignment(old_text, new_text, units["q3"], new_unit)
    assert result["removed_entities_at_edit"] == 2  # pem file and openssl
    assert result["entities_kept"] == 1
    assert result["changed_chars_new"] > 0


@pytest.mark.parametrize(
    "a, b",
    [
        ("x.509", "x509"),
        ("pem files", "pem file"),
        ("remote-server", "remote servers"),
        ("setsockopt()", "setsockopt"),
    ],
)
def test_fold_name_meets_spelling_variants(a, b):
    assert mt.fold_name(a) == mt.fold_name(b) is not None


@pytest.mark.parametrize(
    "name", ["rm -rf /*", "--rm", ".env", "$host", "~/.ssh/config", "location ~"]
)
def test_fold_name_leaves_flags_paths_and_patterns_alone(name):
    assert mt.fold_name(name) is None


def test_fold_name_keeps_words_whose_s_is_not_a_plural():
    assert mt.fold_name("https") != mt.fold_name("http")
    assert mt.fold_name("windows") != mt.fold_name("window")
    assert mt.fold_name("process") == "process"


def test_merge_unions_mentions_and_rewires_relations(graph, units):
    variant = mt.extract_units(
        [
            (
                "q4",
                doc(
                    "Pem",
                    "Are pem-files text?",
                    "Yes, pem-files are base64 text read by openssl.",
                ),
            )
        ],
        lambda prompt: reply(
            [
                ent("pem-files", "Concept", "pem-files"),
                ent("openssl", evidence="openssl"),
            ],
            [rel("openssl", "pem-files", "configures", evidence="read by openssl")],
        ),
    )["q4"]
    mt.add_unit(graph, variant)
    groups = mt.duplicate_groups(graph)
    # Both stub calls return the variant, so it has more mentions and is kept.
    assert groups == [["pem-files", "pem file"]]
    report = mt.merge_entities(graph, groups[0])
    assert report.kept == "pem-files"
    assert report.relations_rewired == 1 and report.relations_collapsed == 1
    assert len(graph.entities["pem-files"].mentions) == 3
    assert "pem file" not in graph.entities
    assert len(graph.relations[("openssl", "pem-files", "configures")].mentions) == 3
    assert graph.validate() == []


def test_merge_collapses_parallel_relations_and_drops_loops():
    payload = {
        "entities": [
            {
                "name": n,
                "type": "Software",
                "description": "",
                "mentions": [{"provenance": f"doc:{n}", "quote": "", "located": False}],
            }
            for n in ("a", "a-b", "ab", "c")
        ],
        "relations": [
            {
                "source": s,
                "target": t,
                "type": "fixes",
                "description": "",
                "mentions": [{"provenance": "doc:r", "quote": "", "located": False}],
            }
            for s, t in (("a-b", "c"), ("ab", "c"), ("a-b", "ab"))
        ],
    }
    graph = KnowledgeGraph.from_dict(payload)
    report = mt.merge_entities(graph, ["a-b", "ab"], into="ab")
    assert report.relations_collapsed == 1 and report.self_loops_dropped == 1
    assert list(graph.relations) == [("ab", "c", "fixes")]
    assert len(graph.relations[("ab", "c", "fixes")].mentions) == 2
    with pytest.raises(ValueError, match="not one of"):
        mt.merge_entities(graph, ["a"], into="c")


def test_topology_counts_components_and_hubs(graph):
    stats = mt.topology(graph)
    assert stats["entities"] == len(graph.entities)
    assert stats["components"] == 3  # nginx, openssl and pem file, and .pem file alone
    assert stats["hubs"][0][0] == "nginx"
    delta = mt.topology_delta(stats, stats)
    assert delta["entities"] == 0


def test_the_gate_passes_a_provenance_retraction_and_fails_a_stale_index(graph, units):
    before = mt.topology(graph)
    lookup = mt.EntityLookup(graph)
    question = "Why does php-fpm make nginx return 502?"
    smoke_before = {question: lookup.cite(question)}
    assert "q1" in smoke_before[question]
    changed = mt.copy_graph(graph)
    mt.retract(changed, ["q1"], units=units)
    limits = mt.GateLimits(
        max_entity_loss_share=1.0,
        max_isolated_increase_share=1.0,
        max_largest_component_drop=1.0,
        max_type_share_shift=1.0,
    )
    fresh = mt.EntityLookup(changed)
    ok = mt.maintenance_gate(
        changed,
        before_topology=before,
        retracted=["q1"],
        smoke={question: lambda: fresh.cite(question)},
        smoke_before=smoke_before,
        limits=limits,
    )
    assert ok.passed, ok.to_dict()
    stale = mt.maintenance_gate(
        changed,
        before_topology=before,
        retracted=["q1"],
        smoke={question: lambda: lookup.cite(question)},
        smoke_before=smoke_before,
        limits=limits,
    )
    assert not stale.passed
    assert any("retracted: q1" in c.detail for c in stale.checks)


def test_the_gate_catches_facts_left_citing_nothing(graph):
    before = mt.topology(graph)
    for fact in list(graph.entities.values()) + list(graph.relations.values()):
        fact.mentions = [m for m in fact.mentions if m.provenance.doc_id != "q1"]
    result = mt.maintenance_gate(graph, before_topology=before, retracted=["q1"])
    failed = {c.name for c in result.checks if not c.passed}
    assert "schema: validate()" in failed


def test_an_unarmed_smoke_question_fails(graph):
    before = mt.topology(graph)
    result = mt.maintenance_gate(
        graph,
        before_topology=before,
        retracted=["q1"],
        smoke={"q": lambda: ["q2"]},
        smoke_before={"q": ["q2"]},
    )
    assert not result.passed
    assert "not armed" in result.checks[-1].detail


def test_expected_removals_do_not_count_against_entity_loss(graph):
    before = mt.topology(graph)
    mt.merge_entities(graph, ["pem file", ".pem file"])
    strict = mt.GateLimits(max_entity_loss_share=0.0)
    assert not mt.maintenance_gate(graph, before_topology=before, limits=strict).passed
    assert (
        mt.maintenance_gate(
            graph, before_topology=before, limits=strict, expected_entity_removals=1
        )
        .checks[2]
        .passed
    )


def test_post_history_reads_bodies_and_titles_in_order(tmp_path: Path):
    xml = tmp_path / "PostHistory.xml"
    xml.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<posthistory>\n'
        '<row Id="1" PostHistoryTypeId="2" PostId="7" RevisionGUID="g1" CreationDate="2010-01-01T00:00:00" Text="first **body**" />\n'
        '<row Id="2" PostHistoryTypeId="1" PostId="7" RevisionGUID="g1" CreationDate="2010-01-01T00:00:00" Text="a title" />\n'
        '<row Id="3" PostHistoryTypeId="3" PostId="7" RevisionGUID="g1" CreationDate="2010-01-01T00:00:00" Text="|tag|" />\n'
        '<row Id="4" PostHistoryTypeId="5" PostId="7" RevisionGUID="g2" CreationDate="2011-01-01T00:00:00" Text="second body" />\n'
        '<row Id="5" PostHistoryTypeId="2" PostId="8" RevisionGUID="g3" CreationDate="2011-01-01T00:00:00" Text="other post" />\n'
        "</posthistory>\n",
        encoding="utf-8",
    )
    history = mt.read_post_history(xml, [7])
    assert list(history) == [7]
    assert [r.kind for r in history[7]] == [2, 1, 5]
    assert [r.text for r in mt.body_revisions(history[7])] == [
        "first **body**",
        "second body",
    ]


def test_render_markdown_matches_the_loader_on_rendered_html():
    pytest.importorskip("markdown_it")
    from aieng.advanced_rag.corpora.serverfault import strip_html

    source = "Use `nginx -t` first.\n\n    server { listen 80; }\n\n- one\n- two"
    html = "<p>Use <code>nginx -t</code> first.</p>\n<pre><code>server { listen 80; }\n</code></pre>\n<ul>\n<li>one</li>\n<li>two</li>\n</ul>\n"
    assert mt.render_markdown(source) == strip_html(html)


def test_extract_member_needs_a_7z_program(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mt.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="7z"):
        mt.extract_member(tmp_path / "dump.7z", tmp_path / "out", "PostHistory.xml")
