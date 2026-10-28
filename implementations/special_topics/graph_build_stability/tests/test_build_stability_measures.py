"""The measures, derived graphs, and prompts behave as documented, with no model."""

from __future__ import annotations

import pytest
import stability as st
from aieng.advanced_rag.graphrag import extraction as ex


def ext(doc: str, *triples: tuple[str, str, str], failed: bool = False) -> st.Extraction:
    return st.Extraction.from_raw(doc, [{"Head": h, "Relation": r, "Tail": t} for h, r, t in triples], failed=failed)


def test_normalise_name_folds_case_accents_punctuation_and_articles() -> None:
    assert st.normalise_name("The Beatles") == st.normalise_name("beatles")
    assert st.normalise_name("Café  Müller!") == "cafe muller"
    assert st.normalise_name("Paris, France") != st.normalise_name("Paris")
    assert st.normalise_name("...") == ""


def test_extraction_drops_empty_names_and_self_loops() -> None:
    e = ext("d", ("A", "x", "A"), ("", "x", "B"), ("A", "Likes", "B"))
    assert e.labelled_edges() == {("a", "likes", "b")}
    assert e.nodes() == {"a", "b"}
    assert e.pairs() == {frozenset(("a", "b"))}


def test_jaccard_of_two_empty_sets_is_one() -> None:
    assert st.jaccard(set(), set()) == 1.0
    assert st.jaccard({1}, set()) == 0.0
    assert st.jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)


def test_identical_builds_agree_fully_and_disjoint_builds_not_at_all() -> None:
    a = [ext("d1", ("a", "r", "b")), ext("d2", ("c", "r", "d"))]
    b = [ext("d1", ("x", "r", "y")), ext("d2", ("z", "r", "w"))]
    same = st.compare_builds(a, a)
    assert same["graph"] == {"nodes": 1.0, "labelled_edges": 1.0, "edges": 1.0}
    assert st.compare_builds(a, b)["graph"]["nodes"] == 0.0


def test_a_relabelled_edge_keeps_its_pair_and_loses_its_label() -> None:
    a = [ext("d", ("a", "founded", "b"))]
    b = [ext("d", ("b", "was founded by", "a"))]
    graph = st.compare_builds(a, b)["graph"]
    assert graph["edges"] == 1.0
    assert graph["labelled_edges"] == 0.0


def test_type_agreement_counts_only_entities_both_builds_typed() -> None:
    a = [
        st.Extraction.from_raw(
            "d", [], entities=[{"name": "nginx", "type": "Software"}, {"name": "x", "type": "Concept"}]
        )
    ]
    b = [
        st.Extraction.from_raw(
            "d", [], entities=[{"name": "nginx", "type": "Software"}, {"name": "x", "type": "Command"}]
        )
    ]
    assert st.compare_builds(a, b)["type_agreement"] == 0.5


def test_compare_many_averages_every_pair() -> None:
    a = [ext("d", ("a", "r", "b"))]
    c = [ext("d", ("a", "r", "c"))]
    out = st.compare_many({"r1": a, "r2": a, "r3": c})
    assert set(out["pairs"]) == {"r1|r2", "r1|r3", "r2|r3"}
    assert out["mean"]["graph_labelled_edges"] == pytest.approx(1 / 3, abs=1e-4)


def test_drop_triples_removes_about_the_asked_share() -> None:
    records = [
        st.atlas_record(f"d{i}", "t", [{"Head": "a", "Relation": "r", "Tail": f"b{j}"} for j in range(10)], "")
        for i in range(100)
    ]
    kept = sum(len(r["entity_relation_dict"]) for r in st.drop_triples(records, 0.3, seed=1))
    assert 600 < kept < 800
    assert st.drop_triples(records, 0.0) == [dict(r, entity_relation_output="0% of triples dropped") for r in records]


def test_union_keeps_each_distinct_triple_once() -> None:
    one = [st.atlas_record("d", "t", [{"Head": "a", "Relation": "r", "Tail": "b"}], "")]
    two = [
        st.atlas_record(
            "d", "t", [{"Head": "a", "Relation": "r", "Tail": "b"}, {"Head": "a", "Relation": "s", "Tail": "c"}], ""
        )
    ]
    merged = st.union_records([one, two])
    assert len(merged) == 1
    assert len(merged[0]["entity_relation_dict"]) == 2


def test_title_mentions_match_whole_words_and_never_target_short_titles() -> None:
    passages = [
        ("p1", "Lionel Messi", "He played for Barcelona and met Ann."),
        ("p2", "Barcelona", "A city. Barcelonas is not a title."),
        ("p3", "Ann", "Ann wrote about Lionel Messi."),
    ]
    records = {r["id"]: r["entity_relation_dict"] for r in st.title_mention_records(passages)}
    assert records["p1"] == [{"Head": "Lionel Messi", "Relation": "mentions", "Tail": "Barcelona"}]
    assert records["p2"] == []
    assert records["p3"] == [{"Head": "Ann", "Relation": "mentions", "Tail": "Lionel Messi"}]


def test_fold_relation_maps_aliases_and_rejects_the_rest() -> None:
    assert st.fold_relation("Born In") == "born_in"
    assert st.fold_relation("performed_by") == "created_by"
    assert st.fold_relation("located_in") == "located_in"
    assert st.fold_relation("won") is None
    assert set(st.MUSIQUE_RELATION_ALIASES.values()) <= set(st.MUSIQUE_RELATIONS)


def test_schema_instruction_names_every_relation_type() -> None:
    text = st.schema_instruction("Extract triples.")
    assert text.startswith("Extract triples.")
    assert all(r in text for r in st.MUSIQUE_RELATIONS)


def test_free_type_prompt_changes_only_the_type_lines() -> None:
    schema = ex.PROMPT.format(
        entity_types=", ".join(ex.ENTITY_TYPES), relation_types=", ".join(ex.RELATION_TYPES), text="BODY"
    )
    free = st.free_type_prompt(schema, ex.ENTITY_TYPES, ex.RELATION_TYPES)
    assert "Software" not in free and "diagnoses" not in free
    assert free.endswith("BODY")
    assert "At most 12 entities and 12 relations." in free
    with pytest.raises(ValueError):
        st.free_type_prompt("no type lines", ex.ENTITY_TYPES, ex.RELATION_TYPES)


def test_ranking_agreement_compares_top_k_sets() -> None:
    a = {"q1": ["p1", "p2", "p3"], "q2": ["p1", "p2", "p3"]}
    b = {"q1": ["p2", "p1", "p3"], "q2": ["p4", "p5", "p6"]}
    out = st.ranking_agreement(a, b, k=2)
    assert out["identical_top_k"] == 0.5
    assert out["top_k_jaccard"]["mean"] == 0.5


def test_structure_and_degree_distance() -> None:
    build = [ext("d1", ("a", "r", "b"), ("a", "s", "c")), ext("d2"), ext("d3", ("a", "r", "b"), failed=True)]
    shape = st.structure(build)
    assert shape["nodes"] == 3 and shape["edges"] == 2 and shape["labelled_edges"] == 2
    assert shape["empty_documents"] == 1 and shape["failed_documents"] == 1
    assert shape["top_hubs"][0] == "a"
    assert shape["cross_document_nodes"] == 2
    assert st.degree_distance(build, build) == 0.0


def test_invented_names_are_distinct_stable_and_keep_word_counts() -> None:
    titles = ["Lionel Messi", "FC Barcelona", "Paris", "110th United States Congress"]
    one = st.invent_names(titles, seed=3)
    assert one == st.invent_names(list(reversed(titles)), seed=3)
    assert len(set(one.values())) == len(titles)
    assert all(len(one[t].split()) == len(t.split()) for t in titles)


def test_rename_text_replaces_whole_titles_longest_first() -> None:
    mapping = {"Paris": "Kalo", "Paris Saint-Germain": "Mitu Renka"}
    text = "Paris Saint-Germain plays in Paris, not Parisian suburbs."
    assert st.rename_text(text, mapping) == "Mitu Renka plays in Kalo, not Parisian suburbs."


def test_response_identity_counts_exact_matches_and_where_pairs_diverge() -> None:
    a = {"d1": "same", "d2": "abcdXXXX", "d3": "only in a"}
    b = {"d1": "same", "d2": "abcdYYYY"}
    result = st.response_identity(a, b)
    assert result["documents"] == 2
    assert result["identical"] == 0.5
    assert result["divergence_point"]["mean"] == 0.5


def test_top_hubs_break_degree_ties_by_name() -> None:
    build = [ext("d", *[(f"n{i}", "r", f"m{i}") for i in range(12)])]
    assert st.structure(build)["top_hubs"] == sorted(st.graph_nodes(build))[:10]
