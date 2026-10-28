"""Tests for parent-child chunking and the small-to-big read step."""

from __future__ import annotations

import pytest

from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.pipeline.hierarchical import (
    Hierarchy,
    Node,
    contextualize,
    detect_headings,
    small_to_big,
    split_document,
    split_documents,
    to_evidence,
)
from aieng.advanced_rag.provenance import Provenance


CONTRACT = (
    "ARTICLE I  DEFINITIONS\n"
    + "The term Agreement means this contract between Party A and Party B. " * 12
    + "\n\nARTICLE II  GOVERNING LAW\n"
    + "This Agreement is governed by the laws of the Province of Ontario. " * 12
)


def build(**options) -> Hierarchy:
    defaults = {"parent_chars": 800, "child_chars": 250, "overlap_chars": 50}
    return split_document("doc1", CONTRACT, **{**defaults, **options})


def test_offsets_index_back_into_the_source_document():
    # Span-derived relevance judgments depend on this, so it is the property
    # that has to hold for every node, parent and child alike.
    hierarchy = build()
    for node in (*hierarchy.parents, *hierarchy.children):
        assert CONTRACT[node.start : node.end] == node.text


def test_every_child_sits_inside_exactly_one_parent():
    hierarchy = build()
    for child in hierarchy.children:
        parent = hierarchy.parent_of(child.node_id)
        assert parent is not None
        assert parent.start <= child.start and child.end <= parent.end


def test_headings_become_parent_boundaries_and_label_their_nodes():
    hierarchy = build()
    headings = {parent.heading for parent in hierarchy.parents}
    assert "ARTICLE II  GOVERNING LAW" in headings
    starts = {parent.start for parent in hierarchy.parents}
    assert CONTRACT.index("ARTICLE II") in starts


def test_capitalised_body_lines_do_not_each_become_a_heading():
    shouting = "\n".join(f"THIS LINE IS ENTIRELY IN CAPITALS {n}" for n in range(20))
    assert len(detect_headings(shouting, min_spacing=200)) < 5


def test_headings_can_be_switched_off():
    hierarchy = build(use_headings=False)
    assert {parent.heading for parent in hierarchy.parents} == {None}


def test_a_short_tail_folds_into_the_window_before_it():
    text = "x" * 1010
    hierarchy = split_document(
        "doc1", text, parent_chars=1000, child_chars=1000, overlap_chars=0
    )
    assert len(hierarchy.parents) == 1
    assert hierarchy.parents[0].end == 1010


def test_expand_collapses_sibling_hits_into_one_read_unit():
    hierarchy = build()
    siblings = [
        child.node_id
        for child in hierarchy.children
        if child.parent_id == hierarchy.parents[0].node_id
    ]
    assert len(siblings) > 1
    assert [node.node_id for node in hierarchy.expand(siblings)] == [
        hierarchy.parents[0].node_id
    ]


def test_expand_keeps_rank_order_and_honours_the_limit():
    hierarchy = build()
    last = hierarchy.children[-1].node_id
    first = hierarchy.children[0].node_id
    ordered = hierarchy.expand([last, first])
    assert ordered[0].node_id == hierarchy.parent_of(last).node_id
    assert len(hierarchy.expand([last, first], limit=1)) == 1


def test_expand_skips_identifiers_it_does_not_know():
    assert build().expand(["nothing::here"]) == []


def test_contextualize_prefixes_the_document_and_section():
    hierarchy = build()
    contextualised = contextualize(hierarchy)
    assert contextualised[0].text.startswith("doc1 > ARTICLE I  DEFINITIONS")
    # Offsets are untouched, so judgments derived from them stay valid even
    # though the indexed text no longer starts at ``start``.
    assert contextualised[0].start == hierarchy.children[0].start


def test_contextualize_can_call_a_model_once_per_child():
    hierarchy = build()
    calls: list[tuple[str, str]] = []

    def describe(parent_text: str, child_text: str) -> str:
        calls.append((parent_text, child_text))
        return "clause about definitions"

    contextualised = contextualize(hierarchy, describe=describe)
    assert len(calls) == len(hierarchy.children)
    assert "clause about definitions" in contextualised[0].text


def test_small_to_big_swaps_the_text_and_keeps_the_ranked_identifier():
    hierarchy = build()
    child = hierarchy.children[0]
    parent = hierarchy.parent_of(child.node_id)
    expanded = small_to_big(hierarchy, to_evidence([child]))
    assert len(expanded) == 1
    assert expanded[0].text == parent.text
    assert expanded[0].chunk_id == child.node_id
    assert expanded[0].provenance.start == parent.start
    assert expanded[0].provenance.extra["parent_id"] == parent.node_id


def test_small_to_big_de_duplicates_siblings_and_honours_the_limit():
    hierarchy = build()
    siblings = [
        child
        for child in hierarchy.children
        if child.parent_id == hierarchy.parents[0].node_id
    ]
    assert len(small_to_big(hierarchy, to_evidence(siblings))) == 1
    everything = to_evidence(list(hierarchy.children))
    assert len(small_to_big(hierarchy, everything, limit=1)) == 1


def test_evidence_from_outside_the_hierarchy_passes_through_untouched():
    stray = Evidence(
        text="from somewhere else",
        provenance=Provenance(doc_id="other", chunk_id="other::000"),
    )
    assert small_to_big(build(), [stray]) == [stray]


def test_split_documents_merges_and_keeps_lookups_working():
    hierarchy = split_documents(
        {"a": CONTRACT, "b": CONTRACT}, parent_chars=800, child_chars=250
    )
    assert hierarchy.summary()["documents"] == 2
    for child in hierarchy.children:
        assert hierarchy.parent_of(child.node_id) is not None


def test_node_provenance_carries_the_section_for_a_citation():
    node = Node(
        node_id="doc1::p000",
        doc_id="doc1",
        text="text",
        start=0,
        end=4,
        heading="ARTICLE I",
    )
    assert node.is_parent
    assert node.provenance().citation() == "doc1 > ARTICLE I"


@pytest.mark.parametrize(
    ("child_chars", "overlap_chars", "parent_chars"),
    [(0, 0, 100), (100, 100, 200), (300, 0, 200)],
)
def test_sizes_that_cannot_make_progress_are_rejected(
    child_chars, overlap_chars, parent_chars
):
    with pytest.raises(ValueError):
        split_document(
            "doc1",
            CONTRACT,
            parent_chars=parent_chars,
            child_chars=child_chars,
            overlap_chars=overlap_chars,
        )
