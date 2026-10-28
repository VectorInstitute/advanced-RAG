"""Validate the complete pinned MuSiQue packaging, including its hop graphs.

Run explicitly on compute with ``-m integration_test``. Set
``ADVANCED_RAG_CORPUS_CACHE`` to a writable local cache; files stay out of git.
"""

from __future__ import annotations

from collections import Counter

import pytest

from aieng.advanced_rag.corpora.musique import HOP_COUNTS, load_musique


@pytest.mark.integration_test
def test_pinned_packaging_exposes_all_supports_and_connected_hop_graphs():
    passages, questions = load_musique()
    assert len(passages) == 11656
    assert len(questions) == 1000
    assert Counter(q.hops for q in questions) == HOP_COUNTS
    assert len({q.question_id for q in questions}) == len(questions)
    corpus_ids = {p.passage_id for p in passages}
    assert len(corpus_ids) == len(passages)
    assert sum(len(q.supporting_ids) for q in questions) == 2648
    assert sum(len(q.hop_graph.edges) for q in questions) == 1648
    for question in questions:
        graph = question.hop_graph
        ids = {node.step_id for node in graph.nodes}
        assert None not in ids
        assert len(ids) == question.hops
        assert len(set(question.supporting_ids)) == question.hops
        assert set(question.supporting_ids) <= corpus_ids
        assert (
            tuple(node.support_passage_id for node in graph.nodes)
            == question.supporting_ids
        )
        # References are backwards in topological order; also check connectivity.
        earlier = set()
        for node in graph.nodes:
            assert set(node.depends_on) <= earlier
            earlier.add(node.step_id)
        reached = {graph.nodes[0].step_id}
        while True:
            expanded = reached | {
                endpoint
                for source, target in graph.edges
                if source in reached or target in reached
                for endpoint in (source, target)
            }
            if expanded == reached:
                break
            reached = expanded
        assert reached == ids, question.question_id
