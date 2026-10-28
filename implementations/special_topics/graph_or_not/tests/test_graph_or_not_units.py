"""Unit tests for the graph-or-not helpers; no model, network, or corpus is needed."""

from __future__ import annotations

import numpy as np
import pytest
from gon import readiness, relational, x1


# ---------------------------------------------------------------- X1


def test_interleave_and_rrf() -> None:
    assert x1.interleave([["a", "b", "c"], ["b", "d"]], 3) == ["a", "b", "d"]
    assert x1.rrf([["a", "b"], ["b", "a"]], 2) == ["a", "b"]


def test_top_k_breaks_ties_on_position() -> None:
    assert x1.top_k(np.array([0.1, 0.5, 0.5, 0.2]), ["w", "x", "y", "z"], 3) == ["x", "y", "z"]


def test_supporting_recall_and_table() -> None:
    assert x1.supporting_recall(["a", "b", "c"], ["a", "c"], 2) == (False, 0.5)
    table = x1.recall_table({"q": ["a", "b"]}, {"q": ["a", "b"]})
    assert table["all_supporting_recall_at_5"] == 1.0


def test_strong_stack_arms_shapes() -> None:
    ids = [f"p{i}" for i in range(30)]
    rng = np.random.default_rng(0)
    table = {i: rng.random() for i in ids}

    def scorer(_q: str) -> np.ndarray:
        return np.array([table[i] for i in ids])

    seen: list[list[str]] = []

    def rerank(_q: str, cands: list[str], k: int) -> list[str]:
        seen.append(list(cands))
        return sorted(cands)[:k]

    stack = x1.StrongStack(
        ids, lambda i: i, dense_scores=scorer, splade_scores=scorer, colbert_scores=scorer, rerank=rerank
    )
    arms = stack.arms("question", ["p29", "p28"], k=10)
    assert set(arms) - {"_candidates"} == {
        "splade",
        "colbert",
        "hybrid3",
        "hybrid3-rerank",
        "chain-rerank",
        "union-rerank",
        "fusion",
    }
    assert all(len(v) == 10 and len(set(v)) == 10 for n, v in arms.items() if n != "_candidates")
    # union-rerank is the last reranker call; the graph's candidates must reach it.
    assert {"p29", "p28"} <= set(seen[-1])


# ---------------------------------------------------------------- X3


@pytest.fixture
def graph() -> relational.Graph:
    docs = frozenset
    return relational.Graph(
        types={"nginx": "Software", "selinux": "Software", "502 bad gateway": "ErrorOrSymptom", "php-fpm": "Software"},
        entity_docs={
            "nginx": docs({"q1", "q2"}),
            "selinux": docs({"q2"}),
            "502 bad gateway": docs({"q1"}),
            "php-fpm": docs({"q3"}),
        },
        relations=[
            ("nginx", "causes", "502 bad gateway", docs({"q1"})),
            ("php-fpm", "depends_on", "selinux", docs({"q3"})),
            ("nginx", "depends_on", "selinux", docs({"q2"})),
        ],
    )


def test_name_matching_is_lenient_but_not_loose() -> None:
    assert relational.names_match("PHP-FPM", "php-fpm")
    assert relational.names_match("nginx", "nginx reverse proxy")
    assert not relational.names_match("nginx", "apache")


def test_set_and_count_scores() -> None:
    assert relational.set_scores(["a", "b"], ["a", "c"], str.__eq__) == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert relational.count_score(9, 10)["within_20pct"] == 1.0
    assert relational.count_score(None, 10)["within_20pct"] == 0.0


def test_templates_read_the_sqlite_graph(graph: relational.Graph) -> None:
    db = relational.sqlite_graph(graph)
    count = relational.Question("c", "count", "?", ("nginx",), None, 2)
    co = relational.Question("m", "co_mention", "?", ("nginx", "selinux"), None, ["q2"])
    rel = relational.Question("r", "relation", "?", ("nginx",), "causes", ["502 bad gateway"])
    path = relational.Question("p", "path", "?", ("nginx", "php-fpm"), None, ["selinux"])
    assert relational.template_answer(count, db) == 2
    assert relational.template_answer(co, db) == ["q2"]
    assert relational.template_answer(rel, db) == ["502 bad gateway"]
    assert relational.template_answer(path, db) == ["selinux"]


def test_lexical_docs_whole_word() -> None:
    texts = {"q1": "Restart nginx now", "q2": "nginx-extras package", "q3": "NGINX."}
    assert relational.lexical_docs("nginx", texts) == {"q1", "q3"}


def test_sql_guard_and_extraction() -> None:
    db = relational.sqlite_graph(relational.Graph({}, {}, []))
    assert relational.run_sql(db, "DROP TABLE entities")[1] == "is not a SELECT query"
    assert relational.extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    rows, error = relational.run_sql(db, "SELECT COUNT(*) FROM entities")
    assert error is None and relational.rows_to_answer("count", rows) == 0


def test_text2sql_repairs_an_empty_result(graph: relational.Graph) -> None:
    db = relational.sqlite_graph(graph)
    replies = iter(
        ["SELECT doc_id FROM mentions WHERE entity = 'NGINX'", "SELECT doc_id FROM mentions WHERE entity = 'nginx'"]
    )
    q = relational.Question("m", "co_mention", "Which posts mention nginx?", ("nginx",), None, ["q1", "q2"])
    answer, trace = relational.text2sql_answer(q, db, lambda _p: next(replies), ["nginx"], repair=True)
    assert answer == ["q1", "q2"] and trace["empty"] and trace["repaired"]


def test_parse_and_reduce() -> None:
    assert relational.parse_read("count", "About 12 posts") == 12
    assert relational.parse_read("co_mention", '{"posts": ["q1", "x", "q2"]}') == ["q1", "q2"]
    q = relational.Question("p", "path", "?", ("a", "b"), None, [])
    replies = {
        "q1": '{"related_to_a": ["selinux"], "related_to_b": []}',
        "q2": '{"related_to_a": [], "related_to_b": ["SELinux"]}',
    }
    assert relational.reduce_map(q, replies) == ["selinux"]


def test_generate_is_deterministic() -> None:
    names = [f"tool{i}" for i in range(40)]
    types = dict.fromkeys(names, "Software")
    entity_docs = {n: frozenset(f"q{j}" for j in range(i % 7 + 6)) for i, n in enumerate(names)}
    texts = {f"q{j}": " ".join(names[: (j % 10) + 5]) for j in range(60)}
    graph = relational.Graph(types, entity_docs, [])
    first = [q.to_dict() for q in relational.generate(graph, texts, per_family=3)]
    assert first == [q.to_dict() for q in relational.generate(graph, texts, per_family=3)]
    assert all(q["family"] in relational.FAMILIES for q in first)


# ---------------------------------------------------------------- X2


def test_candidate_names_and_relational_classifier() -> None:
    names = readiness.candidate_names("The Apache HTTP Server reads /etc/hosts and sysctl.conf on boot.")
    assert {"Apache HTTP Server", "/etc/hosts", "sysctl.conf"} <= names
    assert readiness.is_relational("How many posts mention nginx?")
    assert not readiness.is_relational("Why does nginx return 502?")


def test_graph_readiness_statistics() -> None:
    docs = {
        "d1": "Nginx proxies to Gunicorn.",
        "d2": "Gunicorn runs Django.",
        "d3": "Postfix relays mail.",
    }
    report = readiness.graph_readiness(
        docs,
        names=["nginx", "gunicorn", "django", "postfix"],
        questions={"q": "What framework does the app server behind Nginx run?"},
        gold={"q": ["d1", "d2"]},
        search=lambda _q, _k: ["d1", "d3"],
        hub_share=1.0,
    )
    assert report.recurring_name_share == 0.25
    assert report.linked_document_share == round(2 / 3, 4)
    assert report.bridge_question_share == 1.0
    assert report.dense_all_supporting_at_k == 0.0
    assert len(report.reading()) == 4
