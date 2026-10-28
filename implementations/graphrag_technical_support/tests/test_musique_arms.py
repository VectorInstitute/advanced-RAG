"""The retrieval arms, with a fake encoder and a fake model so nothing downloads."""

from __future__ import annotations

import asyncio

import networkx as nx
import numpy as np
import pytest
from aieng.advanced_rag.corpora.musique import Passage
from aieng.advanced_rag.graphrag.evidence import Evidence
from aieng.advanced_rag.provenance import Provenance
from musique_arms import (
    DenseArm,
    HybridArm,
    IterativeDenseArm,
    LightRagArm,
    SyncEvidenceArm,
    answer,
    interleave,
    parse_steps,
    passage_id_map,
    substitute,
)


WORDS = ["messi", "barcelona", "maradona", "napoli", "signed", "goal", "paris", "river"]


class FakeEncoder:
    """Encode a text as counts of known words, so similarity is word overlap."""

    def __init__(self):
        self.encoded = 0

    def _vector(self, text):
        lowered = text.lower()
        return [float(lowered.count(word)) for word in WORDS]

    def encode(self, texts):
        self.encoded += len(texts)
        return [self._vector(t) for t in texts]

    def encode_queries(self, texts):
        return [self._vector(t) for t in texts]


PASSAGES = [
    Passage("musique::00000", "Messi", "Messi scored a goal for Barcelona."),
    Passage("musique::00001", "Maradona", "Maradona signed for Napoli."),
    Passage("musique::00002", "Paris", "Paris is on the river."),
    Passage("musique::00003", "Napoli", "Napoli is in Italy. Maradona signed here."),
]


def run(coro):
    return asyncio.run(coro)


def test_dense_ranks_by_similarity_and_breaks_ties_by_position(tmp_path):
    arm = DenseArm(PASSAGES, FakeEncoder(), cache=tmp_path / "e.npy")
    assert run(arm.rank("Who signed for Napoli", 2))[0] in {"musique::00001", "musique::00003"}
    ranked = run(arm.rank("zzz nothing", 4))
    assert ranked == ["musique::00000", "musique::00001", "musique::00002", "musique::00003"]


def test_dense_reads_its_embeddings_from_disk_the_second_time(tmp_path):
    first = FakeEncoder()
    DenseArm(PASSAGES, first, cache=tmp_path / "e.npy")
    assert first.encoded == 4 and (tmp_path / "e.npy").is_file()
    second = FakeEncoder()
    arm = DenseArm(PASSAGES, second, cache=tmp_path / "e.npy")
    assert second.encoded == 0
    assert run(arm.rank("messi goal", 1)) == ["musique::00000"]
    assert np.isfinite(np.load(tmp_path / "e.npy")).all()


def test_steps_parse_from_fenced_or_wrapped_json_and_fall_back_to_the_question():
    assert parse_steps('{"steps": ["a", "b"]}', "q") == ["a", "b"]
    assert parse_steps('```json\n{"steps": ["a"]}\n```', "q") == ["a"]
    assert parse_steps('Sure: {"steps": ["a", "b"]}', "q") == ["a", "b"]
    assert parse_steps("no json", "the question") == ["the question"]
    assert parse_steps('{"steps": []}', "q") == ["q"]
    assert len(parse_steps('{"steps": ["1","2","3","4","5","6"]}', "q")) == 4


def test_a_later_step_names_an_earlier_answer():
    assert substitute("When was #1 signed?", ["Maradona"]) == "When was Maradona signed?"
    assert substitute("About #2", ["only one"]) == "About #2"
    assert substitute("About #1 and #2", ["x", "y"]) == "About x and y"


def test_interleaving_alternates_between_steps_and_skips_repeats():
    assert interleave([["a", "b", "c"], ["x", "a", "y"]], 5) == ["a", "x", "b", "c", "y"]
    assert interleave([["a"], []], 3) == ["a"]
    assert interleave([], 3) == []


def test_the_iterative_arm_retrieves_each_step_with_the_previous_answer(tmp_path):
    asked = []

    class RecordingDense(DenseArm):
        async def rank(self, question, k):
            asked.append(question)
            return await super().rank(question, k)

    dense = RecordingDense(PASSAGES, FakeEncoder())
    replies = iter(
        [
            '{"steps": ["Which club did Messi play for?", "Where is #1?"]}',
            "Barcelona",
            "Spain",
        ]
    )
    texts = {p.passage_id: p.text for p in PASSAGES}
    arm = IterativeDenseArm(dense, lambda prompt: next(replies), texts.__getitem__, step_k=2)
    ranked = run(arm.rank("Where is the club Messi played for?", 4))
    assert asked == ["Which club did Messi play for?", "Where is Barcelona?"]
    assert ranked and len(ranked) == len(set(ranked)) <= 4


def test_the_hybrid_fuses_two_rankings():
    class Fixed:
        def __init__(self, name, ids):
            self.name, self._ids = name, ids

        async def rank(self, question, k):
            return self._ids[:k]

    fused = HybridArm(Fixed("a", ["x", "y", "z"]), Fixed("b", ["y", "x", "w"]))
    assert run(fused.rank("q", 3))[:2] == ["x", "y"] or run(fused.rank("q", 3))[:2] == ["y", "x"]
    top = run(fused.rank("q", 4))
    assert set(top) == {"x", "y", "z", "w"}


def test_the_hybrid_respects_a_weight_of_zero():
    class Fixed:
        def __init__(self, ids):
            self.name, self._ids = "f", ids

        async def rank(self, question, k):
            return self._ids[:k]

    arm = HybridArm(Fixed(["a", "b"]), Fixed(["z"]), weights=(1.0, 0.0))
    assert run(arm.rank("q", 3)) == ["a", "b"]


def test_a_synchronous_retriever_becomes_an_arm():
    class Retriever:
        def retrieve(self, question, k):
            return [Evidence("t", Provenance(doc_id=f"d{i}", chunk_id=f"p{i}")) for i in range(k)]

    arm = SyncEvidenceArm("hipporag2", Retriever())
    assert arm.name == "hipporag2" and run(arm.rank("q", 3)) == ["p0", "p1", "p2"]


def test_lightrag_is_asked_for_its_default_depth_and_cut_to_k():
    class Fake:
        def __init__(self):
            self.asked = None

        async def aretrieve(self, question, k):
            self.asked = k
            return [Evidence("t", Provenance(doc_id=f"p{i}")) for i in range(k)]

    fake = Fake()
    ranked = run(LightRagArm(fake).rank("q", 10))
    assert fake.asked == 40 and ranked == [f"p{i}" for i in range(10)]


def test_an_answer_is_the_first_line_of_the_models_reply():
    texts = {"p1": "text one", "p2": "text two"}
    seen = []

    def complete(prompt):
        seen.append(prompt)
        return "Diego Maradona\nbecause the passage says so"

    assert run(answer(complete, "Who?", ["p1", "p2"], texts.__getitem__)) == "Diego Maradona"
    assert "[1] text one" in seen[0] and "[2] text two" in seen[0] and "Who?" in seen[0]


def test_an_empty_reply_is_an_empty_answer():
    assert run(answer(lambda p: "  ", "Who?", [], {}.__getitem__)) == ""


@pytest.mark.parametrize("n", [0, 1])
def test_rank_with_a_small_k_returns_that_many(n, tmp_path):
    assert len(run(DenseArm(PASSAGES, FakeEncoder()).rank("messi", n))) == n


def test_a_synchronous_arm_can_map_the_retrievers_ids_back_to_passages():
    class Retriever:
        def retrieve(self, question, k):
            return [Evidence("t", Provenance(doc_id="hashA")), Evidence("t", Provenance(doc_id="hashB"))]

    arm = SyncEvidenceArm("hipporag2", Retriever(), id_map={"hashA": "musique::00001", "hashB": "musique::00002"})
    assert run(arm.rank("q", 2)) == ["musique::00001", "musique::00002"]


def test_an_id_the_map_does_not_hold_is_an_error_not_a_miss():
    class Retriever:
        def retrieve(self, question, k):
            return [Evidence("t", Provenance(doc_id="unknown"))]

    with pytest.raises(KeyError):
        run(SyncEvidenceArm("hipporag2", Retriever(), id_map={}).rank("q", 1))


def test_the_graph_file_maps_node_keys_to_passages_by_text(tmp_path):
    graph = nx.Graph()
    for key, passage in (("h1", PASSAGES[0]), ("h2", PASSAGES[1])):
        graph.add_node(key, type="passage", id=f"{passage.title}\n{passage.text}")
    graph.add_node("e1", type="entity", id="Messi")
    nx.write_graphml(graph, tmp_path / "g.graphml")
    assert passage_id_map(tmp_path / "g.graphml", PASSAGES) == {
        "h1": "musique::00000",
        "h2": "musique::00001",
    }


def test_a_passage_node_that_matches_no_passage_is_an_error(tmp_path):
    graph = nx.Graph()
    graph.add_node("h1", type="passage", id="text nobody indexed")
    nx.write_graphml(graph, tmp_path / "g.graphml")
    with pytest.raises(ValueError, match="match no passage text"):
        passage_id_map(tmp_path / "g.graphml", PASSAGES)
