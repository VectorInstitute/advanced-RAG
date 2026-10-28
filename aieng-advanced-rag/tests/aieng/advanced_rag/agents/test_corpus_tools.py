"""The corpus tool server: grep backends, search, open, the guards, and the trace."""

from __future__ import annotations

import shutil
from dataclasses import dataclass

import numpy as np
import pytest

from aieng.advanced_rag.agents.corpus_tools import (
    STANDARD_ARMS,
    ArmConfig,
    Corpus,
    CorpusDocument,
    CorpusToolServer,
    DenseSearch,
    RegexGrep,
    RipgrepGrep,
    RipgrepUnavailableError,
    materialize,
)
from aieng.advanced_rag.retrieval.interface import ChunkStore


@dataclass(frozen=True)
class FakePassage:
    passage_id: str
    title: str
    text: str


PASSAGES = [
    FakePassage(
        "p0",
        "Lothair II",
        "Lothair II was King of Lotharingia. His mother was Ermengarde of Tours.",
    ),
    FakePassage(
        "p1",
        "Ermengarde of Tours",
        "Ermengarde of Tours died on 20 March 851 at Prum Abbey.",
    ),
    FakePassage("p2", "Prum Abbey", "Prum Abbey is a Benedictine abbey in the Eifel."),
    FakePassage("p3", "Eifel", "The Eifel is a low mountain range in western Germany."),
]


@pytest.fixture
def corpus() -> Corpus:
    return Corpus.from_passages(PASSAGES, "fake")


class KeywordEmbedder:
    """Embeds a text as counts of a fixed vocabulary, so search is predictable."""

    VOCAB = ("king", "mother", "died", "abbey", "mountain")

    def encode(self, texts):
        return [[t.lower().count(w) for w in self.VOCAB] for t in texts]

    def encode_queries(self, texts):
        return self.encode(texts)


def test_corpus_puts_titles_on_the_first_line(corpus: Corpus) -> None:
    assert corpus.get("p1").text.splitlines()[0] == "Ermengarde of Tours"
    assert len(corpus) == 4
    assert "p3" in corpus


def test_corpus_rejects_duplicate_identifiers() -> None:
    with pytest.raises(ValueError, match="share an identifier"):
        Corpus([CorpusDocument("a", "a", "x"), CorpusDocument("a", "a", "y")])


def test_regex_grep_counts_documents_and_matches(corpus: Corpus) -> None:
    found = RegexGrep(corpus).grep("ermengarde", max_results=10)
    assert [m.doc_id for m in found.matches] == ["p0", "p1"]
    assert found.total_hits == 2
    assert found.total_matches == 3
    assert found.matches[0].line == 2
    assert "Ermengarde" in found.matches[0].preview


def test_regex_grep_reads_an_invalid_pattern_literally(corpus: Corpus) -> None:
    found = RegexGrep(corpus).grep("Lothair (II", max_results=5)
    assert found.total_hits == 0
    assert "literally" in found.note


def test_regex_grep_never_matches_across_documents(corpus: Corpus) -> None:
    assert RegexGrep(corpus).grep(r"Lotharingia[\s\S]*Prum").total_hits == 1
    assert RegexGrep(corpus).grep(r"Eifel\.\s+Eifel").total_hits == 0


def test_regex_grep_stops_at_the_scan_cap(corpus: Corpus) -> None:
    found = RegexGrep(corpus, scan_cap=2).grep("the|of|is", max_results=10)
    assert found.capped
    assert found.total_hits == 2


def test_regex_grep_within_restricts_documents(corpus: Corpus) -> None:
    found = RegexGrep(corpus).grep("abbey", within=["p2"])
    assert [m.doc_id for m in found.matches] == ["p2"]


def test_chunk_store_documents_are_rebuilt() -> None:
    @dataclass(frozen=True)
    class Chunk:
        chunk_id: str
        doc_id: str
        text: str
        start: int
        end: int

    store = ChunkStore(
        [Chunk("c0", "d", "hello wor", 0, 9), Chunk("c1", "d", "world", 6, 11)]
    )
    corpus = Corpus.from_chunk_store(store)
    assert corpus.get("d").text == "hello world"
    assert RegexGrep(corpus).grep("world").total_matches == 1


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")
def test_ripgrep_matches_the_regex_backend(corpus: Corpus, tmp_path) -> None:
    mapping = materialize(corpus, tmp_path)
    rg = RipgrepGrep(tmp_path, doc_ids=mapping, corpus=corpus)
    ours = RegexGrep(corpus).grep("abbey")
    theirs = rg.grep("abbey")
    assert [m.doc_id for m in theirs.matches] == [m.doc_id for m in ours.matches]
    assert theirs.total_matches == ours.total_matches
    assert rg.grep("Lothair (II").note


def test_ripgrep_names_the_fallback_when_missing(tmp_path) -> None:
    with pytest.raises(RipgrepUnavailableError, match="RegexGrep"):
        RipgrepGrep(tmp_path, rg="rg-that-does-not-exist")


def test_materialize_round_trips_identifiers(tmp_path) -> None:
    corpus = Corpus(
        [CorpusDocument("musique::00001", "t", "x"), CorpusDocument("a/../b", "t", "y")]
    )
    mapping = materialize(corpus, tmp_path)
    assert sorted(mapping.values()) == ["a/../b", "musique::00001"]
    for rel in mapping:
        assert (tmp_path / rel).resolve().is_relative_to(tmp_path.resolve())


def test_dense_search_ranks_by_similarity(corpus: Corpus) -> None:
    search = DenseSearch(corpus, KeywordEmbedder())
    assert search("mountain", 1)[0][0] == "p3"


def test_dense_search_checks_the_matrix_shape(corpus: Corpus) -> None:
    with pytest.raises(ValueError, match="embedding rows"):
        DenseSearch(corpus, KeywordEmbedder(), matrix=np.zeros((2, 5)))


def test_c5_offers_grep_and_open_only(corpus: Corpus) -> None:
    session = CorpusToolServer(corpus).session(STANDARD_ARMS["C5"])
    assert [tool.__name__ for tool in session.tools] == ["grep", "open"]


def test_an_arm_with_search_needs_a_backend(corpus: Corpus) -> None:
    with pytest.raises(ValueError, match="no search backend"):
        CorpusToolServer(corpus).session(STANDARD_ARMS["C0"])


def test_arm_config_rejects_unknown_tools() -> None:
    with pytest.raises(ValueError, match="unknown tools"):
        ArmConfig("bad", ("search", "teleport"))


def test_the_trace_separates_seen_from_read(corpus: Corpus) -> None:
    session = CorpusToolServer(corpus).session(STANDARD_ARMS["C5"])
    session.call("grep", {"pattern": "Ermengarde"})
    session.call("open", {"doc_id": "p1"})
    assert session.trace.seen_ids == ["p0", "p1"]
    assert session.trace.read_ids == ["p1"]
    assert session.trace.seen_not_read(["p0", "p1", "p2"]) == ["p0"]
    first = session.trace.events[0]
    assert first.arguments == {"pattern": "Ermengarde", "ignore_case": True}
    assert first.total_hits == 2
    assert len(first.result_digest) == 16


def test_search_counts_whole_hits_as_read(corpus: Corpus) -> None:
    server = CorpusToolServer(corpus, search=DenseSearch(corpus, KeywordEmbedder()))
    session = server.session(STANDARD_ARMS["C0"].with_(search_k=2))
    result = session.call("search", {"query": "died abbey"})
    assert len(result["results"]) == 2
    assert session.trace.read_ids == session.trace.seen_ids


def test_search_previews_long_hits_without_reading_them(corpus: Corpus) -> None:
    server = CorpusToolServer(corpus, search=DenseSearch(corpus, KeywordEmbedder()))
    session = server.session(STANDARD_ARMS["C0"].with_(search_preview_tokens=2))
    session.call("search", {"query": "mountain"})
    assert session.trace.read_ids == []
    assert session.trace.seen_ids


def test_narrow_the_query_replaces_previews(corpus: Corpus) -> None:
    session = CorpusToolServer(corpus).session(
        STANDARD_ARMS["C5"].with_(narrow_above=1)
    )
    result = session.call("grep", {"pattern": "of"})
    assert result["matches"] == []
    assert "narrow" in result["note"]
    assert result["total_hits"] > 1


def test_open_caps_output_and_names_the_next_line() -> None:
    corpus = Corpus(
        [
            CorpusDocument(
                "long", "long", "\n".join(f"line {i} " * 20 for i in range(50))
            )
        ]
    )
    session = CorpusToolServer(corpus).session(
        STANDARD_ARMS["C5"].with_(output_cap_tokens=100)
    )
    result = session.call("open", {"doc_id": "long"})
    assert "start_line=" in result["note"]
    assert len(result["text"]) <= 100 * 4 + 200


def test_open_twice_gets_the_already_read_notice(corpus: Corpus) -> None:
    session = CorpusToolServer(corpus).session(STANDARD_ARMS["C5"])
    session.call("open", {"doc_id": "p2"})
    again = session.call("open", {"doc_id": "p2"})
    assert "already opened" in again["note"]
    assert "text" not in again


def test_open_reports_an_unknown_document(corpus: Corpus) -> None:
    session = CorpusToolServer(corpus).session(STANDARD_ARMS["C5"])
    assert "no document" in session.call("open", {"doc_id": "nope"})["error"]
    assert session.trace.events[0].error


def test_bad_arguments_come_back_as_an_error(corpus: Corpus) -> None:
    session = CorpusToolServer(corpus).session(STANDARD_ARMS["C5"])
    assert "bad arguments" in session.call("grep", {"query": "x"})["error"]
    assert "no tool" in session.call("search", {"query": "x"})["error"]
    assert len(session.trace.events) == 2


def test_searched_scope_greps_only_search_results(corpus: Corpus) -> None:
    server = CorpusToolServer(corpus, search=DenseSearch(corpus, KeywordEmbedder()))
    session = server.session(STANDARD_ARMS["C7"].with_(search_k=1))
    assert "search first" in session.call("grep", {"pattern": "abbey"})["note"]
    session.call("search", {"query": "mountain"})
    found = session.call("grep", {"pattern": "e"})
    assert [m["doc_id"] for m in found["matches"]] == ["p3"]
