"""MuSiQue loader: pins, parsing, judgments, and the stratified draw, all offline."""

from __future__ import annotations

import hashlib
import json
import re

import pytest

from aieng.advanced_rag.corpora import musique, sources
from aieng.advanced_rag.corpora.musique import (
    fetch_musique,
    load_musique,
    parse_corpus,
    parse_questions,
    passage_id,
    stratified_test_set,
)
from aieng.advanced_rag.corpora.sources import ChecksumMismatchError, PinnedSource


def passage(i):
    return {"title": f"Title {i}", "text": f"Text of passage {i}."}


def record(qid, supporting, extra=(), decomposition=None):
    """Build a question over passages ``supporting`` plus distractors ``extra``."""
    order = list(extra) + list(supporting)
    paragraphs = [
        {
            "idx": idx,
            "title": passage(i)["title"],
            "paragraph_text": passage(i)["text"],
            "is_supporting": i in supporting,
        }
        for idx, i in enumerate(order)
    ]
    steps = decomposition or [
        {
            "id": n,
            "question": f"step {n}",
            "answer": f"a{n}",
            "paragraph_support_idx": order.index(i),
        }
        for n, i in enumerate(supporting)
    ]
    return {
        "id": qid,
        "question": f"question {qid}",
        "answer": "yes",
        "answer_aliases": [],
        "answerable": True,
        "paragraphs": paragraphs,
        "question_decomposition": steps,
    }


CORPUS = [passage(i) for i in range(12)]


def test_the_pins_name_a_revision_and_never_main():
    assert re.fullmatch(r"[0-9a-f]{40}", musique.MUSIQUE_REVISION)
    for source in (musique.MUSIQUE_QUESTIONS, musique.MUSIQUE_CORPUS):
        assert musique.MUSIQUE_REVISION in source.url
        assert "/main/" not in source.url
        assert re.fullmatch(r"[0-9a-f]{64}", source.sha256)


def test_the_licence_records_the_terms_and_the_packaging_card():
    licence = musique.MUSIQUE_LICENCE
    assert licence.name == "CC BY 4.0"
    assert "states no licence" in licence.notes


def test_the_hop_split_adds_up():
    assert sum(musique.HOP_COUNTS.values()) == 1000
    assert sum(musique.TEST_PER_HOP.values()) == 300


def test_passage_ids_are_stable_and_ordered():
    passages = parse_corpus(CORPUS)
    assert [p.passage_id for p in passages[:2]] == ["musique::00000", "musique::00001"]
    assert passage_id(11655) == "musique::11655"


def test_a_duplicate_passage_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        parse_corpus([passage(1), passage(1)])


def test_a_question_is_judged_against_its_supporting_passages():
    passages = parse_corpus(CORPUS)
    [q] = parse_questions([record("2hop__1_2", [3, 7], extra=[0, 1])], passages)
    assert q.hops == 2
    assert q.supporting_ids == (passage_id(3), passage_id(7))
    assert [s.support_passage_id for s in q.decomposition] == list(q.supporting_ids)
    assert q.kind == "2hop"
    assert q.is_answerable


def test_the_hop_count_must_match_the_marked_paragraphs():
    passages = parse_corpus(CORPUS)
    with pytest.raises(ValueError, match="supporting paragraphs for a 3-hop"):
        parse_questions([record("3hop1__1_2_3", [3, 7])], passages)


def test_a_paragraph_missing_from_the_corpus_is_rejected():
    passages = parse_corpus(CORPUS[:5])
    with pytest.raises(ValueError, match="not in the corpus"):
        parse_questions([record("2hop__1_2", [3, 7])], passages)


def test_a_decomposition_that_disagrees_with_the_marks_is_rejected():
    passages = parse_corpus(CORPUS)
    bad = record(
        "2hop__1_2",
        [3, 7],
        extra=[0],
        decomposition=[
            {"id": 0, "question": "s", "answer": "a", "paragraph_support_idx": 0},
            {"id": 1, "question": "s", "answer": "a", "paragraph_support_idx": 1},
        ],
    )
    with pytest.raises(ValueError, match="disagree"):
        parse_questions([bad], passages)


def many_questions():
    passages = parse_corpus([passage(i) for i in range(40)])
    records = []
    for hops, count in ((2, 30), (3, 20), (4, 12)):
        for n in range(count):
            records.append(record(f"{hops}hop__{n}", list(range(hops))))
    return parse_questions(records, passages)


def test_the_stratified_draw_takes_the_requested_number_per_hop():
    drawn = stratified_test_set(many_questions(), {2: 10, 3: 5, 4: 5})
    assert [q.hops for q in drawn].count(2) == 10
    assert [q.hops for q in drawn].count(3) == 5
    assert [q.hops for q in drawn].count(4) == 5


def test_the_draw_is_reproducible_and_depends_on_the_seed():
    questions = many_questions()
    per_hop = {2: 10, 3: 5, 4: 5}
    first = stratified_test_set(questions, per_hop, seed=1)
    assert stratified_test_set(questions, per_hop, seed=1) == first
    assert stratified_test_set(questions, per_hop, seed=2) != first


def test_asking_for_more_than_a_hop_count_holds_is_rejected():
    with pytest.raises(ValueError, match="only 12 4-hop"):
        stratified_test_set(many_questions(), {2: 1, 3: 1, 4: 13})


@pytest.fixture
def served(monkeypatch, tmp_path):
    corpus_bytes = json.dumps(CORPUS).encode()
    questions_bytes = json.dumps([record("2hop__1_2", [3, 7])]).encode()
    digest = lambda data: hashlib.sha256(data).hexdigest()  # noqa: E731
    pins = {
        "MUSIQUE_QUESTIONS": (questions_bytes, "https://example.invalid/q.json"),
        "MUSIQUE_CORPUS": (corpus_bytes, "https://example.invalid/c.json"),
    }
    payloads = {}
    for name, (data, url) in pins.items():
        monkeypatch.setattr(
            musique,
            name,
            PinnedSource(
                name=name.lower(),
                url=url,
                sha256=digest(data),
                licence=musique.MUSIQUE_LICENCE,
            ),
        )
        payloads[url] = data
    calls = []

    def download(url):
        calls.append(url)
        return payloads[url]

    monkeypatch.setattr(sources, "_download", download)
    return payloads, calls, tmp_path


def test_fetch_verifies_then_serves_from_the_cache_and_loads(served):
    payloads, calls, root = served
    paths = fetch_musique(dest=root)
    passages, questions = load_musique(paths)
    assert len(passages) == 12 and len(questions) == 1
    fetch_musique(dest=root)
    assert len(calls) == 2, "a verified cached file is not downloaded again"


def test_a_changed_upstream_file_fails_the_digest(served):
    payloads, calls, root = served
    payloads["https://example.invalid/c.json"] = b"[]"
    with pytest.raises(ChecksumMismatchError):
        fetch_musique(dest=root)


def test_graph_preserves_original_ids_and_resolves_positional_references():
    raw = record("4hop2__50_10_90_20", [0, 1, 2, 3])
    for step, step_id, question in zip(
        raw["question_decomposition"],
        [50, 10, 90, 20],
        ["root", "another root", "combine #1 and #2 then #1", "combine #1 and #3"],
    ):
        step.update(id=step_id, question=question)
    [q] = parse_questions([raw], parse_corpus(CORPUS))
    assert [node.step_id for node in q.hop_graph.nodes] == [50, 10, 90, 20]
    assert q.hop_graph.edges == ((50, 90), (10, 90), (50, 20), (90, 20))
    assert q.decomposition[2].depends_on == (50, 10)
    assert q.hop_graph.nodes[2].answer == "a2"
    assert q.hop_graph.nodes[2].support_passage_id == passage_id(2)


@pytest.mark.parametrize("reference", ["#0", "#2", "#3", "#99"])
def test_invalid_dependencies_are_rejected(reference):
    raw = record("2hop__1_2", [0, 1])
    raw["question_decomposition"][1]["question"] = f"use {reference}"
    with pytest.raises(ValueError, match="non-earlier step"):
        parse_questions([raw], parse_corpus(CORPUS))


def test_duplicate_step_ids_are_rejected():
    raw = record("2hop__1_2", [0, 1])
    raw["question_decomposition"][1]["id"] = 0
    with pytest.raises(ValueError, match="duplicate decomposition step ID"):
        parse_questions([raw], parse_corpus(CORPUS))


def test_duplicate_paragraph_indices_are_rejected():
    raw = record("2hop__1_2", [0, 1])
    raw["paragraphs"][1]["idx"] = 0
    with pytest.raises(ValueError, match="duplicate paragraph index"):
        parse_questions([raw], parse_corpus(CORPUS))


def test_duplicate_supporting_passages_are_rejected():
    raw = record("2hop__1_2", [0, 1])
    raw["paragraphs"][1].update(
        title=passage(0)["title"], paragraph_text=passage(0)["text"]
    )
    with pytest.raises(ValueError, match="duplicate supporting passage"):
        parse_questions([raw], parse_corpus(CORPUS))


def test_missing_decomposition_support_is_rejected():
    raw = record("2hop__1_2", [0, 1])
    raw["question_decomposition"][1]["paragraph_support_idx"] = 99
    with pytest.raises(ValueError, match="missing supporting paragraph index"):
        parse_questions([raw], parse_corpus(CORPUS))


@pytest.mark.parametrize("length", [0, 1, 3])
def test_decomposition_length_must_match_hops(length):
    raw = record("2hop__1_2", [0, 1])
    raw["question_decomposition"] = (raw["question_decomposition"] * 2)[:length]
    with pytest.raises(ValueError, match="decomposition length"):
        parse_questions([raw], parse_corpus(CORPUS))


def test_duplicate_question_ids_are_rejected():
    raw = record("2hop__1_2", [0, 1])
    with pytest.raises(ValueError, match="duplicate question identifier"):
        parse_questions([raw, raw], parse_corpus(CORPUS))


@pytest.mark.parametrize("qid", ["", "5hop__1_2", "2hop_bad", "2hop__abc"])
def test_invalid_question_ids_are_rejected(qid):
    with pytest.raises(ValueError, match="invalid MuSiQue question identifier"):
        parse_questions([record(qid, [0, 1])], parse_corpus(CORPUS))


def test_unanswerable_questions_are_rejected():
    raw = record("2hop__1_2", [0, 1])
    raw["answerable"] = False
    with pytest.raises(ValueError, match="expected an answerable question"):
        parse_questions([raw], parse_corpus(CORPUS))


@pytest.mark.parametrize("file_field", ["questions", "corpus"])
def test_explicit_local_paths_are_verified_before_parsing(served, file_field):
    _, _, root = served
    paths = fetch_musique(dest=root)
    getattr(paths, file_field).write_bytes(b"not even JSON")
    with pytest.raises(ChecksumMismatchError):
        load_musique(paths)


# ---------------------------------------------------------------------------
# The 139,416-passage pool, built offline from a hand-made release archive.
# ---------------------------------------------------------------------------


def release_line(qid, indices):
    return json.dumps(
        {
            "id": qid,
            "paragraphs": [
                {
                    "idx": n,
                    "title": passage(i)["title"],
                    "paragraph_text": passage(i)["text"],
                }
                for n, i in enumerate(indices)
            ],
        }
    )


def make_release(path, members):
    """Write a zip whose members are JSONL files of the given paragraph indices."""
    import zipfile

    digests = {}
    with zipfile.ZipFile(path, "w") as archive:
        for name, questions in members.items():
            data = "\n".join(
                release_line(f"{name}-{n}", q) for n, q in enumerate(questions)
            ).encode()
            archive.writestr(name, data)
            digests[name] = hashlib.sha256(data).hexdigest()
    return digests


def test_the_release_pins_its_archive_and_all_six_members():
    assert re.fullmatch(r"[0-9a-f]{64}", musique.MUSIQUE_RELEASE.sha256)
    assert musique.MUSIQUE_RELEASE_FILE_ID in musique.MUSIQUE_RELEASE.url
    assert len(musique.POOL_MEMBERS) == 6
    assert all(re.fullmatch(r"[0-9a-f]{64}", d) for d in musique.POOL_MEMBERS.values())
    assert re.fullmatch(r"[0-9a-f]{64}", musique.POOL_SHA256)
    assert (musique.SUBSET_SIZE, musique.POOL_SIZE) == (11_656, 139_416)
    assert passage_id(139_415) == "musique::139415"


def test_the_pool_opens_with_the_subset_and_numbers_the_rest_in_order(tmp_path):
    subset = parse_corpus([passage(4), passage(2)])
    digests = make_release(
        tmp_path / "r.zip",
        {"a.jsonl": [[2, 9, 4], [9, 1]], "b.jsonl": [[1, 7, 2]]},
    )
    pool = musique.build_pool(
        tmp_path / "r.zip", subset, members=digests, expected_size=5
    )
    assert pool[:2] == subset
    assert [p.title for p in pool[2:]] == ["Title 9", "Title 1", "Title 7"]
    assert [p.passage_id for p in pool] == [passage_id(i) for i in range(5)]


def test_the_pool_size_is_checked(tmp_path):
    subset = parse_corpus([passage(1)])
    digests = make_release(tmp_path / "r.zip", {"a.jsonl": [[1, 2]]})
    with pytest.raises(ValueError, match="holds 2 paragraphs, expected 3"):
        musique.build_pool(tmp_path / "r.zip", subset, members=digests, expected_size=3)


def test_a_subset_passage_missing_from_the_release_is_rejected(tmp_path):
    subset = parse_corpus([passage(5)])
    digests = make_release(tmp_path / "r.zip", {"a.jsonl": [[1, 2]]})
    with pytest.raises(ValueError, match="1 packaged passages are not in the release"):
        musique.build_pool(
            tmp_path / "r.zip", subset, members=digests, expected_size=None
        )


def test_a_changed_member_fails_its_digest(tmp_path):
    subset = parse_corpus([passage(1)])
    digests = make_release(tmp_path / "r.zip", {"a.jsonl": [[1, 2]]})
    digests["a.jsonl"] = "0" * 64
    with pytest.raises(ChecksumMismatchError, match="a.jsonl"):
        musique.build_pool(tmp_path / "r.zip", subset, members=digests)


def test_fetch_pool_builds_verifies_and_reuses_the_cache(served, monkeypatch):
    payloads, calls, root = served
    subset = parse_corpus(CORPUS)
    digests = make_release(
        root / "release.zip",
        {"a.jsonl": [list(range(12)) + [20]], "b.jsonl": [[21, 3]]},
    )
    release = (root / "release.zip").read_bytes()
    payloads["https://example.invalid/r.zip"] = release
    monkeypatch.setattr(
        musique,
        "MUSIQUE_RELEASE",
        PinnedSource(
            name="release-fixture.zip",
            url="https://example.invalid/r.zip",
            sha256=hashlib.sha256(release).hexdigest(),
            licence=musique.MUSIQUE_LICENCE,
        ),
    )
    monkeypatch.setattr(musique, "POOL_MEMBERS", digests)
    monkeypatch.setattr(musique, "POOL_SIZE", 14)
    expected = musique.pool_bytes(
        subset
        + [
            musique.Passage(passage_id(12), "Title 20", "Text of passage 20."),
            musique.Passage(passage_id(13), "Title 21", "Text of passage 21."),
        ]
    )
    monkeypatch.setattr(musique, "POOL_SHA256", hashlib.sha256(expected).hexdigest())
    path = musique.fetch_pool(dest=root)
    assert path.read_bytes() == expected
    pool = musique.load_pool(path)
    assert [p.passage_id for p in pool][-1] == passage_id(13)
    before = len(calls)
    musique.fetch_pool(dest=root)
    assert len(calls) == before, "a verified cached pool is not rebuilt"


def test_load_pool_rejects_a_file_of_the_wrong_size(tmp_path):
    path = tmp_path / "pool.json"
    path.write_bytes(musique.pool_bytes(parse_corpus(CORPUS)))
    with pytest.raises(ValueError, match="expected 139416"):
        musique.load_pool(path)
