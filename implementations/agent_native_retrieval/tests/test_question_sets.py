"""The question-set helpers and builders, offline, with scripted models."""

from __future__ import annotations

import json

import build_near_miss_nulls as nulls_build
import build_paraphrase_control as para_build
import build_serverfault_errors as sf_build
import pytest
import question_sets as qs
from aieng.advanced_rag.corpora import musique
from aieng.advanced_rag.corpora.serverfault import Document as PostDocument
from aieng.advanced_rag.evaluation.synthetic import Document, Model, ProbeIndex
from aieng.advanced_rag.retrieval.embedders import HashEmbedder


def test_capitalized_spans_drop_leading_function_words():
    spans = qs.capitalized_spans("The Battle of Hastings was fought by William the Conqueror.")
    assert spans == ["Battle of Hastings", "William the Conqueror"]


def test_a_rare_word_inside_a_span_leaks_and_a_common_one_does_not():
    filler = [f"passage {i} about a county" for i in range(300)]
    lexicon = qs.Lexicon.build(filler + ["César Gaytan was born in Jalisco County."])
    gold = ["César Gaytan was born in Jalisco County."]
    assert qs.leaked_spans("Where was Gaytan born?", gold, lexicon) == ["Gaytan"]
    assert qs.leaked_spans("Which county was he born in?", gold, lexicon) == []
    assert qs.leaked_spans("Where in jalisco county?", gold, lexicon) == ["Jalisco", "Jalisco County"]


def test_without_a_lexicon_every_word_of_a_span_counts():
    assert qs.leaked_spans("Which county?", ["Jalisco County is large."]) == ["County"]


def test_a_sentence_initial_ordinary_word_is_not_a_name():
    lexicon = qs.Lexicon.build(["located here", "located there", "Located in Paris."])
    assert qs.leaked_spans("Where is it located?", ["Located in Paris."], lexicon) == []
    assert qs.leaked_spans("Is it in paris?", ["Located in Paris."], lexicon) == ["Paris"]


def test_new_terms_are_the_spans_and_years_an_edit_introduced():
    assert qs.new_terms("Who directed the 1994 film set in Lisbon?", "Who directed the 1993 film set in Lisbon?") == [
        "1994"
    ]


def test_grep_counts_case_insensitive_substrings():
    corpus = ["alpha beta", "Beta gamma", "delta"]
    lower = [t.lower() for t in corpus]
    assert qs.grep_count(lower, "BETA") == 2
    assert qs.grep_hits(lower, "gamma") == [1]
    assert qs.grep_count(lower, " ") == 0


def test_token_f1_takes_the_best_reference():
    assert qs.token_f1("the Mississippi River", ["Mississippi River", "Big Muddy"]) == 1.0
    assert qs.token_f1("nothing", ["Mississippi"]) == 0.0


def passage(i, title, text):
    return musique.Passage(musique.passage_id(i), title, text)


def scripted(replies):
    """Return a model that answers by the first matching marker in the prompt."""

    def complete(system, prompt):
        for marker, reply in replies:
            if marker in system or marker in prompt:
                return reply(prompt) if callable(reply) else reply
        return "{}"

    return Model("stub", complete)


PASSAGES = [
    passage(0, "Gaytan", "César Gaytan is a boxer born in Jalisco."),
    passage(1, "Jalisco", "Jalisco is a state of Mexico."),
    passage(2, "Filler", "Nothing relevant here."),
]

QUESTION = musique.MusiqueQuestion(
    question_id="2hop__1_2",
    question="In which country is the state where César Gaytan was born?",
    answer="Mexico",
    answer_aliases=(),
    hops=2,
    supporting_ids=(PASSAGES[0].passage_id, PASSAGES[1].passage_id),
    decomposition=(
        musique.SubQuestion("Where was César Gaytan born?", "Jalisco", PASSAGES[0].passage_id),
        musique.SubQuestion("Which country is Jalisco in?", "Mexico", PASSAGES[1].passage_id),
    ),
)


def rewrite_kwargs(generator, checker):
    return {
        "by_id": {p.passage_id: p for p in PASSAGES},
        "lexicon": qs.Lexicon.build(f"{p.title}\n{p.text}" for p in PASSAGES),
        "corpus_lower": [f"{p.title}\n{p.text}".lower() for p in PASSAGES],
        "generator": generator,
        "checker": checker,
        "attempts": 3,
    }


CHECKER = scripted(
    [
        ("answer questions strictly", '{"answerable": true, "answer": "Mexico"}'),
        ("grade answers", '{"verdict": "match", "reason": "same"}'),
    ]
)


def test_a_leak_free_rewrite_that_round_trips_is_accepted():
    generator = scripted([("reword", '{"question": "In which country is the birthplace state of the boxer?"}')])
    record = para_build.rewrite_one(QUESTION, **rewrite_kwargs(generator, CHECKER))
    assert record["accepted"] and record["leaked_before"]
    assert record["first_hop_grep_reaches_gold_before"] is True
    assert record["first_hop_grep_reaches_gold_after"] is False


def test_a_rewrite_that_keeps_a_name_is_retried_with_the_context_prompt():
    seen = []

    def obfuscated(prompt):
        seen.append(prompt)
        return '{"question": "In which country is the state where the boxer was born?"}'

    generator = scripted(
        [
            ("cannot be answered by", obfuscated),
            ("reword", '{"question": "Where is Gaytan from, by country?"}'),
        ]
    )
    record = para_build.rewrite_one(QUESTION, **rewrite_kwargs(generator, CHECKER))
    assert record["attempts"] == 2 and record["accepted"]
    assert "Jalisco" in seen[0], "the context prompt bans every gold name"


def test_a_rewrite_that_points_at_a_document_is_rejected():
    generator = scripted([("reword", '{"question": "Which country is in the first document?"}')])
    record = para_build.rewrite_one(QUESTION, **rewrite_kwargs(generator, CHECKER))
    assert not record["accepted"] and record["stage"] == "rewrite"


def test_a_rewrite_the_checker_cannot_answer_is_rejected():
    generator = scripted([("reword", '{"question": "Which country is it?"}')])
    checker = scripted([("answer questions strictly", '{"answerable": false, "answer": ""}')])
    record = para_build.rewrite_one(QUESTION, **rewrite_kwargs(generator, checker))
    assert not record["accepted"] and record["verdict"] == "unanswerable"


def test_the_candidates_start_with_the_shared_four_hop_questions(monkeypatch):
    def question(qid, hops):
        return musique.MusiqueQuestion(qid, "q", "a", (), hops, (), ())

    pool = [question(f"{h}hop__{n}", h) for h, count in ((2, 2), (3, 3), (4, 4)) for n in range(count)]
    shared = [pool[-1], pool[3]]
    monkeypatch.setattr(musique, "stratified_test_set", lambda questions: shared)
    ordered = [q.question_id for q in para_build.candidates(pool)]
    assert ordered == ["4hop__3", "4hop__0", "4hop__1", "4hop__2", "3hop__1", "3hop__0", "3hop__2"]


def null_corpus():
    documents = [Document(p.passage_id, f"{p.title}\n{p.text}", p.title) for p in PASSAGES]
    probe = ProbeIndex(documents, HashEmbedder())
    return nulls_build.Corpus(PASSAGES, PASSAGES, probe)


def test_a_near_miss_the_corpus_does_not_answer_is_kept_with_its_grep_counts():
    generator = scripted(
        [
            (
                "realistic questions",
                '{"question": "In which country is the state where César Gaytan died in 1999?", "changed": "born to died"}',
            )
        ]
    )
    checker = scripted([("collection answers", '{"answered": "no", "passage": null, "reason": "none"}')])
    record = nulls_build.build_one(
        QUESTION,
        by_id={p.passage_id: p for p in PASSAGES},
        corpus=null_corpus(),
        generator=generator,
        checker=checker,
    )
    assert record["accepted"]
    assert record["new_terms"] == [{"term": "1999", "hits_11656": 0, "hits_139416": 0}]
    assert record["grep_zero_hits"] is True


def test_a_near_miss_the_corpus_answers_is_rejected():
    generator = scripted(
        [("realistic questions", '{"question": "Which state was César Gaytan born in?", "changed": "asks the state"}')]
    )
    checker = scripted([("collection answers", '{"answered": "yes", "passage": 1, "reason": "it says"}')])
    record = nulls_build.build_one(
        QUESTION,
        by_id={p.passage_id: p for p in PASSAGES},
        corpus=null_corpus(),
        generator=generator,
        checker=checker,
    )
    assert not record["accepted"] and record["grep_zero_hits"] is None


def post(doc_id, title, question, answer="Restart it."):
    return PostDocument(
        doc_id=doc_id,
        question_id=int(doc_id[1:]),
        title=title,
        tags=("linux",),
        question_text=question,
        answer_text=answer,
        score=1,
        link=f"https://serverfault.com/q/{doc_id[1:]}",
        licence="CC BY-SA 4.0",
    )


POSTS = [
    post("q1", "Apache AH00558 at start", "apache2: Could not reliably determine the server name\nhelp"),
    post("q2", "Unrelated", "Nothing to see."),
    post("q3", "Samba NT_STATUS_ACCESS_DENIED", "smbclient says NT_STATUS_ACCESS_DENIED"),
]


def test_an_error_line_is_cut_at_its_clause_and_prose_is_skipped():
    assert sf_build.error_line_key("[Tue] apache2: Could not reliably determine the name") == (
        "Could not reliably determine the name",
        15,
    )
    assert sf_build.error_line_key("I tried it and it failed with nothing useful at all") is None


def test_extraction_finds_codes_identifiers_and_error_lines_with_offsets():
    found = {(c.key, c.kind, c.field) for c in sf_build.extract(POSTS)}
    assert ("AH00558", "code", "title") in found
    assert ("NT_STATUS_ACCESS_DENIED", "identifier", "question") in found
    assert ("Could not reliably determine the server name", "message", "question") in found
    for candidate in sf_build.extract(POSTS):
        document = next(p for p in POSTS if p.doc_id == candidate.doc_id)
        text = document.title if candidate.field == "title" else document.question_text
        assert text[candidate.start : candidate.start + len(candidate.key)] == candidate.key


def test_holders_require_the_key_in_a_title_or_question():
    assert sf_build.holders("AH00558", POSTS) == (["q1"], True)
    assert sf_build.holders("Restart it", POSTS) == (["q1", "q2", "q3"], False)


def committed_row(key, field, start, gold):
    return {
        "query_id": "sf-err-001",
        "kind": "code",
        "key_sha256": qs.sha256_text(key),
        "key_location": {"doc_id": gold[0], "field": field, "start": start, "end": start + len(key)},
        "gold_doc_ids": gold,
    }


def test_the_serverfault_set_materialises_from_the_slice(tmp_path):
    path = tmp_path / "set.jsonl"
    path.write_text(json.dumps(committed_row("AH00558", "title", 7, ["q1"])) + "\n")
    [question] = qs.load_serverfault_questions(POSTS, path)
    assert question.key == "AH00558"
    assert question.question == "My server reports AH00558. What does it mean, and how do I resolve it?"


def test_a_moved_key_or_a_new_holder_fails_the_load(tmp_path):
    path = tmp_path / "set.jsonl"
    path.write_text(json.dumps(committed_row("AH00558", "title", 6, ["q1"])) + "\n")
    with pytest.raises(ValueError, match="changed"):
        qs.load_serverfault_questions(POSTS, path)
    extra = [*POSTS, post("q4", "Also AH00558", "same")]
    path.write_text(json.dumps(committed_row("AH00558", "title", 7, ["q1"])) + "\n")
    with pytest.raises(ValueError, match="occurs in"):
        qs.load_serverfault_questions(extra, path)


def test_a_title_contributes_only_a_quoted_error():
    title = 'Unable to ssh to GCE: "Permission denied (publickey) for user"'
    assert sf_build.error_line_key(title, quoted_only=True) == (
        "Permission denied (publickey) for user",
        23,
    )
    assert sf_build.error_line_key("Unable to reach the GCE host at all", quoted_only=True) is None


def test_the_committed_serverfault_rows_hold_no_key_text():
    rows = qs.load_jsonl(qs.SERVERFAULT_ERRORS)
    assert len(rows) == 30
    for row in rows:
        assert set(row) >= {"key_sha256", "key_location", "gold_doc_ids", "checker"}
        assert "{key}" in row["template"] and row["template"] == qs.TEMPLATES[row["kind"]]
        assert row["key_location"]["doc_id"] == row["gold_doc_ids"][0]
        assert len(row["gold_links"]) == len(row["gold_doc_ids"]) == row["hits_in_slice"]
        assert not any(isinstance(v, str) and len(v) > 80 for v in row.values() if v != row["template"])


def test_the_committed_paraphrase_control_is_leak_free_and_checked():
    rows = qs.load_jsonl(qs.PARAPHRASE_CONTROL)
    assert len(rows) == 50 and {r["hops"] for r in rows} <= {3, 4}
    assert all(r["verdict"] == "match" and r["question"] for r in rows)
    assert all(r["generator"] == "Qwen3_8-27B" and r["checker"] == "Qwen3_8-Flash-Next" for r in rows)
    assert len({r["source_question_id"] for r in rows}) == 50


def test_the_committed_near_miss_nulls_are_checked_nulls_with_grep_counts():
    rows = qs.load_jsonl(qs.NEAR_MISS_NULLS)
    assert len(rows) == 50
    assert all(not r["is_answerable"] and r["null_check"]["answered"] == "no" for r in rows)
    for r in rows:
        assert (r["grep_zero_hits"] is None) == (not r["new_terms"])
        assert all(t["hits_139416"] >= t["hits_11656"] for t in r["new_terms"])
