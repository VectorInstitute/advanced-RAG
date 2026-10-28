"""ServerFault loader: selection, collection, slice digests, and document text."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from aieng.advanced_rag.corpora import serverfault as sf
from aieng.advanced_rag.corpora.sources import ChecksumMismatchError


def row(**attrs: object) -> str:
    inner = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    return f"  <row {inner} />"


POSTS = "\n".join(
    [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<posts>",
        # Question 1: score 50, accepted answer 11, plus a rival answer 12.
        row(
            Id=1,
            PostTypeId=1,
            AcceptedAnswerId=11,
            Score=50,
            Title="Why &amp; how",
            Body="&lt;p&gt;Body one&lt;/p&gt;",
            Tags="&lt;linux&gt;&lt;dns&gt;",
            CreationDate="2020-01-01T00:00:00.000",
            ContentLicense="CC BY-SA 4.0",
            OwnerUserId=7,
            ViewCount=10,
            AnswerCount=2,
        ),
        row(
            Id=11,
            PostTypeId=2,
            ParentId=1,
            Score=9,
            Body="&lt;p&gt;Accepted&lt;/p&gt;",
            CreationDate="2020-01-02T00:00:00.000",
            ContentLicense="CC BY-SA 3.0",
            OwnerDisplayName="Ann",
        ),
        row(
            Id=12,
            PostTypeId=2,
            ParentId=1,
            Score=20,
            Body="Rival",
            CreationDate="2020-01-03T00:00:00.000",
            ContentLicense="CC BY-SA 3.0",
        ),
        # Question 2: same score as 3, lower id, so it sorts first.
        row(
            Id=2,
            PostTypeId=1,
            AcceptedAnswerId=21,
            Score=30,
            Title="Second",
            Body="Body two",
            Tags="|nginx|",
            CreationDate="2019-05-05T00:00:00.000",
            ContentLicense="CC BY-SA 2.5",
        ),
        row(
            Id=21,
            PostTypeId=2,
            ParentId=2,
            Score=1,
            Body="Answer two",
            ContentLicense="CC BY-SA 2.5",
        ),
        row(
            Id=3,
            PostTypeId=1,
            AcceptedAnswerId=31,
            Score=30,
            Title="Third",
            Body="Body three",
            Tags="",
            CreationDate="2021-05-05T00:00:00.000",
        ),
        row(Id=31, PostTypeId=2, ParentId=3, Score=1, Body="Answer three"),
        # No accepted answer: never selected.
        row(Id=4, PostTypeId=1, Score=99, Title="Unanswered", Body="x"),
        # Score below the floor.
        row(Id=5, PostTypeId=1, AcceptedAnswerId=51, Score=0, Title="Low", Body="x"),
        "</posts>",
    ]
)


@pytest.fixture
def posts(tmp_path: Path) -> Path:
    path = tmp_path / "Posts.xml"
    path.write_text(POSTS, encoding="utf-8")
    return path


def test_selection_is_by_score_then_ascending_id_among_accepted_questions(posts):
    selected = sf.select_questions(posts, wanted=10)
    assert list(selected) == [1, 2, 3]
    assert selected[2]["accepted_answer_id"] == 21


def test_selection_takes_only_the_requested_number_and_honours_the_score_floor(posts):
    assert list(sf.select_questions(posts, wanted=2)) == [1, 2]
    assert list(sf.select_questions(posts, wanted=10, min_score=40)) == [1]


def test_a_date_window_splits_the_corpus_in_time(posts):
    before = sf.to_epoch("2020-06-01T00:00:00.000")
    after = sf.to_epoch("2020-06-01T00:00:00.000")
    assert list(sf.select_questions(posts, 10, created_before=before)) == [1, 2]
    assert list(sf.select_questions(posts, 10, created_after=after)) == [3]


def test_collection_keeps_the_accepted_answer_and_each_posts_attribution(posts):
    selected = sf.select_questions(posts, 10)
    questions = sf.collect(posts, selected)
    first = questions[0]
    assert [q["question_id"] for q in questions] == [1, 2, 3]
    assert [a["answer_id"] for a in first["answers"]] == [11]
    assert first["title"] == "Why & how"
    assert first["tags"] == ["linux", "dns"]
    assert first["content_license"] == "CC BY-SA 4.0"
    assert first["owner"] == {
        "display_name": "user7",
        "user_id": 7,
        "link": "https://serverfault.com/users/7",
    }
    assert first["answers"][0]["owner"]["display_name"] == "Ann"
    assert first["link"] == "https://serverfault.com/questions/1"


def test_keeping_every_answer_orders_the_accepted_one_first(posts):
    selected = sf.select_questions(posts, 10)
    first = sf.collect(posts, selected, accepted_only=False)[0]
    assert [a["answer_id"] for a in first["answers"]] == [11, 12]


def test_both_tag_encodings_parse():
    assert sf.parse_tags("<a><b>") == ["a", "b"]
    assert sf.parse_tags("|a|b|") == ["a", "b"]
    assert sf.parse_tags("") == []


def test_a_slice_digest_depends_on_its_content(posts):
    payload = sf.build_slice(posts, questions=3)
    assert sf.slice_sha256(payload) == sf.slice_sha256(
        sf.build_slice(posts, questions=3)
    )
    assert sf.slice_sha256(payload) != sf.slice_sha256(
        sf.build_slice(posts, questions=2)
    )
    assert payload["selection"].startswith("top 3 by score")
    assert payload["answers_kept"] == "accepted only"


def test_licence_counts_cover_questions_and_kept_answers(posts):
    counts = sf.licence_counts(sf.build_slice(posts, questions=3))
    assert counts["CC BY-SA 4.0"] >= 1 and counts["CC BY-SA 2.5"] == 2
    assert sum(counts.values()) == 6


def test_html_becomes_text_with_code_kept():
    html = "<p>Run:</p><pre><code>ls -la</code></pre><ul><li>one</li><li>two</li></ul>a&amp;b"
    text = sf.strip_html(html)
    assert "ls -la" in text and "- one" in text and "a&b" in text and "<" not in text


def test_a_document_is_a_question_with_its_accepted_answer(posts):
    payload = sf.build_slice(posts, questions=3)
    docs = sf.documents(payload)
    assert [d.doc_id for d in docs] == ["q1", "q2", "q3"]
    assert docs[0].text.startswith("# Why & how")
    assert "## Accepted answer\nAccepted" in docs[0].text
    assert docs[0].licence == "CC BY-SA 4.0" and docs[0].link.endswith("/questions/1")
    assert len(sf.documents(payload, max_documents=2)) == 2


def test_the_ten_thousand_slice_is_held_to_its_recorded_statistics(posts):
    payload = sf.build_slice(posts, questions=3)
    with pytest.raises(ChecksumMismatchError, match="statistics"):
        sf._check_slice(payload, sf.SLICE_10K_QUESTIONS)
    sf._check_slice(payload, 3)


def test_a_slice_matching_the_statistics_but_not_the_digest_is_a_serialisation_change(
    posts, monkeypatch
):
    payload = sf.build_slice(posts, questions=3)
    monkeypatch.setattr(sf, "SLICE_10K_QUESTIONS", 3)
    monkeypatch.setattr(sf, "SLICE_10K_STATS", sf.slice_stats(payload))
    with pytest.raises(ChecksumMismatchError, match="serialisation"):
        sf._check_slice(payload, 3)
    monkeypatch.setattr(sf, "SLICE_10K_SHA256", sf.slice_sha256(payload))
    sf._check_slice(payload, 3)


def test_slice_statistics_count_what_the_record_counts(posts):
    stats = sf.slice_stats(sf.build_slice(posts, questions=3))
    assert stats["questions"] == 3 and stats["answers"] == 3
    assert stats["score_range"] == [30, 50]
    assert stats["distinct_tags"] == 3


def test_a_cached_slice_is_returned_without_the_dump(posts, tmp_path):
    payload = sf.build_slice(posts, questions=3)
    (tmp_path / "serverfault-slice-3.json").write_bytes(sf.slice_bytes(payload))
    assert sf.fetch_slice(3, dest=tmp_path)["questions"][0]["question_id"] == 1


def test_a_truncated_dump_fails_the_size_check(tmp_path):
    short = tmp_path / sf.DUMP_FILENAME
    short.write_bytes(b"not a dump")
    with pytest.raises(ChecksumMismatchError, match="bytes"):
        sf._verify(short)


@pytest.mark.skipif(shutil.which("7z") is None, reason="7z is not installed")
def test_posts_are_extracted_from_an_archive(posts, tmp_path):
    archive = tmp_path / "dump.7z"
    subprocess.run(
        ["7z", "a", str(archive), str(posts)], check=True, stdout=subprocess.DEVNULL
    )
    out = sf.extract_posts(archive, tmp_path / "out")
    assert out.read_text(encoding="utf-8") == POSTS
    assert json.dumps(sf.build_slice(out, 3)["questions"][0]["question_id"]) == "1"


def test_user_names_come_from_users_xml_for_slice_authors_only(tmp_path):
    users = tmp_path / "Users.xml"
    users.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<users>\n'
        '  <row Id="5" DisplayName="Ada" />\n'
        '  <row Id="6" DisplayName="Grace" />\n'
        '  <row Id="7" DisplayName="Not in the slice" />\n'
        "</users>\n",
        encoding="utf-8",
    )
    payload = {
        "questions": [
            {
                "owner": {"display_name": "user5", "user_id": 5},
                "answers": [{"owner": {"user_id": 6}}],
            },
            {"owner": {"display_name": "gone"}, "answers": []},
        ]
    }
    assert sf.slice_user_ids(payload) == {5, 6}
    assert sf.read_user_names(users, sf.slice_user_ids(payload)) == {
        5: "Ada",
        6: "Grace",
    }


def test_user_names_are_read_from_their_cache(tmp_path):
    (tmp_path / "serverfault-user-names-1.json").write_text(
        '{"5": "Ada"}', encoding="utf-8"
    )
    payload = {"questions": [{"owner": {"user_id": 5}, "answers": []}]}
    assert sf.fetch_user_names(payload, dest=tmp_path) == {5: "Ada"}
