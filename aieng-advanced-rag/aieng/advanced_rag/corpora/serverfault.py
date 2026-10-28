"""ServerFault as a technical-support corpus, built from a pinned Stack Exchange dump.

The corpus is read from the Internet Archive's ``stackexchange`` collection, which
is frozen and checksummed. The dump used here was published on 2024-04-06, and its
MD5 and SHA-256 are checked on every fetch, so a corpus built from this module is
reproducible by anyone and a silently different dump is an error. Two earlier
sources are not used: the live API, whose "top by votes" is a different set every
month and which caps at 300 anonymous requests a day, and the Hugging Face
``common-pile/stackexchange`` copy, which pins no dump.

A document is one question paired with its accepted answer. That pairing is the
unit a support knowledge base holds. The slice is the top ``N`` questions that have
an accepted answer, by score and then by ascending identifier, so the order is
total and a prefix of a large slice is exactly a smaller slice.

Posts are CC BY-SA at the version each post records (2.5, 3.0, and 4.0 all occur).
Every post keeps its author, its link, and its licence version. Share-alike applies
to derived text, graphs, and questions that quote a post. This module is not
registered in ``corpora.SOURCES`` for that reason.

Extracting ``Posts.xml`` needs the ``7z`` program on the path.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import subprocess
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .sources import ChecksumMismatchError, Licence, cache_dir


DUMP_URL = "https://archive.org/download/stackexchange/serverfault.com.7z"
DUMP_FILENAME = "serverfault.com.7z"
DUMP_PUBLISHED = "2024-04-06"
DUMP_BYTES = 859_606_467
#: The checksum archive.org publishes for the file.
DUMP_MD5 = "1b075d4c7ae7396da965874f156e0fdf"
#: SHA-256 of the same file, computed when the pin was written.
DUMP_SHA256 = "8a2fed8620c93c9d94de7fa1722b24e77f802f64cc5ca652652af3adfd946cd8"
SITE_URL = "https://serverfault.com"

SERVERFAULT_LICENCE = Licence(
    name="CC BY-SA (2.5, 3.0, or 4.0, as each post records)",
    url="https://creativecommons.org/licenses/by-sa/4.0/",
    permits_commercial_use=True,
    attribution=(
        "Stack Exchange user contributions to serverfault.com, licensed CC BY-SA at "
        "the version recorded per post, from the Stack Exchange data dump published "
        "2024-04-06. https://archive.org/details/stackexchange"
    ),
    notes=(
        "Attribution and share-alike apply to anything derived from the posts, "
        "including chunks, extracted graphs, and questions that quote a post. Each "
        "post's author, link, and licence version stay with it."
    ),
)

#: The slice size: 10,000 questions with accepted answers.
SLICE_10K_QUESTIONS = 10_000
#: SHA-256 of the 10,000-question slice as :func:`slice_bytes` serialises it. This
#: pins this module's own output.
SLICE_10K_SHA256 = "7bfb51e583bd9a245cc389afaa7244293ab335f496e6c397353a3b91c9d1dd36"
#: Statistics recorded when 10,000 questions were first built from this dump. That
#: slice file was not kept, so its bytes cannot be compared, but these aggregates
#: can, and a rebuild here matches all of them. They check the content
#: independently of how it is serialised.
SLICE_10K_STATS: dict[str, Any] = {
    "questions": 10_000,
    "answers": 9_999,
    "question_body_chars": 10_957_078,
    "answer_body_chars": 11_517_692,
    "distinct_tags": 2_082,
    "score_range": [9, 1831],
    "creation_date_range": [1_217_608_671, 1_707_778_079],
}

_TAG = re.compile(r"<([^<>]+)>")
_TAGS = re.compile(r"<[^>]+>")
_BLANKS = re.compile(r"\n{3,}")


def parse_tags(raw: str) -> list[str]:
    """Parse either dump tag encoding, ``<a><b>`` or ``|a|b|``, into a list."""
    if not raw:
        return []
    if raw.startswith("<"):
        return _TAG.findall(raw)
    return [tag for tag in raw.split("|") if tag]


def to_epoch(value: str | None) -> int | None:
    """Convert a dump timestamp to epoch seconds."""
    if not value:
        return None
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp())


def strip_html(text: str) -> str:
    """Return plain text from a rendered Stack Exchange body.

    Code blocks are kept, because on ServerFault the command, configuration
    stanza, or error string in a ``<code>`` block is often the answer.
    """
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"</(p|div|li|h[1-6]|pre)>", "\n", text)
    text = re.sub(r"<li>", "- ", text)
    text = _TAGS.sub("", text)
    return _BLANKS.sub("\n\n", html.unescape(text)).strip()


def fetch_dump(*, dest: Path | None = None) -> Path:
    """Download the dump into the corpus cache and verify it.

    The file is streamed to disk and then hashed in one pass, because it is 860 MB.
    A cached file is verified again on every call.

    Returns
    -------
    Path
        The verified archive.

    Raises
    ------
    ChecksumMismatchError
        If the size, the MD5, or the SHA-256 differs from the pin. A truncated
        download is the usual cause.
    """
    target = cache_dir(dest) / DUMP_FILENAME
    if target.exists():
        try:
            _verify(target)
        except ChecksumMismatchError:
            target.unlink()
        else:
            return target
    partial = target.with_suffix(".part")
    with (
        urllib.request.urlopen(DUMP_URL, timeout=600) as response,
        partial.open("wb") as out,
    ):  # noqa: S310
        while block := response.read(1 << 20):
            out.write(block)
    try:
        _verify(partial)
    except ChecksumMismatchError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(target)
    return target


def _verify(path: Path) -> None:
    size = path.stat().st_size
    if size != DUMP_BYTES:
        raise ChecksumMismatchError(f"{path}: {size} bytes, expected {DUMP_BYTES}")
    md5 = hashlib.md5()  # noqa: S324 - matching the checksum archive.org publishes
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            md5.update(block)
            sha.update(block)
    if md5.hexdigest() != DUMP_MD5 or sha.hexdigest() != DUMP_SHA256:
        raise ChecksumMismatchError(
            f"{path}: md5 {md5.hexdigest()} and sha256 {sha.hexdigest()} differ from the "
            "pins. The dump changed or the download was truncated."
        )


def extract_posts(dump: Path, dest: Path, member: str = "Posts.xml") -> Path:
    """Extract only ``Posts.xml``, or another ``member``, from the archive, with ``7z``.

    Raises
    ------
    RuntimeError
        If ``7z`` is not installed.
    """
    posts = dest / member
    if posts.exists():
        return posts
    program = shutil.which("7z") or shutil.which("7zr")
    if program is None:
        raise RuntimeError(
            "extracting the dump needs the 7z program (p7zip) on the path"
        )
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603
        [program, "x", "-y", f"-o{dest}", str(dump), member],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return posts


def iter_rows(posts: Path) -> Iterator[Mapping[str, str]]:
    """Stream ``<row>`` elements, clearing each so memory stays flat."""
    for _, element in ET.iterparse(posts, events=("end",)):
        if element.tag != "row":
            continue
        yield element.attrib
        element.clear()


def _owner(attrib: Mapping[str, str]) -> dict[str, Any]:
    owner: dict[str, Any] = {
        "display_name": attrib.get("OwnerDisplayName", "")
        or "user" + attrib.get("OwnerUserId", "")
    }
    if "OwnerUserId" in attrib:
        owner["user_id"] = int(attrib["OwnerUserId"])
        owner["link"] = f"{SITE_URL}/users/{attrib['OwnerUserId']}"
    return owner


def select_questions(
    posts: Path,
    wanted: int,
    min_score: int = 1,
    *,
    created_before: int | None = None,
    created_after: int | None = None,
) -> dict[int, dict[str, int]]:
    """Pick the questions in one pass, without holding their bodies in memory.

    The top ``wanted`` questions with an accepted answer, ordered by score and then
    by ascending identifier. The dump is frozen, so this is a total order.
    """
    candidates: list[tuple[int, int, int]] = []
    for attrib in iter_rows(posts):
        if attrib.get("PostTypeId") != "1" or "AcceptedAnswerId" not in attrib:
            continue
        score = int(attrib.get("Score", 0))
        if score < min_score:
            continue
        created = to_epoch(attrib.get("CreationDate"))
        if created_before is not None and (
            created is None or created >= created_before
        ):
            continue
        if created_after is not None and (created is None or created < created_after):
            continue
        candidates.append((-score, int(attrib["Id"]), int(attrib["AcceptedAnswerId"])))
    candidates.sort()
    return {
        question_id: {"score": -negative, "accepted_answer_id": accepted}
        for negative, question_id, accepted in candidates[:wanted]
    }


def collect(
    posts: Path,
    selected: Mapping[int, Mapping[str, int]],
    *,
    accepted_only: bool = True,
) -> list[dict[str, Any]]:
    """Pull the selected questions and their answers in a second pass.

    ``accepted_only`` drops the other answers, which multiply a large slice by
    about ten and mostly restate or contradict the accepted one.
    """
    questions: dict[int, dict[str, Any]] = {}
    answers: dict[int, list[dict[str, Any]]] = {}
    for attrib in iter_rows(posts):
        post_type = attrib.get("PostTypeId")
        if post_type == "1":
            question_id = int(attrib["Id"])
            if question_id not in selected:
                continue
            questions[question_id] = {
                "question_id": question_id,
                "title": html.unescape(attrib.get("Title", "")),
                "body": attrib.get("Body", ""),
                "tags": parse_tags(attrib.get("Tags", "")),
                "score": int(attrib.get("Score", 0)),
                "view_count": int(attrib.get("ViewCount", 0)),
                "answer_count": int(attrib.get("AnswerCount", 0)),
                "creation_date": to_epoch(attrib.get("CreationDate")),
                "last_edit_date": to_epoch(attrib.get("LastEditDate")),
                "last_activity_date": to_epoch(attrib.get("LastActivityDate")),
                "accepted_answer_id": selected[question_id]["accepted_answer_id"],
                "content_license": attrib.get("ContentLicense", "CC BY-SA 4.0"),
                "link": f"{SITE_URL}/questions/{question_id}",
                "owner": _owner(attrib),
                "answers": [],
            }
        elif post_type == "2":
            parent = int(attrib.get("ParentId", 0))
            if parent not in selected:
                continue
            answer_id = int(attrib["Id"])
            if accepted_only and answer_id != selected[parent]["accepted_answer_id"]:
                continue
            answers.setdefault(parent, []).append(
                {
                    "answer_id": answer_id,
                    "question_id": parent,
                    "body": attrib.get("Body", ""),
                    "score": int(attrib.get("Score", 0)),
                    "is_accepted": answer_id == selected[parent]["accepted_answer_id"],
                    "creation_date": to_epoch(attrib.get("CreationDate")),
                    "last_edit_date": to_epoch(attrib.get("LastEditDate")),
                    "last_activity_date": to_epoch(attrib.get("LastActivityDate")),
                    "content_license": attrib.get("ContentLicense", "CC BY-SA 4.0"),
                    "owner": _owner(attrib),
                }
            )
    for question_id, question in questions.items():
        question["answers"] = sorted(
            answers.get(question_id, []),
            key=lambda a: (not a["is_accepted"], -a["score"], a["answer_id"]),
        )
    return sorted(questions.values(), key=lambda q: (-q["score"], q["question_id"]))


def build_slice(
    posts: Path,
    questions: int = SLICE_10K_QUESTIONS,
    *,
    min_score: int = 1,
    accepted_only: bool = True,
) -> dict[str, Any]:
    """Build the slice payload from an extracted ``Posts.xml``."""
    selected = select_questions(posts, questions, min_score)
    rows = collect(posts, selected, accepted_only=accepted_only)
    return {
        "site": "serverfault",
        "source": "stackexchange data dump",
        "dump_url": DUMP_URL,
        "dump_md5": DUMP_MD5,
        "dump_published": DUMP_PUBLISHED,
        "selection": (
            f"top {questions} by score among questions with an accepted answer, "
            "ties by ascending id"
        ),
        "created_before": None,
        "created_after": None,
        "answers_kept": "accepted only" if accepted_only else "all",
        "questions": rows,
        "license": "CC BY-SA, per-post version in content_license; see ATTRIBUTION.md",
    }


def slice_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialise a slice the way its recorded digest was computed."""
    return json.dumps(payload, indent=1).encode("utf-8")


def slice_sha256(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 of a slice's serialised bytes."""
    return hashlib.sha256(slice_bytes(payload)).hexdigest()


def licence_counts(payload: Mapping[str, Any]) -> dict[str, int]:
    """Count posts by CC BY-SA version, over questions and kept answers."""
    counts: dict[str, int] = {}
    for question in payload["questions"]:
        for post in (question, *question["answers"]):
            name = post["content_license"]
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


@dataclass(frozen=True)
class Document:
    """One question and its accepted answer, as plain text.

    Parameters
    ----------
    doc_id : str
        ``q`` followed by the question identifier.
    question_id : int
        The Stack Exchange question identifier.
    title : str
        The question title.
    tags : tuple of str
        The question's tags.
    question_text : str
        The question body as plain text.
    answer_text : str
        The accepted answer as plain text.
    score : int
        The question's score in the dump.
    link : str
        The question's URL, which attribution requires.
    licence : str
        The CC BY-SA version the question records.
    """

    doc_id: str
    question_id: int
    title: str
    tags: tuple[str, ...]
    question_text: str
    answer_text: str
    score: int
    link: str
    licence: str

    @property
    def text(self) -> str:
        """Return the document as one string, with the answer under its own heading."""
        return (
            f"# {self.title}\n\n"
            f"Tags: {', '.join(self.tags)}\n\n"
            f"## Question\n{self.question_text}\n\n"
            f"## Accepted answer\n{self.answer_text}\n"
        )


def documents(
    payload: Mapping[str, Any], max_documents: int | None = None
) -> list[Document]:
    """Return the slice's questions that have an accepted answer, in slice order."""
    found: list[Document] = []
    for question in payload["questions"]:
        accepted_id = question.get("accepted_answer_id")
        accepted = next(
            (a for a in question.get("answers", []) if a["answer_id"] == accepted_id),
            None,
        )
        if not accepted_id or accepted is None:
            continue
        found.append(
            Document(
                doc_id=f"q{question['question_id']}",
                question_id=int(question["question_id"]),
                title=html.unescape(question["title"]),
                tags=tuple(question.get("tags", [])),
                question_text=strip_html(question.get("body", "")),
                answer_text=strip_html(accepted.get("body", "")),
                score=int(question.get("score", 0)),
                link=str(question.get("link", "")),
                licence=str(question.get("content_license", "")),
            )
        )
        if max_documents and len(found) >= max_documents:
            break
    return found


def fetch_slice(
    questions: int = SLICE_10K_QUESTIONS, *, dest: Path | None = None
) -> dict[str, Any]:
    """Fetch the dump, build a slice from it, and cache the result.

    The 10,000-question slice is checked against its recorded statistics and its
    SHA-256. Other sizes are built from the verified dump and carry no recorded
    digest.

    Raises
    ------
    ChecksumMismatchError
        If the dump differs from its pin, or the 10,000-question slice differs
        from its recorded digest.
    """
    root = cache_dir(dest)
    cached = root / f"serverfault-slice-{questions}.json"
    if cached.exists():
        payload: dict[str, Any] = json.loads(cached.read_text(encoding="utf-8"))
        _check_slice(payload, questions)
        return payload
    dump = fetch_dump(dest=dest)
    posts = extract_posts(dump, root / "serverfault-dump")
    try:
        payload = build_slice(posts, questions)
    finally:
        posts.unlink(missing_ok=True)
    _check_slice(payload, questions)
    cached.write_bytes(slice_bytes(payload))
    return payload


def read_user_names(users: Path, wanted: set[int]) -> dict[int, str]:
    """Return the display name of each wanted user id from an extracted ``Users.xml``.

    ``Posts.xml`` names an author only when the account is gone, so the names
    attribution needs come from ``Users.xml``.
    """
    names: dict[int, str] = {}
    for attrib in iter_rows(users):
        user_id = int(attrib.get("Id", "0"))
        if user_id in wanted:
            names[user_id] = attrib.get("DisplayName", "")
    return names


def slice_user_ids(payload: Mapping[str, Any]) -> set[int]:
    """Return the user id of every question and kept answer author in a slice."""
    ids: set[int] = set()
    for question in payload["questions"]:
        for post in (question, *question.get("answers", [])):
            user_id = (post.get("owner") or {}).get("user_id")
            if user_id is not None:
                ids.add(int(user_id))
    return ids


def fetch_user_names(
    payload: Mapping[str, Any], *, dest: Path | None = None
) -> dict[int, str]:
    """Return the display names of a slice's authors, cached beside the slice.

    The names come from ``Users.xml`` in the same pinned dump. They are kept
    apart from the slice, so the slice's recorded digest stays as it is.
    """
    root = cache_dir(dest)
    count = len(payload["questions"])
    cached = root / f"serverfault-user-names-{count}.json"
    if cached.exists():
        stored = json.loads(cached.read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in stored.items()}
    dump = fetch_dump(dest=dest)
    users = extract_posts(dump, root / "serverfault-dump", member="Users.xml")
    try:
        names = read_user_names(users, slice_user_ids(payload))
    finally:
        users.unlink(missing_ok=True)
    cached.write_text(
        json.dumps({str(k): v for k, v in sorted(names.items())}, ensure_ascii=False),
        encoding="utf-8",
    )
    return names


def slice_stats(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return aggregate statistics of a slice, for comparison across rebuilds."""
    rows = payload["questions"]
    return {
        "questions": len(rows),
        "answers": sum(len(q["answers"]) for q in rows),
        "question_body_chars": sum(len(q["body"]) for q in rows),
        "answer_body_chars": sum(len(a["body"]) for q in rows for a in q["answers"]),
        "distinct_tags": len({tag for q in rows for tag in q["tags"]}),
        "score_range": [rows[-1]["score"], rows[0]["score"]] if rows else [],
        "creation_date_range": [
            min(q["creation_date"] for q in rows),
            max(q["creation_date"] for q in rows),
        ]
        if rows
        else [],
    }


def _check_slice(payload: Mapping[str, Any], questions: int) -> None:
    if questions != SLICE_10K_QUESTIONS:
        return
    if slice_stats(payload) != SLICE_10K_STATS:
        raise ChecksumMismatchError(
            f"the {questions}-question slice has statistics {slice_stats(payload)}, "
            f"expected {SLICE_10K_STATS}. The selection or the dump changed."
        )
    if slice_sha256(payload) != SLICE_10K_SHA256:
        raise ChecksumMismatchError(
            f"the {questions}-question slice hashes to {slice_sha256(payload)}, expected "
            f"{SLICE_10K_SHA256}. Its content matches the recorded statistics, so the "
            "serialisation changed."
        )


__all__ = [
    "DUMP_BYTES",
    "DUMP_MD5",
    "DUMP_SHA256",
    "DUMP_URL",
    "SERVERFAULT_LICENCE",
    "SLICE_10K_QUESTIONS",
    "SLICE_10K_SHA256",
    "SLICE_10K_STATS",
    "Document",
    "build_slice",
    "collect",
    "documents",
    "extract_posts",
    "fetch_dump",
    "fetch_slice",
    "fetch_user_names",
    "iter_rows",
    "licence_counts",
    "parse_tags",
    "read_user_names",
    "select_questions",
    "slice_bytes",
    "slice_sha256",
    "slice_stats",
    "slice_user_ids",
    "strip_html",
    "to_epoch",
]
