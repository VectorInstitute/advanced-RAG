"""Guards for the describe-your-data profiles, question builders, and outputs."""

from __future__ import annotations

import json
from pathlib import Path

import describe_your_data as d
import pytest
from aieng.advanced_rag.agents.profile import ProfileTools, load_profile


def test_every_committed_profile_loads_with_the_structures_it_is_meant_to_declare() -> None:
    expected = {
        "serverfault": ("identifiers", "tables", "sections"),
        "financebench": ("tables", "sections"),
        "click": ("sections", "code"),
    }
    for name, structures in expected.items():
        assert load_profile(d.profile_path(name)).structures == structures


def test_the_template_profile_loads_and_declares_every_structure() -> None:
    profile = load_profile(d.PROFILES / "template.yaml")
    assert profile.structures == ("identifiers", "tables", "sections", "code")


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("$ 1,577", 1577.0),
        ("(370)", -370.0),
        ("$ (1,577)", -1577.0),
        ("3.2 %", 3.2),
        ("-12", -12.0),
        ("2018", 2018.0),
        ("Total revenues", None),
        ("", None),
        ("\u2014", None),
    ],
)
def test_a_table_cell_reads_as_a_number_with_parentheses_negative(text: str, value: float | None) -> None:
    assert d.parse_number(text) == value


def cell(text: str, header: bool = False) -> dict:
    return {"text": text, "column_header": header}


def test_leading_rows_of_text_or_years_are_headers_even_when_docling_misses_them() -> None:
    grid = [
        [cell(""), cell("2018"), cell("2017")],
        [cell("Sales"), cell("$ 12,267"), cell("$ 11,866")],
        [cell("Cost"), cell("(5,000)"), cell("(4,900)")],
    ]
    assert d.header_row_indexes(grid) == [0]
    marked = [[cell(""), cell("Revenue", True), cell("Margin", True)], *grid]
    assert d.header_row_indexes(marked) == [0, 1]


def test_a_table_becomes_cells_with_row_and_column_labels() -> None:
    table = {
        "prov": [{"page_no": 47}],
        "data": {
            "grid": [
                [cell(""), cell("2018"), cell("2017")],
                [cell("Purchases of PP&E"), cell("(1,577)"), cell("(1,373)")],
            ]
        },
    }
    summary, cells = d.table_rows("3M_2018_10K", 3, table)
    assert summary["table_id"] == "3M_2018_10K#t003"
    assert summary["page"] == 47
    assert summary["columns"] == "2018 | 2017"
    assert cells[0][5:] == ("Purchases of PP&E", "2018", "(1,577)", -1577.0)


def test_a_question_round_trips_through_its_record() -> None:
    question = d.Question("q1", "click", "Which file?", "src/click/core.py", "defining_file", "structured", ("x",))
    assert d.Question.from_dict(json.loads(json.dumps(question.to_dict()))) == question


def test_only_cheap_and_on_prem_models_run_every_question_live() -> None:
    assert d.live_cap("gemini-3.5-flash-lite") is None
    assert d.live_cap("Qwen3_6-35B-A3B") is None
    assert d.live_cap("claude-sonnet-5-5") == d.LIVE_CAP_ABOVE_CHEAP
    assert d.live_cap("gemini-4-argon") == d.LIVE_CAP_ABOVE_CHEAP


CLICK_SOURCE = """
class BaseThing:
    def __init__(self, width: int = 80, *, color: bool = False) -> None:
        pass


class Thing(BaseThing):
    def show(self, prefix: str = "> ") -> None:
        pass


def helper_only_here() -> None:
    pass
"""


def test_click_questions_read_their_answers_from_the_source(tmp_path: Path) -> None:
    src = tmp_path / "src" / "click"
    src.mkdir(parents=True)
    (src / "core.py").write_text(CLICK_SOURCE)
    (src / "other.py").write_text(
        "class A(Exception):\n    pass\n\n\nclass B(A):\n    def f(self, x=1):\n        pass\n"
        "\n\nclass C(B):\n    pass\n\n\nclass D(C):\n    pass\n\n\nclass E(D):\n    pass\n"
        "\n\ndef g1():\n    pass\n\n\ndef g2():\n    pass\n\n\ndef g3(y='a', z=2.5):\n    pass\n"
    )
    (src / "more.py").write_text("class F:\n    def h(self, a=None, b=3, c='x', e=True):\n        pass\n")
    questions = d.click_code_questions(tmp_path)
    by_type = {}
    for q in questions:
        by_type.setdefault(q.qtype, []).append(q)
    assert len(by_type["default_value"]) == 6
    assert len(by_type["base_class"]) == 5
    assert len(by_type["defining_file"]) == 5
    width = [q for q in by_type["default_value"] if "`width`" in q.question]
    if width:
        assert width[0].reference == "80"
    thing = [q for q in by_type["base_class"] if "`Thing`" in q.question]
    if thing:
        assert thing[0].reference == "BaseThing"
    assert all(q.gold_docs for q in questions)


def write_serverfault(folder: Path) -> None:
    posts = folder / "posts"
    posts.mkdir(parents=True)
    rows = ["doc_id,question_id,title,tags,score,answer_score,views,answers,year,link,licence"]
    tags_rows = ["doc_id,tag"]
    for i in range(60):
        tags = ["alpha"] if i < 12 else ["beta"] if i < 30 else ["gamma", "delta"]
        if i % 5 == 0:
            tags.append("epsilon")
        doc = f"posts/q{i}.md"
        path = f"/etc/service{i}/conf.d/settings.conf"
        (folder / doc).write_text(f"# Title {i}\n\nTags: {', '.join(tags)}\n\n## Question\nSee {path}\n")
        rows.append(f"{doc},{i},Title {i},{'|'.join(tags)},{i},0,0,1,2015,https://example.org/{i},CC BY-SA 4.0")
        tags_rows += [f"{doc},{t}" for t in tags]
    (folder / "posts.csv").write_text("\n".join(rows) + "\n")
    (folder / "post_tags.csv").write_text("\n".join(tags_rows) + "\n")


def test_serverfault_structured_questions_have_computed_answers(tmp_path: Path) -> None:
    write_serverfault(tmp_path)
    profile = load_profile(d.profile_path("serverfault"), root=tmp_path)
    tools = ProfileTools(profile, profile.corpus())
    questions = d.serverfault_structured(tmp_path, tools, per_type=1)
    by_type = {q.qtype: q for q in questions}
    assert set(by_type) == {"count", "count_two_tags", "top_score", "identifier"}
    count = by_type["count"]
    tag = count.question.split("`")[1]
    assert count.reference == str(
        sum(1 for line in (tmp_path / "post_tags.csv").read_text().splitlines()[1:] if line.endswith("," + tag))
    )
    top = by_type["top_score"]
    assert top.gold_docs and top.reference.startswith("Title ")


def test_the_summary_and_committed_record_hold_no_question_or_answer_text() -> None:
    records = [
        {
            "qid": f"q{i}",
            "corpus": "click",
            "qtype": "default_value",
            "source": "structured",
            "arm": arm,
            "model": "m",
            "answer": "secret answer text",
            "model_calls": 3,
            "tool_calls": 2,
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "seconds": 1.0,
            "exhausted": False,
            "tool_counts": {"grep": 1},
            "gold_read": i == 0,
            "gold_seen": True,
            "errors": 0,
        }
        for i in range(2)
        for arm in ("G", "P")
    ]
    verdicts = [{"qid": r["qid"], "arm": r["arm"], "correct": r["arm"] == "P", "judge": "j"} for r in records]
    summaries = d.summarise(records, verdicts)
    assert [(s.arm, s.accuracy) for s in summaries] == [("G", 0.0), ("P", 1.0)]
    committed = json.dumps(d.committed_record(summaries, records, verdicts, {"model": "m"}))
    assert "secret answer text" not in committed


def test_the_committed_results_label_their_models_and_carry_no_text() -> None:
    path = d.OUTPUTS / "results.json"
    if not path.is_file():
        pytest.skip("no committed results yet")
    payload = json.loads(path.read_text())
    assert payload["model"] and payload["judge"]
    banned = {"question", "answer", "reference", "text", "transcript"}
    for row in payload["per_question"]:
        assert not banned & set(row)
    assert {s["corpus"] for s in payload["summaries"]} == set(d.CORPUS_NAMES)
