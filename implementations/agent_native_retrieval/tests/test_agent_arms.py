"""RI #5's arms and report, with no model and no network."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from agent_arms import (
    ANSWER,
    ARM_NAMES,
    DECOMPOSE,
    STEP,
    GrepIterative,
    arm_config,
    arm_step,
    live_questions,
    spans,
)
from agent_report import (
    compare_f1,
    compare_models,
    format_comparison,
    format_table,
    joined_table,
    public_row,
    ri4_rows,
    summarise,
)
from aieng.advanced_rag.agents.corpus_tools import STANDARD_ARMS, Corpus
from aieng.advanced_rag.corpora.musique import MusiqueQuestion


HERE = Path(__file__).resolve().parents[1]
RI4 = HERE.parent / "graphrag_technical_support"


def ri4_constant(name: str) -> str:
    """Read a string constant from RI #4's arms module without importing it."""
    tree = ast.parse((RI4 / "musique_arms.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


@pytest.mark.parametrize(("ours", "name"), [(DECOMPOSE, "DECOMPOSE"), (STEP, "STEP"), (ANSWER, "ANSWER")])
def test_grep_iterative_sends_ri4s_prompts(ours: str, name: str) -> None:
    assert ours == ri4_constant(name)


@dataclass(frozen=True)
class P:
    passage_id: str
    title: str
    text: str


CORPUS = Corpus.from_passages(
    [
        P("p0", "Lothair II", "Lothair II's mother was Ermengarde of Tours."),
        P("p1", "Ermengarde of Tours", "Ermengarde of Tours died in 851 at Prum."),
        P("p2", "Tours", "Tours is a city in France. Many people from Tours."),
    ]
)


def question(qid: str, hops: int) -> MusiqueQuestion:
    return MusiqueQuestion(qid, "q", "a", (), hops, tuple(f"g{i}" for i in range(hops)), ())


def test_spans_finds_capitalized_names() -> None:
    assert "Lothair II" in spans("When did Lothair II's mother die?")
    assert "When" not in spans("When did it happen?")


def test_grep_iterative_greps_the_bridge_and_substitutes() -> None:
    prompts: list[str] = []

    def complete(prompt: str) -> tuple[str, int, int]:
        prompts.append(prompt)
        if prompt.startswith("Break"):
            return json.dumps({"steps": ["Who was the mother of Lothair II?", "When did #1 die?"]}), 10, 5
        if "When did Ermengarde of Tours die?" in prompt:
            return "851", 10, 1
        if "mother of Lothair II" in prompt:
            return "Ermengarde of Tours", 10, 3
        return "851", 10, 1

    record = GrepIterative(CORPUS, complete).run("When did Lothair II's mother die?")
    assert record["answer"] == "851"
    assert record["steps"][0]["span"] == "Lothair II"
    assert record["steps"][1]["span"] == "Ermengarde of Tours"
    assert "p1" in record["steps"][1]["hits"]
    assert record["model_calls"] == 4 == len(prompts)
    assert record["prompt_tokens"] == 40
    assert set(record["read_ids"]) >= {"p0", "p1"}


def test_grep_iterative_returns_nothing_when_no_span_matches() -> None:
    ids, span = GrepIterative(CORPUS, lambda p: ("", 0, 0)).retrieve("Who founded Zzyzx?", [])
    assert (ids, span) == ([], "")


def test_live_questions_caps_by_tier() -> None:
    pool = [question(f"{h}-{i}", h) for h in (2, 3, 4) for i in range(20)]
    cheap = live_questions(pool, "gemini-3.5-flash-lite")
    assert len(cheap) == 30
    assert {q.hops for q in cheap} == {2, 3, 4}
    assert len(live_questions(pool, "Qwen3_6-35B-A3B")) == 30
    frontier = live_questions(pool, "claude-opus-5-5")
    assert len(frontier) == 10
    assert {q.hops for q in frontier} == {2, 3, 4}


def test_arm_steps_differ_by_arm_and_questions() -> None:
    qs = [question("2-0", 2)]
    keys = {
        arm_step(arm_config("C0"), CORPUS, qs, "x").key(_choice()).digest,
        arm_step(arm_config("C5"), CORPUS, qs, "x").key(_choice()).digest,
        arm_step(arm_config("C5"), CORPUS, [*qs, question("2-1", 2)], "x").key(_choice()).digest,
    }
    assert len(keys) == 3


def _choice():
    from aieng.advanced_rag.models import REFERENCE_MODELS  # noqa: PLC0415

    return REFERENCE_MODELS.generation


RUNS = [
    {
        "question_id": "a",
        "hops": 2,
        "em": 1.0,
        "f1": 1.0,
        "model_calls": 4,
        "tool_calls": 3,
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "seconds": 2.0,
        "exhausted": False,
        "read_ids": ["g0", "g1"],
        "seen_ids": ["g0", "g1", "x"],
        "answer": "secret answer text",
        "transcript": [{"role": "user", "content": "question text"}],
        "trace": [{"name": "grep", "total_hits": 0}, {"name": "open", "total_hits": None}],
    },
    {
        "question_id": "b",
        "hops": 4,
        "em": 0.0,
        "f1": 0.5,
        "model_calls": 30,
        "tool_calls": 29,
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "seconds": 20.0,
        "exhausted": True,
        "read_ids": ["h0"],
        "seen_ids": ["h0", "h1"],
    },
]
GOLD = {"a": ["g0", "g1"], "b": ["h0", "h1", "h2", "h3"]}


def test_public_row_drops_every_text_field() -> None:
    row = public_row(RUNS[0])
    assert "answer" not in row
    assert "transcript" not in row
    assert "trace" not in row
    assert row["tools"] == {"grep": 1, "open": 1}
    assert row["zero_hit_greps"] == 1


def test_summarise_reads_recall_over_passages_read() -> None:
    summary = summarise([public_row(r) for r in RUNS], GOLD)
    assert summary["2hop"]["all_supporting_recall_read"] == 1.0
    assert summary["4hop"]["all_supporting_recall_read"] == 0.0
    assert summary["4hop"]["mean_supporting_recall_read"] == 0.25
    assert summary["4hop"]["seen_not_read_rate"] == 1.0
    assert summary["all"]["exhaustion_rate"] == 0.5
    assert summary["all"]["calls_per_question"] == 17.0
    assert summary["all"]["tokens_per_question"] == 605.0


def test_compare_f1_pairs_questions() -> None:
    other = [dict(r, f1=0.0) for r in RUNS]
    result = compare_f1(RUNS, other)
    assert result["questions"] == 2
    assert result["mean_difference"] == 0.75


def test_joined_table_puts_ri4_rows_first() -> None:
    ri4 = ri4_rows(RI4 / "outputs" / "musique_graph_test.json")
    ri5 = {"arms": {"C5": {"model": "Qwen3_6-35B-A3B", "summary": summarise(RUNS, GOLD)}}}
    rows = joined_table(ri5, ri4)
    assert [r["arm"] for r in rows] == ["dense", "dense-iterative", "hipporag2", "C5"]
    assert rows[0]["calls_per_question"] == 1.0
    assert rows[-1]["recall_seen"] == 0.5
    assert rows[0]["recall_read"] == ri4["arms"]["dense"]["summary"]["all"]["all_supporting_recall_at_5"]
    assert "C5" in format_table(rows)


def test_the_probe_follows_bridge_answers() -> None:
    from aieng.advanced_rag.corpora.musique import SubQuestion  # noqa: PLC0415
    from grep_probe import reachability  # noqa: PLC0415

    q = MusiqueQuestion(
        "2hop__1",
        "When did Lothair II's mother die?",
        "851",
        (),
        2,
        ("p0", "p1"),
        (
            SubQuestion("Who was the mother of Lothair II?", "Ermengarde of Tours", "p0"),
            SubQuestion("When did #1 die?", "851", "p1"),
        ),
    )
    result = reachability([q], CORPUS)
    assert result["by_hop"][2] == {"questions": 1, "first_hop": 1.0, "bridge": 1.0, "chain": 1.0}
    assert result["rows"][0]["chain"] is True


def _result(model: str, label: str, judged: float | None) -> dict:
    summary = {
        "all": {
            "questions": 2,
            "exact_match": 0.5,
            "f1": 0.6,
            "calls_per_question": 3.0,
            "tokens_per_question": 100.0,
            "exhaustion_rate": 0.0,
        }
    }
    entry: dict = {"model": model, "summary": summary}
    if judged is not None:
        entry["judged"] = {"all": judged}
    return {"produced_by": {"label": label}, "arms": {"C0": entry}}


def test_compare_models_puts_each_arm_in_both_models() -> None:
    rows = compare_models(
        {
            "qwen": _result("Qwen3_6-35B-A3B", "reference trajectories", None),
            "gemini": _result("gemini-3.5-flash-lite", "paid-model rerun", 0.55),
        }
    )
    assert [(r["arm"], r["model"], r["judged"]) for r in rows] == [
        ("C0", "Qwen3_6-35B-A3B", None),
        ("C0", "gemini-3.5-flash-lite", 0.55),
    ]
    text = format_comparison(rows)
    assert "paid-model rerun" in text
    assert "0.550" in text


def test_the_committed_gemini_table_is_labelled_and_judged() -> None:
    results = json.loads((HERE / "outputs" / "musique_agent_arms_gemini.json").read_text(encoding="utf-8"))
    assert results["produced_by"]["model"] == "gemini-3.5-flash-lite"
    assert results["produced_by"]["judge"] == "gemini-3.8-flash"
    assert set(results["arms"]) == {"C0", "C5", "grep-iterative"}
    assert all("judged" in entry for entry in results["arms"].values())
    assert results["test"]["questions"] == 300


def test_the_ri4_join_picks_the_file_of_the_answering_model() -> None:
    from run_agent_arms import ri4_for  # noqa: PLC0415

    assert {e["model"] for e in ri4_for("Qwen3_6-35B-A3B")["arms"].values()} == {"Qwen3_6-35B-A3B"}
    assert {e["model"] for e in ri4_for("gemini-3.5-flash-lite")["arms"].values()} == {"gemini-3.5-flash-lite"}
    assert ri4_for("no-such-model") == {"arms": {}}


def test_the_ladder_report_keeps_each_size_apart() -> None:
    from run_agent_arms import ladder_report  # noqa: PLC0415

    questions = [
        MusiqueQuestion("q2", "?", "a", (), 2, ("p0", "p1"), ()),
        MusiqueQuestion("q4", "?", "a", (), 4, ("p0", "p1", "p2", "p3"), ()),
    ]

    def run(qid: str, hops: int, f1: float) -> dict:
        return {
            "question_id": qid,
            "hops": hops,
            "answer": "a",
            "em": f1,
            "f1": f1,
            "read_ids": ["p0"],
            "seen_ids": [],
            "model_calls": 2,
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "exhausted": False,
            "seconds": 1.0,
        }

    def arms(f1: float) -> dict:
        return {
            name: {"model": "gemini-3.5-flash-lite", "runs": [run("q2", 2, f1), run("q4", 4, f1)]}
            for name in ("C0", "C5")
        }

    out = ladder_report(
        {11_656: arms(1.0), 139_416: arms(0.0)},
        questions,
        {11_656: {"C0": {"q2": True}, "C5": {}}, 139_416: {"C0": {}, "C5": {}}},
        "gemini-3.8-flash",
    )
    assert out["produced_by"]["label"] == "paid-model rerun"
    assert out["produced_by"]["judge"] == "gemini-3.8-flash"
    assert out["test"]["label"] == "paid-model rerun"
    assert set(out["sizes"]) == {"11656", "139416"}
    assert out["sizes"]["11656"]["arms"]["C0"]["summary"]["all"]["f1"] == 1.0
    assert out["sizes"]["139416"]["arms"]["C0"]["summary"]["all"]["f1"] == 0.0
    assert out["sizes"]["11656"]["arms"]["C0"]["judged"]["all"] == 0.5
    assert "C5_minus_C0_f1_4hop" in out["sizes"]["139416"]["comparisons"]


def test_the_committed_gemini_ladder_is_labelled_and_judged() -> None:
    ladder = json.loads((HERE / "outputs" / "musique_scale_ladder_gemini.json").read_text(encoding="utf-8"))
    assert ladder["produced_by"]["model"] == "gemini-3.5-flash-lite"
    assert ladder["produced_by"]["label"] == "paid-model rerun"
    assert ladder["produced_by"]["judge"] == "gemini-3.8-flash"
    for size in ("11656", "139416"):
        assert set(ladder["sizes"][size]["arms"]) == {"C0", "C5", "C7"}
        for entry in ladder["sizes"][size]["arms"].values():
            assert entry["summary"]["all"]["questions"] == 120
            assert "judged" in entry


def test_variant_arms_change_only_what_they_test() -> None:
    top2 = arm_config("C0-top2")
    assert top2.search_k == 2
    assert top2.tools == STANDARD_ARMS["C0"].tools
    full = arm_config("C7-top5-full")
    assert (full.search_k, full.search_preview_tokens) == (5, 512)
    assert full.tools == STANDARD_ARMS["C7"].tools
    assert full.grep_scope == STANDARD_ARMS["C7"].grep_scope
    assert {"C0-top2", "C7-top5-full"} <= set(ARM_NAMES)


def test_an_agent_arm_is_keyed_on_the_tools_adk_declares() -> None:
    from agent_arms import AGENT_LOOP  # noqa: PLC0415
    from aieng.advanced_rag.agents.corpus_tools import CorpusToolServer  # noqa: PLC0415
    from aieng.advanced_rag.agents.toolset import tool_declarations  # noqa: PLC0415

    session = CorpusToolServer(CORPUS, search=lambda q, k: []).session(STANDARD_ARMS["C5"])
    declared = tool_declarations(session.tools)
    assert [d["name"] for d in declared] == [t.__name__ for t in session.tools]
    assert all(d.get("description") for d in declared)
    assert AGENT_LOOP == "adk"
