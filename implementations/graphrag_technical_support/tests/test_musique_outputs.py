"""Check the committed MuSiQue graph-test results for internal consistency."""

import json
from pathlib import Path

import pytest


OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
ARMS = ("dense", "dense-iterative", "hipporag2", "lightrag", "hybrid")


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads((OUTPUTS / "musique_graph_test.json").read_text())


def test_every_arm_covers_the_same_questions(results: dict) -> None:
    assert set(results["arms"]) == set(ARMS)
    assert set(results["rows"]) == set(ARMS)
    ids = [sorted(row["question_id"] for row in results["rows"][arm]) for arm in ARMS]
    assert all(group == ids[0] for group in ids)
    assert len(ids[0]) == results["test"]["questions"] == 300


def test_hop_groups_match_the_draw(results: dict) -> None:
    assert results["test"]["per_hop"] == {"2": 150, "3": 75, "4": 75}
    for arm in ARMS:
        summary = results["arms"][arm]["summary"]
        assert [summary[h]["questions"] for h in ("2hop", "3hop", "4hop")] == [150, 75, 75]
        assert summary["all"]["questions"] == 300


def test_summaries_follow_from_the_rows(results: dict) -> None:
    for arm in ARMS:
        rows = results["rows"][arm]
        summary = results["arms"][arm]["summary"]["all"]
        assert summary["f1"] == pytest.approx(sum(r["f1"] for r in rows) / len(rows), abs=1e-3)
        assert summary["exact_match"] == pytest.approx(sum(r["em"] for r in rows) / len(rows), abs=1e-3)
        assert all(len(r["ranked"]) <= results["test"]["rank_depth"] for r in rows)


def test_recall_at_ten_never_trails_recall_at_five(results: dict) -> None:
    for arm in ARMS:
        for group in results["arms"][arm]["summary"].values():
            assert group["all_supporting_recall_at_10"] >= group["all_supporting_recall_at_5"]
            assert group["mean_supporting_recall_at_10"] >= group["mean_supporting_recall_at_5"]


def test_cost_figures_agree(results: dict) -> None:
    for arm in ARMS:
        cost = results["arms"][arm]["cost"]
        assert cost["prompt_tokens"] + cost["completion_tokens"] == cost["total_tokens"]
        assert cost["calls_per_question"] == pytest.approx(cost["calls"] / 300, abs=0.01)


def test_comparisons_name_only_graph_arms_and_carry_no_text(results: dict) -> None:
    comparisons = results["comparisons_with_dense"]
    assert set(comparisons) == {f"{arm}_vs_dense" for arm in ARMS if arm != "dense"}
    text = (OUTPUTS / "musique_graph_test.json").read_text()
    assert '"question"' not in text
    assert '"passage_text"' not in text


def test_index_costs_are_consistent() -> None:
    hippo = json.loads((OUTPUTS / "musique_hipporag2_index_cost.json").read_text())
    light = json.loads((OUTPUTS / "musique_lightrag_index_cost.json").read_text())
    assert hippo["passages"] == light["passages"] == 11656
    assert hippo["cost"]["prompt_tokens"] + hippo["cost"]["completion_tokens"] == hippo["cost"]["total_tokens"]
    assert light["prompt_tokens"] + light["completion_tokens"] == light["total_tokens"]
    assert round(light["calls"] / light["passages"], 2) == light["calls_per_passage"]


def test_the_gemini_rerun_is_labelled_judged_and_text_free() -> None:
    path = OUTPUTS / "musique_graph_test_gemini.json"
    gemini = json.loads(path.read_text())
    assert gemini["produced_by"] == {
        "model": "gemini-3.5-flash-lite",
        "endpoint": "vector-proxy-paid",
        "producer": "run_musique_test.py",
        "label": "paid-model rerun",
        "judge": "gemini-3.8-flash",
    }
    assert set(gemini["arms"]) == {"dense", "dense-iterative", "hipporag2"}
    assert set(gemini["test"]["index_keys"]) == {"hipporag2"}
    for arm, entry in gemini["arms"].items():
        assert entry["summary"]["all"]["questions"] == 300
        assert 0.0 <= entry["judged"]["all"] <= 1.0
        rows = gemini["rows"][arm]
        assert entry["summary"]["all"]["f1"] == pytest.approx(sum(r["f1"] for r in rows) / len(rows), abs=1e-3)
    text = path.read_text()
    assert '"question"' not in text
    assert '"answer"' not in text


@pytest.mark.parametrize("reader", ["onprem", "gemini"])
def test_the_abstention_cost_runs_are_consistent_and_text_free(reader: str) -> None:
    path = OUTPUTS / f"musique_abstention_cost_{reader}.json"
    results = json.loads(path.read_text())
    assert results["produced_by"]["model"] == results["test"]["model"]
    assert results["test"]["questions"] == 300
    for arm, prompts in results["summary"].items():
        assert set(prompts) == {"standard", "best-guess", "flagged"}
        for prompt, modes in prompts.items():
            for mode, summary in modes.items():
                shares = summary["declined"] + summary["flagged"] + summary["answered_unflagged"]
                assert shares == pytest.approx(1.0, abs=1e-3)
                rows = results["per_question"][arm][f"{prompt}/{mode}"]
                assert len(rows) == summary["questions"] == 300
                assert summary["f1"] == pytest.approx(sum(r["f1"] for r in rows.values()) / 300, abs=1e-3)
            # Only the flagged prompt can mark a guess.
            if prompt != "flagged":
                assert all(m["flagged"] == 0 for m in modes.values())
    text = path.read_text()
    assert '"question"' not in text
    assert '"passage_text"' not in text
