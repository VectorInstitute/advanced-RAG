"""Gather every build and evaluation into the committed results file.

The results hold counts, overlaps, scores, costs, and identifiers. They hold no
passage, post, or question text and no entity names, so the corpora's licences
travel with the corpora and not with this file.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "graphrag_technical_support"))

import stability as st  # noqa: E402


A3B = "Qwen3_6-35B-A3B"
#: The build every other build is compared with: same-day, same model, atlas-rag's
#: prompt at its default temperature.
ANCHOR = f"atlas-{A3B}-t0.7-r1"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(directory: Path) -> list[dict[str, Any]]:
    rows = [json.loads(x) for x in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines() if x]
    return sorted(rows, key=lambda r: r["id"])


def _build(directory: Path, raw: bool = False) -> list[st.Extraction]:
    out = []
    for rec in _records(directory):
        triples = rec.get("raw_triples") if raw and "raw_triples" in rec else rec["entity_relation_dict"]
        out.append(st.Extraction.from_raw(rec["id"], triples or [], failed=bool(rec.get("failed"))))
    return out


def _responses(directory: Path) -> dict[str, str]:
    return {r["id"]: r.get("entity_relation_output") or "" for r in _records(directory)}


def _response_identity(directories: Sequence[Path]) -> dict[str, Any]:
    """Average :func:`stability.response_identity` over every pair of builds."""
    raw = {d.name: _responses(d) for d in directories}
    pairs = [st.response_identity(raw[a], raw[b]) for a, b in combinations(sorted(raw), 2)]
    points = [p["divergence_point"]["median"] for p in pairs if p["divergence_point"].get("n")]
    return {
        "identical": round(statistics.fmean(p["identical"] for p in pairs), 4),
        "median_divergence_point": round(statistics.fmean(points), 4) if points else None,
    }


def _rows(path: Path) -> list[Any]:
    from musique_report import Row  # noqa: PLC0415

    return [
        Row(r["question_id"], r["hops"], tuple(r["ranked"]), float(r["em"]), float(r["f1"]))
        for r in _load(path)["rows"]
    ]


def _gold(per_hop: Mapping[int, int]) -> dict[str, tuple[str, ...]]:
    from run_stability import musique_subset  # noqa: PLC0415

    questions, _ = musique_subset(per_hop)
    return {q.question_id: q.supporting_ids for q in questions}


def _price(model: str, cost: Mapping[str, Any]) -> float | None:
    from aieng.advanced_rag import spend  # noqa: PLC0415
    from aieng.advanced_rag.metering import CallUsage  # noqa: PLC0415

    rate = spend.price_of(model, spend.price_table())
    if rate is None:
        return None
    usage = CallUsage(int(cost.get("prompt_tokens", 0)), int(cost.get("completion_tokens", 0)))
    return round(float(rate.cost_of(usage)), 4)


def _headline(summary: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group, entry in summary.items():
        out[group] = {
            "all_supporting_at_5": entry["all_supporting_recall_at_5"],
            "all_supporting_at_10": entry["all_supporting_recall_at_10"],
            "exact_match": entry["exact_match"],
            "f1": entry["f1"],
        }
    return out


# ---------------------------------------------------------------------------
# ServerFault
# ---------------------------------------------------------------------------


def serverfault_section(root: Path) -> dict[str, Any]:
    """Compare repeated ServerFault builds under each arm and way of reading them."""
    from run_stability import schema_conformance, serverfault_jobs, serverfault_views  # noqa: PLC0415

    directory = root / "serverfault"
    if not directory.is_dir():
        return {}
    jobs = serverfault_jobs()
    groups: dict[str, dict[str, Path]] = {}
    for d in sorted(directory.iterdir()):
        if not (d / "meta.json").is_file():
            continue
        name, rep = d.name.rsplit("-r", 1)
        groups.setdefault(name, {})[rep] = d
    out: dict[str, Any] = {}
    for name, reps in sorted(groups.items()):
        metas = {r: _load(d / "meta.json") for r, d in reps.items()}
        views = {r: serverfault_views(d, jobs) for r, d in reps.items()}
        entry: dict[str, Any] = {
            "arm": next(iter(metas.values()))["arm"],
            "model": next(iter(metas.values()))["model"],
            "temperature_sent": next(iter(metas.values()))["temperature_sent"],
            "repeats": sorted(reps),
            "windows": next(iter(metas.values()))["windows"],
            "calls_per_build": round(statistics.fmean(m["cost"]["calls"] for m in metas.values()), 1),
            "tokens_per_build": round(statistics.fmean(m["cost"]["total_tokens"] for m in metas.values())),
        }
        # "as_returned" keeps the model's labels; "folded" applies the RI #4 fold.
        readings = (
            (("as_returned", "raw"),) if entry["arm"] == "atlas" else (("as_returned", "raw"), ("folded", "folded"))
        )
        for label, reading in readings:
            builds = {r: v[reading] for r, v in views.items()}
            entry[label] = {
                "agreement": st.compare_many(builds)["mean"],
                "structure": {r: _small_structure(b) for r, b in builds.items()},
            }
        if entry["arm"] != "atlas":
            entry["schema_conformance"] = {r: schema_conformance(d) for r, d in reps.items()}
        out[name] = entry
    return out


def _small_structure(build: Sequence[st.Extraction]) -> dict[str, Any]:
    s: dict[str, Any] = st.structure(build)
    s.pop("top_hubs")
    return s


# ---------------------------------------------------------------------------
# MuSiQue
# ---------------------------------------------------------------------------


def musique_section(root: Path) -> dict[str, Any]:  # noqa: PLR0912, PLR0915
    """Describe every MuSiQue build, compare the groups, and attach the evaluations."""
    from musique_report import compare  # noqa: PLC0415
    from run_stability import SUBSET_PER_HOP  # noqa: PLC0415

    builds_dir = root / "musique" / "builds"
    evals = root / "musique" / "eval"
    gold = _gold(SUBSET_PER_HOP)
    names = sorted(
        d.name for d in builds_dir.iterdir() if (d / "records.jsonl").is_file() and (d / "meta.json").is_file()
    )
    loaded = {n: _build(builds_dir / n) for n in names}
    anchor = loaded.get(ANCHOR)
    rows = {n: _rows(evals / f"{n}.json") for n in names if (evals / f"{n}.json").is_file()}
    dense_rows = _rows(evals / "dense.json") if (evals / "dense.json").is_file() else None

    per_build: dict[str, Any] = {}
    for n in names:
        meta = _load(builds_dir / n / "meta.json")
        entry: dict[str, Any] = {
            "structure": st.structure(loaded[n]) | {"top_hubs": None},
            "model": meta.get("model"),
            "builder": meta.get("builder") or meta.get("derived"),
            "temperature_sent": meta.get("temperature_sent"),
            "endpoint": meta.get("endpoint"),
            "cost": meta.get("cost"),
            "cost_usd": _price(meta["model"], meta["cost"]) if meta.get("cost") and meta.get("model") else 0.0,
        }
        entry["structure"].pop("top_hubs")
        if anchor is not None and n != ANCHOR:
            entry["against_anchor"] = st.compare_builds(anchor, loaded[n])
            entry["against_anchor"]["degree_ks"] = st.degree_distance(anchor, loaded[n])
            entry["against_anchor"]["hub_overlap"] = st.hub_overlap(anchor, loaded[n])
        if n in rows:
            entry["retrieval"] = _headline(_load(evals / f"{n}.json")["summary"])
            entry["eval_cost"] = _load(evals / f"{n}.json")["cost"]
            if ANCHOR in rows and n != ANCHOR:
                entry["vs_anchor"] = compare(rows[n], rows[ANCHOR], gold)["all"]
                entry["ranking_vs_anchor"] = st.ranking_agreement(
                    {r.question_id: r.ranked for r in rows[n]}, {r.question_id: r.ranked for r in rows[ANCHOR]}
                )
            if dense_rows is not None:
                entry["vs_dense"] = compare(rows[n], dense_rows, gold)["all"]
        per_build[n] = entry

    def group(prefix: str) -> list[str]:
        return [n for n in names if n.startswith(prefix)]

    repeat_groups = {
        "atlas_A3B_t0.7_same_day": [n for n in group(f"atlas-{A3B}-t0.7-r") if not n.endswith("-r0")],
        "atlas_A3B_t0.7_with_reference": group(f"atlas-{A3B}-t0.7-r"),
        "atlas_A3B_t0": group(f"atlas-{A3B}-t0-r"),
        "schema_A3B_t0.7": group(f"schema-{A3B}-t0.7-r"),
    }
    for model in sorted({n.split("-t")[0].removeprefix("atlas-") for n in group("atlas-")} - {A3B}):
        reps = group(f"atlas-{model}-t")
        if len(reps) > 1:
            repeat_groups[f"atlas_{model}"] = reps
    repeats: dict[str, Any] = {}
    for label, members in repeat_groups.items():
        if len(members) < 2:
            continue
        repeats[label] = {
            "builds": members,
            "agreement": st.compare_many({n: loaded[n] for n in members})["mean"],
            "retrieval_spread": _spread([per_build[n].get("retrieval") for n in members]),
            "ranking_agreement": _pairwise_rankings({n: rows[n] for n in members if n in rows}),
            "response_identity": _response_identity([builds_dir / n for n in members]),
        }
        if label.startswith("schema"):
            raw = {n: _build(builds_dir / n, raw=True) for n in members}
            repeats[label]["agreement_before_fold"] = st.compare_many(raw)["mean"]
    if "atlas_A3B_t0.7_with_reference" in repeats:
        ref = f"atlas-{A3B}-t0.7-r0"
        same_day = list(repeat_groups["atlas_A3B_t0.7_same_day"])
        if ref in loaded:
            repeats["reference_vs_same_day"] = {
                "graph_labelled_edges": [
                    st.compare_builds(loaded[ref], loaded[n])["graph"]["labelled_edges"] for n in same_day
                ],
                "graph_nodes": [st.compare_builds(loaded[ref], loaded[n])["graph"]["nodes"] for n in same_day],
            }

    renamed_dir = root / "musique" / "renamed"
    if renamed_dir.is_dir():
        renamed = {d.name: _build(d) for d in sorted(renamed_dir.iterdir()) if (d / "meta.json").is_file()}
        if len(renamed) > 1:
            originals = [n for n in repeat_groups["atlas_A3B_t0.7_same_day"] if n in loaded]
            repeats["renamed_titles_A3B_t0.7"] = {
                "builds": sorted(renamed),
                "agreement": st.compare_many(renamed)["mean"],
                "same_settings_original_titles": st.compare_many({n: loaded[n] for n in originals})["mean"],
                "structure": {n: _small_structure(b) for n, b in renamed.items()},
            }

    return {
        "questions": len(gold),
        "per_hop": {str(k): v for k, v in SUBSET_PER_HOP.items()},
        "passages": len(next(iter(loaded.values()))) if loaded else 0,
        "anchor": ANCHOR,
        "dense": _headline(_load(evals / "dense.json")["summary"]) if (evals / "dense.json").is_file() else None,
        "builds": per_build,
        "repeats": repeats,
        "rows": {n: [r.to_dict() for r in rows[n]] for n in rows}
        | ({"dense": [r.to_dict() for r in dense_rows]} if dense_rows else {}),
    }


def greedy_section(root: Path) -> dict[str, Any] | None:
    """Summarise the greedy probe: how many distinct responses each setting gives."""
    from collections import defaultdict  # noqa: PLC0415

    path = root / "musique" / "greedy" / "responses.jsonl"
    if not path.is_file():
        return None
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            groups[r["arm"]][r["passage_id"]].append(r["response"])
    out: dict[str, Any] = {}
    for arm, by_passage in groups.items():
        distinct = [len(set(v)) for v in by_passage.values()]
        pairs = [st.response_identity({"d": x}, {"d": y}) for v in by_passage.values() for x, y in combinations(v, 2)]
        out[arm] = {
            "passages": len(by_passage),
            "repeats": len(next(iter(by_passage.values()))),
            "distinct_responses": st.describe([float(d) for d in distinct]),
            "passages_always_identical": round(sum(d == 1 for d in distinct) / len(distinct), 4),
            "identical_pairs": round(statistics.fmean(p["identical"] for p in pairs), 4),
        }
    return out


def _spread(retrievals: Sequence[Mapping[str, Any] | None]) -> dict[str, Any] | None:
    found = [r["all"] for r in retrievals if r]
    if len(found) < 2:
        return None
    return {
        k: {"min": min(f[k] for f in found), "max": max(f[k] for f in found)}
        for k in ("all_supporting_at_5", "all_supporting_at_10", "exact_match", "f1")
    }


def _pairwise_rankings(rows: Mapping[str, Sequence[Any]]) -> dict[str, Any] | None:
    if len(rows) < 2:
        return None
    values = []
    identical = []
    for a, b in combinations(sorted(rows), 2):
        agreement = st.ranking_agreement(
            {r.question_id: r.ranked for r in rows[a]}, {r.question_id: r.ranked for r in rows[b]}
        )
        values.append(agreement["top_k_jaccard"]["mean"])
        identical.append(agreement["identical_top_k"])
    return {
        "mean_top5_jaccard": round(statistics.fmean(values), 4),
        "mean_identical_top5": round(statistics.fmean(identical), 4),
    }


# ---------------------------------------------------------------------------
# LightRAG
# ---------------------------------------------------------------------------


def lightrag_section(root: Path) -> dict[str, Any]:
    """Compare LightRAG builds over the 60-question sub-corpus, one insert or five."""
    from musique_report import compare  # noqa: PLC0415
    from run_stability import LIGHTRAG_PER_HOP, lightrag_build  # noqa: PLC0415

    directory = root / "lightrag"
    if not directory.is_dir():
        return {}
    names = sorted(d.name for d in directory.iterdir() if (d / "meta.json").is_file())
    if not names:
        return {}
    gold = _gold(LIGHTRAG_PER_HOP)
    loaded = {n: lightrag_build(directory / n) for n in names}
    evals = directory / "eval"
    rows = {n: _rows(evals / f"{n}.json") for n in names if (evals / f"{n}.json").is_file()}
    out: dict[str, Any] = {"builds": {}, "pairs": {}}
    for n in names:
        meta = _load(directory / n / "meta.json")
        out["builds"][n] = {
            "mode": meta["mode"],
            "cost": meta["cost"],
            "wall_seconds": meta["wall_seconds"],
            "document_statuses": meta["document_statuses"],
            "structure": _small_structure(loaded[n]),
            "retrieval": _headline(_load(evals / f"{n}.json")["summary"]) if n in rows else None,
        }
    for a, b in combinations(names, 2):
        pair = st.compare_builds(loaded[a], loaded[b])
        pair["degree_ks"] = st.degree_distance(loaded[a], loaded[b])
        if a in rows and b in rows:
            pair["retrieval_difference"] = compare(rows[b], rows[a], gold)["all"]
            pair["ranking"] = st.ranking_agreement(
                {r.question_id: r.ranked for r in rows[a]}, {r.question_id: r.ranked for r in rows[b]}
            )
        out["pairs"][f"{a}|{b}"] = pair
    small = root / "musique" / "eval"
    if (small / "dense-small.json").is_file():
        out["dense"] = _headline(_load(small / "dense-small.json")["summary"])
    if (small / f"{ANCHOR}-small.json").is_file():
        out["hipporag2_anchor"] = _headline(_load(small / f"{ANCHOR}-small.json")["summary"])
    return out


def write_report(path: Path) -> dict[str, Any]:
    """Write the results file and return it."""
    from run_stability import READER, work_root  # noqa: PLC0415

    root = work_root()
    musique = musique_section(root)
    paid = sorted(
        {
            b["model"]
            for b in musique.get("builds", {}).values()
            if b.get("endpoint") == "vector-proxy-paid" and b.get("model")
        }
    )
    usd = round(sum(b.get("cost_usd") or 0.0 for b in musique.get("builds", {}).values()), 2)
    result = {
        "produced_by": {
            "model": A3B,
            "reader": READER,
            "build_models": sorted({b["model"] for b in musique.get("builds", {}).values() if b.get("model")}),
            "paid_build_models": paid,
            "embeddings": "BAAI/bge-small-en-v1.5 (local)",
            "endpoint": "vector-cluster and vector-proxy-paid",
            "producer": "implementations/special_topics/graph_build_stability/run_stability.py",
            "mode": "explore: one build per model unless listed as a repeat; repeats are independent generations",
            "independence": (
                "extraction calls go to the model with no response cache in front of them; "
                "each LightRAG build has its own empty response cache"
            ),
            "paid_build_usd_estimate": usd,
            "date": date.today().isoformat(),
        },
        "serverfault": serverfault_section(root),
        "musique": musique,
        "lightrag": lightrag_section(root),
        "greedy_probe": greedy_section(root),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return result


if __name__ == "__main__":
    from run_stability import OUT  # noqa: PLC0415

    write_report(OUT)
