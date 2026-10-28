"""Draw the results figures in the READMEs from committed outputs.

Every number comes from a JSON file under an RI's ``outputs/`` folder. The RI #1
lever figure rescores the committed rankings in
``regulatory_policy_qa/outputs/baseline_cache`` against the CUAD and ContractNLI
answer spans, so it needs those two corpora on disk
(``python -m aieng.advanced_rag.corpora.download 1``) and no model.

Each figure is written twice, for GitHub's light and dark themes, as
``docs/assets/<name>-light.png`` and ``docs/assets/<name>-dark.png``. The READMEs
pick one with a ``<picture>`` element.

matplotlib is not a project dependency. Run from the repository root::

    uv run --with matplotlib python scripts/make_doc_figures.py

Pass figure names to draw a subset: ``ri1-levers``, ``ri2-shared-300``,
``ri4-graph-vs-dense``, ``ri5-scale-ladder``, ``visual-documents``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl


mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from aieng.advanced_rag.cache import CacheKey, TieredResultCache  # noqa: E402
from aieng.advanced_rag.corpora import contract_nli, cuad  # noqa: E402
from aieng.advanced_rag.evaluation.metrics import score_retrieval  # noqa: E402
from aieng.advanced_rag.retrieval.retrievers import run_retriever  # noqa: E402
from aieng.advanced_rag.retrieval.sparse import BM25Retriever, weighted_rrf  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
IMPL = REPO / "implementations"
ASSETS = REPO / "docs" / "assets"


@dataclass(frozen=True)
class Theme:
    """Surface, ink, and series colours for one GitHub theme."""

    name: str
    surface: str
    text: str
    muted: str
    grid: str
    # Brand blue, pink, and a teal step, checked for colour-vision separation
    # and 3:1 contrast against this surface.
    series: tuple[str, str, str]
    quiet: str


LIGHT = Theme("light", "#ffffff", "#1f2328", "#59636e", "#d1d9e0", ("#313CFF", "#EB088A", "#1C8FA8"), "#a2a9b1")
DARK = Theme("dark", "#0d1117", "#f0f6fc", "#9198a1", "#3d444d", ("#7C83FF", "#F0479F", "#2299B0"), "#656c76")


def load(relative: str) -> Any:
    """Read one committed output file."""
    return json.loads((IMPL / relative).read_text())


def style(theme: Theme) -> None:
    """Set matplotlib defaults for one theme."""
    families = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update(
        {
            "font.family": "Open Sans" if "Open Sans" in families else "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "text.color": theme.text,
            "axes.labelcolor": theme.muted,
            "axes.edgecolor": theme.grid,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlecolor": theme.text,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "xtick.labelcolor": theme.muted,
            "ytick.labelcolor": theme.text,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "lines.solid_capstyle": "round",
        }
    )


def save(fig: Figure, path: Path) -> None:
    """Write a PNG reduced to a 256-colour palette, which keeps it small."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)


# --------------------------------------------------------------------- RI #1


def ri1_lever_scores() -> list[tuple[str, float, float]]:  # noqa: PLR0915
    """Rescore notebook 02's lever table.

    Each row is (lever, CUAD 20 failure at 10, ContractNLI 20 MRR).
    """
    k, depth = 10, 50
    shallow = {"CUAD 20": 10, "ContractNLI 20": 3}
    cuad_path, nli_path = cuad.fetch_cuad_test(), contract_nli.fetch_contract_nli()
    slices = {
        "CUAD 20": (cuad_path, "cuad", cuad.build_slice(cuad_path, max_documents=20)),
        "ContractNLI 20": (nli_path, "contract-nli", contract_nli.build_slice(nli_path, max_documents=20)),
    }
    # An empty local tier, so only the committed measurements are read.
    with tempfile.TemporaryDirectory() as empty:
        cache = TieredResultCache(
            global_dir=IMPL / "regulatory_policy_qa" / "outputs" / "baseline_cache", local_dir=Path(empty)
        )

        def config(name: str) -> dict[str, Any]:
            path, corpus, _ = slices[name]
            return {
                "corpus": corpus,
                "corpus_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "max_documents": 20,
                "chunk_chars": 1200,
                "overlap_chars": 200,
                "scope": "contract",
            }

        def committed(key: CacheKey) -> Any:
            found = cache.get(key)
            if found is None:
                raise FileNotFoundError(f"no committed artifact for {key}")
            return found.value

        models = {
            "dense": "BAAI/bge-small-en-v1.5",
            "splade": "prithivida/Splade_PP_en_v1",
            "colbert": "answerdotai/answerai-colbert-small-v1",
        }
        rerankers = {
            "MiniLM-L6": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
            "ColBERT MaxSim": "answerdotai/answerai-colbert-small-v1",
        }
        rankings: dict[str, dict[str, dict[str, list[str]]]] = {}
        rerank_scores: dict[tuple[str, str], dict[str, list[float]]] = {}
        for name, (_, _, prepared) in slices.items():
            chunks = prepared.chunks
            bm25 = BM25Retriever()
            bm25.index(chunks)
            rankings[name] = {"bm25": run_retriever(bm25, prepared.answerable, k=depth)}
            for arm, model in models.items():
                key = CacheKey(
                    "ri1",
                    "lever-rankings",
                    model.split("/")[-1],
                    {**config(name), "arm": arm, "model": model, "depth": depth},
                )
                runs = committed(key)["runs"]
                rankings[name][arm] = {q: [chunks[i].chunk_id for i in ids] for q, ids in runs.items()}
            for label, model in rerankers.items():
                key = CacheKey("ri1", "rerank-scores", model.split("/")[-1], {**config(name), "candidates": "contract"})
                rerank_scores[(label, name)] = committed(key)["scores"]

    def score(runs: dict[str, list[str]], name: str) -> tuple[float, float]:
        qrels = slices[name][2].qrels
        failure = score_retrieval({q: ids[: shallow[name]] for q, ids in runs.items()}, qrels, k=shallow[name])
        mrr = score_retrieval({q: ids[:k] for q, ids in runs.items()}, qrels, k=k).mrr
        return failure.retrieval_failure_rate, mrr

    def fuse(arms: list[dict[str, list[str]]], weights: list[float], limit: int) -> dict[str, list[str]]:
        return {
            q: [c for c, _ in weighted_rrf([arm[q][:depth] for arm in arms], weights, limit=limit)] for q in arms[0]
        }

    def rerank(first: dict[str, list[str]], label: str, name: str) -> dict[str, list[str]]:
        prepared = slices[name][2]
        by_doc: dict[str, list[str]] = {}
        for chunk in prepared.chunks:
            by_doc.setdefault(chunk.doc_id, []).append(chunk.chunk_id)
        doc_of = {query.query_id: query.doc_id for query in prepared.answerable}
        out = {}
        for q, ranked in first.items():
            lookup = dict(zip(by_doc[doc_of[q]], rerank_scores[(label, name)][q], strict=True))
            head = sorted(ranked[:20], key=lambda c: (-lookup[c], c))
            out[q] = head + [c for c in ranked[20:] if c not in head]
        return out

    levers: dict[str, Callable[[str], dict[str, list[str]]]] = {
        "BM25": lambda n: rankings[n]["bm25"],
        "Dense": lambda n: rankings[n]["dense"],
        "Hybrid BM25 + dense": lambda n: fuse([rankings[n]["dense"], rankings[n]["bm25"]], [0.2, 0.8], k),
        "SPLADE": lambda n: rankings[n]["splade"],
        "ColBERT": lambda n: rankings[n]["colbert"],
        "Dense + SPLADE": lambda n: fuse([rankings[n]["dense"], rankings[n]["splade"]], [0.3, 0.7], k),
    }
    rows = []
    for label, make in levers.items():
        rows.append((label, score(make("CUAD 20"), "CUAD 20")[0], score(make("ContractNLI 20"), "ContractNLI 20")[1]))
    for label in rerankers:
        runs = {
            n: rerank(fuse([rankings[n]["dense"], rankings[n]["splade"]], [0.3, 0.7], depth), label, n) for n in slices
        }
        rows.append(
            (
                f"Dense + SPLADE, {label}",
                score(runs["CUAD 20"], "CUAD 20")[0],
                score(runs["ContractNLI 20"], "ContractNLI 20")[1],
            )
        )
    return rows


def ri1_levers(theme: Theme, rows: list[tuple[str, float, float]]) -> Figure:
    """Dot plot of each retrieval lever on the two contract slices."""
    labels = [row[0] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
    panels = [
        ("CUAD 20 failure at 10\n(lower is better)", [row[1] for row in rows], min),
        ("ContractNLI 20 MRR\n(higher is better)", [row[2] for row in rows], max),
    ]
    ys = list(range(len(rows)))[::-1]
    for ax, (title, values, pick) in zip(axes, panels, strict=True):
        best = pick(values)
        colours = [theme.series[1] if v == best else theme.series[0] for v in values]
        ax.scatter(values, ys, s=46, c=colours, zorder=3, edgecolors=theme.surface, linewidths=1.5)
        for v, y in zip(values, ys, strict=True):
            if v == best:
                ax.annotate(f"{v:.3f}", (v, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        lo, hi = min(values), max(values)
        pad = (hi - lo) * 0.15
        ax.set_xlim(lo - pad, hi + pad)
    axes[0].set_yticks(ys, labels)
    fig.tight_layout(w_pad=2.5)
    return fig


# --------------------------------------------------------------------- RI #2


def ri2_shared_300(theme: Theme) -> Figure:
    """F1 minus dense with paired 95% intervals on the shared 300 MuSiQue questions."""
    data = load("cross_document_synthesis/outputs/evaluation/shared_300.json")["comparisons_with_dense"]
    names = {
        "ircot": "IRCoT",
        "hipporag2": "HippoRAG 2 (atlas-rag)",
        "decomposition": "Decomposition",
        "lightrag": "LightRAG (mix)",
    }
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    for y, key in enumerate(reversed(names)):
        f1 = data[f"{key}_minus_dense"]["token_f1"]
        clear = f1["low"] > 0
        colour = theme.series[1] if clear else theme.quiet
        ax.plot([f1["low"], f1["high"]], [y, y], color=colour, linewidth=2)
        ax.scatter([f1["mean"]], [y], s=46, color=colour, zorder=3, edgecolors=theme.surface, linewidths=1.5)
        ax.annotate(f"{f1['mean']:+.3f}", (f1["high"], y), xytext=(6, 0), textcoords="offset points", va="center")
    ax.axvline(0, color=theme.muted, linewidth=1)
    ax.set_yticks(range(len(names)), list(reversed(names.values())))
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.05, 0.33)
    ax.set_xlabel("F1 minus dense, with 95% paired interval")
    ax.set_title("Shared 300 MuSiQue questions, Qwen3_6-35B-A3B reader")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- RI #4


def ri4_graph_vs_dense(theme: Theme) -> Figure:
    """All-supporting recall at 10 and F1 by hop count for each RI #4 arm."""
    arms = load("graphrag_technical_support/outputs/musique_graph_test.json")["arms"]
    order = ["hipporag2", "dense", "dense-iterative", "hybrid", "lightrag"]
    colour = {"hipporag2": theme.series[1], "dense": theme.series[0]}
    hops = ["2hop", "3hop", "4hop"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    panels = [("All supporting passages in top 10", "all_supporting_recall_at_10"), ("Answer F1", "f1")]
    for ax, (title, metric) in zip(axes, panels, strict=True):
        for arm in reversed(order):
            values = [arms[arm]["summary"][h][metric] for h in hops]
            main = arm in colour
            ax.plot(
                [2, 3, 4],
                values,
                color=colour.get(arm, theme.quiet),
                linewidth=2 if main else 1.4,
                marker="o",
                markersize=5.5 if main else 4,
                markeredgecolor=theme.surface,
                markeredgewidth=1.2,
                zorder=3 if main else 2,
            )
            if main:
                ax.annotate(
                    arm, (2, values[0]), xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=9
                )
        ax.set_title(title)
        ax.set_xticks([2, 3, 4], ["2-hop", "3-hop", "4-hop"])
        ax.set_xlim(1.35, 4.2)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="y", labelcolor=theme.muted)
    axes[0].set_ylim(0, 0.66)
    others = ", ".join(order[2:])
    fig.text(0.01, -0.02, f"Grey lines: {others}.", color=theme.muted, fontsize=9)
    fig.tight_layout(w_pad=2.5)
    return fig


# --------------------------------------------------------------------- RI #5


def ri5_scale_ladder(theme: Theme) -> Figure:
    """F1 for C0, C5, and C7 at two corpus sizes on gemini-3.5-flash-lite."""
    sizes = load("agent_native_retrieval/outputs/musique_scale_ladder_gemini.json")["sizes"]
    arms = {"C0": theme.series[0], "C5": theme.series[1], "C7": theme.series[2]}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    panels = [("All 120 ladder questions", "all"), ("40 four-hop questions", "4hop")]
    for ax, (title, group) in zip(axes, panels, strict=True):
        for arm, colour in arms.items():
            values = [sizes[size]["arms"][arm]["summary"][group]["f1"] for size in ("11656", "139416")]
            ax.plot(
                [0, 1],
                values,
                color=colour,
                linewidth=2,
                marker="o",
                markersize=6,
                markeredgecolor=theme.surface,
                markeredgewidth=1.2,
            )
            ax.annotate(
                f"{arm} {values[1]:.3f}", (1, values[1]), xytext=(8, 0), textcoords="offset points", va="center"
            )
        ax.set_title(title)
        ax.set_xticks([0, 1], ["11,656 passages", "139,416 passages"])
        ax.set_xlim(-0.25, 1.45)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="y", labelcolor=theme.muted)
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0.4, 0.66)
    fig.tight_layout(w_pad=2.5)
    return fig


# ------------------------------------------------------------- visual documents


def visual_documents(theme: Theme) -> Figure:
    """Answer accuracy per arm on figure-only and table questions."""
    answers = load("special_topics/visual_documents/outputs/results.json")["answers"]
    arms = ["T", "E1", "E2", "L", "TV", "VT", "VV", "H"]
    pages = {"VT", "VV", "H"}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=True)
    panels = [("Figure-only questions (76)", "figure-only"), ("Table questions (60)", "table")]
    ys = list(range(len(arms)))[::-1]
    for ax, (title, group) in zip(axes, panels, strict=True):
        values = [answers[arm]["accuracy"][group] for arm in arms]
        colours = [theme.series[1] if arm in pages else theme.series[0] for arm in arms]
        ax.barh(ys, values, height=0.5, color=colours)
        best = max(values)
        for v, y in zip(values, ys, strict=True):
            ax.annotate(
                f"{v:.2f}",
                (v, y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                fontweight="bold" if v == best else "normal",
                color=theme.text if v == best else theme.muted,
            )
        ax.set_title(title)
        ax.set_xlim(0, 1.08)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys, arms)
    handles = [
        Rectangle((0, 0), 1, 1, color=theme.series[0], label="Text retrieval"),
        Rectangle((0, 0), 1, 1, color=theme.series[1], label="Page-image retrieval"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(w_pad=2.5, rect=(0, 0.04, 1, 1))
    return fig


FIGURES: dict[str, Callable[..., Figure]] = {
    "ri1-levers": ri1_levers,
    "ri2-shared-300": ri2_shared_300,
    "ri4-graph-vs-dense": ri4_graph_vs_dense,
    "ri5-scale-ladder": ri5_scale_ladder,
    "visual-documents": visual_documents,
}


def main() -> None:
    """Draw the requested figures in both themes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help=f"figures to draw, from {', '.join(FIGURES)} (default: all)")
    parser.add_argument("--out", type=Path, default=ASSETS, help="output directory (default: docs/assets)")
    args = parser.parse_args()
    unknown = set(args.names) - set(FIGURES)
    if unknown:
        parser.error(f"unknown figure: {', '.join(sorted(unknown))}")
    args.out.mkdir(parents=True, exist_ok=True)
    lever_rows = ri1_lever_scores() if not args.names or "ri1-levers" in args.names else []
    for row in lever_rows:
        print(f"{row[0]:<36}{row[1]:.4f}  {row[2]:.4f}")
    for name in args.names or FIGURES:
        for theme in (LIGHT, DARK):
            style(theme)
            fig = FIGURES[name](theme, lever_rows) if name == "ri1-levers" else FIGURES[name](theme)
            save(fig, args.out / f"{name}-{theme.name}.png")


if __name__ == "__main__":
    main()
