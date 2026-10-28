"""Plot the saved paired full-dataset comparison without running any models."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    """Export a standalone scientific figure from audited confidence intervals."""
    import matplotlib.pyplot as plt  # noqa: PLC0415 - optional plotting dependency

    output = Path(__file__).resolve().parents[1] / "outputs/full_evaluation"
    report = json.loads((output / "comparison.json").read_text())
    groups = ["overall", "2", "3", "4"]
    labels = [
        f"{g.capitalize() if g == 'overall' else g + ' hops'} (n={report['groups'][g]['dense']['questions']:,})"
        for g in groups
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True, layout="constrained")
    for ax, metric, title in zip(
        axes, ("em", "f1", "recall_at_5"), ("Exact match", "Token F1", "Support recall@5"), strict=True
    ):
        for row, group in enumerate(groups):
            interval = report["groups"][group]["paired_differences"][metric]
            center, low, high = [100 * interval[k] for k in ("difference", "low", "high")]
            ax.errorbar(center, row, xerr=[[center - low], [high - center]], fmt="o", color="#215e91", capsize=4)
        ax.axvline(0, color="#777777", linewidth=1, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Decomposition − dense (percentage points)")
        ax.set_yticks(range(len(groups)), labels)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
    axes[0].invert_yaxis()
    fig.suptitle("Full MuSiQue: paired differences and 95% question-bootstrap intervals")
    fig.savefig(output / "paired_differences.png", dpi=300, facecolor="white")
    fig.savefig(output / "paired_differences.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
