"""Draw the reference-implementation diagrams in the root README as SVG.

Each diagram shows which retrieval elements a reference implementation (RI)
combines, as a chain of overlapping circles. The script uses only the standard
library and writes the same files on every run.

Run from the repository root::

    python scripts/make_doc_graphics.py

The files land in ``docs/assets/``. Pass ``--out <dir>`` to write somewhere else.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets"

# Each element keeps one colour in every diagram. Fills are dark enough to carry
# white text at 4.5:1, and the diagrams are opaque, so they read the same on
# GitHub's light and dark themes.
ELEMENTS = {
    "Lexical": "#313CFF",
    "Dense": "#0F6E82",
    "Rerank": "#5B5B66",
    "Tables": "#0B7350",
    "Graph": "#8A25C9",
    "Agent": "#C40072",
}

RI_ELEMENTS = {
    1: ("Lexical", "Dense", "Rerank"),
    2: ("Dense", "Agent", "Graph"),
    3: ("Tables", "Lexical", "Dense"),
    4: ("Dense", "Graph", "Agent"),
    5: ("Dense", "Agent", "Lexical"),
}

RI_TITLES = {
    1: "RI #1 combines lexical retrieval, dense retrieval, and reranking",
    2: "RI #2 combines dense retrieval, an agent loop, and a graph index",
    3: "RI #3 combines table-aware chunking, lexical retrieval, and dense retrieval",
    4: "RI #4 combines dense retrieval, a graph index, and an agent loop",
    5: "RI #5 gives an agent dense search and lexical grep as tools",
}


def ri_diagram(number: int) -> str:
    """Return one RI's chain of three overlapping circles."""
    names = RI_ELEMENTS[number]
    radius, step, cy = 27.0, 45.0, 30.0
    centres = [radius + 2 + i * step for i in range(len(names))]
    width = centres[-1] + radius + 2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} 60" width="{width:.0f}" height="60" '
        f'role="img" aria-labelledby="t{number}">',
        f'<title id="t{number}">{RI_TITLES[number]}</title>',
        "<defs>",
    ]
    for i, x in enumerate(centres[1:], start=1):
        parts.append(f'<clipPath id="k{number}{i}"><circle cx="{x:.0f}" cy="{cy:.0f}" r="{radius:.0f}"/></clipPath>')
    parts.append("</defs>")
    for name, x in zip(names, centres, strict=True):
        parts.append(f'<circle cx="{x:.0f}" cy="{cy:.0f}" r="{radius:.0f}" fill="{ELEMENTS[name]}"/>')
    # Overlaps are the earlier circle clipped to the later one, darkened.
    for i, x in enumerate(centres[:-1]):
        parts.append(
            f'<circle cx="{x:.0f}" cy="{cy:.0f}" r="{radius:.0f}" fill="#000" fill-opacity=".38" '
            f'clip-path="url(#k{number}{i + 1})"/>'
        )
    parts.append(
        "<g fill=\"#fff\" font-family=\"-apple-system,BlinkMacSystemFont,'Segoe UI','Open Sans',Helvetica,Arial,"
        'sans-serif" font-size="11" font-weight="600" text-anchor="middle">'
    )
    for name, x in zip(names, centres, strict=True):
        parts.append(f'<text x="{x:.0f}" y="{cy + 3.9:.1f}">{name}</text>')
    parts.append("</g></svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    """Write the five RI diagrams."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ASSETS, help="output directory (default: docs/assets)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for number in RI_ELEMENTS:
        (args.out / f"ri-{number}.svg").write_text(ri_diagram(number))


if __name__ == "__main__":
    main()
