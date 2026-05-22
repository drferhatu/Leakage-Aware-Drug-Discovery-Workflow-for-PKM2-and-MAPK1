#!/usr/bin/env python3
"""Create a publication-style workflow figure for the PKM2/MAPK1 study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PKM2_COLOR = "#15616d"
MAPK1_COLOR = "#c75000"
SLATE = "#475569"
LIGHT = "#f8fafc"
GRID = "#e2e8f0"
TEXT = "#0f172a"


def add_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body,
    facecolor,
    edgecolor,
    title_color="white",
    title_size=11,
    body_size=8.8,
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.1,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.03 * w,
        y + h - 0.22 * h,
        title,
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        va="top",
        ha="left",
        linespacing=1.15,
    )
    ax.text(
        x + 0.03 * w,
        y + h - 0.42 * h,
        body,
        fontsize=body_size,
        color=title_color,
        va="top",
        ha="left",
        linespacing=1.35,
    )


def draw_workflow(out_dir: Path) -> None:
    fig = plt.figure(figsize=(14.8, 7.1))
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.03,
        0.95,
        "Leakage-aware drug discovery workflow for PKM2 and MAPK1",
        fontsize=17.5,
        fontweight="bold",
        color=TEXT,
        ha="left",
        va="top",
    )
    ax.text(
        0.03,
        0.905,
        "The study combines dataset auditing, scaffold-aware benchmarking, calibrated ranking, ADMET triage, docking and pose-level contact checks.",
        fontsize=10.2,
        color=SLATE,
        ha="left",
        va="top",
    )

    # Top row
    add_box(
        ax,
        0.03,
        0.64,
        0.28,
        0.17,
        "1. Target extraction and curation",
        "LIT-PCBA subsets for PKM2 and MAPK1\nCanonical SMILES audit\nDuplicates and label conflicts removed",
        facecolor=LIGHT,
        edgecolor=GRID,
        title_color=TEXT,
        title_size=11.2,
        body_size=8.8,
    )
    add_box(
        ax,
        0.35,
        0.64,
        0.28,
        0.17,
        "2. EDA and scaffold mapping",
        "Descriptor distributions\nClass imbalance inspection\nBemis-Murcko scaffold coverage",
        facecolor=LIGHT,
        edgecolor=GRID,
        title_color=TEXT,
        title_size=11.2,
        body_size=8.8,
    )
    add_box(
        ax,
        0.67,
        0.64,
        0.28,
        0.17,
        "3. Benchmark, modelling and scoring",
        "Nearest-active Tanimoto baseline\nRandom/scaffold split evaluation\nAP, EF1%, BEDROC20 and repeated seeds",
        facecolor=LIGHT,
        edgecolor=GRID,
        title_color=TEXT,
        title_size=10.9,
        body_size=8.6,
    )

    # Middle row
    add_box(
        ax,
        0.18,
        0.37,
        0.28,
        0.18,
        "4. Calibration and uncertainty",
        "Isotonic probability mapping\nECE10, Brier score, log-loss\nEnsemble disagreement as ranking support",
        facecolor="#ecfeff",
        edgecolor="#a5f3fc",
        title_color=TEXT,
        title_size=11.2,
        body_size=8.8,
    )
    add_box(
        ax,
        0.54,
        0.37,
        0.34,
        0.18,
        "5. ADMET-aware shortlist and structure-aware triage",
        "Drug-likeness filters\nADMET-AI support score\nAutoDock Vina docking and residue-contact checks",
        facecolor="#f0fdf4",
        edgecolor="#bbf7d0",
        title_color=TEXT,
        title_size=10.9,
        body_size=8.7,
    )

    # Outcome bands
    add_box(
        ax,
        0.05,
        0.07,
        0.40,
        0.13,
        "PKM2 branch",
        "Stronger scaffold-level ML gains\nCoherent support across calibration, ADMET and docking",
        facecolor=PKM2_COLOR,
        edgecolor=PKM2_COLOR,
        title_size=12.4,
        body_size=9.8,
    )
    add_box(
        ax,
        0.55,
        0.07,
        0.40,
        0.13,
        "MAPK1 branch",
        "Biologically important contrast target\nSimilarity remains competitive and downstream triage becomes more decisive",
        facecolor=MAPK1_COLOR,
        edgecolor=MAPK1_COLOR,
        title_size=12.4,
        body_size=9.8,
    )

    # Side note
    note = FancyBboxPatch(
        (0.72, 0.855),
        0.24,
        0.09,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=0.9,
        facecolor="#ffffff",
        edgecolor=GRID,
    )
    ax.add_patch(note)
    ax.text(
        0.735,
        0.902,
        "Key principle",
        fontsize=9.6,
        fontweight="bold",
        color=TEXT,
        va="top",
        ha="left",
    )
    ax.text(
        0.735,
        0.872,
        "No single score is treated\nas a discovery claim.",
        fontsize=8.4,
        color=SLATE,
        va="top",
        ha="left",
    )

    # Section labels to preserve flow without arrows
    ax.text(
        0.03,
        0.595,
        "Core workflow",
        fontsize=9.4,
        fontweight="bold",
        color=SLATE,
        ha="left",
        va="center",
    )
    ax.text(
        0.03,
        0.325,
        "Downstream triage",
        fontsize=9.4,
        fontweight="bold",
        color=SLATE,
        ha="left",
        va="center",
    )
    ax.text(
        0.03,
        0.235,
        "Target-specific interpretation",
        fontsize=9.4,
        fontweight="bold",
        color=SLATE,
        ha="left",
        va="center",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig00_workflow_overview.png", dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / "fig00_workflow_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="scripts/figures/workflow")
    args = parser.parse_args()
    draw_workflow(Path(args.out_dir))


if __name__ == "__main__":
    main()
