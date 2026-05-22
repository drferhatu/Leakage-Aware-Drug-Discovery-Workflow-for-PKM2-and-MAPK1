#!/usr/bin/env python3
"""Plot repeated-seed robustness summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_LABELS = {
    "logreg_balanced": "Logistic regression",
    "random_forest_balanced": "Random forest",
    "lightgbm_weighted": "LightGBM",
    "xgboost_weighted": "XGBoost",
}

MODEL_ORDER = ["Logistic regression", "Random forest", "LightGBM", "XGBoost"]
METRIC_LABELS = {
    "ap": "Average precision",
    "ef_1pct": "EF1%",
    "bedroc20": "BEDROC20",
}
SPLIT_PALETTE = {"random": "#4c78a8", "scaffold": "#f58518"}


def set_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#2d3748",
            "axes.labelcolor": "#1a202c",
            "xtick.color": "#1a202c",
            "ytick.color": "#1a202c",
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        },
    )


def prepare(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["model_label"] = summary["model"].map(MODEL_LABELS)
    summary["model_label"] = pd.Categorical(summary["model_label"], MODEL_ORDER, ordered=True)
    return summary


def plot(summary: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["ap", "ef_1pct", "bedroc20"]
    targets = ["PKM2", "MAPK1"]
    fig, axes = plt.subplots(
        len(metrics),
        len(targets),
        figsize=(10.5, 8.2),
        sharex=True,
        constrained_layout=True,
    )
    offsets = {"random": -0.14, "scaffold": 0.14}
    for row_idx, metric in enumerate(metrics):
        for col_idx, target in enumerate(targets):
            ax = axes[row_idx, col_idx]
            subset = summary[summary["target"] == target]
            for split, split_subset in subset.groupby("split"):
                color = SPLIT_PALETTE[split]
                for _, row in split_subset.iterrows():
                    x = MODEL_ORDER.index(str(row["model_label"])) + offsets[split]
                    mean = float(row[f"{metric}_mean"])
                    std = float(row[f"{metric}_std"])
                    ax.errorbar(
                        x,
                        mean,
                        yerr=std,
                        fmt="o",
                        color=color,
                        ecolor=color,
                        elinewidth=1.3,
                        capsize=3,
                        markersize=5.2,
                        markeredgecolor="white",
                        markeredgewidth=0.7,
                        zorder=3,
                    )
                    ax.vlines(x, 0, mean, color=color, alpha=0.16, linewidth=2)
            ax.set_title(target if row_idx == 0 else "")
            ax.set_ylabel(METRIC_LABELS[metric] if col_idx == 0 else "")
            ax.set_xlabel("")
            ax.set_xticks(range(len(MODEL_ORDER)))
            ax.set_xticklabels(MODEL_ORDER, rotation=28, ha="right")
            if row_idx != len(metrics) - 1:
                ax.set_xticklabels([])
            ax.grid(axis="x", visible=False)
            ax.set_ylim(bottom=0)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=SPLIT_PALETTE["random"], markersize=6),
        plt.Line2D([0], [0], marker="o", linestyle="", color=SPLIT_PALETTE["scaffold"], markersize=6),
    ]
    fig.legend(
        handles,
        ["Random split", "Scaffold split"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.035),
    )
    fig.suptitle("Repeated-seed model robustness (mean +/- SD)", y=1.08, fontsize=13)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig07_repeated_seed_robustness.png", dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / "fig07_repeated_seed_robustness.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="scripts/data/processed/repeated_seed/repeated_seed_ml_summary.csv",
    )
    parser.add_argument("--out-dir", default="scripts/figures/model_evaluation")
    args = parser.parse_args()

    set_style()
    summary = prepare(pd.read_csv(args.summary))
    plot(summary, Path(args.out_dir))
    print(f"Wrote repeated-seed figure to: {args.out_dir}")


if __name__ == "__main__":
    main()

