#!/usr/bin/env python3
"""Plot model-vs-similarity evaluation panels for PKM2/MAPK1."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_LABELS = {
    "max_similarity": "Max-similarity",
    "logreg_balanced": "Logistic regression",
    "random_forest_balanced": "Random forest",
    "lightgbm_weighted": "LightGBM",
    "xgboost_weighted": "XGBoost",
}

MODEL_ORDER = [
    "Max-similarity",
    "Logistic regression",
    "Random forest",
    "LightGBM",
    "XGBoost",
]

METRIC_LABELS = {
    "ap": "Average precision",
    "ef_1pct": "EF1%",
    "bedroc20": "BEDROC20",
}

SPLIT_PALETTE = {
    "random": "#4c78a8",
    "scaffold": "#f58518",
}


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


def load_results(similarity_path: Path, ml_path: Path) -> pd.DataFrame:
    similarity = pd.read_csv(similarity_path)
    similarity["model"] = "max_similarity"
    ml = pd.read_csv(ml_path)
    frame = pd.concat([similarity, ml], ignore_index=True, sort=False)
    frame["model_label"] = frame["model"].map(MODEL_LABELS).fillna(frame["model"])
    frame["model_label"] = pd.Categorical(frame["model_label"], MODEL_ORDER, ordered=True)
    frame["split"] = pd.Categorical(frame["split"], ["random", "scaffold"], ordered=True)
    return frame


def plot_metric_grid(frame: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["ap", "ef_1pct", "bedroc20"]
    targets = ["PKM2", "MAPK1"]
    fig, axes = plt.subplots(
        len(metrics),
        len(targets),
        figsize=(10.8, 8.6),
        sharex=True,
        constrained_layout=True,
    )

    for row_idx, metric in enumerate(metrics):
        for col_idx, target in enumerate(targets):
            ax = axes[row_idx, col_idx]
            subset = frame[frame["target"] == target].copy()
            split_offsets = {"random": -0.14, "scaffold": 0.14}
            for split, split_subset in subset.groupby("split", observed=True):
                color = SPLIT_PALETTE[str(split)]
                for _, row in split_subset.iterrows():
                    model_idx = MODEL_ORDER.index(str(row["model_label"]))
                    x = model_idx + split_offsets[str(split)]
                    y = float(row[metric])
                    ax.vlines(
                        x,
                        0,
                        y,
                        color=color,
                        alpha=0.22,
                        linewidth=2.0,
                        zorder=1,
                    )
                    ax.scatter(
                        x,
                        y,
                        s=38,
                        color=color,
                        edgecolor="white",
                        linewidth=0.8,
                        zorder=2,
                    )
            ax.set_title(target if row_idx == 0 else "")
            ax.set_xlabel("")
            ax.set_ylabel(METRIC_LABELS[metric] if col_idx == 0 else "")
            ax.set_xticks(range(len(MODEL_ORDER)))
            ax.set_xticklabels(MODEL_ORDER, rotation=28, ha="right")
            ax.grid(axis="x", visible=False)
            if metric == "ap":
                ax.set_ylim(bottom=0)
            if metric == "bedroc20":
                ax.set_ylim(0, 0.62)
            if metric == "ef_1pct":
                ax.set_ylim(bottom=0)
            if row_idx != len(metrics) - 1:
                ax.set_xticklabels([])

            best_scaffold = subset[subset["split"] == "scaffold"].sort_values(metric, ascending=False).head(1)
            if not best_scaffold.empty:
                x_label = str(best_scaffold["model_label"].iloc[0])
                y_value = float(best_scaffold[metric].iloc[0])
                x = MODEL_ORDER.index(x_label) + split_offsets["scaffold"]
                ax.scatter(
                    [x],
                    [y_value],
                    s=115,
                    facecolor="none",
                    edgecolor="#1a202c",
                    linewidth=1.1,
                    zorder=4,
                )

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
    fig.suptitle(
        "Model ranking changes when analog similarity is controlled",
        y=1.08,
        fontsize=13,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig05_model_evaluation_panel.png", dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / "fig05_model_evaluation_panel.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_scaffold_delta(frame: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    metrics = ["ap", "ef_1pct", "bedroc20"]
    rows = []
    for target, target_frame in frame.groupby("target", observed=True):
        for model_label, model_frame in target_frame.groupby("model_label", observed=True):
            if set(model_frame["split"].astype(str)) != {"random", "scaffold"}:
                continue
            random_row = model_frame[model_frame["split"].astype(str) == "random"].iloc[0]
            scaffold_row = model_frame[model_frame["split"].astype(str) == "scaffold"].iloc[0]
            for metric in metrics:
                rows.append(
                    {
                        "target": target,
                        "model": model_label,
                        "metric": METRIC_LABELS[metric],
                        "random": random_row[metric],
                        "scaffold": scaffold_row[metric],
                        "delta_scaffold_minus_random": scaffold_row[metric] - random_row[metric],
                    }
                )
    deltas = pd.DataFrame(rows)
    deltas.to_csv(out_dir / "model_random_scaffold_deltas.csv", index=False)

    plot_data = deltas[deltas["metric"].isin(["Average precision", "BEDROC20"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True, constrained_layout=True)
    for ax, metric in zip(axes, ["Average precision", "BEDROC20"]):
        subset = plot_data[plot_data["metric"] == metric]
        sns.barplot(
            data=subset,
            x="delta_scaffold_minus_random",
            y="model",
            hue="target",
            palette={"PKM2": "#15616d", "MAPK1": "#c75000"},
            ax=ax,
        )
        ax.axvline(0, color="#1a202c", linewidth=1)
        ax.set_title(metric)
        ax.set_xlabel("Scaffold minus random")
        ax.set_ylabel("")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.savefig(out_dir / "fig06_random_scaffold_delta.png", dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / "fig06_random_scaffold_delta.pdf", bbox_inches="tight")
    plt.close(fig)
    return deltas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--similarity",
        default="scripts/data/processed/similarity_baseline_results.csv",
    )
    parser.add_argument("--ml", default="scripts/data/processed/ml_baseline_results.csv")
    parser.add_argument("--out-dir", default="scripts/figures/model_evaluation")
    args = parser.parse_args()

    set_style()
    out_dir = Path(args.out_dir)
    frame = load_results(Path(args.similarity), Path(args.ml))
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "combined_model_evaluation_results.csv", index=False)
    plot_metric_grid(frame, out_dir)
    plot_scaffold_delta(frame, out_dir)
    print(f"Wrote model evaluation figures to: {out_dir}")


if __name__ == "__main__":
    main()
