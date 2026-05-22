#!/usr/bin/env python3
"""Plot calibration and ensemble uncertainty diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_LABELS = {
    "lightgbm_weighted": "LightGBM",
    "xgboost_weighted": "XGBoost",
    "ensemble": "Ensemble",
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


def plot_calibration_curves(curves: pd.DataFrame, out_dir: Path) -> None:
    curves = curves.copy()
    curves["model_label"] = curves["model"].map(MODEL_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharex=True, sharey=True, constrained_layout=True)
    palette = {"LightGBM": "#15616d", "XGBoost": "#c75000"}
    for ax, target in zip(axes, ["PKM2", "MAPK1"]):
        subset = curves[curves["target"] == target]
        sns.lineplot(
            data=subset,
            x="mean_predicted",
            y="fraction_active",
            hue="model_label",
            marker="o",
            palette=palette,
            linewidth=1.8,
            ax=ax,
        )
        ax.plot([0, 1], [0, 1], color="#64748b", linestyle="--", linewidth=1)
        ax.set_title(target)
        ax.set_xlabel("Mean calibrated probability")
        ax.set_ylabel("Observed active fraction" if target == "PKM2" else "")
        ax.set_xlim(0, max(0.12, subset["mean_predicted"].max() * 1.08))
        ax.set_ylim(0, max(0.12, subset["fraction_active"].max() * 1.08))
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Calibration after isotonic probability mapping", y=1.15, fontsize=12)
    fig.savefig(out_dir / "fig08_calibration_curves.png", dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / "fig08_calibration_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty(predictions: pd.DataFrame, out_dir: Path) -> None:
    top_rows = []
    for target, group in predictions.groupby("target"):
        ranked = group.sort_values("ensemble_calibrated_mean", ascending=False).head(500).copy()
        ranked["rank"] = range(1, len(ranked) + 1)
        top_rows.append(ranked)
    top = pd.concat(top_rows, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), constrained_layout=True)
    palette = {0: "#94a3b8", 1: "#0b6e4f"}
    for ax, target in zip(axes, ["PKM2", "MAPK1"]):
        subset = top[top["target"] == target]
        sns.scatterplot(
            data=subset,
            x="ensemble_calibrated_mean",
            y="ensemble_calibrated_std",
            hue="label",
            palette=palette,
            alpha=0.72,
            s=22,
            edgecolor=None,
            ax=ax,
            legend=False,
        )
        ax.set_title(f"{target}: top 500 by ensemble score")
        ax.set_xlabel("Calibrated ensemble mean")
        ax.set_ylabel("Model disagreement" if target == "PKM2" else "")
    fig.suptitle("High-scoring molecules can be separated by ensemble uncertainty", y=1.08, fontsize=12)
    fig.savefig(out_dir / "fig09_uncertainty_scatter.png", dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / "fig09_uncertainty_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_change(metrics: pd.DataFrame, out_dir: Path) -> None:
    subset = metrics[metrics["model"] != "ensemble"].copy()
    subset["model_label"] = subset["model"].map(MODEL_LABELS)
    metric_long = subset.melt(
        id_vars=["target", "model_label", "score_type"],
        value_vars=["brier", "log_loss", "ece10"],
        var_name="metric",
        value_name="value",
    )
    metric_long["metric"] = metric_long["metric"].map(
        {"brier": "Brier score", "log_loss": "Log loss", "ece10": "ECE10"}
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
    for ax, metric in zip(axes, ["Brier score", "Log loss", "ECE10"]):
        sns.barplot(
            data=metric_long[metric_long["metric"] == metric],
            x="model_label",
            y="value",
            hue="score_type",
            palette={"raw": "#94a3b8", "calibrated": "#14b8a6"},
            ax=ax,
        )
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["Raw", "Calibrated"], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Calibration improves probability quality", y=1.15, fontsize=12)
    fig.savefig(out_dir / "fig10_calibration_metric_change.png", dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / "fig10_calibration_metric_change.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="scripts/data/processed/calibration")
    parser.add_argument("--out-dir", default="scripts/figures/calibration")
    args = parser.parse_args()

    set_style()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = pd.read_csv(input_dir / "calibration_curves.csv")
    predictions = pd.read_csv(input_dir / "calibrated_test_predictions.csv")
    metrics = pd.read_csv(input_dir / "calibration_metrics.csv")
    plot_calibration_curves(curves, out_dir)
    plot_uncertainty(predictions, out_dir)
    plot_metric_change(metrics, out_dir)
    print(f"Wrote calibration figures to: {out_dir}")


if __name__ == "__main__":
    main()

