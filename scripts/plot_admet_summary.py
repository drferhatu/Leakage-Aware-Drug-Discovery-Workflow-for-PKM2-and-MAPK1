#!/usr/bin/env python3
"""Plot ADMET-aware prioritization diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def plot_activity_admet_space(frame: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True, constrained_layout=True)
    palette = {"PKM2": "#15616d", "MAPK1": "#c75000"}
    for ax, target in zip(axes, ["PKM2", "MAPK1"]):
        target_frame = frame[frame["target"] == target]
        sns.scatterplot(
            data=target_frame,
            x="ensemble_calibrated_mean",
            y="admet_support_score",
            size="toxicity_burden",
            sizes=(35, 165),
            style="label",
            color=palette[target],
            alpha=0.86,
            edgecolor="white",
            linewidth=0.6,
            legend=False,
            ax=ax,
        )
        top = target_frame[target_frame["admet_priority_rank"] <= 3]
        for _, row in top.iterrows():
            x_offset = 4 if target == "PKM2" else -10
            y_offset = 5 if target == "PKM2" else 4
            ax.annotate(
                f"r{int(row['admet_priority_rank'])}",
                xy=(row["ensemble_calibrated_mean"], row["admet_support_score"]),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="left" if target == "PKM2" else "right",
                va="bottom",
                fontsize=7,
                color="#1a202c",
            )
        x_values = target_frame["ensemble_calibrated_mean"]
        x_pad = max((x_values.max() - x_values.min()) * 0.08, 0.002)
        ax.set_xlim(x_values.min() - x_pad, x_values.max() + x_pad)
        ax.set_title(target)
        ax.set_xlabel("Calibrated ensemble score")
        ax.set_ylabel("ADMET support score" if target == "PKM2" else "")
    fig.suptitle("ADMET-aware prioritization of scaffold-diverse candidates", y=1.05, fontsize=12)
    fig.savefig(out_dir / "fig13_admet_activity_space.png", dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / "fig13_admet_activity_space.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_activity_admet_shared_axis(frame: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    palette = {"PKM2": "#15616d", "MAPK1": "#c75000"}
    sns.scatterplot(
        data=frame,
        x="ensemble_calibrated_mean",
        y="admet_support_score",
        hue="target",
        style="label",
        size="toxicity_burden",
        sizes=(35, 170),
        palette=palette,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )
    ax.set_xlim(0, 0.88)
    ax.set_xlabel("Calibrated ensemble score (shared x-axis)")
    ax.set_ylabel("ADMET support score")
    ax.set_title("ADMET-aware prioritization on a shared calibrated-score scale")
    ax.legend(
        title="Target / label / toxicity",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0,
        fontsize=8,
        title_fontsize=8,
    )
    fig.savefig(out_dir / "supp_fig_s1_admet_shared_xaxis.png", dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / "supp_fig_s1_admet_shared_xaxis.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_risk_heatmap(frame: pd.DataFrame, out_dir: Path, top_n: int) -> None:
    top = frame[frame["admet_priority_rank"] <= top_n].copy()
    top["candidate"] = top["target"] + "-r" + top["admet_priority_rank"].astype(int).astype(str).str.zfill(2)
    metrics = [
        "AMES",
        "DILI",
        "hERG",
        "ClinTox",
        "Skin_Reaction",
        "structural_alert_flags",
        "cyp_inhibition_flags",
    ]
    heat = top.set_index("candidate")[metrics]
    heat["structural_alert_flags"] = heat["structural_alert_flags"].clip(0, 3) / 3
    heat["cyp_inhibition_flags"] = heat["cyp_inhibition_flags"].clip(0, 5) / 5
    heat = heat.rename(
        columns={
            "Skin_Reaction": "Skin",
            "structural_alert_flags": "Alerts",
            "cyp_inhibition_flags": "CYP",
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    sns.heatmap(
        heat,
        cmap=sns.color_palette("rocket_r", as_cmap=True),
        vmin=0,
        vmax=1,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Predicted risk / normalized flags"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"ADMET risk profile for top {top_n} candidates per target")
    ax.tick_params(axis="x", rotation=28, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.savefig(out_dir / f"fig14_admet_risk_heatmap_top{top_n}.png", dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / f"fig14_admet_risk_heatmap_top{top_n}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_target_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        "median_admet_support",
        "median_toxicity_burden",
        "compounds_with_no_structural_alerts",
        "compounds_with_low_toxicity_flags",
    ]
    long = summary.melt(id_vars="target", value_vars=metrics, var_name="metric", value_name="value")
    labels = {
        "median_admet_support": "Median ADMET support",
        "median_toxicity_burden": "Median toxicity burden",
        "compounds_with_no_structural_alerts": "No structural alerts",
        "compounds_with_low_toxicity_flags": "Low toxicity flags",
    }
    long["metric"] = long["metric"].map(labels)

    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.4), constrained_layout=True)
    palette = {"PKM2": "#15616d", "MAPK1": "#c75000"}
    for ax, metric in zip(axes, labels.values()):
        sns.barplot(
            data=long[long["metric"] == metric],
            x="target",
            y="value",
            hue="target",
            palette=palette,
            legend=False,
            ax=ax,
        )
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Target-level ADMET profile of the scaffold-diverse shortlist", y=1.12, fontsize=12)
    fig.savefig(out_dir / "fig15_admet_target_summary.png", dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / "fig15_admet_target_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="scripts/data/processed/admet")
    parser.add_argument("--out-dir", default="scripts/figures/admet")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    set_style()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_dir / "combined_diverse_top30_admet_annotated.csv")
    summary = pd.read_csv(input_dir / "admet_target_summary.csv")
    plot_activity_admet_space(frame, out_dir)
    plot_activity_admet_shared_axis(frame, out_dir)
    plot_risk_heatmap(frame, out_dir, args.top_n)
    plot_target_summary(summary, out_dir)
    print(f"Wrote ADMET figures to: {out_dir}")


if __name__ == "__main__":
    main()
