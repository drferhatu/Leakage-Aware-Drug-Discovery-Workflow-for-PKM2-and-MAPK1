#!/usr/bin/env python3
"""Plot prioritized hit-selection diagnostics."""

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


def plot_priority_space(pool: pd.DataFrame, shortlist: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    for ax, target in zip(axes, ["PKM2", "MAPK1"]):
        target_pool = pool[pool["target"] == target]
        target_short = shortlist[shortlist["target"] == target]
        background = target_pool.sample(n=min(6000, len(target_pool)), random_state=42)
        ax.scatter(
            background["ensemble_calibrated_mean"],
            background["ensemble_calibrated_std"],
            s=7,
            color="#cbd5e1",
            alpha=0.28,
            linewidth=0,
            label="Screened pool",
        )
        ax.scatter(
            target_short["ensemble_calibrated_mean"],
            target_short["ensemble_calibrated_std"],
            s=34,
            color="#0b6e4f",
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
            label="Prioritized top 50",
        )
        ax.set_title(target)
        ax.set_xlabel("Calibrated ensemble mean")
        ax.set_ylabel("Model disagreement" if target == "PKM2" else "")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Prioritization favors high-score, low-disagreement molecules", y=1.15, fontsize=12)
    fig.savefig(out_dir / "fig11_prioritized_score_uncertainty.png", dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / "fig11_prioritized_score_uncertainty.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_property_profile(shortlist: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["mw", "logp", "tpsa", "rotb"]
    labels = {"mw": "MW", "logp": "LogP", "tpsa": "TPSA", "rotb": "Rotatable bonds"}
    long = shortlist.melt(
        id_vars=["target"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    long["metric"] = long["metric"].map(labels)
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.4), constrained_layout=True)
    for ax, metric in zip(axes, labels.values()):
        sns.boxenplot(
            data=long[long["metric"] == metric],
            x="target",
            y="value",
            palette={"PKM2": "#15616d", "MAPK1": "#c75000"},
            ax=ax,
        )
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Drug-likeness profile of prioritized top-ranked compounds", y=1.12, fontsize=12)
    fig.savefig(out_dir / "fig12_prioritized_property_profile.png", dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / "fig12_prioritized_property_profile.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(shortlist: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        shortlist.groupby("target")
        .agg(
            n=("compound_id", "count"),
            known_actives=("label", "sum"),
            median_priority_score=("priority_score", "median"),
            median_calibrated_score=("ensemble_calibrated_mean", "median"),
            median_disagreement=("ensemble_calibrated_std", "median"),
            lipinski_zero_viol=("lipinski_violations", lambda s: int((s == 0).sum())),
            veber_pass=("veber_pass", "sum"),
            median_mw=("mw", "median"),
            median_logp=("logp", "median"),
            median_tpsa=("tpsa", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "prioritized_top50_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="scripts/data/processed/prioritized_hits")
    parser.add_argument("--out-dir", default="scripts/figures/prioritized_hits")
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()

    set_style()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = pd.read_csv(input_dir / "calibrated_predictions_with_druglikeness.csv")
    shortlist = pd.read_csv(input_dir / f"combined_prioritized_top{args.top_n}.csv")
    plot_priority_space(pool, shortlist, out_dir)
    plot_property_profile(shortlist, out_dir)
    write_summary(shortlist, out_dir)
    print(f"Wrote prioritized-hit figures to: {out_dir}")


if __name__ == "__main__":
    main()

