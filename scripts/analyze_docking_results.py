#!/usr/bin/env python3
"""Merge Vina scores with ML/ADMET priorities and plot docking diagnostics."""

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


def robust_minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    values = series.astype(float)
    lo, hi = values.min(), values.max()
    if hi == lo:
        scaled = pd.Series(0.5, index=series.index)
    else:
        scaled = (values - lo) / (hi - lo)
    return 1 - scaled if invert else scaled


def merge_results(candidates: pd.DataFrame, docking: pd.DataFrame) -> pd.DataFrame:
    merged = candidates.merge(docking, on=["target", "docking_id"], how="left")
    merged["docking_support"] = merged.groupby("target")["best_vina_score_kcal_mol"].transform(
        lambda s: robust_minmax(s, invert=True)
    )
    merged["final_consensus_score"] = (
        0.35 * merged.groupby("target")["ensemble_calibrated_mean"].transform(robust_minmax)
        + 0.25 * merged.groupby("target")["admet_support_score"].transform(robust_minmax)
        + 0.25 * merged["docking_support"]
        + 0.15 * (1 - merged.groupby("target")["toxicity_burden"].transform(robust_minmax))
    )
    merged["consensus_rank"] = (
        merged.groupby("target")["final_consensus_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return merged.sort_values(["target", "consensus_rank"])


def plot_docking_panel(frame: pd.DataFrame, out_dir: Path) -> None:
    palette = {"PKM2": "#15616d", "MAPK1": "#c75000"}
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), constrained_layout=True)

    sns.scatterplot(
        data=frame,
        x="ensemble_calibrated_mean",
        y="best_vina_score_kcal_mol",
        hue="target",
        size="admet_support_score",
        sizes=(50, 180),
        style="label",
        palette=palette,
        edgecolor="white",
        linewidth=0.7,
        legend=False,
        ax=axes[0],
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Calibrated ensemble score")
    axes[0].set_ylabel("Vina score (kcal/mol, lower is better)")
    axes[0].set_title("Docking support vs ligand-based score")

    ranked = frame[frame["consensus_rank"] <= 5].sort_values(["target", "consensus_rank"]).copy()
    ranked["candidate"] = ranked["target"] + "-r" + ranked["consensus_rank"].astype(str)
    sns.barplot(
        data=ranked,
        y="candidate",
        x="final_consensus_score",
        hue="target",
        palette=palette,
        dodge=False,
        ax=axes[1],
    )
    axes[1].set_xlabel("Final consensus score")
    axes[1].set_ylabel("")
    axes[1].set_title("Consensus-ranked docking subset")
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].tick_params(axis="y", labelsize=7)

    fig.suptitle("Structure-aware triage after ML, uncertainty, and ADMET filtering", y=1.06, fontsize=12)
    fig.savefig(out_dir / "fig16_docking_consensus_panel.png", dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / "fig16_docking_consensus_panel.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        default="scripts/data/processed/docking_candidates/docking_candidates_top8_per_target.csv",
    )
    parser.add_argument("--docking", default="scripts/docking/results/vina_docking_summary.csv")
    parser.add_argument("--out-dir", default="scripts/docking/analysis")
    parser.add_argument("--fig-dir", default="scripts/figures/docking")
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates)
    docking = pd.read_csv(args.docking)
    merged = merge_results(candidates, docking)

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    merged.to_csv(out_dir / "docking_ml_admet_consensus.csv", index=False)
    top = merged[merged["consensus_rank"] <= 5].copy()
    top.to_csv(out_dir / "final_consensus_top5_per_target.csv", index=False)
    set_style()
    plot_docking_panel(merged, fig_dir)

    cols = [
        "target",
        "consensus_rank",
        "docking_id",
        "label",
        "ensemble_calibrated_mean",
        "admet_support_score",
        "toxicity_burden",
        "best_vina_score_kcal_mol",
        "final_consensus_score",
    ]
    print(top[cols].to_string(index=False))
    print(f"\nWrote docking analysis to: {out_dir}")


if __name__ == "__main__":
    main()
