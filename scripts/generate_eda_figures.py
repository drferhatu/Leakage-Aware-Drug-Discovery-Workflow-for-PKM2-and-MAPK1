#!/usr/bin/env python3
"""Generate EDA tables and figures for the PKM2/MAPK1 LIT-PCBA study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


PALETTE = {
    "PKM2": "#15616d",
    "MAPK1": "#c75000",
    "Active": "#0b6e4f",
    "Inactive": "#7a8699",
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


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def mol_from_smiles(smiles: str):
    return Chem.MolFromSmiles(smiles)


def scaffold(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return "INVALID"
    Chem.RemoveStereochemistry(mol)
    try:
        value = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except RuntimeError:
        return "SCAFFOLD_ERROR"
    return value if value else "NO_SCAFFOLD"


def descriptors(smiles: str) -> dict[str, float]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return {}
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
        "Rings": rdMolDescriptors.CalcNumRings(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
    }


def load_or_build_descriptors(frame: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    rows = []
    for idx, row in frame.iterrows():
        if idx % 25000 == 0 and idx:
            print(f"Computed descriptors for {idx:,} molecules...")
        item = {
            "target": row["target"],
            "compound_id": row["compound_id"],
            "canonical_smiles": row["canonical_smiles"],
            "label": row["label"],
            "activity": "Active" if row["label"] == 1 else "Inactive",
            "scaffold": scaffold(row["canonical_smiles"]),
        }
        item.update(descriptors(row["canonical_smiles"]))
        rows.append(item)
    out = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return out


def balanced_sample(frame: pd.DataFrame, per_target_inactives: int, seed: int) -> pd.DataFrame:
    parts = []
    rng = np.random.default_rng(seed)
    for target, target_frame in frame.groupby("target"):
        active = target_frame[target_frame["label"] == 1]
        inactive = target_frame[target_frame["label"] == 0]
        n = min(per_target_inactives, len(inactive))
        sampled_inactive = inactive.sample(n=n, random_state=int(rng.integers(0, 1_000_000)))
        parts.extend([active, sampled_inactive])
    return pd.concat(parts, ignore_index=True)


def plot_dataset_landscape(raw_summary: pd.DataFrame, clean_summary: pd.DataFrame, out_dir: Path) -> None:
    clean = clean_summary.copy()
    clean_long = clean.melt(
        id_vars="target",
        value_vars=["actives", "inactives"],
        var_name="class",
        value_name="count",
    )
    clean_long["class"] = clean_long["class"].map({"actives": "Active", "inactives": "Inactive"})

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw={"width_ratios": [1.08, 1.12]})
    sns.barplot(
        data=clean_long,
        x="target",
        y="count",
        hue="class",
        palette={"Active": PALETTE["Active"], "Inactive": PALETTE["Inactive"]},
        ax=axes[0],
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Molecules, log scale")
    axes[0].set_title("Clean LIT-PCBA target sets")
    axes[0].legend(title="")

    flow_rows = []
    for _, raw in raw_summary.iterrows():
        clean_row = clean_summary[clean_summary["target"] == raw["target"]].iloc[0]
        flow_rows.extend(
            [
                {"target": raw["target"], "step": "Raw rows", "count": raw["rows"]},
                {
                    "target": raw["target"],
                    "step": "Duplicates removed",
                    "count": raw["duplicate_rows"],
                },
                {
                    "target": raw["target"],
                    "step": "Label conflicts removed",
                    "count": raw["label_conflicts"],
                },
                {"target": raw["target"], "step": "Clean rows", "count": clean_row["rows"]},
            ]
        )
    flow = pd.DataFrame(flow_rows)

    step_order = ["Raw rows", "Duplicates removed", "Label conflicts removed", "Clean rows"]
    target_order = ["MAPK1", "PKM2"]
    colors = {
        "Raw rows": "#dbeafe",
        "Duplicates removed": "#ede9fe",
        "Label conflicts removed": "#fee2e2",
        "Clean rows": "#ccfbf1",
    }
    text_colors = {
        "Raw rows": "#1e3a8a",
        "Duplicates removed": "#5b21b6",
        "Label conflicts removed": "#991b1b",
        "Clean rows": "#115e59",
    }
    ax = axes[1]
    for row_idx, target in enumerate(target_order):
        for col_idx, step in enumerate(step_order):
            count = int(
                flow[(flow["target"] == target) & (flow["step"] == step)]["count"].iloc[0]
            )
            rect = plt.Rectangle(
                (col_idx - 0.47, row_idx - 0.38),
                0.94,
                0.76,
                facecolor=colors[step],
                edgecolor="white",
                linewidth=1.4,
            )
            ax.add_patch(rect)
            ax.text(
                col_idx,
                row_idx,
                f"{count:,}",
                ha="center",
                va="center",
                fontsize=8.5,
                color=text_colors[step],
                fontweight="bold" if step in {"Raw rows", "Clean rows"} else "normal",
            )
    ax.set_xticks(range(len(step_order)))
    ax.set_xticklabels(["Raw", "Duplicate\nrows", "Label\nconflicts", "Clean"], rotation=0)
    ax.set_yticks(range(len(target_order)))
    ax.set_yticklabels(target_order)
    ax.set_ylim(len(target_order) - 0.5, -0.5)
    ax.set_xlim(-0.55, len(step_order) - 0.45)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Curation flow")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig01_dataset_landscape")


def plot_property_distributions(desc: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotB"]
    sample = balanced_sample(desc, per_target_inactives=3000, seed=42)
    long = sample.melt(
        id_vars=["target", "activity"],
        value_vars=metrics,
        var_name="descriptor",
        value_name="value",
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.0))
    for ax, metric in zip(axes.ravel(), metrics):
        subset = long[long["descriptor"] == metric]
        sns.violinplot(
            data=subset,
            x="target",
            y="value",
            hue="activity",
            split=True,
            inner="quartile",
            linewidth=0.8,
            palette={"Active": PALETTE["Active"], "Inactive": PALETTE["Inactive"]},
            ax=ax,
        )
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=10)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if not handles:
        handles = [
            plt.Line2D([0], [0], color=PALETTE["Active"], lw=8),
            plt.Line2D([0], [0], color=PALETTE["Inactive"], lw=8),
        ]
        labels = ["Active", "Inactive"]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=10.5,
    )
    fig.suptitle("Physicochemical distributions in balanced target samples", y=1.08, fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig02_property_distributions")


def plot_scaffold_landscape(desc: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    active = desc[desc["label"] == 1].copy()
    scaffold_counts = (
        active.groupby(["target", "scaffold"])
        .size()
        .reset_index(name="active_count")
        .sort_values(["target", "active_count"], ascending=[True, False])
    )
    scaffold_counts["rank"] = scaffold_counts.groupby("target")["active_count"].rank(
        method="first", ascending=False
    )
    top = scaffold_counts[scaffold_counts["rank"] <= 15].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for ax, target in zip(axes, ["PKM2", "MAPK1"]):
        data = top[top["target"] == target].sort_values("active_count", ascending=True)
        labels = [f"{target}-S{int(rank):02d}" for rank in data["rank"]]
        bars = ax.barh(labels, data["active_count"], color=PALETTE[target], alpha=0.88)
        ax.bar_label(
            bars,
            labels=[str(int(v)) for v in data["active_count"]],
            padding=3,
            fontsize=8,
        )
        ax.set_title(f"{target}: top active scaffolds")
        ax.set_xlabel("Active molecules")
        ax.set_ylabel("")
    fig.tight_layout()
    save_figure(fig, out_dir, "fig03_top_active_scaffolds")

    coverage_rows = []
    for target, group in scaffold_counts.groupby("target"):
        ordered = group.sort_values("active_count", ascending=False).reset_index(drop=True)
        total = ordered["active_count"].sum()
        ordered["cumulative_fraction"] = ordered["active_count"].cumsum() / total
        ordered["rank"] = np.arange(1, len(ordered) + 1)
        coverage_rows.append(ordered[["target", "rank", "cumulative_fraction"]])
    coverage = pd.concat(coverage_rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    sns.lineplot(
        data=coverage[coverage["rank"] <= 200],
        x="rank",
        y="cumulative_fraction",
        hue="target",
        palette={k: PALETTE[k] for k in ["PKM2", "MAPK1"]},
        linewidth=2.2,
        ax=ax,
    )
    ax.set_xlabel("Active scaffold rank")
    ax.set_ylabel("Cumulative active coverage")
    ax.set_title("How concentrated are active chemotypes?")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig04_active_scaffold_coverage")
    return scaffold_counts


def write_tables(desc: pd.DataFrame, scaffold_counts: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        desc.groupby(["target", "activity"])[["MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "Rings"]]
        .agg(["median", "mean", "std"])
        .round(3)
    )
    summary.to_csv(out_dir / "eda_descriptor_summary.csv")
    scaffold_counts.to_csv(out_dir / "eda_active_scaffold_counts.csv", index=False)
    (
        desc.groupby("target")
        .agg(
            molecules=("canonical_smiles", "count"),
            actives=("label", "sum"),
            unique_scaffolds=("scaffold", "nunique"),
            active_scaffolds=("scaffold", lambda s: s[desc.loc[s.index, "label"].eq(1)].nunique()),
        )
        .to_csv(out_dir / "eda_scaffold_summary.csv")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="scripts/data/processed/lit_pcba_pkm2_mapk1_clean.csv")
    parser.add_argument("--raw-summary", default="scripts/data/processed/lit_pcba_pkm2_mapk1_summary.csv")
    parser.add_argument(
        "--clean-summary",
        default="scripts/data/processed/lit_pcba_pkm2_mapk1_clean_summary.csv",
    )
    parser.add_argument("--out-dir", default="scripts/figures/eda")
    parser.add_argument(
        "--descriptor-cache",
        default="scripts/data/processed/lit_pcba_pkm2_mapk1_descriptors.csv",
    )
    args = parser.parse_args()

    set_style()
    out_dir = Path(args.out_dir)
    frame = pd.read_csv(args.input)
    raw_summary = pd.read_csv(args.raw_summary)
    clean_summary = pd.read_csv(args.clean_summary)

    desc = load_or_build_descriptors(frame, Path(args.descriptor_cache))
    plot_dataset_landscape(raw_summary, clean_summary, out_dir)
    plot_property_distributions(desc, out_dir)
    scaffold_counts = plot_scaffold_landscape(desc, out_dir)
    write_tables(desc, scaffold_counts, out_dir)

    print(f"Wrote EDA figures and tables to: {out_dir}")


if __name__ == "__main__":
    main()
