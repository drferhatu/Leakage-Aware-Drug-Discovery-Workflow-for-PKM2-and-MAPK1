#!/usr/bin/env python3
"""Prioritize calibrated ML hits before ADMET and docking."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


def descriptors(smiles: str) -> dict[str, float | int | bool]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "valid": False,
            "mw": np.nan,
            "logp": np.nan,
            "tpsa": np.nan,
            "hbd": np.nan,
            "hba": np.nan,
            "rotb": np.nan,
            "rings": np.nan,
            "heavy_atoms": np.nan,
        }
    return {
        "valid": True,
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotb": Lipinski.NumRotatableBonds(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
    }


def add_drug_likeness(frame: pd.DataFrame) -> pd.DataFrame:
    desc = pd.DataFrame([descriptors(s) for s in frame["canonical_smiles"]])
    out = pd.concat([frame.reset_index(drop=True), desc], axis=1)
    out["lipinski_violations"] = (
        (out["mw"] > 500).astype(int)
        + (out["logp"] > 5).astype(int)
        + (out["hbd"] > 5).astype(int)
        + (out["hba"] > 10).astype(int)
    )
    out["veber_pass"] = (out["rotb"] <= 10) & (out["tpsa"] <= 140)
    out["lead_like_soft"] = (
        (out["mw"].between(250, 550))
        & (out["logp"].between(-1, 5.5))
        & (out["tpsa"].between(20, 160))
        & (out["lipinski_violations"] <= 1)
        & out["veber_pass"]
    )
    return out


def minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    values = series.astype(float)
    lo = values.min()
    hi = values.max()
    if hi == lo:
        scaled = pd.Series(0.5, index=series.index)
    else:
        scaled = (values - lo) / (hi - lo)
    return 1 - scaled if invert else scaled


def prioritize_target(frame: pd.DataFrame, target: str, top_n: int) -> pd.DataFrame:
    target_frame = frame[frame["target"] == target].copy()
    candidates = target_frame[
        (target_frame["lead_like_soft"])
        & (target_frame["ensemble_calibrated_mean"] > 0)
    ].copy()
    if candidates.empty:
        candidates = target_frame.copy()

    candidates["score_component_activity"] = minmax(candidates["ensemble_calibrated_mean"])
    raw_cols = [col for col in candidates.columns if col.endswith("_raw")]
    if raw_cols:
        candidates["ensemble_raw_mean"] = candidates[raw_cols].mean(axis=1)
        candidates["score_component_raw_rank"] = minmax(candidates["ensemble_raw_mean"])
    else:
        candidates["score_component_raw_rank"] = 0.5
    candidates["score_component_confidence"] = minmax(
        candidates["ensemble_calibrated_std"], invert=True
    )
    candidates["score_component_lipinski"] = 1 - (candidates["lipinski_violations"] / 4)
    candidates["score_component_veber"] = candidates["veber_pass"].astype(float)
    candidates["score_component_property_window"] = (
        minmax(candidates["mw"].clip(200, 600), invert=False) * 0.0 + 1.0
    )
    candidates["priority_score"] = (
        0.47 * candidates["score_component_activity"]
        + 0.23 * candidates["score_component_confidence"]
        + 0.10 * candidates["score_component_raw_rank"]
        + 0.12 * candidates["score_component_lipinski"]
        + 0.08 * candidates["score_component_veber"]
    )
    candidates = candidates.sort_values(
        ["priority_score", "ensemble_calibrated_mean", "ensemble_calibrated_std"],
        ascending=[False, False, True],
    )
    candidates["priority_rank"] = range(1, len(candidates) + 1)
    return candidates.head(top_n)


def scaffold_diverse_selection(ranked: pd.DataFrame, top_n: int, max_per_scaffold: int) -> pd.DataFrame:
    selected = []
    scaffold_counts: dict[str, int] = {}
    for _, row in ranked.iterrows():
        count = scaffold_counts.get(row["scaffold"], 0)
        if count >= max_per_scaffold:
            continue
        selected.append(row)
        scaffold_counts[row["scaffold"]] = count + 1
        if len(selected) >= top_n:
            break
    if not selected:
        return ranked.head(0).copy()
    out = pd.DataFrame(selected).reset_index(drop=True)
    out["diverse_rank"] = range(1, len(out) + 1)
    return out


def summarize_pool(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in frame.groupby("target"):
        rows.append(
            {
                "target": target,
                "test_rows": len(group),
                "test_actives": int(group["label"].sum()),
                "lead_like_soft": int(group["lead_like_soft"].sum()),
                "lipinski_pass_0viol": int((group["lipinski_violations"] == 0).sum()),
                "veber_pass": int(group["veber_pass"].sum()),
                "median_ensemble_score": group["ensemble_calibrated_mean"].median(),
                "p95_ensemble_score": group["ensemble_calibrated_mean"].quantile(0.95),
                "median_disagreement": group["ensemble_calibrated_std"].median(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="scripts/data/processed/calibration/calibrated_test_predictions.csv",
    )
    parser.add_argument("--out-dir", default="scripts/data/processed/prioritized_hits")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--diverse-top-n", type=int, default=30)
    parser.add_argument("--max-per-scaffold", type=int, default=2)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    annotated = add_drug_likeness(predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shortlists = []
    diverse_shortlists = []
    for target in sorted(annotated["target"].unique()):
        target_ranked = prioritize_target(annotated, target, max(args.top_n, args.diverse_top_n * 10))
        shortlist = target_ranked.head(args.top_n).copy()
        diverse = scaffold_diverse_selection(
            target_ranked, args.diverse_top_n, args.max_per_scaffold
        )
        shortlist.to_csv(out_dir / f"{target.lower()}_prioritized_top{args.top_n}.csv", index=False)
        diverse.to_csv(
            out_dir / f"{target.lower()}_prioritized_diverse_top{args.diverse_top_n}.csv",
            index=False,
        )
        shortlists.append(shortlist)
        diverse_shortlists.append(diverse)

    combined_shortlist = pd.concat(shortlists, ignore_index=True)
    combined_diverse = pd.concat(diverse_shortlists, ignore_index=True)
    annotated.to_csv(out_dir / "calibrated_predictions_with_druglikeness.csv", index=False)
    combined_shortlist.to_csv(out_dir / f"combined_prioritized_top{args.top_n}.csv", index=False)
    combined_diverse.to_csv(
        out_dir / f"combined_prioritized_diverse_top{args.diverse_top_n}.csv",
        index=False,
    )
    summarize_pool(annotated).to_csv(out_dir / "prioritization_pool_summary.csv", index=False)

    display_cols = [
        "target",
        "priority_rank",
        "compound_id",
        "label",
        "priority_score",
        "ensemble_calibrated_mean",
        "ensemble_calibrated_std",
        "mw",
        "logp",
        "tpsa",
        "lipinski_violations",
        "veber_pass",
    ]
    print(combined_shortlist[display_cols].head(20).to_string(index=False))
    print(f"\nWrote prioritized hit tables to: {out_dir}")


if __name__ == "__main__":
    main()
