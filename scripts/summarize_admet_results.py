#!/usr/bin/env python3
"""Summarize ADMET-AI predictions for the scaffold-diverse hit shortlist."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TOXICITY_RISK_COLUMNS = [
    "AMES",
    "Carcinogens_Lagunin",
    "ClinTox",
    "DILI",
    "hERG",
    "Skin_Reaction",
]

NUCLEAR_RECEPTOR_RISK_COLUMNS = [
    "NR-AR-LBD",
    "NR-AR",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER-LBD",
    "NR-ER",
    "NR-PPAR-gamma",
]

STRESS_RESPONSE_RISK_COLUMNS = [
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

CYP_RISK_COLUMNS = [
    "CYP1A2_Veith",
    "CYP2C19_Veith",
    "CYP2C9_Veith",
    "CYP2D6_Veith",
    "CYP3A4_Veith",
]


def robust_minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    values = series.astype(float)
    lo = values.quantile(0.05)
    hi = values.quantile(0.95)
    clipped = values.clip(lo, hi)
    if hi == lo:
        scaled = pd.Series(0.5, index=series.index)
    else:
        scaled = (clipped - lo) / (hi - lo)
    return 1 - scaled if invert else scaled


def add_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    risk_groups = {
        "toxicity_risk_flags": TOXICITY_RISK_COLUMNS,
        "nuclear_receptor_risk_flags": NUCLEAR_RECEPTOR_RISK_COLUMNS,
        "stress_response_risk_flags": STRESS_RESPONSE_RISK_COLUMNS,
        "cyp_inhibition_flags": CYP_RISK_COLUMNS,
    }
    for flag_name, cols in risk_groups.items():
        present = [col for col in cols if col in out.columns]
        out[flag_name] = (out[present] >= 0.5).sum(axis=1) if present else 0

    out["structural_alert_flags"] = (
        (out.get("PAINS_alert", 0) > 0).astype(int)
        + (out.get("BRENK_alert", 0) > 0).astype(int)
        + (out.get("NIH_alert", 0) > 0).astype(int)
    )
    out["absorption_support"] = (
        0.45 * out.get("HIA_Hou", 0)
        + 0.30 * out.get("PAMPA_NCATS", 0)
        + 0.25 * robust_minmax(out.get("Caco2_Wang", pd.Series(0.0, index=out.index)))
    )
    out["toxicity_burden"] = (
        0.22 * out.get("AMES", 0)
        + 0.24 * out.get("DILI", 0)
        + 0.22 * out.get("hERG", 0)
        + 0.12 * out.get("ClinTox", 0)
        + 0.10 * out.get("Carcinogens_Lagunin", 0)
        + 0.10 * out.get("Skin_Reaction", 0)
    )
    out["property_support"] = (
        0.40 * out.get("QED", 0)
        + 0.25 * robust_minmax(out.get("Solubility_AqSolDB", pd.Series(0.0, index=out.index)))
        + 0.20 * robust_minmax(out.get("LD50_Zhu", pd.Series(0.0, index=out.index)))
        + 0.15 * (1 - (out["structural_alert_flags"].clip(0, 3) / 3))
    )
    out["admet_support_score"] = (
        0.40 * out["absorption_support"]
        + 0.35 * out["property_support"]
        + 0.25 * (1 - out["toxicity_burden"].clip(0, 1))
    )
    out["integrated_admet_priority"] = (
        0.55 * robust_minmax(out["priority_score"])
        + 0.30 * robust_minmax(out["ensemble_calibrated_mean"])
        + 0.15 * out["admet_support_score"]
    )
    out["admet_priority_rank"] = (
        out.groupby("target")["integrated_admet_priority"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return out.sort_values(["target", "admet_priority_rank"])


def summarize_target(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in frame.groupby("target"):
        rows.append(
            {
                "target": target,
                "n": len(group),
                "known_actives": int(group["label"].sum()),
                "median_qed": group["QED"].median(),
                "median_solubility_aqsol": group["Solubility_AqSolDB"].median(),
                "median_admet_support": group["admet_support_score"].median(),
                "median_toxicity_burden": group["toxicity_burden"].median(),
                "compounds_with_no_structural_alerts": int(
                    (group["structural_alert_flags"] == 0).sum()
                ),
                "compounds_with_low_toxicity_flags": int(
                    (group["toxicity_risk_flags"] <= 1).sum()
                ),
                "compounds_with_high_hia": int((group["HIA_Hou"] >= 0.8).sum()),
                "compounds_with_high_pampa": int((group["PAMPA_NCATS"] >= 0.8).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="scripts/data/processed/admet/combined_diverse_top30_admet_predictions.csv",
    )
    parser.add_argument("--out-dir", default="scripts/data/processed/admet")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    annotated = add_flags(predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated.to_csv(out_dir / "combined_diverse_top30_admet_annotated.csv", index=False)
    summarize_target(annotated).to_csv(out_dir / "admet_target_summary.csv", index=False)

    top = (
        annotated.sort_values(["target", "admet_priority_rank"])
        .groupby("target", as_index=False)
        .head(args.top_n)
    )
    top.to_csv(out_dir / f"admet_prioritized_top{args.top_n}_per_target.csv", index=False)

    display_cols = [
        "target",
        "admet_priority_rank",
        "compound_id",
        "label",
        "priority_score",
        "ensemble_calibrated_mean",
        "admet_support_score",
        "toxicity_burden",
        "structural_alert_flags",
        "toxicity_risk_flags",
        "QED",
        "Solubility_AqSolDB",
        "hERG",
        "DILI",
        "AMES",
    ]
    print(top[display_cols].to_string(index=False))
    print(f"\nWrote ADMET summaries to: {out_dir}")


if __name__ == "__main__":
    main()
