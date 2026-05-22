#!/usr/bin/env python3
"""Prepare ADMET-AI inputs from scaffold-diverse prioritized compounds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_admet_input(shortlist: pd.DataFrame) -> pd.DataFrame:
    out = shortlist.copy()
    out["smiles"] = out["canonical_smiles"]
    out["admet_id"] = (
        out["target"].astype(str)
        + "_rank"
        + out["diverse_rank"].astype(int).astype(str).str.zfill(2)
        + "_cid"
        + out["compound_id"].astype(str)
    )
    keep_cols = [
        "admet_id",
        "target",
        "compound_id",
        "diverse_rank",
        "priority_score",
        "ensemble_calibrated_mean",
        "ensemble_calibrated_std",
        "label",
        "smiles",
        "canonical_smiles",
        "scaffold",
        "mw",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "rotb",
        "lipinski_violations",
        "veber_pass",
    ]
    return out[keep_cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shortlist",
        default="scripts/data/processed/prioritized_hits/combined_prioritized_diverse_top30.csv",
    )
    parser.add_argument("--out-dir", default="scripts/data/processed/admet")
    args = parser.parse_args()

    shortlist = pd.read_csv(args.shortlist)
    admet_input = build_admet_input(shortlist)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "combined_diverse_top30_admet_input.csv"
    admet_input.to_csv(out_path, index=False)

    print(f"Wrote {len(admet_input)} compounds to {out_path}")
    print(admet_input.groupby("target").size().to_string())


if __name__ == "__main__":
    main()
