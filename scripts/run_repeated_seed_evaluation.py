#!/usr/bin/env python3
"""Run repeated-seed ligand ML evaluation and summarize robustness."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from train_ligand_ml_baselines import evaluate_target


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["ap", "roc_auc", "ef_0_5pct", "ef_1pct", "ef_2pct", "bedroc20"]
    grouped = (
        results.groupby(["target", "split", "model"], as_index=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else col
        for col in grouped.columns
    ]
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="scripts/data/processed/lit_pcba_pkm2_mapk1_clean.csv")
    parser.add_argument("--out-dir", default="scripts/data/processed/repeated_seed")
    parser.add_argument("--targets", nargs="+", default=["PKM2", "MAPK1"])
    parser.add_argument("--splits", nargs="+", default=["random", "scaffold"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 42, 67, 101])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=80000,
        help="Per-target cap for repeated runs. Actives are retained.",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed in args.seeds:
        args.seed = seed
        print(f"\n=== Seed {seed} ===")
        for target in args.targets:
            for split in args.splits:
                rows = evaluate_target(frame, target, split, args)
                for row in rows:
                    row["seed"] = seed
                all_rows.extend(rows)
                partial = pd.DataFrame(all_rows)
                partial.to_csv(out_dir / "repeated_seed_ml_results_partial.csv", index=False)

    results = pd.DataFrame(all_rows)
    summary = summarize(results)
    results.to_csv(out_dir / "repeated_seed_ml_results.csv", index=False)
    summary.to_csv(out_dir / "repeated_seed_ml_summary.csv", index=False)

    print("\nRepeated-seed summary")
    print(summary.to_string(index=False))
    print(f"\nWrote: {out_dir / 'repeated_seed_ml_results.csv'}")
    print(f"Wrote: {out_dir / 'repeated_seed_ml_summary.csv'}")


if __name__ == "__main__":
    main()

