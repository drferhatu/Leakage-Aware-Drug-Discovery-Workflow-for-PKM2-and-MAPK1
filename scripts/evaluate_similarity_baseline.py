#!/usr/bin/env python3
"""Evaluate analog-similarity baselines for PKM2/MAPK1 LIT-PCBA screening."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Scoring import Scoring
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


def morgan_fingerprint(smiles: str, radius: int, n_bits: int):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return generator.GetFingerprint(mol)


def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    Chem.RemoveStereochemistry(mol)
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except RuntimeError:
        return "SCAFFOLD_ERROR"
    return scaffold if scaffold else "NO_SCAFFOLD"


def enrichment_factor(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    n = len(y_true)
    top_n = max(1, int(np.ceil(n * fraction)))
    total_actives = float(np.sum(y_true))
    if total_actives == 0:
        return np.nan
    order = np.argsort(scores)[::-1]
    top_actives = float(np.sum(y_true[order[:top_n]]))
    expected_random = total_actives * (top_n / n)
    return top_actives / expected_random if expected_random else np.nan


def bedroc(y_true: np.ndarray, scores: np.ndarray, alpha: float = 20.0) -> float:
    order = np.argsort(scores)[::-1]
    ranked = [[int(y_true[idx])] for idx in order]
    return float(Scoring.CalcBEDROC(ranked, 0, alpha))


def random_split(frame: pd.DataFrame, seed: int, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(frame))
    labels = frame["label"].to_numpy()
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return train_idx, test_idx


def scaffold_split(frame: pd.DataFrame, seed: int, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, scaffold in enumerate(frame["scaffold"]):
        scaffold_to_indices[scaffold].append(idx)

    groups = list(scaffold_to_indices.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    target_test_size = int(round(len(frame) * test_size))
    test_indices: list[int] = []
    train_indices: list[int] = []
    test_actives = 0
    total_actives = int(frame["label"].sum())
    min_test_actives = max(1, int(round(total_actives * test_size * 0.5)))

    for group in groups:
        group_actives = int(frame.iloc[group]["label"].sum())
        should_add_to_test = len(test_indices) + len(group) <= target_test_size
        needs_actives = test_actives < min_test_actives and group_actives > 0
        if should_add_to_test or needs_actives:
            test_indices.extend(group)
            test_actives += group_actives
        else:
            train_indices.extend(group)

    return np.array(train_indices, dtype=int), np.array(test_indices, dtype=int)


def max_similarity_scores(test_fps, train_active_fps, batch_size: int) -> np.ndarray:
    scores = np.zeros(len(test_fps), dtype=float)
    for start in range(0, len(test_fps), batch_size):
        stop = min(start + batch_size, len(test_fps))
        for offset, fp in enumerate(test_fps[start:stop]):
            scores[start + offset] = max(DataStructs.BulkTanimotoSimilarity(fp, train_active_fps))
    return scores


def evaluate_target(
    frame: pd.DataFrame,
    target: str,
    split_name: str,
    seed: int,
    test_size: float,
    radius: int,
    n_bits: int,
    batch_size: int,
) -> dict[str, float | int | str]:
    target_frame = frame[frame["target"] == target].reset_index(drop=True)
    target_frame["scaffold"] = target_frame["canonical_smiles"].map(murcko_scaffold)

    if split_name == "random":
        train_idx, test_idx = random_split(target_frame, seed, test_size)
    elif split_name == "scaffold":
        train_idx, test_idx = scaffold_split(target_frame, seed, test_size)
    else:
        raise ValueError(f"Unknown split: {split_name}")

    train = target_frame.iloc[train_idx].reset_index(drop=True)
    test = target_frame.iloc[test_idx].reset_index(drop=True)
    train_active = train[train["label"] == 1]
    if train_active.empty or test["label"].sum() == 0:
        raise ValueError(f"Split {split_name} for {target} has no train/test actives.")

    train_active_fps = [
        morgan_fingerprint(smiles, radius, n_bits)
        for smiles in train_active["canonical_smiles"]
    ]
    test_fps = [
        morgan_fingerprint(smiles, radius, n_bits)
        for smiles in test["canonical_smiles"]
    ]

    scores = max_similarity_scores(test_fps, train_active_fps, batch_size)
    y_test = test["label"].to_numpy()

    return {
        "target": target,
        "split": split_name,
        "seed": seed,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_actives": int(train["label"].sum()),
        "test_actives": int(test["label"].sum()),
        "train_scaffolds": train["scaffold"].nunique(),
        "test_scaffolds": test["scaffold"].nunique(),
        "ap": average_precision_score(y_test, scores),
        "roc_auc": roc_auc_score(y_test, scores),
        "ef_0_5pct": enrichment_factor(y_test, scores, 0.005),
        "ef_1pct": enrichment_factor(y_test, scores, 0.01),
        "ef_2pct": enrichment_factor(y_test, scores, 0.02),
        "bedroc20": bedroc(y_test, scores, alpha=20.0),
        "median_test_similarity": float(np.median(scores)),
        "p95_test_similarity": float(np.percentile(scores, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="scripts/data/processed/lit_pcba_pkm2_mapk1_clean.csv",
        help="Clean input table from prepare_lit_pcba_targets.py.",
    )
    parser.add_argument(
        "--out",
        default="scripts/data/processed/similarity_baseline_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--targets", nargs="+", default=["PKM2", "MAPK1"])
    parser.add_argument("--splits", nargs="+", default=["random", "scaffold"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    results = []
    for target in args.targets:
        for split_name in args.splits:
            print(f"Evaluating {target} with {split_name} split...")
            results.append(
                evaluate_target(
                    frame=frame,
                    target=target,
                    split_name=split_name,
                    seed=args.seed,
                    test_size=args.test_size,
                    radius=args.radius,
                    n_bits=args.n_bits,
                    batch_size=args.batch_size,
                )
            )

    result_frame = pd.DataFrame(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(out, index=False)
    print(result_frame.to_string(index=False))
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
