#!/usr/bin/env python3
"""Train ligand-based ML baselines against the max-similarity baseline."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Scoring import Scoring
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    Chem.RemoveStereochemistry(mol)
    try:
        value = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except RuntimeError:
        return "SCAFFOLD_ERROR"
    return value if value else "NO_SCAFFOLD"


def fingerprint_matrix(smiles_values: list[str], radius: int, n_bits: int) -> sparse.csr_matrix:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows = []
    cols = []
    for row_idx, smiles in enumerate(smiles_values):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = generator.GetFingerprint(mol)
        on_bits = list(fp.GetOnBits())
        rows.extend([row_idx] * len(on_bits))
        cols.extend(on_bits)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(smiles_values), n_bits), dtype=np.float32
    )


def split_indices(frame: pd.DataFrame, split: str, seed: int, test_size: float):
    indices = np.arange(len(frame))
    labels = frame["label"].to_numpy()
    if split == "random":
        return train_test_split(indices, test_size=test_size, random_state=seed, stratify=labels)
    if split != "scaffold":
        raise ValueError(f"Unknown split: {split}")

    rng = np.random.default_rng(seed)
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, value in enumerate(frame["scaffold"]):
        scaffold_to_indices[value].append(idx)
    groups = list(scaffold_to_indices.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    target_test_size = int(round(len(frame) * test_size))
    total_actives = int(frame["label"].sum())
    min_test_actives = max(1, int(round(total_actives * test_size * 0.5)))
    train_idx: list[int] = []
    test_idx: list[int] = []
    test_actives = 0
    for group in groups:
        group_actives = int(frame.iloc[group]["label"].sum())
        if len(test_idx) + len(group) <= target_test_size or (
            test_actives < min_test_actives and group_actives > 0
        ):
            test_idx.extend(group)
            test_actives += group_actives
        else:
            train_idx.extend(group)
    return np.array(train_idx), np.array(test_idx)


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


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "ap": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "ef_0_5pct": enrichment_factor(y_true, scores, 0.005),
        "ef_1pct": enrichment_factor(y_true, scores, 0.01),
        "ef_2pct": enrichment_factor(y_true, scores, 0.02),
        "bedroc20": bedroc(y_true, scores, alpha=20.0),
    }


def make_models(seed: int):
    models = {
        "logreg_balanced": LogisticRegression(
            class_weight="balanced",
            solver="saga",
            penalty="l2",
            C=1.0,
            max_iter=1000,
            n_jobs=-1,
            random_state=seed,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
    }
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm_weighted"] = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.7,
            objective="binary",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        pass
    try:
        from xgboost import XGBClassifier

        models["xgboost_weighted"] = XGBClassifier(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    except Exception:
        pass
    return models


def evaluate_target(frame: pd.DataFrame, target: str, split: str, args) -> list[dict[str, object]]:
    target_frame = frame[frame["target"] == target].reset_index(drop=True).copy()
    if args.max_rows and len(target_frame) > args.max_rows:
        actives = target_frame[target_frame["label"] == 1]
        inactives = target_frame[target_frame["label"] == 0].sample(
            n=args.max_rows - len(actives), random_state=args.seed
        )
        target_frame = pd.concat([actives, inactives], ignore_index=True)

    target_frame["scaffold"] = target_frame["canonical_smiles"].map(scaffold)
    train_idx, test_idx = split_indices(target_frame, split, args.seed, args.test_size)
    train = target_frame.iloc[train_idx].reset_index(drop=True)
    test = target_frame.iloc[test_idx].reset_index(drop=True)

    x_train = fingerprint_matrix(train["canonical_smiles"].tolist(), args.radius, args.n_bits)
    x_test = fingerprint_matrix(test["canonical_smiles"].tolist(), args.radius, args.n_bits)
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    results = []
    for name, model in make_models(args.seed).items():
        print(f"Training {target} {split} {name}...")
        fit_kwargs = {}
        if name in {"lightgbm_weighted", "xgboost_weighted"}:
            negatives = int((y_train == 0).sum())
            positives = int((y_train == 1).sum())
            if name == "lightgbm_weighted":
                model.set_params(scale_pos_weight=negatives / max(1, positives))
            if name == "xgboost_weighted":
                model.set_params(scale_pos_weight=negatives / max(1, positives))
        else:
            fit_kwargs["sample_weight"] = sample_weight

        model.fit(x_train, y_train, **fit_kwargs)
        scores = model.predict_proba(x_test)[:, 1]
        row = {
            "target": target,
            "split": split,
            "model": name,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_actives": int(y_train.sum()),
            "test_actives": int(y_test.sum()),
        }
        row.update(evaluate_scores(y_test, scores))
        results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="scripts/data/processed/lit_pcba_pkm2_mapk1_clean.csv")
    parser.add_argument("--out", default="scripts/data/processed/ml_baseline_results.csv")
    parser.add_argument("--targets", nargs="+", default=["PKM2", "MAPK1"])
    parser.add_argument("--splits", nargs="+", default=["random", "scaffold"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=120000,
        help="Optional per-target cap for fast first-pass modeling. Actives are always retained.",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    rows = []
    for target in args.targets:
        for split in args.splits:
            rows.extend(evaluate_target(frame, target, split, args))

    result_frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(out, index=False)
    print(result_frame.to_string(index=False))
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()

