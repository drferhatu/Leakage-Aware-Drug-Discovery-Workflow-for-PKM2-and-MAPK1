#!/usr/bin/env python3
"""Build calibrated scaffold-split ensembles and uncertainty summaries."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from train_ligand_ml_baselines import bedroc, enrichment_factor


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


def scaffold_split(frame: pd.DataFrame, seed: int, test_size: float):
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


def make_model(model_name: str, seed: int, scale_pos_weight: float):
    if model_name == "lightgbm_weighted":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.7,
            objective="binary",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            scale_pos_weight=scale_pos_weight,
        )
    if model_name == "xgboost_weighted":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )
    raise ValueError(f"Unknown model: {model_name}")


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        bin_confidence = float(np.mean(y_prob[mask]))
        bin_accuracy = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(bin_confidence - bin_accuracy)
    return ece


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    return {
        "ap": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "ef_1pct": enrichment_factor(y_true, scores, 0.01),
        "bedroc20": bedroc(y_true, scores, alpha=20.0),
        "brier": brier_score_loss(y_true, clipped),
        "log_loss": log_loss(y_true, clipped, labels=[0, 1]),
        "ece10": expected_calibration_error(y_true, clipped, n_bins=10),
    }


def prepare_target(frame: pd.DataFrame, target: str, max_rows: int, seed: int) -> pd.DataFrame:
    target_frame = frame[frame["target"] == target].reset_index(drop=True).copy()
    if max_rows and len(target_frame) > max_rows:
        active = target_frame[target_frame["label"] == 1]
        inactive = target_frame[target_frame["label"] == 0].sample(
            n=max_rows - len(active), random_state=seed
        )
        target_frame = pd.concat([active, inactive], ignore_index=True)
    target_frame["scaffold"] = target_frame["canonical_smiles"].map(scaffold)
    return target_frame


def run_target(frame: pd.DataFrame, target: str, args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_frame = prepare_target(frame, target, args.max_rows, args.seed)
    train_idx, test_idx = scaffold_split(target_frame, args.seed, args.test_size)
    train = target_frame.iloc[train_idx].reset_index(drop=True)
    test = target_frame.iloc[test_idx].reset_index(drop=True)

    train_inner_idx, calib_idx = train_test_split(
        np.arange(len(train)),
        test_size=args.calibration_size,
        random_state=args.seed,
        stratify=train["label"].to_numpy(),
    )
    train_inner = train.iloc[train_inner_idx].reset_index(drop=True)
    calib = train.iloc[calib_idx].reset_index(drop=True)

    x_train = fingerprint_matrix(train_inner["canonical_smiles"].tolist(), args.radius, args.n_bits)
    x_calib = fingerprint_matrix(calib["canonical_smiles"].tolist(), args.radius, args.n_bits)
    x_test = fingerprint_matrix(test["canonical_smiles"].tolist(), args.radius, args.n_bits)
    y_train = train_inner["label"].to_numpy()
    y_calib = calib["label"].to_numpy()
    y_test = test["label"].to_numpy()

    prediction_rows = test[
        ["target", "compound_id", "canonical_smiles", "label", "scaffold"]
    ].copy()
    metric_rows = []
    calibration_rows = []

    model_score_columns = []
    calibrated_score_columns = []
    for model_name in args.models:
        negatives = int((y_train == 0).sum())
        positives = int((y_train == 1).sum())
        model = make_model(model_name, args.seed, negatives / max(1, positives))
        print(f"Training {target} {model_name}...")
        model.fit(x_train, y_train)

        raw_calib = model.predict_proba(x_calib)[:, 1]
        raw_test = model.predict_proba(x_test)[:, 1]

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_calib, y_calib)
        calibrated_test = calibrator.transform(raw_test)

        raw_col = f"{model_name}_raw"
        cal_col = f"{model_name}_calibrated"
        prediction_rows[raw_col] = raw_test
        prediction_rows[cal_col] = calibrated_test
        model_score_columns.append(raw_col)
        calibrated_score_columns.append(cal_col)

        raw_metrics = evaluate_scores(y_test, raw_test)
        cal_metrics = evaluate_scores(y_test, calibrated_test)
        metric_rows.extend(
            [
                {
                    "target": target,
                    "model": model_name,
                    "score_type": "raw",
                    "train_rows": len(train_inner),
                    "calibration_rows": len(calib),
                    "test_rows": len(test),
                    "test_actives": int(y_test.sum()),
                    **raw_metrics,
                },
                {
                    "target": target,
                    "model": model_name,
                    "score_type": "calibrated",
                    "train_rows": len(train_inner),
                    "calibration_rows": len(calib),
                    "test_rows": len(test),
                    "test_actives": int(y_test.sum()),
                    **cal_metrics,
                },
            ]
        )

        prob_true, prob_pred = calibration_curve(y_test, calibrated_test, n_bins=10, strategy="quantile")
        for bin_idx, (pred, true) in enumerate(zip(prob_pred, prob_true), start=1):
            calibration_rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "bin": bin_idx,
                    "mean_predicted": pred,
                    "fraction_active": true,
                }
            )

    prediction_rows["ensemble_raw_mean"] = prediction_rows[model_score_columns].mean(axis=1)
    prediction_rows["ensemble_calibrated_mean"] = prediction_rows[calibrated_score_columns].mean(axis=1)
    prediction_rows["ensemble_calibrated_std"] = prediction_rows[calibrated_score_columns].std(axis=1)
    prediction_rows["ensemble_confidence"] = 1 - prediction_rows["ensemble_calibrated_std"]

    ensemble_metrics = evaluate_scores(y_test, prediction_rows["ensemble_calibrated_mean"].to_numpy())
    metric_rows.append(
        {
            "target": target,
            "model": "ensemble",
            "score_type": "calibrated_mean",
            "train_rows": len(train_inner),
            "calibration_rows": len(calib),
            "test_rows": len(test),
            "test_actives": int(y_test.sum()),
            **ensemble_metrics,
        }
    )

    return prediction_rows, pd.DataFrame(metric_rows), pd.DataFrame(calibration_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="scripts/data/processed/lit_pcba_pkm2_mapk1_clean.csv")
    parser.add_argument("--out-dir", default="scripts/data/processed/calibration")
    parser.add_argument("--targets", nargs="+", default=["PKM2", "MAPK1"])
    parser.add_argument("--models", nargs="+", default=["lightgbm_weighted", "xgboost_weighted"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--calibration-size", type=float, default=0.2)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--max-rows", type=int, default=80000)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    all_metrics = []
    all_calibration = []
    for target in args.targets:
        predictions, metrics, calibration = run_target(frame, target, args)
        predictions.to_csv(out_dir / f"{target.lower()}_calibrated_test_predictions.csv", index=False)
        all_predictions.append(predictions)
        all_metrics.append(metrics)
        all_calibration.append(calibration)

    pd.concat(all_predictions, ignore_index=True).to_csv(
        out_dir / "calibrated_test_predictions.csv", index=False
    )
    pd.concat(all_metrics, ignore_index=True).to_csv(
        out_dir / "calibration_metrics.csv", index=False
    )
    pd.concat(all_calibration, ignore_index=True).to_csv(
        out_dir / "calibration_curves.csv", index=False
    )
    print(pd.concat(all_metrics, ignore_index=True).to_string(index=False))
    print(f"\nWrote calibration outputs to: {out_dir}")


if __name__ == "__main__":
    main()

