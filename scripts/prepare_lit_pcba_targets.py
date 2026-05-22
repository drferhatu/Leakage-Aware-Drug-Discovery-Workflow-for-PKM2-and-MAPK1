#!/usr/bin/env python3
"""Prepare PKM2 and MAPK1 LIT-PCBA tables for leakage-aware screening."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


@dataclass(frozen=True)
class TargetInput:
    target: str
    active_file: Path
    inactive_file: Path


def parse_smi(path: Path, target: str, label: int) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            smiles = parts[0]
            compound_id = parts[1] if len(parts) > 1 else f"{target}_{label}_{line_number}"
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                records.append(
                    {
                        "target": target,
                        "compound_id": compound_id,
                        "smiles": smiles,
                        "canonical_smiles": None,
                        "label": label,
                        "valid": False,
                    }
                )
                continue
            records.append(
                {
                    "target": target,
                    "compound_id": compound_id,
                    "smiles": smiles,
                    "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
                    "label": label,
                    "valid": True,
                }
            )
    return pd.DataFrame.from_records(records)


def add_basic_descriptors(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for smiles in frame["canonical_smiles"]:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            rows.append({})
            continue
        rows.append(
            {
                "mol_wt": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "tpsa": rdMolDescriptors.CalcTPSA(mol),
                "hbd": Lipinski.NumHDonors(mol),
                "hba": Lipinski.NumHAcceptors(mol),
                "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
                "rings": rdMolDescriptors.CalcNumRings(mol),
                "heavy_atoms": mol.GetNumHeavyAtoms(),
            }
        )
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["valid"]].copy()
    duplicate_counts = (
        valid.groupby(["target", "canonical_smiles"])["label"]
        .nunique()
        .reset_index(name="label_count")
    )
    conflicts = duplicate_counts[duplicate_counts["label_count"] > 1]
    summary_rows = []
    for target, target_frame in frame.groupby("target"):
        target_valid = target_frame[target_frame["valid"]]
        summary_rows.append(
            {
                "target": target,
                "rows": len(target_frame),
                "valid_rows": len(target_valid),
                "invalid_rows": int((~target_frame["valid"]).sum()),
                "actives": int((target_valid["label"] == 1).sum()),
                "inactives": int((target_valid["label"] == 0).sum()),
                "unique_canonical_smiles": target_valid["canonical_smiles"].nunique(),
                "duplicate_rows": len(target_valid)
                - target_valid["canonical_smiles"].nunique(),
                "label_conflicts": int((conflicts["target"] == target).sum()),
            }
        )
    return pd.DataFrame(summary_rows)


def find_label_conflicts(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["valid"]].copy()
    label_counts = (
        valid.groupby(["target", "canonical_smiles"])["label"]
        .nunique()
        .reset_index(name="label_count")
    )
    conflict_keys = label_counts[label_counts["label_count"] > 1][
        ["target", "canonical_smiles"]
    ]
    if conflict_keys.empty:
        return valid.iloc[0:0].copy()
    return valid.merge(conflict_keys, on=["target", "canonical_smiles"], how="inner")


def make_clean_table(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["valid"]].copy()
    conflicts = find_label_conflicts(valid)[["target", "canonical_smiles"]].drop_duplicates()
    if not conflicts.empty:
        valid = valid.merge(
            conflicts.assign(has_conflict=True),
            on=["target", "canonical_smiles"],
            how="left",
        )
        valid = valid[valid["has_conflict"].isna()].drop(columns=["has_conflict"])
    valid = valid.sort_values(["target", "label", "canonical_smiles", "compound_id"])
    return valid.drop_duplicates(["target", "canonical_smiles", "label"], keep="first")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lit-pcba-dir",
        default="scripts/data/Lit-PCBA",
        help="Path containing extracted LIT-PCBA target folders.",
    )
    parser.add_argument(
        "--out-dir",
        default="scripts/data/processed",
        help="Directory for processed CSV outputs.",
    )
    parser.add_argument(
        "--with-descriptors",
        action="store_true",
        help="Also compute basic RDKit descriptors. This is slower for the full dataset.",
    )
    args = parser.parse_args()

    root = Path(args.lit_pcba_dir)
    targets = [
        TargetInput("PKM2", root / "PKM2" / "actives.smi", root / "PKM2" / "inactives.smi"),
        TargetInput("MAPK1", root / "MAPK1" / "actives.smi", root / "MAPK1" / "inactives.smi"),
    ]

    frames = []
    for item in targets:
        frames.append(parse_smi(item.active_file, item.target, 1))
        frames.append(parse_smi(item.inactive_file, item.target, 0))

    combined = pd.concat(frames, ignore_index=True)
    if args.with_descriptors:
        combined = add_basic_descriptors(combined)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_name = (
        "lit_pcba_pkm2_mapk1_raw_descriptors.csv"
        if args.with_descriptors
        else "lit_pcba_pkm2_mapk1_raw.csv"
    )
    combined.to_csv(out_dir / table_name, index=False)
    clean = make_clean_table(combined)
    clean.to_csv(out_dir / "lit_pcba_pkm2_mapk1_clean.csv", index=False)
    conflicts = find_label_conflicts(combined)
    conflicts.to_csv(out_dir / "lit_pcba_pkm2_mapk1_label_conflicts.csv", index=False)
    summarize(combined).to_csv(out_dir / "lit_pcba_pkm2_mapk1_summary.csv", index=False)
    summarize(clean).to_csv(
        out_dir / "lit_pcba_pkm2_mapk1_clean_summary.csv", index=False
    )

    print("Raw summary")
    print(summarize(combined).to_string(index=False))
    print("\nClean summary")
    print(summarize(clean).to_string(index=False))
    print(f"\nWrote: {out_dir / table_name}")
    print(f"Wrote: {out_dir / 'lit_pcba_pkm2_mapk1_clean.csv'}")
    print(f"Wrote: {out_dir / 'lit_pcba_pkm2_mapk1_label_conflicts.csv'}")
    print(f"Wrote: {out_dir / 'lit_pcba_pkm2_mapk1_summary.csv'}")
    print(f"Wrote: {out_dir / 'lit_pcba_pkm2_mapk1_clean_summary.csv'}")


if __name__ == "__main__":
    main()
