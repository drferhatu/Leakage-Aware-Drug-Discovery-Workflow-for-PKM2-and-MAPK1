#!/usr/bin/env python3
"""Select ADMET-aware docking candidates and write 3D ligand SDF files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


TARGET_STRUCTURES = {
    "PKM2": {
        "protein": "scripts/data/Lit-PCBA/PKM2/4g1n_protein.pdb",
        "pocket": "scripts/data/Lit-PCBA/PKM2/4g1n_pocket.pdb",
        "reference_ligand": "scripts/data/Lit-PCBA/PKM2/4g1n_ligand.mol2",
    },
    "MAPK1": {
        "protein": "scripts/data/Lit-PCBA/MAPK1/4qte_protein.pdb",
        "pocket": "scripts/data/Lit-PCBA/MAPK1/4qte_pocket.pdb",
        "reference_ligand": "scripts/data/Lit-PCBA/MAPK1/4qte_ligand.mol2",
    },
}


def make_3d_mol(smiles: str, name: str, seed: int) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    mol.SetProp("_Name", name)
    return mol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--admet-table",
        default="scripts/data/processed/admet/admet_prioritized_top10_per_target.csv",
    )
    parser.add_argument("--out-dir", default="scripts/data/processed/docking_candidates")
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    frame = pd.read_csv(args.admet_table)
    out_dir = Path(args.out_dir)
    sdf_dir = out_dir / "ligands_sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)

    selected = (
        frame.sort_values(["target", "admet_priority_rank"])
        .groupby("target", as_index=False)
        .head(args.top_n)
        .copy()
    )
    selected["docking_id"] = (
        selected["target"]
        + "_r"
        + selected["admet_priority_rank"].astype(int).astype(str).str.zfill(2)
        + "_cid"
        + selected["compound_id"].astype(str)
    )
    for col in ["protein", "pocket", "reference_ligand"]:
        selected[col] = selected["target"].map(lambda target: TARGET_STRUCTURES[target][col])

    selected.to_csv(out_dir / f"docking_candidates_top{args.top_n}_per_target.csv", index=False)

    failures = []
    for target, group in selected.groupby("target"):
        writer = Chem.SDWriter(str(sdf_dir / f"{target.lower()}_top{args.top_n}.sdf"))
        for index, row in group.reset_index(drop=True).iterrows():
            mol = make_3d_mol(row["smiles"], row["docking_id"], seed=2026 + index)
            if mol is None:
                failures.append(row["docking_id"])
                continue
            for key in [
                "docking_id",
                "target",
                "compound_id",
                "admet_priority_rank",
                "priority_score",
                "ensemble_calibrated_mean",
                "admet_support_score",
                "toxicity_burden",
                "label",
            ]:
                mol.SetProp(key, str(row[key]))
            writer.write(mol)
        writer.close()

    if failures:
        pd.DataFrame({"docking_id": failures}).to_csv(out_dir / "sdf_embedding_failures.csv", index=False)
        print(f"3D embedding failed for {len(failures)} molecules")
    print(selected[["target", "docking_id", "label", "ensemble_calibrated_mean", "admet_support_score"]].to_string(index=False))
    print(f"\nWrote docking candidates to: {out_dir}")


if __name__ == "__main__":
    main()
