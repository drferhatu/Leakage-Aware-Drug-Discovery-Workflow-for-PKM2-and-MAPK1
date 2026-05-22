#!/usr/bin/env python3
"""Prepare receptor and ligand PDBQT files plus Vina configs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


ENV_BIN = Path("/opt/miniconda3/envs/ferhat_ml/bin")
MEEKO_RECEPTOR = ENV_BIN / "mk_prepare_receptor.py"
MEEKO_LIGAND = ENV_BIN / "mk_prepare_ligand.py"


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def pocket_box(path: Path, padding: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    coords: list[tuple[float, float, float]] = []
    with path.open() as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    if not coords:
        raise ValueError(f"No atom coordinates found in {path}")
    xs, ys, zs = zip(*coords)
    center = (
        (min(xs) + max(xs)) / 2,
        (min(ys) + max(ys)) / 2,
        (min(zs) + max(zs)) / 2,
    )
    size = (
        max(xs) - min(xs) + 2 * padding,
        max(ys) - min(ys) + 2 * padding,
        max(zs) - min(zs) + 2 * padding,
    )
    return center, size


def write_vina_config(
    path: Path,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    exhaustiveness: int,
    num_modes: int,
) -> None:
    lines = [
        f"receptor = {receptor_pdbqt}",
        f"ligand = {ligand_pdbqt}",
        "",
        f"center_x = {center[0]:.3f}",
        f"center_y = {center[1]:.3f}",
        f"center_z = {center[2]:.3f}",
        "",
        f"size_x = {size[0]:.3f}",
        f"size_y = {size[1]:.3f}",
        f"size_z = {size[2]:.3f}",
        "",
        f"exhaustiveness = {exhaustiveness}",
        f"num_modes = {num_modes}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        default="scripts/data/processed/docking_candidates/docking_candidates_top8_per_target.csv",
    )
    parser.add_argument("--sdf-dir", default="scripts/data/processed/docking_candidates/ligands_sdf")
    parser.add_argument("--out-dir", default="scripts/docking")
    parser.add_argument("--padding", type=float, default=4.0)
    parser.add_argument("--exhaustiveness", type=int, default=16)
    parser.add_argument("--num-modes", type=int, default=10)
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates)
    sdf_dir = Path(args.sdf_dir)
    out_dir = Path(args.out_dir)
    receptors_dir = out_dir / "receptors"
    ligands_dir = out_dir / "ligands_pdbqt"
    configs_dir = out_dir / "configs"
    for path in [receptors_dir, ligands_dir, configs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    box_rows = []
    for target, group in candidates.groupby("target"):
        row = group.iloc[0]
        protein = Path(row["protein"])
        pocket = Path(row["pocket"])
        center, size = pocket_box(pocket, args.padding)
        receptor_base = receptors_dir / target.lower()
        receptor_pdbqt = receptors_dir / f"{target.lower()}.pdbqt"

        run(
            [
                str(MEEKO_RECEPTOR),
                "--read_pdb",
                str(protein),
                "-o",
                str(receptor_base),
                "-p",
                str(receptor_pdbqt),
                "--box_center",
                f"{center[0]:.3f}",
                f"{center[1]:.3f}",
                f"{center[2]:.3f}",
                "--box_size",
                f"{size[0]:.3f}",
                f"{size[1]:.3f}",
                f"{size[2]:.3f}",
                "-v",
                str(configs_dir / f"{target.lower()}_box.txt"),
                "-a",
                "--default_altloc",
                "A",
            ]
        )

        target_sdf = sdf_dir / f"{target.lower()}_top{len(group)}.sdf"
        target_ligand_dir = ligands_dir / target.lower()
        target_ligand_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                str(MEEKO_LIGAND),
                "-i",
                str(target_sdf),
                "--name_from_prop",
                "docking_id",
                "--multimol_outdir",
                str(target_ligand_dir),
            ]
        )

        for ligand_pdbqt in sorted(target_ligand_dir.glob("*.pdbqt")):
            config_path = configs_dir / f"{target.lower()}__{ligand_pdbqt.stem}.txt"
            write_vina_config(
                config_path,
                receptor_pdbqt,
                ligand_pdbqt,
                center,
                size,
                args.exhaustiveness,
                args.num_modes,
            )

        box_rows.append(
            {
                "target": target,
                "protein": protein,
                "pocket": pocket,
                "receptor_pdbqt": receptor_pdbqt,
                "center_x": center[0],
                "center_y": center[1],
                "center_z": center[2],
                "size_x": size[0],
                "size_y": size[1],
                "size_z": size[2],
                "padding": args.padding,
                "ligand_pdbqt_count": len(list(target_ligand_dir.glob("*.pdbqt"))),
            }
        )

    pd.DataFrame(box_rows).to_csv(out_dir / "vina_box_summary.csv", index=False)
    print(f"Wrote Vina docking inputs to: {out_dir}")


if __name__ == "__main__":
    main()
