#!/usr/bin/env python3
"""Compute direct heavy-atom RMSD between prepared reference and redocked PDBQT poses."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCORE_RE = re.compile(r"REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)")


def atom_name(line: str) -> str:
    return line[12:16].strip()


def element_from_pdbqt(line: str) -> str:
    parts = line.split()
    if parts:
        token = parts[-1]
        if token:
            return token[0].upper()
    name = atom_name(line)
    return name[0].upper() if name else ""


def is_heavy_atom(line: str) -> bool:
    return line.startswith(("ATOM", "HETATM")) and element_from_pdbqt(line) != "H"


def xyz(line: str) -> tuple[float, float, float]:
    return (float(line[30:38]), float(line[38:46]), float(line[46:54]))


def read_reference(path: Path) -> list[tuple[float, float, float]]:
    coords = []
    for line in path.read_text().splitlines():
        if is_heavy_atom(line):
            coords.append(xyz(line))
    if not coords:
        raise ValueError(f"No heavy atoms found in {path}")
    return coords


def read_models(path: Path) -> list[dict]:
    models: list[dict] = []
    current: dict | None = None
    for line in path.read_text().splitlines():
        if line.startswith("MODEL"):
            current = {"coords": [], "score": None}
        elif line.startswith("ENDMDL"):
            if current is not None:
                models.append(current)
                current = None
        elif current is not None:
            match = SCORE_RE.search(line)
            if match:
                current["score"] = float(match.group(1))
            if is_heavy_atom(line):
                current["coords"].append(xyz(line))
    if current is not None:
        models.append(current)
    if not models:
        # Vina sometimes writes a single pose without explicit MODEL/ENDMDL.
        coords = []
        score = None
        for line in path.read_text().splitlines():
            match = SCORE_RE.search(line)
            if match:
                score = float(match.group(1))
            if is_heavy_atom(line):
                coords.append(xyz(line))
        if coords:
            models.append({"coords": coords, "score": score})
    return models


def rmsd(a: list[tuple[float, float, float]], b: list[tuple[float, float, float]]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Atom count mismatch: {len(a)} vs {len(b)}")
    total = 0.0
    for pa, pb in zip(a, b):
        total += sum((pa[i] - pb[i]) ** 2 for i in range(3))
    return math.sqrt(total / len(a))


def main() -> None:
    targets = [
        {
            "target": "PKM2",
            "pdb_id": "4G1N",
            "reference": ROOT / "pkm2_ref_ligand.pdbqt",
            "redocked": ROOT / "pkm2_ref_redocked.pdbqt",
        },
        {
            "target": "MAPK1",
            "pdb_id": "4QTE",
            "reference": ROOT / "mapk1_ref_ligand.pdbqt",
            "redocked": ROOT / "mapk1_ref_redocked.pdbqt",
        },
    ]
    rows = []
    for item in targets:
        ref = read_reference(item["reference"])
        models = read_models(item["redocked"])
        scored = []
        for idx, model in enumerate(models, start=1):
            scored.append(
                {
                    "target": item["target"],
                    "pdb_id": item["pdb_id"],
                    "mode": idx,
                    "heavy_atoms": len(ref),
                    "vina_score_kcal_mol": model["score"],
                    "direct_heavy_atom_rmsd_angstrom": rmsd(ref, model["coords"]),
                }
            )
        rows.extend(scored)

    out = ROOT / "redocking_rmsd_by_mode.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for target in ["PKM2", "MAPK1"]:
        target_rows = [row for row in rows if row["target"] == target]
        best = target_rows[0]
        min_rmsd = min(target_rows, key=lambda row: row["direct_heavy_atom_rmsd_angstrom"])
        summary.append(
            {
                "target": target,
                "pdb_id": best["pdb_id"],
                "best_pose_score_kcal_mol": best["vina_score_kcal_mol"],
                "best_pose_rmsd_angstrom": best["direct_heavy_atom_rmsd_angstrom"],
                "minimum_rmsd_mode": min_rmsd["mode"],
                "minimum_rmsd_angstrom": min_rmsd["direct_heavy_atom_rmsd_angstrom"],
                "heavy_atoms": best["heavy_atoms"],
            }
        )

    summary_out = ROOT / "redocking_validation_summary.csv"
    with summary_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    for row in summary:
        print(row)
    print(f"Wrote {out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
