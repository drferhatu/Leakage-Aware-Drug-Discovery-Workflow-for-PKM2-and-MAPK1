#!/usr/bin/env python3
"""Run AutoDock Vina for prepared candidate configs and summarize scores."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import pandas as pd


VINA = Path("/opt/miniconda3/envs/ferhat_ml/bin/vina")
SCORE_RE = re.compile(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+")


def run(cmd: list[str], log_path: Path) -> None:
    with log_path.open("w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def parse_best_score(log_path: Path) -> float | None:
    for line in log_path.read_text(errors="replace").splitlines():
        match = SCORE_RE.match(line)
        if match:
            return float(match.group(1))
    return None


def target_from_config(config_path: Path) -> str:
    return config_path.name.split("__", 1)[0].upper()


def docking_id_from_config(config_path: Path) -> str:
    return config_path.stem.split("__", 1)[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs-dir", default="scripts/docking/configs")
    parser.add_argument("--out-dir", default="scripts/docking/results")
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--only-target", choices=["PKM2", "MAPK1"], default=None)
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)
    out_dir = Path(args.out_dir)
    poses_dir = out_dir / "poses"
    logs_dir = out_dir / "logs"
    poses_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    configs = sorted(configs_dir.glob("*__*.txt"))
    for config in configs:
        target = target_from_config(config)
        if args.only_target and target != args.only_target:
            continue
        docking_id = docking_id_from_config(config)
        pose_path = poses_dir / f"{docking_id}_vina_out.pdbqt"
        log_path = logs_dir / f"{docking_id}.log"
        cmd = [
            str(VINA),
            "--config",
            str(config),
            "--out",
            str(pose_path),
            "--cpu",
            str(args.cpu),
        ]
        print(f"Docking {docking_id}")
        run(cmd, log_path)
        rows.append(
            {
                "target": target,
                "docking_id": docking_id,
                "best_vina_score_kcal_mol": parse_best_score(log_path),
                "pose_path": pose_path,
                "log_path": log_path,
            }
        )

    summary = pd.DataFrame(rows).sort_values(["target", "best_vina_score_kcal_mol"])
    summary.to_csv(out_dir / "vina_docking_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote docking results to: {out_dir}")


if __name__ == "__main__":
    main()
