#!/usr/bin/env python3
"""Export consensus-ranked Vina PDBQT poses to PDB for visual inspection."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


OBABEL = Path("/opt/miniconda3/envs/ferhat_ml/bin/obabel")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--consensus", default="scripts/docking/analysis/final_consensus_top5_per_target.csv")
    parser.add_argument("--out-dir", default="scripts/docking/visual_inspection/top_poses_pdb")
    args = parser.parse_args()

    consensus = pd.read_csv(args.consensus)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in consensus.iterrows():
        source = Path(row["pose_path"])
        target_name = f"{row['target']}_r{int(row['consensus_rank']):02d}_{row['docking_id']}.pdb"
        output = out_dir / target_name
        subprocess.run(
            [str(OBABEL), "-ipdbqt", str(source), "-opdb", "-O", str(output), "-m"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        rows.append(
            {
                "target": row["target"],
                "consensus_rank": row["consensus_rank"],
                "docking_id": row["docking_id"],
                "pose_pdb": output,
            }
        )

    pd.DataFrame(rows).to_csv(out_dir.parent / "top_pose_pdb_index.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nWrote visual-inspection PDB files to: {out_dir}")


if __name__ == "__main__":
    main()
