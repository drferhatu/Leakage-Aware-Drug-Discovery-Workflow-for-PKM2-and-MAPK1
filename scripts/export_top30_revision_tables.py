#!/usr/bin/env python3
"""Export revision supplementary tables for the expanded top-30 docking set."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / "MDPI_Latex" / "supplementary"
SOURCE = ROOT / "scripts" / "docking_top30" / "analysis" / "docking_ml_admet_consensus.csv"


def esc(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def write_consensus_table(frame: pd.DataFrame, path: Path, title: str, note: str) -> None:
    lines = [
        title,
        note,
        "",
        "\\begingroup",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.0pt}",
        "\\begin{longtable}{@{}p{0.9cm}p{0.55cm}>{\\raggedright\\arraybackslash}p{5.7cm}p{0.5cm}p{0.78cm}p{0.78cm}p{0.78cm}p{0.75cm}p{0.85cm}@{}}",
        "\\toprule",
        "Target & Rank & SMILES & Label & Calib. & ADMET & Toxic. & Vina & Consensus \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Target & Rank & SMILES & Label & Calib. & ADMET & Toxic. & Vina & Consensus \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{esc(row['target'])} & {int(row['consensus_rank'])} & "
            f"\\path{{{row['canonical_smiles']}}} & {int(row['label'])} & "
            f"{row['ensemble_calibrated_mean']:.3f} & {row['admet_support_score']:.3f} & "
            f"{row['toxicity_burden']:.3f} & {row['best_vina_score_kcal_mol']:.3f} & "
            f"{row['final_consensus_score']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{longtable}", "\\endgroup"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    df = pd.read_csv(SOURCE).sort_values(["target", "consensus_rank"]).copy()
    cols = [
        "target",
        "consensus_rank",
        "docking_id",
        "canonical_smiles",
        "label",
        "ensemble_calibrated_mean",
        "admet_support_score",
        "toxicity_burden",
        "best_vina_score_kcal_mol",
        "final_consensus_score",
    ]
    out = df[cols].copy()
    top5 = out[out["consensus_rank"] <= 5].copy()

    top5.to_csv(SUPP / "supp_table_s2_top_consensus_candidates.csv", index=False)
    out.to_csv(SUPP / "supp_table_s5_full_top30_consensus.csv", index=False)

    write_consensus_table(
        top5,
        SUPP / "supp_table_s2_top_consensus_candidates.tex",
        "\\section*{Supplementary Table S2}",
        "Top five consensus-ranked docked candidates for each target after integrating calibrated activity, ADMET support, toxicity burden and Vina docking support. The Label column reports the retrospective LIT-PCBA benchmark label, not experimental validation in this study.",
    )
    write_consensus_table(
        out,
        SUPP / "supp_table_s5_full_top30_consensus.tex",
        "\\section*{Supplementary Table S5}",
        "Full scaffold-diverse top-30 consensus table for each target after ADMET-aware prioritization and Vina docking. The Label column reports the retrospective LIT-PCBA benchmark label and is included to audit false-positive risk in the computational ranking.",
    )

    print(
        top5[
            [
                "target",
                "consensus_rank",
                "docking_id",
                "label",
                "ensemble_calibrated_mean",
                "admet_support_score",
                "toxicity_burden",
                "best_vina_score_kcal_mol",
                "final_consensus_score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
