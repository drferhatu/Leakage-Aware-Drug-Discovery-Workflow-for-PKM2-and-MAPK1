#!/usr/bin/env python3
"""Export revision supplementary tables for reproducibility."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / "MDPI_Latex" / "supplementary"


def esc(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def top_candidate_table() -> None:
    source = ROOT / "scripts" / "docking" / "analysis" / "final_consensus_top5_per_target.csv"
    df = pd.read_csv(source)
    cols = [
        "target",
        "consensus_rank",
        "docking_id",
        "smiles",
        "label",
        "ensemble_calibrated_mean",
        "admet_support_score",
        "best_vina_score_kcal_mol",
        "final_consensus_score",
    ]
    out = df[cols].sort_values(["target", "consensus_rank"]).copy()
    out.to_csv(SUPP / "supp_table_s2_top_consensus_candidates.csv", index=False)

    lines = [
        r"\section*{Supplementary Table S2}",
        "Top five consensus-ranked candidates for each target. The table reports canonical SMILES and the main ligand-based, ADMET and docking quantities used in final prioritization.",
        "",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\begin{longtable}{llp{6.2cm}rrrr}",
        r"\toprule",
        r"Target & Rank & SMILES & Label & Cal. score & ADMET & Vina & Consensus \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Target & Rank & SMILES & Label & Cal. score & ADMET & Vina & Consensus \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in out.iterrows():
        lines.append(
            f"{esc(row['target'])} & {int(row['consensus_rank'])} & "
            f"\\texttt{{{esc(row['smiles'])}}} & {int(row['label'])} & "
            f"{row['ensemble_calibrated_mean']:.3f} & {row['admet_support_score']:.3f} & "
            f"{row['best_vina_score_kcal_mol']:.3f} & {row['final_consensus_score']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup", ""])
    (SUPP / "supp_table_s2_top_consensus_candidates.tex").write_text("\n".join(lines))


def manifest_table() -> None:
    rows = [
        ("Processed target tables", "scripts/data/processed/lit_pcba_pkm2_mapk1_clean.csv; scripts/data/processed/lit_pcba_pkm2_mapk1_clean_summary.csv", "Cleaned canonical-SMILES tables and target-level counts."),
        ("Train/test and seed-dependent outputs", "scripts/data/processed/ml_baseline_results.csv; scripts/data/processed/repeated_seed/repeated_seed_ml_results.csv", "Model outputs for random/scaffold splits and repeated seeds."),
        ("Similarity baseline", "scripts/data/processed/similarity_baseline_results.csv", "Nearest-active Tanimoto baseline results for random and scaffold splits."),
        ("Representation sensitivity", "scripts/data/processed/ml_baseline_ecfp6_sensitivity_results.csv", "Additional ECFP6 scaffold-split sensitivity check."),
        ("Calibration outputs", "scripts/data/processed/calibration/calibration_metrics.csv; scripts/data/processed/calibration/calibrated_test_predictions.csv", "Calibration metrics and calibrated test predictions."),
        ("ADMET outputs", "scripts/data/processed/admet/combined_diverse_top30_admet_predictions.csv; scripts/data/processed/admet/combined_diverse_top30_admet_annotated.csv", "ADMET-AI predictions and annotated top-30 candidate tables."),
        ("Top candidate SMILES", "MDPI_Latex/supplementary/supp_table_s2_top_consensus_candidates.csv", "Consensus-ranked candidate SMILES and final prioritization scores."),
        ("Docking inputs and logs", "scripts/docking/configs/; scripts/docking/receptors/; scripts/docking/ligands_pdbqt/; scripts/docking/results/logs/", "Vina configuration files, prepared receptors/ligands and docking logs."),
        ("Docking poses and contact fingerprints", "scripts/docking/results/poses/; scripts/docking/interactions/pose_residue_contacts.csv", "Docked poses and residue-level contact tables."),
        ("Redocking validation", "scripts/docking/redocking_validation/redocking_validation_summary.csv; scripts/docking/redocking_validation/redocking_rmsd_by_mode.csv", "Reference-ligand redocking RMSD summaries."),
        ("Analysis scripts", "scripts/*.py", "Scripts used for curation, model training, calibration, prioritization, docking analysis and figure generation."),
    ]
    df = pd.DataFrame(rows, columns=["Item", "File(s)", "Purpose"])
    df.to_csv(SUPP / "supp_table_s3_reproducibility_manifest.csv", index=False)

    lines = [
        r"\section*{Supplementary Table S3}",
        "Reproducibility manifest for the revised manuscript. File paths refer to the project archive that accompanies the revision and will be mirrored in the public repository and Zenodo archive.",
        "",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\begin{longtable}{p{3.2cm}p{6.2cm}p{5.0cm}}",
        r"\toprule",
        r"Item & File(s) & Purpose \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Item & File(s) & Purpose \\",
        r"\midrule",
        r"\endhead",
    ]
    for item, files, purpose in rows:
        lines.append(f"{esc(item)} & \\texttt{{{esc(files)}}} & {esc(purpose)} \\\\")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup", ""])
    (SUPP / "supp_table_s3_reproducibility_manifest.tex").write_text("\n".join(lines))


def main() -> None:
    SUPP.mkdir(parents=True, exist_ok=True)
    top_candidate_table()
    manifest_table()
    print(f"Wrote revision supplementary tables to {SUPP}")


if __name__ == "__main__":
    main()
