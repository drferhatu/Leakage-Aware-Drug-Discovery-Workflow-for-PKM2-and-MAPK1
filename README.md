# Leakage-Aware Drug Discovery Workflow for PKM2 and MAPK1

This repository contains the analysis scripts and computational outputs associated with the manuscript:

**A Leakage-Aware Drug Discovery Workflow for PKM2 and MAPK1 Integrating Scaffold Validation, Molecular Docking and Structural Triage**

Manuscript ID: `ijms-4310641`

## Scope

The repository supports a computational drug discovery workflow for PKM2 and MAPK1, including dataset curation, exploratory analysis, scaffold-aware benchmarking, ligand-based machine learning, calibrated ranking, ADMET-aware prioritization, molecular docking, and structure-aware triage.

The scripts are provided to make the analysis transparent and reproducible after publication. They are organized under `scripts/`, together with processed outputs and generated figures where applicable.

## Data

The original LIT-PCBA benchmark should be obtained from its public source. The raw compressed LIT-PCBA archive is not included in this repository.

Processed target tables, model outputs, ADMET outputs, docking summaries, AutoDock Vina logs, prepared docking files, generated figures, and reproducibility manifests are provided under `scripts/` where applicable.

## Software Environment

The analyses were carried out using a conda-based Python environment. The main package versions used for the accepted manuscript were:

- Python 3.12.8
- RDKit 2025.09.5
- scikit-learn 1.7.2
- LightGBM 4.6.0
- XGBoost 3.1.2
- AutoDock Vina 1.2.5
- Meeko 0.7.1
- pandas 2.3.3
- NumPy 2.3.4
- matplotlib 3.10.8
- seaborn 0.13.2

The Python dependencies are also listed in `requirements.txt`.

## Notes

The LIT-PCBA labels are used here for retrospective benchmark auditing. Candidate rankings should not be interpreted as experimentally validated activity claims.

Docking scores and interaction summaries are used as a structural triage layer, not as standalone evidence of biological activity.

## Citation

If you use this repository, its scripts, processed outputs, or figures, please cite the associated article:

Ucar, F.; Kati, N. **A Leakage-Aware Drug Discovery Workflow for PKM2 and MAPK1 Integrating Scaffold Validation, Molecular Docking and Structural Triage.** *International Journal of Molecular Sciences*, 2026.

The final DOI, volume, issue, and article number will be added to this repository after publication.
