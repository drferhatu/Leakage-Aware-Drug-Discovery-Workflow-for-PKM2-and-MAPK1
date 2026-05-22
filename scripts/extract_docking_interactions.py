#!/usr/bin/env python3
"""Extract lightweight protein-ligand interaction fingerprints from Vina PDBQT poses."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR"}
AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS"}
POLAR_RESIDUES = {"SER", "THR", "ASN", "GLN", "CYS", "TYR", "HIS", "LYS", "ARG", "ASP", "GLU"}
DONOR_TYPES = {"N", "NA", "HD"}
ACCEPTOR_TYPES = {"OA", "NA", "N", "O", "SA"}
CARBON_TYPES = {"C", "A"}
HALOGEN_ELEMENTS = {"F", "CL", "BR", "I"}


@dataclass(frozen=True)
class Atom:
    serial: int
    name: str
    resname: str
    chain: str
    resnum: int
    x: float
    y: float
    z: float
    atom_type: str

    @property
    def residue_id(self) -> str:
        chain = self.chain if self.chain.strip() else "_"
        return f"{self.resname}{chain}{self.resnum}"

    @property
    def element_like(self) -> str:
        atom_type = self.atom_type.upper()
        if atom_type in {"CL", "BR"}:
            return atom_type
        first = atom_type[0] if atom_type else self.name.strip()[:1]
        return first.upper()


def parse_pdbqt_atom(line: str) -> Atom:
    return Atom(
        serial=int(line[6:11]),
        name=line[12:16].strip(),
        resname=line[17:20].strip(),
        chain=line[21:22].strip(),
        resnum=int(line[22:26]),
        x=float(line[30:38]),
        y=float(line[38:46]),
        z=float(line[46:54]),
        atom_type=line[77:].strip().split()[0] if len(line) >= 78 and line[77:].strip() else line[12:16].strip()[:1],
    )


def read_receptor(path: Path) -> list[Atom]:
    atoms = []
    with path.open() as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                atoms.append(parse_pdbqt_atom(line))
    return atoms


def read_ligand_model_1(path: Path) -> list[Atom]:
    atoms = []
    in_first_model = False
    saw_model = False
    with path.open() as handle:
        for line in handle:
            if line.startswith("MODEL"):
                saw_model = True
                in_first_model = line.split()[1] == "1"
                continue
            if saw_model and line.startswith("ENDMDL"):
                if in_first_model:
                    break
                in_first_model = False
            if (not saw_model or in_first_model) and line.startswith(("ATOM", "HETATM")):
                atoms.append(parse_pdbqt_atom(line))
    return atoms


def distance(a: Atom, b: Atom) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def classify_pair(lig: Atom, rec: Atom, dist: float) -> set[str]:
    interactions: set[str] = set()
    lig_type = lig.atom_type.upper()
    rec_type = rec.atom_type.upper()
    lig_element = lig.element_like
    rec_element = rec.element_like

    if dist <= 4.0 and lig_type in CARBON_TYPES and rec_type in CARBON_TYPES and rec.resname in HYDROPHOBIC_RESIDUES:
        interactions.add("hydrophobic")
    if dist <= 4.0 and rec.resname in AROMATIC_RESIDUES and lig_type in {"A", "C"}:
        interactions.add("aromatic_contact")
    if dist <= 3.5 and (
        (lig_type in DONOR_TYPES and rec_type in ACCEPTOR_TYPES)
        or (lig_type in ACCEPTOR_TYPES and rec_type in DONOR_TYPES)
    ):
        interactions.add("polar_hbond_like")
    if dist <= 3.8 and (lig_element in HALOGEN_ELEMENTS and rec_element in {"O", "N", "S"}):
        interactions.add("halogen_polar_contact")
    if dist <= 4.0 and rec.resname in POLAR_RESIDUES and (
        lig_element in {"N", "O", "S", "F"} or rec_element in {"N", "O", "S"}
    ):
        interactions.add("polar_contact")
    if dist <= 4.5 and not interactions:
        interactions.add("close_contact")
    return interactions


def extract_for_pose(docking_id: str, target: str, receptor_atoms: list[Atom], ligand_atoms: list[Atom]) -> tuple[list[dict], dict]:
    residue_records: dict[str, dict] = {}
    all_distances = []
    for lig in ligand_atoms:
        for rec in receptor_atoms:
            dist = distance(lig, rec)
            if dist > 4.5:
                continue
            all_distances.append(dist)
            interaction_types = classify_pair(lig, rec, dist)
            rec_key = rec.residue_id
            record = residue_records.setdefault(
                rec_key,
                {
                    "target": target,
                    "docking_id": docking_id,
                    "residue_id": rec_key,
                    "resname": rec.resname,
                    "chain": rec.chain,
                    "resnum": rec.resnum,
                    "min_distance": dist,
                    "contact_atom_pairs": 0,
                    "hydrophobic": 0,
                    "aromatic_contact": 0,
                    "polar_hbond_like": 0,
                    "polar_contact": 0,
                    "halogen_polar_contact": 0,
                    "close_contact": 0,
                },
            )
            record["min_distance"] = min(record["min_distance"], dist)
            record["contact_atom_pairs"] += 1
            for interaction_type in interaction_types:
                record[interaction_type] = 1

    summary = {
        "target": target,
        "docking_id": docking_id,
        "ligand_atoms_model1": len(ligand_atoms),
        "contact_residues_4p5A": len(residue_records),
        "min_contact_distance": min(all_distances) if all_distances else float("nan"),
        "polar_hbond_like_residues": sum(rec["polar_hbond_like"] for rec in residue_records.values()),
        "hydrophobic_residues": sum(rec["hydrophobic"] for rec in residue_records.values()),
        "aromatic_contact_residues": sum(rec["aromatic_contact"] for rec in residue_records.values()),
        "halogen_polar_contact_residues": sum(rec["halogen_polar_contact"] for rec in residue_records.values()),
    }
    return list(residue_records.values()), summary


def set_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#2d3748",
            "grid.color": "#e2e8f0",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        },
    )


def plot_interaction_heatmap(contact_frame: pd.DataFrame, consensus: pd.DataFrame, fig_dir: Path) -> None:
    top = consensus[consensus["consensus_rank"] <= 5][["target", "docking_id", "consensus_rank"]]
    filtered = contact_frame.merge(top, on=["target", "docking_id"], how="inner")
    filtered["candidate"] = filtered["target"] + "-r" + filtered["consensus_rank"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9), constrained_layout=True)
    for ax, target in zip(axes, ["PKM2", "MAPK1"]):
        target_contacts = filtered[filtered["target"] == target].copy()
        residue_counts = (
            target_contacts.groupby("residue_id")["docking_id"]
            .nunique()
            .reset_index(name="candidate_count")
            .sort_values("candidate_count", ascending=False)
            .head(12)
        )
        target_contacts = target_contacts.merge(residue_counts[["residue_id"]], on="residue_id", how="inner")
        pivot = (
            target_contacts.assign(contact=1)
            .pivot_table(index="candidate", columns="residue_id", values="contact", aggfunc="max", fill_value=0)
        )
        ordered_rows = (
            top[top["target"] == target]
            .assign(candidate=lambda x: x["target"] + "-r" + x["consensus_rank"].astype(str))
            ["candidate"]
        )
        ordered_cols = residue_counts["residue_id"].tolist()
        pivot = pivot.reindex(index=[row for row in ordered_rows if row in pivot.index], columns=ordered_cols, fill_value=0)

        sns.heatmap(
            pivot,
            cmap=sns.color_palette(["#f8fafc", "#15616d"]),
            linewidths=0.5,
            linecolor="white",
            cbar=False,
            ax=ax,
        )
        ax.set_title(target, fontsize=12, fontweight="bold")
        ax.set_xlabel("Contact residues", fontsize=11)
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=40, labelsize=10.5)
        ax.tick_params(axis="y", labelsize=10, rotation=0)
    fig.suptitle("Residue-level contact fingerprints for consensus-ranked docking poses", y=1.06, fontsize=13)
    fig.savefig(fig_dir / "fig17_pose_contact_fingerprint.png", dpi=400, bbox_inches="tight")
    fig.savefig(fig_dir / "fig17_pose_contact_fingerprint.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docking-dir", default="scripts/docking")
    parser.add_argument("--consensus", default="scripts/docking/analysis/docking_ml_admet_consensus.csv")
    parser.add_argument("--out-dir", default="scripts/docking/interactions")
    parser.add_argument("--fig-dir", default="scripts/figures/docking")
    args = parser.parse_args()

    docking_dir = Path(args.docking_dir)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    consensus = pd.read_csv(args.consensus)
    receptor_paths = {
        "PKM2": docking_dir / "receptors" / "pkm2.pdbqt",
        "MAPK1": docking_dir / "receptors" / "mapk1.pdbqt",
    }
    receptors = {target: read_receptor(path) for target, path in receptor_paths.items()}

    all_records = []
    summaries = []
    for _, row in consensus.iterrows():
        target = row["target"]
        docking_id = row["docking_id"]
        pose_path = Path(row["pose_path"])
        ligand_atoms = read_ligand_model_1(pose_path)
        records, summary = extract_for_pose(docking_id, target, receptors[target], ligand_atoms)
        all_records.extend(records)
        summaries.append(summary)

    contacts = pd.DataFrame(all_records).sort_values(["target", "docking_id", "resnum"])
    pose_summary = pd.DataFrame(summaries).merge(
        consensus[
            [
                "target",
                "docking_id",
                "consensus_rank",
                "best_vina_score_kcal_mol",
                "ensemble_calibrated_mean",
                "admet_support_score",
                "final_consensus_score",
            ]
        ],
        on=["target", "docking_id"],
        how="left",
    )
    contacts.to_csv(out_dir / "pose_residue_contacts.csv", index=False)
    pose_summary.to_csv(out_dir / "pose_interaction_summary.csv", index=False)

    set_style()
    plot_interaction_heatmap(contacts, consensus, fig_dir)

    display_cols = [
        "target",
        "consensus_rank",
        "docking_id",
        "contact_residues_4p5A",
        "polar_hbond_like_residues",
        "hydrophobic_residues",
        "aromatic_contact_residues",
        "halogen_polar_contact_residues",
    ]
    print(pose_summary.sort_values(["target", "consensus_rank"])[display_cols].to_string(index=False))
    print(f"\nWrote interaction fingerprints to: {out_dir}")


if __name__ == "__main__":
    main()
