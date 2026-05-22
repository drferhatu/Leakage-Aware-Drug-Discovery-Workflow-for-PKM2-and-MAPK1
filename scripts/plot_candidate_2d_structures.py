#!/usr/bin/env python3
"""Draw 2D structures for consensus-ranked candidates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import Draw


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "MDPI_Latex" / "supplementary" / "supp_table_s2_top_consensus_candidates.csv"
OUT_DIR = ROOT / "MDPI_Latex" / "figs" / "generated"


def draw_molecule(smiles: str, legend: str, size: tuple[int, int] = (360, 270)) -> Image.Image:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    Chem.rdDepictor.Compute2DCoords(mol)
    return Draw.MolToImage(mol, size=size, legend=legend)


def main() -> None:
    df = pd.read_csv(SOURCE).sort_values(["target", "consensus_rank"]).copy()
    rows = []
    for target in ["PKM2", "MAPK1"]:
        target_df = df[df["target"] == target].head(5)
        row_images = []
        for _, item in target_df.iterrows():
            legend = (
                f"{target}-r{int(item['consensus_rank'])} | "
                f"label={int(item['label'])} | "
                f"S={item['final_consensus_score']:.3f}"
            )
            row_images.append(draw_molecule(item["canonical_smiles"], legend))
        rows.append((target, row_images))

    cell_w, cell_h = 360, 270
    title_h = 54
    width = cell_w * 5
    height = (cell_h + title_h) * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    y = 0
    for target, images in rows:
        color = "#15616d" if target == "PKM2" else "#c75000"
        draw.rectangle([0, y, width, y + title_h], fill="#f8fafc")
        draw.text((18, y + 15), f"{target} consensus-ranked candidates", fill=color)
        y += title_h
        for idx, image in enumerate(images):
            canvas.paste(image.convert("RGB"), (idx * cell_w, y))
        y += cell_h

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "fig18_candidate_2d_structures.png"
    pdf_path = OUT_DIR / "fig18_candidate_2d_structures.pdf"
    canvas.save(png_path, dpi=(300, 300))
    canvas.save(pdf_path, "PDF", resolution=300.0)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
