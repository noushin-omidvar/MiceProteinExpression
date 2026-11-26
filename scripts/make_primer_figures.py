#!/usr/bin/env python3
"""Explanatory figures for readers new to the study: the experiment, the measurement, the data.

Facts are taken from the original articles (Ahmed et al. 2015; Higuera et al. 2015):
context fear conditioning in a novel cage, 3 min exploration and a 2 s, 0.7 mA shock;
memantine 5 mg/kg i.p. or saline 15 min before training; brains collected 60 min after
training; cortex nuclear fraction; reverse-phase protein arrays printed as a five-point
dilution series in three replicates (15 readings per protein).

The timeline and measurement panels are schematics; the data-table panel shows real rows
from the UCI workbook.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from mice_protein import plotting as P
from mice_protein.data import load_data, protein_columns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures/primer"


def _box(ax, x, y, w, h, text, fc="white", ec=P.RULE, color=P.INK, size=7, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color,
            fontweight=weight, linespacing=1.15)


def _arrow(ax, x0, y0, x1, y1, color=P.SUBTLE_TEXT):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                 color=color, lw=0.9))


def timeline():
    """One training session, for a CS and an SC mouse (schematic)."""
    fig = plt.figure(figsize=(P.DOUBLE, 70 * P.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.015, 0.95, "One training session", fontfamily=P.SERIF, fontweight="semibold",
            fontsize=9.5, va="top")
    rows = [("CS  ·  context, then shock", 0.66, ["explore the new cage\n3 min", "shock\n2 s"],
             [0.26, 0.08], "learns to fear the cage", P.COBALT),
            ("SC  ·  shock, then context", 0.34, ["shock\n2 s", "explore the new cage\n3 min"],
             [0.08, 0.26], "cannot link shock to cage", P.SUBTLE_TEXT)]
    for label, y, steps, widths, outcome, c in rows:
        ax.text(0.015, y + 0.115, label, fontsize=7, fontweight="semibold", color=c)
        _box(ax, 0.015, y - 0.05, 0.14, 0.12, "memantine 5 mg/kg\nor saline", size=6.2)
        _arrow(ax, 0.16, y + 0.01, 0.2, y + 0.01)
        ax.text(0.18, y + 0.06, "15 min", ha="center", fontsize=5.8, color=P.SUBTLE_TEXT)
        x = 0.205
        for step, w in zip(steps, widths, strict=True):
            shock = step.startswith("shock")
            _box(ax, x, y - 0.05, w, 0.12, step, fc=P.VERMILION if shock else "#EEF2FB",
                 ec="none", color="white" if shock else P.INK, size=6.2,
                 weight="semibold" if shock else "normal")
            x += w + 0.008
        _arrow(ax, x + 0.005, y + 0.01, x + 0.06, y + 0.01)
        ax.text(x + 0.033, y + 0.06, "60 min", ha="center", fontsize=5.8, color=P.SUBTLE_TEXT)
        _box(ax, x + 0.065, y - 0.05, 0.17, 0.12, "brain collected\n(cortex)", size=6.2)
        ax.text(x + 0.25, y + 0.01, outcome, fontsize=6.4, va="center", color=c,
                fontstyle="italic")
    ax.text(0.015, 0.1, "Both groups receive the same shock in the same cage; only the order "
            "differs. Comparing CS with SC therefore isolates learning the association.",
            fontsize=6.4, color=P.SUBTLE_TEXT)
    ax.text(0.985, 0.02, "SCHEMATIC · protocol from Ahmed et al. 2015", ha="right",
            fontfamily=P.MONO, fontsize=5, color=P.SUBTLE_TEXT)
    P.save(fig, OUT, "experiment_timeline")


def measurement():
    """From one brain to 15 numbers per protein (schematic)."""
    fig = plt.figure(figsize=(P.DOUBLE, 64 * P.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.015, 0.95, "From one brain to 15 numbers per protein", fontfamily=P.SERIF,
            fontweight="semibold", fontsize=9.5, va="top")
    stages = [(0.02, "Cortex", "outer layer\nof the brain"),
              (0.215, "Nuclear fraction", "proteins from\ncell nuclei"),
              (0.41, "Protein array", "sample spotted on\na slide, stained per\nprotein (RPPA)"),
              (0.62, "15 spots", "3 replicates ×\n5 dilutions"),
              (0.815, "15 rows", "in the data table,\nfor each mouse")]
    for i, (x, title, sub) in enumerate(stages):
        ax.text(x + 0.08, 0.78, title, ha="center", fontsize=7.2, fontweight="semibold")
        ax.text(x + 0.08, 0.2, sub, ha="center", va="top", fontsize=6.2, color=P.SUBTLE_TEXT,
                linespacing=1.2)
        if i < len(stages) - 1:
            _arrow(ax, x + 0.165, 0.5, x + 0.2, 0.5, P.COBALT)
    # glyphs
    ax.add_patch(plt.matplotlib.patches.Ellipse((0.1, 0.5), 0.12, 0.3, fc="#EEF2FB", ec=P.INK,
                                                 lw=0.8))
    ax.add_patch(plt.matplotlib.patches.Ellipse((0.1, 0.56), 0.08, 0.1, fc=P.MUTED, ec="none"))
    for k in range(5):
        ax.add_patch(Circle((0.26 + (k % 3) * 0.035, 0.44 + (k // 3) * 0.1), 0.016, fc="white",
                            ec=P.INK, lw=0.7))
        ax.add_patch(Circle((0.26 + (k % 3) * 0.035, 0.44 + (k // 3) * 0.1), 0.007, fc=P.COBALT,
                            ec="none"))
    ax.add_patch(Rectangle((0.42, 0.32), 0.14, 0.36, fc="white", ec=P.INK, lw=0.8))
    for r in range(6):
        for c in range(5):
            ax.add_patch(Circle((0.44 + c * 0.025, 0.36 + r * 0.05), 0.008, fc=P.MUTED, ec="none",
                                alpha=0.5 + 0.08 * c))
    for r in range(3):
        for c in range(5):
            ax.add_patch(Circle((0.64 + c * 0.028, 0.4 + r * 0.08), 0.011, fc=P.COBALT, ec="none",
                                alpha=1 - 0.17 * c))
    ax.text(0.7, 0.68, "dilution →", ha="center", fontsize=5.6, color=P.SUBTLE_TEXT)
    ax.text(0.62, 0.48, "replicate", rotation=90, ha="center", va="center", fontsize=5.6,
            color=P.SUBTLE_TEXT)
    for r in range(15):
        ax.add_patch(Rectangle((0.84, 0.31 + r * 0.025), 0.13, 0.018, fc=P.MUTED,
                               alpha=0.35 + 0.04 * (r % 3), ec="none"))
    ax.text(0.985, 0.02, "SCHEMATIC · methods from Ahmed et al. 2015; Higuera et al. 2015",
            ha="right", fontfamily=P.MONO, fontsize=5, color=P.SUBTLE_TEXT)
    P.save(fig, OUT, "measurement")


def data_anatomy():
    """Real excerpt of the UCI table, annotated."""
    raw = load_data(ROOT / "Data/Raw_Data/Data_Cortex_Nuclear.xls")
    proteins = protein_columns(raw)
    first = raw[raw.AnimalID == raw.AnimalID.iloc[0]]
    show_rows = list(range(4)) + [None] + [14]
    show_cols = proteins[:3] + [None] + [proteins[-1]]
    fig = plt.figure(figsize=(P.DOUBLE, 74 * P.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.015, 0.95, "What the data table looks like (real rows)", fontfamily=P.SERIF,
            fontweight="semibold", fontsize=9.5, va="top")
    headers = ["MouseID"] + [(c.removesuffix("_N") if c else "…") for c in show_cols] + \
              ["Genotype", "Treatment", "Behavior", "class"]
    xs = np.linspace(0.06, 0.9, len(headers))
    y0, dy = 0.76, 0.075
    for x, h in zip(xs, headers, strict=True):
        ax.text(x, y0, h, ha="center", fontsize=6.4, fontweight="semibold",
                fontfamily=P.MONO, color=P.INK)
    ax.plot([0.03, 0.95], [y0 - 0.03, y0 - 0.03], color=P.RULE, lw=0.8)
    for k, r in enumerate(show_rows):
        y = y0 - (k + 1) * dy
        if r is None:
            for x in xs:
                ax.text(x, y + 0.01, "…", ha="center", va="center", fontsize=8, rotation=90,
                        color=P.SUBTLE_TEXT)
            continue
        row = first.iloc[r]
        cells = [row.MouseID] + [(f"{row[c]:.3f}" if c else "…") for c in show_cols] + \
                [row.Genotype, row.Treatment, row.Behavior, row["class"]]
        for x, c in zip(xs, cells, strict=True):
            ax.text(x, y, str(c), ha="center", fontsize=6.2, fontfamily=P.MONO, color=P.INK)
    ytop, ybot = y0 - dy + 0.03, y0 - len(show_rows) * dy - 0.02
    ax.plot([0.02, 0.02], [ybot, ytop], color=P.VERMILION, lw=1.2)
    ax.text(0.012, (ytop + ybot) / 2, "15 rows = one mouse", rotation=90, ha="right",
            va="center", fontsize=6.2, color=P.VERMILION)
    ax.plot([xs[1] - 0.03, xs[len(show_cols)] + 0.03], [y0 + 0.06] * 2, color=P.COBALT, lw=1.2)
    ax.text((xs[1] + xs[len(show_cols)]) / 2, y0 + 0.08, f"{len(proteins)} protein columns",
            ha="center", fontsize=6.4, color=P.COBALT)
    ax.plot([xs[-4] - 0.03, xs[-1] + 0.03], [y0 + 0.06] * 2, color=P.INK, lw=1.2)
    ax.text((xs[-4] + xs[-1]) / 2, y0 + 0.08, "experimental labels", ha="center", fontsize=6.4)
    ax.text(0.03, 0.06, f"{len(raw):,} rows in total = {raw.AnimalID.nunique()} mice × 15 "
            "readings. The MouseID suffix (_1 … _15) numbers the readings of one mouse.",
            fontsize=6.6, color=P.INK)
    ax.text(0.985, 0.01, "DATA · UCI Mice Protein Expression workbook", ha="right",
            fontfamily=P.MONO, fontsize=5, color=P.SUBTLE_TEXT)
    P.save(fig, OUT, "data_anatomy")


def main() -> None:
    P.apply_style()
    for f in (timeline, measurement, data_anatomy):
        f()
        print("drew", f.__name__)


if __name__ == "__main__":
    main()
