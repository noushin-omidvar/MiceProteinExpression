#!/usr/bin/env python3
"""Explainer graphics for the biology study (schematics for readers new to the topic).

Facts shown: Down syndrome = three copies of chromosome 21; Ts65Dn mice carry an extra segment
of mouse chromosome 16 with ~90 orthologous genes; phosphorylation tags (``p``-prefixed
names); memantine is an NMDA-receptor antagonist. Numbers in the evidence summary come from
results/mechanisms/summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mice_protein import explainers as X
from mice_protein import plotting as P

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures/explainers"
S = json.loads((ROOT / "results/mechanisms/summary.json").read_text())


def story():
    X.story_map(OUT, "story_map", [
        ("Question", "what changes in\nthe trisomic brain?"),
        ("Why", "learning deficit\nand a drug"),
        ("Background", "chromosomes,\nproteins, drug"),
        ("Data", "72 mice ×\n77 proteins"),
        ("Approach", "four rules for\nsmall samples"),
        ("Findings", "four questions,\nfour answers"),
        ("How sure", "strength of\nevidence"),
        ("Next", "what a follow-up\nneeds"),
    ], "The story in eight steps")


def chromosomes():
    fig, ax = X.canvas(183, 70)
    panels = [(0.03, "Typical", "two copies of\nchromosome 21", 2, P.COBALT, None),
              (0.36, "Down syndrome", "three copies of\nchromosome 21", 3, P.VERMILION, None),
              (0.69, "Ts65Dn mouse", ("two copies of chromosome 16\nplus an extra piece carrying\n"
               "~90 of the same genes"), 2, P.COBALT, "extra")]
    for x0, title, desc, n, color, extra in panels:
        X.heading(ax, x0, 0.95, title, size=8)
        for k in range(n):
            X.chromosome(ax, x0 + 0.06 + k * 0.05, 0.32, 0.42, color, w=0.026)
        if extra:
            X.chromosome(ax, x0 + 0.06 + 2 * 0.05, 0.47, 0.18, P.VERMILION, w=0.026)
            X.label(ax, x0 + 0.2, 0.56, "extra piece", color=P.VERMILION, size=6.2, ha="left")
        X.label(ax, x0 + 0.12, 0.17, desc, size=6.2)
    X.schematic_tag(ax)
    P.save(fig, OUT, "chromosomes")


def gene_protein():
    fig, ax = X.canvas(183, 62)
    X.figure_title(ax, "Genes make proteins; a chemical tag switches many of them on")
    # gene
    xs = np.linspace(0.04, 0.2, 40)
    ax.plot(xs, 0.55 + 0.05 * np.sin(xs * 120), color=P.COBALT, lw=1.6)
    ax.plot(xs, 0.55 - 0.05 * np.sin(xs * 120), color=P.MUTED, lw=1.6)
    X.label(ax, 0.12, 0.33, "gene (DNA)\nthe recipe", size=6.4)
    X.arrow(ax, 0.24, 0.55, 0.34, 0.55)
    X.label(ax, 0.29, 0.63, "made into", size=5.8)
    ax.add_patch(plt_ellipse(0.45, 0.55, 0.13, 0.22, P.MUTED))
    X.label(ax, 0.45, 0.33, "protein\nthe machine (off)", size=6.4)
    X.arrow(ax, 0.55, 0.55, 0.65, 0.55)
    X.label(ax, 0.6, 0.63, "+ phosphate tag", size=5.8, color=P.VERMILION)
    ax.add_patch(plt_ellipse(0.77, 0.55, 0.13, 0.22, P.COBALT))
    X.icon_circle(ax, 0.82, 0.64, 0.025, "p", P.VERMILION)
    X.label(ax, 0.77, 0.33, "tagged protein\nswitched on (e.g. pERK)", size=6.4)
    X.label(ax, 0.5, 0.1, "The study measured 77 proteins and tagged forms; names starting with “p” "
            "are tagged forms.", size=6.2, color=P.INK)
    X.schematic_tag(ax)
    P.save(fig, OUT, "gene_protein")


def plt_ellipse(x, y, w, h, color):
    from matplotlib.patches import Ellipse

    return Ellipse((x, y), w, h, fc=color, ec="none")


def memantine():
    fig, ax = X.canvas(183, 64)
    X.figure_title(ax, "Memantine partly blocks a receptor that controls signalling for memory")
    for x0, title, blocked in ((0.08, "Without memantine", False), (0.55, "With memantine", True)):
        X.label(ax, x0 + 0.17, 0.82, title, color=P.INK, size=7, fontweight="semibold")
        ax.add_patch(plt_rect(x0, 0.42, 0.34, 0.05, P.RULE))  # membrane
        ax.add_patch(plt_rect(x0 + 0.13, 0.35, 0.03, 0.2, P.COBALT))
        ax.add_patch(plt_rect(x0 + 0.18, 0.35, 0.03, 0.2, P.COBALT))
        X.label(ax, x0 + 0.17, 0.27, "NMDA receptor (a channel)", size=6)
        ys = [0.72, 0.66, 0.6]
        for k, y in enumerate(ys):
            X.dot(ax, x0 + 0.11 + (k - 1) * 0.02, y, 0.008, P.VERMILION)
        if blocked:
            X.dot(ax, x0 + 0.17, 0.5, 0.016, P.INK)
            X.label(ax, x0 + 0.28, 0.5, "memantine\nplugs the channel", size=6, color=P.INK,
                    ha="left")
            X.arrow(ax, x0 + 0.17, 0.4, x0 + 0.17, 0.33, color=P.RULE)
        else:
            X.arrow(ax, x0 + 0.17, 0.58, x0 + 0.17, 0.3, color=P.VERMILION)
            X.label(ax, x0 + 0.28, 0.5, "ions flow in,\nsignal to the cell", size=6, ha="left")
    X.label(ax, 0.5, 0.1, "Earlier studies found that a memantine injection restores learning in "
            "Ts65Dn mice.", size=6.2, color=P.INK)
    X.schematic_tag(ax)
    P.save(fig, OUT, "memantine")


def plt_rect(x, y, w, h, color):
    from matplotlib.patches import Rectangle

    return Rectangle((x, y), w, h, fc=color, ec="none")


def signature():
    fig, ax = X.canvas(183, 66)
    X.figure_title(ax, "A signature turns 70 protein levels into one score per mouse")
    rng = np.random.default_rng(3)
    for m in range(3):
        y0 = 0.62 - m * 0.2
        vals = rng.normal(0, 1, 12)
        for k, v in enumerate(vals):
            ax.add_patch(plt_rect(0.04 + k * 0.016, y0, 0.012, 0.04 + 0.05 * (v + 2) / 4,
                                  P.MUTED))
        X.label(ax, 0.02, y0 + 0.04, f"mouse {m + 1}", size=5.8, ha="right")
    X.label(ax, 0.14, 0.2, "70 protein levels\nper mouse", size=6.2)
    X.arrow(ax, 0.27, 0.5, 0.36, 0.5)
    X.box(ax, 0.37, 0.38, 0.22, 0.24, "weighted sum\n(weights = the signature:\nhow much each "
          "protein\nmoves with the effect)", ec=P.COBALT, size=6)
    X.arrow(ax, 0.6, 0.5, 0.69, 0.5)
    ax.plot([0.71, 0.97], [0.5, 0.5], color=P.INK, lw=0.9)
    for x, c in ((0.75, P.MUTED), (0.8, P.MUTED), (0.88, P.COBALT), (0.93, P.COBALT)):
        X.dot(ax, x, 0.5, 0.012, c)
    X.label(ax, 0.84, 0.38, "one score per mouse", size=6.2)
    X.label(ax, 0.5, 0.06, "Why: 7–10 mice per group are too few to test 70 proteins one by one; "
            "one score per mouse is far more reliable.", size=6.2, color=P.INK)
    X.schematic_tag(ax)
    P.save(fig, OUT, "signature")


def approach():
    fig, ax = X.canvas(183, 62)
    X.figure_title(ax, "Four rules for trustworthy answers from 72 mice")
    rules = [("1", "Count mice,\nnot rows", "15 readings of a mouse\nbecome one value"),
             ("2", "Combine\nproteins", "one signature score\nper mouse"),
             ("3", "No mouse\nscores itself", "signature rebuilt\nwithout that mouse"),
             ("4", "Let chance\nspeak", ("shuffle labels thousands\nof times to see what\n"
              "chance produces"))]
    for i, (num, title, desc) in enumerate(rules):
        x = 0.13 + i * 0.25
        X.icon_circle(ax, x, 0.66, 0.06, num, P.COBALT)
        X.label(ax, x, 0.44, title, color=P.INK, size=7, fontweight="semibold")
        X.label(ax, x, 0.2, desc, size=6)
    X.schematic_tag(ax)
    P.save(fig, OUT, "approach")


def evidence():
    d, L, R, N = S["dosage"], S["learning"], S["rescue"], S["network"]
    X.evidence_summary(OUT, "evidence", [
        ("Q1 · Dosage", (f"Extra gene copies add only ~{100 * d['set_fraction_of_expected']:.0f}% "
         "of the expected protein"), 3, f"replicated in separate mice (r = {d['replication_pearson']:.2f})"),
        ("Q2 · Learning", (f"Weaker learning response ({L['relative']['Ts65Dn saline']:.2f}×); "
         "near normal with memantine"), 1,
         f"two-sided p = {L['attenuation_p']:.3f}; interval reaches 1; exploratory"),
        ("Q3 · Rescue", (f"Memantine shifts trisomic mice ~{100 * R['rescue_fraction']:.0f}% "
         "toward normal"), 1,
         f"p = {R['rescue_shift_p']:.3f}, but interval includes no shift; controls move too"),
        ("Q4 · Networks", "No detectable rewiring", 1,
         f"p = {N['p_global']:.2f}; only large changes detectable"),
    ], "How sure are we? Strength of evidence for each answer")


def main() -> None:
    P.apply_style()
    for f in (story, chromosomes, gene_protein, memantine, signature, approach, evidence):
        f()
        print("drew", f.__name__)


if __name__ == "__main__":
    main()
