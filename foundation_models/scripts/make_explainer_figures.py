#!/usr/bin/env python3
"""Explainer graphics for the foundation-model study (schematics for readers new to the topic).

Writes to foundation_models/results/figures/explainers/. The evidence summary reads its numbers
from foundation_models/results/; every other figure is a labelled schematic.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_presentation import numbers

from mice_protein import explainers as X
from mice_protein import plotting as P

OUT = ROOT / "foundation_models/results/figures/explainers"


def story():
    X.story_map(OUT, "story_map", [
        ("Question", "do pretrained\nmodels help?"),
        ("Why", "biology tables\nare small"),
        ("Background", "what a foundation\nmodel does"),
        ("Data", "72 mice ×\n77 proteins"),
        ("Approach", "four rules for a\nfair comparison"),
        ("Findings", "four questions,\nfour answers"),
        ("How sure", "planned vs\nadded later"),
        ("Next", "lessons and\nopen questions"),
    ], "The story in eight steps")


def small_data():
    fig, ax = X.canvas(183, 66)
    X.figure_title(ax, "Biology tables are small because every sample is an animal")
    panels = [(0.05, "Typical machine-learning table", 40, 10, P.MUTED,
               "thousands to millions of rows"),
              (0.4, "This study", 9, 8, P.COBALT, "72 mice (independent samples)"),
              (0.72, "What the raw table suggests", 9, 8, P.COBALT,
               "1,080 rows = the same 72 mice,\neach read 15 times")]
    for x0, title, ncol, nrow, color, desc in panels:
        X.label(ax, x0 + 0.12, 0.8, title, color=P.INK, size=6.8, ha="center",
                fontweight="semibold")
        for i in range(ncol):
            for j in range(nrow):
                if title.startswith("Typical"):
                    X.dot(ax, x0 + 0.005 + i * 0.006, 0.36 + j * 0.035, 0.0022, color)
                elif title.startswith("What"):
                    for k in range(5):
                        ax.plot([x0 + 0.032 + i * 0.02, x0 + 0.048 + i * 0.02],
                                [0.352 + j * 0.035 + k * 0.004] * 2, color=P.MUTED, lw=0.6)
                else:
                    X.dot(ax, x0 + 0.04 + i * 0.02, 0.36 + j * 0.035, 0.006, color)
        X.label(ax, x0 + 0.12, 0.25, desc, size=6.2)
    X.label(ax, 0.5, 0.08, "Pretrained “foundation” models are said to help most in exactly this "
            "situation. We tested that claim.", size=6.4, color=P.INK)
    X.schematic_tag(ax)
    P.save(fig, OUT, "small_data")


def fair_rules():
    fig, ax = X.canvas(183, 62)
    X.figure_title(ax, "Four rules for a fair comparison on 72 mice")
    rules = [("1", "Hold out\nwhole mice", "a mouse’s 15 readings\nnever sit on both sides"),
             ("2", "Same splits for\nevery model", "5 repeats × 5 folds,\nidentical test mice"),
             ("3", "Statistics for\noverlapping splits",
              "corrected paired test,\nadjusted for 6 comparisons"),
             ("4", "Plan first,\nlabel the rest", "hypotheses written down\nbefore any run")]
    for i, (num, title, desc) in enumerate(rules):
        x = 0.13 + i * 0.25
        X.icon_circle(ax, x, 0.66, 0.06, num, P.COBALT)
        X.label(ax, x, 0.44, title, color=P.INK, size=7, fontweight="semibold")
        X.label(ax, x, 0.2, desc, size=6)
    X.schematic_tag(ax)
    P.save(fig, OUT, "fair_rules")


def timeline():
    fig, ax = X.canvas(183, 64)
    X.figure_title(ax, "What was planned, and what was added after seeing results")
    ax.plot([0.04, 0.96], [0.55, 0.55], color=P.RULE, lw=2.2, zorder=0)
    steps = [(0.08, "Protocol\ncommitted", "4 questions, hypotheses,\npass/fail rules", P.COBALT),
             (0.3, "Planned\nexperiments", "8 models · learning\ncurves · 3 controls", P.COBALT),
             (0.52, "Results\nseen", "planned verdicts\nrecorded", P.INK),
             (0.74, "Added later\n(exploratory)", ("tuned baselines · matched\ncurve · "
              "sensitivity"), P.VERMILION),
             (0.93, "Report", "both kinds,\nclearly labelled", P.INK)]
    for x, title, desc, color in steps:
        X.dot(ax, x, 0.55, 0.018, color)
        X.label(ax, x, 0.74, title, color=color, size=6.8, fontweight="semibold")
        X.label(ax, x, 0.33, desc, size=6)
    ax.axvspan(0.52, 0.985, ymin=0.47, ymax=0.63, color=P.VERMILION, alpha=0.07, lw=0)
    X.label(ax, 0.5, 0.1, "Planned results test the hypotheses. Later analyses can only suggest; "
            "they are marked “exploratory” everywhere.", size=6.2, color=P.INK)
    X.schematic_tag(ax)
    P.save(fig, OUT, "timeline")


def evidence():
    n = numbers()
    kt = n["kumo_vs_tuned"]
    X.evidence_summary(OUT, "evidence", [
        ("RQ1 · Accuracy", (f"Kumo led every model ({n['kumo']:.2f} vs logistic "
         f"{n['logistic']:.2f})"), 2,
         f"planned test clear; vs tuned logistic p = {kt['p']:.2f} (exploratory)"),
        ("RQ2 · Probabilities", f"Kumo had the lowest log loss ({n['ll_kumo']:.2f})", 2,
         "log loss mixes accuracy and calibration"),
        ("RQ3 · Small data", "No consistent small-data advantage", 1,
         "hypothesis not supported; matched curve varies widely"),
        ("RQ4 · Controls", "Shuffled labels → chance; leakage inflates", 3,
         "cannot rule out exposure to this public dataset"),
    ], "How sure are we? Strength of evidence for each answer")


def main() -> None:
    P.apply_style()
    for f in (story, small_data, fair_rules, timeline, evidence):
        f()
        print("drew", f.__name__)


if __name__ == "__main__":
    main()
