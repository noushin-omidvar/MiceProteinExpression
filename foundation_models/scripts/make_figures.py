#!/usr/bin/env python3
"""Figures for the foundation-model study (paper panels and single-message slide figures).

Every chart is a ``draw_*`` function that fills one Axes, so the same code produces the
multi-panel paper figures and the large-type slide figures. Inputs are saved results only.
Colour roles: foundation models = vermilion (the subject of the study), conventional
baselines = cobalt / muted blue, chance and references = gray.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from mice_protein import plotting as P
from mice_protein.benchmark import models as M

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "foundation_models/results"
TAB = RES / "tables"
FIG = RES / "figures"
SLIDE = FIG / "slides"
S = json.loads((RES / "summary.json").read_text())
LABEL = {k: v.label for k, v in M.SPECS.items()}
SHORT = {"dummy": "Prior", "logistic": "Logistic", "elastic_net": "Elastic\nnet", "svm": "SVM",
         "random_forest": "Random\nforest", "gradient_boosting": "Boosting", "tabpfn": "TabPFN",
         "kumo": "Kumo"}
ORDER = list(M.SPECS)


def color(model: str) -> str:
    if model in M.FOUNDATION:
        return P.VERMILION
    return P.SUBTLE_TEXT if model == "dummy" else P.COBALT


def marker(model: str) -> str:
    return {"tabpfn": "D", "kumo": "s"}.get(model, "o")


# ---------------------------------------------------------------------------
# Drawing functions (one Axes each)
# ---------------------------------------------------------------------------

def draw_accuracy(ax, big=False):
    """Eight-group balanced accuracy per model: per-repeat dots and mean."""
    preds = pd.read_csv(RES / "predictions_class.csv.gz")
    from mice_protein.benchmark.harness import pooled_scores

    pooled = pooled_scores(preds.drop(columns="animal"))
    means = pooled.groupby("model").balanced_accuracy.mean().reindex(ORDER).sort_values()
    for i, model in enumerate(means.index):
        vals = pooled[pooled.model == model].balanced_accuracy
        ax.scatter(vals, np.full(len(vals), i), s=26 if big else 9, color=color(model),
                   marker=marker(model), alpha=0.45, lw=0)
        ax.scatter(means[model], i, s=110 if big else 34, color=color(model),
                   marker=marker(model), zorder=3, edgecolors="white", linewidths=0.8)
        P.annotate_stat(ax, max(vals.max(), means[model]) + 0.03, i, f"{means[model]:.2f}",
                        va="center", fontsize=12 if big else 6)
    ax.axvline(0.125, color=P.SUBTLE_TEXT, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.13, -0.6, "chance", color=P.SUBTLE_TEXT, fontsize=11 if big else 6)
    ax.set_yticks(range(len(means)), [LABEL[m] for m in means.index])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Eight-group balanced accuracy (held-out mice)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)


def draw_h1(ax, tfm, metric="H1", big=False):
    """Paired difference (foundation model − baseline) with corrected 95% CI."""
    tests = pd.DataFrame(S[metric][tfm]["tests"]).set_index("baseline").reindex(
        [b for b in ORDER if b in M.BASELINES])[::-1]
    y = np.arange(len(tests))
    sig = tests.significant_better
    cols = np.where(sig, P.VERMILION, P.MUTED)
    ax.hlines(y, tests.lower, tests.upper, color=cols, lw=2.4 if big else 1.4)
    ax.scatter(tests.mean_difference, y, color=cols, s=80 if big else 22, zorder=3,
               marker=marker(tfm))
    from matplotlib.transforms import blended_transform_factory

    margin = blended_transform_factory(ax.transAxes, ax.transData)
    for i, (_, r) in enumerate(tests.iterrows()):
        P.annotate_stat(ax, 1.03, i,
                        f"p = {r.p_holm:.3f}" if r.p_holm >= 0.001 else "p < 0.001",
                        va="center", fontsize=11 if big else 5.8, transform=margin,
                        color=P.INK if r.significant_better else P.SUBTLE_TEXT)
    P.zero_line(ax, "v")
    ax.set_yticks(y, [f"vs {LABEL[b]}" for b in tests.index])
    unit = "balanced accuracy" if metric == "H1" else "log loss (lower is better)"
    ax.set_xlabel(f"{LABEL[tfm]} advantage in {unit}\n(corrected 95% CI; Holm-adjusted p)")
    span = tests.upper.max() - min(tests.lower.min(), 0)
    ax.set_xlim(min(tests.lower.min(), 0) - 0.05 * span, tests.upper.max() + 0.05 * span)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)


def draw_logloss(ax, big=False):
    perf = pd.DataFrame(S["performance"]).T.reindex(ORDER)
    perf = perf.drop(index="dummy").sort_values("log_loss_mean", ascending=False)
    y = np.arange(len(perf))
    ax.barh(y, perf.log_loss_mean, color=[color(m) for m in perf.index], height=0.6)
    for i, v in enumerate(perf.log_loss_mean):
        P.annotate_stat(ax, v + 0.02, i, f"{v:.2f}", va="center", fontsize=11 if big else 6)
    ax.axvline(np.log(8), color=P.SUBTLE_TEXT, lw=0.8, ls=(0, (3, 2)))
    ax.text(np.log(8) - 0.02, len(perf) - 0.5, "uniform guess", ha="right",
            color=P.SUBTLE_TEXT, fontsize=11 if big else 6)
    ax.set_yticks(y, [LABEL[m] for m in perf.index])
    ax.set_xlabel("Log loss (lower = better probabilities)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)


def draw_reliability(ax, models, big=False):
    rel = pd.read_csv(TAB / "reliability.csv")
    ax.plot([0, 1], [0, 1], color=P.SUBTLE_TEXT, lw=0.8, ls=(0, (3, 2)))
    for model in models:
        r = rel[rel.model == model]
        ax.plot(r.confidence, r.accuracy, color=color(model) if model in M.FOUNDATION else
                (P.COBALT if model == "logistic" else P.MUTED), marker=marker(model),
                ms=7 if big else 3, lw=2 if big else 1, label=LABEL[model])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.grid(axis="x")
    ax.legend(loc="upper left", fontsize=11 if big else 5.8)


def draw_learning(ax, big=False):
    lc = pd.read_csv(RES / "learning_curve.csv")
    for model in [m for m in ORDER if m in lc.model.unique()]:
        g = lc[lc.model == model].groupby("k").balanced_accuracy
        mean, se = g.mean(), g.std() / np.sqrt(g.count())
        c = color(model) if model in M.FOUNDATION else (P.COBALT if model == "logistic"
                                                         else P.MUTED)
        ax.plot(mean.index, mean, color=c, marker=marker(model), ms=7 if big else 3,
                lw=2.2 if big else 1.1, label=LABEL[model],
                ls="-" if model in M.FOUNDATION or model == "logistic" else (0, (3, 2)))
        ax.fill_between(mean.index, mean - 1.96 * se, mean + 1.96 * se, color=c, alpha=0.12, lw=0)
    ax.axhline(0.125, color=P.SUBTLE_TEXT, lw=0.8, ls=(0, (1, 2)))
    ax.set_xticks(sorted(lc.k.unique()))
    ax.set_xlabel("Training mice per group")
    ax.set_ylabel("Balanced accuracy\n(held-out mice; mean ± 95% CI)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=11 if big else 5.6, ncol=2)


def draw_gap(ax, big=False):
    for tfm in M.FOUNDATION:
        gap = pd.Series(S["H3"][tfm]["gap_by_k"]).rename(int).sort_index()
        ax.plot(gap.index, gap, color=P.VERMILION, marker=marker(tfm), ms=8 if big else 3.5,
                lw=2 if big else 1.1, label=f"{LABEL[tfm]} − logistic",
                ls="-" if tfm == "tabpfn" else (0, (3, 2)))
    P.zero_line(ax)
    ax.set_xlabel("Training mice per group")
    ax.set_ylabel("Accuracy advantage over\nlogistic regression")
    ax.legend(loc="upper right", fontsize=11 if big else 5.8)


def draw_permutation(ax, big=False):
    perm = pd.read_csv(RES / "permutation_null.csv")
    models = [m for m in ORDER if m in perm.model.unique()]
    for i, m in enumerate(models):
        v = perm[perm.model == m].balanced_accuracy
        ax.scatter(np.full(len(v), i), v, color=color(m), marker=marker(m), s=24 if big else 7,
                   alpha=0.6, lw=0)
        ax.hlines(v.mean(), i - 0.3, i + 0.3, color=P.INK, lw=1.4)
    ax.axhspan(0.075, 0.175, color=P.RULE, alpha=0.6, lw=0)
    ax.axhline(0.125, color=P.SUBTLE_TEXT, lw=0.8, ls=(0, (3, 2)))
    ax.set_xticks(range(len(models)), [SHORT[m] for m in models], fontsize=10 if big else 5.6)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("Balanced accuracy\nwith shuffled labels")


def draw_leakage(ax, big=False):
    lk = pd.DataFrame(S["H4b"]["by_model"]).T
    models = [m for m in ORDER if m in lk.index]
    x = np.arange(len(models))
    ax.bar(x - 0.18, lk.loc[models, "random_rows"], 0.34, color=P.MUTED, label="Random rows")
    ax.bar(x + 0.18, lk.loc[models, "mouse_held_out"], 0.34,
           color=[color(m) for m in models], label="Whole mice held out")
    ax.set_xticks(x, [SHORT[m] for m in models], fontsize=10 if big else 5.8)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Balanced accuracy (all 1,080 rows)")
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color=P.MUTED, label="Random rows (leaky)"),
                       Line2D([], [], marker="s", ls="", color=P.INK, label="Whole mice held out")],
              loc="upper center", ncol=2, fontsize=11 if big else 5.8)


def draw_memorization(ax, big=False):
    mem = pd.DataFrame(S["H4c"]["by_model"]).T
    models = [m for m in ORDER if m in mem.index and m != "dummy"]
    y = np.arange(len(models))[::-1]
    for yi, m in zip(y, models, strict=True):
        o, b = mem.loc[m, "original"], mem.loc[m, "obfuscated"]
        ax.plot([o, b], [yi, yi], color=P.RULE, lw=2)
        ax.scatter(o, yi, facecolors="none", edgecolors=P.INK, s=150 if big else 46, lw=1.1,
                   zorder=4)
        ax.scatter(b, yi, color=color(m), marker=marker(m) if m in M.FOUNDATION else "D",
                   s=60 if big else 18, zorder=3)
    ax.set_yticks(y, [LABEL[m] for m in models])
    ax.set_xlim(0.3, 0.95)
    ax.set_xlabel("Balanced accuracy\n(open circle = original; filled = transformed representation)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)


def draw_cost(ax, big=False):
    perf = pd.DataFrame(S["performance"]).T.reindex(ORDER).drop(index="dummy")
    for m, r in perf.iterrows():
        ax.scatter(r.seconds_per_fold, r.balanced_accuracy_mean, color=color(m), marker=marker(m),
                   s=120 if big else 30, zorder=3, edgecolors="white", linewidths=0.6)
        right = r.seconds_per_fold > 0.5
        ax.annotate(LABEL[m], (r.seconds_per_fold, r.balanced_accuracy_mean),
                    xytext=(-7 if right else 7, 0), textcoords="offset points",
                    ha="right" if right else "left", va="center", fontsize=11 if big else 5.8)
    ax.set_xscale("log")
    ax.set_xlabel("Seconds to fit and predict one fold (CPU, log scale)")
    ax.set_ylabel("Eight-group balanced accuracy")
    ax.grid(axis="x")


def draw_recall(ax, big=False):
    rec = pd.read_csv(TAB / "recall_by_group.csv")
    wide = rec.pivot(index="model", columns="group", values="recall").reindex(
        [m for m in ORDER if m != "dummy"])
    im = ax.imshow(wide.to_numpy(), cmap=mpl.colors.LinearSegmentedColormap.from_list(
        "w_c", ["#FFFFFF", P.MUTED, P.COBALT]), vmin=0, vmax=1, aspect="auto")
    P.no_grid(ax)
    for i in range(wide.shape[0]):
        for j in range(wide.shape[1]):
            v = wide.iloc[i, j]
            P.annotate_stat(ax, j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=10 if big else 5.2, color="white" if v > 0.6 else P.INK)
    ax.set_xticks(range(wide.shape[1]), [P.CELL_LABEL[c] for c in wide.columns], rotation=40,
                  ha="right")
    ax.set_yticks(range(wide.shape[0]), [LABEL[m] for m in wide.index])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


# ---------------------------------------------------------------------------
# Composite paper figures
# ---------------------------------------------------------------------------

def fig_overview():
    fig = plt.figure(figsize=(P.DOUBLE, 82 * P.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    cols = [(1, "1  The claim"), (26, "2  The difficulty"), (51, "3  Our questions"),
            (76, "4  Our design")]
    for x, head in cols:
        ax.text(x, 47, head, fontfamily=P.SERIF, fontweight="semibold", fontsize=9, va="top")
        ax.plot([x, x + 22], [43.5, 43.5], color=P.RULE, lw=1)
    for x in (23.6, 48.6, 73.6):
        ax.annotate("", (x + 1.2, 45.8), (x - 0.4, 45.8),
                    arrowprops={"arrowstyle": "-|>", "color": P.SUBTLE_TEXT, "lw": 0.8,
                                "mutation_scale": 8})
    blocks = {
        1: ["Pretrained tabular models", "(TabPFN, Kumo) learn from millions of",
            "tables and predict new ones in", "context, without training.", "",
            "They are claimed to excel on", "small datasets, as in biology."],
        26: ["Few independent units:", "1,080 rows, but only 72 mice.", "",
             "Noisy comparisons:", "small differences vs resampling noise.", "",
             "Possible contamination:", "a public dataset may be in pretraining."],
        51: ["RQ1  Are they more accurate?", "RQ2  Are their probabilities", "        better?",
             "RQ3  Is the gain largest with", "        the least data?", "RQ4  Are the scores valid?",
             "RQ5  What do they cost?"],
        76: ["Protocol fixed before running", "8 models, identical folds", "Whole mice held out",
             "Corrected paired t-tests", "Learning curves (2–6 mice/group)",
             "Permutation, leakage and", "representation controls"],
    }
    for x, lines in blocks.items():
        y = 40
        for line in lines:
            weight = "semibold" if line.endswith(":") else "normal"
            fam = P.MONO if line.startswith(("RQ", "        ")) else P.SANS
            ax.text(x, y, line, fontsize=6.4, va="top", fontweight=weight, fontfamily=fam,
                    color=P.INK)
            y -= 4.6
    ax.text(99, 1.2, "SCHEMATIC", ha="right", fontfamily=P.MONO, fontsize=5.2,
            color=P.SUBTLE_TEXT)
    P.save(fig, FIG, "fig0_overview")


def fig_accuracy():
    fig = plt.figure(figsize=(P.DOUBLE, 135 * P.MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.55, wspace=0.95)
    ax = fig.add_subplot(gs[0, :])
    draw_accuracy(ax)
    P.title(ax, "Accuracy by model (dots: 5 repeats; large marker: mean)")
    P.panel_label(ax, "a", x=-0.14)
    for k, tfm in enumerate(M.FOUNDATION):
        ax = fig.add_subplot(gs[1, k])
        draw_h1(ax, tfm)
        P.title(ax, f"{LABEL[tfm]}: advantage over each baseline")
        P.panel_label(ax, "bc"[k], x=-0.55)
    P.save(fig, FIG, "fig1_accuracy")


def fig_calibration():
    fig = plt.figure(figsize=(P.DOUBLE, 135 * P.MM))
    gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.95)
    ax = fig.add_subplot(gs[0, 0])
    draw_logloss(ax)
    P.title(ax, "Probability quality")
    P.panel_label(ax, "a", x=-0.5)
    ax = fig.add_subplot(gs[0, 1])
    draw_reliability(ax, ["logistic", *M.FOUNDATION])
    P.title(ax, "Reliability")
    P.panel_label(ax, "b", x=-0.35)
    for k, tfm in enumerate(M.FOUNDATION):
        ax = fig.add_subplot(gs[1, k])
        draw_h1(ax, tfm, metric="H2")
        P.title(ax, f"{LABEL[tfm]}: log-loss advantage")
        P.panel_label(ax, "cd"[k], x=-0.55)
    P.save(fig, FIG, "fig2_calibration")


def fig_learning():
    fig = plt.figure(figsize=(P.DOUBLE, 66 * P.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0])
    draw_learning(ax)
    P.title(ax, "Learning curves")
    P.panel_label(ax, "a", x=-0.16)
    ax = fig.add_subplot(gs[1])
    draw_gap(ax)
    P.title(ax, "Foundation-model advantage")
    P.panel_label(ax, "b", x=-0.25)
    P.save(fig, FIG, "fig3_learning_curves")


def fig_validity():
    fig = plt.figure(figsize=(P.DOUBLE, 130 * P.MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1], hspace=0.55, wspace=0.6)
    ax = fig.add_subplot(gs[0, :])
    draw_permutation(ax)
    P.title(ax, "Shuffled labels give chance accuracy for every model")
    P.panel_label(ax, "a", x=-0.08)
    ax = fig.add_subplot(gs[1, 0])
    draw_leakage(ax)
    P.title(ax, "Leakage inflates every model")
    P.panel_label(ax, "b", x=-0.2)
    ax = fig.add_subplot(gs[1, 1])
    draw_memorization(ax)
    P.title(ax, "Hiding dataset identity barely changes accuracy")
    P.panel_label(ax, "c", x=-0.55)
    P.save(fig, FIG, "fig4_validity")


def fig_cost_and_errors():
    fig = plt.figure(figsize=(P.DOUBLE, 72 * P.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.25], wspace=0.95)
    ax = fig.add_subplot(gs[0])
    draw_cost(ax)
    P.title(ax, "Accuracy versus compute")
    P.panel_label(ax, "a", x=-0.25)
    ax = fig.add_subplot(gs[1])
    im = draw_recall(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Recall", fontsize=6)
    cb.outline.set_linewidth(0.4)
    P.title(ax, "Where errors fall (exploratory)")
    P.panel_label(ax, "b", x=-0.3)
    P.save(fig, FIG, "fig5_cost_errors")


def slide_figures():
    with mpl.rc_context({"font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "legend.fontsize": 11, "axes.linewidth": 0.8}):
        specs = {
            "accuracy": (lambda ax: draw_accuracy(ax, big=True), (8, 4.6)),
            "h1_tabpfn": (lambda ax: draw_h1(ax, "tabpfn", big=True), (8, 4.4)),
            "h1_kumo": (lambda ax: draw_h1(ax, "kumo", big=True), (8, 4.4)),
            "logloss": (lambda ax: draw_logloss(ax, big=True), (8, 4.6)),
            "learning": (lambda ax: draw_learning(ax, big=True), (8, 4.6)),
            "gap": (lambda ax: draw_gap(ax, big=True), (8, 4.4)),
            "permutation": (lambda ax: draw_permutation(ax, big=True), (8, 4.4)),
            "leakage": (lambda ax: draw_leakage(ax, big=True), (8, 4.4)),
            "memorization": (lambda ax: draw_memorization(ax, big=True), (8, 4.4)),
            "cost": (lambda ax: draw_cost(ax, big=True), (8, 4.6)),
        }
        for name, (draw, size) in specs.items():
            fig, ax = plt.subplots(figsize=size)
            draw(ax)
            P.save(fig, SLIDE, name)


def fig_tfm_primer():
    """How a conventional model and a tabular foundation model make predictions (schematic)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    fig = plt.figure(figsize=(P.DOUBLE, 96 * P.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def table(x, y, w, h, rows, color, label=None):
        for r in range(rows):
            ax.add_patch(Rectangle((x, y + r * h / rows), w, h / rows * 0.75, color=color,
                                   alpha=0.35 + 0.5 * (r % 2), lw=0))
        if label:
            ax.text(x + w / 2, y - 0.04, label, ha="center", va="top", fontsize=6.2,
                    color=P.SUBTLE_TEXT)

    def box(x, y, w, h, text, ec, fc="white", color=P.INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                    fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6, color=color,
                linespacing=1.2)

    def arrow(x0, y0, x1, y1, color=P.SUBTLE_TEXT):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                     color=color, lw=0.9))

    ax.text(0.015, 0.97, "Conventional model (e.g. logistic regression)", fontsize=8,
            fontweight="semibold", color=P.COBALT, va="top")
    table(0.03, 0.66, 0.12, 0.2, 8, P.COBALT, "training mice\n(proteins + group)")
    arrow(0.17, 0.76, 0.24, 0.76)
    box(0.25, 0.69, 0.2, 0.14, "fit weights to\nthese mice only", P.COBALT)
    arrow(0.47, 0.76, 0.54, 0.76)
    table(0.55, 0.7, 0.1, 0.12, 4, P.MUTED, "new mice")
    arrow(0.67, 0.76, 0.74, 0.76)
    box(0.75, 0.69, 0.2, 0.14, "predicted group", P.COBALT)
    ax.plot([0.015, 0.985], [0.53, 0.53], color=P.RULE, lw=0.8)
    ax.text(0.015, 0.5, "Tabular foundation model (e.g. TabPFN, Kumo)", fontsize=8,
            fontweight="semibold", color=P.VERMILION, va="top")
    for k in range(3):
        table(0.03 + k * 0.02, 0.1 + k * 0.03, 0.1, 0.12, 5, P.SUBTLE_TEXT)
    ax.text(0.11, 0.07, "millions of synthetic tables\n(pretraining, done once)", ha="center",
            va="top", fontsize=6.2, color=P.SUBTLE_TEXT)
    arrow(0.19, 0.18, 0.24, 0.18)
    box(0.25, 0.1, 0.2, 0.16, "pretrained network:\na learned sense of\nhow tables behave",
        P.VERMILION)
    table(0.3, 0.33, 0.1, 0.08, 4, P.VERMILION)
    ax.text(0.42, 0.37, "training mice,\nread as context", va="center", fontsize=6.2,
            color=P.VERMILION)
    arrow(0.35, 0.32, 0.35, 0.27, P.VERMILION)
    arrow(0.47, 0.18, 0.54, 0.18)
    table(0.55, 0.12, 0.1, 0.12, 4, P.MUTED, "new mice")
    arrow(0.67, 0.18, 0.74, 0.18)
    box(0.75, 0.1, 0.2, 0.16, "predicted group\n(one pass, no training)", P.VERMILION)
    ax.text(0.985, 0.01, "SCHEMATIC", ha="right", fontfamily=P.MONO, fontsize=5,
            color=P.SUBTLE_TEXT)
    P.save(fig, FIG, "primer_tfm")


def main() -> None:
    P.apply_style()
    for f in (fig_tfm_primer, fig_overview, fig_accuracy, fig_calibration, fig_learning, fig_validity,
              fig_cost_and_errors, slide_figures):
        f()
        print("drew", f.__name__)


if __name__ == "__main__":
    main()
