#!/usr/bin/env python3
"""Figures for the prediction study (results/figures), in the house style.

Inputs are the saved outputs of ``run_analysis.py``, ``study.py`` and
``factorial_analysis.py``; nothing is re-estimated here except a PCA used only for
display. Every figure is written as SVG, PDF and 300-dpi PNG.
"""

from __future__ import annotations

import json
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from mice_protein import plotting as P
from mice_protein.annotations import display
from mice_protein.data import aggregate_animals, load_data, protein_columns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures"
COBALT_SEQ = LinearSegmentedColormap.from_list("white_cobalt", ["#FFFFFF", P.MUTED, P.COBALT])
DESIGN_LABEL = {"random_observation": "Random rows\n(mice leak)",
                "grouped_animal": "Held-out mice\n(15 rows each)",
                "mouse_mean": "Held-out mice\n(mouse means)"}
MODEL_STYLE = {"dummy": (P.SUBTLE_TEXT, "s", "Class prior"),
               "logistic": (P.COBALT, "o", "Logistic"),
               "random_forest": (P.MUTED, "D", "Random forest")}


def missingness(df, features):
    """Missing data concentrate in six proteins."""
    miss = df[features].isna().mean().sort_values(ascending=False).head(12)[::-1] * 100
    fig, ax = plt.subplots(figsize=(P.SINGLE, 62 * P.MM))
    colors = [P.VERMILION if v > 5 else P.MUTED for v in miss]
    ax.hlines(range(len(miss)), 0, miss, color=colors, lw=2.2)
    ax.scatter(miss, range(len(miss)), color=colors, s=14, zorder=3)
    for i, v in enumerate(miss):
        P.annotate_stat(ax, v + 0.6, i, f"{v:.1f}%", va="center", fontsize=5.6)
    ax.set_yticks(range(len(miss)), [display(p) for p in miss.index])
    ax.set_xlabel("Measurements missing (% of 1,080 rows)")
    ax.set_xlim(0, miss.max() * 1.2)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    P.title(ax, "Missing data concentrate in six proteins")
    P.save(fig, OUT, "missingness")


def pca(mice, features):
    """Mouse-mean profiles separate mainly by conditioning order."""
    z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(mice[features]))
    model = PCA(2).fit(z)
    pcs = model.transform(z)
    fig, ax = plt.subplots(figsize=(P.SINGLE, 75 * P.MM))
    for (g, b, t), idx in mice.groupby(["Genotype", "Behavior", "Treatment"]).groups.items():
        rows = mice.index.get_indexer(idx)
        color = P.GENOTYPE_COLOR[g]
        ax.scatter(pcs[rows, 0], pcs[rows, 1], s=18, marker="o" if b == "C/S" else "^",
                   edgecolors=color, facecolors=color if t == "Memantine" else "white",
                   linewidths=0.8, zorder=3)
    ev = model.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.0f}% of variance)")
    ax.grid(axis="x")
    handles = [Line2D([], [], marker="s", ls="", color=P.CONTROL, label="Control", ms=4),
               Line2D([], [], marker="s", ls="", color=P.TS65DN, label="Ts65Dn", ms=4),
               Line2D([], [], marker="o", ls="", mfc="white", mec=P.INK, label="Saline", ms=4),
               Line2D([], [], marker="o", ls="", mfc=P.INK, mec=P.INK, label="Memantine", ms=4),
               Line2D([], [], marker="o", ls="", mfc="none", mec=P.SUBTLE_TEXT, label="CS", ms=4),
               Line2D([], [], marker="^", ls="", mfc="none", mec=P.SUBTLE_TEXT, label="SC", ms=4)]
    ax.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              columnspacing=0.8, handletextpad=0.2)
    P.title(ax, "72 mice, one point each (PCA of mouse means)")
    P.save(fig, OUT, "pca_mouse_level")


def validation(metrics):
    """Random row splits inflate accuracy; held-out mice do not."""
    fig, ax = plt.subplots(figsize=(P.DOUBLE * 0.7, 68 * P.MM))
    rng = np.random.default_rng(0)
    designs = list(DESIGN_LABEL)
    for i, design in enumerate(designs):
        for k, model in enumerate(MODEL_STYLE):
            vals = metrics[(metrics.design == design) & (metrics.model == model)].balanced_accuracy
            color, marker, _ = MODEL_STYLE[model]
            x = i + (k - 1) * 0.26
            ax.scatter(x + rng.uniform(-0.06, 0.06, len(vals)), vals, s=7, color=color,
                       marker=marker, lw=0, alpha=0.85)
            ax.hlines(vals.mean(), x - 0.1, x + 0.1, color=P.INK, lw=1.4, zorder=4)
            if model != "dummy":
                P.annotate_stat(ax, x, vals.max() + 0.03, f"{vals.mean():.2f}", ha="center",
                                fontsize=5.6)
    ax.set_xticks(range(3), [DESIGN_LABEL[d] for d in designs])
    ax.set_ylabel("Eight-group balanced accuracy\n(one dot per fold; bar = mean)")
    ax.set_ylim(0, 1.1)
    ax.legend(handles=[Line2D([], [], marker=m, ls="", color=c, label=lab, ms=3.5)
                       for c, m, lab in MODEL_STYLE.values()], loc="center left",
              bbox_to_anchor=(0.0, 0.5))
    P.title(ax, "Validation design changes the conclusion")
    P.save(fig, OUT, "validation_comparison")


def genotype_effects(effects):
    """Largest mouse-level genotype differences (Welch, marginal over other factors)."""
    eff = effects.assign(a=effects.hedges_like_d.abs()).nlargest(15, "a").sort_values("hedges_like_d")
    fig, ax = plt.subplots(figsize=(P.SINGLE, 72 * P.MM))
    colors = np.where(eff.q_value < 0.05, P.VERMILION, P.MUTED)
    ax.hlines(range(len(eff)), 0, eff.hedges_like_d, color=colors, lw=2)
    ax.scatter(eff.hedges_like_d, range(len(eff)), color=colors, s=14, zorder=3)
    ax.set_yticks(range(len(eff)), [display(p) for p in eff.protein])
    P.zero_line(ax, "v")
    ax.set_xlabel("Standardised difference, Ts65Dn − control (pooled SD)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=P.VERMILION, label="q < 0.05", ms=4),
                       Line2D([], [], marker="o", ls="", color=P.MUTED, label="q ≥ 0.05", ms=4)],
              loc="lower right")
    P.title(ax, "Largest genotype differences")
    P.save(fig, OUT, "genotype_effects")


def importance(imp):
    """Permutation importance of the full-data logistic model (descriptive)."""
    imp = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(P.SINGLE, 72 * P.MM))
    y = np.arange(len(imp))
    ax.hlines(y, imp.importance_mean - imp.importance_sd, imp.importance_mean + imp.importance_sd,
              color=P.MUTED, lw=1.2)
    ax.scatter(imp.importance_mean, y, color=P.COBALT, s=14, zorder=3)
    ax.set_yticks(y, [display(p) for p in imp.protein])
    ax.set_xlabel("Drop in balanced accuracy when permuted (mean ± SD)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    P.title(ax, "Predictive importance (not causal)")
    P.save(fig, OUT, "feature_importance")


def confusion(records):
    """Which groups the held-out-mouse classifier confuses."""
    yt = list(chain.from_iterable(r["y_true"] for r in records if r["repeat"] == 0))
    yp = list(chain.from_iterable(r["y_pred"] for r in records if r["repeat"] == 0))
    labels = sorted(set(yt))
    cm = confusion_matrix(yt, yp, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(P.SINGLE * 1.05, 82 * P.MM))
    im = ax.imshow(cm, cmap=COBALT_SEQ, vmin=0, vmax=1)
    P.no_grid(ax)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] >= 0.05:
                P.annotate_stat(ax, j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=5.4,
                                color="white" if cm[i, j] > 0.55 else P.INK)
    ax.set_xticks(range(len(labels)), [P.CELL_LABEL[c] for c in labels], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), [P.CELL_LABEL[c] for c in labels])
    ax.set_xlabel("Predicted group")
    ax.set_ylabel("Observed group")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Fraction of observed group (row)", fontsize=6)
    cb.outline.set_linewidth(0.4)
    P.title(ax, "Errors cluster within conditioning order")
    P.save(fig, OUT, "grouped_confusion_matrix")


def factorial(contrasts):
    """Top FDR discoveries for each factorial term."""
    selected = contrasts.query("q < 0.05").groupby("term", sort=False).head(3).iloc[::-1]
    fig, ax = plt.subplots(figsize=(P.SINGLE * 1.2, 80 * P.MM))
    y = np.arange(len(selected))
    color = np.where(selected.term.str.contains("Genotype"), P.VERMILION, P.COBALT)
    ax.hlines(y, selected.lower_95, selected.upper_95, color=color, lw=1.1)
    ax.scatter(selected.effect_sd, y, color=color, s=14, zorder=3)
    ax.set_yticks(y, [f"{display(p)} · {t.replace('Behavior', 'Conditioning')}"
                     for p, t in zip(selected.protein, selected.term)],
                  fontsize=6)
    P.zero_line(ax, "v")
    ax.set_xlabel("Equal-cell contrast, protein SD units (95% CI)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    P.title(ax, "Strongest contrasts per term (q < 0.05)")
    P.save(fig, OUT, "factorial_contrasts")


def factor_recovery(study):
    """Balanced accuracy for each experimental factor in held-out mice."""
    scores = (study.query("representation == 'mean' and target != 'class'")
              .groupby(["model", "target", "method"]).balanced_accuracy.mean()
              .unstack(["target", "method"]))
    fig, ax = plt.subplots(figsize=(P.DOUBLE * 0.7, 46 * P.MM))
    im = ax.imshow(scores.to_numpy(), cmap=COBALT_SEQ, vmin=0.5, vmax=1, aspect="auto")
    P.no_grid(ax)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            v = scores.iloc[i, j]
            P.annotate_stat(ax, j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if v > 0.82 else P.INK)
    ax.set_xticks(range(scores.shape[1]),
                  [f"{f.replace('Behavior', 'Conditioning')}\n{m}" for f, m in scores.columns],
                  fontsize=6)
    names = {"kumo": "Kumo Tabular Small", "logistic": "Logistic", "tabpfn": "TabPFN v2"}
    ax.set_yticks(range(scores.shape[0]), [names.get(m, m) for m in scores.index])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Balanced accuracy", fontsize=6)
    cb.outline.set_linewidth(0.4)
    P.title(ax, "Conditioning is recovered best, treatment worst")
    P.save(fig, OUT, "factor_recovery")


def main():
    P.apply_style()
    df = load_data(ROOT / "Data/Raw_Data/Data_Cortex_Nuclear.xls")
    features = protein_columns(df)
    mice = aggregate_animals(df)
    missingness(df, features)
    pca(mice, features)
    validation(pd.read_csv(ROOT / "results/metrics/fold_metrics.csv"))
    genotype_effects(pd.read_csv(ROOT / "results/tables/genotype_effects.csv"))
    importance(pd.read_csv(ROOT / "results/tables/permutation_importance.csv"))
    confusion(json.loads((ROOT / "results/metrics/predictions.json").read_text())[
        "grouped_animal:logistic"])
    factorial(pd.read_csv(ROOT / "results/tables/factorial_contrasts.csv"))
    study_path = ROOT / "results/metrics/study.csv"
    if study_path.is_file():
        factor_recovery(pd.read_csv(study_path))
    print(f"wrote figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
