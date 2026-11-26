#!/usr/bin/env python3
"""Figures generated only from saved exploratory extension predictions and scores."""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mice_protein import plotting as P

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "foundation_models/results"
FIG = RES / "figures"
# House colour roles (docs/figure_style.md): foundation models = vermilion family,
# logistic regression = cobalt, other conventional baselines = muted blue.
COLORS = {"logistic": P.COBALT, "svm": P.MUTED, "random_forest": P.MUTED,
          "gradient_boosting": P.MUTED, "factorized_logistic": P.MUTED,
          "tabpfn": "#F09A9B", "kumo": P.VERMILION}
LABELS = {"logistic": "Logistic regression", "svm": "RBF SVM",
          "random_forest": "Random forest", "gradient_boosting": "Histogram boosting",
          "factorized_logistic": "Factorized logistic",
          "tabpfn": "TabPFN v2",
          "kumo": "Kumo Tabular Small"}


def learning_curve() -> None:
    scores = pd.read_csv(RES / "matched_holdout_review/scores.csv")
    scores["n_train"] = scores.n_train.astype(int)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for model in ["logistic", "tabpfn", "kumo"]:
        part = scores[scores.model == model]
        stat = part.groupby("n_train").balanced_accuracy.agg(
            mean="mean", lo=lambda x: np.quantile(x, 0.1),
            hi=lambda x: np.quantile(x, 0.9)).sort_index()
        color = COLORS[model]
        ax.plot(stat.index, stat["mean"], marker="o", lw=2.0,
                color=color, label=LABELS[model])
        ax.fill_between(stat.index.to_numpy(), stat.lo.to_numpy(), stat.hi.to_numpy(),
                        color=color, alpha=0.08, linewidth=0)
    ax.axhline(0.125, color=P.SUBTLE_TEXT, ls="--", lw=0.8)
    ax.set(xlim=(14, 66), ylim=(0.08, 1.02),
           xlabel="Independent labeled training mice (total across eight groups)",
           ylabel="Eight-group balanced accuracy on fixed held-out mice")
    ax.set_title("Performance as the number of labeled mice grows", fontfamily=P.SERIF,
                 fontsize=14, pad=15)
    ax.text(0, -0.23, "Seven-fold outer CV, two repeats, five nested subset draws. Bands are split/subset spread, not population confidence intervals.",
            transform=ax.transAxes, fontsize=8, color=P.SUBTLE_TEXT)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    P.save(fig, FIG, "fig4_matched_learning")
    P.save(learning_linkedin(), FIG, "linkedin_learning")


def learning_linkedin():
    scores = pd.read_csv(RES / "matched_holdout_review/scores.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in ("logistic", "tabpfn", "kumo"):
        stat = scores[scores.model == model].groupby("n_train").balanced_accuracy.mean()
        ax.plot(stat.index, stat.values, lw=3.2, marker="o", ms=7,
                color=COLORS[model], label=LABELS[model])
    ax.axhline(0.125, color=P.SUBTLE_TEXT, ls="--", lw=1)
    ax.set(xlim=(14, 66), ylim=(0.08, 1.0), xlabel="Independent labeled training mice",
           ylabel="Eight-group balanced accuracy")
    ax.set_title("More rows are not more mice", fontfamily=P.SERIF, fontsize=20, pad=22)
    ax.text(0, -0.22, "Matched held-out mice · five subset draws per split · 72 mice total · descriptive averages over overlapping folds",
            transform=ax.transAxes, fontsize=10, color=P.SUBTLE_TEXT)
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    return fig


def reliability() -> None:
    predictions = pd.read_csv(RES / "predictions_class.csv.gz")
    classes = sorted(c for c in predictions if c.startswith(("c-", "t-")))
    # Each biological animal contributes once: average its five out-of-fold predictions.
    pooled = predictions.groupby(["model", "animal", "observed"], as_index=False)[classes].mean()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for model in ("logistic", "tabpfn", "kumo"):
        part = pooled[pooled.model == model]
        p = part[classes].to_numpy()
        confidence = p.max(axis=1)
        correct = (np.asarray(classes)[p.argmax(axis=1)] == part.observed.to_numpy())
        edges = np.linspace(0, 1, 6)
        x, y, n = [], [], []
        for lo, hi in itertools.pairwise(edges):
            mask = (confidence > lo) & (confidence <= hi)
            if mask.sum() >= 3:
                x.append(confidence[mask].mean())
                y.append(correct[mask].mean())
                n.append(mask.sum())
        axes[0].plot(x, y, "o-", color=COLORS[model], label=LABELS[model])
        axes[1].hist(confidence, bins=edges, alpha=0.32, color=COLORS[model],
                     label=LABELS[model])
    axes[0].plot([0, 1], [0, 1], "--", color=P.SUBTLE_TEXT, lw=0.8)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted confidence",
                ylabel="Fraction correct", title="Top-label reliability")
    axes[1].set(xlim=(0, 1), xlabel="Predicted confidence", ylabel="Mice",
                title="Confidence distribution")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Probability reliability is uncertain with 72 mice", fontfamily=P.SERIF,
                 fontsize=13)
    fig.text(0.1, -0.04, "Each mouse contributes once after averaging five held-out predictions. Bins with fewer than three mice are omitted.",
             fontsize=7, color=P.SUBTLE_TEXT)
    P.save(fig, FIG, "fig5_reliability_72_mice")


def sensitivity() -> None:
    scores = pd.read_csv(RES / "sensitivity/scores.csv")
    order = ["arithmetic", "geometric", "geometric_no_duplicate_sparse", "missingness_only"]
    names = ["Arithmetic mean", "Geometric mean", "Geometric; remove duplicate/sparse",
             "Missingness only"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, model in enumerate(("logistic", "kumo")):
        means = scores[scores.model == model].groupby("version").balanced_accuracy.mean().reindex(order)
        ax.scatter(means, np.arange(4) + (i - 0.5) * 0.14, s=80, color=COLORS[model],
                   label=LABELS[model])
    ax.axvline(0.125, color=P.SUBTLE_TEXT, ls="--", lw=0.8)
    ax.set_yticks(np.arange(4), names)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel="Held-out mouse balanced accuracy (mean of two CV repeats)")
    ax.set_title("Preprocessing and missingness sensitivity", fontfamily=P.SERIF, fontsize=13)
    ax.legend(frameon=False, loc="lower right")
    P.save(fig, FIG, "fig7_sensitivity")


def leakage() -> None:
    rows = pd.read_csv(RES / "leakage.csv")
    means = rows.groupby(["model", "design"]).balanced_accuracy.mean().unstack()
    order = ["logistic", "random_forest", "tabpfn", "kumo"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(order))
    ax.scatter(means.loc[order, "random_rows"], y - 0.12, color=P.MUTED,
               s=65, label="Random measurement rows")
    ax.scatter(means.loc[order, "mouse_held_out"], y + 0.12, color=P.INK,
               s=65, label="Held-out mice, same row features")
    for i, model in enumerate(order):
        ax.plot([means.loc[model, "mouse_held_out"], means.loc[model, "random_rows"]],
                [i + 0.12, i - 0.12], color=P.RULE, lw=1.2, zorder=0)
    ax.set_yticks(y, [LABELS[m] for m in order])
    ax.invert_yaxis()
    ax.set(xlim=(0.5, 1.03), xlabel="Row-weighted eight-group balanced accuracy")
    ax.set_title("A random row split tests familiar mice", fontfamily=P.SERIF, fontsize=13)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.92))
    fig.text(0.1, -0.03, "Same 1,080 measurement rows in both analyses; only the split rule changes. One five-fold CV repeat.",
             fontsize=7, color=P.SUBTLE_TEXT)
    P.save(fig, FIG, "fig2_leakage")


def factors() -> None:
    table = pd.read_csv(RES / "tables/factor_targets.csv", index_col=0)
    targets = [("genotype", "Genotype"), ("treatment", "Treatment"),
               ("behavior", "Conditioning")]
    models = ["logistic", "tabpfn", "kumo"]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    y = np.arange(3)
    for i, model in enumerate(models):
        values = [table.loc[model, key] for key, _ in targets]
        ax.scatter(values, y + (i - 1) * 0.16, color=COLORS[model], s=70,
                   label=LABELS[model])
    ax.axvline(0.5, color=P.SUBTLE_TEXT, ls="--", lw=0.8)
    ax.set_yticks(y, [label for _, label in targets])
    ax.invert_yaxis()
    ax.set(xlim=(0.4, 1.04), xlabel="Binary balanced accuracy on held-out mice")
    ax.set_title("The experimental factors differ in predictability",
                 fontfamily=P.SERIF, fontsize=13)
    ax.legend(frameon=False, loc="lower right")
    fig.text(0.1, -0.03, "Exploratory binary targets; five repeats of five-fold CV on the same 72 mice.",
             fontsize=7, color=P.SUBTLE_TEXT)
    P.save(fig, FIG, "fig6_factors")


def tuned_benchmark() -> None:
    """Matched outer folds; tuned conventional models are explicitly post hoc."""
    original = pd.read_csv(RES / "predictions_class.csv.gz")
    classes = sorted(c for c in original if c.startswith(("c-", "t-")))
    rows = []
    for model in ("tabpfn", "kumo"):
        part = original[original.model == model]
        for repeat, block in part.groupby("repeat"):
            predicted = np.asarray(classes)[block[classes].to_numpy().argmax(axis=1)]
            from sklearn.metrics import balanced_accuracy_score

            rows.append({"model": model, "repeat": repeat,
                         "balanced_accuracy": balanced_accuracy_score(block.observed, predicted)})
    for model in ("logistic", "svm", "random_forest", "gradient_boosting"):
        part = pd.read_csv(RES / f"tuned_baselines/predictions_{model}.csv.gz")
        pcols = [f"p_{c}" for c in classes]
        direct_svm = None
        if model == "svm":
            direct_svm = pd.read_csv(RES / "svm_main_audit/predictions.csv")
            direct_svm = direct_svm[direct_svm.version == "post_hoc_wider_grid"]
        for repeat, block in part.groupby("repeat"):
            predicted = np.asarray(classes)[block[pcols].to_numpy().argmax(axis=1)]
            if direct_svm is not None:
                chosen = direct_svm[direct_svm.repeat == repeat]
                predicted = chosen.predicted
                block = chosen
            rows.append({"model": model, "repeat": repeat,
                         "balanced_accuracy": balanced_accuracy_score(block.observed, predicted)})
    factorized = pd.read_csv(RES / "factorized/predictions.csv.gz")
    for repeat, block in factorized.groupby("repeat"):
        rows.append({"model": "factorized_logistic", "repeat": repeat,
                     "balanced_accuracy": balanced_accuracy_score(block.observed,
                                                                 block.predicted)})
    data = pd.DataFrame(rows)
    order = ["gradient_boosting", "random_forest", "svm", "factorized_logistic",
             "logistic", "tabpfn", "kumo"]
    fig, ax = plt.subplots(figsize=(7.6, 4))
    for i, model in enumerate(order):
        values = data[data.model == model].balanced_accuracy.to_numpy()
        ax.scatter(values, np.full(len(values), i), color=COLORS.get(model, P.MUTED),
                   s=25, alpha=0.5)
        ax.scatter(values.mean(), i, color=COLORS.get(model, P.MUTED),
                   s=95, edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(values.mean() + 0.025, i, f"{values.mean():.3f}", va="center", fontsize=8)
    ax.axvline(0.125, color=P.SUBTLE_TEXT, ls="--", lw=0.8)
    ax.set_yticks(np.arange(len(order)), [LABELS.get(m, "Histogram boosting") for m in order])
    ax.set(xlim=(0, 1.0), xlabel="Eight-group balanced accuracy on held-out mice")
    ax.set_title("A wider conventional search leaves Kumo ahead in this cohort",
                 fontfamily=P.SERIF, fontsize=13)
    fig.text(0.1, -0.02, "Five dots per model are overlapping CV repeats, not independent cohorts. Conventional tuning is post hoc; foundation inference settings are original.",
             fontsize=7, color=P.SUBTLE_TEXT)
    P.save(fig, FIG, "fig3_tuned_benchmark")


if __name__ == "__main__":
    P.apply_style()
    learning_curve()
    reliability()
    sensitivity()
    leakage()
    factors()
    if all((RES / f"tuned_baselines/predictions_{m}.csv.gz").exists()
           for m in ("logistic", "svm", "random_forest", "gradient_boosting")):
        tuned_benchmark()
