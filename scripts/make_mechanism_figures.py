#!/usr/bin/env python3
"""Draw the paper's figures from saved result tables (no analysis is re-run here).

Each function draws one figure; its docstring is the figure's purpose. All inputs
come from ``results/mechanisms/`` and ``results/metrics/`` so figures can be
regenerated or restyled without recomputing statistics.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from mice_protein import plotting as P
from mice_protein.annotations import EXPECTED_DOSAGE_LOG2FC, display
from mice_protein.data import CELLS

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/mechanisms"
TAB = RES / "tables"
FIG = RES / "figures"
SUMMARY = json.loads((RES / "summary.json").read_text())
ARMS = ["control saline", "control memantine", "Ts65Dn saline", "Ts65Dn memantine"]
ARM_LABEL = {"control saline": "Control\nsaline", "control memantine": "Control\nmemantine",
             "Ts65Dn saline": "Ts65Dn\nsaline", "Ts65Dn memantine": "Ts65Dn\nmemantine"}
ARM_CODE = {"control saline": ("c", "s"), "control memantine": ("c", "m"),
            "Ts65Dn saline": ("t", "s"), "Ts65Dn memantine": ("t", "m")}


def mouse_marker(ax, x, y, genotype, behavior, treatment, size=14, alpha=0.9, z=3):
    color = P.GENOTYPE_COLOR[genotype]
    ax.scatter(x, y, s=size, marker=P.BEHAVIOR_MARKER[behavior], linewidths=0.7,
               edgecolors=color, facecolors=color if treatment == "m" else "white",
               alpha=alpha, zorder=z)


def mean_ci(ax, x, values, color, width=0.18):
    values = np.asarray(values)
    mean = values.mean()
    half = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    ax.hlines(mean, x - width, x + width, color=color, lw=1.4, zorder=4)
    ax.vlines(x, mean - half, mean + half, color=color, lw=0.9, zorder=4)


def design_legend(ax, loc="upper left", **kw):
    handles = [
        Line2D([], [], marker="s", ls="", color=P.CONTROL, label="Control", ms=4),
        Line2D([], [], marker="s", ls="", color=P.TS65DN, label="Ts65Dn", ms=4),
        Line2D([], [], marker="o", ls="", mfc="white", mec=P.INK, label="Saline", ms=4),
        Line2D([], [], marker="o", ls="", mfc=P.INK, mec=P.INK, label="Memantine", ms=4),
        Line2D([], [], marker="o", ls="", mfc="none", mec=P.SUBTLE_TEXT, label="CS", ms=4),
        Line2D([], [], marker="^", ls="", mfc="none", mec=P.SUBTLE_TEXT, label="SC", ms=4),
    ]
    ax.legend(handles=handles, loc=loc, ncol=3, columnspacing=0.8, handletextpad=0.2, **kw)


# ---------------------------------------------------------------------------
def figure1():
    """Design, technical structure, and why the mouse is the experimental unit."""
    fig = plt.figure(figsize=(P.DOUBLE, 120 * P.MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.25, 1.0], hspace=0.6, wspace=0.55)

    ax = fig.add_subplot(gs[0, 0])
    n = SUMMARY["data"]["cell_n"]
    for i, (g, gname) in enumerate((("c", "Control"), ("t", "Ts65Dn"))):
        for j, (b, t) in enumerate((("CS", "s"), ("CS", "m"), ("SC", "s"), ("SC", "m"))):
            color = P.GENOTYPE_COLOR[g]
            ax.add_patch(Rectangle((j, 1 - i), 0.92, 0.85, facecolor=color if t == "m" else "white",
                                   edgecolor=color, lw=1.0, alpha=0.9 if t == "m" else 1))
            P.annotate_stat(ax, j + 0.46, 1 - i + 0.42, f"n={n[f'{g}-{b}-{t}']}", ha="center",
                            va="center", color="white" if t == "m" else P.INK, fontsize=6.2)
        ax.text(-0.1, 1 - i + 0.42, gname, ha="right", va="center", fontsize=6.5)
    for j, label in enumerate(("saline", "mem.", "saline", "mem.")):
        ax.text(j + 0.46, 1.95, label, ha="center", fontsize=5.8, color=P.SUBTLE_TEXT)
    for j, label in ((0.96, "CS"), (2.96, "SC")):
        ax.text(j, 2.25, label, ha="center", va="bottom", fontsize=6.2, linespacing=1.1)
        ax.plot([j - 0.9, j + 0.9], [2.2, 2.2], color=P.RULE, lw=0.8)
    ax.text(1.96, -0.25, "CS: context then shock (associative learning)\n"
            "SC: shock then context (no association)\n72 mice × 15 measurements = 1,080 rows",
            ha="center", va="top", fontsize=5.8, color=P.SUBTLE_TEXT, linespacing=1.3)
    ax.set_xlim(-1.2, 4.0)
    ax.set_ylim(-1.35, 2.7)
    ax.axis("off")
    P.title(ax, "2 × 2 × 2 factorial design", pad=10)
    ax.text(4.0, 2.7, "SCHEMATIC", ha="right", va="bottom",
            fontfamily=P.MONO, fontsize=5.2, color=P.SUBTLE_TEXT)
    P.panel_label(ax, "a", x=-0.3, y=1.0)

    ax = fig.add_subplot(gs[0, 1:])
    profile = pd.read_csv(TAB / "qc_block_profile.csv", index_col=0)
    keep = [c for c in profile.columns if c not in SUMMARY["data"]["duplicates"]]
    mat = profile[keep].T
    order = leaves_list(linkage(mat.fillna(0).to_numpy(), "average", metric="correlation"))
    mat = mat.iloc[order]
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap=P.DIVERGING, vmin=-1.2, vmax=1.2,
                   interpolation="nearest")
    for edge in (2.5, 5.5, 8.5, 11.5):
        ax.axvline(edge, color="white", lw=1.2)
    ax.set_xticks(range(15), range(1, 16))
    ax.set_yticks([])
    ax.set_xlabel("Measurement index within mouse (MouseID suffix)")
    ax.set_ylabel(f"{len(keep)} proteins (clustered)")
    for k in range(5):
        ax.text(3 * k + 1, -1.2, f"block {k + 1}", ha="center", fontsize=6, color=P.SUBTLE_TEXT)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Mean within-mouse z", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)
    P.title(ax, "Rows come in five technical blocks of three", pad=12)
    P.panel_label(ax, "b", x=-0.06, y=1.04)

    ax = fig.add_subplot(gs[1, 0])
    comp = pd.read_csv(TAB / "qc_variance_components.csv")
    comp = comp[~comp.protein.isin(SUMMARY["data"]["duplicates"])]
    rng = np.random.default_rng(3)
    rows = [("Between mice\n(share of total)", comp.icc, P.COBALT),
            ("Technical block\n(share of within-mouse)", comp.block_share, P.MUTED)]
    for i, (label, values, color) in enumerate(rows):
        ax.scatter(values, i + rng.uniform(-0.18, 0.18, len(values)), s=6, color=color, lw=0,
                   alpha=0.9)
        med = float(np.median(values))
        ax.vlines(med, i - 0.3, i + 0.3, color=P.INK, lw=1.3)
        P.annotate_stat(ax, 0.02, i - 0.3, f"median {med:.2f}", ha="left", va="center",
                        fontsize=5.6)
    ax.set_yticks(range(2), [r[0] for r in rows], fontsize=6)
    ax.set_ylim(-0.5, 1.75)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Variance fraction (one dot per protein)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    P.title(ax, "Most variance is between mice")
    P.panel_label(ax, "c", x=-0.62)

    ax = fig.add_subplot(gs[1, 1:])
    folds = pd.read_csv(ROOT / "results/metrics/fold_metrics.csv")
    folds = folds[folds.model.isin(["logistic", "random_forest"])]
    designs = [("random_observation", "Random rows\n(mice leak)"),
               ("grouped_animal", "Held-out mice\n(15 rows each)"),
               ("mouse_mean", "Held-out mice\n(mouse means)")]
    colors = {"logistic": P.CONTROL, "random_forest": P.MUTED}
    rng = np.random.default_rng(0)
    for i, (design, label) in enumerate(designs):
        for k, (model, mlabel) in enumerate((("logistic", "Logistic"), ("random_forest", "Random forest"))):
            vals = folds[(folds.design == design) & (folds.model == model)].balanced_accuracy
            x = i + (k - 0.5) * 0.32
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(vals)), vals, s=7, color=colors[model],
                       alpha=0.8, lw=0, marker="o" if k == 0 else "D")
            ax.hlines(vals.mean(), x - 0.11, x + 0.11, color=P.INK, lw=1.4, zorder=4)
            if k == 0:
                P.annotate_stat(ax, x - 0.14, vals.mean(), f"{vals.mean():.2f}", ha="right",
                                va="center", fontsize=5.8)
    ax.axhline(0.125, color=P.SUBTLE_TEXT, ls=(0, (3, 2)), lw=0.6)
    ax.text(2.6, 0.14, "chance (1/8)", fontsize=5.8, color=P.SUBTLE_TEXT, ha="right")
    ax.set_xticks(range(3), [d[1] for d in designs])
    ax.set_ylabel("Eight-group balanced accuracy\n(one dot per CV fold; bar = mean)")
    ax.set_ylim(0, 1.05)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=P.CONTROL, label="Logistic", ms=3.5),
                       Line2D([], [], marker="D", ls="", color=P.MUTED, label="Random forest", ms=3)],
              loc="lower left")
    P.title(ax, "Leakage between technical replicates inflates accuracy")
    P.panel_label(ax, "d", x=-0.1)
    P.save(fig, FIG, "fig1_design_qc")


# ---------------------------------------------------------------------------
def callout_column(ax, points, x_text, y_range, colors, fontsize=5.8, ha="left"):
    """Label points in an evenly spaced column with thin leader lines (no collisions)."""
    points = sorted(points, key=lambda p: p[2])
    ys = np.linspace(*y_range, len(points)) if len(points) > 1 else [np.mean(y_range)]
    for (name, x, y, cls), yt in zip(points, ys):
        ax.annotate(name, (x, y), xytext=(x_text, yt), textcoords="data", fontsize=fontsize,
                    color=colors[cls], ha=ha, va="center",
                    arrowprops={"arrowstyle": "-", "color": P.RULE, "lw": 0.6,
                                "shrinkA": 1, "shrinkB": 2})


def figure2():
    """Gene dosage of triplicated proteins with independent replication."""
    d = pd.read_csv(TAB / "dosage_fold_changes.csv").dropna(subset=["estimate_saline"])
    colors = {"Triplicated in Ts65Dn": P.TS65DN, "HSA21 ortholog, disomic": P.INK,
              "Not on HSA21": P.MUTED}
    markers = {"Triplicated in Ts65Dn": "o", "HSA21 ortholog, disomic": "D", "Not on HSA21": "o"}
    fig = plt.figure(figsize=(P.DOUBLE, 140 * P.MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.25, 1.0],
                          hspace=0.45, wspace=0.42)

    ax = fig.add_subplot(gs[0, :])
    d = d.sort_values("estimate_saline").reset_index(drop=True)
    for cls, color in colors.items():
        sub = d[d.dosage_class == cls]
        background = cls == "Not on HSA21"
        ax.vlines(sub.index, sub.lower_saline, sub.upper_saline, color=color,
                  lw=0.6 if background else 1.1)
        ax.scatter(sub.index, sub.estimate_saline, s=7 if background else 18, marker=markers[cls],
                   color=color, zorder=3, label=cls, edgecolors="white", linewidths=0.4)
    highlighted = d[d.dosage_class != "Not on HSA21"].sort_values("estimate_saline")
    previous_x, previous_y = -99, -99.0
    for i, row in highlighted.iterrows():
        y_text = max(row.upper_saline, 0.3) + 0.12
        if i - previous_x < 6:  # stack labels of neighbouring points
            y_text = max(y_text, previous_y + 0.14)
        previous_x, previous_y = i, y_text
        ax.annotate(display(row.protein), (i, row.upper_saline), xytext=(i, y_text),
                    ha="center", fontsize=6, color=colors[row.dosage_class],
                    arrowprops={"arrowstyle": "-", "color": P.RULE, "lw": 0.6, "shrinkA": 0,
                                "shrinkB": 1})
    P.zero_line(ax)
    ax.axhline(EXPECTED_DOSAGE_LOG2FC, color=P.TS65DN, lw=0.6, ls=(0, (1, 1.5)))
    ax.text(len(d) * 0.55, EXPECTED_DOSAGE_LOG2FC + 0.03, "strict dosage: 1.5× = +0.58 log₂",
            fontsize=6, color=P.TS65DN, ha="center")
    ax.set_xticks([])
    ax.set_xlim(-1.5, len(d) + 0.5)
    ax.set_xlabel(f"{len(d)} proteins, ordered by estimated effect")
    ax.set_ylabel("log₂ fold change, Ts65Dn / control\n(saline mice; 95% CI)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), handletextpad=0.1)
    P.title(ax, "Triplicated proteins rise, but far less than gene copy number predicts")
    P.panel_label(ax, "a", x=-0.07, y=1.02)

    ax = fig.add_subplot(gs[1, 0])
    bg = d[d.dosage_class == "Not on HSA21"]
    ax.scatter(bg.estimate_saline, bg.estimate_memantine, s=7, color=P.MUTED, lw=0)
    points = []
    for cls in ("HSA21 ortholog, disomic", "Triplicated in Ts65Dn"):
        sub = d[d.dosage_class == cls]
        ax.errorbar(sub.estimate_saline, sub.estimate_memantine,
                    xerr=[sub.estimate_saline - sub.lower_saline, sub.upper_saline - sub.estimate_saline],
                    yerr=[sub.estimate_memantine - sub.lower_memantine,
                          sub.upper_memantine - sub.estimate_memantine],
                    fmt=markers[cls], ms=3.6, color=colors[cls], elinewidth=0.5, capsize=0,
                    mec="white", mew=0.4, alpha=0.95, zorder=3)
        points += [(display(r.protein), r.estimate_saline, r.estimate_memantine, cls)
                   for r in sub.itertuples()]
    lim = np.array([-0.5, 0.8])
    ax.plot(lim, lim, color=P.SUBTLE_TEXT, lw=0.5, ls=(0, (3, 2)))
    P.zero_line(ax)
    P.zero_line(ax, "v")
    callout_column(ax, points, 0.86, (-0.42, 0.72), colors)
    ax.set_xlim(lim[0], 1.12)
    ax.set_ylim(*lim)
    ax.spines["bottom"].set_bounds(*lim)
    ax.set_xticks([-0.4, 0, 0.4, 0.8])
    r = SUMMARY["dosage"]["replication_pearson"]
    P.annotate_stat(ax, 0.02, 0.97, f"Pearson r = {r:.2f}, {len(d)} proteins", va="top",
                    transform=ax.transAxes)
    ax.set_xlabel("log₂FC in saline mice (discovery)")
    ax.set_ylabel("log₂FC in memantine mice (replication)")
    ax.grid(axis="x")
    P.title(ax, "Effects replicate in independent mice")
    P.panel_label(ax, "b", x=-0.2, y=1.02)

    ax = fig.add_subplot(gs[1, 1])
    sets = SUMMARY["dosage"]["sets"]
    names = [("triplicated", "Triplicated", P.TS65DN, "o"),
             ("hsa21_disomic", "HSA21,\ndisomic", P.INK, "D"),
             ("background", "Other\nproteins", P.MUTED, "o")]
    for i, (key, label, color, marker) in enumerate(names):
        s = sets[key]
        ax.errorbar(i, s["estimate"], yerr=[[s["estimate"] - s["lower"]], [s["upper"] - s["estimate"]]],
                    fmt=marker, color=color, ms=5.5, elinewidth=1.2, capsize=0, mec="white", mew=0.5)
        P.annotate_stat(ax, i, s["upper"] + 0.02, f"{s['estimate']:+.2f}\nn={s['n']}",
                        ha="center", va="bottom", fontsize=5.8)
    P.zero_line(ax)
    ax.axhline(EXPECTED_DOSAGE_LOG2FC, color=P.TS65DN, lw=0.6, ls=(0, (1, 1.5)))
    ax.text(2.6, EXPECTED_DOSAGE_LOG2FC - 0.03, "strict dosage", fontsize=6, color=P.TS65DN,
            ha="right", va="top")
    ax.set_xticks(range(3), [n[1] for n in names])
    ax.set_xlim(-0.5, 2.7)
    ax.set_ylim(-0.15, 0.65)
    ax.set_ylabel("Mean log₂FC across set\n(72 mice; mouse bootstrap 95% CI)")
    frac = SUMMARY["dosage"]["set_fraction_of_expected"]
    P.title(ax, f"Set mean reaches {100 * frac:.0f}% of the dosage expectation")
    P.panel_label(ax, "c", x=-0.25, y=1.02)
    P.save(fig, FIG, "fig2_gene_dosage")


# ---------------------------------------------------------------------------
def figure3():
    """Learning signature: magnitude and direction of the CS-SC response per arm."""
    L = SUMMARY["learning"]
    scores = pd.read_csv(TAB / "learning_scores.csv")
    parts = scores.cell.str.split("-", expand=True)
    scores[["g", "b", "t"]] = parts
    fig = plt.figure(figsize=(P.DOUBLE, 112 * P.MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1.0, 1.0], height_ratios=[1, 1], hspace=0.7,
                          wspace=0.75)

    ax = fig.add_subplot(gs[:, 0])
    rng = np.random.default_rng(1)
    for i, arm in enumerate(ARMS):
        g, t = ARM_CODE[arm]
        for k, b in enumerate(("SC", "CS")):
            sub = scores[(scores.g == g) & (scores.t == t) & (scores.b == b)]
            x = i + (k - 0.5) * 0.36
            mouse_marker(ax, x + rng.uniform(-0.06, 0.06, len(sub)), sub.learning_score, g, b, t,
                         size=12, alpha=0.85)
            mean_ci(ax, x, sub.learning_score, P.INK, width=0.11)
        lo = scores[(scores.g == g) & (scores.t == t) & (scores.b == "SC")].learning_score.mean()
        hi = scores[(scores.g == g) & (scores.t == t) & (scores.b == "CS")].learning_score.mean()
        ax.add_patch(FancyArrowPatch((i + 0.33, lo), (i + 0.33, hi), arrowstyle="-|>",
                                     mutation_scale=6, color=P.GENOTYPE_COLOR[g], lw=0.9))
        ax.text(i + 0.38, (lo + hi) / 2, f"{L['response'][arm]:.1f}", fontsize=6,
                color=P.GENOTYPE_COLOR[g], va="center")
    ax.set_xticks(range(4), [ARM_LABEL[a] for a in ARMS])
    ax.set_xlim(-0.5, 3.75)
    ax.set_ylabel("Score on control-saline CS–SC axis\n(within-cell SD units, leave-one-out)")
    design_legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    P.title(ax, "Ts65Dn saline mice respond least")
    P.panel_label(ax, "a", x=-0.14, y=1.02)

    ax = fig.add_subplot(gs[0, 1])
    for i, arm in enumerate(ARMS):
        g, _ = ARM_CODE[arm]
        lo, hi = L["relative_ci"][arm]
        est = L["relative"][arm]
        if arm == "control saline":
            ax.scatter(i, est, color=P.GENOTYPE_COLOR[g], s=18, marker="D", zorder=3)
        else:
            ax.errorbar(i, est, yerr=[[est - lo], [hi - est]], fmt="o", ms=4.5, capsize=0,
                        color=P.GENOTYPE_COLOR[g], mfc="white" if arm.endswith("saline") else
                        P.GENOTYPE_COLOR[g], elinewidth=1)
    ax.axhline(1, color=P.SUBTLE_TEXT, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(range(4), ["Ctrl\nsal", "Ctrl\nmem", "Ts\nsal", "Ts\nmem"])
    ax.set_ylabel("Response relative to\ncontrol saline")
    top = max(v[1] for v in L["relative_ci"].values())
    y1, y2 = top + 0.5, top + 0.12
    for (a, b), y, pval in (((0, 2), y1, L["attenuation_p"]), ((2, 3), y2, L["restoration_p"])):
        ax.plot([a, a, b, b], [y - 0.05, y, y, y - 0.05], color=P.INK, lw=0.6)
        P.annotate_stat(ax, (a + b) / 2, y + 0.03, f"p = {pval:.3f}", ha="center", fontsize=5.8)
    ax.set_ylim(0, y1 + 0.28)
    P.title(ax, "Blunted, then restored")
    P.panel_label(ax, "b", x=-0.42)

    ax = fig.add_subplot(gs[1, 1])
    arms3 = [a for a in ARMS if a != "control saline"]
    for i, arm in enumerate(arms3):
        g, _ = ARM_CODE[arm]
        est = L["cosine"][arm]
        lo, hi = L["cosine_ci"][arm]
        ax.errorbar(i, est, yerr=[[est - lo], [hi - est]], fmt="o", ms=4.5, capsize=0,
                    color=P.GENOTYPE_COLOR[g], mfc="white" if arm.endswith("saline") else
                    P.GENOTYPE_COLOR[g], elinewidth=1)
    ax.axhline(1, color=P.SUBTLE_TEXT, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(range(3), ["Ctrl\nmem", "Ts\nsal", "Ts\nmem"])
    ax.set_ylim(0, 1.35)
    ax.set_ylabel("Cosine with control saline\n(noise-corrected)")
    P.title(ax, "Same direction in every arm")
    P.panel_label(ax, "c", x=-0.42)

    ax = fig.add_subplot(gs[:, 2])
    load = pd.read_csv(TAB / "learning_axis_loadings.csv").head(15).iloc[::-1]
    y = np.arange(len(load))
    color = np.where(load.loading > 0, P.TS65DN, P.CONTROL)
    ax.hlines(y, load.lower, load.upper, color=color, lw=0.9)
    ax.scatter(load.loading, y, color=color, s=12, zorder=3)
    P.zero_line(ax, "v")
    ax.set_yticks(y, [display(p) for p in load.protein], fontsize=6)
    ax.set_xlabel("Axis loading (bootstrap 95% CI)")
    ax.text(0.99, 0.995, "higher in CS →", transform=ax.transAxes, ha="right", va="top",
            fontsize=5.8, color=P.TS65DN)
    ax.text(0.01, 0.005, "← higher in SC", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=5.8, color=P.CONTROL)
    ax.grid(axis="x")
    P.title(ax, "Axis-defining proteins")
    P.panel_label(ax, "d", x=-0.45, y=1.03)
    P.save(fig, FIG, "fig3_learning_signature")


# ---------------------------------------------------------------------------
def figure4():
    """Memantine rescue along the trisomy axis and per-protein persistence."""
    R = SUMMARY["rescue"]
    scores = pd.read_csv(TAB / "trisomy_axis_scores.csv")
    scores[["g", "b", "t"]] = scores.cell.str.split("-", expand=True)
    fig = plt.figure(figsize=(P.DOUBLE, 118 * P.MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.25], hspace=0.75, wspace=0.95)

    ax = fig.add_subplot(gs[:, 0])
    rng = np.random.default_rng(2)
    for i, arm in enumerate(ARMS):
        g, t = ARM_CODE[arm]
        sub = scores[(scores.g == g) & (scores.t == t)]
        for b in ("CS", "SC"):
            s = sub[sub.b == b]
            mouse_marker(ax, i + rng.uniform(-0.17, 0.17, len(s)), s.trisomy_score, g, b, t, size=12)
        m = R["arm_means"][arm]
        lo, hi = R["arm_means_ci"][arm]
        ax.hlines(m, i - 0.25, i + 0.25, color=P.INK, lw=1.4, zorder=4)
        ax.vlines(i + 0.3, lo, hi, color=P.INK, lw=0.9)
    ax.axhline(R["arm_means"]["control saline"], color=P.CONTROL, lw=0.6, ls=(0, (3, 2)))
    ax.add_patch(FancyArrowPatch((2.0, R["arm_means"]["Ts65Dn saline"]),
                                 (3.0, R["arm_means"]["Ts65Dn memantine"]),
                                 arrowstyle="-|>", mutation_scale=8, color=P.INK, lw=0.8,
                                 connectionstyle="arc3,rad=-0.25", zorder=5))
    P.annotate_stat(ax, 3.45, -4.9, f"{100 * R['rescue_fraction']:.0f}% of saline gap\n"
                    f"closed, p = {R['rescue_shift_p']:.3f}", ha="right", va="bottom", fontsize=5.8)
    ax.set_xticks(range(4), [ARM_LABEL[a] for a in ARMS])
    ax.set_ylabel("Score on trisomy axis (saline Ts65Dn − control)\n(within-cell SD units, leave-one-out)")
    design_legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.set_xlim(-0.5, 3.5)
    ax.set_xticks(range(4), ["Ctrl\nsal", "Ctrl\nmem", "Ts\nsal", "Ts\nmem"])
    P.title(ax, "Drug moves Ts65Dn")
    P.panel_label(ax, "a", x=-0.2, y=1.02)

    ax = fig.add_subplot(gs[0, 1])
    rows = [("Rescue\n(vs control saline)", R["rescue_fraction"], R["rescue_fraction_ci"], P.TS65DN),
            ("Rescue in CS", R["rescue_by_conditioning"]["CS"], R["rescue_by_conditioning_ci"]["CS"], P.TS65DN),
            ("Rescue in SC", R["rescue_by_conditioning"]["SC"], R["rescue_by_conditioning_ci"]["SC"], P.TS65DN),
            ("Residual gap\n(vs control memantine)", R["persistence"], R["persistence_ci"], P.INK),
            ("Control shift\nwith memantine", R["control_shift"], R["control_shift_ci"], P.CONTROL)]
    for i, (label, est, ci, color) in enumerate(rows):
        ax.errorbar(est, -i, xerr=[[est - ci[0]], [ci[1] - est]], fmt="o", ms=4, color=color,
                    elinewidth=1, capsize=0)
    ax.axvline(0, color=P.SUBTLE_TEXT, lw=0.6, ls=(0, (3, 2)))
    ax.axvline(1, color=P.SUBTLE_TEXT, lw=0.6, ls=(0, (1, 1.5)))
    ax.set_yticks(-np.arange(len(rows)), [r[0] for r in rows], fontsize=6)
    ax.set_xlabel("Fraction of saline trisomy gap\n(bootstrap 95% CI)")
    ax.set_xlim(-1.5, 3.0)
    for i, (_, est, ci, color) in enumerate(rows):
        if ci[0] < -1.5:
            ax.annotate("", (-1.5, -i), (-1.2, -i), arrowprops={"arrowstyle": "-|>", "color": color,
                                                                "lw": 0.8, "mutation_scale": 5})
        if ci[1] > 3.0:
            ax.annotate("", (3.0, -i), (2.7, -i), arrowprops={"arrowstyle": "-|>", "color": color,
                                                              "lw": 0.8, "mutation_scale": 5})
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    P.title(ax, "But controls shift too")
    P.panel_label(ax, "b", x=-0.85)

    ax = fig.add_subplot(gs[1, 1])
    mags = R["magnitudes"]
    labels = ["Trisomy\n(saline)", "Drug in\nTs65Dn", "Drug in\ncontrol"]
    vals = [mags["Trisomy (saline)"], mags["Memantine in Ts65Dn"], mags["Memantine in control"]]
    ax.bar(range(3), vals, color=[P.TS65DN, P.INK, P.CONTROL], width=0.55, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.08, f"{v:.1f}", ha="center", fontsize=6)
    ax.set_xticks(range(3), labels, fontsize=5.8)
    ax.set_ylabel("Effect size |Δ|, SD units\n(noise-corrected)")
    P.title(ax, "Drug effect rivals trisomy")
    P.panel_label(ax, "c", x=-0.42)

    ax = fig.add_subplot(gs[:, 2])
    per = pd.read_csv(TAB / "rescue_per_protein.csv")
    cat_color = {"Not affected (saline)": P.MUTED, "Persists under memantine": P.TS65DN,
                 "Not detected under memantine": P.INK}
    cat_marker = {"Not affected (saline)": "o", "Persists under memantine": "s",
                  "Not detected under memantine": "D"}
    for cat, color in cat_color.items():
        sub = per[per.category == cat]
        ax.scatter(sub.estimate_saline, sub.estimate_memantine, s=10 if cat != "Not affected (saline)" else 6,
                   color=color, marker=cat_marker[cat], lw=0, alpha=0.9,
                   label=f"{cat} ({len(sub)})")
    affected = per[per.category != "Not affected (saline)"]
    for _, row in affected.iterrows():
        ax.annotate(display(row.protein), (row.estimate_saline, row.estimate_memantine), xytext=(2, 2),
                    textcoords="offset points", fontsize=5.3, color=cat_color[row.category])
    lim = np.array([per[["estimate_saline", "estimate_memantine"]].min().min() - 0.05,
                    per[["estimate_saline", "estimate_memantine"]].max().max() + 0.05])
    ax.plot(lim, lim, color=P.SUBTLE_TEXT, lw=0.6, ls=(0, (3, 2)), label="No rescue (y = x)")
    ax.plot(lim, R["persistence_slope_corrected"] * lim, color=P.INK, lw=0.8,
            label=f"Fitted slope {R['persistence_slope_corrected']:.2f} (error-corrected)")
    P.zero_line(ax)
    P.zero_line(ax, "v")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("log₂FC Ts65Dn/control, saline mice")
    ax.set_ylabel("log₂FC Ts65Dn/control, memantine mice")
    ax.legend(loc="lower right", fontsize=5.6, handletextpad=0.2)
    ax.grid(axis="x")
    P.title(ax, "Per-protein trisomy effects persist")
    P.panel_label(ax, "d", x=-0.25, y=1.03)
    P.save(fig, FIG, "fig4_rescue")


# ---------------------------------------------------------------------------
def heatmap(ax, table, rows, cols, row_labels, vmax, title):
    est = table.pivot(index="feature", columns="contrast", values="estimate").loc[rows, cols]
    q = table.pivot(index="feature", columns="contrast", values="q").loc[rows, cols]
    im = ax.imshow(est.to_numpy(), cmap=P.DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    P.no_grid(ax)
    for i in range(len(rows)):
        for j in range(len(cols)):
            mark = "**" if q.iloc[i, j] < 0.01 else "*" if q.iloc[i, j] < 0.05 else ""
            if mark:
                ax.text(j, i + 0.08, mark, ha="center", va="center", fontsize=6.5, color=P.INK)
    ax.set_xticks(range(len(cols)), [c.replace("Conditioning", "Cond.").replace("Treatment", "Drug")
                                     .replace("Genotype", "Geno.") for c in cols],
                  rotation=40, ha="right", fontsize=6)
    ax.set_yticks(range(len(rows)), row_labels, fontsize=6)
    ax.set_xticks(np.arange(len(cols)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=1)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    P.title(ax, title)
    return im


def figure5():
    """Pathway activity indices across the factorial design."""
    path = pd.read_csv(TAB / "pathway_contrasts.csv")
    ratio = pd.read_csv(TAB / "phospho_ratio_contrasts.csv")
    terms = ["Genotype", "Conditioning", "Treatment", "Genotype×Conditioning",
             "Genotype×Treatment", "Conditioning×Treatment", "Genotype×Conditioning×Treatment"]
    fig = plt.figure(figsize=(P.DOUBLE, 100 * P.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1], wspace=0.75)

    def colorbar(im, ax, label):
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.27,
                          aspect=30)
        cb.set_label(label, fontsize=6)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_linewidth(0.4)

    pathways = list(path.feature.unique())
    ax = fig.add_subplot(gs[0])
    im = heatmap(ax, path, pathways, terms, pathways, 1.5, "Conditioning dominates pathway activity")
    colorbar(im, ax, "Contrast estimate (pathway score, SD units)")
    P.panel_label(ax, "a", x=-0.62, y=1.02)
    ax = fig.add_subplot(gs[1])
    ratios = list(ratio.feature.unique())
    est = ratio.pivot(index="feature", columns="contrast", values="estimate").loc[ratios, terms]
    vmax = float(np.nanpercentile(np.abs(est.to_numpy()), 98))
    im2 = heatmap(ax, ratio, ratios, terms, ratios, round(vmax, 1), "Individual phospho/total ratios")
    colorbar(im2, ax, "Contrast estimate (log₂ ratio)")
    P.panel_label(ax, "b", x=-0.42, y=1.02)
    fig.text(0.99, 0.005, "* q < 0.05   ** q < 0.01 (Benjamini–Hochberg within each contrast)",
             ha="right", va="bottom", fontsize=5.8, fontfamily=P.MONO, color=P.SUBTLE_TEXT)
    P.save(fig, FIG, "fig5_pathways")


# ---------------------------------------------------------------------------
def figure6():
    """Network rewiring: residual co-variation within each genotype."""
    N = SUMMARY["network"]
    control = pd.read_csv(TAB / "network_corr_control.csv", index_col=0)
    trisomy = pd.read_csv(TAB / "network_corr_ts65dn.csv", index_col=0)
    null = pd.read_csv(TAB / "network_null.csv").null_statistic
    pairs = pd.read_csv(TAB / "network_pairs.csv")
    order = leaves_list(linkage(squareform(1 - control.to_numpy(), checks=False), "average"))
    c = control.to_numpy()[np.ix_(order, order)]
    t = trisomy.to_numpy()[np.ix_(order, order)]
    combined = np.where(np.tril(np.ones_like(c, bool), -1), c, t)
    np.fill_diagonal(combined, np.nan)

    fig = plt.figure(figsize=(P.DOUBLE, 72 * P.MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.95, 0.95], wspace=0.62)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(combined, cmap=P.DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
    P.no_grid(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.02, 0.02, "Control\n(lower)", transform=ax.transAxes, fontsize=6, color=P.CONTROL,
            fontweight="semibold")
    ax.text(0.98, 0.98, "Ts65Dn\n(upper)", transform=ax.transAxes, fontsize=6, color=P.TS65DN,
            fontweight="semibold", ha="right", va="top")
    ax.set_xlabel(f"{len(order)} proteins, clustered on control mice")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)
    P.title(ax, "Residual correlations (r)")
    P.panel_label(ax, "a", x=-0.08, y=1.02)

    ax = fig.add_subplot(gs[1])
    ax.scatter(pairs.r_control, pairs.r_ts65dn, s=1.5, color=P.MUTED, alpha=0.6, lw=0)
    top = pairs.nsmallest(10, "p")
    ax.scatter(top.r_control, top.r_ts65dn, s=9, color=P.VERMILION, lw=0,
               label=f"10 largest changes\n(min q = {pairs.q.min():.2f})")
    ax.plot([-1, 1], [-1, 1], color=P.SUBTLE_TEXT, lw=0.5, ls=(0, (3, 2)))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.grid(axis="x")
    ax.set_xlabel("r in control mice")
    ax.set_ylabel("r in Ts65Dn mice")
    ax.legend(loc="lower right", fontsize=5.6)
    P.title(ax, f"{len(pairs):,} protein pairs")
    P.panel_label(ax, "b", x=-0.3, y=1.02)

    ax = fig.add_subplot(gs[2])
    ax.hist(null, bins=40, color=P.MUTED, edgecolor="white", lw=0.3)
    ax.axvline(N["statistic"], color=P.VERMILION, lw=1.2)
    P.annotate_stat(ax, N["statistic"] + 0.003, ax.get_ylim()[1] * 0.97,
                    f"observed\np = {N['p_global']:.2f}", color=P.VERMILION, va="top", fontsize=5.8)
    ax.set_xlabel("RMS Fisher-z difference")
    ax.set_ylabel("Permutations (genotype shuffled)")
    P.title(ax, "No detectable rewiring")
    P.panel_label(ax, "c", x=-0.3, y=1.02)
    P.save(fig, FIG, "fig6_network")


# ---------------------------------------------------------------------------
def figure7():
    """Power of a follow-up experiment as a function of mice per cell."""
    power = pd.read_csv(TAB / "power_curves.csv")
    styles = {
        "Learning attenuation (observed)": (P.TS65DN, "-"),
        "Learning attenuation (50% of control response)": (P.TS65DN, (0, (3, 2))),
        "Protein 3-way interaction, 1 SD": (P.CONTROL, (0, (3, 2))),
        "Protein 3-way interaction, 2 SD": (P.CONTROL, "-"),
    }
    fig, ax = plt.subplots(figsize=(P.SINGLE, 62 * P.MM))
    for name, (color, ls) in styles.items():
        sub = power[power.scenario == name]
        ax.plot(sub.n_per_cell, sub.power, color=color, ls=ls, lw=1.2, marker="o", ms=2.5, label=name)
    ax.axhline(0.8, color=P.SUBTLE_TEXT, lw=0.6, ls=(0, (1, 1.5)))
    ax.axvspan(7, 10, color=P.RULE, zorder=0)
    ax.text(8.5, 0.03, "this\nstudy", ha="center", fontsize=6, color=P.SUBTLE_TEXT)
    ax.set_xlabel("Mice per experimental cell")
    ax.set_ylabel("Power (two-sided α = 0.05)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=5.6)
    P.title(ax, "Interactions need larger cohorts")
    P.save(fig, FIG, "fig7_power")


# ---------------------------------------------------------------------------
def figure_s1():
    """Missingness by experimental cell for proteins with missing values."""
    miss = pd.read_csv(TAB / "qc_missingness_by_cell.csv")
    miss = miss[miss.missing_rows >= 30].sort_values("missing_rows", ascending=False)
    cols = [f"missing_frac_{c}" for c in CELLS]
    fig, ax = plt.subplots(figsize=(P.SINGLE * 1.2, 55 * P.MM))
    im = ax.imshow(miss[cols].to_numpy(), cmap=P.SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    P.no_grid(ax)
    P.title(ax, "Missingness does not track experimental cell")
    ax.set_xticks(range(8), [P.CELL_LABEL[c] for c in CELLS], rotation=40, ha="right", fontsize=6)
    ax.set_yticks(range(len(miss)), [f"{display(p)} (q={q:.2g})" for p, q in zip(miss.protein, miss.q)],
                  fontsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("Fraction of measurements missing", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    for s in ax.spines.values():
        s.set_visible(False)
    P.save(fig, FIG, "figS1_missingness")


def figure_s2():
    """Memorisation test: original vs obfuscated data for each model (per CV repeat)."""
    models = [("logistic", "Logistic\n(control)"), ("tabpfn", "TabPFN v2"), ("kumo", "Kumo Tabular\nSmall")]
    fig, ax = plt.subplots(figsize=(P.SINGLE * 1.25, 58 * P.MM))
    for i, (model, label) in enumerate(models):
        path = ROOT / f"results/metrics/memorization_{model}.csv"
        if not path.exists():
            continue
        t = pd.read_csv(path)
        orig = t[t.version == "original"].balanced_accuracy
        obf = t[t.version != "original"].balanced_accuracy
        ax.scatter(np.full(len(orig), i - 0.14), orig, color=P.INK, s=12, lw=0,
                   label="Original" if i == 0 else None)
        ax.scatter(i + 0.14 + np.random.default_rng(i).uniform(-0.04, 0.04, len(obf)), obf,
                   color=P.VERMILION, marker="D", s=10, lw=0,
                   label="Obfuscated (3 keys)" if i == 0 else None)
        for x, vals in ((i - 0.14, orig), (i + 0.14, obf)):
            ax.hlines(vals.mean(), x - 0.08, x + 0.08, color=P.INK, lw=1.3)
        P.annotate_stat(ax, i, max(orig.max(), obf.max()) + 0.02,
                        f"{orig.mean():.3f} → {obf.mean():.3f}", ha="center", fontsize=5.8)
    ax.set_xticks(range(len(models)), [m[1] for m in models])
    ax.set_ylabel("Eight-group balanced accuracy\n(one dot per CV repeat)")
    ax.set_ylim(0.5, 0.95)
    ax.legend(loc="lower right")
    P.title(ax, "Hiding the dataset’s identity leaves mean accuracy unchanged")
    P.save(fig, FIG, "figS2_obfuscation")



# ---------------------------------------------------------------------------
def figure_overview():
    """Schematic overview: experiment -> problem -> questions -> answers (numbers from results)."""
    L, R, D, N = SUMMARY["learning"], SUMMARY["rescue"], SUMMARY["dosage"], SUMMARY["network"]
    fig = plt.figure(figsize=(P.DOUBLE, 92 * P.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    cols = [(1, 22, "1  The experiment"), (25, 22, "2  The problem"),
            (50, 23, "3  Our questions"), (76, 23, "4  What we found")]
    for x, w, head in cols:
        ax.text(x, 52.5, head, fontfamily=P.SERIF, fontweight="semibold", fontsize=9, va="top")
        ax.plot([x, x + w], [48.8, 48.8], color=P.RULE, lw=1.0)
    for x in (23.4, 48.4, 74.4):
        ax.annotate("", (x + 1.2, 51.3), (x - 0.4, 51.3),
                    arrowprops={"arrowstyle": "-|>", "color": P.SUBTLE_TEXT, "lw": 0.8,
                                "mutation_scale": 8})
    experiment = [("72", "male mice"), ("2", "genotypes: control vs Ts65Dn\n(Down syndrome model)"),
                  ("2", "conditioning: learns (CS)\nvs cannot learn (SC)"),
                  ("2", "treatments: memantine\nvs saline"),
                  ("77", "cortical proteins, each\nmeasured 15 times per mouse")]
    y = 45.5
    for big, text in experiment:
        ax.text(1, y, big, fontsize=12, fontweight="semibold", color=P.COBALT, va="top")
        ax.text(6.3, y - 0.3, text, fontsize=6.4, va="top", linespacing=1.25)
        y -= 8.6
    problem = [
        ("Used as a benchmark,\nnot as an experiment",
         "Papers ask “can a model tell the 8 groups apart?” instead of asking what the biology is."),
        ("1,080 rows are not\n1,080 mice",
         ("Random row splits leak each mouse into training and test: accuracy "
         "0.99 → 0.60 when mice are held out.")),
        ("The real questions\nare unanswered",
         "What does trisomy do to the proteome, to the learning response, and does the drug fix it?"),
    ]
    y = 45.5
    for head, body in problem:
        ax.text(25, y, head, fontsize=6.8, fontweight="semibold", va="top", linespacing=1.2,
                color=P.VERMILION if "rows" in head else P.INK)
        ax.text(25, y - 5.6, textwrap.fill(body, 40), fontsize=6.0, va="top", color=P.SUBTLE_TEXT,
                linespacing=1.25)
        y -= 14.5
    qa = [
        ("Q1", "Does an extra gene copy\nmean 50% more protein?",
         f"Only {100 * D['set_fraction_of_expected']:.0f}% of that; non-triplicated\ngenes unchanged"),
        ("Q2", "Does trisomy change the\nbrain’s response to learning?",
         (f"Weaker ({L['relative']['Ts65Dn saline']:.2f}×); memantine\nrestores it "
         f"({L['relative']['Ts65Dn memantine']:.2f}×)")),
        ("Q3", "Does memantine repair the\ntrisomic proteome?",
         f"Partly offsets it ({100 * R['rescue_fraction']:.0f}%), but\nmoves controls too"),
        ("Q4", "Does trisomy rewire how\nproteins co-vary?",
         f"No evidence\n(p = {N['p_global']:.2f})"),
    ]
    y = 45.5
    for tag, question, answer in qa:
        ax.text(50, y, tag, fontfamily=P.MONO, fontsize=7, color=P.COBALT, va="top")
        ax.text(54, y, question, fontsize=6.4, va="top", linespacing=1.25)
        ax.text(76, y, answer, fontsize=6.4, va="top", linespacing=1.25,
                color=P.INK, fontweight="semibold")
        y -= 11
    ax.text(99, 1.2, "SCHEMATIC · numbers from results/mechanisms/summary.json", ha="right",
            fontfamily=P.MONO, fontsize=5.2, color=P.SUBTLE_TEXT)
    ax.grid(False)
    P.save(fig, FIG, "fig0_overview")


# ---------------------------------------------------------------------------
# Single-message figures for slides (larger type, one panel each)
SLIDE = FIG / "slides"


def slide_axes(width=7.2, height=4.2):
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def slide_style():
    import matplotlib as mpl

    mpl.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "legend.fontsize": 10.5, "axes.linewidth": 0.8})


def slide_dosage():
    sets = SUMMARY["dosage"]["sets"]
    fig, ax = slide_axes()
    names = [("triplicated", "Triplicated\ngenes", P.TS65DN), ("hsa21_disomic", "Chromosome-21\ngenes, not\ntriplicated", P.INK),
             ("background", "All other\nproteins", P.MUTED)]
    for i, (k, label, color) in enumerate(names):
        e = sets[k]
        ax.errorbar(i, e["estimate"], yerr=[[e["estimate"] - e["lower"]], [e["upper"] - e["estimate"]]],
                    fmt="o", ms=10, color=color, elinewidth=2.2, capsize=0)
        P.annotate_stat(ax, i + 0.12, e["estimate"], f"{e['estimate']:+.2f}", va="center", fontsize=11)
    ax.axhline(EXPECTED_DOSAGE_LOG2FC, color=P.TS65DN, lw=1.2, ls=(0, (2, 2)))
    ax.text(2.45, EXPECTED_DOSAGE_LOG2FC + 0.02, "expected if 3 copies = 1.5× protein", ha="right",
            va="bottom", color=P.TS65DN, fontsize=11)
    P.zero_line(ax)
    ax.set_xticks(range(3), [n[1] for n in names])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.15, 0.7)
    ax.set_ylabel("Change in Ts65Dn vs control\n(log₂ fold change, 95% CI)")
    P.save(fig, SLIDE, "dosage")


def slide_learning():
    L = SUMMARY["learning"]
    fig, ax = slide_axes()
    labels = ["Control\nsaline", "Control\nmemantine", "Ts65Dn\nsaline", "Ts65Dn\nmemantine"]
    for i, arm in enumerate(ARMS):
        g, t = ARM_CODE[arm]
        est = L["relative"][arm]
        lo, hi = L["relative_ci"][arm]
        color = P.GENOTYPE_COLOR[g]
        if arm == "control saline":
            ax.scatter(i, est, s=110, color=color, marker="D", zorder=3)
        else:
            ax.errorbar(i, est, yerr=[[est - lo], [hi - est]], fmt="o", ms=10, color=color,
                        mfc="white" if t == "s" else color, elinewidth=2.2, capsize=0, mew=2)
    ax.axhline(1, color=P.SUBTLE_TEXT, lw=1, ls=(0, (3, 2)))
    for (a, b), y, pv in (((0, 2), 1.9, L["attenuation_p"]), ((2, 3), 1.62, L["restoration_p"])):
        ax.plot([a, a, b, b], [y - 0.05, y, y, y - 0.05], color=P.INK, lw=1)
        P.annotate_stat(ax, (a + b) / 2, y + 0.03, f"p = {pv:.3f}", ha="center", fontsize=11)
    ax.set_xticks(range(4), labels)
    ax.set_ylim(0, 2.1)
    ax.set_ylabel("Size of the learning response\n(relative to control saline, 95% CI)")
    P.save(fig, SLIDE, "learning")


def slide_rescue():
    R = SUMMARY["rescue"]
    fig, ax = slide_axes()
    labels = ["Control\nsaline", "Control\nmemantine", "Ts65Dn\nsaline", "Ts65Dn\nmemantine"]
    for i, arm in enumerate(ARMS):
        g, t = ARM_CODE[arm]
        est = R["arm_means"][arm]
        lo, hi = R["arm_means_ci"][arm]
        color = P.GENOTYPE_COLOR[g]
        ax.errorbar(i, est, yerr=[[est - lo], [hi - est]], fmt="o", ms=10, color=color,
                    mfc="white" if t == "s" else color, elinewidth=2.2, capsize=0, mew=2)
    ax.annotate("", (2.9, R["arm_means"]["Ts65Dn memantine"]), (2.1, R["arm_means"]["Ts65Dn saline"]),
                arrowprops={"arrowstyle": "-|>", "color": P.INK, "lw": 1.2, "mutation_scale": 12,
                            "connectionstyle": "arc3,rad=-0.2"})
    ax.annotate("", (0.9, R["arm_means"]["control memantine"]), (0.1, R["arm_means"]["control saline"]),
                arrowprops={"arrowstyle": "-|>", "color": P.COBALT, "lw": 1.2, "mutation_scale": 12,
                            "connectionstyle": "arc3,rad=0.2"})
    ax.axhline(R["arm_means"]["control saline"], color=P.COBALT, lw=1, ls=(0, (3, 2)))
    ax.set_xticks(range(4), labels)
    ax.set_ylabel("Position on the trisomy axis\n(0 ≈ average mouse; 95% CI)")
    P.save(fig, SLIDE, "rescue")


def slide_network():
    N = SUMMARY["network"]
    null = pd.read_csv(TAB / "network_null.csv").null_statistic
    fig, ax = slide_axes()
    ax.hist(null, bins=40, color=P.MUTED, edgecolor="white", lw=0.4)
    ax.axvline(N["statistic"], color=P.VERMILION, lw=2)
    ax.text(N["statistic"] + 0.003, ax.get_ylim()[1] * 0.95, f"observed\np = {N['p_global']:.2f}",
            color=P.VERMILION, va="top", fontsize=11, fontfamily=P.MONO)
    ax.set_xlabel("How different the two genotypes’ correlation patterns are")
    ax.set_ylabel("Count when genotype\nlabels are shuffled")
    P.save(fig, SLIDE, "network")


def slides_all():
    import matplotlib as mpl

    with mpl.rc_context():
        slide_style()
        for f in (slide_dosage, slide_learning, slide_rescue, slide_network):
            f()


def main():
    P.apply_style()
    for f in (figure_overview, figure1, figure2, figure3, figure4, figure5, figure6, figure7,
              figure_s1, figure_s2, slides_all):
        f()
        print("drew", f.__name__)


if __name__ == "__main__":
    main()
