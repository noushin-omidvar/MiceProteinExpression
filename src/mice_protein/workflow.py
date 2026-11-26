"""Workflow diagrams: one self-explanatory, left-to-right schematic per research question.

Design rules (so a first-time reader can follow without the paper):

* The **question** heads the diagram; the **result** ends it, drawn as a small chart of the
  real numbers (never a generic tick, so a negative or null answer is visible as such).
* Each step has a plain **verb** ("Average", "Compare"), a short title, a labelled drawing,
  one plain-language sentence, and then the defining **equation** with a "where" line that
  defines every symbol.
* Colour keeps one meaning per diagram: cobalt = control / baseline, vermilion = Ts65Dn /
  foundation model; the result step is marked by its heading, not by recolouring data.
* Steps are labelled as schematics; the result panel shows data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from . import plotting as P

GLYPHS: dict[str, callable] = {}
SMALL = 5.4


def glyph(name):
    def register(fn):
        GLYPHS[name] = fn
        return fn
    return register


@dataclass
class Stage:
    verb: str  # plain action word shown in the step heading, e.g. "AVERAGE"
    title: str
    glyph: str
    text: str  # one plain-language sentence
    equation: str = ""  # mathtext, optional
    where: str = ""  # definitions of the symbols in the equation
    args: dict = field(default_factory=dict)  # glyph parameters (e.g. real results)
    result: bool = False


def _rng():
    return np.random.default_rng(7)


def _label(ax, x, y, text, color=P.SUBTLE_TEXT, **kw):
    kw.setdefault("fontsize", SMALL)
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    ax.text(x, y, text, color=color, **kw)


# ---------------------------------------------------------------------------
# Schematic glyphs (labelled)
# ---------------------------------------------------------------------------

@glyph("rows_to_mean")
def _rows_to_mean(ax, **_):
    rng = _rng()
    for r in range(15):
        ax.add_patch(Rectangle((0.04, 0.9 - r * 0.05), 0.3, 0.035, color=P.MUTED,
                               alpha=0.4 + 0.5 * rng.random(), lw=0))
    _label(ax, 0.19, 0.1, "15 measurements")
    ax.add_patch(FancyArrowPatch((0.4, 0.55), (0.56, 0.55), arrowstyle="-|>", mutation_scale=7,
                                 color=P.INK, lw=0.8))
    ax.add_patch(Rectangle((0.62, 0.51), 0.34, 0.08, color=P.INK, lw=0))
    _label(ax, 0.79, 0.36, "1 value\nper mouse")


@glyph("row_leak")
def _row_leak(ax, **_):
    """One mouse's 15 rows scattered across training and test sets."""
    rng = _rng()
    side = rng.permutation([0] * 10 + [1] * 5)
    for r, s in enumerate(side):
        ax.add_patch(Rectangle((0.04 + 0.52 * s, 0.9 - r * 0.05), 0.36, 0.035,
                               color=P.VERMILION, alpha=0.85, lw=0))
    _label(ax, 0.22, 0.1, "train")
    _label(ax, 0.74, 0.1, "test")
    _label(ax, 0.5, 1.03, "rows of one mouse", color=P.VERMILION)


@glyph("copy_number")
def _copy_number(ax, **_):
    for k, (name, n, c) in enumerate((("control", 2, P.COBALT), ("Ts65Dn", 3, P.VERMILION))):
        x0 = 0.1 + k * 0.48
        for i in range(n):
            ax.add_patch(Rectangle((x0 + i * 0.1, 0.42), 0.06, 0.42, color=c, lw=0))
        _label(ax, x0 + 0.1, 0.27, name)
        _label(ax, x0 + 0.1, 0.12, f"{n} gene copies", color=c)


@glyph("two_groups")
def _two_groups(ax, left="control", right="Ts65Dn", **_):
    x = np.linspace(0, 1, 120)
    for mu, c, lab in ((0.36, P.COBALT, left), (0.62, P.VERMILION, right)):
        ax.fill_between(x, 0.18, 0.18 + 0.55 * np.exp(-((x - mu) / 0.1) ** 2), color=c,
                        alpha=0.3, lw=0)
        ax.plot([mu, mu], [0.18, 0.73], color=c, lw=1)
        _label(ax, mu, 0.06, lab, color=c)
    ax.add_patch(FancyArrowPatch((0.36, 0.84), (0.62, 0.84), arrowstyle="<|-|>",
                                 mutation_scale=6, color=P.INK, lw=0.8))
    _label(ax, 0.49, 0.95, "difference", color=P.INK)
    ax.plot([0.05, 0.95], [0.18, 0.18], color=P.INK, lw=0.6)


@glyph("protein_matrix")
def _protein_matrix(ax, **_):
    ax.imshow(_rng().normal(size=(9, 12)), extent=(0.12, 0.92, 0.18, 0.92), cmap=P.DIVERGING,
              vmin=-2, vmax=2, aspect="auto")
    _label(ax, 0.04, 0.55, "mice", rotation=90)
    _label(ax, 0.52, 0.06, "proteins")


@glyph("signature")
def _signature(ax, from_label="SC", to_label="CS", **_):
    rng = _rng()
    a = rng.normal([0.28, 0.3], 0.07, (10, 2))
    b = rng.normal([0.72, 0.7], 0.07, (10, 2))
    ax.scatter(*a.T, s=7, color=P.MUTED, lw=0)
    ax.scatter(*b.T, s=7, color=P.COBALT, lw=0)
    ax.add_patch(FancyArrowPatch((0.28, 0.3), (0.7, 0.68), arrowstyle="-|>", mutation_scale=8,
                                 color=P.INK, lw=1.3))
    _label(ax, 0.16, 0.12, from_label)
    _label(ax, 0.84, 0.9, to_label, color=P.COBALT)
    _label(ax, 0.66, 0.4, "signature", color=P.INK, rotation=40)


@glyph("leave_one_out")
def _leave_one_out(ax, **_):
    ax.plot([0.06, 0.94], [0.18, 0.82], color=P.INK, lw=1.1)
    rng = _rng()
    for _i in range(6):
        x, y = rng.uniform(0.18, 0.85), rng.uniform(0.15, 0.75)
        t = ((x - 0.06) * 0.88 + (y - 0.18) * 0.64) / (0.88**2 + 0.64**2)
        px, py = 0.06 + t * 0.88, 0.18 + t * 0.64
        ax.plot([x, px], [y, py], color=P.RULE, lw=0.6)
        ax.scatter([x], [y], s=7, color=P.COBALT, lw=0, zorder=3)
        ax.scatter([px], [py], s=6, color=P.INK, lw=0, zorder=3)
    ax.scatter([0.22], [0.72], s=16, facecolors="white", edgecolors=P.INK, lw=1, zorder=4)
    _label(ax, 0.26, 0.92, "scored mouse is\nleft out of the line", color=P.INK)
    _label(ax, 0.8, 0.06, "score = position\non the line")


@glyph("four_arms")
def _four_arms(ax, **_):
    for i, name in enumerate(["Ctrl\nsal", "Ctrl\nmem", "Ts\nsal", "Ts\nmem"]):
        c = P.COBALT if i < 2 else P.VERMILION
        x = 0.2 + i * 0.22
        ax.scatter([x] * 4, 0.64 + 0.05 * np.arange(4), s=6, color=c, marker="o", lw=0)
        ax.scatter([x] * 4, 0.32 + 0.05 * np.arange(4), s=7, color=c, marker="^", lw=0)
        _label(ax, x, 0.08, name, color=c, linespacing=1.0)
    _label(ax, 0.02, 0.72, "CS", ha="left")
    _label(ax, 0.02, 0.4, "SC", ha="left")


@glyph("shuffle")
def _shuffle(ax, n="5,000", **_):
    left = ["Ctrl", "Ts", "Ctrl", "Ts", "Ts"]
    right = ["Ts", "Ctrl", "Ts", "Ts", "Ctrl"]
    for i, (a, b) in enumerate(zip(left, right, strict=True)):
        y = 0.88 - i * 0.16
        _label(ax, 0.14, y, a, color=P.COBALT if a == "Ctrl" else P.VERMILION)
        _label(ax, 0.86, y, b, color=P.COBALT if b == "Ctrl" else P.VERMILION)
        ax.add_patch(FancyArrowPatch((0.25, y), (0.75, 0.88 - ((i * 2) % 5) * 0.16),
                                     arrowstyle="-", color=P.RULE, lw=0.6))
    _label(ax, 0.5, 0.0, f"labels shuffled {n}×")


@glyph("network_pairs")
def _network_pairs(ax, **_):
    rng = _rng()
    for k, (c, lab) in enumerate(((P.COBALT, "control"), (P.VERMILION, "Ts65Dn"))):
        pts = rng.uniform(0.1, 0.9, (6, 2)) * [0.42, 0.75] + [0.03 + k * 0.5, 0.18]
        for i in range(6):
            for j in range(i + 1, 6):
                if rng.random() < 0.4:
                    ax.plot(*pts[[i, j]].T, color=c, lw=0.6, alpha=0.6)
        ax.scatter(*pts.T, s=9, color=c, zorder=3, lw=0)
        _label(ax, 0.24 + k * 0.5, 0.05, lab, color=c)


@glyph("residuals")
def _residuals(ax, **_):
    rng = _rng()
    for g in range(8):
        mu = 0.35 + 0.35 * rng.random()
        x = 0.1 + g * 0.11
        ax.plot([x - 0.035, x + 0.035], [mu, mu], color=P.INK, lw=1)
        ax.scatter(x + rng.uniform(-0.02, 0.02, 4), mu + rng.normal(0, 0.06, 4), s=5,
                   color=P.MUTED, lw=0)
    _label(ax, 0.5, 0.08, "subtract each group’s mean")


@glyph("cv_folds")
def _cv_folds(ax, **_):
    for r in range(5):
        for f in range(5):
            ax.add_patch(Rectangle((0.1 + f * 0.16, 0.8 - r * 0.14), 0.14, 0.1,
                                   color=P.INK if f == r else P.MUTED,
                                   alpha=0.85 if f == r else 0.4, lw=0))
    _label(ax, 0.5, 0.06, "dark = held-out mice")


@glyph("model_zoo")
def _model_zoo(ax, left="6 conventional\n(tuned)", right="2 foundation\n(as-is)", **_):
    for i, (c, lab, n) in enumerate(((P.COBALT, left, 3), (P.VERMILION, right, 2))):
        x0 = 0.08 + i * 0.48
        for k in range(n):
            ax.add_patch(Rectangle((x0 + k * 0.04, 0.45 + k * 0.08), 0.3, 0.2, fc="white",
                                   ec=c, lw=1))
        _label(ax, x0 + 0.18, 0.18, lab, color=c, linespacing=1.1)


@glyph("paired_diff")
def _paired_diff(ax, **_):
    rng = _rng()
    a = 0.45 + rng.normal(0, 0.08, 8)
    b = a + 0.1 + rng.normal(0, 0.04, 8)
    for i in range(8):
        ax.plot([0.25, 0.75], [a[i], b[i]], color=P.RULE, lw=0.6)
    ax.scatter([0.25] * 8, a, s=7, color=P.COBALT, lw=0, zorder=3)
    ax.scatter([0.75] * 8, b, s=7, color=P.VERMILION, lw=0, zorder=3, marker="D")
    _label(ax, 0.25, 0.08, "baseline", color=P.COBALT)
    _label(ax, 0.75, 0.08, "foundation", color=P.VERMILION)
    _label(ax, 0.5, 0.96, "same fold → paired")


@glyph("reliability")
def _reliability(ax, **_):
    ax.plot([0.14, 0.92], [0.14, 0.92], color=P.SUBTLE_TEXT, lw=0.6, ls=(0, (2, 1.5)))
    x = np.linspace(0.2, 0.9, 6)
    ax.plot(x, x - 0.15 * np.sin(x * 3), color=P.COBALT, marker="o", ms=2, lw=0.9)
    ax.plot([0.14, 0.14, 0.92], [0.92, 0.14, 0.14], color=P.INK, lw=0.6)
    _label(ax, 0.53, 0.03, "confidence")
    _label(ax, 0.04, 0.53, "accuracy", rotation=90)
    _label(ax, 0.6, 0.82, "perfect", rotation=40)


@glyph("subsample")
def _subsample(ax, **_):
    for g in range(8):
        for m in range(7):
            chosen = m < 2
            ax.add_patch(Circle((0.14 + m * 0.11, 0.9 - g * 0.095), 0.027,
                                color=P.INK if chosen else P.MUTED, alpha=1 if chosen else 0.45,
                                lw=0))
    _label(ax, 0.5, 0.06, "dark = training mice")


@glyph("curve")
def _curve(ax, **_):
    k = np.linspace(0.18, 0.9, 6)
    ax.plot(k, 0.3 + 0.48 * (1 - np.exp(-3 * k)), color=P.VERMILION, marker="D", ms=2, lw=1)
    ax.plot(k, 0.2 + 0.45 * (1 - np.exp(-2 * k)), color=P.COBALT, marker="o", ms=2, lw=1)
    ax.plot([0.12, 0.12, 0.94], [0.94, 0.14, 0.14], color=P.INK, lw=0.6)
    _label(ax, 0.53, 0.03, "training mice per group")
    _label(ax, 0.04, 0.55, "accuracy", rotation=90)


@glyph("three_controls")
def _three_controls(ax, **_):
    items = [("shuffle labels", "→ chance"), ("let rows leak", "→ inflated"),
             ("hide identity", "→ unchanged")]
    for i, (a, b) in enumerate(items):
        y = 0.8 - i * 0.3
        ax.add_patch(Rectangle((0.02, y - 0.1), 0.96, 0.21, fc="white", ec=P.RULE, lw=0.8))
        _label(ax, 0.06, y, a, color=P.INK, ha="left")
        _label(ax, 0.94, y, b, ha="right")


@glyph("obfuscate")
def _obfuscate(ax, **_):
    heads, tokens = ["APP", "SOD1", "pERK"], ["f07", "f42", "f13"]
    for i in range(3):
        _label(ax, 0.17 + i * 0.33, 0.84, heads[i], color=P.INK)
        _label(ax, 0.17 + ((i + 1) % 3) * 0.33, 0.34, tokens[i], fontfamily=P.MONO)
        ax.add_patch(FancyArrowPatch((0.17 + i * 0.33, 0.75), (0.17 + ((i + 1) % 3) * 0.33, 0.43),
                                     arrowstyle="-|>", mutation_scale=5, color=P.RULE, lw=0.6))
    _label(ax, 0.5, 0.08, "rename · reorder · rescale")


# ---------------------------------------------------------------------------
# Result glyphs (real numbers)
# ---------------------------------------------------------------------------

@glyph("result_bars")
def _result_bars(ax, values, labels, colors, ref=None, ref_label="", fmt="{:.2f}", ymax=None,
                 **_):
    n = len(values)
    top = ymax or max(max(values), ref or 0) * 1.3
    for i, (v, lab, c) in enumerate(zip(values, labels, colors, strict=True)):
        x = 0.15 + i * (0.7 / (n - 1)) if n > 1 else 0.5
        h = 0.6 * max(v, 0) / top
        ax.add_patch(Rectangle((x - 0.09, 0.22), 0.18, h, color=c, lw=0))
        inside = h > 0.14 and n <= 3
        _label(ax, x, 0.22 + h - 0.06 if inside else 0.22 + h + 0.06, fmt.format(v),
               color="white" if inside else P.INK, fontfamily=P.MONO)
        _label(ax, x, 0.08, lab, color=c, linespacing=1.0)
    if ref is not None:
        y = 0.22 + 0.6 * ref / top
        ax.plot([0.02, 0.98], [y, y], color=P.SUBTLE_TEXT, lw=0.7, ls=(0, (2, 1.5)), zorder=3)
        _label(ax, 0.02, y + 0.075, ref_label, ha="left")
    ax.plot([0.02, 0.98], [0.22, 0.22], color=P.INK, lw=0.6)


@glyph("result_null")
def _result_null(ax, observed, null, p, **_):
    null = np.asarray(null)
    lo, hi = np.percentile(null, [0.5, 99.5])
    lo, hi = min(lo, observed), max(hi, observed)
    counts, edges = np.histogram(null, bins=18, range=(lo, hi))
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:], strict=True):
        x0 = 0.06 + 0.88 * (e0 - lo) / (hi - lo)
        ax.add_patch(Rectangle((x0, 0.2), 0.88 * (e1 - e0) / (hi - lo) * 0.9,
                               0.55 * c / counts.max(), color=P.MUTED, lw=0))
    xo = 0.06 + 0.88 * (observed - lo) / (hi - lo)
    ax.plot([xo, xo], [0.18, 0.82], color=P.INK, lw=1.3)
    _label(ax, xo, 0.9, f"observed · p = {p:.2f}", color=P.INK, fontfamily=P.MONO)
    _label(ax, 0.5, 0.07, "differences expected by chance")


@glyph("result_checks")
def _result_checks(ax, items, **_):
    for i, (name, ok) in enumerate(items):
        y = 0.82 - i * 0.3
        _label(ax, 0.06, y + 0.05, name, color=P.INK, ha="left")
        _label(ax, 0.06, y - 0.07, "as expected" if ok else "not as expected", ha="left",
               fontfamily=P.MONO, color=P.COBALT if ok else P.VERMILION)


# ---------------------------------------------------------------------------
# Diagram
# ---------------------------------------------------------------------------

def draw_workflow(stages: list[Stage], out_dir: Path, name: str, question: str = "") -> None:
    """Draw question -> labelled steps -> result chart; save SVG/PDF/PNG."""
    n = len(stages)
    fig = plt.figure(figsize=(P.DOUBLE, 82 * P.MM))
    base = fig.add_axes([0, 0, 1, 1])
    base.set_xlim(0, 1)
    base.set_ylim(0, 1)
    base.axis("off")
    col = 1 / n
    fs = 6.3
    top = 0.97
    if question:
        base.text(0.012, 0.975, question, ha="left", va="top", fontsize=9.6,
                  fontfamily=P.SERIF, fontweight="semibold", color=P.INK)
        base.plot([0.012, 0.988], [0.9, 0.9], color=P.RULE, lw=0.8)
        top = 0.865
    for i, st in enumerate(stages):
        cx = col * (i + 0.5)
        base.text(cx, top, "RESULT" if st.result else f"STEP {i + 1} · {st.verb}",
                  ha="center", va="top", fontsize=fs * 0.9, fontfamily=P.MONO,
                  color=P.INK if st.result else P.COBALT)
        base.text(cx, top - 0.05, st.title, ha="center", va="top", fontsize=fs * 1.22,
                  fontfamily=P.SERIF, fontweight="semibold", color=P.INK)
        g_bottom = top - 0.43
        g_ax = fig.add_axes([cx - col * 0.4, g_bottom, col * 0.8, 0.3])
        g_ax.set_xlim(0, 1)
        g_ax.set_ylim(0, 1)
        g_ax.axis("off")
        if st.result:
            g_ax.add_patch(Rectangle((-0.06, -0.06), 1.12, 1.16, fc="#F4F6F8", ec="none",
                                     zorder=-1, clip_on=False))
        GLYPHS[st.glyph](g_ax, **st.args)
        base.text(cx, g_bottom - 0.035, st.text, ha="center", va="top", fontsize=fs * 0.95,
                  color=P.INK, linespacing=1.3,
                  fontweight="semibold" if st.result else "normal")
        if st.equation:
            base.text(cx, 0.17, st.equation, ha="center", va="top", fontsize=fs, color=P.INK)
        if st.where:
            base.text(cx, 0.09, st.where, ha="center", va="top", fontsize=fs * 0.8,
                      color=P.SUBTLE_TEXT, linespacing=1.2)
        if i < n - 1:
            ya = g_bottom + 0.15
            base.add_patch(FancyArrowPatch((col * (i + 0.92), ya), (col * (i + 1.08), ya),
                                           arrowstyle="-|>", mutation_scale=8, color=P.COBALT,
                                           lw=1.0))
    base.text(0.995, 0.004, f"Steps 1–{n - 1}: schematic · Result: data",
              ha="right", va="bottom", fontfamily=P.MONO, fontsize=4.6, color=P.SUBTLE_TEXT)
    P.save(fig, out_dir, name)
