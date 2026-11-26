"""Drawing helpers for explainer graphics (schematics for readers new to the topic).

Shared by both studies so that every concept diagram, story map and evidence summary looks the
same. All outputs are schematics and are labelled as such.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from . import plotting as P


def canvas(width_mm: float = 183, height_mm: float = 80):
    fig = plt.figure(figsize=(width_mm * P.MM, height_mm * P.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", ec=P.RULE, fc="white", color=P.INK, size=6.6, weight="normal",
        radius=0.015, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={radius}",
                                fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
                color=color, fontweight=weight, linespacing=1.2)


def arrow(ax, x0, y0, x1, y1, color=P.SUBTLE_TEXT, lw=1.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9,
                                 color=color, lw=lw))


def heading(ax, x, y, text, color=P.INK, size=8.4):
    ax.text(x, y, text, fontfamily=P.SERIF, fontweight="semibold", fontsize=size, color=color,
            va="top")


def label(ax, x, y, text, color=P.SUBTLE_TEXT, size=6.2, ha="center", va="center", **kw):
    ax.text(x, y, text, fontsize=size, color=color, ha=ha, va=va, linespacing=1.2, **kw)


def chromosome(ax, x, y, h, color, w=0.018, band=None):
    """A stylised chromosome: two rounded arms joined at a centromere."""
    for y0, hh in ((y, h * 0.46), (y + h * 0.54, h * 0.46)):
        ax.add_patch(FancyBboxPatch((x - w / 2, y0), w, hh,
                                    boxstyle=f"round,pad=0,rounding_size={w / 2}",
                                    fc=color, ec="none"))
    ax.add_patch(Rectangle((x - w * 0.28, y + h * 0.45), w * 0.56, h * 0.1, fc=color, ec="none"))
    if band is not None:
        ax.add_patch(Rectangle((x - w / 2, y + band[0] * h), w, (band[1] - band[0]) * h,
                               fc=P.VERMILION, ec="none"))


def dot(ax, x, y, r, color, **kw):
    """A true circle of radius ``r`` (in x units) on a non-square canvas."""
    from matplotlib.patches import Ellipse

    w, h = ax.figure.get_size_inches()
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r * w / h, fc=color, ec=kw.pop("ec", "none"), **kw))


def icon_circle(ax, x, y, r, text, color):
    dot(ax, x, y, r, color)
    ax.text(x, y, text, ha="center", va="center", fontsize=9, color="white",
            fontweight="semibold", fontfamily=P.SERIF)


def schematic_tag(ax, text="SCHEMATIC"):
    ax.text(0.99, 0.01, text, ha="right", va="bottom", fontfamily=P.MONO, fontsize=5,
            color=P.SUBTLE_TEXT)


def story_map(out: Path, name: str, steps: list[tuple[str, str]], title: str) -> None:
    """Horizontal outline of a talk or report: numbered stops with one-line descriptions."""
    fig, ax = canvas(183, 58)
    heading(ax, 0.015, 0.95, title)
    n = len(steps)
    xs = [0.07 + i * (0.86 / (n - 1)) for i in range(n)]
    ax.plot([xs[0], xs[-1]], [0.55, 0.55], color=P.RULE, lw=2.2, zorder=0)
    for i, (x, (name_, desc)) in enumerate(zip(xs, steps, strict=True)):
        icon_circle(ax, x, 0.55, 0.045, str(i + 1), P.COBALT)
        label(ax, x, 0.36, name_, color=P.INK, size=7.6, fontweight="semibold")
        label(ax, x, 0.2, desc, size=6.6)
    schematic_tag(ax, "OUTLINE")
    P.save(fig, out, name)


def evidence_summary(out: Path, name: str, rows: list[tuple[str, str, int, str]],
                     title: str) -> None:
    """Answers with a three-level evidence meter (1 = weak, 2 = moderate, 3 = strong)."""
    fig, ax = canvas(183, 26 + 15 * len(rows))
    heading(ax, 0.015, 0.97, title)
    top = 0.84
    step = (top - 0.06) / len(rows)
    label(ax, 0.83, top + 0.02, "strength of evidence", size=5.8, ha="center")
    for i, (question, answer, level, note) in enumerate(rows):
        y = top - (i + 0.5) * step
        label(ax, 0.015, y, question, color=P.COBALT, size=6.8, ha="left", fontweight="semibold")
        label(ax, 0.3, y + 0.025, answer, color=P.INK, size=6.8, ha="left")
        label(ax, 0.3, y - 0.04, note, size=5.8, ha="left")
        for k in range(3):
            ax.add_patch(Rectangle((0.75 + k * 0.055, y - 0.02), 0.048, 0.04,
                                   fc=P.COBALT if k < level else P.RULE, ec="none"))
        label(ax, 0.92, y, ["", "weak", "moderate", "strong"][level], size=5.8, ha="left")
        if i < len(rows) - 1:
            ax.plot([0.015, 0.985], [y - step / 2, y - step / 2], color=P.RULE, lw=0.6)
    schematic_tag(ax, "SUMMARY")
    P.save(fig, out, name)
