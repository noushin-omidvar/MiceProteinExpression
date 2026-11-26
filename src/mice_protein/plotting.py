"""House visual identity for every figure (paper, README, notebooks, slides).

Style: minimalist editorial scientific figures (Nature-style anatomy with modern,
story-first titling). The full guide, with rationale, is ``docs/figure_style.md``.

Color roles (used by *meaning*, never cycled)
---------------------------------------------
``COBALT``   #2457D6  primary information; control mice
``VERMILION``#E33D3F  meaningful highlight; Ts65Dn mice, the effect being shown
``MUTED``    #8EA8EC  secondary data; background proteins, reference series
``INK``      #161B1F  text, axes, means and intervals
``RULE``     #D9DDE0  gridlines, borders, separators only (never data)

Identity is never carried by color alone: treatment is marker fill (open =
saline, filled = memantine) and conditioning is marker shape (circle = CS,
triangle = SC).

Typography
----------
IBM Plex Sans for labels and body, Fraunces for editorial headings (figure and
panel titles), IBM Plex Mono for technical annotations (statistics, n, p).
Fonts ship in ``assets/fonts`` (SIL OFL 1.1). IBM Plex is distributed as IBM's
unmodified WOFF files; matplotlib's FreeType cannot read WOFF, so they are
decompressed to an untracked local cache on first use (no redistribution of a
modified font).

Output
------
Every figure is written as SVG (editable text), PDF (embedded TrueType), and
300-dpi PNG, on a white (#FFFFFF) background with a small margin so nothing is
cropped.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
FONT_DIR = ROOT / "assets/fonts"
FONT_CACHE = ROOT / ".cache/fonts"

MM = 1 / 25.4
SINGLE = 89 * MM
DOUBLE = 183 * MM

INK = "#161B1F"
COBALT = "#2457D6"
VERMILION = "#E33D3F"
MUTED = "#8EA8EC"
RULE = "#D9DDE0"
SUBTLE_TEXT = "#5B6670"  # secondary text (captions inside plots); passes 4.5:1 on white
BACKGROUND = "#FFFFFF"

# Semantic aliases used by figure code.
CONTROL = COBALT
TS65DN = VERMILION
GENOTYPE_COLOR = {"c": CONTROL, "t": TS65DN, "Control": CONTROL, "Ts65Dn": TS65DN}
BEHAVIOR_MARKER = {"CS": "o", "SC": "^"}

SANS = "IBM Plex Sans"
SERIF = "Fraunces"
MONO = "IBM Plex Mono"

DIVERGING = LinearSegmentedColormap.from_list(
    "cobalt_rule_vermilion", [COBALT, "#9DB2EE", "#F1F2F3", "#F09A9B", VERMILION]
)
SEQUENTIAL = LinearSegmentedColormap.from_list("rule_ink", ["#FFFFFF", RULE, "#7C8790", INK])

CELL_LABEL = {
    "c-CS-s": "Ctrl CS sal", "c-SC-s": "Ctrl SC sal", "c-CS-m": "Ctrl CS mem",
    "c-SC-m": "Ctrl SC mem", "t-CS-s": "Ts CS sal", "t-SC-s": "Ts SC sal",
    "t-CS-m": "Ts CS mem", "t-SC-m": "Ts SC mem",
}


def register_fonts() -> list[str]:
    """Make the house fonts available to matplotlib; returns registered families."""
    families = []
    FONT_CACHE.mkdir(parents=True, exist_ok=True)
    for path in sorted(FONT_DIR.glob("*")):
        if path.suffix == ".woff":
            target = FONT_CACHE / f"{path.stem}.ttf"
            if not target.exists():
                from fontTools.ttLib import TTFont

                font = TTFont(path)
                font.flavor = None
                font.save(target)
            path = target
        elif path.suffix not in (".ttf", ".otf"):
            continue
        font_manager.fontManager.addfont(str(path))
        families.append(font_manager.FontProperties(fname=str(path)).get_name())
    return sorted(set(families))


def apply_style() -> None:
    """Apply the house style globally (call once per process before plotting)."""
    available = register_fonts()
    sans = [SANS] if SANS in available else []
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": sans + ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "text.color": INK,
            "axes.titlesize": 8,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 6,
            "axes.labelsize": 7,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": RULE,
            "grid.linewidth": 0.4,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 6.3,
            "legend.frameon": False,
            "legend.handlelength": 1.1,
            "legend.borderaxespad": 0.2,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "mathtext.fontset": "custom",
            "mathtext.rm": SANS,
            "mathtext.it": f"{SANS}:style=italic",
            "mathtext.cal": SANS,
            "mathtext.bf": f"{SANS}:weight=bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
            "figure.dpi": 150,
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "svg.hashsalt": "mice-protein",  # stable element IDs in SVG output
        }
    )


def title(ax, text: str, **kwargs) -> None:
    """Editorial panel heading in Fraunces."""
    ax.set_title(text, fontfamily=SERIF, fontweight="semibold", fontsize=8, loc="left",
                 color=INK, **kwargs)


def annotate_stat(ax, x, y, text: str, transform=None, **kwargs) -> None:
    """Technical annotation (statistics, sample sizes) in IBM Plex Mono."""
    kwargs.setdefault("fontsize", 6)
    kwargs.setdefault("color", INK)
    ax.text(x, y, text, fontfamily=MONO, transform=transform or ax.transData, **kwargs)


def panel_label(ax, letter: str, x: float = -0.18, y: float = 1.06) -> None:
    """Bold lowercase panel letter at the top-left corner."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            fontfamily=SANS, va="bottom", ha="left", color=INK)


def schematic_tag(ax) -> None:
    """Mark a panel as an illustrative schematic, not experimental data."""
    ax.text(1.0, 1.02, "SCHEMATIC", transform=ax.transAxes, ha="right", va="bottom",
            fontfamily=MONO, fontsize=5.5, color=SUBTLE_TEXT)


def zero_line(ax, orientation: str = "h", value: float = 0.0) -> None:
    kwargs = {"color": SUBTLE_TEXT, "lw": 0.5, "ls": (0, (3, 2)), "zorder": 1}
    (ax.axhline if orientation == "h" else ax.axvline)(value, **kwargs)


def no_grid(ax) -> None:
    ax.grid(False)


def save(fig, directory: Path, name: str) -> None:
    """Write SVG, PDF and 300-dpi PNG with a margin (never cropped)."""
    directory.mkdir(parents=True, exist_ok=True)
    # Fixed metadata so unchanged figures are byte-identical across rebuilds.
    metadata = {"svg": {"Date": None}, "pdf": {"Creator": "mice_protein", "CreationDate": None},
                "png": {"Software": None}}
    for ext in ("svg", "pdf", "png"):
        fig.savefig(directory / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.06,
                    dpi=300, metadata=metadata[ext])
    plt.close(fig)
