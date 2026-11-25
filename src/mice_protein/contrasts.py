"""Planned contrasts of the eight experimental cell means.

Every group comparison in the mechanistic analyses is a linear contrast
``L = sum_c w_c * mean_c`` over the eight genotype x conditioning x treatment cells,
with one value per mouse as the experimental unit. Inference uses Welch-type
standard errors that allow each cell its own variance,

    Var(L) = sum_c w_c^2 s_c^2 / n_c,

with Welch-Satterthwaite degrees of freedom. This is the classical unequal-variance
planned-contrast test; for a two-cell contrast it reduces to Welch's t-test.

Sign conventions: Ts65Dn, memantine, and context-shock (CS) are the "+" levels, so
a positive genotype effect means higher in Ts65Dn, a positive treatment effect
higher with memantine, and a positive conditioning effect higher in CS.

Weights are scaled so that
* main effects are differences of equally weighted level means,
* two-way interactions are the *average difference of differences*, and
* the three-way interaction is the difference of two difference-of-differences.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

from .data import CELLS

_SIGN = {"c": -1, "t": 1, "SC": -1, "CS": 1, "s": -1, "m": 1}


def _signs(cell: str) -> tuple[int, int, int]:
    genotype, behavior, treatment = cell.split("-")
    return _SIGN[genotype], _SIGN[behavior], _SIGN[treatment]


def _factorial_weights() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    names = ("Genotype", "Conditioning", "Treatment")
    for mask in product((0, 1), repeat=3):
        order = sum(mask)
        if order == 0:
            continue
        label = "×".join(n for n, m in zip(names, mask) if m)
        scale = {1: 4.0, 2: 2.0, 3: 1.0}[order]
        out[label] = {
            cell: float(np.prod([s for s, m in zip(_signs(cell), mask) if m])) / scale
            for cell in CELLS
        }
    return out


FACTORIAL: dict[str, dict[str, float]] = _factorial_weights()
"""Main effects and all interactions of the 2x2x2 design."""


def _within(fixed: dict[int, str], vary: int, scale: float) -> dict[str, float]:
    weights = {}
    for cell in CELLS:
        parts = cell.split("-")
        if all(parts[i] == level for i, level in fixed.items()):
            weights[cell] = _signs(cell)[vary] / scale
    return weights


MECHANISTIC: dict[str, dict[str, float]] = {
    # Ts65Dn - control, averaged over conditioning, within one treatment arm.
    "Trisomy (saline)": _within({2: "s"}, vary=0, scale=2),
    "Trisomy (memantine)": _within({2: "m"}, vary=0, scale=2),
    # Memantine - saline, averaged over conditioning, within one genotype.
    "Memantine in control": _within({0: "c"}, vary=2, scale=2),
    "Memantine in Ts65Dn": _within({0: "t"}, vary=2, scale=2),
    # CS - SC within each genotype x treatment cell.
    "Learning: control saline": _within({0: "c", 2: "s"}, vary=1, scale=1),
    "Learning: control memantine": _within({0: "c", 2: "m"}, vary=1, scale=1),
    "Learning: Ts65Dn saline": _within({0: "t", 2: "s"}, vary=1, scale=1),
    "Learning: Ts65Dn memantine": _within({0: "t", 2: "m"}, vary=1, scale=1),
}
"""Simple-effect contrasts that map onto the mechanistic questions."""


def cell_summary(values: pd.DataFrame, cells: pd.Series) -> dict[str, pd.DataFrame]:
    """Per-cell mean, variance (ddof=1), and non-missing count for each feature."""
    grouped = values.groupby(cells.reindex(values.index))
    return {
        "mean": grouped.mean().reindex(CELLS),
        "var": grouped.var(ddof=1).reindex(CELLS),
        "n": grouped.count().reindex(CELLS),
    }


def contrast_table(
    values: pd.DataFrame,
    cells: pd.Series,
    contrasts: dict[str, dict[str, float]],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Estimate every contrast for every feature (column) of ``values``.

    Parameters
    ----------
    values
        One row per mouse, one column per feature (protein, ratio, or score).
    cells
        Cell code for each mouse (index aligned with ``values``).
    contrasts
        Mapping from contrast name to ``{cell: weight}``.

    Returns
    -------
    Long table with estimate, standard error, Welch df, t, two-sided p, and a
    ``1 - alpha`` confidence interval. Features with fewer than two observed mice in
    any cell used by a contrast get NaN for that contrast.
    """
    summary = cell_summary(values, cells)
    rows = []
    for name, weights in contrasts.items():
        w = pd.Series(weights, dtype=float)
        mean = summary["mean"].loc[w.index]
        var = summary["var"].loc[w.index]
        n = summary["n"].loc[w.index]
        valid = (n >= 2).all()
        estimate = mean.mul(w, axis=0).sum()
        components = var.div(n).mul(w**2, axis=0)
        se = np.sqrt(components.sum())
        df = components.sum() ** 2 / (components**2).div(n - 1).sum()
        t = estimate / se
        p = 2 * stats.t.sf(np.abs(t), df)
        half = stats.t.ppf(1 - alpha / 2, df) * se
        frame = pd.DataFrame(
            {
                "feature": values.columns,
                "contrast": name,
                "estimate": estimate.to_numpy(),
                "se": se.to_numpy(),
                "df": df.to_numpy(),
                "t": t.to_numpy(),
                "p": p,
                "lower": (estimate - half).to_numpy(),
                "upper": (estimate + half).to_numpy(),
                "n_mice": n.sum().to_numpy(),
            }
        )
        frame.loc[~valid.to_numpy(), ["estimate", "se", "df", "t", "p", "lower", "upper"]] = np.nan
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def add_fdr(table: pd.DataFrame, by: str | None = None, column: str = "p") -> pd.DataFrame:
    """Add Benjamini-Hochberg q-values, optionally within groups of ``by``."""
    from .statistics import benjamini_hochberg

    table = table.copy()
    table["q"] = np.nan
    groups = [table.index] if by is None else table.groupby(by).groups.values()
    for index in groups:
        subset = table.loc[index, column].dropna()
        if len(subset):
            table.loc[subset.index, "q"] = benjamini_hochberg(subset.to_numpy())
    return table
