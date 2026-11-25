"""Mouse-level effect sizes and multiplicity control."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from .data import aggregate_animals, protein_columns


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def genotype_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute mouse-level standardized genotype effects and FDR-adjusted tests."""
    mice = aggregate_animals(frame)
    rows = []
    for protein in protein_columns(frame):
        control = mice.loc[mice.Genotype == "Control", protein].dropna().to_numpy()
        trisomy = mice.loc[mice.Genotype == "Ts65Dn", protein].dropna().to_numpy()
        pooled = np.sqrt(
            ((len(control) - 1) * control.var(ddof=1) + (len(trisomy) - 1) * trisomy.var(ddof=1))
            / (len(control) + len(trisomy) - 2)
        )
        effect = (trisomy.mean() - control.mean()) / pooled if pooled else np.nan
        _, pvalue = ttest_ind(trisomy, control, equal_var=False)
        rows.append(
            {
                "protein": protein,
                "hedges_like_d": effect,
                "p_value": pvalue,
                "control_mean": control.mean(),
                "trisomy_mean": trisomy.mean(),
            }
        )
    result = pd.DataFrame(rows)
    result["q_value"] = benjamini_hochberg(result.p_value.to_numpy())
    return result.sort_values("q_value")
