"""Phospho/total activity ratios and pathway-level activity scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .annotations import PATHWAYS, PHOSPHO_PAIRS, ratio_name


def phospho_ratios(log2_values: pd.DataFrame) -> pd.DataFrame:
    """Per-mouse ``log2(phospho) - log2(total)`` for every phospho/total pair.

    Inputs are per-mouse means of log2 intensities, so each ratio is the log2 ratio
    of geometric-mean intensities across a mouse's technical measurements.
    """
    return pd.DataFrame(
        {ratio_name(total): log2_values[phospho] - log2_values[total]
         for phospho, total in PHOSPHO_PAIRS},
        index=log2_values.index,
    )


def pathway_scores(ratios: pd.DataFrame, cells: pd.Series) -> pd.DataFrame:
    """Average standardized member ratios into one activity score per pathway.

    Each ratio is centered at its grand mean and scaled by its pooled within-cell
    SD before averaging, so every member contributes equally and the score is in
    within-cell SD units of an average member.
    """
    from .signatures import standardize

    z = standardize(ratios, cells)
    out = {}
    for pathway, stems in PATHWAYS.items():
        members = [ratio_name(f"{stem}_N") for stem in stems]
        out[pathway] = z[members].mean(axis=1)
    return pd.DataFrame(out, index=ratios.index)


def pathway_coherence(ratios: pd.DataFrame, cells: pd.Series) -> pd.Series:
    """Mean pairwise within-cell correlation among each multi-member pathway's ratios.

    Positive coherence supports summarizing members by one score; values near zero
    indicate the members carry largely independent information.
    """
    residual = ratios - ratios.groupby(cells.reindex(ratios.index)).transform("mean")
    out = {}
    for pathway, stems in PATHWAYS.items():
        members = [ratio_name(f"{stem}_N") for stem in stems]
        if len(members) < 2:
            continue
        corr = residual[members].corr().to_numpy()
        out[pathway] = corr[np.triu_indices(len(members), 1)].mean()
    return pd.Series(out)
