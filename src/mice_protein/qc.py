"""Measurement quality control: technical structure, missingness, duplicates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import dilution_block, protein_columns


def variance_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Decompose each protein's log2 variance into between-mouse and technical parts.

    For each protein, total variance of the 1,080 log2 measurements is split into
    * between-mouse variance (variance of mouse means),
    * within-mouse variance explained by the inferred technical block, and
    * residual within-block (replicate) variance.

    ``icc`` is the between-mouse share of total variance. ``block_share`` is the
    share of *within-mouse* variance explained by block membership.
    """
    proteins = protein_columns(frame)
    logged = np.log2(frame[proteins].where(frame[proteins] > 0))
    mouse = frame["AnimalID"]
    block = dilution_block(frame)
    rows = []
    for protein in proteins:
        y = logged[protein]
        mouse_mean = y.groupby(mouse).transform("mean")
        cell_mean = y.groupby([mouse, block]).transform("mean")
        within = (y - mouse_mean).pow(2).sum()
        replicate = (y - cell_mean).pow(2).sum()
        total = (y - y.mean()).pow(2).sum()
        rows.append(
            {
                "protein": protein,
                "icc": 1 - within / total,
                "block_share": 1 - replicate / within if within > 0 else np.nan,
                "replicate_sd": np.sqrt(replicate / y.notna().sum()),
                "between_mouse_sd": y.groupby(mouse).mean().std(),
            }
        )
    return pd.DataFrame(rows)


def block_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean within-mouse standardized log2 level per replicate index (1-15) and protein."""
    proteins = protein_columns(frame)
    logged = np.log2(frame[proteins].where(frame[proteins] > 0))
    grouped = logged.groupby(frame["AnimalID"])
    z = (logged - grouped.transform("mean")) / grouped.transform("std")
    index = frame["MouseID"].str.rsplit("_", n=1).str[1].astype(int)
    return z.groupby(index).mean()


def missingness_by_cell(frame: pd.DataFrame) -> pd.DataFrame:
    """Test whether missing measurements concentrate in particular experimental cells.

    For each protein with missing values, the per-mouse fraction of missing
    measurements is compared across the eight cells with a Kruskal-Wallis test
    (mouse is the unit). Informative missingness would bias complete-case summaries.
    """
    proteins = protein_columns(frame)
    fraction = frame[proteins].isna().groupby(frame["AnimalID"]).mean()
    cells = frame.groupby("AnimalID")["class"].first()
    rows = []
    for protein in fraction.columns[fraction.sum() > 0]:
        groups = [g.to_numpy() for _, g in fraction[protein].groupby(cells)]
        if len({tuple(g) for g in groups}) > 1 and np.ptp(fraction[protein]) > 0:
            h, p = stats.kruskal(*groups)
        else:
            h, p = np.nan, np.nan
        rows.append(
            {
                "protein": protein,
                "missing_rows": int(frame[protein].isna().sum()),
                "mice_fully_missing": int((fraction[protein] == 1).sum()),
                "kruskal_h": h,
                "p": p,
                **{f"missing_frac_{c}": fraction[protein][cells == c].mean()
                   for c in sorted(cells.unique())},
            }
        )
    return pd.DataFrame(rows)
