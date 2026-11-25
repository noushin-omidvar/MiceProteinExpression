"""Gene-dosage test: are proteins from triplicated genes elevated 1.5-fold?

Trisomic/disomic fold changes are estimated twice from **disjoint mice**: once among
saline-treated mice (discovery) and once among memantine-treated mice
(replication). Memantine is not expected to change gene copy number, so agreement
between the two estimates measures reproducibility without reusing any animal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .annotations import EXPECTED_DOSAGE_LOG2FC, HSA21_DISOMIC_TS65DN, TRISOMIC_TS65DN
from .contrasts import MECHANISTIC, contrast_table


def dosage_class(protein: str) -> str:
    if protein in TRISOMIC_TS65DN:
        return "Triplicated in Ts65Dn"
    if protein in HSA21_DISOMIC_TS65DN:
        return "HSA21 ortholog, disomic"
    return "Not on HSA21"


def dosage_table(log2_values: pd.DataFrame, cells: pd.Series) -> pd.DataFrame:
    """Per-protein log2 fold change (Ts65Dn / control) in each treatment arm."""
    contrasts = {k: MECHANISTIC[k] for k in ("Trisomy (saline)", "Trisomy (memantine)")}
    table = contrast_table(log2_values, cells, contrasts)
    wide = table.pivot(index="feature", columns="contrast", values=["estimate", "lower", "upper", "p"])
    wide.columns = [f"{stat}_{'saline' if 'saline' in c else 'memantine'}" for stat, c in wide.columns]
    wide = wide.reset_index().rename(columns={"feature": "protein"})
    wide["dosage_class"] = wide.protein.map(dosage_class)
    wide["fold_change_saline"] = 2 ** wide["estimate_saline"]
    wide["fraction_of_expected_saline"] = wide["estimate_saline"] / EXPECTED_DOSAGE_LOG2FC
    return wide


def set_mean_bootstrap(
    log2_values: pd.DataFrame,
    cells: pd.Series,
    members: list[str],
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """Mean genotype log2FC over a protein set, with a mouse-level bootstrap CI.

    The genotype effect is averaged over all four treatment x conditioning strata
    (all 72 mice). Mice are resampled within cells, so the interval reflects
    biological sampling variability and respects the correlation among proteins.
    """
    from .contrasts import FACTORIAL

    weights = pd.Series(FACTORIAL["Genotype"])
    cells = cells.reindex(log2_values.index)

    def estimate(values: pd.DataFrame, labels: pd.Series) -> float:
        means = values[members].groupby(labels).mean().reindex(weights.index)
        return float(means.mul(weights, axis=0).sum().mean())

    point = estimate(log2_values, cells)
    draws = np.empty(n_boot)
    positions = {c: np.flatnonzero(cells.to_numpy() == c) for c in weights.index}
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(p, len(p)) for p in positions.values()])
        draws[b] = estimate(log2_values.iloc[idx], cells.iloc[idx])
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return {"estimate": point, "lower": lower, "upper": upper}
