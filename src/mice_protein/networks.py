"""Differential co-expression (network rewiring) between genotypes.

Question: does trisomy change how proteins co-vary across mice, beyond changing
their means?

Correlations are computed on *residuals* after removing each of the eight cell
means, so that co-variation induced by the experimental design (e.g. two proteins
that both respond to conditioning) is not mistaken for coordination. Within each
genotype the residuals have ``n_mice - 4`` degrees of freedom.

The global test statistic is the root-mean-square difference of Fisher-z
transformed correlations over all protein pairs. Its null distribution comes from
permuting genotype labels within treatment x conditioning strata and recomputing
the residuals each time, which respects the design and the dependence among pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .signatures import permute_factor


def residual_correlations(
    z: np.ndarray, codes: np.ndarray, genotype_t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Within-genotype correlation matrices of cell-mean-centered profiles."""
    residual = z.copy()
    for k in np.unique(codes):
        residual[codes == k] -= residual[codes == k].mean(axis=0)
    control = np.corrcoef(residual[~genotype_t].T)
    trisomy = np.corrcoef(residual[genotype_t].T)
    return control, trisomy


def rewiring_statistic(control: np.ndarray, trisomy: np.ndarray) -> float:
    """Root-mean-square Fisher-z difference over the upper triangle."""
    upper = np.triu_indices_from(control, 1)
    diff = np.arctanh(np.clip(trisomy[upper], -0.999, 0.999)) - np.arctanh(
        np.clip(control[upper], -0.999, 0.999)
    )
    return float(np.sqrt(np.mean(diff**2)))


def is_trisomic(codes: np.ndarray) -> np.ndarray:
    """Boolean genotype indicator from cell codes (cells 4-7 are Ts65Dn)."""
    return codes >= 4


def rewiring_test(
    z: pd.DataFrame, codes: np.ndarray, n_permutations: int, rng: np.random.Generator
) -> dict:
    """Global permutation test and per-pair Fisher-z tests of rewiring."""
    x = z.to_numpy()
    control, trisomy = residual_correlations(x, codes, is_trisomic(codes))
    observed = rewiring_statistic(control, trisomy)
    everyone = np.ones(len(codes), dtype=bool)
    null = np.empty(n_permutations)
    for b in range(n_permutations):
        permuted = permute_factor(codes, 0, everyone, rng)
        null[b] = rewiring_statistic(*residual_correlations(x, permuted, is_trisomic(permuted)))
    p_global = (1 + np.sum(null >= observed)) / (1 + n_permutations)

    n_c = int((~is_trisomic(codes)).sum()) - 4
    n_t = int(is_trisomic(codes).sum()) - 4
    upper = np.triu_indices_from(control, 1)
    zc, zt = np.arctanh(control[upper]), np.arctanh(trisomy[upper])
    se = np.sqrt(1 / (n_c - 3) + 1 / (n_t - 3))
    pair_z = (zt - zc) / se
    names = np.array(z.columns)
    pairs = pd.DataFrame(
        {
            "protein_a": names[upper[0]],
            "protein_b": names[upper[1]],
            "r_control": control[upper],
            "r_ts65dn": trisomy[upper],
            "z": pair_z,
            "p": 2 * stats.norm.sf(np.abs(pair_z)),
        }
    )
    return {
        "control": pd.DataFrame(control, index=names, columns=names),
        "trisomy": pd.DataFrame(trisomy, index=names, columns=names),
        "statistic": observed,
        "null": null,
        "p_global": p_global,
        "pairs": pairs,
    }
