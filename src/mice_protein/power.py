"""Power of planned contrasts for designing a follow-up experiment."""

from __future__ import annotations

import numpy as np
from scipy import stats


def contrast_power(
    effect: float,
    sd: float,
    weights: np.ndarray,
    n_per_cell: int,
    alpha: float = 0.05,
) -> float:
    """Two-sided power of a cell-mean contrast with a common within-cell SD.

    The contrast estimate has standard error ``sd * sqrt(sum(w^2) / n)``; the test
    uses a t reference with the pooled residual degrees of freedom
    ``n_cells * (n - 1)``, and power is computed from the non-central t
    distribution.
    """
    weights = np.asarray(weights, dtype=float)
    used = int(np.count_nonzero(weights))
    df = used * (n_per_cell - 1)
    se = sd * np.sqrt(np.sum(weights**2) / n_per_cell)
    ncp = effect / se
    critical = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(critical, df, ncp) + stats.nct.cdf(-critical, df, ncp))


def required_n(
    effect: float, sd: float, weights: np.ndarray, target: float = 0.8, alpha: float = 0.05,
    max_n: int = 500,
) -> int | None:
    """Smallest per-cell sample size reaching the target power (None if > max_n)."""
    for n in range(2, max_n + 1):
        if contrast_power(effect, sd, weights, n, alpha) >= target:
            return n
    return None
