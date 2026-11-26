"""Statistical comparison of models evaluated on the same cross-validation folds."""

from __future__ import annotations

import numpy as np
from scipy import stats


def corrected_resampled_ttest(
    differences: np.ndarray, n_train: int, n_test: int
) -> dict[str, float]:
    """Nadeau-Bengio corrected t-test for repeated k-fold cross-validation.

    Fold-level scores share training data, so their differences are positively correlated
    and the naive t-test is anti-conservative. The correction inflates the variance by
    ``n_test / n_train``:

        t = mean(d) / sqrt((1 / J + n_test / n_train) * var(d)),   df = J - 1,

    where J is the number of folds (repeats x splits). Nadeau & Bengio (2003);
    Bouckaert & Frank (2004).
    """
    d = np.asarray(differences, dtype=float)
    j = len(d)
    var = d.var(ddof=1)
    se = np.sqrt((1 / j + n_test / n_train) * var)
    t = d.mean() / se if se > 0 else np.inf * np.sign(d.mean())
    p = 2 * stats.t.sf(abs(t), j - 1)
    half = stats.t.ppf(0.975, j - 1) * se
    return {"mean_difference": float(d.mean()), "se": float(se), "t": float(t), "df": j - 1,
            "p": float(p), "lower": float(d.mean() - half), "upper": float(d.mean() + half)}


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def bootstrap_interval(values, rng: np.random.Generator, n_boot: int = 4000):
    """Percentile 95% interval of the mean by resampling the given values."""
    v = np.asarray(values, dtype=float)
    draws = rng.choice(v, size=(n_boot, len(v))).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
