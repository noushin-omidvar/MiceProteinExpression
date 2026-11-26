"""Multivariate protein signatures: learning response and memantine rescue.

Both analyses reduce the 70-protein profile of each mouse to a single score: its
projection on an *axis* defined as a contrast of cell-mean profiles. Using one
score per mouse gives far more power than testing 70 proteins separately, and lets
three-way questions (does trisomy blunt the conditioning response, and does
memantine restore it?) be asked with 7-10 mice per cell.

Avoiding self-projection bias
-----------------------------
A mouse used to estimate an axis is biased toward that axis when projected on it.
Every mouse in a cell that defines the axis is therefore scored on an axis
re-estimated *without that mouse* (leave-one-out); all other mice are scored
without contributing to that axis. Because the leave-one-out cell mean has a
closed form, this costs a single pass. Pooled within-cell standardization below
is fitted once on the full labeled cohort, however, so these projections are
descriptive and are not fully out-of-sample scores.

Profiles are standardized per protein by the *pooled within-cell standard
deviation*, so one unit is one SD of biological mouse-to-mouse variability within
an experimental cell, and the grand mean is zero. The handful of missing mouse
means (2 of 5,040 in the analysis panel) are set to that zero after scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import CELLS


def standardize(values: pd.DataFrame, cells: pd.Series) -> pd.DataFrame:
    """Center by the grand mean and scale by the pooled within-cell SD."""
    cells = cells.reindex(values.index)
    residual = values - values.groupby(cells).transform("mean")
    dof = values.notna().groupby(cells).sum().sub(1).clip(lower=0).sum()
    pooled_sd = np.sqrt((residual**2).sum() / dof)
    return ((values - values.mean()) / pooled_sd).fillna(0.0)


def _cell_stats(z: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.stack([z[codes == k].mean(axis=0) for k in range(len(CELLS))])
    counts = np.bincount(codes, minlength=len(CELLS)).astype(float)
    return means, counts


def axis_scores(
    z: np.ndarray,
    codes: np.ndarray,
    weights: np.ndarray,
    multiplicity: np.ndarray | None = None,
) -> np.ndarray:
    """Project each mouse on the contrast axis ``sum_c w_c * mean_c``.

    Mice in cells with non-zero weight are projected on a leave-one-out axis.

    Parameters
    ----------
    z
        Standardized profiles, shape (mice, proteins).
    codes
        Integer cell index (position in :data:`CELLS`) for each mouse.
    weights
        Contrast weight for each of the eight cells.
    multiplicity
        Number of copies of each row's mouse in ``z`` (bootstrap resamples contain
        duplicates). All copies are left out together; defaults to 1.
    """
    means, counts = _cell_stats(z, codes)
    axis = weights @ means
    w = weights[codes][:, None]
    m = np.ones(len(z)) if multiplicity is None else np.asarray(multiplicity, float)
    m = m[:, None]
    # Leave-out axis: mean_c(-i) - mean_c = m_i (mean_c - x_i) / (n_c - m_i).
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(w != 0, w * m * (means[codes] - z) / (counts[codes][:, None] - m), 0.0)
    loo_axis = axis[None, :] + shift
    return np.einsum("ij,ij->i", z, loo_axis) / np.linalg.norm(loo_axis, axis=1)


def cell_score_means(scores: np.ndarray, codes: np.ndarray) -> pd.Series:
    """Mean projection score per cell."""
    return pd.Series([scores[codes == k].mean() for k in range(len(CELLS))], index=CELLS)


def weight_vector(weights: dict[str, float]) -> np.ndarray:
    """Dense eight-cell weight vector from a ``{cell: weight}`` mapping."""
    return np.array([weights.get(cell, 0.0) for cell in CELLS])


def debiased_squared_norm(z: np.ndarray, codes: np.ndarray, weights: np.ndarray) -> float:
    """Unbiased estimate of ``||sum_c w_c mu_c||^2`` for a contrast of cell means.

    The plug-in squared norm is inflated by sampling noise:
    ``E||d_hat||^2 = ||d||^2 + sum_c w_c^2 tr(Sigma_c) / n_c``. Subtracting the
    unbiased noise term gives an unbiased estimate (which can be negative when the
    true effect is near zero).
    """
    means, counts = _cell_stats(z, codes)
    plug_in = float(np.sum((weights @ means) ** 2))
    noise = 0.0
    for k, w in enumerate(weights):
        if w:
            trace = z[codes == k].var(axis=0, ddof=1).sum()
            noise += w**2 * trace / counts[k]
    return plug_in - noise


def disattenuated_cosine(
    z: np.ndarray, codes: np.ndarray, first: np.ndarray, second: np.ndarray
) -> float:
    """Cosine between two contrast vectors, corrected for sampling noise.

    For contrasts built from *disjoint* cells, the plug-in inner product is unbiased,
    while each norm is inflated by noise; dividing by debiased norms removes the
    attenuation toward zero that noise causes in a raw cosine.
    """
    if np.any((first != 0) & (second != 0)):
        raise ValueError("Contrasts must use disjoint cells for an unbiased inner product")
    means, _ = _cell_stats(z, codes)
    inner = float((first @ means) @ (second @ means))
    n1 = debiased_squared_norm(z, codes, first)
    n2 = debiased_squared_norm(z, codes, second)
    if n1 <= 0 or n2 <= 0:
        return np.nan
    return inner / np.sqrt(n1 * n2)


# ---------------------------------------------------------------------------
# Analysis definitions
# ---------------------------------------------------------------------------

LEARNING_AXIS = weight_vector({"c-CS-s": 1.0, "c-SC-s": -1.0})
"""Reference conditioning (CS - SC) axis: saline-treated control mice."""

TRISOMY_AXIS = weight_vector({"t-CS-s": 0.5, "t-SC-s": 0.5, "c-CS-s": -0.5, "c-SC-s": -0.5})
"""Trisomy axis: Ts65Dn - control in saline mice, averaged over conditioning."""

ARMS = {"control saline": ("c", "s"), "control memantine": ("c", "m"),
        "Ts65Dn saline": ("t", "s"), "Ts65Dn memantine": ("t", "m")}


def learning_vector(genotype: str, treatment: str) -> np.ndarray:
    """CS - SC contrast within one genotype x treatment arm."""
    return weight_vector({f"{genotype}-CS-{treatment}": 1.0, f"{genotype}-SC-{treatment}": -1.0})


@dataclass
class LearningResult:
    scores: np.ndarray
    response: pd.Series  # CS - SC score difference per arm
    relative: pd.Series  # response / control-saline response
    cosine: pd.Series  # disattenuated cosine of each arm's vector with control saline
    loadings: pd.Series  # unit reference axis (protein weights)


def learning_analysis(
    z: pd.DataFrame, codes: np.ndarray, multiplicity: np.ndarray | None = None
) -> LearningResult:
    """Score all mice on the control-saline conditioning axis.

    The response of each genotype x treatment arm is the CS - SC difference of
    mean scores, expressed relative to the (leave-one-out) control-saline response.
    """
    x = z.to_numpy()
    scores = axis_scores(x, codes, LEARNING_AXIS, multiplicity)
    means = cell_score_means(scores, codes)
    response = pd.Series(
        {arm: means[f"{g}-CS-{t}"] - means[f"{g}-SC-{t}"] for arm, (g, t) in ARMS.items()}
    )
    reference = learning_vector("c", "s")
    cosine = pd.Series(
        {
            arm: disattenuated_cosine(x, codes, reference, learning_vector(g, t))
            for arm, (g, t) in ARMS.items()
            if arm != "control saline"
        }
    )
    cell_means, _ = _cell_stats(x, codes)
    axis = LEARNING_AXIS @ cell_means
    loadings = pd.Series(axis / np.linalg.norm(axis), index=z.columns)
    return LearningResult(scores, response, response / response["control saline"], cosine,
                          loadings)


@dataclass
class RescueResult:
    scores: np.ndarray
    arm_means: pd.Series  # mean trisomy-axis score per arm (averaged over conditioning)
    rescue_fraction: float  # (Ts-sal - Ts-mem) / (Ts-sal - Ctrl-sal)
    persistence: float  # (Ts-mem - Ctrl-mem) / (Ts-sal - Ctrl-sal)
    control_shift: float  # (Ctrl-mem - Ctrl-sal) / (Ts-sal - Ctrl-sal)
    rescue_by_conditioning: pd.Series
    magnitudes: pd.Series  # debiased Euclidean effect sizes (pooled-SD units)
    along_axis_share: float  # share of memantine-in-Ts effect lying on the trisomy axis
    loadings: pd.Series


def rescue_analysis(
    z: pd.DataFrame, codes: np.ndarray, multiplicity: np.ndarray | None = None
) -> RescueResult:
    """Quantify how far memantine moves Ts65Dn profiles back along the trisomy axis."""
    x = z.to_numpy()
    scores = axis_scores(x, codes, TRISOMY_AXIS, multiplicity)
    means = cell_score_means(scores, codes)

    def arm(g: str, t: str, behaviors=("CS", "SC")) -> float:
        return float(np.mean([means[f"{g}-{b}-{t}"] for b in behaviors]))

    arm_means = pd.Series({name: arm(g, t) for name, (g, t) in ARMS.items()})
    gap = arm_means["Ts65Dn saline"] - arm_means["control saline"]
    by_conditioning = pd.Series(
        {
            b: (arm("t", "s", (b,)) - arm("t", "m", (b,))) / (arm("t", "s", (b,)) - arm("c", "s", (b,)))
            for b in ("CS", "SC")
        }
    )
    drug_t = weight_vector({"t-CS-m": .5, "t-SC-m": .5, "t-CS-s": -.5, "t-SC-s": -.5})
    drug_c = weight_vector({"c-CS-m": .5, "c-SC-m": .5, "c-CS-s": -.5, "c-SC-s": -.5})
    magnitudes = pd.Series(
        {
            "Trisomy (saline)": debiased_squared_norm(x, codes, TRISOMY_AXIS),
            "Memantine in Ts65Dn": debiased_squared_norm(x, codes, drug_t),
            "Memantine in control": debiased_squared_norm(x, codes, drug_c),
        }
    ).clip(lower=0) ** 0.5
    parallel = arm_means["Ts65Dn memantine"] - arm_means["Ts65Dn saline"]
    total = magnitudes["Memantine in Ts65Dn"] ** 2
    share = parallel**2 / total if total > 0 else np.nan
    cell_means, _ = _cell_stats(x, codes)
    axis = TRISOMY_AXIS @ cell_means
    return RescueResult(
        scores=scores,
        arm_means=arm_means,
        rescue_fraction=(arm_means["Ts65Dn saline"] - arm_means["Ts65Dn memantine"]) / gap,
        persistence=(arm_means["Ts65Dn memantine"] - arm_means["control memantine"]) / gap,
        control_shift=(arm_means["control memantine"] - arm_means["control saline"]) / gap,
        rescue_by_conditioning=by_conditioning,
        magnitudes=magnitudes,
        along_axis_share=float(min(share, 1.0)) if np.isfinite(share) else np.nan,
        loadings=pd.Series(axis / np.linalg.norm(axis), index=z.columns),
    )


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def stratified_bootstrap(
    codes: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Resample mice with replacement within each cell.

    Returns the resampled row indices and, for each resampled row, how many copies
    of that mouse the resample contains (needed for leave-out projection).
    """
    index = np.concatenate(
        [rng.choice(np.flatnonzero(codes == k), size=(codes == k).sum()) for k in np.unique(codes)]
    )
    copies = np.bincount(index, minlength=len(codes))[index]
    return index, copies


def permute_factor(
    codes: np.ndarray, factor: int, within: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Permute one factor's level among mice, holding the other two factors fixed.

    ``factor`` is 0 (genotype), 1 (conditioning) or 2 (treatment). Labels are
    shuffled only among mice selected by the boolean mask ``within`` and only inside
    strata defined by the remaining factors, so the null hypothesis is "this factor
    has no effect within these strata".
    """
    parts = np.array([cell.split("-") for cell in CELLS])[codes]
    others = [i for i in range(3) if i != factor]
    strata = ["-".join(p[others]) for p in parts]
    new = parts.copy()
    for stratum in sorted(set(np.array(strata)[within])):  # sorted: hash order varies
        idx = np.flatnonzero(within & (np.array(strata) == stratum))
        new[idx, factor] = rng.permutation(parts[idx, factor])
    lookup = {cell: k for k, cell in enumerate(CELLS)}
    return np.array([lookup["-".join(p)] for p in new])
