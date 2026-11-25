"""Correctness tests for the mechanistic-analysis machinery.

Each estimator is checked against a brute-force implementation or a simulation
with known ground truth, so the closed-form shortcuts used for speed are verified.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mice_protein import contrasts as C
from mice_protein import signatures as S
from mice_protein.annotations import PHOSPHO_PAIRS
from mice_protein.data import (
    CELLS,
    analysis_panel,
    dilution_block,
    duplicate_columns,
    load_data,
    log2_animal_matrix,
)
from mice_protein.memorization import obfuscate
from mice_protein.pathways import phospho_ratios
from mice_protein.power import contrast_power

RAW = "Data/Raw_Data/Data_Cortex_Nuclear.xls"


@pytest.fixture(scope="module")
def raw():
    return load_data(RAW)


def synthetic(n_per_cell=6, p=5, seed=0, effects=None):
    rng = np.random.default_rng(seed)
    codes = np.repeat(np.arange(8), n_per_cell)
    means = np.zeros((8, p)) if effects is None else effects
    z = means[codes] + rng.normal(size=(len(codes), p))
    return z, codes


# --- data contracts -----------------------------------------------------------

def test_duplicate_column_is_ps6_copy_of_arc(raw):
    assert duplicate_columns(raw) == {"pS6_N": "ARC_N"}


def test_dilution_blocks_are_five_blocks_of_three(raw):
    blocks = dilution_block(raw)
    assert set(blocks) == set(range(5))
    assert raw.assign(b=blocks).groupby(["AnimalID", "b"]).size().eq(3).all()


def test_log2_matrix_and_panel(raw):
    values, labels = log2_animal_matrix(raw)
    assert values.shape == (72, 77) and labels.index.equals(values.index)
    panel = analysis_panel(raw)
    assert len(panel) == 70 and "pS6_N" not in panel and "BCL2_N" not in panel


# --- contrasts -----------------------------------------------------------------

@pytest.mark.parametrize("name", list(C.FACTORIAL) + list(C.MECHANISTIC))
def test_contrast_weights_sum_to_zero(name):
    weights = (C.FACTORIAL | C.MECHANISTIC)[name]
    assert abs(sum(weights.values())) < 1e-12


def test_main_effect_is_difference_of_level_means():
    z, codes = synthetic()
    frame = pd.DataFrame(z[:, :1], columns=["x"])
    cells = pd.Series(np.array(CELLS)[codes])
    est = C.contrast_table(frame, cells, {"g": C.FACTORIAL["Genotype"]}).estimate[0]
    cell_means = frame.x.groupby(cells).mean()
    expected = cell_means[[c for c in CELLS if c[0] == "t"]].mean() - \
        cell_means[[c for c in CELLS if c[0] == "c"]].mean()
    assert est == pytest.approx(expected)


def test_two_cell_contrast_equals_welch_t_test():
    z, codes = synthetic(seed=3)
    z[codes == 0] *= 3  # unequal variances
    frame = pd.DataFrame(z[:, :1], columns=["x"])
    cells = pd.Series(np.array(CELLS)[codes])
    row = C.contrast_table(frame, cells, {"d": {CELLS[1]: 1, CELLS[0]: -1}}).iloc[0]
    ref = stats.ttest_ind(z[codes == 1, 0], z[codes == 0, 0], equal_var=False)
    assert row.t == pytest.approx(ref.statistic) and row.p == pytest.approx(ref.pvalue)


# --- projections ---------------------------------------------------------------

def brute_force_scores(z, codes, weights, drop_groups):
    scores = np.empty(len(z))
    for i in range(len(z)):
        keep = drop_groups != drop_groups[i] if weights[codes[i]] else np.ones(len(z), bool)
        means = np.stack([z[keep & (codes == k)].mean(axis=0) for k in range(8)])
        axis = weights @ means
        scores[i] = z[i] @ axis / np.linalg.norm(axis)
    return scores


def test_leave_one_out_scores_match_refitting():
    z, codes = synthetic(seed=1)
    fast = S.axis_scores(z, codes, S.TRISOMY_AXIS)
    slow = brute_force_scores(z, codes, S.TRISOMY_AXIS, np.arange(len(z)))
    np.testing.assert_allclose(fast, slow)


def test_bootstrap_leave_out_removes_all_copies():
    z, codes = synthetic(seed=2)
    idx, copies = S.stratified_bootstrap(codes, np.random.default_rng(0))
    fast = S.axis_scores(z[idx], codes[idx], S.LEARNING_AXIS, copies)
    slow = brute_force_scores(z[idx], codes[idx], S.LEARNING_AXIS, idx)
    np.testing.assert_allclose(fast, slow)


def test_debiased_norm_is_unbiased():
    effects = np.zeros((8, 20))
    effects[0, :] = 0.3  # true ||mu_0 - mu_1||^2 = 20 * 0.09 = 1.8
    weights = S.weight_vector({CELLS[0]: 1, CELLS[1]: -1})
    draws = [S.debiased_squared_norm(*synthetic(8, 20, s, effects), weights) for s in range(400)]
    plug = [np.sum((weights @ np.stack([z[c == k].mean(0) for k in range(8)])) ** 2)
            for z, c in (synthetic(8, 20, s, effects) for s in range(400))]
    assert np.mean(draws) == pytest.approx(1.8, abs=0.15)
    assert np.mean(plug) > 1.8 + 3  # plug-in is badly inflated (noise term = 20 * 2/8)


def test_disattenuated_cosine_recovers_identical_directions():
    effects = np.zeros((8, 30))
    effects[0, :] = effects[2, :] = 0.6
    a = S.weight_vector({CELLS[0]: 1, CELLS[1]: -1})
    b = S.weight_vector({CELLS[2]: 1, CELLS[3]: -1})
    values = []
    for seed in range(200):
        z, codes = synthetic(8, 30, seed, effects)
        values.append(S.disattenuated_cosine(z, codes, a, b))
    assert np.nanmedian(values) == pytest.approx(1.0, abs=0.1)


def test_permutation_preserves_other_factors_and_counts():
    _, codes = synthetic(seed=4)
    within = codes >= 4
    new = S.permute_factor(codes, 2, within, np.random.default_rng(1))
    old_parts = np.array([c.split("-") for c in np.array(CELLS)[codes]])
    new_parts = np.array([c.split("-") for c in np.array(CELLS)[new]])
    np.testing.assert_array_equal(old_parts[:, :2], new_parts[:, :2])
    np.testing.assert_array_equal(new[~within], codes[~within])
    assert np.bincount(new, minlength=8).tolist() == np.bincount(codes, minlength=8).tolist()


def test_rescue_fraction_recovers_simulated_partial_rescue():
    p = 40
    direction = np.ones(p) / np.sqrt(p)
    effects = np.zeros((8, p))
    for k, cell in enumerate(CELLS):
        g, _, t = cell.split("-")
        if g == "t":
            effects[k] = direction * (3.0 if t == "s" else 1.5)  # 50% rescue
    estimates = [S.rescue_analysis(pd.DataFrame(z), c).rescue_fraction
                 for z, c in (synthetic(10, p, s, effects) for s in range(100))]
    assert np.median(estimates) == pytest.approx(0.5, abs=0.1)


# --- other modules -------------------------------------------------------------

def test_phospho_ratio_is_log_difference(raw):
    values, _ = log2_animal_matrix(raw)
    ratios = phospho_ratios(values)
    phospho, total = PHOSPHO_PAIRS[0]
    np.testing.assert_allclose(ratios.iloc[:, 0], values[phospho] - values[total])


def test_power_matches_simulation():
    weights = S.weight_vector({CELLS[0]: 1, CELLS[1]: -1, CELLS[4]: -1, CELLS[5]: 1})
    analytic = contrast_power(1.5, 1.0, weights, 8)
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(3000):
        groups = [rng.normal(m, 1, 8) for m in (1.5, 0, 0, 0)]
        est = groups[0].mean() - groups[1].mean() - groups[2].mean() + groups[3].mean()
        pooled = np.mean([g.var(ddof=1) for g in groups])
        t = est / np.sqrt(pooled * 4 / 8)
        hits += abs(t) > stats.t.ppf(0.975, 28)
    assert analytic == pytest.approx(hits / 3000, abs=0.03)


def test_obfuscation_is_invertible():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(10, 4)), columns=list("abcd"))
    obf, key = obfuscate(frame, np.random.default_rng(5))
    assert list(obf.columns) == ["f00", "f01", "f02", "f03"]
    for _, row in key.iterrows():
        recovered = (obf[row.token] - row.offset) / row.scale
        np.testing.assert_allclose(recovered, frame[row.original])


def test_permutation_is_independent_of_python_hash_seed():
    """String-set iteration order varies with PYTHONHASHSEED; results must not."""
    import os
    import subprocess
    import sys

    code = ("import numpy as np; from mice_protein.signatures import permute_factor;"
            "c=np.repeat(np.arange(8),6);"
            "print(permute_factor(c,0,np.ones(48,bool),np.random.default_rng(3)).tolist())")
    outputs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                       env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": "src"}).stdout
        for seed in ("1", "2")
    }
    assert len(outputs) == 1
