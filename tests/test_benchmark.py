"""Tests for the foundation-model study's metrics, statistics and fold construction."""

import numpy as np
import pandas as pd
import pytest

from mice_protein.benchmark import harness as H
from mice_protein.benchmark.factorized import FactorizedLogistic
from mice_protein.benchmark.metrics import brier, expected_calibration_error, score
from mice_protein.benchmark.stats import corrected_resampled_ttest, holm


def test_brier_perfect_and_uniform():
    classes = np.array(["a", "b"])
    y = np.array(["a", "b"])
    assert brier(y, np.array([[1.0, 0.0], [0.0, 1.0]]), classes) == 0
    assert brier(y, np.full((2, 2), 0.5), classes) == pytest.approx(0.5)


def test_ece_zero_when_confidence_matches_accuracy():
    classes = np.array([0, 1])
    proba = np.array([[0.8, 0.2]] * 10)
    y = np.array([0] * 8 + [1] * 2)  # 80% correct at 0.8 confidence
    assert expected_calibration_error(y, proba, classes) == pytest.approx(0.0, abs=1e-12)


def test_score_balanced_accuracy_and_logloss():
    classes = np.array([0, 1])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    result = score(np.array([0, 1, 1, 1]), proba, classes)
    assert result["balanced_accuracy"] == pytest.approx((1 + 2 / 3) / 2)
    assert result["log_loss"] > 0


def test_corrected_ttest_matches_formula_and_is_conservative():
    d = np.array([0.05, 0.02, 0.08, 0.01, 0.04, 0.06, 0.03, 0.07, 0.02, 0.05])
    out = corrected_resampled_ttest(d, n_train=58, n_test=14)
    se = np.sqrt((1 / 10 + 14 / 58) * d.var(ddof=1))
    assert out["t"] == pytest.approx(d.mean() / se)
    naive_se = np.sqrt(d.var(ddof=1) / 10)
    assert out["se"] > naive_se  # correction widens the interval


def test_holm_adjustment():
    adjusted = holm([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def test_mouse_folds_are_identical_and_disjoint():
    strata = pd.Series(np.repeat(np.arange(8), 9))
    a = H.mouse_folds(strata, 2, 5, 7)
    b = H.mouse_folds(strata, 2, 5, 7)
    assert all(np.array_equal(x[3], y[3]) for x, y in zip(a, b, strict=True))
    for _, _, train, test in a:
        assert set(train).isdisjoint(test)


def test_row_level_grouped_folds_hold_out_whole_mice():
    groups = pd.Series(np.repeat(np.arange(40), 3))
    strata = pd.Series(np.repeat(np.arange(40) % 4, 3))
    for _, _, train, test in H.row_level_folds(groups, strata, grouped=True, seed=1):
        assert set(groups[train]).isdisjoint(groups[test])


def test_matched_holdout_plan_keeps_tests_fixed_and_training_nested():
    """The exploratory learning curves must compare the same mice at each size."""
    import importlib.util
    from pathlib import Path

    from mice_protein.data import aggregate_animals, load_data

    root = Path(__file__).resolve().parents[1]
    script = root / "foundation_models/scripts/matched_holdout.py"
    spec = importlib.util.spec_from_file_location("matched_holdout", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mice = aggregate_animals(load_data(root / "Data/Raw_Data/Data_Cortex_Nuclear.xls"))
    mice = mice.set_index("AnimalID")
    plan = module.make_plan(mice)
    for _, part in plan.groupby(["repeat", "fold", "draw"]):
        tests = [set(g.loc[g.role == "test", "animal"])
                 for _, g in part.groupby("size")]
        assert all(test == tests[0] for test in tests)
        previous = set()
        for size in ["2", "3", "4", "5", "6", "full"]:
            train = set(part.loc[(part.role == "train") & (part["size"] == size), "animal"])
            assert previous <= train
            assert not train & tests[0]
            previous = train


def test_factorized_baseline_returns_normalized_eight_class_probabilities():
    labels = np.array([f"{g}-{b}-{t}" for g in ("c", "t")
                       for b in ("CS", "SC") for t in ("m", "s")])
    x = pd.DataFrame(np.eye(8))
    model = FactorizedLogistic().fit(x, labels)
    probability = model.predict_proba(x)
    assert probability.shape == (8, 8)
    np.testing.assert_allclose(probability.sum(axis=1), 1)
    assert set(model.predict(x)) <= set(labels)
