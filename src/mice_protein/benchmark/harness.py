"""Evaluation harness: identical folds for every model, out-of-fold probabilities, timing."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold

from . import models as M
from .metrics import score


def mouse_folds(strata: pd.Series, n_repeats: int, n_splits: int, seed: int):
    """List of (repeat, fold, train_idx, test_idx), stratified on ``strata``."""
    folds = []
    for r in range(n_repeats):
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        for f, (train, test) in enumerate(splitter.split(np.zeros(len(strata)), strata)):
            folds.append((r, f, train, test))
    return folds


def fit_predict(name: str, x_train, y_train, x_test, strata_train, seed: int, inner_seed: int):
    """Fit one model (with inner tuning if specified) and return test probabilities."""
    spec = M.SPECS[name]
    estimator = M.build(name, seed)
    if spec.grid:
        inner = StratifiedKFold(3, shuffle=True, random_state=inner_seed)
        estimator = GridSearchCV(estimator, spec.grid, scoring="balanced_accuracy",
                                 cv=list(inner.split(x_train, strata_train)), error_score="raise")
    start = time.perf_counter()
    estimator.fit(x_train, y_train)
    proba = np.asarray(estimator.predict_proba(x_test), dtype=float)
    seconds = time.perf_counter() - start
    return proba / proba.sum(axis=1, keepdims=True), np.asarray(estimator.classes_), seconds


def cross_validate(name: str, x: pd.DataFrame, y: pd.Series, strata: pd.Series, folds, seed: int):
    """Run one model over all folds.

    Returns fold-level metrics and pooled out-of-fold probabilities for every repeat.
    """
    classes = np.unique(y)
    fold_rows, predictions = [], []
    for r, f, train, test in folds:
        proba, model_classes, seconds = fit_predict(
            name, x.iloc[train], y.iloc[train], x.iloc[test], strata.iloc[train], seed,
            inner_seed=seed + 100 * r + f)
        if not np.array_equal(model_classes, classes):
            raise ValueError(f"{name}: training fold is missing a class")
        fold_rows.append({"model": name, "repeat": r, "fold": f, "seconds": seconds,
                          "n_train": len(train), "n_test": len(test),
                          **score(y.iloc[test].to_numpy(), proba, classes)})
        frame = pd.DataFrame(proba, columns=classes, index=x.index[test])
        predictions.append(frame.assign(model=name, repeat=r, fold=f, observed=y.iloc[test]))
    return pd.DataFrame(fold_rows), pd.concat(predictions)


def pooled_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """Metrics on pooled out-of-fold predictions, one row per model and repeat."""
    rows = []
    for (model, repeat), part in predictions.groupby(["model", "repeat"]):
        classes = np.array([c for c in part.columns if c not in ("model", "repeat", "fold",
                                                                  "observed")])
        rows.append({"model": model, "repeat": repeat,
                     **score(part.observed.to_numpy(), part[classes].to_numpy(), classes)})
    return pd.DataFrame(rows)


def row_level_folds(groups: pd.Series, strata: pd.Series, grouped: bool, seed: int, n_splits=5):
    """Five folds over measurement rows: random (leaky) or holding out whole mice."""
    splitter = (StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed) if grouped
                else StratifiedKFold(n_splits, shuffle=True, random_state=seed))
    args = (np.zeros(len(strata)), strata) + ((groups,) if grouped else ())
    return [(0, f, tr, te) for f, (tr, te) in enumerate(splitter.split(*args))]


def clone_spec_check() -> None:
    """Fail fast if any model cannot be cloned (sklearn contract)."""
    for name in M.SPECS:
        if name not in M.FOUNDATION:
            clone(M.build(name))
