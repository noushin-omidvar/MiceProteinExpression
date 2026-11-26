"""Group-aware evaluation and uncertainty summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def repeated_cv_predictions(
    X, y, groups, model, *, grouped: bool, repeats: int = 5, splits: int = 5, seed: int = 2025
):
    """Generate out-of-fold predictions across reproducible repeated CV partitions."""
    records = []
    y_array = np.asarray(y)
    groups_array = np.asarray(groups)
    for repeat in range(repeats):
        cv = (
            StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed + repeat)
            if grouped
            else StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed + repeat)
        )
        iterator = cv.split(X, y_array, groups_array) if grouped else cv.split(X, y_array)
        for fold, (train, test) in enumerate(iterator):
            if grouped and set(groups_array[train]) & set(groups_array[test]):
                raise RuntimeError("Animal leakage detected")
            fitted = clone(model).fit(X.iloc[train], y_array[train])
            pred = fitted.predict(X.iloc[test])
            proba = fitted.predict_proba(X.iloc[test])
            labels = fitted.classes_
            records.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "balanced_accuracy": balanced_accuracy_score(y_array[test], pred),
                    "macro_f1": f1_score(y_array[test], pred, average="macro", zero_division=0),
                    "log_loss": log_loss(y_array[test], proba, labels=labels),
                    "n_test_rows": len(test),
                    "n_test_animals": len(set(groups_array[test])),
                    "y_true": y_array[test].tolist(),
                    "y_pred": pred.tolist(),
                    "animal_ids": groups_array[test].tolist(),
                }
            )
    return records


def summarize(records: list[dict]) -> dict[str, float]:
    """Summarize fold metrics with empirical 95% intervals."""
    result = {}
    for metric in ["balanced_accuracy", "macro_f1", "log_loss"]:
        values = np.array([r[metric] for r in records])
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_sd"] = float(values.std(ddof=1))
        result[f"{metric}_q025"] = float(np.quantile(values, 0.025))
        result[f"{metric}_q975"] = float(np.quantile(values, 0.975))
    return result


def animal_vote_predictions(record: dict) -> tuple[list[str], list[str]]:
    """Collapse repeated predictions to one majority vote per mouse."""
    f = pd.DataFrame(
        {"animal": record["animal_ids"], "true": record["y_true"], "pred": record["y_pred"]}
    )
    agg = f.groupby("animal").agg(
        true=("true", "first"), pred=("pred", lambda s: s.value_counts().index[0])
    )
    return agg["true"].tolist(), agg["pred"].tolist()


MOUSE_CV_SEED = 2025


def mouse_level_cv(x: pd.DataFrame, y: pd.Series, strata: pd.Series, model, *,
                   repeats: int = 5, splits: int = 5, seed: int = MOUSE_CV_SEED):
    """Out-of-fold class probabilities from repeated stratified CV on one row per mouse.

    Returns a DataFrame indexed like ``x`` (repeated once per repeat) with one probability
    column per class plus ``repeat``, ``fold`` and ``observed`` columns.
    """
    labels = np.unique(y)
    parts = []
    for repeat in range(repeats):
        splitter = StratifiedKFold(splits, shuffle=True, random_state=seed + repeat)
        for fold, (train, test) in enumerate(splitter.split(x, strata)):
            fitted = clone(model).fit(x.iloc[train], y.iloc[train])
            if not np.array_equal(fitted.classes_, labels):
                raise ValueError("A training fold is missing an outcome class")
            proba = np.asarray(fitted.predict_proba(x.iloc[test]), dtype=float)
            part = pd.DataFrame(proba / proba.sum(axis=1, keepdims=True), columns=labels,
                                index=x.index[test])
            parts.append(part.assign(repeat=repeat, fold=fold, observed=y.iloc[test]))
    return pd.concat(parts)
