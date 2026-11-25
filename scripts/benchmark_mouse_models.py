#!/usr/bin/env python3
"""Compare mouse-level classifiers with shared, nested stratified splits."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from mice_protein.data import aggregate_animals, load_data, protein_columns
from mice_protein.models import build_model

ROOT = Path(__file__).resolve().parents[1]
SEED = 2025


def evaluate(x, y, strata, model, grid, *, prediction_sink=None):
    """Return pooled out-of-fold metrics, separately for each repeat."""
    labels = np.unique(y)
    for repeat in range(5):
        outer = StratifiedKFold(5, shuffle=True, random_state=SEED + repeat)
        probabilities = np.empty((len(y), len(labels)))
        for fold, (train, test) in enumerate(outer.split(x, strata)):
            inner = StratifiedKFold(3, shuffle=True, random_state=SEED + repeat * 5 + fold)
            search = (
                GridSearchCV(
                    model,
                    grid,
                    cv=list(inner.split(x.iloc[train], strata.iloc[train])),
                    scoring="balanced_accuracy",
                    error_score="raise",
                )
                if grid
                else clone(model)
            )
            search.fit(x.iloc[train], y.iloc[train])
            if not np.array_equal(search.classes_, labels):
                raise ValueError("A training fold is missing an outcome class")
            fold_probabilities = np.asarray(search.predict_proba(x.iloc[test]), dtype=float)
            totals = fold_probabilities.sum(axis=1, keepdims=True)
            if not np.allclose(totals, 1, atol=1e-5):
                raise ValueError("Model returned probabilities that do not sum to one")
            probabilities[test] = fold_probabilities / totals
            if prediction_sink is not None:
                prediction_sink(repeat, fold, test, labels, probabilities[test])
        predictions = labels[probabilities.argmax(axis=1)]
        yield {
            "repeat": repeat,
            "balanced_accuracy": balanced_accuracy_score(y, predictions),
            "log_loss": log_loss(y, probabilities, labels=labels),
        } | {
            f"recall_{label}": recall
            for label, recall in zip(
                labels, recall_score(y, predictions, labels=labels, average=None), strict=True
            )
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tabpfn", action="store_true", help="Include locally installed TabPFN")
    args = parser.parse_args()
    frame = load_data(ROOT / "Data/Raw_Data/Data_Cortex_Nuclear.xls")
    features = protein_columns(frame)
    mice = aggregate_animals(frame).set_index("AnimalID")
    grouped = frame.groupby("AnimalID")[features]
    means = mice[features]
    medians = grouped.median().reindex(mice.index)
    missing = frame.set_index("AnimalID")[features].isna().groupby(level=0).mean()
    missing = missing.reindex(mice.index)
    baseline = build_model("logistic", SEED)
    elastic = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            solver="saga", penalty="elasticnet", l1_ratio=0.5, max_iter=15000, random_state=SEED
        ),
    )
    svm = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC())
    # SVC's built-in probability estimation operates on independent mouse summaries.
    svm.set_params(svc__probability=True, svc__random_state=SEED)
    boosting = HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=7, early_stopping=False, random_state=SEED
    )
    candidates = [
        ("logistic_mean", means, baseline, {}),
        ("logistic_median", medians, baseline, {}),
        ("missingness_only", missing, baseline, {}),
        ("forest_mean", means, build_model("random_forest", SEED), {}),
        ("elastic_net_mean", means, elastic, {"logisticregression__C": [0.1, 1, 10]}),
        ("rbf_svm_mean", means, svm, {"svc__C": [0.3, 1, 3], "svc__gamma": ["scale"]}),
        ("gradient_boosting_mean", means, boosting, {"min_samples_leaf": [4, 8]}),
    ]
    if args.tabpfn:
        from tabpfn import TabPFNClassifier

        candidates.append(("tabpfn_mean", means, TabPFNClassifier(random_state=SEED), {}))
    rows = []
    with threadpool_limits(limits=1):
        for target in ("class", "Genotype", "Treatment", "Behavior"):
            for name, x, model, grid in candidates:
                rows.extend(
                    {"target": target, "model": name, **result}
                    for result in evaluate(x, mice[target], mice["class"], model, grid)
                )
    results = pd.DataFrame(rows)
    suffix = "_tabpfn" if args.tabpfn else ""
    results.to_csv(ROOT / f"results/metrics/mouse_model_benchmark{suffix}.csv", index=False)
    print(results.groupby(["target", "model"])[["balanced_accuracy", "log_loss"]].mean())


if __name__ == "__main__":
    main()
