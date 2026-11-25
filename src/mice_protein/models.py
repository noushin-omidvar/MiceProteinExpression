"""Leakage-safe model pipelines."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(name: str, seed: int = 2025):
    """Build a model with preprocessing fitted inside each training fold."""
    if name == "dummy":
        estimator = DummyClassifier(strategy="prior")
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", estimator)])
    if name == "logistic":
        estimator = LogisticRegression(
            C=0.1, max_iter=5000, class_weight="balanced", random_state=seed
        )
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )
    if name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        return Pipeline(
            [("impute", SimpleImputer(strategy="median", add_indicator=True)), ("model", estimator)]
        )
    raise ValueError(f"Unknown model: {name}")
