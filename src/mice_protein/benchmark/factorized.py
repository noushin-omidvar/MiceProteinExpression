"""Design-informed eight-class baseline from three binary experimental factors."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class FactorizedLogistic(ClassifierMixin, BaseEstimator):
    """Fit genotype, conditioning and treatment separately.

The joint probability is the product of three marginal probabilities and
therefore assumes conditional independence of the factors given the features.
Its argmax equals the tuple of the three binary decisions. It is an exploratory,
design-informed baseline, not a claim that the factors are biologically independent.
    """

    def __init__(self, c: float = 1.0, random_state: int = 2026):
        self.c = c
        self.random_state = random_state

    def fit(self, x, y):
        labels = np.asarray(y, dtype=str)
        self.classes_ = np.unique(labels)
        parts = [label.split("-") for label in labels]
        if any(len(part) != 3 for part in parts):
            raise ValueError("Expected genotype-conditioning-treatment class codes")
        self.models_ = []
        for factor in range(3):
            target = np.asarray([part[factor] for part in parts])
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(C=self.c, max_iter=5000,
                                                     random_state=self.random_state))
            model.fit(x, target)
            self.models_.append(model)
        return self

    def predict_proba(self, x):
        marginals = [dict(zip(model.classes_, model.predict_proba(x).T, strict=True))
                     for model in self.models_]
        joint = np.column_stack([
            marginals[0][g] * marginals[1][b] * marginals[2][t]
            for g, b, t in (label.split("-") for label in self.classes_)
        ])
        return joint / joint.sum(axis=1, keepdims=True)

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]
