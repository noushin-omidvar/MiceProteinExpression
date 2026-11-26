"""Model zoo for the foundation-model study (specifications fixed in the study protocol).

Conventional baselines impute (median) and scale inside each training fold. Tuned models
carry a small grid that is searched by 3-fold inner cross-validation. Foundation models
are used with default settings and no tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .provenance import pinned_tabpfn_v2_path, verify_kumo_small_cache

SEED = 2026
N_ESTIMATORS_TFM = 4


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    family: str  # "baseline" or "foundation"
    grid: dict = field(default_factory=dict)


SPECS: dict[str, ModelSpec] = {
    "dummy": ModelSpec("dummy", "Class prior", "baseline"),
    "logistic": ModelSpec("logistic", "Logistic (L2)", "baseline"),
    "elastic_net": ModelSpec("elastic_net", "Elastic net", "baseline",
                             {"model__C": [0.1, 1.0, 10.0]}),
    "svm": ModelSpec("svm", "RBF SVM", "baseline", {"model__C": [0.3, 1.0, 3.0]}),
    "random_forest": ModelSpec("random_forest", "Random forest", "baseline"),
    "gradient_boosting": ModelSpec("gradient_boosting", "Gradient boosting", "baseline",
                                   {"model__min_samples_leaf": [4, 8]}),
    "tabpfn": ModelSpec("tabpfn", "TabPFN v2", "foundation"),
    "kumo": ModelSpec("kumo", "Kumo Tabular Small", "foundation"),
}
BASELINES = [k for k, s in SPECS.items() if s.family == "baseline"]
FOUNDATION = [k for k, s in SPECS.items() if s.family == "foundation"]


def _pipeline(estimator, scale: bool = True) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline([*steps, ("model", estimator)])


def build(name: str, seed: int = SEED, probability: bool = True):
    """Return an unfitted estimator for a model name."""
    if name == "dummy":
        return _pipeline(DummyClassifier(strategy="prior"), scale=False)
    if name == "logistic":
        return _pipeline(LogisticRegression(C=1.0, max_iter=5000, random_state=seed))
    if name == "elastic_net":
        return _pipeline(LogisticRegression(solver="saga", penalty="elasticnet", l1_ratio=0.5,
                                            C=1.0, max_iter=20000, random_state=seed))
    if name == "svm":
        return _pipeline(SVC(C=1.0, probability=probability, random_state=seed))
    if name == "random_forest":
        return _pipeline(RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                                random_state=seed, n_jobs=2), scale=False)
    if name == "gradient_boosting":
        return _pipeline(HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
                                                        early_stopping=False,
                                                        random_state=seed), scale=False)
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(device="cpu", n_estimators=N_ESTIMATORS_TFM,
                                model_path=pinned_tabpfn_v2_path(), random_state=seed)
    if name == "kumo":
        return KumoClassifier(seed=seed)
    raise ValueError(f"Unknown model {name!r}")


@lru_cache(maxsize=1)
def _load_kumo():
    from sdm.models import KumoTabular

    model = KumoTabular(task="classification", size="small", device="cpu")
    verify_kumo_small_cache()
    return model


class KumoClassifier(ClassifierMixin, BaseEstimator):
    """scikit-learn wrapper around Kumo Tabular (in-context prediction, no training)."""

    def __init__(self, seed: int = SEED, n_estimators: int = N_ESTIMATORS_TFM):
        self.seed = seed
        self.n_estimators = n_estimators

    def fit(self, x, y):
        self.context_ = pd.DataFrame(x).reset_index(drop=True)
        self.classes_, self.labels_ = np.unique(np.asarray(y), return_inverse=True)
        return self

    def predict_proba(self, x):
        import sdm
        import torch

        torch.manual_seed(self.seed)
        x = pd.DataFrame(x).reset_index(drop=True).set_axis(self.context_.columns, axis=1)
        context = self.context_.assign(
            target=pd.Categorical(self.labels_, categories=np.arange(len(self.classes_))))
        numeric = dict.fromkeys(self.context_.columns, "numerical")
        table = sdm.TableTensor.from_pandas(df=context, stypes={**numeric, "target": "categorical"},
                                            device="cpu")
        query = sdm.TableTensor.from_pandas(df=x, stypes=numeric, device="cpu")
        with torch.inference_mode():
            probabilities = _load_kumo()(x_context=table.drop_columns("target"),
                                         y_context=table[:, "target"], x_query=query,
                                         num_estimators=self.n_estimators)
        columns = [str(i) for i in range(len(self.classes_))]
        return probabilities.to_pandas().loc[:, columns].to_numpy(dtype=float)

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]
