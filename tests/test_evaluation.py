import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from mice_protein.evaluation import repeated_cv_predictions
from mice_protein.models import build_model


def test_grouped_split_has_no_overlap():
    groups = np.repeat(np.arange(24), 2)
    y = np.repeat(np.tile(["a", "b", "c"], 8), 2)
    X = pd.DataFrame({"x": np.arange(48.0)})
    cv = StratifiedGroupKFold(3, shuffle=True, random_state=7)
    for train, test in cv.split(X, y, groups):
        assert set(groups[train]).isdisjoint(groups[test])


def test_pipeline_imputes_inside_fit_and_is_reproducible():
    groups = np.repeat(np.arange(24), 2)
    y = np.repeat(np.tile(["a", "b", "c"], 8), 2)
    X = pd.DataFrame({"x": np.arange(48.0), "missing": [np.nan, 1.0] * 24})
    kwargs = {"grouped": True, "repeats": 1, "splits": 3, "seed": 11}
    a = repeated_cv_predictions(X, y, groups, build_model("logistic", 11), **kwargs)
    b = repeated_cv_predictions(X, y, groups, build_model("logistic", 11), **kwargs)
    assert [r["y_pred"] for r in a] == [r["y_pred"] for r in b]
    assert "impute" in build_model("logistic").named_steps
