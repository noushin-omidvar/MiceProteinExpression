#!/usr/bin/env python3
"""Post hoc nested tuning of available conventional baselines on saved outer folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from threadpoolctl import threadpool_limits

from mice_protein.benchmark import models as M
from mice_protein.data import aggregate_animals, load_data, protein_columns

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "foundation_models"
CFG = yaml.safe_load((STUDY / "configs/tuned_baselines.yaml").read_text())
OUT = STUDY / "results/tuned_baselines"


def run(models: list[str], pilot: bool) -> None:
    raw = load_data(ROOT / "Data/Raw_Data/Data_Cortex_Nuclear.xls")
    mice = aggregate_animals(raw).set_index("AnimalID")
    features = protein_columns(raw)
    saved = pd.read_csv(STUDY / "results/folds.csv")
    classes = np.unique(mice["class"])
    scenarios = list(saved.groupby(["repeat", "fold"], sort=True))
    if pilot:
        scenarios = scenarios[:1]
    else:
        OUT.mkdir(parents=True, exist_ok=True)
    for model in models:
        if model not in CFG["models"]:
            raise ValueError(f"Unknown configured model: {model}")
        predictions, scores = [], []
        for repeat_fold, part in scenarios:
            repeat, fold = repeat_fold
            test = part.animal.to_numpy()
            train = mice.index.difference(test)
            assert not set(train) & set(test)
            estimator = M.build(model, CFG["seed"] + repeat * 100 + fold)
            inner = StratifiedKFold(n_splits=CFG["inner_folds"], shuffle=True,
                                    random_state=CFG["seed"] + repeat * 100 + fold)
            search = GridSearchCV(estimator, CFG["models"][model], scoring=CFG["scoring"],
                                  cv=inner, n_jobs=1, error_score="raise", refit=True)
            search.fit(mice.loc[train, features], mice.loc[train, "class"])
            proba = np.asarray(search.predict_proba(mice.loc[test, features]), dtype=float)
            if not np.array_equal(search.classes_, classes):
                raise ValueError("Probability columns are misaligned")
            for animal, observed, probs in zip(test, mice.loc[test, "class"], proba, strict=True):
                predictions.append({"model": model, "repeat": repeat, "fold": fold,
                                    "animal": animal, "observed": observed,
                                    **{f"p_{c}": p for c, p in zip(classes, probs, strict=True)}})
            scores.append({"model": model, "repeat": repeat, "fold": fold,
                           "balanced_accuracy": balanced_accuracy_score(
                               mice.loc[test, "class"], classes[proba.argmax(axis=1)]),
                           "log_loss": log_loss(mice.loc[test, "class"], proba, labels=classes),
                           "best_params": json.dumps(search.best_params_, sort_keys=True)})
            print(f"{model}: repeat {repeat}, fold {fold} done", flush=True)
        if not pilot:
            pd.DataFrame(predictions).to_csv(OUT / f"predictions_{model}.csv.gz", index=False)
            pd.DataFrame(scores).to_csv(OUT / f"scores_{model}.csv", index=False)
    if pilot:
        print("Pilot passed: inner tuning and class-aligned probabilities work.")
    else:
        import platform

        import numpy
        import sklearn

        (OUT / "provenance.json").write_text(json.dumps(
            {"config": CFG, "outer_folds": "foundation_models/results/folds.csv",
             "python": platform.python_version(), "scikit_learn": sklearn.__version__,
             "numpy": numpy.__version__, "pandas": pd.__version__}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(CFG["models"]))
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.models, args.pilot)
