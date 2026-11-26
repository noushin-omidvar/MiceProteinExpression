#!/usr/bin/env python3
"""Separate checkpoint load, task fit, and warm prediction on one fixed mouse split."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from threadpoolctl import threadpool_limits

from mice_protein.benchmark import models as M
from mice_protein.data import aggregate_animals, load_data, protein_columns

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "foundation_models/results"
OUT = RES / "runtime_audit"


def timed(callback):
    start = time.perf_counter()
    result = callback()
    return result, time.perf_counter() - start


def main() -> None:
    raw = load_data(ROOT / "Data/Raw_Data/Data_Cortex_Nuclear.xls")
    mice = aggregate_animals(raw).set_index("AnimalID")
    features = protein_columns(raw)
    test = pd.read_csv(RES / "folds.csv").query("repeat == 0 and fold == 0").animal.to_numpy()
    train = mice.index.difference(test)
    x_train, y_train = mice.loc[train, features], mice.loc[train, "class"]
    x_test = mice.loc[test, features]
    rows = []
    for model_name in ("logistic", "tabpfn", "kumo"):
        for trial in range(4):
            estimator, construction = timed(lambda name=model_name: M.build(name))
            _, fit = timed(lambda fitted=estimator: fitted.fit(x_train, y_train))
            _, prediction = timed(lambda fitted=estimator: fitted.predict_proba(x_test))
            rows.append({"model": model_name, "trial": trial,
                         "construction_seconds": construction,
                         "fit_or_context_seconds": fit,
                         "prediction_seconds": prediction,
                         "warm": trial > 0})
            print(f"{model_name} trial {trial} done", flush=True)
    # Search time is distinct from inference and from fixed-C logistic fit.
    inner = StratifiedKFold(3, shuffle=True, random_state=20261129)
    search = GridSearchCV(M.build("logistic"), {"model__C": [0.01, 0.1, 1, 10, 100]},
                          scoring="balanced_accuracy", cv=inner, refit=True)
    _, tuning = timed(lambda: search.fit(x_train, y_train))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "timings.csv", index=False)
    (OUT / "provenance.json").write_text(json.dumps({
        "status": "descriptive_single_machine", "outer_split": "repeat 0 fold 0",
        "n_train": len(train), "n_test": len(test), "threads": 2,
        "platform": platform.platform(), "processor": platform.processor(),
        "tuned_logistic_search_seconds": tuning,
        "scope": "warm trials reuse OS and model caches; no monetary or energy cost estimate",
    }, indent=2))


if __name__ == "__main__":
    try:
        import torch
        torch.set_num_threads(2)
    except ImportError:
        pass
    with threadpool_limits(limits=2):
        main()
