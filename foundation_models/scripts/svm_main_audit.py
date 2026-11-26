#!/usr/bin/env python3
"""Re-evaluate main SVM classification using SVC.predict with probability=False."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from threadpoolctl import threadpool_limits

from mice_protein.benchmark import models as M
from mice_protein.data import aggregate_animals, load_data, protein_columns

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "foundation_models/results"
OUT = RES / "svm_main_audit"


def main() -> None:
    raw = load_data(ROOT / "Data/Raw_Data/Data_Cortex_Nuclear.xls")
    mice = aggregate_animals(raw).set_index("AnimalID")
    features = protein_columns(raw)
    folds = pd.read_csv(RES / "folds.csv")
    tuned = pd.read_csv(RES / "tuned_baselines/scores_svm.csv")
    rows = []
    for (repeat, fold), block in folds.groupby(["repeat", "fold"]):
        test = block.animal.to_numpy()
        train = mice.index.difference(test)
        assert not set(train) & set(test)
        for version in ("original_grid", "post_hoc_wider_grid"):
            model = M.build("svm", probability=False)
            if version == "original_grid":
                inner = StratifiedKFold(3, shuffle=True, random_state=2026 + 100 * repeat + fold)
                search = GridSearchCV(model, M.SPECS["svm"].grid,
                                      scoring="balanced_accuracy", cv=inner,
                                      error_score="raise")
                search.fit(mice.loc[train, features], mice.loc[train, "class"])
                model = search.best_estimator_
            else:
                params = json.loads(tuned.loc[(tuned.repeat == repeat) &
                                              (tuned.fold == fold), "best_params"].iloc[0])
                model.set_params(**params)
                model.fit(mice.loc[train, features], mice.loc[train, "class"])
            predicted = model.predict(mice.loc[test, features])
            for animal, observed, guess in zip(test, mice.loc[test, "class"], predicted,
                                               strict=True):
                rows.append({"version": version, "repeat": repeat, "fold": fold,
                             "animal": animal, "observed": observed, "predicted": guess})
    predictions = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    scores = []
    for (version, repeat), part in predictions.groupby(["version", "repeat"]):
        scores.append({"version": version, "repeat": repeat,
                       "balanced_accuracy": balanced_accuracy_score(part.observed,
                                                                   part.predicted)})
    pd.DataFrame(scores).to_csv(OUT / "pooled_by_repeat.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
