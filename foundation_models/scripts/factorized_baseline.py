#!/usr/bin/env python3
"""Nested-CV factorized logistic baseline on the saved mouse-level outer splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from threadpoolctl import threadpool_limits

from mice_protein.benchmark.factorized import FactorizedLogistic
from mice_protein.data import aggregate_animals, load_data, protein_columns

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "foundation_models"
CFG = yaml.safe_load((STUDY / "configs/factorized.yaml").read_text())
OUT = STUDY / "results/factorized"


def main() -> None:
    raw = load_data(ROOT / "Data/Raw_Data/Data_Cortex_Nuclear.xls")
    mice = aggregate_animals(raw).set_index("AnimalID")
    features = protein_columns(raw)
    classes = np.unique(mice["class"])
    folds = pd.read_csv(ROOT / CFG["outer_splits"])
    predictions, scores = [], []
    for (repeat, fold), block in folds.groupby(["repeat", "fold"]):
        test = block.animal.to_numpy()
        train = mice.index.difference(test)
        assert not set(train) & set(test)
        inner = StratifiedKFold(n_splits=CFG["inner_folds"], shuffle=True,
                                random_state=CFG["seed"] + repeat * 100 + fold)
        search = GridSearchCV(FactorizedLogistic(random_state=CFG["seed"]),
                              {"c": CFG["c_grid"]}, scoring=CFG["scoring"],
                              cv=inner, refit=True, error_score="raise")
        search.fit(mice.loc[train, features], mice.loc[train, "class"])
        probability = search.predict_proba(mice.loc[test, features])
        decision = search.predict(mice.loc[test, features])
        if not np.array_equal(search.classes_, classes):
            raise ValueError("Probability columns are not aligned")
        for animal, observed, predicted, p in zip(test, mice.loc[test, "class"], decision,
                                                  probability, strict=True):
            predictions.append({"model": "factorized_logistic", "repeat": repeat,
                                "fold": fold, "animal": animal, "observed": observed,
                                "predicted": predicted,
                                **{f"p_{c}": v for c, v in zip(classes, p, strict=True)}})
        scores.append({"repeat": repeat, "fold": fold, "n_train": len(train),
                       "n_test": len(test), "balanced_accuracy": balanced_accuracy_score(
                           mice.loc[test, "class"], decision),
                       "log_loss": log_loss(mice.loc[test, "class"], probability,
                                             labels=classes), "best_c": search.best_params_["c"]})
        print(f"repeat {repeat}, fold {fold} complete", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(OUT / "predictions.csv.gz", index=False)
    pd.DataFrame(scores).to_csv(OUT / "scores.csv", index=False)
    (OUT / "provenance.json").write_text(json.dumps({"config": CFG}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
