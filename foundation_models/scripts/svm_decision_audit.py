#!/usr/bin/env python3
"""Audit SVC classification decisions against saved probability-argmax decisions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from mice_protein.benchmark import models as M
from mice_protein.data import aggregate_animals, load_data, protein_columns

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "foundation_models/results"
OUT = RES / "svm_decision_audit"


def main() -> None:
    raw = load_data(ROOT / "Data/Raw_Data/Data_Cortex_Nuclear.xls")
    mice = aggregate_animals(raw).set_index("AnimalID")
    features = protein_columns(raw)
    plan = pd.read_csv(RES / "matched_holdout/splits.csv", dtype={"size": str})
    old = pd.read_csv(RES / "matched_holdout/predictions.csv.gz", dtype={"size": str})
    old = old[old.model == "svm"]
    classes = np.unique(mice["class"])
    pcols = [f"p_{c}" for c in classes]
    old = old.assign(probability_decision=classes[old[pcols].to_numpy().argmax(axis=1)])
    rows = []
    for key, part in plan.groupby(["repeat", "fold", "draw", "size"], sort=False):
        train = part.loc[part.role == "train", "animal"].to_numpy()
        test = part.loc[part.role == "test", "animal"].to_numpy()
        assert not set(train) & set(test)
        model = M.build("svm", probability=False)
        model.fit(mice.loc[train, features], mice.loc[train, "class"])
        predicted = model.predict(mice.loc[test, features])
        for animal, actual, decision in zip(test, mice.loc[test, "class"], predicted,
                                            strict=True):
            rows.append(dict(zip(["repeat", "fold", "draw", "size"], key, strict=True),
                             animal=animal, observed=actual, svm_decision=decision,
                             n_train=len(train)))
    audited = pd.DataFrame(rows).merge(
        old[["repeat", "fold", "draw", "size", "animal", "probability_decision"]],
        on=["repeat", "fold", "draw", "size", "animal"], validate="one_to_one")
    OUT.mkdir(parents=True, exist_ok=True)
    audited.to_csv(OUT / "predictions.csv", index=False)
    scores = []
    for key, part in audited.groupby(["repeat", "fold", "draw", "size"], sort=False):
        scores.append(dict(zip(["repeat", "fold", "draw", "size"], key, strict=True),
                           n_train=int(part.n_train.iloc[0]),
                           svm_decision_ba=balanced_accuracy_score(part.observed,
                                                                   part.svm_decision),
                           probability_argmax_ba=balanced_accuracy_score(
                               part.observed, part.probability_decision),
                           disagreement_count=int((part.svm_decision !=
                                                   part.probability_decision).sum())))
    pd.DataFrame(scores).to_csv(OUT / "scores.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
