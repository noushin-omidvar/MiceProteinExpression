#!/usr/bin/env python3
"""Post hoc representation and missingness controls on fixed held-out mice."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from mice_protein.benchmark import models as M
from mice_protein.benchmark.harness import mouse_folds
from mice_protein.data import aggregate_animals, load_data, protein_columns

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "foundation_models"
CFG = yaml.safe_load((STUDY / "configs/sensitivity.yaml").read_text())
OUT = STUDY / "results/sensitivity"


def main() -> None:
    raw = load_data(ROOT / "Data/Raw_Data/Data_Cortex_Nuclear.xls")
    mice = aggregate_animals(raw).set_index("AnimalID")
    proteins = protein_columns(raw)
    arithmetic = mice[proteins]
    geometric = np.exp(np.log(raw[proteins].where(raw[proteins] > 0))
                       .groupby(raw.AnimalID).mean()).loc[mice.index]
    missingness = raw[proteins].isna().groupby(raw.AnimalID).mean().loc[mice.index]
    versions = {"arithmetic": arithmetic, "geometric": geometric,
                "missingness_only": missingness,
                "geometric_no_duplicate_sparse": geometric}
    folds = mouse_folds(mice["class"], CFG["repeats"], CFG["folds"], CFG["seed"])
    rows = []
    for version in CFG["representations"]:
        x = versions[version]
        for repeat, fold, train, test in folds:
            columns = proteins
            if version == "geometric_no_duplicate_sparse":
                train_values = x.iloc[train]
                duplicated = set(train_values.columns[train_values.T.duplicated()])
                columns = [c for c in proteins if c not in duplicated and
                           train_values[c].isna().sum() < 5]
            for model in CFG["models"]:
                estimator = M.build(model, seed=CFG["seed"] + repeat * 100 + fold)
                estimator.fit(x.iloc[train][columns], mice["class"].iloc[train])
                probability = estimator.predict_proba(x.iloc[test][columns])
                classes = np.asarray(estimator.classes_)
                predicted = classes[np.asarray(probability).argmax(axis=1)]
                for animal, truth, pred in zip(mice.index[test], mice["class"].iloc[test],
                                               predicted, strict=True):
                    rows.append({"version": version, "repeat": repeat, "fold": fold,
                                 "model": model, "animal": animal,
                                 "observed": truth, "predicted": pred,
                                 "n_features": len(columns)})
            print(f"{version}: repeat {repeat}, fold {fold} done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(rows)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    score_rows = []
    for (version, repeat, model), part in predictions.groupby(["version", "repeat", "model"]):
        score_rows.append({"version": version, "repeat": repeat, "model": model,
                           "balanced_accuracy": balanced_accuracy_score(
                               part.observed, part.predicted)})
    pd.DataFrame(score_rows).to_csv(OUT / "scores.csv", index=False)
    (OUT / "provenance.json").write_text(json.dumps({"config": CFG}, indent=2))


if __name__ == "__main__":
    try:
        import torch
        torch.set_num_threads(2)
    except ImportError:
        pass
    with threadpool_limits(limits=2):
        main()
