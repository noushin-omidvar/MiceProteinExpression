#!/usr/bin/env python3
"""Post hoc matched-holdout learning curves; see configs/matched_holdout.yaml.

The original study's learning curves used changing test sets. Here each seven-fold
outer split fixes the held-out mice while nested training subsets grow within its
training partition. Subset selection uses labels only for stratification.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from mice_protein.benchmark import models as M
from mice_protein.benchmark.provenance import checkpoint_inventory
from mice_protein.data import aggregate_animals, load_data, protein_columns

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "foundation_models"
CFG = yaml.safe_load((BASE / "configs/matched_holdout.yaml").read_text())
OUT = BASE / "results/matched_holdout"


def make_plan(mice: pd.DataFrame) -> pd.DataFrame:
    labels = mice["class"].to_numpy()
    ids = mice.index.to_numpy()
    classes = np.unique(labels)
    rows = []
    for repeat in range(CFG["outer_repeats"]):
        splitter = StratifiedKFold(
            n_splits=CFG["outer_folds"], shuffle=True,
            random_state=CFG["seed"] + repeat,
        )
        for fold, (train, test) in enumerate(splitter.split(ids, labels)):
            assert not set(ids[train]) & set(ids[test])
            assert set(labels[test]) == set(classes)
            for draw in range(CFG["subset_draws"]):
                rng = np.random.default_rng(CFG["seed"] + 1000 * repeat + 100 * fold + draw)
                order = {c: rng.permutation(train[labels[train] == c]) for c in classes}
                for size in [*CFG["per_class_sizes"], "full"]:
                    if size == "full" and draw > 0:
                        continue  # identical full training set within an outer fold
                    selected = train if size == "full" else np.concatenate(
                        [order[c][:size] for c in classes]
                    )
                    assert len(selected) == len(set(selected))
                    assert set(selected).issubset(set(train))
                    for position in selected:
                        rows.append((repeat, fold, draw, str(size), "train", ids[position]))
                    for position in test:
                        rows.append((repeat, fold, draw, str(size), "test", ids[position]))
    return pd.DataFrame(rows, columns=["repeat", "fold", "draw", "size", "role", "animal"])


def run(models: list[str], pilot: bool) -> None:
    if not set(models) <= set(CFG["models"]):
        raise ValueError("Models must be listed in the frozen configuration")
    raw = load_data(ROOT / "Data/Raw_Data/Data_Cortex_Nuclear.xls")
    mice = aggregate_animals(raw).set_index("AnimalID")
    features = protein_columns(raw)
    plan = make_plan(mice)
    scenarios = list(plan.groupby(["repeat", "fold", "draw", "size"], sort=False))
    if pilot:
        scenarios = scenarios[:1]
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        plan.to_csv(OUT / "splits.csv", index=False)
    results = []
    for number, (key, partition) in enumerate(scenarios, 1):
        repeat, fold, draw, size = key
        train = partition.loc[partition.role == "train", "animal"].to_numpy()
        test = partition.loc[partition.role == "test", "animal"].to_numpy()
        assert len(train) == len(set(train)) and not set(train) & set(test)
        y_train = mice.loc[train, "class"]
        y_test = mice.loc[test, "class"]
        classes = np.unique(y_train)
        assert len(classes) == 8
        for name in models:
            estimator = M.build(name, seed=CFG["seed"] + repeat * 100 + fold)
            estimator.fit(mice.loc[train, features], y_train)
            proba = np.asarray(estimator.predict_proba(mice.loc[test, features]), dtype=float)
            model_classes = np.asarray(estimator.classes_)
            if not np.array_equal(model_classes, classes):
                raise ValueError(f"Probability columns misaligned for {name}: {model_classes}")
            proba /= proba.sum(axis=1, keepdims=True)
            for animal, observed, values in zip(test, y_test, proba, strict=True):
                results.append({"repeat": repeat, "fold": fold, "draw": draw,
                                "size": size, "n_train": len(train), "model": name,
                                "animal": animal, "observed": observed,
                                **{f"p_{c}": v for c, v in zip(classes, values, strict=True)}})
        if number % 10 == 0 or pilot:
            print(f"{number}/{len(scenarios)} scenarios complete", flush=True)
    if pilot:
        print("Pilot passed: every model fitted and returned aligned probabilities.")
        return
    predictions = pd.DataFrame(results)
    predictions.to_csv(OUT / "predictions.csv.gz", index=False)
    scores = []
    pcols = [f"p_{c}" for c in np.unique(mice["class"])]
    for key, part in predictions.groupby(["repeat", "fold", "draw", "size", "model"], sort=False):
        probability = part[pcols].to_numpy()
        predicted = np.unique(mice["class"])[probability.argmax(axis=1)]
        scores.append(dict(zip(["repeat", "fold", "draw", "size", "model"], key, strict=True),
                           n_train=int(part.n_train.iloc[0]), n_test=len(part),
                           balanced_accuracy=balanced_accuracy_score(part.observed, predicted),
                           log_loss=log_loss(part.observed, probability,
                                             labels=np.unique(mice["class"]))))
    pd.DataFrame(scores).to_csv(OUT / "scores.csv", index=False)
    provenance = {"config": CFG, "python": sys.version.split()[0],
                  "scikit_learn": sklearn.__version__, "platform": platform.platform()}
    for module in ("tabpfn", "sdm", "torch"):
        package = __import__(module)
        provenance[module] = getattr(package, "__version__", "unknown")
    provenance["checkpoint_inventory"] = checkpoint_inventory()
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--config", default="matched_holdout.yaml")
    parser.add_argument("--output", default="matched_holdout")
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    CFG = yaml.safe_load((BASE / "configs" / args.config).read_text())
    OUT = BASE / "results" / args.output
    try:
        import torch
        torch.set_num_threads(2)
    except ImportError:
        pass
    with threadpool_limits(limits=2):
        run(args.models or CFG["models"], args.pilot)
