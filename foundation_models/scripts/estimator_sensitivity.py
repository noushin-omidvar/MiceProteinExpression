#!/usr/bin/env python3
"""Exploratory sensitivity: foundation models at their package-default ensemble sizes.

The pre-specified protocol fixed 4 inference estimators for both foundation models. TabPFN
2.2.1 defaults to 8, and Kumo Tabular resolves its own default when ``num_estimators`` is
None. This script re-runs the eight-group comparison with package defaults on the identical
outer folds, so any change is attributable to ensemble size. It is post hoc (2026-11-25) and
does not replace the pre-specified results.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from mice_protein.benchmark import harness as H
from mice_protein.benchmark import models as M
from mice_protein.benchmark.provenance import checkpoint_inventory, pinned_tabpfn_v2_path
from mice_protein.data import aggregate_animals, load_data, protein_columns

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "foundation_models/configs/study.yaml").read_text())
OUT = ROOT / "foundation_models/results/estimator_sensitivity"


def build(name: str):
    if name == "tabpfn_default":
        from tabpfn import TabPFNClassifier

        # n_estimators omitted: the package default (8 in tabpfn 2.2.1)
        return TabPFNClassifier(device="cpu", model_path=pinned_tabpfn_v2_path(),
                                random_state=CFG["seed"])
    if name == "kumo_default":
        return M.KumoClassifier(seed=CFG["seed"], n_estimators=None)
    raise ValueError(name)


def main() -> None:
    import torch

    torch.set_num_threads(CFG["threads"])
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_data(ROOT / CFG["raw_data"])
    mice = aggregate_animals(raw).set_index("AnimalID")
    x, y = mice[protein_columns(raw)], mice["class"]
    folds = H.mouse_folds(y, CFG["n_repeats"], CFG["n_splits"], CFG["seed"])
    classes = np.unique(y)
    parts = []
    with threadpool_limits(limits=CFG["threads"]):
        for name in ("tabpfn_default", "kumo_default"):
            for r, f, train, test in folds:
                start = time.perf_counter()
                model = build(name).fit(x.iloc[train], y.iloc[train])
                proba = np.asarray(model.predict_proba(x.iloc[test]), dtype=float)
                seconds = time.perf_counter() - start
                frame = pd.DataFrame(proba / proba.sum(axis=1, keepdims=True), columns=classes,
                                     index=x.index[test])
                parts.append(frame.assign(model=name, repeat=r, fold=f, observed=y.iloc[test],
                                          seconds=seconds))
            print(name, "done", flush=True)
    preds = pd.concat(parts)
    preds.to_csv(OUT / "predictions.csv.gz", index_label="animal")
    pooled = H.pooled_scores(preds.drop(columns="seconds"))
    pooled.to_csv(OUT / "scores.csv", index=False)
    summary = pooled.groupby("model")[["balanced_accuracy", "log_loss"]].mean()
    original = pd.read_csv(ROOT / "foundation_models/results/tables/performance_class.csv",
                           index_col=0)
    summary["original_4_estimators_ba"] = [original.loc[m.split("_")[0], "balanced_accuracy_mean"]
                                           for m in summary.index]
    summary.to_csv(OUT / "summary.csv")
    import platform

    import sklearn
    import tabpfn

    (OUT / "provenance.json").write_text(json.dumps(
        {"status": "post_hoc_exploratory", "date": "2026-11-25",
         "python": platform.python_version(), "scikit_learn": sklearn.__version__,
         "tabpfn": tabpfn.__version__, "torch": torch.__version__,
         "checkpoints": checkpoint_inventory()}, indent=2))
    print(summary.round(3))


if __name__ == "__main__":
    main()
