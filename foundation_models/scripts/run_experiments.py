#!/usr/bin/env python3
"""Run the pre-specified experiments of the foundation-model study.

Each subcommand implements one section of ``foundation_models/docs/study_protocol.md``:

    accuracy      4.1/4.2/4.6  repeated CV on mouse means, all targets and models
    learning      4.3          learning curves (k training mice per group)
    permutation   4.4          label-permutation null
    leakage       4.4          row-level random split vs mouse-held-out split
    memorization  4.5          original vs obfuscated features

Run with the environment that has TabPFN and Kumo installed, for example::

    PYTHONPATH=src .venv-foundation/bin/python foundation_models/scripts/run_experiments.py all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from threadpoolctl import threadpool_limits

from mice_protein.benchmark import harness as H
from mice_protein.benchmark import models as M
from mice_protein.benchmark.provenance import checkpoint_inventory
from mice_protein.data import aggregate_animals, load_data, protein_columns
from mice_protein.memorization import obfuscate

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "foundation_models"
CFG = yaml.safe_load((STUDY / "configs/study.yaml").read_text())
OUT = STUDY / "results"
TARGETS = {"class": "Eight groups", "Genotype": "Genotype", "Treatment": "Treatment",
           "Behavior": "Conditioning"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load():
    raw = load_data(ROOT / CFG["raw_data"])
    mice = aggregate_animals(raw).set_index("AnimalID")
    return raw, mice, protein_columns(raw)


def run_accuracy(mice, features, models):
    folds = H.mouse_folds(mice["class"], CFG["n_repeats"], CFG["n_splits"], CFG["seed"])
    pd.DataFrame([{"repeat": r, "fold": f, "animal": a}
                  for r, f, _, test in folds for a in mice.index[test]]).to_csv(
        OUT / "folds.csv", index=False)
    for target in TARGETS:
        rows, preds = [], []
        for name in models:
            log(f"accuracy · {target} · {name}")
            fold_metrics, predictions = H.cross_validate(name, mice[features], mice[target],
                                                         mice["class"], folds, CFG["seed"])
            rows.append(fold_metrics.assign(target=target))
            preds.append(predictions)
        tag = target.lower()
        _merge_write(pd.concat(rows), OUT / f"accuracy_folds_{tag}.csv", ["model"])
        _merge_write(pd.concat(preds).reset_index(names="animal"),
                     OUT / f"predictions_{tag}.csv.gz", ["model"])


def run_learning(mice, features, models):
    rng = np.random.default_rng(CFG["seed"])
    groups = mice.groupby("class").groups
    rows = []
    for k in CFG["learning_curve_k"]:
        for draw in range(CFG["learning_curve_draws"]):
            train = np.concatenate([rng.choice(np.asarray(idx), k, replace=False)
                                    for idx in groups.values()])
            test = mice.index.difference(train)
            for name in models:
                est = M.build(name, CFG["seed"], probability=False)
                start = time.perf_counter()
                est.fit(mice.loc[train, features], mice.loc[train, "class"])
                predicted = est.predict(mice.loc[test, features])
                seconds = time.perf_counter() - start
                from sklearn.metrics import balanced_accuracy_score

                rows.append({"model": name, "k": k, "draw": draw, "n_train": len(train),
                             "seconds": seconds,
                             "balanced_accuracy": balanced_accuracy_score(
                                 mice.loc[test, "class"], predicted)})
        log(f"learning · k={k} done")
    _merge_write(pd.DataFrame(rows), OUT / "learning_curve.csv", ["model"])


def run_permutation(mice, features, models):
    rows = []
    for p in range(CFG["n_permutations"]):
        rng = np.random.default_rng(CFG["seed"] + 1000 + p)
        y = pd.Series(rng.permutation(mice["class"].to_numpy()), index=mice.index)
        folds = H.mouse_folds(y, 1, CFG["n_splits"], CFG["seed"] + 1000 + p)
        for name in models:
            fm, _ = H.cross_validate(name, mice[features], y, y, folds, CFG["seed"])
            rows.append({"model": name, "permutation": p,
                         "balanced_accuracy": fm.balanced_accuracy.mean()})
        log(f"permutation {p} done")
    _merge_write(pd.DataFrame(rows), OUT / "permutation_null.csv", ["model"])


def run_leakage(raw, features, models):
    x = raw[features].reset_index(drop=True)
    y = raw["class"].reset_index(drop=True)
    groups = raw["AnimalID"].reset_index(drop=True)
    rows = []
    for design, grouped in (("random_rows", False), ("mouse_held_out", True)):
        folds = H.row_level_folds(groups, y, grouped, CFG["seed"])
        for name in models:
            log(f"leakage · {design} · {name}")
            fm, _ = H.cross_validate(name, x, y, y, folds, CFG["seed"])
            rows.append(fm.assign(design=design))
    _merge_write(pd.concat(rows), OUT / "leakage.csv", ["model"])


def run_memorization(mice, features, models):
    folds = H.mouse_folds(mice["class"], CFG["n_repeats"], CFG["n_splits"], CFG["seed"])
    rows = []
    versions = {"original": mice[features]}
    for key in range(CFG["obfuscation_keys"]):
        versions[f"obfuscated_{key}"], _ = obfuscate(mice[features],
                                                     np.random.default_rng(CFG["seed"] + key))
    for name in models:
        for version, x in versions.items():
            log(f"memorization · {name} · {version}")
            _fm, preds = H.cross_validate(name, x, mice["class"], mice["class"], folds, CFG["seed"])
            pooled = H.pooled_scores(preds)
            rows.append(pooled.assign(version=version))
    _merge_write(pd.concat(rows), OUT / "memorization.csv", ["model"])


def _merge_write(new: pd.DataFrame, path: Path, keys: list[str]) -> None:
    """Write results, replacing rows for the models just run and keeping the others."""
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["model"].isin(new["model"].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def _run_identity() -> dict:
    """Identity checked before any partial rerun can replace saved model rows."""
    raw_path = ROOT / CFG["raw_data"]
    identity = {"config": CFG, "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                "python": sys.version.split()[0], "scikit_learn": sklearn.__version__,
                "checkpoint_inventory": checkpoint_inventory()}
    for pkg in ("tabpfn", "sdm", "torch"):
        try:
            identity[pkg] = __import__(pkg).__version__
        except (ImportError, AttributeError):
            identity[pkg] = "unavailable"
    identity["run_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def _check_compatible(identity: dict) -> None:
    path = OUT / "provenance.json"
    if not path.exists():
        if any(OUT.glob("*.csv*")):
            raise RuntimeError("Saved results exist without provenance; refusing to merge")
        return
    old = json.loads(path.read_text())
    for key in ("config", "python", "scikit_learn", "tabpfn", "sdm", "torch"):
        if old.get(key) != identity[key]:
            raise RuntimeError(f"Incompatible existing results: {key} changed")
    prior_checkpoints = old.get("checkpoint_inventory", old.get(
        "checkpoint_inventory_recorded_after_run"))
    if prior_checkpoints is not None and prior_checkpoints != identity["checkpoint_inventory"]:
        raise RuntimeError("Incompatible existing results: checkpoint identity changed")
    if "raw_sha256" in old and old["raw_sha256"] != identity["raw_sha256"]:
        raise RuntimeError("Incompatible existing results: raw data changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["accuracy", "learning", "permutation", "leakage",
                                               "memorization", "all"])
    parser.add_argument("--models", nargs="+", help="Subset of models (default: per protocol)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    identity = _run_identity()
    _check_compatible(identity)
    try:
        import torch

        torch.set_num_threads(CFG["threads"])
    except ImportError:
        pass
    raw, mice, features = load()
    plan = {
        "accuracy": lambda m: run_accuracy(mice, features, m or list(M.SPECS)),
        "learning": lambda m: run_learning(mice, features, m or CFG["learning_curve_models"]),
        "permutation": lambda m: run_permutation(mice, features, m or list(M.SPECS)),
        "leakage": lambda m: run_leakage(raw, features, m or CFG["leakage_models"]),
        "memorization": lambda m: run_memorization(mice, features, m or list(M.SPECS)),
    }
    todo = list(plan) if args.experiment == "all" else [args.experiment]
    with threadpool_limits(limits=CFG["threads"]):
        for name in todo:
            plan[name](args.models)
    provenance = {**identity, "platform": platform.platform(),
                  "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
