#!/usr/bin/env python3
"""Contamination (memorization) test for tabular models on this public dataset.

Question: does a model's accuracy depend on recognising *this particular dataset*
(column names, column order, exact values) rather than on learning from the data?

Protocol (pre-specified in docs/analysis_decisions.md before any foundation-model run):
  * Inputs are mouse-mean protein levels, the same as the original benchmark.
  * Each model is scored on the original table and on three obfuscated copies
    (``memorization.obfuscate``: generic names, shuffled columns, random positive
    affine rescaling per protein). Information content is unchanged.
  * All versions use the identical 5 x 5-fold stratified splits.
  * Decision rule: evidence of reliance on memorised identity if the mean
    balanced-accuracy drop (original - obfuscated) exceeds 0.05 AND every
    obfuscation key shows a drop. Otherwise: no evidence of memorisation.

Run in an environment with the model installed, for example::

    .venv-foundation/bin/python scripts/memorization_test.py tabpfn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_mouse_models import SEED, evaluate

from mice_protein.data import aggregate_animals, load_data, protein_columns
from mice_protein.memorization import obfuscate

THRESHOLD = 0.05
N_KEYS = 3


def build(name: str):
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(device="cpu", n_estimators=4, random_state=SEED)
    if name == "kumo":
        from benchmark_foundation_models import KumoClassifier

        return KumoClassifier()
    if name == "logistic":
        from mice_protein.models import build_model

        return build_model("logistic", SEED)
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=["tabpfn", "kumo", "logistic"])
    args = parser.parse_args()
    try:
        import torch

        torch.set_num_threads(2)
    except ImportError:
        pass
    raw = load_data(ROOT / "Data/Raw_Data/Data_Cortex_Nuclear.xls")
    mice = aggregate_animals(raw)
    features = mice[protein_columns(raw)]
    y = mice["class"]
    versions = {"original": features}
    for key in range(N_KEYS):
        versions[f"obfuscated_{key}"], _ = obfuscate(features, np.random.default_rng(SEED + key))
    rows = []
    for version, x in versions.items():
        for result in evaluate(x, y, y, build(args.model), {}):
            rows.append({"model": args.model, "version": version, **result})
            print(version, result["repeat"], round(result["balanced_accuracy"], 3), flush=True)
    table = pd.DataFrame(rows)
    out = ROOT / f"results/metrics/memorization_{args.model}.csv"
    table.to_csv(out, index=False)

    wide = table.pivot(index="repeat", columns="version", values="balanced_accuracy")
    drops = wide[[c for c in wide if c != "original"]].rsub(wide["original"], axis=0)
    mean_drop = float(drops.to_numpy().mean())
    every_key_drops = bool((drops.mean() > 0).all())
    verdict = ("evidence of reliance on dataset identity"
               if mean_drop > THRESHOLD and every_key_drops else "no evidence of memorisation")
    print(f"\n{args.model}: original {wide['original'].mean():.3f}, obfuscated "
          f"{wide.drop(columns='original').to_numpy().mean():.3f}, mean drop {mean_drop:+.3f} "
          f"-> {verdict}\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
