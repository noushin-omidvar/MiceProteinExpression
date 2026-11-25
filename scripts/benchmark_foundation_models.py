#!/usr/bin/env python3
"""Score local foundation models on the shared mouse-level benchmark folds."""

import argparse
from functools import lru_cache

import numpy as np
import pandas as pd
from benchmark_mouse_models import ROOT, SEED, evaluate
from sklearn.base import BaseEstimator, ClassifierMixin

from mice_protein.data import aggregate_animals, load_data, protein_columns


@lru_cache(maxsize=1)
def load_kumo():
    from sdm.models import KumoTabular

    return KumoTabular(task="classification", size="small", device="cpu")


class KumoClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, x, y):
        self.context_ = x.reset_index(drop=True)
        self.classes_, self.labels_ = np.unique(y, return_inverse=True)
        return self

    def predict_proba(self, x):
        import sdm
        import torch

        torch.manual_seed(SEED)
        context = self.context_.assign(
            target=pd.Categorical(self.labels_, categories=np.arange(len(self.classes_)))
        )
        table = sdm.TableTensor.from_pandas(
            df=context,
            stypes={**dict.fromkeys(x.columns, "numerical"), "target": "categorical"},
            device="cpu",
        )
        query = sdm.TableTensor.from_pandas(
            df=x.reset_index(drop=True),
            stypes=dict.fromkeys(x.columns, "numerical"),
            device="cpu",
        )
        with torch.inference_mode():
            probabilities = load_kumo()(
                x_context=table.drop_columns("target"),
                y_context=table[:, "target"],
                x_query=query,
                num_estimators=4,
            )
        columns = [str(index) for index in range(len(self.classes_))]
        return probabilities.to_pandas().loc[:, columns].to_numpy(dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["tabpfn", "kumo"])
    parser.add_argument("--checkpoint", help="Local TabPFN v2 checkpoint")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    if args.model == "tabpfn":
        from tabpfn import TabPFNClassifier

        model = TabPFNClassifier(
            device="cpu", model_path=args.checkpoint or "auto", n_estimators=4, random_state=SEED
        )
    else:
        model = KumoClassifier()
    raw = load_data(ROOT / "Data/Raw_Data/Data_Cortex_Nuclear.xls")
    mice = aggregate_animals(raw)
    rows = list(evaluate(mice[protein_columns(raw)], mice["class"], mice["class"], model, {}))
    results = pd.DataFrame(rows).assign(target="class", model=f"{args.model}_mean")
    results.to_csv(ROOT / f"results/metrics/{args.model}_mouse_benchmark.csv", index=False)
    print(results[["balanced_accuracy", "log_loss"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
