#!/usr/bin/env python3
"""Decompose joint mouse-group predictions into experimental factors."""

import argparse
from pathlib import Path

import pandas as pd
from benchmark_foundation_models import KumoClassifier
from benchmark_mouse_models import ROOT, SEED, evaluate
from sklearn.metrics import balanced_accuracy_score, log_loss

from mice_protein.data import aggregate_animals, load_data, protein_columns
from mice_protein.models import build_model

FACTORS = ("Genotype", "Treatment", "Behavior")


def score(y, probabilities):
    labels = list(probabilities.columns)
    return {
        "balanced_accuracy": balanced_accuracy_score(y, probabilities.idxmax(axis=1)),
        "log_loss": log_loss(y, probabilities, labels=labels),
    }


def marginalize(joint, mapping):
    """Sum group probabilities over each level of an experimental factor."""
    return pd.DataFrame(
        {
            level: joint.loc[:, mapping.index[mapping == level]].sum(axis=1)
            for level in sorted(mapping.unique())
        },
        index=joint.index,
    )


def predict(x, y, strata, model):
    parts = []

    def save(repeat, fold, test, labels, probabilities):
        part = pd.DataFrame(probabilities, columns=labels, index=x.index[test])
        parts.append(part.assign(repeat=repeat, fold=fold, observed=y.iloc[test]))

    list(evaluate(x, y, strata, model, {}, prediction_sink=save))
    return pd.concat(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    raw = load_data(ROOT / "Data/Raw_Data/Data_Cortex_Nuclear.xls")
    features = protein_columns(raw)
    mice = aggregate_animals(raw).set_index("AnimalID")
    strata = mice["class"]
    design = mice.drop_duplicates("class").set_index("class")[list(FACTORS)].sort_index()
    missing = raw.set_index("AnimalID")[features].isna().groupby(level=0).mean()
    inputs = {"mean": mice[features], "missingness": missing.reindex(mice.index)}
    models = {"logistic": build_model("logistic", SEED)}
    if args.foundation:
        import torch
        from tabpfn import TabPFNClassifier

        torch.set_num_threads(2)
        models.update(
            kumo=KumoClassifier(),
            tabpfn=TabPFNClassifier(
                device="cpu",
                n_estimators=4,
                random_state=SEED,
                model_path=args.checkpoint or "auto",
            ),
        )
    rows, predictions = [], []
    for name, model in models.items():
        for representation, x in inputs.items():
            targets = ("class", *FACTORS) if representation == "mean" else ("class",)
            runs = {target: predict(x, mice[target], strata, model) for target in targets}
            for target, run in runs.items():
                predictions.append(
                    run.melt(
                        id_vars=["repeat", "fold", "observed"],
                        var_name="label",
                        value_name="probability",
                        ignore_index=False,
                    ).assign(model=name, representation=representation, target=target)
                )
            for repeat in range(5):
                joint = runs["class"].query("repeat == @repeat")[list(design.index)]
                base = {"model": name, "representation": representation, "repeat": repeat}
                rows.append(
                    base
                    | {
                        "target": "class",
                        "method": "joint",
                        **score(strata.loc[joint.index], joint),
                    }
                )
                if representation == "missingness":
                    continue
                recombined = pd.DataFrame(1.0, index=joint.index, columns=joint.columns)
                for factor in FACTORS:
                    y = mice.loc[joint.index, factor]
                    direct = (
                        runs[factor].query("repeat == @repeat").loc[joint.index, sorted(y.unique())]
                    )
                    rows.append(base | {"target": factor, "method": "direct", **score(y, direct)})
                    rows.append(
                        base
                        | {
                            "target": factor,
                            "method": "marginal",
                            **score(y, marginalize(joint, design[factor])),
                        }
                    )
                    recombined *= direct.loc[:, design[factor]].set_axis(joint.columns, axis=1)
                rows.append(
                    base
                    | {
                        "target": "class",
                        "method": "recombined",
                        **score(strata.loc[joint.index], recombined),
                    }
                )
    out = ROOT / "results/metrics"
    suffix = "" if args.foundation else "_core"
    pd.DataFrame(rows).to_csv(out / f"study{suffix}.csv", index=False)
    pd.concat(predictions).to_csv(out / f"study_predictions{suffix}.csv", index_label="animal_id")


if __name__ == "__main__":
    main()
