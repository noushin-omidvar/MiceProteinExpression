#!/usr/bin/env python3
"""Decompose joint mouse-group predictions into experimental factors (logistic regression).

Compares predicting the eight groups jointly with predicting each factor directly, using
repeated stratified cross-validation on one row per mouse.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss

from mice_protein.data import aggregate_animals, load_data, protein_columns
from mice_protein.evaluation import MOUSE_CV_SEED as SEED
from mice_protein.evaluation import mouse_level_cv
from mice_protein.models import build_model

ROOT = Path(__file__).resolve().parents[1]

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
    return mouse_level_cv(x, y, strata, model, seed=SEED)


def main():
    raw = load_data(ROOT / "Data/Raw_Data/Data_Cortex_Nuclear.xls")
    features = protein_columns(raw)
    mice = aggregate_animals(raw).set_index("AnimalID")
    strata = mice["class"]
    design = mice.drop_duplicates("class").set_index("class")[list(FACTORS)].sort_index()
    missing = raw.set_index("AnimalID")[features].isna().groupby(level=0).mean()
    inputs = {"mean": mice[features], "missingness": missing.reindex(mice.index)}
    models = {"logistic": build_model("logistic", SEED)}
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
    pd.DataFrame(rows).to_csv(out / "study.csv", index=False)
    pd.concat(predictions).to_csv(out / "study_predictions.csv", index_label="animal_id")


if __name__ == "__main__":
    main()
