#!/usr/bin/env python3
"""Estimate exploratory factorial contrasts using one protein mean per mouse."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from mice_protein.data import aggregate_animals, load_data, protein_columns

ROOT = Path(__file__).resolve().parents[1]
TERMS = ("Genotype", "Treatment", "Behavior", "Genotype×Treatment", "Genotype×Behavior")


def main():
    raw = load_data(ROOT / "Data/Raw_Data/Data_Cortex_Nuclear.xls")
    mice = aggregate_animals(raw)
    features = raw[protein_columns(raw)]
    # Keep the first copy of an exact duplicate in the hypothesis family.
    unique_features = features.columns[~features.T.duplicated()]
    design = pd.DataFrame(
        {
            "Genotype": (mice.Genotype == "Ts65Dn").astype(float) - 0.5,
            "Treatment": (mice.Treatment == "Memantine").astype(float) - 0.5,
            "Behavior": (mice.Behavior == "C/S").astype(float) - 0.5,
        }
    )
    for first, second in (
        ("Genotype", "Treatment"),
        ("Genotype", "Behavior"),
        ("Treatment", "Behavior"),
    ):
        design[f"{first}×{second}"] = design[first] * design[second]
    design["three_way"] = design.Genotype * design.Treatment * design.Behavior
    design = sm.add_constant(design)
    rows, variability = [], []
    for protein in unique_features:
        observed = mice[protein].notna()
        cell_counts = mice.loc[observed].groupby("class").size()
        values = mice.loc[observed, protein]
        if len(cell_counts) == 8 and cell_counts.min() >= 3 and values.std() > 0:
            result = sm.OLS((values - values.mean()) / values.std(), design.loc[observed]).fit(
                cov_type="HC3", use_t=True
            )
            intervals = result.conf_int()
            for term in TERMS:
                rows.append(
                    {
                        "protein": protein,
                        "term": term,
                        "effect_sd": result.params[term],
                        "lower_95": intervals.loc[term, 0],
                        "upper_95": intervals.loc[term, 1],
                        "p": result.pvalues[term],
                        "n_mice": observed.sum(),
                        "minimum_cell_n": cell_counts.min(),
                    }
                )
        centered = raw[protein] - raw.groupby("AnimalID")[protein].transform("mean")
        variability.append(
            {
                "protein": protein,
                "within_mouse_rms": np.sqrt(centered.pow(2).mean()),
                "between_mouse_sd": values.std(),
                "n_mice": observed.sum(),
            }
        )
    effects = pd.DataFrame(rows)
    effects["q"] = multipletests(effects.p, method="fdr_bh")[1]
    effects.sort_values("q").to_csv(ROOT / "results/tables/factorial_contrasts.csv", index=False)
    pd.DataFrame(variability).to_csv(ROOT / "results/tables/protein_variability.csv", index=False)
    print(effects.groupby("term").q.apply(lambda values: (values < 0.05).sum()))


if __name__ == "__main__":
    main()
