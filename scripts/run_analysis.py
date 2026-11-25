#!/usr/bin/env python3
"""Run all empirical analyses and save traceable result tables."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, f1_score

from mice_protein.data import aggregate_animals, load_data, protein_columns
from mice_protein.evaluation import animal_vote_predictions, repeated_cv_predictions, summarize
from mice_protein.models import build_model
from mice_protein.statistics import genotype_effects

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    frame = load_data(ROOT / cfg["raw_data"])
    features = protein_columns(frame)
    X, y, groups = frame[features], frame[cfg["primary_target"]], frame.AnimalID
    out = ROOT / "results"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    summaries, fold_rows = [], []
    all_records = {}
    for design, grouped in [("random_observation", False), ("grouped_animal", True)]:
        for name in cfg["models"]:
            records = repeated_cv_predictions(
                X,
                y,
                groups,
                build_model(name, cfg["seed"]),
                grouped=grouped,
                repeats=cfg["n_repeats"],
                splits=cfg["n_splits"],
                seed=cfg["seed"],
            )
            all_records[f"{design}:{name}"] = records
            summary = {"design": design, "model": name, **summarize(records)}
            if grouped:
                votes = [animal_vote_predictions(r) for r in records]
                summary["mouse_balanced_accuracy_mean"] = float(
                    np.mean([balanced_accuracy_score(a, b) for a, b in votes])
                )
                summary["mouse_macro_f1_mean"] = float(
                    np.mean([f1_score(a, b, average="macro", zero_division=0) for a, b in votes])
                )
            summaries.append(summary)
            fold_rows.extend(
                {k: v for k, v in r.items() if k not in {"y_true", "y_pred", "animal_ids"}}
                | {"design": design, "model": name}
                for r in records
            )

    # Scientifically transparent mouse-mean sensitivity analysis.
    mice = aggregate_animals(frame)
    Xm, ym = mice[features], mice["class"]
    for name in cfg["models"]:
        rec = repeated_cv_predictions(
            Xm,
            ym,
            mice.AnimalID,
            build_model(name, cfg["seed"]),
            grouped=False,
            repeats=cfg["n_repeats"],
            splits=cfg["n_splits"],
            seed=cfg["seed"],
        )
        summaries.append(
            {
                "design": "mouse_mean",
                "model": name,
                **summarize(rec),
                "mouse_balanced_accuracy_mean": summarize(rec)["balanced_accuracy_mean"],
                "mouse_macro_f1_mean": summarize(rec)["macro_f1_mean"],
            }
        )
        fold_rows.extend(
            {k: v for k, v in r.items() if k not in {"y_true", "y_pred", "animal_ids"}}
            | {"design": "mouse_mean", "model": name}
            for r in rec
        )

    pd.DataFrame(summaries).to_csv(out / "metrics/model_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "metrics/fold_metrics.csv", index=False)
    (out / "metrics/predictions.json").write_text(json.dumps(all_records, indent=2))

    effects = genotype_effects(frame)
    effects.to_csv(out / "tables/genotype_effects.csv", index=False)
    missing = pd.DataFrame({"protein": features, "missing_n": frame[features].isna().sum().values})
    missing["missing_fraction"] = missing.missing_n / len(frame)
    missing.sort_values("missing_n", ascending=False).to_csv(
        out / "tables/missingness.csv", index=False
    )
    class_counts = (
        frame.groupby("class")
        .agg(rows=("MouseID", "size"), animals=("AnimalID", "nunique"))
        .reset_index()
    )
    class_counts.to_csv(out / "tables/class_counts.csv", index=False)

    # Full-data model is for interpretation/deployment only, never for performance estimation.
    final = build_model("logistic", cfg["seed"]).fit(X, y)
    joblib.dump({"model": final, "features": features}, out / "models/final_logistic.joblib")
    perm = permutation_importance(
        final, X, y, scoring="balanced_accuracy", n_repeats=20, random_state=cfg["seed"], n_jobs=-1
    )
    pd.DataFrame(
        {
            "protein": features,
            "importance_mean": perm.importances_mean,
            "importance_sd": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).to_csv(
        out / "tables/permutation_importance.csv", index=False
    )

    provenance = {
        "seed": cfg["seed"],
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "rows": len(frame),
        "animals": frame.AnimalID.nunique(),
        "proteins": len(features),
    }
    (out / "metrics/run_metadata.json").write_text(json.dumps(provenance, indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
