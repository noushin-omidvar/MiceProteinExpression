#!/usr/bin/env python3
"""Summarize post hoc analyses from saved probabilities and scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss

from mice_protein.benchmark.stats import corrected_resampled_ttest

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "foundation_models/results"


def main() -> None:
    original = pd.read_csv(RES / "predictions_class.csv.gz")
    classes = sorted(c for c in original if c.startswith(("c-", "t-")))
    rows = []
    for model in ("tabpfn", "kumo"):
        frame = original[original.model == model]
        for repeat, part in frame.groupby("repeat"):
            p = part[classes].to_numpy()
            rows.append({"model": model, "repeat": repeat,
                         "balanced_accuracy": balanced_accuracy_score(
                             part.observed, np.asarray(classes)[p.argmax(axis=1)]),
                         "log_loss": log_loss(part.observed, p, labels=classes)})
    for model in ("logistic", "svm", "random_forest", "gradient_boosting"):
        path = RES / f"tuned_baselines/predictions_{model}.csv.gz"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        svm_decisions = None
        if model == "svm":
            svm_decisions = pd.read_csv(RES / "svm_main_audit/predictions.csv")
            svm_decisions = svm_decisions[svm_decisions.version == "post_hoc_wider_grid"]
        for repeat, part in frame.groupby("repeat"):
            p = part[[f"p_{c}" for c in classes]].to_numpy()
            decision_ba = balanced_accuracy_score(part.observed,
                                                   np.asarray(classes)[p.argmax(axis=1)])
            if svm_decisions is not None:
                direct = svm_decisions[svm_decisions.repeat == repeat]
                decision_ba = balanced_accuracy_score(direct.observed, direct.predicted)
            rows.append({"model": model, "repeat": repeat,
                         "balanced_accuracy": decision_ba,
                         "log_loss": log_loss(part.observed, p, labels=classes)})
    factorized = pd.read_csv(RES / "factorized/predictions.csv.gz")
    for repeat, part in factorized.groupby("repeat"):
        p = part[[f"p_{c}" for c in classes]].to_numpy()
        rows.append({"model": "factorized_logistic", "repeat": repeat,
                     "balanced_accuracy": balanced_accuracy_score(part.observed,
                                                                 part.predicted),
                     "log_loss": log_loss(part.observed, p, labels=classes)})
    per_repeat = pd.DataFrame(rows)
    out = RES / "tuned_baselines"
    per_repeat.to_csv(out / "pooled_by_repeat.csv", index=False)
    performance = per_repeat.groupby("model")["balanced_accuracy"].agg(
        ["mean", "std"]).round(4)
    performance.to_csv(out / "performance.csv")
    original_folds = pd.read_csv(RES / "accuracy_folds_class.csv")
    tuned_logistic = pd.read_csv(out / "scores_logistic.csv")
    paired = []
    for model in ("kumo", "tabpfn"):
        left = original_folds.loc[original_folds.model == model,
                                  ["repeat", "fold", "balanced_accuracy"]]
        right = tuned_logistic[["repeat", "fold", "balanced_accuracy"]]
        match = left.merge(right, on=["repeat", "fold"], suffixes=("_foundation", "_tuned"),
                           validate="one_to_one")
        result = corrected_resampled_ttest(
            (match.balanced_accuracy_foundation - match.balanced_accuracy_tuned).to_numpy(),
            n_train=58, n_test=14)
        paired.append({"foundation": model, "baseline": "tuned_logistic",
                       "status": "post_hoc_exploratory", **result})
    pd.DataFrame(paired).to_csv(out / "paired_vs_tuned_logistic.csv", index=False)
    curve_dir = RES / "matched_holdout_review"
    curve = pd.read_csv(curve_dir / "scores.csv")
    matched = curve.groupby(["n_train", "model"]).balanced_accuracy.agg(
        ["mean", "std"]).round(4)
    matched.to_csv(curve_dir / "summary.csv")
    wide = curve.pivot_table(index=["repeat", "fold", "draw", "size", "n_train"],
                             columns="model", values="balanced_accuracy").reset_index()
    for model in ("tabpfn", "kumo"):
        wide[f"{model}_minus_logistic"] = wide[model] - wide["logistic"]
    gap_cols = ["repeat", "fold", "draw", "size", "n_train",
                "tabpfn_minus_logistic", "kumo_minus_logistic"]
    wide[gap_cols].to_csv(curve_dir / "paired_gaps.csv", index=False)
    gap_summary = wide.groupby("n_train")[["tabpfn_minus_logistic",
                                            "kumo_minus_logistic"]].agg(["mean", "std"])
    gap_summary.to_csv(curve_dir / "paired_gap_summary.csv")
    sensitivity = pd.read_csv(RES / "sensitivity/scores.csv")
    controls = sensitivity.groupby(["version", "model"]).balanced_accuracy.agg(
        ["mean", "std"]).round(4)
    controls.to_csv(RES / "sensitivity/summary.csv")
    error_rows = []
    for model, part in original.groupby("model"):
        per_mouse = part.groupby(["animal", "observed"], as_index=False)[classes].mean()
        probability = per_mouse[classes].to_numpy()
        predicted = np.asarray(classes)[probability.argmax(axis=1)]
        for animal, truth, guess, confidence in zip(
            per_mouse.animal, per_mouse.observed, predicted, probability.max(axis=1), strict=True
        ):
            error_rows.append({"model": model, "animal": animal, "observed": truth,
                               "predicted": guess, "correct": truth == guess,
                               "confidence": confidence})
    pd.DataFrame(error_rows).to_csv(RES / "error_audit_by_mouse.csv", index=False)
    manifest = {"status": "post_hoc_exploratory", "independent_mice": 72,
                "tuned_baselines": performance.to_dict(orient="index"),
                "paired_vs_tuned_logistic": paired,
                "matched_holdout": matched.reset_index().to_dict(orient="records"),
                "sensitivity": controls.reset_index().to_dict(orient="records")}
    (RES / "extension_summary.json").write_text(json.dumps(manifest, indent=2))
    print(performance.to_string())


if __name__ == "__main__":
    main()
