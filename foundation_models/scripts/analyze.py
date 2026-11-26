#!/usr/bin/env python3
"""Apply the pre-specified tests and decision rules; write tables and summary.json.

Every hypothesis verdict below follows the rule written in docs/study_protocol.md before the
experiments were run. Exploratory summaries are marked as such.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mice_protein.benchmark import models as M
from mice_protein.benchmark.harness import pooled_scores
from mice_protein.benchmark.metrics import reliability_table
from mice_protein.benchmark.stats import corrected_resampled_ttest, holm

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "foundation_models/results"
TABLES = RES / "tables"
ALPHA = 0.05


def comparisons(folds: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    """Each foundation model vs each baseline: corrected t-test on paired fold differences."""
    wide = folds.pivot_table(index=["repeat", "fold"], columns="model", values=metric)
    n_train = int(folds.n_train.mean())
    n_test = round(folds.n_test.mean())
    rows = []
    for tfm in M.FOUNDATION:
        block = []
        for base in M.BASELINES:
            diff = wide[tfm] - wide[base]
            if not higher_is_better:
                diff = -diff  # positive = foundation model better
            block.append({"foundation": tfm, "baseline": base, "metric": metric,
                          **corrected_resampled_ttest(diff.to_numpy(), n_train, n_test)})
        block = pd.DataFrame(block)
        block["p_holm"] = holm(block.p)
        block["significant_better"] = (block.p_holm < ALPHA) & (block.mean_difference > 0)
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    summary: dict = {"alpha": ALPHA, "labels": {k: s.label for k, s in M.SPECS.items()}}

    # --- RQ1/RQ2/RQ5: accuracy, calibration, cost (eight-group target) -------------------
    folds = pd.read_csv(RES / "accuracy_folds_class.csv")
    preds = pd.read_csv(RES / "predictions_class.csv.gz")
    pooled = pooled_scores(preds.drop(columns="animal"))
    perf = pooled.groupby("model").agg(["mean", "std"])
    perf.columns = [f"{m}_{s}" for m, s in perf.columns]
    perf["seconds_per_fold"] = folds.groupby("model").seconds.median()
    perf = perf.reindex(list(M.SPECS))
    perf.to_csv(TABLES / "performance_class.csv")
    summary["performance"] = perf.round(4).to_dict(orient="index")

    acc_tests = comparisons(folds, "balanced_accuracy", higher_is_better=True)
    cal_tests = comparisons(folds, "log_loss", higher_is_better=False)
    acc_tests.to_csv(TABLES / "h1_accuracy_tests.csv", index=False)
    cal_tests.to_csv(TABLES / "h2_logloss_tests.csv", index=False)
    summary["H1"] = {t: {"supported": bool(g.significant_better.all()),
                         "n_baselines_beaten": int(g.significant_better.sum()),
                         "tests": g.drop(columns=["foundation", "metric"]).round(4)
                         .to_dict(orient="records")}
                     for t, g in acc_tests.groupby("foundation")}
    summary["H2"] = {t: {"supported": bool(g.significant_better.all()),
                         "n_baselines_beaten": int(g.significant_better.sum()),
                         "tests": g.drop(columns=["foundation", "metric"]).round(4)
                         .to_dict(orient="records")}
                     for t, g in cal_tests.groupby("foundation")}

    # Reliability diagrams (pooled over repeats)
    classes = sorted(c for c in preds.columns if c not in ("animal", "model", "repeat", "fold",
                                                            "observed"))
    rel = []
    for model, part in preds.groupby("model"):
        for row in reliability_table(part.observed.to_numpy(), part[classes].to_numpy(),
                                     np.array(classes)):
            rel.append({"model": model, **row})
    pd.DataFrame(rel).to_csv(TABLES / "reliability.csv", index=False)

    # Exploratory: per-group recall (pooled over repeats)
    recall = []
    for model, part in preds.groupby("model"):
        predicted = np.array(classes)[part[classes].to_numpy().argmax(axis=1)]
        for group in classes:
            mask = part.observed.to_numpy() == group
            recall.append({"model": model, "group": group,
                           "recall": float((predicted[mask] == group).mean())})
    pd.DataFrame(recall).to_csv(TABLES / "recall_by_group.csv", index=False)

    # Exploratory: factor targets
    factor = {}
    for tag in ("genotype", "treatment", "behavior"):
        p = pd.read_csv(RES / f"predictions_{tag}.csv.gz").drop(columns="animal")
        s = pooled_scores(p).groupby("model").balanced_accuracy.mean()
        factor[tag] = s.reindex(list(M.SPECS)).round(4).to_dict()
    pd.DataFrame(factor).to_csv(TABLES / "factor_targets.csv")
    summary["factor_targets"] = factor

    # --- RQ3: learning curves ----------------------------------------------------------
    lc = pd.read_csv(RES / "learning_curve.csv")
    curve = lc.groupby(["model", "k"]).balanced_accuracy.agg(["mean", "std", "count"]).reset_index()
    curve.to_csv(TABLES / "learning_curve_summary.csv", index=False)
    rng = np.random.default_rng(2026)
    wide = lc.pivot_table(index=["k", "draw"], columns="model", values="balanced_accuracy")
    h3 = {}
    for tfm in M.FOUNDATION:
        gap = (wide[tfm] - wide["logistic"]).rename("gap").reset_index()
        g2 = gap[gap.k == min(gap.k)].gap.to_numpy()
        g6 = gap[gap.k == max(gap.k)].gap.to_numpy()
        boots = np.array([rng.choice(g2, len(g2)).mean() - rng.choice(g6, len(g6)).mean()
                          for _ in range(4000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        h3[tfm] = {"gap_small": float(g2.mean()), "gap_large": float(g6.mean()),
                   "difference": float(g2.mean() - g6.mean()), "lower": float(lo),
                   "upper": float(hi), "supported": bool(lo > 0),
                   "gap_by_k": gap.groupby("k").gap.mean().round(4).to_dict()}
    summary["H3"] = h3
    summary["learning_curve"] = {m: g.set_index("k")["mean"].round(4).to_dict()
                                 for m, g in curve.groupby("model")}

    # --- RQ4: validity -------------------------------------------------------------------
    perm = pd.read_csv(RES / "permutation_null.csv")
    pm = perm.groupby("model").balanced_accuracy.agg(["mean", "min", "max"])
    summary["H4a"] = {"supported": bool(((pm["mean"] - 0.125).abs() <= 0.05).all()),
                      "by_model": pm.round(4).to_dict(orient="index")}
    leak = pd.read_csv(RES / "leakage.csv")
    lk = leak.groupby(["model", "design"]).balanced_accuracy.mean().unstack()
    lk["inflation"] = lk["random_rows"] - lk["mouse_held_out"]
    summary["H4b"] = {"supported": bool((lk.inflation > 0.2).all()),
                      "by_model": lk.round(4).to_dict(orient="index")}
    mem = pd.read_csv(RES / "memorization.csv")
    mw = mem.pivot_table(index=["model", "repeat"], columns="version", values="balanced_accuracy")
    mem_rows = {}
    for model, g in mw.groupby(level="model"):
        drops = g.drop(columns="original").rsub(g["original"], axis=0)
        flagged = drops.to_numpy().mean() > 0.05 and bool((drops.mean() > 0).all())
        mem_rows[model] = {"original": float(g["original"].mean()),
                           "obfuscated": float(g.drop(columns="original").to_numpy().mean()),
                           "mean_drop": float(drops.to_numpy().mean()), "flagged": bool(flagged)}
    summary["H4c"] = {"supported": not any(v["flagged"] for v in mem_rows.values()),
                      "by_model": mem_rows}

    summary["provenance"] = json.loads((RES / "provenance.json").read_text())
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k]["supported"] if isinstance(summary[k], dict) and
                      "supported" in summary[k] else None
                      for k in ("H4a", "H4b", "H4c")}, indent=1))
    for h in ("H1", "H2", "H3"):
        print(h, {m: v["supported"] for m, v in summary[h].items()})


if __name__ == "__main__":
    main()
