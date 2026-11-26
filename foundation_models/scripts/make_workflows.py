#!/usr/bin/env python3
"""Workflow diagrams for the foundation-model study (one per research question RQ1-RQ4).

Same design as the biology study: plain-language steps with defined equations, ending in a
result panel that plots the real numbers from foundation_models/results/summary.json.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from mice_protein import plotting as P
from mice_protein.workflow import Stage, draw_workflow

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "foundation_models/results"
OUT = RES / "figures/workflows"
S = json.loads((RES / "summary.json").read_text())


def w(text: str, width: int = 27) -> str:
    return textwrap.fill(text, width)


def supported(h: str) -> str:
    block = S[h]
    ok = [m for m, v in block.items() if isinstance(v, dict) and v.get("supported")]
    if len(ok) == 2:
        return "holds for both models"
    if not ok:
        return "does not hold"
    return "holds for " + S["labels"][ok[0]] + " only"


def workflows():
    perf = S["performance"]
    baselines = [m for m in perf if m not in ("tabpfn", "kumo", "dummy")]
    best_acc = max(baselines, key=lambda m: perf[m]["balanced_accuracy_mean"])
    best_ll = min(baselines, key=lambda m: perf[m]["log_loss_mean"])
    h3 = S["H3"]
    return {
        "rq1_accuracy": ("RQ1 · Are foundation models more accurate than tuned conventional models?", [
            Stage("SPLIT", "Same held-out mice", "cv_folds",
                  w("Split the 72 mice into 5 folds, 5 times; every model is tested on exactly "
                    "the same held-out mice."), r"$J = 5\times 5 = 25$ folds", "J = number of folds"),
            Stage("FIT", "Eight models", "model_zoo",
                  w("Tune six conventional models inside each training fold; use the two "
                    "foundation models as they come."), r"$\hat y = f(x)$", "x = 77 protein means"),
            Stage("SCORE", "Balanced accuracy", "four_arms",
                  w("Score each fold by balanced accuracy: the average hit rate over the "
                    "8 groups (chance = 0.125)."),
                  r"$\mathrm{BA}=\frac{1}{8}\sum_{g}\mathrm{recall}_g$", "g = experimental group"),
            Stage("TEST", "Paired comparison", "paired_diff",
                  w("Compare models fold by fold with a t-test corrected for overlapping "
                    "training sets; adjust for 6 comparisons (Holm)."),
                  r"$t=\bar d\,/\,[(\frac{1}{J}+\frac{n_{te}}{n_{tr}})\,s_d^{2}]^{1/2}$",
                  "d = fold difference; n = mice in test / train"),
            Stage("RESULT", "Accuracy", "result_bars",
                  w(f"H1 {supported('H1')}: better than every baseline after correction. "
                    "Chance = 0.125."),
                  args={"values": [perf[best_acc]["balanced_accuracy_mean"],
                                   perf["tabpfn"]["balanced_accuracy_mean"],
                                   perf["kumo"]["balanced_accuracy_mean"]],
                        "labels": [S["labels"][best_acc].split(" (")[0] + "\n(best baseline)",
                                   "TabPFN", "Kumo"],
                        "colors": [P.COBALT, P.VERMILION, P.VERMILION], "ymax": 1.0},
                  result=True),
        ]),
        "rq2_calibration": ("RQ2 · Are their predicted probabilities better?", [
            Stage("COLLECT", "Predicted probabilities", "protein_matrix",
                  w("Keep each model’s probability for every group, for every held-out mouse."),
                  r"$\hat p_{ik}$", "i = mouse, k = group"),
            Stage("SCORE", "Log loss", "two_groups",
                  w("Penalise confident wrong answers: log loss is low only when the true group "
                    "gets high probability."),
                  r"$\mathrm{LL}=-\frac{1}{n}\sum_i \log \hat p_{i,y_i}$", "y = true group",
                  args={"left": "baseline", "right": "foundation"}),
            Stage("CHECK", "Reliability", "reliability",
                  w("Check that predictions made with 80% confidence are right about 80% of "
                    "the time."), r"$\mathrm{ECE}=\sum_b \frac{n_b}{n}\,|\mathrm{acc}_b-\mathrm{conf}_b|$",
                  "b = confidence bin"),
            Stage("TEST", "Paired comparison", "paired_diff",
                  w("Same corrected, Holm-adjusted test as RQ1, on fold-level log loss."),
                  r"lower $\mathrm{LL}$ = better", ""),
            Stage("RESULT", "Log loss (lower is better)", "result_bars",
                  w(f"H2 {supported('H2')}."),
                  args={"values": [perf[best_ll]["log_loss_mean"], perf["tabpfn"]["log_loss_mean"],
                                   perf["kumo"]["log_loss_mean"]],
                        "labels": [S["labels"][best_ll].split(" (")[0] + "\n(best baseline)",
                                   "TabPFN", "Kumo"],
                        "colors": [P.COBALT, P.VERMILION, P.VERMILION]},
                  result=True),
        ]),
        "rq3_learning": ("RQ3 · Is their advantage largest when data are scarcest?", [
            Stage("SAMPLE", "Few training mice", "subsample",
                  w("Train on k randomly chosen mice per group (k = 2 to 6) and test on all the "
                    "others; repeat 20 times per k."), r"$k\in\{2,\dots,6\}$", "k = mice per group"),
            Stage("FIT", "Five models", "model_zoo",
                  w("Fit logistic regression, random forest, SVM, TabPFN and Kumo with default "
                    "settings; tuning is impossible with so few mice."), "", "",
                  args={"left": "3 conventional\n(defaults)", "right": "2 foundation\n(as-is)"}),
            Stage("PLOT", "Learning curves", "curve",
                  w("Follow accuracy as the number of training mice grows."),
                  r"$\mathrm{BA}(k)$", ""),
            Stage("TEST", "Gap at 2 vs 6", "two_groups",
                  w("Compare the foundation models’ lead over logistic regression with 2 and "
                    "with 6 mice per group (bootstrap interval)."),
                  r"$\mathrm{gap}_k=\mathrm{BA}^{\mathrm{FM}}_k-\mathrm{BA}^{\mathrm{logistic}}_k$",
                  "FM = foundation model", args={"left": "k = 6", "right": "k = 2"}),
            Stage("RESULT", "Lead over logistic", "result_bars",
                  w(f"H3 {supported('H3')}: the lead grows with more data (Kumo "
                    f"{h3['kumo']['gap_small']:+.2f} at 2 vs {h3['kumo']['gap_large']:+.2f} at 6 "
                    "mice per group)."),
                  args={"values": [h3["tabpfn"]["gap_small"], h3["tabpfn"]["gap_large"],
                                   h3["kumo"]["gap_small"], h3["kumo"]["gap_large"]],
                        "labels": ["TabPFN\nk=2", "TabPFN\nk=6", "Kumo\nk=2", "Kumo\nk=6"],
                        "colors": [P.MUTED, P.VERMILION, P.MUTED, P.VERMILION],
                        "fmt": "{:+.2f}"},
                  result=True),
        ]),
        "rq4_validity": ("RQ4 · Are the scores trustworthy?", [
            Stage("SHUFFLE", "Break the labels", "shuffle",
                  w("Shuffle group labels across mice; any model scoring above chance would "
                    "reveal a bug or leak."), r"$y\rightarrow\pi(y)$", "π = random permutation",
                  args={"n": "10"}),
            Stage("LEAK", "Split by rows", "row_leak",
                  w("Split the 1,080 rows at random instead of by mouse; every model should "
                    "look better than it is."), "random rows vs whole mice", ""),
            Stage("HIDE", "Hide the dataset", "obfuscate",
                  w("Rename, shuffle and rescale the proteins; a model that memorised this public "
                    "dataset would lose accuracy."), r"$x'_j=a_j\,x_{\pi(j)}+b_j$",
                  "a > 0, b = random scale and shift"),
            Stage("RULE", "Decided in advance", "three_controls",
                  w("Each control has a pass/fail rule written before any model was run."),
                  "", ""),
            Stage("RESULT", "Controls", "result_checks",
                  w("All three controls behave as expected." if S["H4a"]["supported"]
                    and S["H4b"]["supported"] and S["H4c"]["supported"]
                    else "At least one control did not behave as expected."),
                  args={"items": [("shuffled → chance", S["H4a"]["supported"]),
                                  ("leaky → inflated", S["H4b"]["supported"]),
                                  ("hidden → unchanged", S["H4c"]["supported"])]},
                  result=True),
        ]),
    }


def main() -> None:
    P.apply_style()
    for name, (question, stages) in workflows().items():
        draw_workflow(stages, OUT, f"wf_{name}", question=question)
        draw_workflow(stages, OUT / "slides", f"wf_{name}")
        print("drew", name)


if __name__ == "__main__":
    main()
