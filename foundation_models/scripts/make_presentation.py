#!/usr/bin/env python3
"""Foundation-model study deck, written for an audience new to the topic.

Story: WHY -> BACKGROUND -> DATA -> QUESTIONS -> METHOD -> RESULTS -> SO WHAT.
Pre-specified results and later exploratory checks are labelled separately on every slide
that shows them. Numbers come from foundation_models/results/ (no hand-typed values).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from make_paper import document_author

from mice_protein.slides import (
    BigNumber,
    Claim,
    ClaimSplit,
    Deck,
    Points,
    Section,
    TableSlide,
    Title,
    render,
)

STUDY = ROOT / "foundation_models"
RES = STUDY / "results"
FIG = RES / "figures"
SLIDE = FIG / "slides"
WF = FIG / "workflows" / "slides"
PRIMER = ROOT / "results/figures/primer"
HIGUERA = ("Reproduced from Higuera, Gardiner & Cios, PLoS ONE 10(6): e0129126 (2015), Fig 1A "
           "(cropped) · CC BY 4.0")


def numbers() -> dict:
    S = json.loads((RES / "summary.json").read_text())
    E = json.loads((RES / "extension_summary.json").read_text())
    perf = S["performance"]
    tuned = pd.read_csv(RES / "tuned_baselines/performance.csv", index_col="model")["mean"]
    vs_tuned = {r["foundation"]: r for r in E["paired_vs_tuned_logistic"]}
    curve = pd.read_csv(RES / "matched_holdout_review/scores.csv")
    m = curve.groupby(["n_train", "model"]).balanced_accuracy.mean()
    h1 = {t["baseline"]: t for t in S["H1"]["kumo"]["tests"]}
    h1t = {t["baseline"]: t for t in S["H1"]["tabpfn"]["tests"]}
    sens = pd.DataFrame(E["sensitivity"]).set_index(["version", "model"])["mean"]
    return {
        "kumo": perf["kumo"]["balanced_accuracy_mean"], "tabpfn": perf["tabpfn"]["balanced_accuracy_mean"],
        "logistic": perf["logistic"]["balanced_accuracy_mean"],
        "ll_kumo": perf["kumo"]["log_loss_mean"], "ll_tabpfn": perf["tabpfn"]["log_loss_mean"],
        "ll_logistic": perf["logistic"]["log_loss_mean"],
        "kumo_vs_log": h1["logistic"], "tab_vs_log": h1t["logistic"],
        "tuned_log": tuned["logistic"], "kumo_vs_tuned": vs_tuned["kumo"],
        "tab_vs_tuned": vs_tuned["tabpfn"],
        "k16": m.loc[16, "kumo"], "l16": m.loc[16, "logistic"], "t16": m.loc[16, "tabpfn"],
        "k48": m.loc[48, "kumo"], "l48": m.loc[48, "logistic"], "t48": m.loc[48, "tabpfn"],
        "h3": S["H3"], "perm": S["H4a"]["by_model"], "leak": S["H4b"]["by_model"],
        "mem": S["H4c"]["by_model"],
        "miss_kumo": sens.get(("missingness_only", "kumo")),
        "verdicts": {h: S[h] for h in ("H1", "H2", "H3", "H4a", "H4b", "H4c")},
    }


def yes(flag: bool) -> str:
    return "Supported" if flag else "Not supported"


def build() -> Deck:
    n = numbers()
    v = n["verdicts"]
    kt = n["kumo_vs_tuned"]
    slides = [
        Title("On 72 mice, one foundation model led every model we tested, but its edge over a "
              "tuned baseline is uncertain",
              "TabPFN v2 and Kumo Tabular compared with conventional models on cortical protein data",
              document_author(),
              notes="This talk asks a practical question: do pretrained ‘foundation’ models for "
                    "tables actually help on a small biological dataset? No machine-learning "
                    "background is assumed."),

        # ------------------------------------------------------------------ WHY
        Section(1, "WHY", "Do pretrained models help when every sample is expensive?",
                notes="Motivation first."),
        Points("Biology produces small tables, and new models promise to handle them",
               [("Each sample is an animal or a patient",
                 "so datasets of tens to hundreds of samples are normal"),
                ("Tabular foundation models are pretrained on millions of tables",
                 "and are often said to work best exactly when data are scarce"),
                ("But testing that claim fairly is hard",
                 ("few independent samples, noisy comparisons, and public data that a model may "
                 "already have seen"))],
               notes="Our job is to test the claim carefully on one real dataset."),

        # ------------------------------------------------------------------ BACKGROUND
        Section(2, "BACKGROUND", "What is a tabular foundation model?",
                notes="One idea and two definitions."),
        Claim("A foundation model reads the training data as context instead of fitting to it",
              FIG / "primer_tfm.png",
              "Schematic. Both foundation models tested here were pretrained on synthetic tables.",
              notes="A conventional model learns weights from your training rows alone. A "
                    "foundation model was trained once, on millions of synthetic tables, and then "
                    "predicts new rows by reading your training rows as context, in one pass."),
        Points("We compared eight models",
               [("Six conventional models",
                 ("class prior (always guess the commonest group), logistic regression, elastic "
                 "net, support-vector machine, random forest, gradient boosting")),
                ("TabPFN v2 (Prior Labs)", "a widely used tabular foundation model"),
                ("Kumo Tabular Small (NVIDIA)", "a newer tabular foundation model"),
                ("Same data, same held-out mice for every model", "so differences are fair")],
               notes="Exact versions: tabpfn 2.2.1, structured-data-models 0.3.0, CPU, defaults."),
        TableSlide("Key terms used in this talk",
                   ["Term", "Meaning"],
                   [["Balanced accuracy", "Average hit rate across the 8 groups; chance = 0.125"],
                    ["Log loss", "Penalty for confident wrong answers; lower is better"],
                    ["Leakage", "Test information reaching the model during training"],
                    ["Pre-specified", "Fixed in writing before any experiment was run"],
                    ["Exploratory", "Added after seeing results; suggestive, not confirmatory"]],
                   [1.4, 4.0],
                   notes="The last two matter: we keep planned and later analyses apart."),

        # ------------------------------------------------------------------ DATA
        Section(3, "DATA", "The test bed: 72 mice, 77 proteins, 8 groups",
                notes="A real biology dataset, described in plain terms."),
        Claim("The task: predict which of eight experimental groups a new mouse belongs to",
              ROOT / "assets/external/higuera2015_fig1A.png", HIGUERA,
              notes="Groups come from three yes/no choices: Down-syndrome model or control, "
                    "learning protocol or not, drug or placebo. 7–10 mice per group."),
        Claim("The table has 1,080 rows, but every 15 rows are the same mouse",
              PRIMER / "data_anatomy.png",
              "Real rows. We average each mouse’s 15 readings into one profile of 77 proteins.",
              notes="This is the single most important fact about the data."),
        BigNumber("So there are only 72 independent samples to learn from and test on",
                  "72 mice",
                  "77 proteins per mouse · 8 groups · chance accuracy 0.125 (1 in 8)",
                  "Every model is always tested on mice it has never seen.",
                  notes="Small by any standard, which is the point."),

        # ------------------------------------------------------------------ QUESTIONS
        Section(4, "QUESTIONS", "Four questions, fixed before running anything",
                notes="Each question has a hypothesis written down in advance."),
        TableSlide("Each question came with a pre-specified hypothesis",
                   ["Question", "Hypothesis written in advance"],
                   [["RQ1 · More accurate?", "Each foundation model beats every conventional model"],
                    ["RQ2 · Better probabilities?", "Lower log loss than every conventional model"],
                    ["RQ3 · Best with least data?", "Bigger lead with 2 than with 6 mice per group"],
                    ["RQ4 · Trustworthy?", ("Chance with shuffled labels; inflated by leakage; "
                                           "no memorisation")]],
                   [1.6, 3.6],
                   notes="Committed to the repository before the experiments."),

        # ------------------------------------------------------------------ METHOD
        Section(5, "METHOD", "How do you compare models fairly with 72 mice?",
                notes="Fair comparison is the core of the method."),
        Points("Four rules for a fair comparison",
               [("Hold out whole mice", "never split a mouse’s readings between training and test"),
                ("Use identical splits for every model", "5 repeats × 5 folds, the same mice each time"),
                ("Use tests that respect overlapping splits",
                 "corrected paired t-test, adjusted for multiple comparisons"),
                ("Keep planned and later analyses apart",
                 "anything added after seeing results is labelled exploratory")],
               notes="Rule 1 matters most, as the next slide shows."),
        Claim("Splitting rows instead of mice makes every model look nearly perfect",
              FIG / "fig2_leakage.png",
              "Same 1,080 rows; only the split rule changes.",
              notes="With random row splits the model has already seen other readings of the "
                    "test mouse. Foundation models are as vulnerable as conventional ones."),

        # ------------------------------------------------------------------ RESULTS
        Section(6, "RESULTS", "What did we find?",
                notes="For each question: how we tested it, then the answer."),
        Claim("RQ1 · Every model is tested on the same held-out mice and compared fold by fold",
              WF / "wf_rq1_accuracy.png", "Read left to right; the last panel is the result.",
              notes="The paired test accounts for overlapping training sets."),
        ClaimSplit("RQ1 · Pre-specified: Kumo beats all six baselines; TabPFN does not beat "
                   "logistic regression",
                   SLIDE / "accuracy.png",
                   [f"Kumo {n['kumo']:.2f}, TabPFN {n['tabpfn']:.2f}, logistic {n['logistic']:.2f}",
                    (f"Kumo vs logistic: +{n['kumo_vs_log']['mean_difference']:.2f} "
                    f"(p = {n['kumo_vs_log']['p_holm']:.3f})"),
                    (f"TabPFN vs logistic: +{n['tab_vs_log']['mean_difference']:.2f} "
                    f"(p = {n['tab_vs_log']['p_holm']:.2f}), not distinguishable"),
                    "The simplest conventional model was the strongest"],
                   "Balanced accuracy on held-out mice; chance = 0.125.",
                   notes="This is the pre-specified result. The baseline logistic regression used "
                         "a fixed setting, which matters on the next slide."),
        Claim(f"RQ1 · Exploratory: against a tuned logistic regression, Kumo’s lead "
              f"(+{kt['mean_difference']:.2f}) is no longer statistically clear",
              FIG / "fig3_tuned_benchmark.png",
              f"Tuned logistic {n['tuned_log']:.3f}. Kumo minus tuned logistic: 95% interval "
              f"{kt['lower']:+.2f} to {kt['upper']:+.2f}, p = {kt['p']:.2f} (post hoc).",
              notes="After seeing the results we tuned the conventional models more widely. The "
                    "point estimate still favours Kumo, but the interval includes zero. CatBoost "
                    "and XGBoost were not tested."),
        Claim("RQ2 · We score the probabilities, not just the guesses",
              WF / "wf_rq2_calibration.png", "Read left to right; the last panel is the result.",
              notes="Log loss rewards confident correct answers and punishes confident wrong ones."),
        ClaimSplit("RQ2 · Kumo gives the best probabilities, though that alone is not proof of "
                   "calibration",
                   SLIDE / "logloss.png",
                   [(f"Log loss: Kumo {n['ll_kumo']:.2f}, TabPFN {n['ll_tabpfn']:.2f}, "
                    f"logistic {n['ll_logistic']:.2f}"),
                    (f"H2: Kumo {yes(v['H2']['kumo']['supported']).lower()}, "
                    f"TabPFN {yes(v['H2']['tabpfn']['supported']).lower()}"),
                    ("Log loss mixes accuracy and calibration; 72 mice are too few for a precise "
                    "calibration test")],
                   "Lower is better.",
                   notes="Show the reliability figure in the paper if asked."),
        Claim("RQ3 · We train on 2 to 6 mice per group and track each model’s accuracy",
              WF / "wf_rq3_learning.png", "Read left to right; the last panel is the result.",
              notes="The hypothesis was that foundation models help most with the least data."),
        ClaimSplit("RQ3 · The foundation models’ lead grows with more data, the opposite of the "
                   "hypothesis",
                   SLIDE / "gap.png",
                   [(f"Kumo’s lead: {n['h3']['kumo']['gap_small']:+.2f} at 2 per group, "
                    f"{n['h3']['kumo']['gap_large']:+.2f} at 6"),
                    (f"TabPFN’s lead: {n['h3']['tabpfn']['gap_small']:+.2f} → "
                    f"{n['h3']['tabpfn']['gap_large']:+.2f}"),
                    "H3 not supported for either model"],
                   "Pre-specified analysis. Lead = foundation model minus logistic regression.",
                   notes="With only 16 training mice, no model has much to work with."),
        Claim("RQ3 · Exploratory: a fairer, matched learning curve shows the same pattern, with "
              "large variation",
              FIG / "fig4_matched_learning.png",
              f"Same test mice at every size. 16 mice: Kumo {n['k16']:.2f}, logistic {n['l16']:.2f}. "
              f"48 mice: Kumo {n['k48']:.2f}, logistic {n['l48']:.2f}.",
              notes="The pre-specified curve changed the test mice as the training set grew. This "
                    "post hoc version fixes that. The average gap grows but varies a lot between "
                    "draws, so there is no dependable threshold."),
        Claim("RQ4 · Three controls, each with a pass/fail rule written in advance",
              WF / "wf_rq4_validity.png", "Read left to right; the last panel is the result.",
              notes="Shuffled labels, leaky splits, and hidden dataset identity."),
        Claim("RQ4 · All three controls behave as expected",
              FIG / "fig4_validity.png",
              "Shuffled labels give chance; leakage inflates every model; hiding the dataset’s "
              "identity barely changes accuracy.",
              notes="Caveat: hiding names and rescaling values cannot prove a model never saw "
                    "this public dataset. NVIDIA reports Kumo was pretrained only on synthetic "
                    "tables."),
        Claim("Exploratory · Missing-value patterns alone predict nothing; preprocessing shifts "
              "scores modestly",
              FIG / "fig7_sensitivity.png",
              f"Missingness-only input: Kumo {n['miss_kumo']:.2f} (chance 0.125)."
              if n["miss_kumo"] is not None else "Post hoc sensitivity checks.",
              notes="So the models are using protein levels, not which values happen to be missing."),

        # ------------------------------------------------------------------ SO WHAT
        Section(7, "SO WHAT", "What should a scientist take from this?",
                notes="Bring it together."),
        TableSlide("Verdicts on the pre-specified hypotheses",
                   ["Hypothesis", "TabPFN v2", "Kumo"],
                   [["H1 · More accurate than every baseline",
                     yes(v["H1"]["tabpfn"]["supported"]), yes(v["H1"]["kumo"]["supported"])],
                    ["H2 · Better probabilities (log loss)",
                     yes(v["H2"]["tabpfn"]["supported"]), yes(v["H2"]["kumo"]["supported"])],
                    ["H3 · Biggest lead with least data",
                     yes(v["H3"]["tabpfn"]["supported"]), yes(v["H3"]["kumo"]["supported"])],
                    ["H4 · Controls behave as expected",
                     yes(v["H4a"]["supported"] and v["H4b"]["supported"] and v["H4c"]["supported"]),
                     yes(v["H4a"]["supported"] and v["H4b"]["supported"] and v["H4c"]["supported"])]],
                   [3.0, 1.4, 1.4], highlight_column=None,
                   notes="And the exploratory caveat: against a tuned logistic regression, Kumo’s "
                         "lead is not statistically clear."),
        Points("Practical lessons for using foundation models on scientific tables",
               [("Count independent units, not rows", "split by animal, patient or experiment"),
                ("Compare against a well-tuned simple model", "logistic regression was hard to beat"),
                ("Report probabilities and test with corrected statistics",
                 "fold-to-fold differences are not independent"),
                ("Check for contamination, but know its limits",
                 "hiding names and scales cannot prove a model never saw the data")],
               notes="Each of these changed or protected a conclusion in this study."),
        Points("What remains open",
               [("Stronger conventional baselines", "CatBoost and XGBoost were not available"),
                ("Newer foundation-model versions", "results apply to the versions tested"),
                ("An independent cohort", "all results come from one set of 72 mice"),
                ("Where the advantage comes from", "why Kumo and TabPFN differ is untested")],
               notes="Close with the open questions."),
        Points("Take-home messages",
               [("Kumo led every model we tested", "and was clearly ahead in the pre-specified test"),
                ("But not clearly ahead of a tuned logistic regression",
                 "in a later exploratory comparison"),
                ("No small-data advantage", "the lead grew with more training mice")],
               notes="Everything is reproducible with make fm-reproduce."),
    ]
    return Deck("Tabular foundation models on 72 mice",
                ["WHY", "BACKGROUND", "DATA", "QUESTIONS", "METHOD", "RESULTS", "SO WHAT"],
                slides, document_author())


def main() -> None:
    deck = build()
    render(deck, STUDY / "presentation")
    print(f"wrote {len(deck.slides)} slides to foundation_models/presentation/")


if __name__ == "__main__":
    main()
