#!/usr/bin/env python3
"""Foundation-model study deck, written for an audience new to the topic.

Outline (shared with the README and the report):
1 QUESTION · 2 WHY · 3 BACKGROUND · 4 DATA · 5 APPROACH · 6 FINDINGS · 7 HOW SURE · 8 NEXT.
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
EXPLAIN = FIG / "explainers"
SECTIONS = ["QUESTION", "WHY", "BACKGROUND", "DATA", "APPROACH", "FINDINGS", "HOW SURE", "NEXT"]
SCHEMATIC = "Schematic, not data."
READ_WF = "How we answered it: read left to right. The last panel shows the measured result."
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
    est = pd.read_csv(RES / "estimator_sensitivity/summary.csv", index_col=0)
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
        "est_tabpfn": est.loc["tabpfn_default", "balanced_accuracy"],
        "est_kumo": est.loc["kumo_default", "balanced_accuracy"],
    }


def yes(flag: bool) -> str:
    return "Supported" if flag else "Not supported"


def build() -> Deck:
    n = numbers()
    v = n["verdicts"]
    kt = n["kumo_vs_tuned"]
    h4 = v["H4a"]["supported"] and v["H4b"]["supported"] and v["H4c"]["supported"]
    slides = [
        Title("Do pretrained models for tables help on a small biological dataset?",
              "TabPFN v2 and Kumo Tabular compared with conventional models on 72 mice",
              document_author(),
              notes="A practical question: do pretrained ‘foundation’ models for tables actually "
                    "help when every sample is an animal? No machine-learning background is "
                    "assumed."),
        Claim("The talk follows eight steps",
              EXPLAIN / "story_map.png", "The same outline is used in the report and README.",
              notes="The footer shows where we are at every point."),

        # ------------------------------------------------------------ 1 QUESTION
        Section(1, "QUESTION", "Do tabular foundation models beat conventional models when "
                "there are only 72 samples?",
                notes="One sentence to remember."),
        TableSlide("Four questions, each with a hypothesis written down in advance",
                   ["Question", "Hypothesis written in advance"],
                   [["RQ1 · More accurate?", "Each foundation model beats every conventional model"],
                    ["RQ2 · Better probabilities?", "Lower log loss than every conventional model"],
                    ["RQ3 · Best with least data?", "Bigger lead with 2 than with 6 mice per group"],
                    ["RQ4 · Trustworthy scores?", ("Chance with shuffled labels; inflated by "
                                                  "leakage; no memorisation")]],
                   [1.6, 3.6],
                   notes="Committed to the repository before the experiments ran."),

        # ------------------------------------------------------------ 2 WHY
        Section(2, "WHY", "Why does this matter for science?",
                notes="Motivation."),
        Claim("Biology tables are small because every sample is an animal",
              EXPLAIN / "small_data.png", SCHEMATIC,
              notes="Tens to hundreds of samples is normal in biology. Foundation models are "
                    "said to help most exactly here, so the claim is worth testing carefully."),

        # ------------------------------------------------------------ 3 BACKGROUND
        Section(3, "BACKGROUND", "What is a tabular foundation model?",
                notes="One idea, then the line-up."),
        Claim("A foundation model reads the training data as context instead of fitting to it",
              FIG / "primer_tfm.png",
              "Schematic. Both foundation models tested here were pretrained on synthetic tables.",
              notes="A conventional model learns weights from your training rows alone. A "
                    "foundation model was trained once on millions of synthetic tables, then "
                    "predicts new rows by reading your training rows as context, in one pass."),
        Points("Eight models, one task",
               [("Six conventional models",
                 ("class prior, logistic regression, elastic net, support-vector machine, "
                  "random forest, gradient boosting")),
                ("TabPFN v2 (Prior Labs)", "a widely used tabular foundation model"),
                ("Kumo Tabular Small (NVIDIA)", "a newer tabular foundation model")],
               notes="Exact versions: tabpfn 2.2.1, structured-data-models 0.3.0, CPU, four "
                     "inference estimators each (TabPFN’s own default is 8)."),
        TableSlide("Two scores used throughout",
                   ["Score", "Meaning"],
                   [["Balanced accuracy", "Average hit rate across the 8 groups; chance = 0.125"],
                    ["Log loss", "Penalty for confident wrong answers; lower is better"]],
                   [1.4, 4.0],
                   notes="Accuracy scores the guesses; log loss scores the probabilities."),

        # ------------------------------------------------------------ 4 DATA
        Section(4, "DATA", "The test bed: 72 mice, 77 proteins, 8 groups",
                notes="Public data from the UCI Machine Learning Repository."),
        Claim("The task: predict which of eight experimental groups a new mouse belongs to",
              ROOT / "assets/external/higuera2015_fig1A.png", HIGUERA,
              notes="Groups come from three yes/no choices: Down-syndrome model or control, "
                    "learning protocol or not, drug or placebo. 7–10 mice per group."),
        Claim("The table has 1,080 rows, but every 15 rows are the same mouse",
              PRIMER / "data_anatomy.png",
              "Real rows (archive.ics.uci.edu/dataset/342). We average each mouse’s 15 "
              "readings into one profile of 77 proteins.",
              notes="This is the single most important fact about the data."),
        BigNumber("So there are only 72 independent samples to learn from and test on",
                  "72 mice",
                  "77 proteins per mouse · 8 groups · chance accuracy 0.125 (1 in 8)",
                  "Every model is always tested on mice it has never seen.",
                  notes="Small by any standard, which is the point."),

        # ------------------------------------------------------------ 5 APPROACH
        Section(5, "APPROACH", "How do you compare models fairly with 72 mice?",
                notes="Fair comparison is the core of the method."),
        Claim("Four rules for a fair comparison",
              EXPLAIN / "fair_rules.png", SCHEMATIC,
              notes="Whole mice held out; identical splits; corrected statistics; planned "
                    "versus later analyses kept apart."),
        Claim("Rule 1 · Splitting rows instead of mice makes every model look nearly perfect",
              FIG / "fig2_leakage.png",
              "Same 1,080 rows; only the split rule changes.",
              notes="With random row splits the model has already seen other readings of the "
                    "test mouse. Foundation models are as vulnerable as conventional ones."),
        Claim("Rule 4 · What was planned, and what was added after seeing results",
              EXPLAIN / "timeline.png", SCHEMATIC,
              notes="Anything added later is labelled exploratory on every slide that shows it."),

        # ------------------------------------------------------------ 6 FINDINGS
        Section(6, "FINDINGS", "Four questions, four answers",
                notes="For each question: how we tested it, then the answer."),
        Claim("RQ1 · Is a foundation model more accurate?",
              WF / "wf_rq1_accuracy.png", READ_WF,
              notes="Every model sees the same held-out mice; the paired test accounts for "
                    "overlapping training sets."),
        ClaimSplit("RQ1 · Planned test: Kumo beats all six baselines; TabPFN does not beat "
                   "logistic regression",
                   SLIDE / "accuracy.png",
                   [f"Kumo {n['kumo']:.2f}, TabPFN {n['tabpfn']:.2f}, logistic {n['logistic']:.2f}",
                    (f"Kumo vs logistic: +{n['kumo_vs_log']['mean_difference']:.2f} "
                     f"(p = {n['kumo_vs_log']['p_holm']:.3f})"),
                    (f"TabPFN vs logistic: +{n['tab_vs_log']['mean_difference']:.2f} "
                     f"(p = {n['tab_vs_log']['p_holm']:.2f}), not distinguishable"),
                    "The simplest conventional model was the strongest"],
                   "Balanced accuracy on held-out mice; chance = 0.125.",
                   notes="Pre-specified result. The baseline logistic regression used a fixed "
                         "setting, which matters on the next slide."),
        Claim(f"RQ1 · Exploratory: against a tuned logistic regression, Kumo’s lead "
              f"(+{kt['mean_difference']:.2f}) is no longer statistically clear",
              FIG / "fig3_tuned_benchmark.png",
              f"Tuned logistic {n['tuned_log']:.3f}. Kumo minus tuned logistic: 95% interval "
              f"{kt['lower']:+.2f} to {kt['upper']:+.2f}, p = {kt['p']:.2f} (post hoc).",
              notes="After seeing the results we tuned the conventional models more widely. The "
                    "estimate still favours Kumo, but the interval includes zero. CatBoost and "
                    "XGBoost were not tested."),
        Claim("RQ2 · Are its probabilities better?",
              WF / "wf_rq2_calibration.png", READ_WF,
              notes="Log loss rewards confident correct answers and punishes confident wrong ones."),
        ClaimSplit("RQ2 · Kumo gives the best probabilities, though that alone does not prove "
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
        Claim("RQ3 · Do they help most when data are scarcest?",
              WF / "wf_rq3_learning.png", READ_WF,
              notes="We train on 2 to 6 mice per group and track each model’s accuracy."),
        ClaimSplit("RQ3 · Planned test: the lead did not shrink with more data, so the "
                   "hypothesis fails",
                   SLIDE / "gap.png",
                   [(f"Kumo’s lead: {n['h3']['kumo']['gap_small']:+.2f} at 2 per group, "
                     f"{n['h3']['kumo']['gap_large']:+.2f} at 6"),
                    (f"TabPFN’s lead: {n['h3']['tabpfn']['gap_small']:+.2f} → "
                     f"{n['h3']['tabpfn']['gap_large']:+.2f}"),
                    "H3 not supported for either model",
                    "Test mice changed with training size, so the curve is only indicative"],
                   "Lead = foundation model minus logistic regression.",
                   notes="With only 16 training mice, no model has much to work with."),
        Claim("RQ3 · Exploratory: a matched curve varies widely, so there is no dependable "
              "small-data pattern",
              FIG / "fig4_matched_learning.png",
              f"Same test mice at every size. 16 mice: Kumo {n['k16']:.2f}, logistic "
              f"{n['l16']:.2f}. 48 mice: Kumo {n['k48']:.2f}, logistic {n['l48']:.2f}.",
              notes="The planned curve changed the test mice as the training set grew; this post "
                    "hoc version fixes that. The average gap grows, but varies a lot between "
                    "draws."),
        Claim("RQ4 · Can the scores be trusted?",
              WF / "wf_rq4_validity.png", READ_WF,
              notes="Three controls, each with a pass/fail rule written in advance: shuffled "
                    "labels, leaky splits, hidden dataset identity."),
        Claim("RQ4 · All three controls behave as expected",
              FIG / "fig4_validity.png",
              "Shuffled labels give chance; leakage inflates every model; hiding the dataset’s "
              "identity barely changes accuracy.",
              notes="Caveat: hiding names and rescaling values cannot prove a model never saw "
                    "this public dataset. NVIDIA reports Kumo was pretrained only on synthetic "
                    "tables."),

        # ------------------------------------------------------------ 7 HOW SURE
        Section(7, "HOW SURE", "How much weight can each answer bear?",
                notes="Planned versus exploratory, informed versus blind."),
        Claim("Two moderate answers, one weak, one strong",
              EXPLAIN / "evidence.png",
              "Strength combines the planned test, later checks and known limits.",
              notes="Accuracy and probabilities are moderate because the tuned comparison is "
                    "not clear. The small-data question is weak. Controls are strong."),
        TableSlide("Verdicts on the pre-specified hypotheses",
                   ["Hypothesis", "TabPFN v2", "Kumo"],
                   [["H1 · More accurate than every baseline",
                     yes(v["H1"]["tabpfn"]["supported"]), yes(v["H1"]["kumo"]["supported"])],
                    ["H2 · Better probabilities (log loss)",
                     yes(v["H2"]["tabpfn"]["supported"]), yes(v["H2"]["kumo"]["supported"])],
                    ["H3 · Biggest lead with least data",
                     yes(v["H3"]["tabpfn"]["supported"]), yes(v["H3"]["kumo"]["supported"])],
                    ["H4 · Controls behave as expected", yes(h4), yes(h4)]],
                   [3.0, 1.4, 1.4], highlight_column=None,
                   notes="And the exploratory caveat: against a tuned logistic regression, Kumo’s "
                         "lead is not statistically clear."),
        Points("What the pre-specification can and cannot claim",
               [("Accuracy and probability hypotheses were informed, not blind",
                 "an earlier benchmark on the same data was known when they were written"),
                ("The learning-curve hypothesis and two controls were blind",
                 "their results had not been seen"),
                ("Ensemble size does not change the conclusions",
                 (f"at package defaults: TabPFN {n['est_tabpfn']:.2f}, Kumo "
                  f"{n['est_kumo']:.2f}"))],
               notes="Be candid about this."),

        # ------------------------------------------------------------ 8 NEXT
        Section(8, "NEXT", "What should a scientist take from this?",
                notes="Lessons and open questions."),
        Points("Four lessons for using foundation models on scientific tables",
               [("Count independent units, not rows", "split by animal, patient or experiment"),
                ("Compare against a well-tuned simple model",
                 "logistic regression was hard to beat"),
                ("Report probabilities and use corrected statistics",
                 "fold-to-fold differences are not independent"),
                ("Check for contamination, but know its limits",
                 "hiding names and scales cannot prove a model never saw the data")],
               notes="Each of these changed or protected a conclusion in this study."),
        TableSlide("What remains open",
                   ["Open question", "Why it matters"],
                   [["Stronger conventional baselines", "CatBoost and XGBoost were not tested"],
                    ["Newer model versions", "results apply to the versions tested"],
                    ["An independent cohort", "all results come from one set of 72 mice"],
                    ["Where Kumo’s edge comes from", "why Kumo and TabPFN differ is untested"]],
                   [2.4, 4.0],
                   notes="Close with the open questions."),
        Points("Three things to remember",
               [("Kumo led every model we tested", "and was clearly ahead in the planned test"),
                ("Its edge over a tuned logistic regression is uncertain",
                 "in a later exploratory comparison"),
                ("No dependable small-data advantage",
                 "the planned hypothesis failed and the matched curve varies widely")],
               notes="Everything is reproducible with make fm-reproduce."),
    ]
    return Deck("Tabular foundation models on 72 mice", SECTIONS, slides, document_author())


def main() -> None:
    deck = build()
    render(deck, STUDY / "presentation")
    print(f"wrote {len(deck.slides)} slides to foundation_models/presentation/")


if __name__ == "__main__":
    main()
