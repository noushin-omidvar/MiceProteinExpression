# Study protocol

**Title.** Do tabular foundation models help on a small biological dataset? A controlled study
of TabPFN and Kumo on cortical protein profiles from Ts65Dn mice.

**Status.** Written and committed on 2026-11-25, before the experiments in Sections 4.1–4.4 and
4.6 were run. The memorisation rule (Section 4.5) was fixed earlier the same day, before
foundation models were evaluated on obfuscated data. Deviations from this protocol are listed
in Section 8 of this document and reported in the paper.

## 1. Background and problem

Tabular foundation models (TFMs) such as TabPFN and Kumo are pretrained on very large
collections of tables and predict a new table's labels *in context*, without task-specific
training. They are claimed to excel on small datasets, which are common in biology, where each
sample is an animal or a patient and collecting more is costly.

Small biological datasets pose three problems for evaluating such claims:

1. **Few independent units.** The Mice Protein Expression data contain 1,080 rows but only
   72 mice; any comparison must hold out whole mice.
2. **Noisy comparisons.** With 72 units, differences between models are small relative to
   resampling noise, so naive comparisons over-claim.
3. **Possible contamination.** Public benchmarks may be part of a TFM's pretraining data, which
   would inflate its scores.

## 2. Research questions and hypotheses

| | Research question | Hypothesis (direction fixed in advance) |
|---|---|---|
| **RQ1** | Are TFMs more accurate than well-tuned conventional models? | H1: each TFM has higher eight-group balanced accuracy than every conventional baseline |
| **RQ2** | Are their probabilities better calibrated? | H2: TFMs have lower log loss than every conventional baseline |
| **RQ3** | Where does their advantage come from: is it largest with the least data? | H3: the TFM advantage over logistic regression is larger with 2 than with 6 training mice per group |
| **RQ4** | Are the scores valid? | H4a: all models fall to chance when labels are permuted; H4b: all models are inflated by row-level leakage; H4c: no TFM relies on dataset identity (memorisation test) |
| **RQ5** | What do they cost? | Descriptive only: fit-and-predict time |

## 3. Data, targets and models

* **Data.** UCI Mice Protein Expression (<https://archive.ics.uci.edu/dataset/342/mice+protein+expression>),
  SHA-256 `68b3ce69…45b7` (`Data/Raw_Data/Data_Cortex_Nuclear.xls`). Inputs: per-mouse mean of each of 77 proteins (72 × 77).
* **Primary target.** Eight experimental groups (genotype × conditioning × treatment).
  **Secondary targets.** Genotype, treatment and conditioning as separate binary problems.
* **Conventional baselines.** Class prior (dummy); L2 logistic regression (C = 1); elastic-net
  logistic regression, RBF support-vector machine and histogram gradient boosting, each tuned
  by 3-fold inner cross-validation on fixed small grids; random forest (500 trees, balanced
  class weights). All use median imputation and scaling fitted inside training folds.
* **Foundation models.** TabPFN v2 (`tabpfn==2.2.1`, 4 estimators, CPU) and Kumo Tabular Small
  (`structured-data-models==0.3.0`, 4 estimators, CPU), default settings, no tuning.

## 4. Experiments

### 4.1 Accuracy (RQ1)
Five repeats of five-fold cross-validation stratified on the eight groups, with **identical
folds for every model**. Primary metric: balanced accuracy over pooled out-of-fold predictions.
**Test:** for each TFM against each baseline, the corrected repeated k-fold *t*-test of
Nadeau and Bengio (2003) on the 25 paired fold-level differences, with Holm correction across
baselines within each TFM. **H1 holds** for a TFM if its advantage over every baseline is
significant at α = 0.05 after correction.

### 4.2 Calibration (RQ2)
From the same predictions: log loss (primary), multi-class Brier score and expected
calibration error (10 bins, top-label). Same test as 4.1 on fold-level log loss.

### 4.3 Data efficiency (RQ3)
For k ∈ {2, 3, 4, 5, 6} training mice per group, draw k mice per group at random for training
and test on all remaining mice; 20 draws per k, identical across models. Models: logistic
regression, random forest, SVM (default settings, no tuning possible at small k), TabPFN, Kumo.
**Test:** the TFM-minus-logistic gap at k = 2 versus k = 6, with a 95% bootstrap interval over
draws; **H3 holds** if the interval for (gap₂ − gap₆) lies above zero.

### 4.4 Validity controls (RQ4a, RQ4b)
* **Label permutation.** Labels are shuffled across mice (10 permutations, one 5-fold split
  each). **H4a holds** if every model's mean balanced accuracy lies within 0.125 ± 0.05.
* **Row-level leakage.** On all 1,080 rows, compare a random row split with a split that holds
  out whole mice (one repeat of five folds). **H4b holds** if every model's random-row accuracy
  exceeds its mouse-held-out accuracy by more than 0.2.

### 4.5 Memorisation (RQ4c), pre-specified earlier
Re-score each model after renaming, reordering and positively rescaling every protein
(3 independent keys), on identical folds. Evidence of reliance on dataset identity if the mean
balanced-accuracy drop exceeds 0.05 *and* every key shows a drop.

### 4.6 Cost (RQ5)
Wall-clock time to fit and predict one fold, single machine, CPU, two threads.

## 5. Exploratory analyses (not hypothesis tests)
Per-group recall and confusion patterns; accuracy on the three binary factor targets.

## 6. Reporting rules
All pre-specified results are reported, including those that do not support a hypothesis.
Exploratory results are labelled as such. Every number in the paper, report and slides is
generated from `results/`.

## 7. Known limitations (stated in advance)
One dataset, one cohort, CPU inference, default TFM settings, and no external test set. The
results describe this dataset and should not be generalised to all small tables.

## 8. Deviations and later work
The original analyses above were run as specified. The original protocol did not include
CatBoost, XGBoost, a newer TabPFN checkpoint, a fixed-holdout learning curve, broader
conventional tuning, or the missingness and aggregation controls. The analyses in
`configs/matched_holdout.yaml`, `configs/tuned_baselines.yaml`, and
`configs/sensitivity.yaml` were specified **after** inspecting the original results on
2026-11-25 and are exploratory. They must not be presented as preregistered tests.

The original RQ2 uses log loss, which evaluates probabilistic prediction quality; it is
**not** a standalone calibration test. The original RQ3 draws different test mice at each
training size, so its gap-versus-size comparison mixes training-size and test-composition
effects. Its bootstrap treats repeated draws from the same 72 animals as independent and
understates biological uncertainty. The fixed-holdout extension addresses the first issue,
but it cannot create independent biological replication. The original obfuscation test
cannot establish absence of pretraining contamination, because the transformations may be
invisible to preprocessing or irrelevant to memorized structure.

### Post hoc amendments (2026-11-25)

* `configs/matched_holdout_review.yaml` increases nested subset draws from one to five for
  the three headline models. It retains the same outer split seed and is post hoc.
* `svm_decision_audit.py` and `svm_main_audit.py` evaluate SVC classification using
  `predict()` with `probability=False`. Earlier balanced-accuracy values obtained from the
  argmax of `predict_proba()` are retained only as an audit trail; SVC documentation allows
  those decisions to differ. Probability metrics remain separate.
* `configs/factorized.yaml` specifies a design-informed baseline that reconstructs the
  eight-class target from three binary logistic classifiers. Product probabilities assume
  conditional independence of factors given protein features.
* The original H1/H2 corrected intervals are individual approximate intervals, not
  simultaneous multiple-comparison-adjusted intervals. Holm correction was applied within
  each foundation model's six original comparisons, not across both foundation models or
  later exploratory analyses. Primary scores pool out-of-fold predictions by repeat;
  original tests average paired **fold** differences. Test folds are small and overlapping,
  and no statistical correction supplies an independent biological cohort.
