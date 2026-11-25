# Analysis decisions

This log records every consequential choice in the mechanistic analysis, why it was made, and
what it protects against. It is written for someone who wants to understand or challenge the
analysis without reading the code. Code locations are given as `module.function`.

## 1. Experimental unit

| Decision | Rationale |
|---|---|
| One value per mouse in every analysis | The 15 rows per mouse are technical repeats. Treating them as independent inflates sample size 15-fold and leaks identity across CV folds (accuracy 0.99 → 0.60). |
| Mouse value = mean of log₂ intensities (`data.log2_animal_matrix`) | Log scale makes group differences interpretable as fold changes; the mean of logs is the geometric mean, robust to multiplicative assay noise. |
| One non-positive value treated as missing | log₂ is undefined at ≤ 0; one RRP1 value is affected. |

## 2. Data quality

| Decision | Rationale |
|---|---|
| Drop `pS6_N` (`data.duplicate_columns`) | It is an exact copy of `ARC_N`; keeping both double-counts one signal. |
| Infer five technical blocks of three rows (`data.dilution_block`) | Rows 1–3, 4–6, … co-vary strongly within mice (Fig. 1b), consistent with 3 replicates × 5 dilutions. Used for QC only; blocks are balanced, so mouse means are unaffected. |
| Exclude proteins missing for ≥ 5 whole mice from multivariate analyses (`data.analysis_panel`) | BAD, BCL2, EGR1, H3AcK18, H3MeK4 and pCFOS lack 5–19 mice; imputing that much would dominate the signal. Missingness was tested and is not associated with experimental cell. They remain in per-protein contrasts, which use available mice. |
| Two remaining missing mouse means set to 0 after standardisation | Label-free; affects 2 of 5,040 values. |

## 3. Inference for group comparisons

| Decision | Rationale |
|---|---|
| Planned contrasts of cell means with Welch SEs (`contrasts.contrast_table`) | Allows unequal variances and unequal cell sizes (7–10); every question maps to an explicit contrast. |
| Main effects = difference of level means; interactions = average difference of differences | Makes effect sizes directly interpretable and comparable across terms. |
| Benjamini–Hochberg FDR within each contrast | Controls false discoveries across proteins for each question separately. |

## 4. Multivariate signatures

| Decision | Rationale |
|---|---|
| Combine proteins into an axis and score each mouse (`signatures.axis_scores`) | One score per mouse gives far more power than 70 separate tests at n = 7–10 per cell, and allows three-way questions. |
| Leave-one-out scoring for mice that define an axis | Otherwise those mice are biased toward the axis (self-projection), which would exaggerate the reference group. The closed form is verified against brute-force refitting in tests. |
| Bootstrap leaves out *all* copies of a resampled mouse | A duplicate left in the axis reintroduces self-projection bias. |
| Standardise by pooled within-cell SD | Units become "SD of biological variation between comparable mice", comparable across proteins. |
| Noise-corrected norms and cosines (`signatures.debiased_squared_norm`, `disattenuated_cosine`) | Raw norms of noisy difference vectors are biased upward and raw cosines toward 0; the corrections are unbiased (verified by simulation). |

## 5. Rescue without regression to the mean

The tempting analysis (select proteins changed by trisomy in saline mice, then test whether
memantine reverses them in the same mice) shares the Ts65Dn-saline cell between selection and
test. Noise that made a protein look trisomy-affected is undone by chance in the drug comparison,
manufacturing "rescue". We therefore:

* measure global rescue on a leave-one-out trisomy axis (memantine mice are out-of-sample);
* compare trisomy effects estimated in *disjoint* mice (saline vs memantine) per protein; and
* correct the across-protein slope for measurement error (reliability-adjusted).

## 6. Resampling

| Decision | Rationale |
|---|---|
| Bootstrap mice within cells (4,000×) | Preserves the design; intervals reflect biological sampling. Percentile intervals for ratios are wide when the denominator is small (e.g. rescue in CS), which is reported rather than hidden. |
| Permute one factor within strata of the other two, recomputing the whole pipeline (5,000×) | Tests exactly the null of interest and accounts for axis estimation. |

## 7. What was deliberately *not* done

* No per-protein "rescued" list (regression to the mean, Section 5).
* No causal claims about proteins: associations only.
* No nested hyperparameter search for the mechanistic analyses: there are no tuned models.
* No claims that the foundation models generalise until the obfuscation protocol has been run
  on them (`memorization.obfuscate`; positive control in Fig. S2).

## 8. Memorisation (contamination) test — pre-specified

Written before running any foundation model on obfuscated data (2026-11-25).

* **Question.** Do TabPFN and Kumo score well because they learn from the data, or partly
  because they recognise this public dataset?
* **Manipulation.** Score each model on the original mouse-mean table and on three obfuscated
  copies (generic names, shuffled columns, random positive affine rescaling per protein), on the
  identical 5 × 5-fold splits (`scripts/memorization_test.py`).
* **Decision rule.** Evidence of reliance on dataset identity if the mean balanced-accuracy
  drop (original − obfuscated) exceeds **0.05** *and* every obfuscation key shows a drop;
  otherwise, no evidence of memorisation.
* **Positive control.** Scaled logistic regression must be exactly invariant (it is: 0.610 →
  0.610).
* **Limitation.** The test detects reliance on names, order and raw values. A model that
  memorised the *relationships* in the data would pass it, so a pass reduces but does not
  eliminate contamination concerns.

## 9. Reproducibility

* All settings: `configs/mechanisms.yaml` (seed, resample counts, thresholds).
* All numbers in paper, report and slides are generated from `results/` (no hand-typed values).
* `make reproduce` runs tests, lint, every analysis, all figures, the paper, the report and the
  slides.
