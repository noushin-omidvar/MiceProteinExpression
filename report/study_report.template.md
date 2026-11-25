---
title: "What protein levels reveal about trisomy, learning and memantine"
subtitle: "A plain-language report on the Ts65Dn mouse protein study"
---

<!-- Generated from report/study_report.template.md by scripts/make_report.py.
     Edit the template, not study_report.md: $$Name placeholders are filled from results/. -->

## Summary

A public dataset of brain proteins from a Down syndrome mouse model is widely used to test
machine-learning methods. We analysed it as the biological experiment it is and found the
following.

- **The usual benchmark is misleading.** The data hold $NRows measurements but only $NMice
  mice. Counting measurements as mice inflates classification accuracy from $GroupedBA to
  $RowBA.
- **Q1 · Extra genes, little extra protein.** Genes present in three copies instead of two
  produce only about $DosageFrac% of the expected extra protein.
- **Q2 · A weaker response to learning.** The trisomic brain's protein response to learning is
  about a third smaller than normal, and the drug memantine restores it.
- **Q3 · A partial, non-specific rescue.** Memantine moves trisomic mice $RescueFrac% of the way
  back toward normal mice, but it moves normal mice in the same direction.
- **Q4 · No network rewiring detected.** Proteins co-vary in the same way in both kinds of mice,
  although only large differences could have been detected.

![The study at a glance. Schematic; numbers are generated from the results.](../results/mechanisms/figures/fig0_overview.png)

## 1. The problem

**What the data are.** In 2015 a research team measured 77 proteins in the brains of 72 mice.
Some mice model Down syndrome, some had learned to fear a place, and some received a drug. They
published the data in the UCI Machine Learning Repository
(<https://archive.ics.uci.edu/dataset/342/mice+protein+expression>) so that others could
analyse them.

**How the data have been used.** Most later work treats the dataset as a puzzle for algorithms:
given a mouse's protein levels, can a model guess which of the eight groups it came from?

**What is wrong with that.**

1. *It overstates accuracy.* Each mouse was measured 15 times. When the measurements are split
   at random into training and test data, the same mouse appears in both, and the model can
   score well by recognising individual mice rather than learning the biology.
2. *It answers the wrong question.* Even an accurate classifier tells us only that the groups
   differ. It does not tell us *how* they differ, which is what the experiment was designed to
   find out.

**Why it matters.** Down syndrome is the most common genetic cause of intellectual disability.
Knowing what an extra chromosome does to brain proteins, and whether a drug can undo it, is
the kind of evidence that guides treatment research.

## 2. Background you need

| Term | What it means here |
|---|---|
| **Down syndrome** | Caused by an extra copy of human chromosome 21 |
| **Ts65Dn mouse** | Carries an extra copy of about 90 genes that are also on human chromosome 21; widely used to model Down syndrome |
| **Fear conditioning** | A memory test. **CS** mice explore a box and then receive a mild shock, so they learn to fear the box. **SC** mice are shocked immediately and cannot link the shock to the box. Comparing CS with SC isolates the effect of learning |
| **Memantine** | A drug that blocks NMDA receptors, a type of receptor involved in memory. Earlier work showed it restores learning in Ts65Dn mice |
| **The experiment** | 2 genotypes × 2 conditioning types × 2 treatments = 8 groups of 7–10 mice; 77 proteins measured in the cortex |

## 3. Our questions

| | Question | Why it matters |
|---|---|---|
| **Q1** | Do three copies of a gene give 1.5 times as much protein? | Tests the simplest explanation of trisomy |
| **Q2** | Does trisomy change the brain's protein response to learning? | Links molecules to the known learning deficit |
| **Q3** | Does memantine move the trisomic brain back toward normal? | Asks how the drug works |
| **Q4** | Does trisomy change how proteins work together? | Effects may lie in coordination, not in levels |

## 4. How we answered them

| Step | What we did | Why |
|---|---|---|
| 1. Count mice, not rows | Averaged each mouse's 15 measurements into one value per protein | Measurements of the same mouse are not independent |
| 2. Check the data | Found that the 15 measurements form five technical blocks of three, removed one duplicated column, and checked that missing values are random | Problems in the data would distort every result |
| 3. Build signatures | Combined proteins into a single score per mouse that captures one effect, such as learning or trisomy | One score per mouse is far more reliable with 7–10 mice per group than 77 separate tests |
| 4. Avoid self-flattery | Scored each mouse on a signature rebuilt without that mouse | A mouse that helped define a signature would look artificially typical on it |
| 5. Measure uncertainty | Re-drew mice $NBoot times (bootstrap) and shuffled group labels $NPerm times (permutation tests) | Shows how much results could change by chance |
| 6. Test the tools | Checked every shortcut against a slow, exact version or a simulation with a known answer | Ensures the method works before it touches real data |

## 5. What we found

### First: why the benchmark misleads

![Accuracy of eight-group prediction under three ways of splitting the data.](../results/figures/validation_comparison.png)

*How to read this figure:* each dot is one test of a classifier. The left group splits rows at
random; the other two hold out whole mice. The dashed line is chance (1 in 8).

When the same mouse can appear in training and test data, accuracy is almost perfect
($RowBA). When whole mice are held out, it falls to $GroupedBA. The near-perfect scores often
reported for this dataset mostly measure recognition of individual mice.

### Q1 · Do three gene copies give 1.5 times as much protein?

**Short answer:** no. They give about $DosageFrac% of the expected increase.

![Average change in protein level in Ts65Dn mice for three groups of proteins.](../results/mechanisms/figures/slides/dosage.png)

*How to read this figure:* each point is the average change (with 95% interval) for a group of
proteins. The dashed red line is where triplicated genes would sit if three copies meant 1.5
times the protein.

The six proteins from triplicated genes rose by $DosageFold-fold on average (95% interval
$DosageTriCI on the log₂ scale). Two genes that are on human chromosome 21 but *not*
triplicated in this mouse did not change ($DosageDis), which shows the effect follows copy
number. Estimates from two separate groups of mice agreed (r = $DosageRep).

**Caveat:** the shortfall could come from the measurement method compressing large differences,
from measuring only the cell nucleus, or from cells compensating for the extra copies. These
data cannot tell these explanations apart.

### Q2 · Does trisomy change the brain's response to learning?

**Short answer:** yes. The response is about a third weaker, and memantine restores it.

![Size of the protein response to learning in each group, relative to normal mice on saline.](../results/mechanisms/figures/slides/learning.png)

*How to read this figure:* 1.0 is the response of normal mice on saline. Points below 1 are
weaker responses. Brackets show permutation tests.

We defined a *learning signature* as the protein difference between mice that learned (CS) and
mice that could not (SC), using normal mice on saline. Measured on that signature, trisomic
mice on saline responded at **$LearnRelTsSal** times the normal level (95% interval
$LearnRelTsSalCI; p = $LearnAttP). With memantine they responded at **$LearnRelTsMem** times
(p = $LearnResP for the improvement). The same proteins respond in every group; only the size
of the response changes. This mirrors behaviour: Ts65Dn mice fail this memory task, and
memantine rescues them.

**Caveat:** the dataset has no learning score for each mouse, so the link to behaviour holds at
the group level only. The upper end of the interval is close to 1, so this result needs
replication.

### Q3 · Does memantine move the trisomic brain back toward normal?

**Short answer:** partly, but it moves normal mice too, so it is not a targeted repair.

![Average position of each group on the trisomy signature.](../results/mechanisms/figures/slides/rescue.png)

*How to read this figure:* the vertical axis measures how "trisomic" a group's protein profile
looks. Arrows show the effect of memantine within each genotype.

Memantine moved trisomic mice $RescueFrac% of the way back toward normal mice on saline
(p = $RescueP). It also moved normal mice the same way, so compared with normal mice that also
received the drug, $Persist% of the trisomy difference remains. Most of memantine's effect on
the trisomic brain points in other directions entirely.

**A trap we avoided.** A tempting shortcut is to pick the proteins most changed by trisomy and
then check whether the drug "fixes" them in the same mice. That makes rescue look larger than it
is, through regression to the mean. We measured the two effects in separate mice instead.

### Q4 · Does trisomy change how proteins work together?

**Short answer:** we found no evidence that it does.

![How different the two genotypes' correlation patterns are, compared with chance.](../results/mechanisms/figures/slides/network.png)

*How to read this figure:* the blue bars show the differences expected by chance, obtained by
shuffling genotype labels. The red line is the observed difference. It falls within the chance
range (p = $NetP).

**Caveat:** with about 34 mice per genotype only large changes would be visible, so this result
is inconclusive rather than a proof of no effect.

### Can we trust the best prediction scores?

Two large pretrained models score highest at predicting a held-out mouse's group (Kumo
$KumoBA, TabPFN $TabpfnBA, against $ElasticBA for the best conventional model). Because the
dataset is public, these models might have seen it during training. We tested this by renaming,
reordering and rescaling every protein, which keeps the information but hides the dataset's
identity. The decision rule was fixed in advance: a drop of more than 0.05 that appears with
every obfuscation key would count as evidence of memorisation.

| Model | Original data | Obfuscated data | Verdict |
|---|---|---|---|
| Logistic regression (control) | $MemLogisticOrig | $MemLogisticObf | $MemLogisticVerdict |
| TabPFN v2 | $MemTabpfnOrig | $MemTabpfnObf | $MemTabpfnVerdict |
| Kumo Tabular Small | $MemKumoOrig | $MemKumoObf | $MemKumoVerdict |

A pass rules out reliance on names, order and exact values. It cannot rule out a model that
memorised the relationships in the data.

## 6. What this means and what comes next

- **For researchers using this dataset:** split by mouse, never by row.
- **For Down syndrome biology:** the trisomic cortex responds to learning in the right way but
  too weakly, and memantine restores the size of that response. That is a concrete molecular
  hypothesis to test.
- **For drug research:** memantine's protein effects are broad rather than targeted to trisomy.
- **For the next experiment:** about $NAttEighty mice per group would confirm the learning
  result with 80% power. Single-protein interaction effects would need $NThreeTwo–$NThreeOne
  mice per group. Include female mice, an independent cohort and per-mouse learning scores.

## 7. Limitations

- One laboratory, one measurement method, male mice only, and no independent replication.
- CS and SC differ in timing as well as in learning, so the learning signature also contains
  protocol differences.
- The replicate and dilution structure is inferred from the data, not documented.

## 8. Glossary

| Term | Meaning |
|---|---|
| Balanced accuracy | Average of the accuracies for each group; 0.125 is chance for 8 groups |
| Bootstrap | Re-drawing mice with replacement many times to measure uncertainty |
| Confidence interval (95%) | A range of values compatible with the data |
| False discovery rate (q) | The expected share of false positives among findings reported as significant |
| log₂ fold change | A ratio on a doubling scale: 1 means double, −1 means half, 0.58 means 1.5 times |
| p-value | How often chance alone would give a result at least this extreme |
| Permutation test | Shuffling group labels to see how often chance produces an effect this large |
| Signature | A weighted combination of proteins that captures one effect |

## 9. Where everything is

| What | Where |
|---|---|
| Full scientific paper | `paper/main.pdf` |
| Slides with speaker notes | `presentation/presentation.pptx` (and `.pdf`) |
| Every decision and its rationale | `docs/analysis_decisions.md` |
| All numbers | `results/mechanisms/summary.json` and the `results/` tables |
| Reproduce everything | `make install && make reproduce` |
