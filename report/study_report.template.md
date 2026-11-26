---
title: "What does an extra chromosome change in the learning brain?"
subtitle: "A plain-language report on brain proteins in the Ts65Dn mouse model of Down syndrome"
---

<!-- Generated from report/study_report.template.md by scripts/make_report.py.
     Edit the template, not study_report.md: $$Name placeholders are filled from results/. -->

## At a glance

We measured how 77 brain proteins differ between mice that carry extra copies of Down-syndrome
genes and normal mice, while they learn and after a drug called memantine. The data are public
(UCI Machine Learning Repository,
<https://archive.ics.uci.edu/dataset/342/mice+protein+expression>).

![The report follows eight steps; the slides and README use the same outline.](../results/figures/explainers/story_map.png)

| Question | Short answer | How sure |
|---|---|---|
| **Q1** Do extra gene copies make extra protein? | Yes, but only about $DosageFrac% of what the extra copies predict | Strong |
| **Q2** Does trisomy change the brain's response to learning? | It looks weaker ($LearnRelTsSal× normal) and near normal with memantine | Weak |
| **Q3** Does memantine move trisomic mice back toward normal? | Possibly ($RescueFrac% of the gap), but by a very uncertain amount | Weak |
| **Q4** Does trisomy rewire how proteins work together? | No difference detected; only large changes were detectable | Weak |

One practical lesson applies to anyone who uses this dataset: **count mice, not rows.** The
table has $NRows rows but only $NMice mice; treating rows as mice inflates prediction accuracy
from $GroupedBA to $RowBA.

## 1. The question

**What changes in the brain's proteins when a mouse carries extra copies of Down-syndrome
genes, while it learns and after the drug memantine?**

![One experiment, four questions. Schematic; numbers are generated from the results.](../results/mechanisms/figures/fig0_overview.png)

## 2. Why it matters

Down syndrome is the most common genetic cause of intellectual disability. Two facts are
already known:

- A mouse model carrying extra copies of the same genes **fails a simple memory test**.
- The drug **memantine restores the mice's learning** in earlier behavioural experiments.

What happens to brain proteins between the extra genes and the behaviour is still unclear.
Proteins are where that missing step can be measured.

## 3. Background in pictures

Four ideas are enough to follow the rest of the report.

**Idea 1 · The mouse model.** People with Down syndrome have three copies of chromosome 21.
Ts65Dn mice carry an extra piece of mouse chromosome 16 with about 90 of the same genes.

![Chromosomes in a typical person, in Down syndrome and in the Ts65Dn mouse. Schematic.](../results/figures/explainers/chromosomes.png)

**Idea 2 · Proteins.** Genes are recipes; proteins are the machines they make. Many proteins
are switched on by a small chemical tag (phosphorylation). Protein names that start with *p*,
such as pERK, are the tagged forms.

![From gene to protein to switched-on protein. Schematic.](../results/figures/explainers/gene_protein.png)

**Idea 3 · The memory test.** Mice learn to fear a cage only if the shock comes *after* they
explore it (CS). Mice shocked first (SC) receive the same shock but cannot make the link. CS
minus SC therefore isolates learning.

![One training session for CS and SC mice. Schematic; protocol from Ahmed et al. 2015.](../results/figures/primer/experiment_timeline.png)

**Idea 4 · The drug.** Memantine partly blocks the NMDA receptor, a channel that drives the
signalling behind memory. It is an approved Alzheimer's drug.

![How memantine acts on the NMDA receptor. Schematic.](../results/figures/explainers/memantine.png)

## 4. The data

Three yes/no choices make **eight groups of 7–10 mice**: Ts65Dn or normal, CS or SC training,
memantine or saline. Brains were collected one hour after training and 77 proteins were
measured in the cortex.

![The eight groups and their learning outcome. Reproduced from Higuera, Gardiner & Cios, PLoS ONE 10(6): e0129126 (2015), Fig 1A (cropped), CC BY 4.0.](../assets/external/higuera2015_fig1A.png)

Each brain sample was measured **15 times** (3 replicates × 5 dilutions), so every mouse fills
15 rows of the table.

![From one brain to 15 numbers per protein. Schematic.](../results/figures/primer/measurement.png)

![The data table (real rows): every 15 rows are one mouse.](../results/figures/primer/data_anatomy.png)

> **Key fact:** $NRows rows, but only **$NMice independent mice**. Every analysis below uses
> one value per mouse.

**What these data cannot tell us:** which individual mouse learned (no per-mouse memory score
was recorded); cause and effect for single proteins; anything about females, other ages or
other brain regions.

## 5. Our approach

Seventy-two mice is a small sample. Four rules keep the answers trustworthy.

![Four rules for trustworthy answers from 72 mice. Schematic.](../results/figures/explainers/approach.png)

**Rule 1 in action.** When rows are split at random, the same mouse sits in both training and
test data, and a classifier learns to recognise mice rather than groups.

![Eight-group prediction accuracy under three ways of splitting the data. Dashed line: chance (1 in 8).](../results/figures/validation_comparison.png)

**Rule 2 in action.** With 7–10 mice per group, 70 separate protein tests are noisy. A
*signature* combines the proteins into one score per mouse for each effect.

![How a signature turns many protein levels into one score per mouse. Schematic.](../results/figures/explainers/signature.png)

Rules 3 and 4: each mouse is scored on a signature rebuilt without it, and chance is measured
by re-drawing mice $NBoot times (bootstrap) and shuffling labels $NPerm times within the design
(permutation tests). Every shortcut was checked against an exact calculation or a simulation.

## 6. What we found

Each question has two pictures: **how we answered it** (a workflow, read left to right) and
**the answer** (the measured result).

### Q1 · Do extra gene copies make extra protein?

![How we answered Q1. Steps 1–4 are schematic; the last panel shows data.](../results/mechanisms/figures/workflows/wf_q1_dosage.png)

![Average change in protein level in Ts65Dn mice for three groups of proteins. Dashed line: the 1.5× expected from three gene copies.](../results/mechanisms/figures/slides/dosage.png)

**Answer.** Six proteins from triplicated genes rose by $DosageFold-fold on average (95%
interval $DosageTriCI on the log₂ scale), about **$DosageFrac% of the simple 1.5-fold
expectation**. Two chromosome-21 genes that are *not* triplicated in this mouse did not change
($DosageDis), as copy number predicts. Two separate groups of mice agreed (r = $DosageRep).

**Why it might fall short:** cells may compensate for the extra copies, or the measurement may
compress large differences. These data cannot tell the two apart.

### Q2 · Does trisomy change the brain's response to learning?

![How we answered Q2. Steps are schematic; the last panel shows data.](../results/mechanisms/figures/workflows/wf_q2_learning.png)

![Size of the learning response in each group, relative to normal mice on saline (1.0).](../results/mechanisms/figures/slides/learning.png)

**Answer.** On a learning signature defined in normal mice, trisomic mice on saline responded
at **$LearnRelTsSal× the normal level** (95% interval $LearnRelTsSalCI; p = $LearnAttP). With
memantine they responded at **$LearnRelTsMem×** (p = $LearnResP).

**Caveat:** the interval nearly reaches 1, and the link to behaviour holds only at the group
level, because no per-mouse learning score exists. This needs replication.

### Q3 · Does memantine move trisomic mice back toward normal?

![How we answered Q3. Steps are schematic; the last panel shows data.](../results/mechanisms/figures/workflows/wf_q3_rescue.png)

![Average position of each group on the trisomy signature. Arrows: effect of memantine within each genotype.](../results/mechanisms/figures/slides/rescue.png)

**Answer.** Memantine shifted trisomic mice about **$RescueFrac%** of the way toward normal
(bootstrap interval $RescueFracCI; p = $RescueP). The interval runs from no shift to beyond a
full return, and normal mice also shifted, so neither the size nor the specificity is settled.
Against drug-treated normal mice, $Persist% of the difference remains.

**A trap we avoided:** picking the proteins most changed by trisomy and then checking whether
the drug "fixes" them in the same mice exaggerates rescue (regression to the mean). We
measured the two effects in separate mice.

### Q4 · Does trisomy rewire how proteins work together?

![How we answered Q4. Steps are schematic; the last panel shows data.](../results/mechanisms/figures/workflows/wf_q4_network.png)

![Observed difference in correlation patterns between genotypes (red) against differences produced by chance (bars).](../results/mechanisms/figures/slides/network.png)

**Answer.** No difference was detected (p = $NetP). With about 34 mice per genotype only large
changes would be visible, so this is inconclusive rather than proof of no effect.

## 7. How sure we are

![Strength of evidence for each answer.](../results/figures/explainers/evidence.png)

- **The p-values are exploratory.** The analyses were refined while looking at the data, are
  not corrected across the four questions and are all two-sided. They suggest hypotheses to
  replicate.
- One laboratory, one measurement method, male mice only, no independent replication.
- CS and SC differ in timing as well as learning, so the learning signature also carries
  protocol differences.
- The replicate and dilution structure is inferred from the data, and print-run identifiers
  are unavailable, so batch effects cannot be excluded.
- Signatures omit the scored mouse, but the standardisation used all mice, so scores are not
  fully out-of-sample.

## 8. What it means and what's next

| For | Take-away |
|---|---|
| Anyone using this dataset | Split by mouse, never by row |
| Down syndrome biology | Extra gene copies show far less extra protein than expected; the learning response may be weaker |
| Drug research | Memantine's protein effects are suggestive, not a demonstrated targeted repair |

**A follow-up study needs:** about **$NAttEighty mice per group** to confirm the weaker learning
response with 80% power; $NThreeTwo–$NThreeOne mice per group for single-protein interaction
effects; female mice; an independent cohort; and a memory score for every mouse.

## Glossary

| Term | Meaning |
|---|---|
| Balanced accuracy | Average of the per-group hit rates; 0.125 is chance for 8 groups |
| Bootstrap | Re-drawing mice with replacement many times to measure uncertainty |
| 95% interval | A range of values compatible with the data |
| log₂ fold change | A ratio on a doubling scale: 1 = double, −1 = half, 0.58 = 1.5 times |
| p-value | How often chance alone would give a result at least this extreme |
| Permutation test | Shuffling labels to see how often chance produces an effect this large |
| Signature | A weighted combination of proteins that captures one effect |

## Where everything is

| What | Where |
|---|---|
| Full scientific paper | `paper/main.pdf` |
| Slides with speaker notes | `presentation/presentation.pptx` (and `.pdf`) |
| Every decision and its rationale | `docs/analysis_decisions.md` |
| All numbers | `results/mechanisms/summary.json` and the `results/` tables |
| Reproduce everything | `make install && make reproduce` |
