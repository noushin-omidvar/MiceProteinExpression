# Data provenance

`Data/Raw_Data/Data_Cortex_Nuclear.xls` is the original UCI Mice Protein
Expression workbook (SHA-256
`68b3ce690985ad5472ba82b3f2eb0e8c4e4eabaa10a4b5fd7f90079b2da245b7`).
It contains 1,080 measurements of 77 proteins from 72 mice. Each animal has
15 dependent technical observations (three replicates of a five-point dilution
series). The code derives the animal ID from the portion of `MouseID` before
the final underscore.

Source: UCI Machine Learning Repository, <https://archive.ics.uci.edu/dataset/342/mice+protein+expression>.

The dataset is distributed by UCI under CC BY 4.0. Attribution: Higuera,
Gardiner, and Cios (2015), DOI: 10.24432/C50S3Z. The workbook is included unmodified;
redistribution requires preserving attribution.

## Known properties of the workbook

* `pS6_N` is an exact copy of `ARC_N`; analyses keep `ARC_N` only.
* One `RRP1_N` value is negative and is treated as missing on the log scale.
* Within each animal, rows 1–3, 4–6, …, 13–15 behave as five technical blocks
  (inferred; not labelled in the workbook).

Every analysis starts from this raw workbook.
