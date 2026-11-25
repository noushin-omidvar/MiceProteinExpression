"""Biological annotations of the 77 measured proteins.

Every annotation used by an analysis lives here so that the biological assumptions
behind a result are explicit, reviewable, and testable.

Gene-dosage annotation
----------------------
Ts65Dn mice carry a freely segregating marker chromosome containing the distal
segment of mouse chromosome 16 (Mmu16, from *Mrpl39* to the telomere, roughly 90
protein-coding orthologs of human chromosome 21 [HSA21]) plus a small centromeric
segment of Mmu17 that has no HSA21 orthologs (Duchon et al., 2011; Reinholdt et al.,
2011). Orthologs of distal HSA21 genes map to Mmu10 and Mmu17 and are **not**
triplicated in Ts65Dn.

* ``TRISOMIC_TS65DN`` - measured proteins whose genes lie in the triplicated Mmu16
  segment (*App*, *Sod1*, *Dyrk1a*, *Itsn1*, *Rcan1* [=DSCR1], *Tiam1*). These are the
  proteins encoded by HSA21 orthologs and also listed as trisomic in Ts65Dn by
  Ahmed et al. (2015).
* ``HSA21_DISOMIC_TS65DN`` - measured HSA21-ortholog proteins whose mouse genes are
  on Mmu10 and therefore remain at two copies in Ts65Dn (*Rrp1*, *Adarb1*). These
  form a built-in negative control: they share the HSA21 annotation but not the
  dosage change.

Phospho/total pairs
-------------------
The panel quantifies several proteins both in total and as a specific phospho-form.
The log-ratio ``log2(phospho) - log2(total)`` is a relative index of the fraction of
the protein in the phosphorylated state, which is a closer proxy for pathway activity
than either abundance alone. It is not a stoichiometry: reverse-phase protein array
(RPPA) intensities are antibody-specific and not on a common absolute scale.

``pS6_N`` is excluded everywhere because it is an exact copy of ``ARC_N`` in the
public workbook (see :func:`mice_protein.data.duplicate_columns`).
"""

from __future__ import annotations

import math

TRISOMIC_TS65DN: tuple[str, ...] = (
    "APP_N",
    "SOD1_N",
    "DYRK1A_N",
    "ITSN1_N",
    "DSCR1_N",
    "TIAM1_N",
)
HSA21_DISOMIC_TS65DN: tuple[str, ...] = ("RRP1_N", "ADARB1_N")

#: Expected trisomic/disomic protein ratio on the log2 scale under strict gene dosage.
EXPECTED_DOSAGE_LOG2FC: float = math.log2(1.5)

#: (phospho-form, total-protein) column pairs.
PHOSPHO_PAIRS: tuple[tuple[str, str], ...] = (
    ("pBRAF_N", "BRAF_N"),
    ("pMEK_N", "MEK_N"),
    ("pERK_N", "ERK_N"),
    ("pRSK_N", "RSK_N"),
    ("pELK_N", "ELK_N"),
    ("pAKT_N", "AKT_N"),
    ("pMTOR_N", "MTOR_N"),
    ("pP70S6_N", "P70S6_N"),
    ("pCAMKII_N", "CAMKII_N"),
    ("pCREB_N", "CREB_N"),
    ("pNR1_N", "NR1_N"),
    ("pNR2A_N", "NR2A_N"),
    ("pNR2B_N", "NR2B_N"),
    ("pJNK_N", "JNK_N"),
    ("pGSK3B_N", "GSK3B_N"),
    ("pPKCAB_N", "PKCA_N"),
    ("pNUMB_N", "NUMB_N"),
)

#: Pathway modules built from phospho/total ratios (keys are display names; values
#: are the total-protein stems of the member ratios). pGSK3B is the inhibitory Ser9
#: site, so a higher ratio indicates *less* active GSK3B.
PATHWAYS: dict[str, tuple[str, ...]] = {
    "MAPK/ERK": ("BRAF", "MEK", "ERK", "RSK", "ELK"),
    "AKT-mTOR": ("AKT", "MTOR", "P70S6"),
    "CaMKII-CREB": ("CAMKII", "CREB"),
    "NMDA receptor": ("NR1", "NR2A", "NR2B"),
    "JNK stress": ("JNK",),
    "GSK3B (Ser9, inhibitory)": ("GSK3B",),
    "PKC": ("PKCA",),
}


def ratio_name(total: str) -> str:
    """Return the display name of a phospho/total ratio from its total column."""
    return f"p{total.removesuffix('_N')}/{total.removesuffix('_N')}"


def display(protein: str) -> str:
    """Strip the ``_N`` (nuclear fraction) suffix for figures and tables."""
    return protein.removesuffix("_N")
