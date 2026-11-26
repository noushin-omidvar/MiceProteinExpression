#!/usr/bin/env python3
"""Biology study deck, written for an audience new to the topic.

Outline (shared with the README and the report):
1 QUESTION · 2 WHY · 3 BACKGROUND · 4 DATA · 5 APPROACH · 6 FINDINGS · 7 HOW SURE · 8 NEXT.

Built with the shared slide engine (``mice_protein.slides``). Every content slide states one
claim as a full sentence and shows one picture that supports it; the spoken explanation is in the
speaker notes. Concept pictures are schematics and are labelled as such. Numbers come from the
same generated values as the paper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_paper import document_author, values

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

MECH = ROOT / "results/mechanisms/figures"
PRED = ROOT / "results/figures"
PRIMER = PRED / "primer"
EXPLAIN = PRED / "explainers"
WF = MECH / "workflows" / "slides"
EXTERNAL = ROOT / "assets/external"
HIGUERA = ("Reproduced from Higuera, Gardiner & Cios, PLoS ONE 10(6): e0129126 (2015), Fig 1A "
           "(cropped) · CC BY 4.0")
SECTIONS = ["QUESTION", "WHY", "BACKGROUND", "DATA", "APPROACH", "FINDINGS", "HOW SURE", "NEXT"]
SCHEMATIC = "Schematic, not data."
READ_WF = "How we answered it: read left to right. The last panel shows the measured result."


def clean(text: str) -> str:
    return text.replace("{,}", ",").replace("--", "–")


def build() -> Deck:
    v = {k: clean(x) for k, x in values(json.loads(
        (ROOT / "results/mechanisms/summary.json").read_text())).items()}
    s = MECH / "slides"
    slides = [
        Title("What does an extra chromosome change in the learning brain?",
              "Cortical protein signatures in the Ts65Dn mouse model of Down syndrome",
              document_author(),
              notes="No background is assumed. I will start with the question, give you the "
                    "four pieces of biology you need, show the data, and then answer four "
                    "questions, each with how sure we are."),
        Claim("The talk follows eight steps",
              EXPLAIN / "story_map.png", "The same outline is used in the report and README.",
              notes="This is the map. The footer shows where we are at every point."),

        # ------------------------------------------------------------ 1 QUESTION
        Section(1, "QUESTION", "What changes in the brain’s proteins when a mouse carries "
                "extra copies of Down-syndrome genes?",
                notes="One sentence to remember: what does trisomy do to the proteins of a "
                      "brain that is learning, and what does the drug memantine do?"),
        Claim("One experiment, four questions",
              MECH / "fig0_overview.png", "Schematic. Numbers are generated from the results.",
              notes="Left to right: the experiment, the problem with the table, our four "
                    "questions and a preview of the answers. We return to each question in "
                    "the findings."),

        # ------------------------------------------------------------ 2 WHY
        Section(2, "WHY", "Why do the proteins of the learning brain matter?",
                notes="Motivation."),
        Points("A learning deficit, a drug that helps, and a gap in the middle",
               [("Down syndrome is the most common genetic cause of intellectual disability",
                 "it comes from an extra copy of chromosome 21"),
                ("A mouse model carries extra copies of the same genes",
                 "and fails a simple memory test"),
                ("The drug memantine restores the mice’s learning",
                 "shown in earlier behavioural experiments"),
                ("What happens to brain proteins in between is unclear",
                 "the gap this study addresses")],
               notes="Genes and behaviour are known; the molecular step between them is not. "
                     "Proteins are where we can look for that step."),

        # ------------------------------------------------------------ 3 BACKGROUND
        Section(3, "BACKGROUND", "Four ideas in pictures",
                notes="Chromosomes, proteins, the memory test, the drug."),
        Claim("Idea 1 · Ts65Dn mice carry extra copies of ~90 Down-syndrome genes",
              EXPLAIN / "chromosomes.png", SCHEMATIC,
              notes="People with Down syndrome have three copies of chromosome 21. Ts65Dn mice "
                    "carry an extra piece of mouse chromosome 16 with about 90 of the same "
                    "genes. Like people with Down syndrome, they have learning deficits."),
        Claim("Idea 2 · Genes make proteins, and a chemical tag switches many of them on",
              EXPLAIN / "gene_protein.png", SCHEMATIC,
              notes="When I say a protein ‘rises’, I mean more of it was measured in that "
                    "group. Names starting with p, such as pERK, are tagged (switched-on) forms."),
        Claim("Idea 3 · Mice learn to fear a place only if the shock comes after exploring it",
              PRIMER / "experiment_timeline.png",
              "CS = context then shock (learning). SC = shock then context (same shock, no "
              "learning).",
              notes="Both groups get the same cage and the same mild shock; only the order "
                    "differs. So CS minus SC isolates learning. Brains were collected one hour "
                    "later."),
        Claim("Idea 4 · Memantine partly blocks a receptor that drives memory signalling",
              EXPLAIN / "memantine.png", SCHEMATIC,
              notes="Memantine is an approved Alzheimer’s drug that blocks NMDA receptors. "
                    "Here mice received 5 mg/kg 15 minutes before training. We ask what it does "
                    "to proteins, not whether it improves behaviour."),

        # ------------------------------------------------------------ 4 DATA
        Section(4, "DATA", "What was measured, and what does the table look like?",
                notes="Public data from the UCI Machine Learning Repository."),
        Claim("Eight groups of 7–10 mice; only three groups learn",
              EXTERNAL / "higuera2015_fig1A.png", HIGUERA,
              notes="Three yes/no choices make eight groups: genotype (c = control, t = Ts65Dn), "
                    "training (CS or SC) and injection (s = saline, m = memantine). Learning: "
                    "normal in control CS, failed in Ts65Dn CS on saline, rescued with "
                    "memantine, absent in every SC group."),
        Claim("Each protein was measured 15 times in every mouse",
              PRIMER / "measurement.png",
              "Reverse-phase protein arrays: each sample printed as 3 replicates × 5 dilutions.",
              notes="This is why the table has 15 rows per mouse: repeated readings of one "
                    "brain sample, not 15 mice."),
        Claim("The table has 1,080 rows, but every 15 rows are the same mouse",
              PRIMER / "data_anatomy.png",
              "Real rows from the public table (archive.ics.uci.edu/dataset/342).",
              notes="Point at the red bracket: rows 309_1 to 309_15 are all mouse 309."),
        BigNumber("So the data hold 72 independent samples, not 1,080",
                  f"{v['NRows']} rows · {v['NMice']} mice",
                  "72 male mice, 3–4 months old, cortex collected 1 hour after training",
                  "Treating rows as separate mice is the most common mistake with this dataset.",
                  notes="Everything that follows uses one value per mouse."),

        # ------------------------------------------------------------ 5 APPROACH
        Section(5, "APPROACH", "How do you get trustworthy answers from 72 mice?",
                notes="Small samples need careful methods. Four rules."),
        Claim("Four rules keep a small study honest",
              EXPLAIN / "approach.png", SCHEMATIC,
              notes=f"Mice, not rows. Signatures, not 70 separate tests. No mouse scores "
                    f"itself. Chance is measured by re-drawing mice {v['NBoot']} times and "
                    f"shuffling labels {v['NPerm']} times within the design."),
        Claim("Rule 1 · Counting rows as mice makes accuracy look near-perfect",
              PRED / "validation_comparison.png",
              f"Random row splits: {v['RowBA']}. Whole mice held out: {v['GroupedBA']}. "
              "Chance: 0.125.",
              notes="When the same mouse is in training and test data, a model recognises the "
                    "mouse rather than its group."),
        Claim("Rule 2 · A signature turns many proteins into one score per mouse",
              EXPLAIN / "signature.png", SCHEMATIC,
              notes="With 7–10 mice per group, testing 70 proteins one by one gives noisy, "
                    "mostly empty answers. A signature asks one question per effect."),

        # ------------------------------------------------------------ 6 FINDINGS
        Section(6, "FINDINGS", "Four questions, four answers",
                notes="For each question: first how we answered it, then the answer."),
        Claim("Q1 · Do extra gene copies make extra protein?",
              WF / "wf_q1_dosage.png", READ_WF,
              notes="We compare each protein’s change with what an extra gene copy predicts."),
        ClaimSplit(f"Q1 · Three gene copies give only ~{v['DosageFrac']}% of the expected "
                   "extra protein",
                   s / "dosage.png",
                   [f"Triplicated genes: +{v['DosageTri']} log₂ (95% CI {v['DosageTriCI']})",
                    f"Chromosome-21 genes that are not triplicated: {v['DosageDis']}",
                    f"Replicates in separate mice (r = {v['DosageRep']})"],
                   "Dashed line: 1.5× protein expected from three gene copies.",
                   notes="Built-in negative control: genes that are not triplicated do not "
                         "change."),
        Claim("Q2 · Does trisomy change the brain’s response to learning?",
              WF / "wf_q2_learning.png", READ_WF,
              notes="Each mouse is scored on a learning signature it did not help build; the "
                    "signature is defined in normal mice on saline."),
        ClaimSplit("Q2 · The learning response looks weaker in trisomic mice and near normal "
                   "with memantine",
                   s / "learning.png",
                   [f"Trisomic, saline: {v['LearnRelTsSal']}× normal (p = {v['LearnAttP']})",
                    f"Trisomic, memantine: {v['LearnRelTsMem']}× (p = {v['LearnResP']})",
                    "The same proteins respond in every group; only the size changes"],
                   "Learning response = CS − SC signature score, relative to control saline.",
                   notes="This mirrors behaviour, but the weakening is borderline: two-sided "
                         "p just under 0.05 and an interval that reaches 1."),
        Claim("Q3 · Does memantine undo the effects of trisomy?",
              WF / "wf_q3_rescue.png", READ_WF,
              notes="We measure how far memantine moves trisomic mice along the trisomy "
                    "signature, built from saline mice only."),
        ClaimSplit("Q3 · Memantine may shift trisomic mice toward normal, but by an uncertain "
                   "amount",
                   s / "rescue.png",
                   [f"Closes {v['RescueFrac']}% of the trisomy gap (p = {v['RescueP']})",
                    f"{v['Persist']}% of the gap remains against drug-treated controls",
                    f"Drug effect ({v['MagDrugTs']} SD) rivals trisomy ({v['MagTri']} SD)"],
                   "Arrows: effect of memantine within each genotype.",
                   notes="The interval is very wide, and normal mice also shift, so neither "
                         "the size nor the specificity of the effect is settled."),
        Claim("Q4 · Does trisomy rewire how proteins work together?",
              WF / "wf_q4_network.png", READ_WF,
              notes="We compare how protein pairs co-vary in each genotype, after removing "
                    "group means, against shuffled genotype labels."),
        ClaimSplit("Q4 · We find no evidence that trisomy rewires protein networks",
                   s / "network.png",
                   [f"Global test p = {v['NetP']}", f"0 of {v['NetPairs']} protein pairs differ",
                    "Only large changes were detectable"],
                   "Bars: differences expected by chance (genotype labels shuffled).",
                   notes="Absence of evidence, not evidence of absence."),

        # ------------------------------------------------------------ 7 HOW SURE
        Section(7, "HOW SURE", "How much weight can each answer bear?",
                notes="Confidence differs a lot between the four answers."),
        Claim("One strong answer and three tentative ones",
              EXPLAIN / "evidence.png",
              "Strength combines effect size, interval width, replication and design.",
              notes="Q1 replicates in separate mice. Q2 is borderline. Q3 points toward "
                    "normal, but its interval includes no shift. Q4 is a null with low power."),
        Points("Read every p-value as exploratory",
               [("The analyses were not pre-registered",
                 "questions and methods were refined while examining the data"),
                ("All tests are two-sided", "the direction was not fixed in advance"),
                ("No correction across the four questions",
                 "each result is a hypothesis to replicate, not a settled fact")],
               notes="Say this plainly: the evidence is suggestive."),

        # ------------------------------------------------------------ 8 NEXT
        Section(8, "NEXT", "What does it mean, and what would settle it?",
                notes="Bring it together."),
        TableSlide("What a follow-up study needs",
                   ["Need", "Why"],
                   [[f"About {v['NAttEighty']} mice per group",
                     "to confirm the weaker learning response with 80% power"],
                    [f"{v['NThreeTwo']}–{v['NThreeOne']} mice per group",
                     "to test interaction effects on single proteins"],
                    ["Female mice and an independent cohort", "so findings can generalise"],
                    ["A memory score for every mouse", "to link proteins to behaviour"]],
                   [2.4, 4.0],
                   notes="This study had 7–10 mice per group."),
        Points("Three things to remember",
               [("Count mice, not rows", "otherwise accuracy is badly inflated"),
                ("Extra gene copies show far less extra protein than expected",
                 "compensation or measurement compression; these data cannot tell which"),
                ("Memantine’s protein effects are suggestive, not settled",
                 "a replication with more mice is the next step")],
               notes="Close here. Everything is reproducible with `make reproduce`."),
    ]
    return Deck("Ts65Dn cortical protein signatures", SECTIONS, slides, document_author())


def main() -> None:
    deck = build()
    render(deck, ROOT / "presentation")
    print(f"wrote {len(deck.slides)} slides to presentation/presentation.(pptx|pdf)")


if __name__ == "__main__":
    main()
