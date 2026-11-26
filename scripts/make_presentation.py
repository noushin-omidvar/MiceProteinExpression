#!/usr/bin/env python3
"""Biology study deck, written for an audience new to the topic.

Story: WHY -> BACKGROUND -> DATA -> QUESTIONS -> METHOD -> RESULTS -> SO WHAT.
Built with the shared slide engine (``mice_protein.slides``): every content slide states its
claim as a full sentence and shows one piece of evidence; the spoken explanation is in the
speaker notes. Numbers come from the same generated values as the paper.
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
WF = MECH / "workflows" / "slides"
EXTERNAL = ROOT / "assets/external"
HIGUERA = ("Reproduced from Higuera, Gardiner & Cios, PLoS ONE 10(6): e0129126 (2015), Fig 1A "
           "(cropped) · CC BY 4.0")


def clean(text: str) -> str:
    return text.replace("{,}", ",").replace("--", "–")


def build() -> Deck:
    v = {k: clean(x) for k, x in values(json.loads(
        (ROOT / "results/mechanisms/summary.json").read_text())).items()}
    s = MECH / "slides"
    slides = [
        Title("An extra chromosome weakens the brain’s protein response to learning, and "
              "memantine restores it",
              "Cortical protein signatures in the Ts65Dn mouse model of Down syndrome",
              document_author(),
              notes="Today I will show what an extra chromosome does to the proteins of the "
                    "learning brain, and what the drug memantine does about it. No background is "
                    "assumed: we start with the biology, then the data, then the findings."),

        # ------------------------------------------------------------------ WHY
        Section(1, "WHY", "An extra chromosome impairs learning. What changes in the brain?",
                notes="Start with the motivation."),
        Points("Down syndrome affects learning, and a drug can restore learning in mice",
               [("Down syndrome is caused by an extra copy of chromosome 21",
                 "It is the most common genetic cause of intellectual disability"),
                ("A mouse model carries extra copies of the same genes",
                 "These mice fail a simple memory test"),
                ("The drug memantine restores their learning",
                 "Shown in earlier experiments"),
                ("What changes in the brain’s proteins is still unclear",
                 "That is the gap this study addresses")],
               notes="The behavioural findings are established; the molecular picture is not."),

        # ------------------------------------------------------------------ BACKGROUND
        Section(2, "BACKGROUND", "Four ideas you need to follow the talk",
                notes="For anyone new to the topic: the model, the memory test, proteins, the drug."),
        Points("Idea 1 · The Ts65Dn mouse is a model of Down syndrome",
               [("People with Down syndrome have three copies of chromosome 21",
                 "instead of two, so many genes are over-active"),
                ("Ts65Dn mice carry a third copy of ~90 of the same genes",
                 "on a piece of mouse chromosome 16"),
                ("They show learning deficits, as people with Down syndrome do",
                 "which makes them useful for testing treatments")],
               notes="Mouse models let us measure the brain directly, which is impossible in people."),
        Claim("Idea 2 · Mice learn to fear a place only if the shock comes after they explore it",
              PRIMER / "experiment_timeline.png",
              "CS = context then shock (learning). SC = shock then context (same shock, no learning).",
              notes="This is context fear conditioning. Both groups get the same cage and the same "
                    "mild shock; only the order differs. So CS minus SC isolates learning the link "
                    "between place and shock. Brains were collected one hour later."),
        Points("Idea 3 · Proteins do the brain’s work, and switch on and off",
               [("Genes are recipes; proteins are the machines they make",
                 "Learning changes which proteins are present and active"),
                ("Many proteins are switched on by phosphorylation",
                 "a small chemical tag; names starting with ‘p’ (e.g. pERK) are tagged forms"),
                ("This study measured 77 proteins and tagged forms",
                 "in the cortex, the brain’s outer layer")],
               notes="When I say a protein ‘rises’ I mean more of it was measured in that group."),
        Points("Idea 4 · Memantine tunes a receptor involved in memory",
               [("Memantine blocks NMDA receptors", "a receptor central to learning and memory"),
                ("It is an approved drug", "used for Alzheimer’s disease"),
                ("In Ts65Dn mice it restores learning",
                 "shown in earlier behavioural studies; here, 5 mg/kg 15 min before training")],
               notes="We ask what memantine does to brain proteins, not whether it improves behaviour."),
        TableSlide("Key terms used in this talk",
                   ["Term", "Meaning"],
                   [["Ts65Dn", "Mouse with extra copies of Down-syndrome genes"],
                    ["CS / SC", "Learns the place–shock link / same shock, no learning"],
                    ["Saline", "Inactive injection, the comparison for the drug"],
                    ["Signature", "A combination of proteins that captures one effect"],
                    ["p-value", "How often chance alone would give a result this large"]],
                   [1.2, 4.0],
                   notes="Keep this slide up briefly; the terms recur throughout."),

        # ------------------------------------------------------------------ DATA
        Section(3, "DATA", "What exactly was measured, and what does the table look like?",
                notes="Now the data themselves."),
        Claim("Eight groups of 7–10 mice; only three of them learn",
              EXTERNAL / "higuera2015_fig1A.png",
              HIGUERA,
              notes="Three yes/no choices make eight groups: genotype (c = control, t = Ts65Dn), "
                    "training (CS or SC) and injection (s = saline, m = memantine). Learning outcome "
                    "on the right: normal in control CS, failed in Ts65Dn CS on saline, rescued "
                    "with memantine, and no learning in any SC group."),
        Claim("Each protein was measured 15 times in every mouse",
              PRIMER / "measurement.png",
              "Reverse-phase protein arrays: each sample printed as 3 replicates × 5 dilutions.",
              notes="This is why the table has 15 rows per mouse. They are repeated readings of the "
                    "same brain sample, not 15 different mice."),
        Claim("The table has 1,080 rows, but every 15 rows are the same mouse",
              PRIMER / "data_anatomy.png",
              "Real rows from the public table. Columns: 77 proteins, then the experimental labels.",
              notes="Point at the red bracket: rows 309_1 to 309_15 are all mouse 309."),
        BigNumber("So the data hold 72 independent samples, not 1,080",
                  f"{v['NRows']} rows · {v['NMice']} mice",
                  "72 male mice, 3–4 months old, cortex collected 1 hour after training",
                  "Treating rows as separate mice is the most common mistake made with this dataset.",
                  notes="Everything that follows uses one value per mouse."),
        Points("What these data can and cannot tell us",
               [("Can: how groups differ in protein levels",
                 "genotype, learning and drug, one hour after training"),
                ("Cannot: which individual mouse learned",
                 "no memory score was recorded for each mouse"),
                ("Cannot: cause and effect for single proteins",
                 "the data show associations"),
                ("Cannot: females, other ages or other brain regions",
                 "all mice were male; one region, one time point")],
               notes="Stating limits up front builds trust in the results."),

        # ------------------------------------------------------------------ QUESTIONS
        Section(4, "QUESTIONS", "Four questions about the trisomic brain",
                notes="Here is what we set out to answer."),
        Claim("One experiment, four questions",
              MECH / "fig0_overview.png",
              "Schematic. Numbers are generated from the results.",
              notes="Walk left to right: experiment, problem, questions, and a preview of the answers."),

        # ------------------------------------------------------------------ METHOD
        Section(5, "METHOD", "How do you get reliable answers from 72 mice?",
                notes="Small samples demand careful methods. Four rules."),
        Points("Four rules keep a small study honest",
               [("Count mice, not rows", "each mouse’s 15 readings become one value per protein"),
                ("Combine proteins into signatures",
                 "one score per mouse is far more reliable than 77 separate tests"),
                ("Never let a mouse score itself",
                 "each mouse is scored on a signature rebuilt without it"),
                ("Let the design define chance",
                 f"re-draw mice {v['NBoot']}× and shuffle labels {v['NPerm']}× within the design")],
               notes="Every shortcut was checked against an exact calculation or a simulation."),
        Claim("Counting rows as mice makes accuracy look near-perfect",
              PRED / "validation_comparison.png",
              f"Random row splits: {v['RowBA']}. Whole mice held out: {v['GroupedBA']}. Chance: 0.125.",
              notes="When the same mouse is in training and test data, a model recognises the "
                    "mouse rather than its group. This is why rule one matters."),

        # ------------------------------------------------------------------ RESULTS
        Section(6, "RESULTS", "What did we find?",
                notes="For each question: first how we answered it, then the answer."),
        Claim("Q1 · We compare each protein’s change with what an extra gene copy predicts",
              WF / "wf_q1_dosage.png",
              "Read left to right: each step, its equation, and the result.",
              notes="The last panel shows the real result."),
        ClaimSplit(f"Q1 · Three gene copies give only ~{v['DosageFrac']}% of the expected extra protein",
                   s / "dosage.png",
                   [f"Triplicated genes: +{v['DosageTri']} log₂ (95% CI {v['DosageTriCI']})",
                    f"Chromosome-21 genes that are not triplicated: {v['DosageDis']}",
                    f"Replicates in separate mice (r = {v['DosageRep']})"],
                   "Dashed line: 1.5× protein expected from three gene copies.",
                   notes="Built-in negative control: non-triplicated genes do not change."),
        Claim("Q2 · We score every mouse on a learning signature it did not help build",
              WF / "wf_q2_learning.png",
              "Read left to right: each step, its equation, and the result.",
              notes="The signature is defined in normal mice on saline."),
        ClaimSplit("Q2 · Trisomy weakens the learning response by a third; memantine restores it",
                   s / "learning.png",
                   [f"Trisomic, saline: {v['LearnRelTsSal']}× normal (p = {v['LearnAttP']})",
                    f"Trisomic, memantine: {v['LearnRelTsMem']}× (p = {v['LearnResP']})",
                    "The same proteins respond in every group; only the size changes"],
                   "Learning response = CS − SC protein signature, relative to control saline.",
                   notes="This mirrors behaviour: Ts65Dn mice fail the test; memantine rescues them. "
                         "Caveat: group-level link; needs replication."),
        Claim("Q3 · We measure how far memantine moves trisomic mice along the trisomy signature",
              WF / "wf_q3_rescue.png",
              "Read left to right: each step, its equation, and the result.",
              notes="The signature is built from saline mice only."),
        ClaimSplit("Q3 · Memantine partly offsets trisomy, but moves normal mice the same way",
                   s / "rescue.png",
                   [f"Closes {v['RescueFrac']}% of the trisomy gap (p = {v['RescueP']})",
                    f"{v['Persist']}% of the gap remains against drug-treated controls",
                    f"Drug effect ({v['MagDrugTs']} SD) rivals trisomy ({v['MagTri']} SD)"],
                   "Arrows: effect of memantine within each genotype.",
                   notes="So memantine is not a targeted repair."),
        Claim("Q4 · We compare how protein pairs co-vary in each genotype against chance",
              WF / "wf_q4_network.png",
              "Read left to right: each step, its equation, and the result.",
              notes="Correlations are computed after removing group means."),
        ClaimSplit("Q4 · We find no evidence that trisomy rewires protein networks",
                   s / "network.png",
                   [f"Global test p = {v['NetP']}", f"0 of {v['NetPairs']} protein pairs differ",
                    "Only large changes were detectable"],
                   "Bars: differences expected by chance (genotype labels shuffled).",
                   notes="Absence of evidence, not evidence of absence."),

        # ------------------------------------------------------------------ SO WHAT
        Section(7, "SO WHAT", "What does it mean, and what comes next?",
                notes="Bring it together."),
        TableSlide("Two clear answers, one partial answer, one null",
                   ["Question", "Answer", "Strength of evidence"],
                   [["Q1 · Extra gene copies → more protein?",
                     f"Yes, but only ~{v['DosageFrac']}% of expected", "Strong; replicated"],
                    ["Q2 · Trisomy changes learning response?",
                     "Weaker; memantine restores it", "Moderate; needs replication"],
                    ["Q3 · Memantine repairs trisomy?", "Partly, not selectively", "Moderate"],
                    ["Q4 · Networks rewired?", "No evidence", "Weak; underpowered"]],
                   [3.2, 3.2, 2.2],
                   notes="Be explicit about confidence for each answer."),
        Points("How much weight the p-values can bear",
               [("These analyses were not pre-registered",
                 "questions and methods were refined while examining the data"),
                ("All tests are two-sided",
                 "the direction of each effect was not fixed in advance"),
                ("No correction across questions",
                 "so every result is a hypothesis to replicate, not a settled fact")],
               notes="Say this plainly: the evidence is suggestive, and a pre-registered "
                     "replication is the next step."),
        Points("What a follow-up study needs",
               [(f"About {v['NAttEighty']} mice per group",
                 "to confirm the weaker learning response with 80% power"),
                (f"{v['NThreeTwo']}–{v['NThreeOne']} mice per group",
                 "for interaction effects on single proteins"),
                ("Female mice and an independent cohort", "so the findings can generalise"),
                ("A memory score for every mouse", "to link proteins to behaviour animal by animal")],
               notes="This study had 7–10 mice per group."),
        Points("Take-home messages",
               [("Count mice, not rows", "otherwise accuracy is badly inflated"),
                ("The trisomic brain responds to learning too weakly",
                 "and memantine restores the size of that response"),
                ("Memantine’s protein effects are broad, not targeted",
                 "it partly offsets trisomy rather than repairing it")],
               notes="Close here. Everything is reproducible with `make reproduce`."),
    ]
    return Deck("Ts65Dn cortical protein signatures",
                ["WHY", "BACKGROUND", "DATA", "QUESTIONS", "METHOD", "RESULTS", "SO WHAT"],
                slides, document_author())


def main() -> None:
    deck = build()
    render(deck, ROOT / "presentation")
    print(f"wrote {len(deck.slides)} slides to presentation/presentation.(pptx|pdf)")


if __name__ == "__main__":
    main()
