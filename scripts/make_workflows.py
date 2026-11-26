#!/usr/bin/env python3
"""Workflow diagrams for the biology study (one per question, Q1-Q4).

Each diagram reads left to right: plain-language steps, each with its equation and symbol
definitions, ending in a result panel that plots the real numbers from
results/mechanisms/summary.json.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd

from mice_protein import plotting as P
from mice_protein.workflow import Stage, draw_workflow

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/mechanisms"
OUT = RES / "figures/workflows"
S = json.loads((RES / "summary.json").read_text())


def w(text: str, width: int = 27) -> str:
    return textwrap.fill(text, width)


def workflows():
    d, L, R, N = S["dosage"], S["learning"], S["rescue"], S["network"]
    null = pd.read_csv(RES / "tables/network_null.csv").null_statistic.to_numpy()
    rel = L["relative"]
    return {
        "q1_dosage": ("Q1 · Do three gene copies give 1.5 times as much protein?", [
            Stage("AVERAGE", "One value per mouse", "rows_to_mean",
                  w("Average each mouse’s 15 measurements of every protein (log scale)."),
                  r"$y_{mp}=\mathrm{mean}_r\,\log_2 x_{mpr}$",
                  "m = mouse, p = protein, r = repeat"),
            Stage("LABEL", "Gene copy number", "copy_number",
                  w("Mark proteins whose genes are triplicated in Ts65Dn; use non-triplicated "
                    "chromosome-21 genes as a control."), r"$c_p\in\{2,3\}$",
                  "c = gene copies in Ts65Dn"),
            Stage("COMPARE", "Ts65Dn vs control", "two_groups",
                  w("For each protein, take the difference of group means, separately in "
                    "saline and memantine mice."),
                  r"$\Delta_p=\bar y_{\mathrm{Ts}}-\bar y_{\mathrm{Ctrl}}$",
                  "Δ = log₂ fold change"),
            Stage("RESULT", "Observed vs expected", "result_bars",
                  w(f"Triplicated proteins rise only {100 * d['set_fraction_of_expected']:.0f}% "
                    "as much as 1.5× predicts; control genes do not change."),
                  args={"values": [d["sets"]["triplicated"]["estimate"],
                                   d["sets"]["hsa21_disomic"]["estimate"]],
                        "labels": ["triplicated", "not\ntriplicated"],
                        "colors": [P.VERMILION, P.SUBTLE_TEXT],
                        "ref": d["expected_log2fc"], "ref_label": "1.5× expected",
                        "fmt": "{:+.2f}", "ymax": 0.75},
                  result=True),
        ]),
        "q2_learning": ("Q2 · Does trisomy change the brain’s protein response to learning?", [
            Stage("BUILD", "Learning signature", "signature",
                  w("In normal mice on saline, find the protein direction that separates "
                    "mice that learned (CS) from mice that could not (SC)."),
                  r"$\mathbf{a}=\bar{\mathbf{z}}_{\mathrm{CS}}-\bar{\mathbf{z}}_{\mathrm{SC}}$",
                  "z = a mouse’s 70 standardised proteins"),
            Stage("SCORE", "Every mouse, fairly", "leave_one_out",
                  w("Place each mouse on the signature, rebuilding it without that mouse so "
                    "no mouse scores itself."),
                  r"$s_i=\mathbf{z}_i\cdot\hat{\mathbf{a}}^{(-i)}$",
                  "s = score; (−i) = built without mouse i"),
            Stage("MEASURE", "Response per group", "four_arms",
                  w("A group’s learning response is its CS mice’s average score minus its SC "
                    "mice’s, relative to normal mice on saline."),
                  r"$R_g=\bar s_{\mathrm{CS}}-\bar s_{\mathrm{SC}}$",
                  "g = genotype × treatment group"),
            Stage("TEST", "Could it be chance?", "shuffle",
                  w("Shuffle genotype or treatment labels within the design, rebuild "
                    "everything, and count how often chance gives an effect this large."),
                  r"$p=\frac{1+\#(T^{*}\geq T)}{1+B}$", "T* = effect after shuffling; B = shuffles",
                  args={"n": f"{S['provenance']['config']['n_permutations']:,}"}),
            Stage("RESULT", "Learning response", "result_bars",
                  w(f"Trisomic response {rel['Ts65Dn saline']:.2f}× normal "
                    f"(p = {L['attenuation_p']:.3f}); near normal with memantine "
                    f"(p = {L['restoration_p']:.3f})."),
                  args={"values": [rel["control saline"], rel["control memantine"],
                                   rel["Ts65Dn saline"], rel["Ts65Dn memantine"]],
                        "labels": ["Ctrl\nsal", "Ctrl\nmem", "Ts\nsal", "Ts\nmem"],
                        "colors": [P.COBALT, P.COBALT, P.VERMILION, P.VERMILION],
                        "ref": 1.0, "ref_label": "normal", "ymax": 1.45},
                  result=True),
        ]),
        "q3_rescue": ("Q3 · Does memantine move the trisomic brain back toward normal?", [
            Stage("BUILD", "Trisomy signature", "signature",
                  w("In saline mice only, find the protein direction that separates Ts65Dn "
                    "from control mice."),
                  r"$\mathbf{t}=\bar{\mathbf{z}}_{\mathrm{Ts,sal}}-\bar{\mathbf{z}}_{\mathrm{Ctrl,sal}}$",
                  "sal = saline-treated",
                  args={"from_label": "control", "to_label": "Ts65Dn"}),
            Stage("SCORE", "Every mouse, fairly", "leave_one_out",
                  w("Place all mice on it; drug-treated mice were not used to build it, and "
                    "saline mice are left out in turn."),
                  r"$s_i=\mathbf{z}_i\cdot\hat{\mathbf{t}}^{(-i)}$", "s = how trisomic a mouse looks"),
            Stage("MEASURE", "Gap closed by the drug", "four_arms",
                  w("Compare how far memantine moves Ts65Dn mice with the size of the trisomy "
                    "gap, and check whether it moves controls too."),
                  r"$F=\frac{\bar s_{\mathrm{Ts,sal}}-\bar s_{\mathrm{Ts,mem}}}"
                  r"{\bar s_{\mathrm{Ts,sal}}-\bar s_{\mathrm{Ctrl,sal}}}$",
                  "F = fraction of the gap closed"),
            Stage("RESULT", "Partial, not specific", "result_bars",
                  w(f"Memantine closes ~{100 * R['rescue_fraction']:.0f}% of the gap "
                    f"(p = {R['rescue_shift_p']:.3f}; wide interval) and shifts controls too; "
                    f"{100 * R['persistence']:.0f}% remains against drug-treated controls."),
                  args={"values": [100 * R["rescue_fraction"], 100 * R["persistence"]],
                        "labels": ["gap closed", "gap remaining\nvs drug controls"],
                        "colors": [P.VERMILION, P.SUBTLE_TEXT], "fmt": "{:.0f}%",
                        "ref": 100, "ref_label": "full gap", "ymax": 125},
                  result=True),
        ]),
        "q4_network": ("Q4 · Does trisomy change how proteins co-vary?", [
            Stage("REMOVE", "Design effects", "residuals",
                  w("Subtract each experimental group’s mean, so shared responses to the design "
                    "cannot create correlations."),
                  r"$e_{mp}=z_{mp}-\bar z_{g(m),p}$", "g(m) = the group of mouse m"),
            Stage("CORRELATE", "Pairs per genotype", "network_pairs",
                  w("Correlate every pair of the 70 proteins, separately in control and in "
                    "Ts65Dn mice."), r"$r^{\mathrm{Ctrl}}_{pq},\ r^{\mathrm{Ts}}_{pq}$",
                  "2,415 protein pairs"),
            Stage("SUMMARISE", "Overall change", "two_groups",
                  w("Average how much the correlations differ between genotypes."),
                  r"$T=\mathrm{RMS}_{pq}\,(\mathrm{atanh}\,r^{\mathrm{Ts}}-\mathrm{atanh}\,r^{\mathrm{Ctrl}})$",
                  "atanh stabilises correlations",
                  args={"left": "Ctrl", "right": "Ts"}),
            Stage("TEST", "Could it be chance?", "shuffle",
                  w("Shuffle genotype labels within the design and recompute T each time."),
                  r"$T^{*}$ from shuffled labels", "",
                  args={"n": f"{S['provenance']['config']['n_network_permutations']:,}"}),
            Stage("RESULT", "No detectable rewiring", "result_null",
                  w(f"The observed change is within the range expected by chance "
                    f"(p = {N['p_global']:.2f}); only large changes were detectable."),
                  args={"observed": N["statistic"], "null": null, "p": N["p_global"]},
                  result=True),
        ]),
    }


def main() -> None:
    P.apply_style()
    for name, (question, stages) in workflows().items():
        draw_workflow(stages, OUT, f"wf_{name}", question=question)
        draw_workflow(stages, OUT / "slides", f"wf_{name}")  # no heading: the slide states it
        print("drew", name)


if __name__ == "__main__":
    main()
