#!/usr/bin/env python3
"""Build the foundation-model study manuscript: generated values and tables, then compile.

Every number in foundation_models/paper/main.tex is a macro written here from
foundation_models/results/summary.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from make_paper import document_author, latex_escape

from mice_protein import plotting
from mice_protein.benchmark import models as M

STUDY = ROOT / "foundation_models"
RES = STUDY / "results"
PAPER = STUDY / "paper"
GEN = PAPER / "generated"
SHORT = {"dummy": "Dummy", "logistic": "Logistic", "elastic_net": "ElasticNet", "svm": "SVM",
         "random_forest": "RandomForest", "gradient_boosting": "GradBoost", "tabpfn": "Tabpfn",
         "kumo": "Kumo"}


def f2(x) -> str:
    return f"{x:.2f}"


def f3(x) -> str:
    return f"{x:.3f}"


def pv(p) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def yes(flag: bool) -> str:
    return "supported" if flag else "not supported"


def values(S: dict) -> dict[str, str]:
    v: dict[str, str] = {}
    perf = S["performance"]
    for m, short in SHORT.items():
        p = perf[m]
        v[f"BA{short}"] = f3(p["balanced_accuracy_mean"])
        v[f"BASD{short}"] = f3(p["balanced_accuracy_std"])
        v[f"LL{short}"] = f2(p["log_loss_mean"])
        v[f"Brier{short}"] = f2(p["brier_mean"])
        v[f"ECE{short}"] = f2(p["ece_mean"])
        v[f"Sec{short}"] = f"{p['seconds_per_fold']:.2f}"
    baselines = [m for m in M.BASELINES if m != "dummy"]
    best = max(baselines, key=lambda m: perf[m]["balanced_accuracy_mean"])
    best_ll = min(baselines, key=lambda m: perf[m]["log_loss_mean"])
    v["BestBaseline"] = latex_escape(S["labels"][best])
    v["BABest"] = f3(perf[best]["balanced_accuracy_mean"])
    v["BestBaselineLL"] = latex_escape(S["labels"][best_ll])
    v["LLBest"] = f2(perf[best_ll]["log_loss_mean"])
    for h in ("H1", "H2"):
        for tfm in M.FOUNDATION:
            block = S[h][tfm]
            key = f"{h.replace('1', 'one').replace('2', 'two')}{SHORT[tfm]}"
            v[key] = yes(block["supported"])
            v[key + "Beaten"] = str(block["n_baselines_beaten"])
            tests = {t["baseline"]: t for t in block["tests"]}
            t = tests[best if h == "H1" else best_ll]
            v[key + "Diff"] = f"{t['mean_difference']:+.3f}"
            v[key + "CI"] = f"{t['lower']:+.3f} to {t['upper']:+.3f}"
            v[key + "P"] = pv(t["p_holm"])
    for tfm in M.FOUNDATION:
        g = S["H3"][tfm]
        k = f"Hthree{SHORT[tfm]}"
        v[k] = yes(g["supported"])
        v[k + "Small"] = f"{g['gap_small']:+.3f}"
        v[k + "Large"] = f"{g['gap_large']:+.3f}"
        v[k + "Diff"] = f"{g['difference']:+.3f}"
        v[k + "CI"] = f"{g['lower']:+.3f} to {g['upper']:+.3f}"
    lc = S["learning_curve"]
    for m in lc:
        v[f"LCtwo{SHORT[m]}"] = f2(lc[m]["2"])
        v[f"LCsix{SHORT[m]}"] = f2(lc[m]["6"])
    v["HfourA"] = yes(S["H4a"]["supported"])
    pm = S["H4a"]["by_model"]
    v["PermMin"] = f3(min(r["mean"] for r in pm.values()))
    v["PermMax"] = f3(max(r["mean"] for r in pm.values()))
    v["HfourB"] = yes(S["H4b"]["supported"])
    lk = S["H4b"]["by_model"]
    for m, r in lk.items():
        v[f"LeakRows{SHORT[m]}"] = f3(r["random_rows"])
        v[f"LeakMice{SHORT[m]}"] = f3(r["mouse_held_out"])
    v["HfourC"] = yes(S["H4c"]["supported"])
    mem = S["H4c"]["by_model"]
    for m in M.FOUNDATION:
        v[f"Mem{SHORT[m]}Orig"] = f3(mem[m]["original"])
        v[f"Mem{SHORT[m]}Obf"] = f3(mem[m]["obfuscated"])
        v[f"Mem{SHORT[m]}Drop"] = f"{mem[m]['mean_drop']:+.3f}"
    for tag, name in (("genotype", "Geno"), ("treatment", "Treat"), ("behavior", "Cond")):
        for m in ("logistic", "tabpfn", "kumo"):
            v[f"Factor{name}{SHORT[m]}"] = f2(S["factor_targets"][tag][m])
    prov = S["provenance"]
    v["SklearnVersion"] = prov["scikit_learn"]
    v["TabpfnVersion"] = str(prov.get("tabpfn", "2.2.1"))
    v["TorchVersion"] = str(prov.get("torch", ""))
    v["DocAuthor"] = latex_escape(document_author())
    est = pd.read_csv(RES / "estimator_sensitivity/summary.csv", index_col=0)
    v["EstTabpfnBA"] = f3(est.loc["tabpfn_default", "balanced_accuracy"])
    v["EstKumoBA"] = f3(est.loc["kumo_default", "balanced_accuracy"])
    return v


def performance_table(S: dict) -> str:
    perf = S["performance"]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Model & BA & (SD) & Log loss & Brier & ECE & s/fold \\", r"\midrule"]
    for m in M.SPECS:
        p = perf[m]
        name = latex_escape(S["labels"][m])
        if m in M.FOUNDATION:
            name = r"\textbf{" + name + "}"
        lines.append(f"{name} & {p['balanced_accuracy_mean']:.3f} & {p['balanced_accuracy_std']:.3f}"
                     f" & {p['log_loss_mean']:.2f} & {p['brier_mean']:.2f} & {p['ece_mean']:.2f}"
                     f" & {p['seconds_per_fold']:.2f} \\\\")
        if m == "gradient_boosting":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def tests_table(S: dict) -> str:
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Comparison & Metric & Difference & 95\% CI & $p_{\mathrm{Holm}}$ & Better? \\",
             r"\midrule"]
    for h, metric in (("H1", "Balanced accuracy"), ("H2", "Log loss")):
        for tfm in M.FOUNDATION:
            for t in S[h][tfm]["tests"]:
                lines.append(
                    f"{latex_escape(S['labels'][tfm])} vs {latex_escape(S['labels'][t['baseline']])}"
                    f" & {metric} & {t['mean_difference']:+.3f} & "
                    f"[{t['lower']:+.3f}, {t['upper']:+.3f}] & {pv(t['p_holm'])} & "
                    f"{'yes' if t['significant_better'] else 'no'} \\\\")
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def hypothesis_table(S: dict) -> str:
    rows = [
        ("H1", "TFMs more accurate than every baseline",
         f"TabPFN: {yes(S['H1']['tabpfn']['supported'])}; Kumo: {yes(S['H1']['kumo']['supported'])}"),
        ("H2", "TFMs have lower log loss than every baseline",
         f"TabPFN: {yes(S['H2']['tabpfn']['supported'])}; Kumo: {yes(S['H2']['kumo']['supported'])}"),
        ("H3", "TFM advantage larger at 2 than at 6 mice per group",
         f"TabPFN: {yes(S['H3']['tabpfn']['supported'])}; Kumo: {yes(S['H3']['kumo']['supported'])}"),
        ("H4a", "All models at chance with shuffled labels", yes(S["H4a"]["supported"])),
        ("H4b", "All models inflated by row-level leakage", yes(S["H4b"]["supported"])),
        ("H4c", "No reliance on dataset identity", yes(S["H4c"]["supported"])),
    ]
    lines = [r"\begin{tabularx}{\linewidth}{lXl}", r"\toprule",
             r"Hypothesis & Statement (fixed in advance) & Verdict \\", r"\midrule"]
    lines += [f"{a} & {b} & {c} \\\\" for a, b, c in rows]
    lines += [r"\bottomrule", r"\end{tabularx}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    S = json.loads((RES / "summary.json").read_text())
    GEN.mkdir(parents=True, exist_ok=True)
    v = values(S)
    lines = ["% Generated by foundation_models/scripts/make_paper.py -- do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{x}}}" for k, x in sorted(v.items())]
    (GEN / "values.tex").write_text("\n".join(lines) + "\n")
    (GEN / "performance_table.tex").write_text(performance_table(S) + "\n")
    (GEN / "tests_table.tex").write_text(tests_table(S) + "\n")
    (GEN / "hypothesis_table.tex").write_text(hypothesis_table(S) + "\n")
    print(f"wrote {len(v)} macros and 3 tables")
    if args.no_pdf:
        return
    plotting.register_fonts()
    if shutil.which("tectonic") is None:
        raise SystemExit("tectonic not found")
    subprocess.run(["tectonic", "main.tex"], cwd=PAPER, check=True)
    print(f"compiled {PAPER / 'main.pdf'}")


if __name__ == "__main__":
    main()
