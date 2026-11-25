#!/usr/bin/env python3
"""Build the manuscript: write numeric macros and tables from results, then compile.

Every number quoted in ``paper/main.tex`` is a macro defined in
``paper/generated/values.tex``, written here from ``results/``. Re-running the
analysis and this script therefore updates the paper without hand-editing.

Usage::

    python scripts/make_paper.py            # write generated files and compile PDF
    python scripts/make_paper.py --no-pdf   # only write generated files
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from mice_protein import plotting
from mice_protein.annotations import display

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
GEN = PAPER / "generated"
RES = ROOT / "results"


def document_author() -> str:
    """Author for document metadata: the repository's git user name (empty if unset)."""
    result = subprocess.run(["git", "config", "user.name"], cwd=ROOT, capture_output=True,
                            text=True, check=False)
    return result.stdout.strip()


def f2(x: float) -> str:
    return f"{x:.2f}"


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def pval(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def ci(pair) -> str:
    """Interval as 'a--b', or 'a to b' when a bound is negative (avoids '-0.1--0.5')."""
    if min(pair) < 0:
        return f"{pair[0]:.2f} to {pair[1]:.2f}"
    return f"{pair[0]:.2f}--{pair[1]:.2f}"


def latex_escape(text: str) -> str:
    return (text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("×", r"$\times$"))


MEMORIZATION_THRESHOLD = 0.05  # pre-specified in docs/analysis_decisions.md


def memorization(model: str) -> dict[str, str]:
    """Original vs obfuscated balanced accuracy and the pre-specified verdict."""
    path = RES / f"metrics/memorization_{model}.csv"
    if not path.exists():
        return {"Orig": "--", "Obf": "--", "Drop": "--", "Verdict": "not run"}
    table = pd.read_csv(path)
    wide = table.pivot(index="repeat", columns="version", values="balanced_accuracy")
    obf = wide.drop(columns="original")
    drops = obf.rsub(wide["original"], axis=0)
    flagged = drops.to_numpy().mean() > MEMORIZATION_THRESHOLD and (drops.mean() > 0).all()
    return {"Orig": f"{wide['original'].mean():.3f}", "Obf": f"{obf.to_numpy().mean():.3f}",
            "Drop": f"{drops.to_numpy().mean():+.3f}",
            "Verdict": "possible memorisation" if flagged else "no evidence of memorisation"}


def values(summary: dict) -> dict[str, str]:
    """Map macro names (letters only) to formatted values."""
    d, q = summary["data"], summary["qc"]
    dos, L, R = summary["dosage"], summary["learning"], summary["rescue"]
    net, pw = summary["network"], summary["power"]
    pc = summary["protein_contrasts"]

    model_summary = pd.read_csv(RES / "metrics/model_summary.csv").set_index(["design", "model"])
    bench = pd.read_csv(RES / "metrics/mouse_model_benchmark.csv")
    bench = bench[bench.target == "class"].groupby("model").balanced_accuracy.mean()
    kumo = pd.read_csv(RES / "metrics/kumo_mouse_benchmark.csv").balanced_accuracy.mean()
    tabpfn = pd.read_csv(RES / "metrics/tabpfn_mouse_benchmark.csv").balanced_accuracy.mean()
    obf = pd.DataFrame(summary["obfuscation"])

    per = dos["per_protein"]
    v = {
        "NMice": str(d["mice"]), "NRows": f"{d['rows']:,}".replace(",", "{,}"),
        "NProteins": str(d["proteins"]), "NUnique": str(d["unique_proteins"]),
        "NPanel": str(d["panel_proteins"]), "NImputed": str(d["imputed_mouse_means"]),
        "ICCMedian": f2(q["icc_median"]), "ICCRange": ci(q["icc_iqr"]),
        "BlockMedian": f2(q["block_share_median"]), "BlockPct": pct(q["block_share_median"]),
        "BlockRange": ci(q["block_share_iqr"]),
        "MissingTested": str(q["missing_tested"]),
        "MissingInformative": str(len(q["missing_informative"])),
        "RowBA": f"{model_summary.loc[('random_observation', 'logistic'), 'balanced_accuracy_mean']:.3f}",
        "GroupedBA": f"{model_summary.loc[('grouped_animal', 'logistic'), 'balanced_accuracy_mean']:.3f}",
        "GroupedRFBA": f"{model_summary.loc[('grouped_animal', 'random_forest'), 'balanced_accuracy_mean']:.3f}",
        "MouseMeanBA": f"{model_summary.loc[('mouse_mean', 'logistic'), 'balanced_accuracy_mean']:.3f}",
        "ElasticBA": f"{bench['elastic_net_mean']:.3f}", "LogisticBA": f"{bench['logistic_mean']:.3f}",
        "KumoBA": f"{kumo:.3f}", "TabpfnBA": f"{tabpfn:.3f}",
        "NCond": str(pc["Conditioning"]), "NGeno": str(pc["Genotype"]), "NTreat": str(pc["Treatment"]),
        "NGxC": str(pc["Genotype×Conditioning"]), "NGxT": str(pc["Genotype×Treatment"]),
        "NGxCxT": str(pc["Genotype×Conditioning×Treatment"]),
        "NTriSaline": str(pc["Trisomy (saline)"]), "NLearnCS": str(pc["Learning: control saline"]),
        "NLearnTS": str(pc["Learning: Ts65Dn saline"]), "NLearnTM": str(pc["Learning: Ts65Dn memantine"]),
        "DosageExpected": f2(dos["expected_log2fc"]),
        "DosageTri": f2(dos["sets"]["triplicated"]["estimate"]),
        "DosageTriCI": ci([dos["sets"]["triplicated"]["lower"], dos["sets"]["triplicated"]["upper"]]),
        "DosageDis": f2(dos["sets"]["hsa21_disomic"]["estimate"]),
        "DosageDisCI": ci([dos["sets"]["hsa21_disomic"]["lower"], dos["sets"]["hsa21_disomic"]["upper"]]),
        "DosageBg": f2(dos["sets"]["background"]["estimate"]),
        "DosageBgCI": ci([dos["sets"]["background"]["lower"], dos["sets"]["background"]["upper"]]),
        "NBackground": str(dos["sets"]["background"]["n"]),
        "DosageFrac": pct(dos["set_fraction_of_expected"]),
        "DosageFold": f"{2 ** dos['sets']['triplicated']['estimate']:.2f}",
        "DosageRep": f2(dos["replication_pearson"]),
        "DosageRepTri": f2(dos["replication_pearson_triplicated"]),
        "DosageTopFifth": str(sum(r >= 0.8 for r in dos["triplicated_rank_percentile"].values())),
        "DosageQSig": str(dos["triplicated_q05_saline"]),
        "APPfc": f2(per["APP_N"]["estimate_saline"]), "DYRKfc": f2(per["DYRK1A_N"]["estimate_saline"]),
        "ITSNfc": f2(per["ITSN1_N"]["estimate_saline"]), "SODfc": f2(per["SOD1_N"]["estimate_saline"]),
        "DSCRfc": f2(per["DSCR1_N"]["estimate_saline"]), "TIAMfc": f2(per["TIAM1_N"]["estimate_saline"]),
        "LearnRespCtrlSal": f"{L['response']['control saline']:.1f}",
        "LearnRelCtrlMem": f2(L["relative"]["control memantine"]),
        "LearnRelCtrlMemCI": ci(L["relative_ci"]["control memantine"]),
        "LearnRelTsSal": f2(L["relative"]["Ts65Dn saline"]),
        "LearnRelTsSalCI": ci(L["relative_ci"]["Ts65Dn saline"]),
        "LearnRelTsMem": f2(L["relative"]["Ts65Dn memantine"]),
        "LearnRelTsMemCI": ci(L["relative_ci"]["Ts65Dn memantine"]),
        "LearnAttP": pval(L["attenuation_p"]), "LearnResP": pval(L["restoration_p"]),
        "LearnAtt": f"{L['attenuation']:.1f}", "LearnSD": f"{L['within_cell_sd']:.2f}",
        "CosCtrlMem": f2(L["cosine"]["control memantine"]), "CosCtrlMemCI": ci(L["cosine_ci"]["control memantine"]),
        "CosTsSal": f2(L["cosine"]["Ts65Dn saline"]), "CosTsSalCI": ci(L["cosine_ci"]["Ts65Dn saline"]),
        "CosTsMem": f2(L["cosine"]["Ts65Dn memantine"]), "CosTsMemCI": ci(L["cosine_ci"]["Ts65Dn memantine"]),
        "RescueFrac": pct(R["rescue_fraction"]), "RescueFracCI": ci(R["rescue_fraction_ci"]),
        "RescueP": pval(R["rescue_shift_p"]),
        "Persist": pct(R["persistence"]), "PersistCI": ci(R["persistence_ci"]),
        "CtrlShift": f2(R["control_shift"]), "CtrlShiftCI": ci(R["control_shift_ci"]),
        "RescueCS": f2(R["rescue_by_conditioning"]["CS"]),
        "RescueCSCI": ci(R["rescue_by_conditioning_ci"]["CS"]),
        "RescueSC": f2(R["rescue_by_conditioning"]["SC"]),
        "RescueSCCI": ci(R["rescue_by_conditioning_ci"]["SC"]),
        "MagTri": f"{R['magnitudes']['Trisomy (saline)']:.1f}",
        "MagDrugTs": f"{R['magnitudes']['Memantine in Ts65Dn']:.1f}",
        "MagDrugCtrl": f"{R['magnitudes']['Memantine in control']:.1f}",
        "AxisShare": pct(R["along_axis_share"]), "AxisShareCI": ci(R["along_axis_share_ci"]),
        "SlopeNaive": f2(R["persistence_slope_naive"]), "Reliability": f2(R["reliability_saline"]),
        "SlopeCorr": f2(R["persistence_slope_corrected"]),
        "NAffected": str(sum(v for k, v in R["per_protein_counts"].items() if k != "Not affected (saline)")),
        "NPersist": str(R["per_protein_counts"].get("Persists under memantine", 0)),
        "NetStat": f"{net['statistic']:.3f}", "NetNull": f"{net['null_mean']:.3f}",
        "NetP": f2(net["p_global"]), "NetPairs": f"{net['pairs_total']:,}".replace(",", "{,}"),
        "NetPairsSig": str(net["pairs_q05"]),
        "PowAttNine": pct(pw["Learning attenuation (observed)"]["power_at_9"]),
        "NAttEighty": str(pw["Learning attenuation (observed)"]["n_for_80"]),
        "PowThreeOneNine": pct(pw["Protein 3-way interaction, 1 SD"]["power_at_9"]),
        "NThreeOne": str(pw["Protein 3-way interaction, 1 SD"]["n_for_80"]),
        "NThreeTwo": str(pw["Protein 3-way interaction, 2 SD"]["n_for_80"]),
        "ObfLog": f"{obf[(obf.model == 'logistic') & ~obf.obfuscated]['mean'].iloc[0]:.3f}",
        "ObfLogObf": f"{obf[(obf.model == 'logistic') & obf.obfuscated]['mean'].iloc[0]:.3f}",
        "ObfRF": f"{obf[(obf.model == 'random_forest') & ~obf.obfuscated]['mean'].iloc[0]:.3f}",
        "ObfRFObfMin": f"{obf[(obf.model == 'random_forest') & obf.obfuscated]['min'].iloc[0]:.3f}",
        "ObfRFObfMax": f"{obf[(obf.model == 'random_forest') & obf.obfuscated]['max'].iloc[0]:.3f}",
        "NBoot": f"{summary['provenance']['config']['n_bootstrap']:,}".replace(",", "{,}"),
        "NPerm": f"{summary['provenance']['config']['n_permutations']:,}".replace(",", "{,}"),
        "NNetPerm": f"{summary['provenance']['config']['n_network_permutations']:,}".replace(",", "{,}"),
        "Seed": str(summary["provenance"]["config"]["seed"]),
        "PyVersion": summary["provenance"]["python"],
        "SklearnVersion": summary["provenance"]["scikit_learn"],
    }
    for model, prefix in (("logistic", "MemLogistic"), ("tabpfn", "MemTabpfn"), ("kumo", "MemKumo")):
        v.update({prefix + k: x for k, x in memorization(model).items()})
    verdicts = {v["MemTabpfnVerdict"], v["MemKumoVerdict"]}
    if verdicts == {"no evidence of memorisation"}:
        v["MemAnswer"] = "neither model lost accuracy beyond the pre-specified threshold"
    elif "not run" in verdicts:
        v["MemAnswer"] = "the test has not been run for every model yet"
    else:
        v["MemAnswer"] = "at least one model lost accuracy beyond the pre-specified threshold"
    loads = list(L["top_loadings"].items())[:6]
    v["LearnTopProteins"] = ", ".join(display(p) for p, _ in loads)
    return v


def pathway_table(summary: dict) -> str:
    table = pd.read_csv(RES / "mechanisms/tables/pathway_contrasts.csv")
    keep = ["Genotype", "Conditioning", "Treatment", "Genotype×Conditioning",
            "Genotype×Treatment", "Conditioning×Treatment", "Genotype×Conditioning×Treatment"]
    table = table[table.contrast.isin(keep) & (table.q < 0.05)].sort_values(["contrast", "q"])
    rows = [
        f"{latex_escape(r.feature)} & {latex_escape(r.contrast)} & {r.estimate:+.2f} & "
        f"[{r.lower:+.2f}, {r.upper:+.2f}] & {r.q:.2g} \\\\"
        for r in table.itertuples()
    ]
    coherence = summary["pathways"]["coherence"]
    note = "; ".join(f"{latex_escape(k)} {v:+.2f}" for k, v in coherence.items())
    return "\n".join([
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"Pathway & Contrast & Estimate (SD) & 95\% CI & $q$ \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}",
        r"\par\smallskip{\footnotesize Mean within-cell correlation of member ratios: " + note + ".}",
    ])


def validation_table() -> str:
    ms = pd.read_csv(RES / "metrics/model_summary.csv").set_index(["design", "model"])
    bench = pd.read_csv(RES / "metrics/mouse_model_benchmark.csv")
    bench = bench[bench.target == "class"].groupby("model").balanced_accuracy.agg(["mean", "std"])
    kumo = pd.read_csv(RES / "metrics/kumo_mouse_benchmark.csv").balanced_accuracy
    tabpfn = pd.read_csv(RES / "metrics/tabpfn_mouse_benchmark.csv").balanced_accuracy
    rows = [
        ("Random rows (leaky)", "Logistic", ms.loc[("random_observation", "logistic")]),
        ("Held-out mice, 15 rows", "Logistic", ms.loc[("grouped_animal", "logistic")]),
        ("Held-out mice, 15 rows", "Random forest", ms.loc[("grouped_animal", "random_forest")]),
        ("Held-out mice, means", "Logistic", ms.loc[("mouse_mean", "logistic")]),
    ]
    lines = [r"\begin{tabular}{llrr}", r"\toprule",
             r"Validation & Model & Balanced accuracy & SD \\", r"\midrule"]
    for design, model, row in rows:
        lines.append(f"{design} & {model} & {row.balanced_accuracy_mean:.3f} & "
                     f"{row.balanced_accuracy_sd:.3f} \\\\")
    lines.append(r"\midrule")
    for name, label in (("elastic_net_mean", "Elastic net"), ("rbf_svm_mean", "RBF SVM"),
                        ("gradient_boosting_mean", "Gradient boosting")):
        lines.append(f"Held-out mice, means & {label} & {bench.loc[name, 'mean']:.3f} & "
                     f"{bench.loc[name, 'std']:.3f} \\\\")
    lines.append(f"Held-out mice, means & TabPFN v2$^\\dagger$ & {tabpfn.mean():.3f} & {tabpfn.std():.3f} \\\\")
    lines.append(f"Held-out mice, means & Kumo Tabular Small$^\\dagger$ & {kumo.mean():.3f} & {kumo.std():.3f} \\\\")
    lines += [r"Chance & -- & 0.125 & -- \\", r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    summary = json.loads((RES / "mechanisms/summary.json").read_text())
    GEN.mkdir(parents=True, exist_ok=True)
    macros = values(summary)
    lines = ["% Generated by scripts/make_paper.py from results/ -- do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    lines.append(f"\\newcommand{{\\DocAuthor}}{{{latex_escape(document_author())}}}")
    (GEN / "values.tex").write_text("\n".join(lines) + "\n")
    (GEN / "pathway_table.tex").write_text(pathway_table(summary) + "\n")
    (GEN / "validation_table.tex").write_text(validation_table() + "\n")
    print(f"wrote {len(macros)} macros and 2 tables to {GEN.relative_to(ROOT)}")
    if args.no_pdf:
        return
    plotting.register_fonts()  # ensures decompressed TrueType fonts exist for XeTeX
    if shutil.which("tectonic") is None:
        raise SystemExit("tectonic not found: install it (https://tectonic-typesetting.github.io)"
                         " or use --no-pdf")
    subprocess.run(["tectonic", "main.tex"], cwd=PAPER, check=True)
    print(f"compiled {PAPER / 'main.pdf'}")


if __name__ == "__main__":
    main()
