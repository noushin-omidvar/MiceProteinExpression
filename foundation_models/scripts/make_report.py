#!/usr/bin/env python3
"""Render the foundation-model study's plain-language report (Markdown and PDF).

``foundation_models/report/study_report.template.md`` holds ``$Name`` placeholders filled from
the study's results. The PDF uses the repository's report typography (``report/style.tex``),
with font paths adjusted for this directory.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from string import Template

import pandas as pd

from mice_protein import plotting

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from make_paper import document_author

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "foundation_models"
RES = STUDY / "results"
REPORT = STUDY / "report"


def verdict(flag: bool) -> str:
    return "supported" if flag else "not supported"


def values() -> dict[str, str]:
    S = json.loads((RES / "summary.json").read_text())
    E = json.loads((RES / "extension_summary.json").read_text())
    perf = S["performance"]
    tuned = pd.read_csv(RES / "tuned_baselines/performance.csv", index_col="model")["mean"]
    kt = {r["foundation"]: r for r in E["paired_vs_tuned_logistic"]}["kumo"]
    h1k = {t["baseline"]: t for t in S["H1"]["kumo"]["tests"]}["logistic"]
    h1t = {t["baseline"]: t for t in S["H1"]["tabpfn"]["tests"]}["logistic"]
    curve = pd.read_csv(RES / "matched_holdout_review/scores.csv")
    m = curve.groupby(["n_train", "model"]).balanced_accuracy.mean()
    h4 = S["H4a"]["supported"] and S["H4b"]["supported"] and S["H4c"]["supported"]
    return {
        "KumoBA": f"{perf['kumo']['balanced_accuracy_mean']:.2f}",
        "TabpfnBA": f"{perf['tabpfn']['balanced_accuracy_mean']:.2f}",
        "LogBA": f"{perf['logistic']['balanced_accuracy_mean']:.2f}",
        "TunedLogBA": f"{tuned['logistic']:.2f}",
        "TunedDiff": f"{kt['mean_difference']:+.2f}",
        "TunedCI": f"{kt['lower']:+.2f} to {kt['upper']:+.2f}",
        "TunedP": f"{kt['p']:.2f}",
        "KumoLogDiff": f"{h1k['mean_difference']:+.2f}", "KumoLogP": f"{h1k['p_holm']:.3f}",
        "TabLogDiff": f"{h1t['mean_difference']:+.2f}", "TabLogP": f"{h1t['p_holm']:.2f}",
        "KumoLL": f"{perf['kumo']['log_loss_mean']:.2f}",
        "TabpfnLL": f"{perf['tabpfn']['log_loss_mean']:.2f}",
        "LogLL": f"{perf['logistic']['log_loss_mean']:.2f}",
        "KumoGapSmall": f"{S['H3']['kumo']['gap_small']:+.2f}",
        "KumoGapLarge": f"{S['H3']['kumo']['gap_large']:+.2f}",
        "M16Kumo": f"{m.loc[16, 'kumo']:.2f}", "M16Log": f"{m.loc[16, 'logistic']:.2f}",
        "M48Kumo": f"{m.loc[48, 'kumo']:.2f}", "M48Log": f"{m.loc[48, 'logistic']:.2f}",
        "HoneTabpfn": verdict(S["H1"]["tabpfn"]["supported"]),
        "HoneKumo": verdict(S["H1"]["kumo"]["supported"]),
        "HtwoTabpfn": verdict(S["H2"]["tabpfn"]["supported"]),
        "HtwoKumo": verdict(S["H2"]["kumo"]["supported"]),
        "HthreeTabpfn": verdict(S["H3"]["tabpfn"]["supported"]),
        "HthreeKumo": verdict(S["H3"]["kumo"]["supported"]),
        "HfourAll": "all three controls as expected" if h4 else "at least one control failed",
    }


def main() -> None:
    REPORT.mkdir(exist_ok=True)
    text = Template((REPORT / "study_report.template.md").read_text()).substitute(values())
    (REPORT / "study_report.md").write_text(text)
    style = (ROOT / "report/style.tex").read_text().replace("Path=../", "Path=../../")
    style = style.replace("PLAIN-LANGUAGE REPORT", "PLAIN-LANGUAGE REPORT")
    style = style.replace("Ts65Dn mouse cortex · 77 proteins · 72 mice",
                          "TabPFN v2 · Kumo Tabular · 72 mice")
    (REPORT / "style.tex").write_text(style)
    print("wrote foundation_models/report/study_report.md")
    if shutil.which("pandoc") is None or shutil.which("tectonic") is None:
        print("pandoc or tectonic not found: skipped PDF")
        return
    plotting.register_fonts()
    subprocess.run(
        ["pandoc", "study_report.md", "-o", "study_report.pdf", "--pdf-engine=tectonic",
         "--include-in-header=style.tex", "--resource-path=.", "-V", "geometry:margin=22mm",
         "-V", "papersize=a4", "-V", "colorlinks=true", "-M", f"author={document_author()}"],
        cwd=REPORT, check=True,
    )
    print("wrote foundation_models/report/study_report.pdf")


if __name__ == "__main__":
    main()
