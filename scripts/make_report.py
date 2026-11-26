#!/usr/bin/env python3
"""Render the plain-language report (Markdown and PDF) from its template.

``report/study_report.template.md`` contains ``$Name`` placeholders that are filled
with the same values used by the paper (``make_paper.values``). The filled Markdown
is written to ``report/study_report.md`` (readable on GitHub) and converted to PDF
with pandoc + tectonic using the house typography (``report/style.tex``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_paper import document_author, values

from mice_protein import plotting

REPORT = ROOT / "report"



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pdf", action="store_true", help="render Markdown only")
    args = parser.parse_args()
    summary = json.loads((ROOT / "results/mechanisms/summary.json").read_text())
    v = {k: x.replace("{,}", ",").replace("--", "–") for k, x in values(summary).items()}
    text = Template((REPORT / "study_report.template.md").read_text()).substitute(v)
    (REPORT / "study_report.md").write_text(text)
    print("wrote report/study_report.md")
    if args.no_pdf:
        return
    if shutil.which("pandoc") is None or shutil.which("tectonic") is None:
        print("pandoc or tectonic not found: skipped PDF")
        return
    plotting.register_fonts()
    subprocess.run(
        ["pandoc", "study_report.md", "-o", "study_report.pdf", "--pdf-engine=tectonic",
         "--include-in-header=style.tex", "--resource-path=.", "-V", "geometry:margin=22mm",
         "-V", "papersize=a4", "-V", "colorlinks=true", "-V", "linkcolor=blue",
         "-M", f"author={document_author()}"],
        cwd=REPORT, check=True,
    )
    print("wrote report/study_report.pdf")


if __name__ == "__main__":
    main()
