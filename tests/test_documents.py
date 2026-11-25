"""Checks that generated documents can be built from the committed results."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/mechanisms/summary.json"


@pytest.mark.skipif(not SUMMARY.exists(), reason="run `make mechanisms` first")
def test_paper_macros_are_valid_latex_and_complete():
    from make_paper import values

    macros = values(json.loads(SUMMARY.read_text()))
    assert all(re.fullmatch(r"[A-Za-z]+", name) for name in macros)
    assert all(str(value).strip() for value in macros.values())
    used = set(re.findall(r"\\([A-Za-z]+)", (ROOT / "paper/main.tex").read_text()))
    report = (ROOT / "report/study_report.template.md").read_text()
    placeholders = set(re.findall(r"(?<!\$)\$([A-Za-z]+)", report))
    assert placeholders <= set(macros), placeholders - set(macros)
    assert {"LearnRelTsSal", "RescueFrac", "DosageFrac"} <= used
