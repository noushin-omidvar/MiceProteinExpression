"""Smoke tests: the workflow-diagram and slide engines render without error."""

import matplotlib
import pytest

matplotlib.use("Agg")

from mice_protein import plotting as P
from mice_protein.workflow import GLYPHS, Stage, draw_workflow


def test_every_schematic_glyph_draws(tmp_path):
    P.apply_style()
    schematic = [g for g in GLYPHS if not g.startswith("result")]
    stages = [Stage("STEP", g, g, "text", r"$x$", "where") for g in schematic[:5]]
    stages.append(Stage("RESULT", "r", "result_bars", "done",
                        args={"values": [0.2, 0.5], "labels": ["a", "b"],
                              "colors": [P.COBALT, P.VERMILION], "ref": 0.4}, result=True))
    draw_workflow(stages, tmp_path, "wf", question="Q?")
    assert (tmp_path / "wf.svg").exists() and (tmp_path / "wf.pdf").exists()


def test_slide_engine_renders_pptx_and_pdf(tmp_path):
    pytest.importorskip("pptx")
    pytest.importorskip("reportlab")
    from mice_protein import slides as SL

    deck = SL.Deck("Test", ["WHY", "WHAT"], [
        SL.Title("A claim", "A subtitle", "Author"),
        SL.Section(1, "WHY", "Why?"),
        SL.Points("Points", [("Lead", "explanation")]),
        SL.BigNumber("Number", "0.99", "label", "context"),
        SL.TableSlide("Table", ["A", "B"], [["1", "yes"]], [1, 1], highlight_column=1),
    ], "Author")
    SL.render(deck, tmp_path, "deck")
    assert (tmp_path / "deck.pptx").stat().st_size > 0
    assert (tmp_path / "deck.pdf").stat().st_size > 0
