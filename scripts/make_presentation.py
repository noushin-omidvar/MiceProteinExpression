#!/usr/bin/env python3
"""Build the slide deck as an editable PowerPoint file and a matching PDF.

One slide specification (``build_slides``) drives both renderers, and all numbers
come from ``results/`` through the same macro table that feeds the paper, so slides,
paper and figures cannot disagree.

Layouts
-------
``title``   opening slide.
``split``   heading, takeaway, bullets on the left, figure on the right.
``figure``  heading, one-sentence takeaway, full-width figure, short footnote.
``text``    heading and bullets (summaries, methods).

Typography and colour follow ``docs/figure_style.md``. The PowerPoint file names
the house fonts; machines without them installed will substitute (install the
fonts in ``assets/fonts`` for exact rendering). The PDF embeds the fonts.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR
from pptx.util import Inches, Pt
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_paper import document_author, values

from mice_protein import plotting as P

OUT = ROOT / "presentation"
TITLE = "Ts65Dn cortical protein signatures"

MECH = ROOT / "results/mechanisms/figures"
PRED = ROOT / "results/figures"
W, H = 13.333, 7.5  # inches, 16:9
MARGIN = 0.6


@dataclass
class Slide:
    layout: str
    kicker: str
    title: str
    takeaway: str = ""
    bullets: list[str] = field(default_factory=list)
    figure: Path | None = None
    footnote: str = ""
    notes: str = ""


def clean(text: str) -> str:
    return text.replace("{,}", ",").replace("--", "–")


def build_slides() -> list[Slide]:
    v = {k: clean(x) for k, x in values(json.loads(
        (ROOT / "results/mechanisms/summary.json").read_text())).items()}
    slide_fig = MECH / "slides"
    return [
        Slide("title", "RESEARCH SUMMARY",
              "What an extra chromosome does to the learning brain, and what memantine does about it",
              "A mouse-level analysis of cortical proteins in a Down syndrome mouse model",
              notes="Plan: the problem with how this dataset is used, four biological questions, the "
                    "answers, and what they mean. Every number is regenerated from the raw data."),
        Slide("text", "THE PROBLEM", "A biology experiment that became a machine-learning puzzle",
              "The data were collected to understand Down syndrome, but are mostly used to test classifiers.",
              ["Researchers measured 77 brain proteins in 72 mice of a Down syndrome model",
               "Most later work only asks: can a model tell the 8 groups of mice apart?",
               (f"Problem 1 · it overstates accuracy: {v['NRows']} rows come from only {v['NMice']} mice, "
               "so random splits let a model recognise mice it has already seen"),
               ("Problem 2 · it answers the wrong question: accuracy says the groups differ, "
               "not how they differ")],
              notes="State the problem plainly before any results. Both issues are general: repeated "
                    "measurements and benchmarks that drift away from the original scientific question."),
        Slide("text", "WHY IT MATTERS", "The biological questions are still open",
              "Down syndrome is the most common genetic cause of intellectual disability.",
              ["What does an extra chromosome actually change in the brain?",
               "Why do trisomic mice fail to learn, and how does a drug restore learning?",
               "Answering this needs the data analysed as an experiment, with the mouse as the unit"],
              notes="Memantine is already approved for Alzheimer's disease; knowing how it acts in the "
                    "trisomic brain matters for treatment research."),
        Slide("figure", "STUDY AT A GLANCE", "The experiment, the problem, our questions and the answers",
              "The rest of the talk unpacks each column.",
              figure=MECH / "fig0_overview.png",
              footnote="Schematic. Numbers are generated from the results.",
              notes="Use this slide as the map for the talk; return to it at the end."),
        Slide("text", "THE EXPERIMENT", "Eight groups from three yes/no choices",
              "2 genotypes × 2 conditioning types × 2 treatments = 8 groups of 7–10 mice.",
              ["Genotype · control, or Ts65Dn (an extra copy of ~90 genes also on human chromosome 21)",
               ("Conditioning · CS mice explore a box, then get a mild shock, and learn to fear the box; "
               "SC mice are shocked at once and cannot form that link"),
               "Treatment · memantine (blocks NMDA receptors, which are involved in memory) or saline",
               "Measurement · 77 proteins in the cortex, each measured 15 times per mouse"],
              notes="CS versus SC is the key contrast: the same shock, with or without learning."),
        Slide("text", "OUR QUESTIONS", "Four questions the benchmark never asked",
              bullets=["Q1 · Do three copies of a gene give 1.5 times as much protein?",
                       "Q2 · Does trisomy change the brain's protein response to learning?",
                       "Q3 · Does memantine move the trisomic brain back toward normal?",
                       "Q4 · Does trisomy change how proteins work together?"],
              notes="Each question gets one slide: short answer, evidence, caveat."),
        Slide("text", "HOW WE ANSWERED THEM", "Four rules that keep the analysis honest",
              bullets=["Count mice, not rows: average each mouse's 15 measurements",
                       ("Combine proteins into one signature score per mouse, which is far more "
                       "reliable than 77 separate tests with 7–10 mice per group"),
                       "Never score a mouse on a signature it helped build (leave-one-out)",
                       (f"Measure uncertainty by re-drawing mice ({v['NBoot']}×) and shuffling labels "
                       f"({v['NPerm']}×)")],
              notes="Every shortcut was checked against an exact calculation or a simulation with a "
                    "known answer before it was used on the real data."),
        Slide("figure", "FIRST, THE PITFALL", "Counting rows as mice makes accuracy look near-perfect",
              f"Random row splits: {v['RowBA']}. Holding out whole mice: {v['GroupedBA']}. Chance: 0.125.",
              figure=PRED / "validation_comparison.png",
              footnote="Each dot is one test. With random splits, the same mouse sits in training and "
                       "test data, so the model recognises the mouse rather than its group.",
              notes="Everything that follows uses one value per mouse."),
        Slide("split", "QUESTION 1 OF 4 · GENE DOSAGE", "Extra genes, but little extra protein",
              f"Short answer: about {v['DosageFrac']}% of the increase that three gene copies predict.",
              [(f"Triplicated genes: +{v['DosageTri']} log₂ (95% CI {v['DosageTriCI']}), "
               f"a {v['DosageFold']}-fold change"),
               (f"Chromosome-21 genes that are not triplicated: {v['DosageDis']}, so the effect "
               "follows copy number"),
               f"Replicates in separate mice (r = {v['DosageRep']})",
               ("Caveat: assay compression, nuclear-only measurement or compensation could explain "
               "the shortfall")],
              slide_fig / "dosage.png",
              notes="The dashed line is strict dosage, 1.5 times the protein."),
        Slide("split", "QUESTION 2 OF 4 · LEARNING", "Trisomy weakens the learning response; memantine restores it",
              f"Short answer: {v['LearnRelTsSal']}× the normal response, back to {v['LearnRelTsMem']}× with memantine.",
              [("Learning signature: protein difference between mice that learned (CS) and could "
               "not (SC), in normal mice on saline"),
               f"Trisomic, saline: {v['LearnRelTsSal']}× (95% CI {v['LearnRelTsSalCI']}; p = {v['LearnAttP']})",
               f"Trisomic, memantine: {v['LearnRelTsMem']}× (p = {v['LearnResP']})",
               "Same proteins respond in every group; only the size changes",
               "Caveat: group-level link to behaviour; needs replication"],
              slide_fig / "learning.png",
              notes="This mirrors behaviour: Ts65Dn mice fail this memory test and memantine rescues "
                    "them. It is the most biologically specific result."),
        Slide("split", "QUESTION 3 OF 4 · RESCUE", "Memantine partly offsets trisomy, but not selectively",
              f"Short answer: it closes {v['RescueFrac']}% of the gap, but moves normal mice the same way.",
              [f"Trisomic mice move {v['RescueFrac']}% of the way back toward normal (p = {v['RescueP']})",
               f"Normal mice shift too, so {v['Persist']}% of the gap remains against drug-treated controls",
               (f"Memantine's effect ({v['MagDrugTs']} SD) is nearly as large as trisomy's "
               f"({v['MagTri']} SD) and mostly points elsewhere"),
               "We avoided the 'pick proteins, then test rescue in the same mice' trap"],
              slide_fig / "rescue.png",
              notes="Arrows show the drug effect within each genotype. Both point the same way."),
        Slide("split", "QUESTION 4 OF 4 · NETWORKS", "No evidence that trisomy rewires protein networks",
              f"Short answer: no detectable change (p = {v['NetP']}).",
              ["Compared how proteins rise and fall together in each genotype",
               (f"Observed difference is within the range expected by chance; "
               f"0 of {v['NetPairs']} protein pairs differ"),
               ("Caveat: with ~34 mice per genotype only large changes would be visible, so this "
               "is inconclusive, not a proof of no effect")],
              slide_fig / "network.png",
              notes="Reporting null results tells the next study where effort is unlikely to pay off."),
        Slide("text", "CAN WE TRUST THE BEST SCORES?", "Testing whether large models memorised this dataset",
              "We hid the dataset's identity (new names, shuffled order, rescaled values) and re-scored.",
              ["Rule fixed in advance: a drop of more than 0.05 at every attempt would count as memorisation",
               f"Logistic regression (control): {v['MemLogisticOrig']} → {v['MemLogisticObf']}",
               f"TabPFN v2: {v['MemTabpfnOrig']} → {v['MemTabpfnObf']} · {v['MemTabpfnVerdict']}",
               f"Kumo Tabular Small: {v['MemKumoOrig']} → {v['MemKumoObf']} · {v['MemKumoVerdict']}",
               "A pass rules out reliance on names, order and values, not memorised relationships"],
              notes="Public benchmarks may be in pretraining data. This check is cheap and should be "
                    "standard practice."),
        Slide("split", "WHAT COMES NEXT", "What a follow-up study needs",
              "Signature scores are far more efficient than protein-by-protein tests.",
              [f"About {v['NAttEighty']} mice per group to confirm the learning result (80% power)",
               f"{v['NThreeTwo']}–{v['NThreeOne']} mice per group for single-protein interactions",
               "Add female mice, an independent cohort and a learning score for every mouse",
               "Release replicate and dilution labels with the data"],
              MECH / "fig7_power.png",
              notes="This study had 7–10 mice per group."),
        Slide("text", "ANSWERS", "The four answers",
              bullets=[f"Q1 · Extra gene copies add only ~{v['DosageFrac']}% of the expected protein",
                       "Q2 · The trisomic brain responds to learning too weakly; memantine restores it",
                       "Q3 · Memantine partly offsets trisomy but is not a targeted repair",
                       "Q4 · No detectable change in how proteins work together",
                       f"Method · Count mice, not rows: {v['NMice']} mice, not {v['NRows']} samples"],
              notes="Return to the overview slide if questions arise. These are results from one cohort "
                    "of male mice."),
        Slide("text", "REPRODUCIBILITY", "Everything is regenerated by one command",
              bullets=["make install   ·   make reproduce",
                       "Raw workbook → tests → analyses → figures → paper, report and slides",
                       "Every estimator checked against brute force or simulation",
                       ("Data: UCI Machine Learning Repository, archive.ics.uci.edu/dataset/342 "
                       "(Higuera, Gardiner & Cios 2015; CC BY 4.0)"),
                       ("Paper: paper/main.pdf · Plain-language report: report/study_report.pdf · "
                       "Decisions: docs/analysis_decisions.md")],
              notes="All code, settings and results are in the repository."),
    ]


# ---------------------------------------------------------------------------
# PowerPoint renderer
# ---------------------------------------------------------------------------

def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color.lstrip("#"))


def add_text(slide, x, y, w, h, text, size, font=P.SANS, color=P.INK, bold=False,
             anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        run = para.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.name = font
        run.font.bold = bold
        run.font.color.rgb = rgb(color)
        para.space_after = Pt(size * 0.6)
    return box


def fit_image(path: Path, w: float, h: float) -> tuple[float, float]:
    with Image.open(path) as im:
        ratio = im.width / im.height
    return (w, w / ratio) if w / ratio <= h else (h * ratio, h)


def regions(s: Slide):
    """Shared geometry so both renderers place elements identically (inches, top-left)."""
    top = 1.75 if s.takeaway else 1.45
    if s.layout == "split":
        return {"text": (MARGIN, top + 0.15, 4.6, H - top - 0.8),
                "figure": (5.6, top + 0.15, W - 5.6 - MARGIN, H - top - 0.85)}
    if s.layout == "figure":
        return {"figure": (MARGIN, top + 0.1, W - 2 * MARGIN, H - top - 1.1)}
    return {"text": (MARGIN, top + 0.2, W - 2 * MARGIN, H - top - 1.0)}


def render_pptx(slides: list[Slide], path: Path) -> None:
    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(W), Inches(H)
    blank = deck.slide_layouts[6]
    for n, s in enumerate(slides, 1):
        slide = deck.slides.add_slide(blank)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = rgb(P.BACKGROUND)
        if s.layout == "title":
            add_text(slide, MARGIN + 0.2, 1.4, 10, 0.4, s.kicker, 12, P.MONO, P.COBALT)
            add_text(slide, MARGIN + 0.2, 2.0, 11.5, 2.4, s.title, 40, P.SERIF, P.INK)
            add_text(slide, MARGIN + 0.2, 4.55, 11, 0.8, s.takeaway, 20, P.SANS, P.SUBTLE_TEXT)
            bar = slide.shapes.add_shape(1, Inches(MARGIN + 0.2), Inches(5.6), Inches(1.2), Inches(0.06))
            bar.fill.solid()
            bar.fill.fore_color.rgb = rgb(P.VERMILION)
            bar.line.fill.background()
        else:
            add_text(slide, MARGIN, 0.42, 10, 0.3, s.kicker, 10.5, P.MONO, P.COBALT)
            add_text(slide, MARGIN, 0.72, W - 2 * MARGIN, 0.7, s.title, 26, P.SERIF, P.INK)
            if s.takeaway:
                add_text(slide, MARGIN, 1.3, W - 2 * MARGIN, 0.4, s.takeaway, 15, P.SANS, P.SUBTLE_TEXT)
            geo = regions(s)
            if "text" in geo:
                x, y, w, h = geo["text"]
                add_text(slide, x, y, w, h, [f"•  {b}" for b in s.bullets],
                         22 if s.layout == "text" else 16)
            if s.figure is not None:
                x, y, w, h = geo["figure"]
                fw, fh = fit_image(s.figure, w, h)
                slide.shapes.add_picture(str(s.figure), Inches(x + (w - fw) / 2), Inches(y),
                                         Inches(fw), Inches(fh))
            if s.footnote:
                add_text(slide, MARGIN, H - 0.95, W - 2 * MARGIN - 0.8, 0.5, s.footnote, 11.5,
                         P.SANS, P.SUBTLE_TEXT)
            add_text(slide, W - MARGIN - 0.6, H - 0.45, 0.6, 0.3, str(n), 10, P.MONO,
                     P.SUBTLE_TEXT)
        slide.notes_slide.notes_text_frame.text = s.notes
    deck.core_properties.title = TITLE
    deck.core_properties.author = document_author()
    deck.core_properties.last_modified_by = document_author()
    deck.core_properties.comments = ""
    deck.save(path)


# ---------------------------------------------------------------------------
# PDF renderer (same geometry, embedded fonts)
# ---------------------------------------------------------------------------

FONTS = {"sans": "IBMPlexSans-Regular", "serif": "Fraunces-SemiBold", "mono": "IBMPlexMono-Regular"}


def register_pdf_fonts() -> None:
    P.register_fonts()
    pdfmetrics.registerFont(TTFont("sans", str(P.FONT_CACHE / "IBMPlexSans-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("mono", str(P.FONT_CACHE / "IBMPlexMono-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("serif", str(P.FONT_DIR / "Fraunces-SemiBold.ttf")))


def wrap(text: str, font: str, size: float, width_pt: float) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if pdfmetrics.stringWidth(trial, font, size) <= width_pt:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line] if line else lines


def draw_text(c, x, y, w, text, font, size, color, leading=1.25, after=0.0):
    """Draw wrapped text with its top at (x, y) in inches; return the new top in inches."""
    c.setFillColor(HexColor(color))
    c.setFont(font, size)
    cursor = H * 72 - y * 72 - size
    for line in wrap(text, font, size, w * 72):
        c.drawString(x * 72, cursor, line)
        cursor -= size * leading
    return (H * 72 - cursor - size) / 72 + after


def render_pdf(slides: list[Slide], path: Path) -> None:
    register_pdf_fonts()
    c = canvas.Canvas(str(path), pagesize=(W * 72, H * 72))
    c.setTitle(TITLE)
    c.setAuthor(document_author())
    c.setCreator(document_author())
    c.setSubject("Mouse-level analysis of cortical proteins in the Ts65Dn model")
    for n, s in enumerate(slides, 1):
        c.setFillColor(HexColor(P.BACKGROUND))
        c.rect(0, 0, W * 72, H * 72, stroke=0, fill=1)
        if s.layout == "title":
            draw_text(c, MARGIN + 0.2, 1.4, 10, s.kicker, "mono", 12, P.COBALT)
            y = draw_text(c, MARGIN + 0.2, 2.0, 11.5, s.title, "serif", 40, P.INK, leading=1.15)
            draw_text(c, MARGIN + 0.2, max(y + 0.3, 4.55), 11, s.takeaway, "sans", 20, P.SUBTLE_TEXT)
            c.setFillColor(HexColor(P.VERMILION))
            c.rect((MARGIN + 0.2) * 72, (H - 5.66) * 72, 1.2 * 72, 0.06 * 72, stroke=0, fill=1)
        else:
            draw_text(c, MARGIN, 0.42, 10, s.kicker, "mono", 10.5, P.COBALT)
            draw_text(c, MARGIN, 0.72, W - 2 * MARGIN, s.title, "serif", 26, P.INK)
            if s.takeaway:
                draw_text(c, MARGIN, 1.3, W - 2 * MARGIN, s.takeaway, "sans", 15, P.SUBTLE_TEXT)
            geo = regions(s)
            if "text" in geo:
                x, y, w, _ = geo["text"]
                size = 22 if s.layout == "text" else 16
                for bullet in s.bullets:
                    y = draw_text(c, x, y, w, f"•  {bullet}", "sans", size, P.INK, after=size * 0.6 / 72)
            if s.figure is not None:
                x, y, w, h = geo["figure"]
                fw, fh = fit_image(s.figure, w, h)
                c.drawImage(str(s.figure), (x + (w - fw) / 2) * 72, (H - y - fh) * 72, fw * 72, fh * 72,
                            preserveAspectRatio=True, mask="auto")
            if s.footnote:
                draw_text(c, MARGIN, H - 0.95, W - 2 * MARGIN - 0.8, s.footnote, "sans", 11.5,
                          P.SUBTLE_TEXT)
            draw_text(c, W - MARGIN - 0.3, H - 0.45, 0.6, str(n), "mono", 10, P.SUBTLE_TEXT)
        c.showPage()
    c.save()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    slides = build_slides()
    missing = [s.figure for s in slides if s.figure and not s.figure.exists()]
    if missing:
        raise SystemExit(f"Missing figures (run `make figures` first): {missing}")
    render_pptx(slides, OUT / "presentation.pptx")
    render_pdf(slides, OUT / "presentation.pdf")
    print(f"wrote {len(slides)} slides to {OUT.relative_to(ROOT)}/presentation.(pptx|pdf)")


if __name__ == "__main__":
    main()
