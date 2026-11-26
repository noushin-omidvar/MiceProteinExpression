"""Slide engine: one deck specification, rendered to editable PowerPoint and to PDF.

Design (assertion-evidence, story-first)
----------------------------------------
* Every content slide has a **claim headline**: a full sentence stating the takeaway.
  The evidence (one figure, one number or one table) sits beneath it.
* The talk is divided into **sections** (e.g. WHY, WHAT, HOW, RESULTS, SO WHAT). A divider
  opens each section and a tracker in the footer shows where the audience is.
* Text is minimal on the slide; the full spoken explanation lives in the **speaker notes**.
* Identity follows ``docs/figure_style.md``: Fraunces headlines, IBM Plex Sans text,
  IBM Plex Mono labels, cobalt for structure, vermilion for the single highlight.

Slide types
-----------
``Title``, ``Section``, ``Claim`` (figure), ``ClaimSplit`` (figure + points),
``BigNumber``, ``Points`` (numbered lead-ins), ``TableSlide``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from . import plotting as P

W, H = 13.333, 7.5
M = 0.7  # outer margin (inches)
HEAD_Y = 0.95  # top of claim headline
BODY_Y = 2.15  # top of evidence area
FOOT_Y = 6.95  # baseline area of footer


# ---------------------------------------------------------------------------
# Slide specifications
# ---------------------------------------------------------------------------

@dataclass
class Title:
    title: str
    subtitle: str
    byline: str = ""
    notes: str = ""


@dataclass
class Section:
    number: int
    name: str  # e.g. "WHY"
    question: str  # one sentence the section answers
    notes: str = ""


@dataclass
class Claim:
    claim: str
    figure: Path
    caption: str = ""
    notes: str = ""


@dataclass
class ClaimSplit:
    claim: str
    figure: Path
    points: list[str] = field(default_factory=list)
    caption: str = ""
    notes: str = ""


@dataclass
class BigNumber:
    claim: str
    number: str
    label: str
    context: str = ""
    notes: str = ""


@dataclass
class Points:
    claim: str
    items: list[tuple[str, str]]  # (lead, explanation)
    numbered: bool = True
    notes: str = ""


@dataclass
class TableSlide:
    claim: str
    header: list[str]
    rows: list[list[str]]
    widths: list[float]  # relative column widths
    highlight_column: int | None = None  # column whose "Yes/No" cells are coloured
    caption: str = ""
    notes: str = ""


@dataclass
class Deck:
    title: str
    sections: list[str]
    slides: list  # sequence of slide specs
    author: str = ""


def _section_of(deck: Deck) -> list[int | None]:
    """Current section index for each slide (None before the first divider)."""
    current, out = None, []
    for s in deck.slides:
        if isinstance(s, Section):
            current = s.number - 1
        out.append(current)
    return out


def _image_fit(path: Path, w: float, h: float) -> tuple[float, float]:
    with Image.open(path) as im:
        ratio = im.width / im.height
    return (w, w / ratio) if w / ratio <= h else (h * ratio, h)


# ---------------------------------------------------------------------------
# PowerPoint renderer
# ---------------------------------------------------------------------------

def _rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color.lstrip("#"))


def _box(slide, x, y, w, h, text, size, font=P.SANS, color=P.INK, bold=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.1):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paragraphs = text if isinstance(text, list) else [text]
    for i, item in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        runs = item if isinstance(item, list) else [(item, {})]
        for run_text, style in runs:
            r = p.add_run()
            r.text = run_text
            r.font.size = Pt(style.get("size", size))
            r.font.name = style.get("font", font)
            r.font.bold = style.get("bold", bold)
            r.font.color.rgb = _rgb(style.get("color", color))
        p.space_after = Pt(size * 0.5)
    return shape


def _rect(slide, x, y, w, h, color):
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = _rgb(color)
    shape.line.fill.background()
    return shape


def _footer_pptx(slide, deck: Deck, n: int, section: int | None) -> None:
    _rect(slide, M, FOOT_Y - 0.12, W - 2 * M, 0.012, P.RULE)
    x = M
    for i, name in enumerate(deck.sections):
        active = section == i
        _box(slide, x, FOOT_Y, 1.4, 0.3, name, 9.5, P.MONO,
             P.COBALT if active else P.SUBTLE_TEXT, bold=active)
        x += 0.32 + 0.085 * len(name)
    _box(slide, W - M - 3.5, FOOT_Y, 3.0, 0.3, deck.title, 9.5, P.MONO, P.SUBTLE_TEXT,
         align=PP_ALIGN.RIGHT)
    _box(slide, W - M - 0.4, FOOT_Y, 0.4, 0.3, str(n), 9.5, P.MONO, P.INK, align=PP_ALIGN.RIGHT)


def _claim_pptx(slide, claim: str) -> None:
    _rect(slide, M, HEAD_Y - 0.28, 0.5, 0.05, P.VERMILION)
    _box(slide, M, HEAD_Y, W - 2 * M, 1.1, claim, 28, P.SERIF, P.INK, spacing=1.0)


def render_pptx(deck: Deck, path: Path) -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    sections = _section_of(deck)
    for n, (s, sec) in enumerate(zip(deck.slides, sections, strict=True), 1):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = _rgb(P.BACKGROUND)
        if isinstance(s, Title):
            _rect(slide, M, 1.6, 0.9, 0.07, P.VERMILION)
            _box(slide, M, 1.9, 11.4, 2.6, s.title, 42, P.SERIF, P.INK, spacing=1.0)
            _box(slide, M, 4.55, 11, 0.9, s.subtitle, 20, P.SANS, P.SUBTLE_TEXT)
            _box(slide, M, 5.7, 11, 0.4, s.byline, 12, P.MONO, P.COBALT)
        elif isinstance(s, Section):
            _box(slide, M, 1.5, 3, 1.6, f"{s.number:02d}", 96, P.SERIF, P.COBALT)
            _box(slide, M, 3.3, 10, 0.6, s.name, 16, P.MONO, P.COBALT, bold=True)
            _box(slide, M, 3.85, 11, 1.6, s.question, 32, P.SERIF, P.INK, spacing=1.0)
        else:
            _claim_pptx(slide, s.claim)
            if isinstance(s, Claim):
                fw, fh = _image_fit(s.figure, W - 2 * M, FOOT_Y - BODY_Y - 0.55)
                slide.shapes.add_picture(str(s.figure), Inches((W - fw) / 2), Inches(BODY_Y),
                                         Inches(fw), Inches(fh))
                if s.caption:
                    _box(slide, M, FOOT_Y - 0.5, W - 2 * M, 0.35, s.caption, 11, P.SANS,
                         P.SUBTLE_TEXT)
            elif isinstance(s, ClaimSplit):
                _box(slide, M, BODY_Y + 0.1, 4.3, 4.3,
                     [[(p, {})] for p in s.points], 17, P.SANS, P.INK, spacing=1.15)
                fw, fh = _image_fit(s.figure, W - M - 5.4, FOOT_Y - BODY_Y - 0.55)
                slide.shapes.add_picture(str(s.figure), Inches(5.4 + (W - M - 5.4 - fw) / 2),
                                         Inches(BODY_Y), Inches(fw), Inches(fh))
                if s.caption:
                    _box(slide, 5.4, FOOT_Y - 0.5, W - M - 5.4, 0.35, s.caption, 11, P.SANS,
                         P.SUBTLE_TEXT)
            elif isinstance(s, BigNumber):
                _box(slide, M, BODY_Y + 0.2, W - 2 * M, 1.8, s.number, 80, P.SERIF, P.COBALT)
                _box(slide, M, BODY_Y + 2.15, W - 2 * M, 0.6, s.label, 22, P.SANS, P.INK)
                _box(slide, M, BODY_Y + 2.9, 10.5, 1.2, s.context, 16, P.SANS, P.SUBTLE_TEXT)
            elif isinstance(s, Points):
                y = BODY_Y + 0.05
                for i, (lead, text) in enumerate(s.items, 1):
                    if s.numbered:
                        _box(slide, M, y, 0.6, 0.5, f"{i}", 22, P.SERIF, P.COBALT)
                    x = M + (0.65 if s.numbered else 0)
                    _box(slide, x, y, W - x - M, 0.45, lead, 19, P.SANS, P.INK, bold=True)
                    _box(slide, x, y + 0.42, W - x - M, 0.5, text, 15, P.SANS, P.SUBTLE_TEXT)
                    y += (FOOT_Y - BODY_Y - 0.3) / max(len(s.items), 4)
            elif isinstance(s, TableSlide):
                _table_pptx(slide, s)
        if not isinstance(s, Title):
            _footer_pptx(slide, deck, n, sec)
        slide.notes_slide.notes_text_frame.text = s.notes
    prs.core_properties.title = deck.title
    prs.core_properties.author = deck.author
    prs.core_properties.last_modified_by = deck.author
    prs.save(path)


def _table_pptx(slide, s: TableSlide) -> None:
    total = sum(s.widths)
    width = W - 2 * M
    rows, cols = len(s.rows) + 1, len(s.header)
    height = min(0.75 * rows, FOOT_Y - BODY_Y - 0.6)
    shape = slide.shapes.add_table(rows, cols, Inches(M), Inches(BODY_Y), Inches(width),
                                   Inches(height))
    table = shape.table
    for j, w in enumerate(s.widths):
        table.columns[j].width = Inches(width * w / total)
    for i in range(rows):
        for j in range(cols):
            cell = table.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = _rgb(P.BACKGROUND if i else "#F3F5F7")
            text = s.header[j] if i == 0 else s.rows[i - 1][j]
            tf = cell.text_frame
            tf.text = ""
            run = tf.paragraphs[0].add_run()
            run.text = text
            run.font.size = Pt(17 if i else 12)
            run.font.name = P.MONO if i == 0 else P.SANS
            color = P.SUBTLE_TEXT if i == 0 else P.INK
            if i and j == s.highlight_column:
                color = P.COBALT if text.lower().startswith(("yes", "supported")) else P.VERMILION
                run.font.bold = True
            run.font.color.rgb = _rgb(color)
    if s.caption:
        _box(slide, M, FOOT_Y - 0.5, W - 2 * M, 0.35, s.caption, 11, P.SANS, P.SUBTLE_TEXT)


# ---------------------------------------------------------------------------
# PDF renderer (same geometry, embedded fonts)
# ---------------------------------------------------------------------------

def _register_pdf_fonts() -> None:
    P.register_fonts()
    for name, path in (("sans", P.FONT_CACHE / "IBMPlexSans-Regular.ttf"),
                       ("sans-bold", P.FONT_CACHE / "IBMPlexSans-SemiBold.ttf"),
                       ("mono", P.FONT_CACHE / "IBMPlexMono-Regular.ttf"),
                       ("mono-bold", P.FONT_CACHE / "IBMPlexMono-Medium.ttf"),
                       ("serif", P.FONT_DIR / "Fraunces-SemiBold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(path)))


_FONT = {P.SANS: "sans", P.MONO: "mono", P.SERIF: "serif"}


def _wrap(text: str, font: str, size: float, width_pt: float) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if pdfmetrics.stringWidth(trial, font, size) <= width_pt or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + ([line] if line else [])


def _text(c, x, y, w, text, size, font=P.SANS, color=P.INK, bold=False, align="left",
          leading=1.18) -> float:
    """Draw wrapped text with its top at y (inches); return the bottom in inches."""
    name = _FONT[font] + ("-bold" if bold and font in (P.SANS, P.MONO) else "")
    c.setFillColor(HexColor(color))
    c.setFont(name, size)
    cursor = H * 72 - y * 72 - size * 0.9
    for line in _wrap(text, name, size, w * 72):
        if align == "right":
            c.drawRightString((x + w) * 72, cursor, line)
        else:
            c.drawString(x * 72, cursor, line)
        cursor -= size * leading
    return (H * 72 - cursor - size * 0.9) / 72 - size * (leading - 1) / 72


def _rect_pdf(c, x, y, w, h, color):
    c.setFillColor(HexColor(color))
    c.rect(x * 72, (H - y - h) * 72, w * 72, h * 72, stroke=0, fill=1)


def _image_pdf(c, path, x, y, w, h):
    c.drawImage(str(path), x * 72, (H - y - h) * 72, w * 72, h * 72, mask="auto")


def _footer_pdf(c, deck: Deck, n: int, section: int | None) -> None:
    _rect_pdf(c, M, FOOT_Y - 0.12, W - 2 * M, 0.012, P.RULE)
    x = M
    for i, name in enumerate(deck.sections):
        active = section == i
        _text(c, x, FOOT_Y, 1.4, name, 9.5, P.MONO, P.COBALT if active else P.SUBTLE_TEXT,
              bold=active)
        x += 0.32 + 0.085 * len(name)
    _text(c, W - M - 3.5, FOOT_Y, 3.0, deck.title, 9.5, P.MONO, P.SUBTLE_TEXT, align="right")
    _text(c, W - M - 0.4, FOOT_Y, 0.4, str(n), 9.5, P.MONO, P.INK, align="right")


def render_pdf(deck: Deck, path: Path) -> None:
    _register_pdf_fonts()
    c = canvas.Canvas(str(path), pagesize=(W * 72, H * 72))
    c.setTitle(deck.title)
    c.setAuthor(deck.author)
    c.setCreator(deck.author)
    sections = _section_of(deck)
    for n, (s, sec) in enumerate(zip(deck.slides, sections, strict=True), 1):
        _rect_pdf(c, 0, 0, W, H, P.BACKGROUND)
        if isinstance(s, Title):
            _rect_pdf(c, M, 1.6, 0.9, 0.07, P.VERMILION)
            bottom = _text(c, M, 1.9, 11.4, s.title, 42, P.SERIF, leading=1.08)
            bottom = _text(c, M, max(bottom + 0.35, 4.55), 11, s.subtitle, 20, P.SANS,
                           P.SUBTLE_TEXT)
            _text(c, M, max(bottom + 0.5, 5.7), 11, s.byline, 12, P.MONO, P.COBALT)
        elif isinstance(s, Section):
            _text(c, M, 1.5, 3, f"{s.number:02d}", 96, P.SERIF, P.COBALT)
            _text(c, M, 3.3, 10, s.name, 16, P.MONO, P.COBALT, bold=True)
            _text(c, M, 3.85, 11, s.question, 32, P.SERIF, leading=1.1)
        else:
            _rect_pdf(c, M, HEAD_Y - 0.28, 0.5, 0.05, P.VERMILION)
            _text(c, M, HEAD_Y, W - 2 * M, s.claim, 28, P.SERIF, leading=1.08)
            if isinstance(s, Claim):
                fw, fh = _image_fit(s.figure, W - 2 * M, FOOT_Y - BODY_Y - 0.55)
                _image_pdf(c, s.figure, (W - fw) / 2, BODY_Y, fw, fh)
                if s.caption:
                    _text(c, M, FOOT_Y - 0.5, W - 2 * M, s.caption, 11, P.SANS, P.SUBTLE_TEXT)
            elif isinstance(s, ClaimSplit):
                y = BODY_Y + 0.1
                for point in s.points:
                    y = _text(c, M, y, 4.3, point, 17, P.SANS, leading=1.2) + 0.22
                fw, fh = _image_fit(s.figure, W - M - 5.4, FOOT_Y - BODY_Y - 0.55)
                _image_pdf(c, s.figure, 5.4 + (W - M - 5.4 - fw) / 2, BODY_Y, fw, fh)
                if s.caption:
                    _text(c, 5.4, FOOT_Y - 0.5, W - M - 5.4, s.caption, 11, P.SANS,
                          P.SUBTLE_TEXT)
            elif isinstance(s, BigNumber):
                _text(c, M, BODY_Y + 0.2, W - 2 * M, s.number, 80, P.SERIF, P.COBALT)
                _text(c, M, BODY_Y + 2.15, W - 2 * M, s.label, 22, P.SANS)
                _text(c, M, BODY_Y + 2.9, 10.5, s.context, 16, P.SANS, P.SUBTLE_TEXT)
            elif isinstance(s, Points):
                y = BODY_Y + 0.05
                step = (FOOT_Y - BODY_Y - 0.3) / max(len(s.items), 4)
                for i, (lead, text) in enumerate(s.items, 1):
                    if s.numbered:
                        _text(c, M, y, 0.6, f"{i}", 22, P.SERIF, P.COBALT)
                    x = M + (0.65 if s.numbered else 0)
                    _text(c, x, y, W - x - M, lead, 19, P.SANS, bold=True)
                    _text(c, x, y + 0.42, W - x - M, text, 15, P.SANS, P.SUBTLE_TEXT)
                    y += step
            elif isinstance(s, TableSlide):
                _table_pdf(c, s)
        if not isinstance(s, Title):
            _footer_pdf(c, deck, n, sec)
        c.showPage()
    c.save()


def _table_pdf(c, s: TableSlide) -> None:
    total = sum(s.widths)
    width = W - 2 * M
    xs = [M]
    for w in s.widths:
        xs.append(xs[-1] + width * w / total)
    y = BODY_Y
    _rect_pdf(c, M, y, width, 0.5, "#F3F5F7")
    for j, head in enumerate(s.header):
        _text(c, xs[j] + 0.12, y + 0.15, xs[j + 1] - xs[j] - 0.24, head, 12, P.MONO,
              P.SUBTLE_TEXT)
    y += 0.5
    for row in s.rows:
        bottoms = []
        for j, cell in enumerate(row):
            color, bold = P.INK, False
            if j == s.highlight_column:
                color = P.COBALT if cell.lower().startswith(("yes", "supported")) else P.VERMILION
                bold = True
            bottoms.append(_text(c, xs[j] + 0.12, y + 0.2, xs[j + 1] - xs[j] - 0.24, cell, 17,
                                 P.SANS, color, bold=bold))
        y = max(bottoms) + 0.22
        _rect_pdf(c, M, y, width, 0.01, P.RULE)
    if s.caption:
        _text(c, M, FOOT_Y - 0.5, W - 2 * M, s.caption, 11, P.SANS, P.SUBTLE_TEXT)


def render(deck: Deck, out_dir: Path, stem: str = "presentation") -> None:
    """Write ``<stem>.pptx`` and ``<stem>.pdf`` and check every figure exists first."""
    missing = [s.figure for s in deck.slides
               if isinstance(s, (Claim, ClaimSplit)) and not Path(s.figure).exists()]
    if missing:
        raise FileNotFoundError(f"Missing figures: {missing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    render_pptx(deck, out_dir / f"{stem}.pptx")
    render_pdf(deck, out_dir / f"{stem}.pdf")
