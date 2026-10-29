from __future__ import annotations

import os
import re
import sys

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, NextPageTemplate,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOURCE = os.path.join(ROOT, "interview_qa.md")
TARGET = os.path.join(ROOT, "interview_qa.pdf")

FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__),
                        "mpl-data", "fonts", "ttf")

SECTION_COLOURS = {
    "A": colors.HexColor("#6c8ebf"),
    "B": colors.HexColor("#c0504d"),
    "C": colors.HexColor("#9bbb59"),
    "D": colors.HexColor("#8064a2"),
    "E": colors.HexColor("#e0973c"),
    "F": colors.HexColor("#2e8b7a"),
    "G": colors.HexColor("#c0504d"),
    "H": colors.HexColor("#4f6228"),
    "I": colors.HexColor("#7f7f7f"),
    "*": colors.HexColor("#7f7f7f"),
}

PAGE_W, PAGE_H = A4
TAB_W = 9 * mm
MARGIN_L = 16 * mm
MARGIN_R = TAB_W + 6 * mm
MARGIN_T = 16 * mm
MARGIN_B = 14 * mm


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("DJ", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DJ-B", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DJ-I", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFontFamily("DJ", normal="DJ", bold="DJ-B", italic="DJ-I")


def styles() -> dict:
    base = dict(fontName="DJ", alignment=TA_LEFT)
    return {
        "section": ParagraphStyle("section", **base, fontSize=17, leading=21,
                                  spaceBefore=0, spaceAfter=5,
                                  textColor=colors.HexColor("#111111")),
        "q_en": ParagraphStyle("q_en", **{**base, "fontName": "DJ-B"}, fontSize=11.5,
                               leading=14.5, spaceBefore=9, spaceAfter=1,
                               textColor=colors.HexColor("#1a1a1a")),
        "q_ru": ParagraphStyle("q_ru", **base, fontSize=8.6, leading=11,
                               spaceAfter=4, textColor=colors.HexColor("#8a8a8a")),
        "a_en": ParagraphStyle("a_en", **base, fontSize=10.6, leading=14.6,
                               textColor=colors.HexColor("#101010")),
        "a_ru": ParagraphStyle("a_ru", **base, fontSize=8.3, leading=11,
                               textColor=colors.HexColor("#6f6f6f")),
        "note": ParagraphStyle("note", **{**base, "fontName": "DJ-I"}, fontSize=8.6,
                               leading=11.4, spaceBefore=3, spaceAfter=2,
                               textColor=colors.HexColor("#b0392f")),
        "toc": ParagraphStyle("toc", **base, fontSize=11, leading=19),
        "toc_h": ParagraphStyle("toc_h", **{**base, "fontName": "DJ-B"}, fontSize=20,
                                leading=26, spaceAfter=10),
        "lead": ParagraphStyle("lead", **base, fontSize=9.4, leading=13,
                               textColor=colors.HexColor("#555555")),
    }


class CallSheet(BaseDocTemplate):

    def __init__(self, path: str) -> None:
        super().__init__(path, pagesize=A4,
                         leftMargin=MARGIN_L, rightMargin=MARGIN_R,
                         topMargin=MARGIN_T, bottomMargin=MARGIN_B,
                         title="Call sheet — Ewa Syta")
        frame = Frame(MARGIN_L, MARGIN_B,
                      PAGE_W - MARGIN_L - MARGIN_R,
                      PAGE_H - MARGIN_T - MARGIN_B, id="body")
        self.addPageTemplates([
            PageTemplate(id="plain", frames=[frame]),
            PageTemplate(id="tabbed", frames=[frame], onPageEnd=self._decorate),
        ])
        self.current = ("*", "")
        self.section_pages: dict = {}

    def afterFlowable(self, flowable) -> None:
        key = getattr(flowable, "section_key", None)
        if key:
            self.current = key
            self.section_pages.setdefault(key[0], self.page)

    def _decorate(self, canvas, doc) -> None:
        letter, title = self.current
        colour = SECTION_COLOURS.get(letter, SECTION_COLOURS["*"])

        slot = "ABCDEFGHI".find(letter)
        slot = 0 if slot < 0 else slot
        tab_h = 34 * mm
        top = PAGE_H - MARGIN_T - slot * (tab_h * 0.62)
        canvas.saveState()
        canvas.setFillColor(colour)
        canvas.rect(PAGE_W - TAB_W, top - tab_h, TAB_W, tab_h, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("DJ-B", 13)
        canvas.drawCentredString(PAGE_W - TAB_W / 2, top - 12 * mm, letter)
        canvas.restoreState()

        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#9a9a9a"))
        canvas.setFont("DJ", 7.6)
        canvas.drawString(MARGIN_L, 8 * mm, title)
        canvas.drawRightString(PAGE_W - MARGIN_R, 8 * mm, str(doc.page))
        canvas.restoreState()


class SectionMark(Spacer):

    def __init__(self, letter: str, title: str) -> None:
        super().__init__(1, 0)
        self.section_key = (letter, title)


def clean(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+?)`", r"<font face='DJ-I'>\1</font>", text)
    text = text.strip().strip("«»").strip()
    return text


def parse(md: str) -> list:
    lines = md.split("\n")
    sections: list = []
    current = None
    block = None
    i = 0

    def close_block():
        nonlocal block
        if block and (block["en"] or block["ru"] or block["notes"]):
            current["blocks"].append(block)
        block = None

    while i < len(lines):
        line = lines[i].rstrip()

        m = re.match(r"^# ([A-H])\.\s*(.+)$", line)
        if m:
            close_block()
            current = {"letter": m.group(1), "title": m.group(2).strip(), "blocks": []}
            sections.append(current)
            i += 1
            continue

        if line.startswith("# "):
            close_block()
            title = line[2:].strip()
            if not sections:
                current = None
                i += 1
                continue
            if sections and sections[-1]["letter"] == "I":
                current = sections[-1]
                current["blocks"].append({"q_en": title, "q_ru": "",
                                          "en": [], "ru": [], "notes": []})
            else:
                current = {"letter": "I", "title": "Справка", "blocks": [
                    {"q_en": title, "q_ru": "", "en": [], "ru": [], "notes": []}]}
                sections.append(current)
            block = current["blocks"][-1]
            i += 1
            continue

        if current is None:
            i += 1
            continue

        if line.startswith("### "):
            q = clean(line[4:])
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if nxt.startswith("### "):
                close_block()
                block = {"q_en": q, "q_ru": clean(nxt[4:]), "en": [], "ru": [], "notes": []}
                i += 2
            else:
                close_block()
                block = {"q_en": q, "q_ru": "", "en": [], "ru": [], "notes": []}
                i += 1
            continue

        if line.startswith(">"):
            body = line.lstrip("> ").strip()
            if block is None:
                block = {"q_en": "", "q_ru": "", "en": [], "ru": [], "notes": []}
            if body.startswith("**EN:**"):
                block["en"].append(clean(body[7:]))
                block["_last"] = "en"
            elif body.startswith("**RU:**"):
                block["ru"].append(clean(body[7:]))
                block["_last"] = "ru"
            elif body:
                block.setdefault("_last", "en")
                block[block["_last"]].append(clean(body))
            i += 1
            continue

        if line.strip() and not line.startswith("---"):
            if block is None:
                block = {"q_en": "", "q_ru": "", "en": [], "ru": [], "notes": []}
            para = [line.strip()]
            while i + 1 < len(lines) and lines[i + 1].strip()                     and not lines[i + 1].startswith(("#", ">", "---")):
                i += 1
                para.append(lines[i].strip())
            block["notes"].append(clean(" ".join(para)))
        i += 1

    close_block()
    return [s for s in sections if s["blocks"]]


def render(sections: list, st: dict, path: str, pages: dict) -> dict:
    doc = CallSheet(path)
    story: list = []

    story.append(Paragraph("Call sheet — Ewa Syta", st["toc_h"]))
    story.append(Paragraph(
        "Английский в рамке — то, что говоришь. Серое под ним — по-русски, "
        "чтобы понимать. <font color='#b0392f'><i>Красный курсив — заметки себе, "
        "вслух не читать.</i></font> Цветная вкладка на правом поле = раздел.",
        st["lead"]))
    story.append(Spacer(1, 7 * mm))

    toc_rows = []
    for s in sections:
        page = pages.get(s["letter"])
        toc_rows.append([
            Paragraph(f"<b>{s['letter']}</b>", st["toc"]),
            Paragraph(s["title"], st["toc"]),
            Paragraph(f"<b>стр. {page}</b>" if page else "—", st["toc"]),
        ])
    toc = Table(toc_rows, colWidths=[12 * mm, 108 * mm, 30 * mm])
    toc.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor("#dddddd")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    for idx, s in enumerate(sections):
        toc.setStyle(TableStyle([
            ("BACKGROUND", (0, idx), (0, idx),
             SECTION_COLOURS.get(s["letter"], SECTION_COLOURS["*"])),
            ("TEXTCOLOR", (0, idx), (0, idx), colors.white),
            ("ALIGN", (0, idx), (0, idx), "CENTER"),
        ]))
    story.append(toc)
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        "Не понял вопрос: <b>Sorry, could you say that again?</b> &nbsp;·&nbsp; "
        "Не знаешь ответа: <b>I haven't tested that. I could check it by…</b> "
        "&nbsp;·&nbsp; Цифру наугад — никогда.", st["lead"]))

    story.append(NextPageTemplate("tabbed"))
    from reportlab.platypus import PageBreak
    story.append(PageBreak())

    for s in sections:
        story.append(SectionMark(s["letter"], f"{s['letter']}. {s['title']}"))
        story.append(Paragraph(f"{s['letter']}. {s['title']}", st["section"]))
        colour = SECTION_COLOURS.get(s["letter"], SECTION_COLOURS["*"])

        for b in s["blocks"]:
            parts = []
            if b["q_en"]:
                parts.append(Paragraph(b["q_en"], st["q_en"]))
            if b["q_ru"]:
                parts.append(Paragraph(b["q_ru"], st["q_ru"]))

            if b["en"]:
                answer = Table(
                    [[Paragraph(" ".join(b["en"]), st["a_en"])]],
                    colWidths=[PAGE_W - MARGIN_L - MARGIN_R])
                answer.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f6f8")),
                    ("LINEBEFORE", (0, 0), (0, -1), 2.2, colour),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                parts.append(answer)
            if b["ru"]:
                parts.append(Spacer(1, 1.6 * mm))
                parts.append(Paragraph(" ".join(b["ru"]), st["a_ru"]))
            for note in b["notes"]:
                parts.append(Paragraph(note, st["note"]))
            parts.append(Spacer(1, 2.4 * mm))

            story.append(KeepTogether(parts))

    doc.build(story)
    return doc.section_pages


def build() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    register_fonts()
    st = styles()

    with open(SOURCE, "r", encoding="utf-8") as handle:
        sections = parse(handle.read())

    scratch = TARGET + ".pass1"
    pages = render(sections, st, scratch, {})
    render(sections, st, TARGET, pages)
    try:
        os.remove(scratch)
    except OSError:
        pass

    print(f"wrote {TARGET}")
    for s in sections:
        print(f"    {s['letter']}  {s['title']:<24} стр. {pages.get(s['letter'], '?')}"
              f"   ({len(s['blocks'])} блоков)")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
