from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
                                PageBreak, KeepTogether)

D = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", D + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", D + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", D + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DV-BI", D + "DejaVuSans-BoldOblique.ttf"))
pdfmetrics.registerFont(TTFont("DVM", D + "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

NAVY = colors.HexColor("#1F3A5F")
BLUE = colors.HexColor("#3B6EA5")
LIGHT = colors.HexColor("#EAF1F8")
RED = colors.HexColor("#D9534F")
GREY = colors.HexColor("#555555")

S = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=22, leading=27, textColor=NAVY, spaceAfter=6),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=11, leading=15, textColor=GREY, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=14.5, leading=19, textColor=NAVY, spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=11.5, leading=15, textColor=BLUE, spaceBefore=9, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.4, leading=14, spaceAfter=5, alignment=TA_LEFT),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=9.4, leading=13.5, leftIndent=14, bulletIndent=3, spaceAfter=2),
    "def": ParagraphStyle("def", fontName="DV", fontSize=9.2, leading=13.5, textColor=colors.black),
    "formula": ParagraphStyle("formula", fontName="DVM", fontSize=9, leading=13, leftIndent=10, textColor=NAVY, spaceAfter=4),
    "cap": ParagraphStyle("cap", fontName="DV-I", fontSize=8, leading=11, textColor=GREY, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.4, leading=11),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=8.4, leading=11, textColor=colors.white),
}


def P(t, s="body"):
    return Paragraph(t, S[s])


def bullets(items):
    return [Paragraph(i, S["bullet"], bulletText="\u2022") for i in items]


def defbox(term, text):
    """Shaded definition box."""
    t = Table([[Paragraph(f"<b>{term}</b><br/>{text}", S["def"])]], colWidths=[17 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("LINEBEFORE", (0, 0), (0, -1), 3, BLUE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return [t, Spacer(1, 6)]


def table(data, widths, header=True, zebra=True, font=8.4):
    rows = []
    for i, r in enumerate(data):
        st = S["cellb"] if (header and i == 0) else S["cell"]
        rows.append([Paragraph(str(c), st) for c in r])
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C5CED8")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F5F8FB")))
    t.setStyle(TableStyle(style))
    return t


def footer(title):
    def _f(canvas, doc):
        canvas.saveState()
        canvas.setFont("DV", 7.5)
        canvas.setFillColor(GREY)
        canvas.drawString(2 * cm, 1.2 * cm, title)
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()
    return _f


def build(path, story, title):
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=2 * cm, title=title, author="Risk Alert Classifier project")
    doc.build(story, onFirstPage=footer(title), onLaterPages=footer(title))
