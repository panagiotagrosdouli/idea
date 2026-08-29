from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "PAPER_DRAFT.md"
OUTPUT = (
    ROOT
    / "output"
    / "pdf"
    / "predictive_pc_fmcw_corrected_research_draft.pdf"
)
STYLES: dict[str, ParagraphStyle] = {}


def _register_fonts() -> None:
    pdfmetrics.registerFont(
        TTFont("DejaVuSerif", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
    )
    pdfmetrics.registerFont(
        TTFont(
            "DejaVuSerif-Bold",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        )
    )
    pdfmetrics.registerFont(
        TTFont(
            "DejaVuSansMono-Oblique",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Oblique.ttf",
        )
    )
    pdfmetrics.registerFont(
        TTFont(
            "DejaVuSansMono",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        )
    )


def _inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(
        r"(?<!\*)\*([^*]+)\*(?!\*)",
        r'<font name="DejaVuSansMono-Oblique">\1</font>',
        escaped,
    )
    escaped = re.sub(
        r"`(.+?)`", r'<font name="DejaVuSansMono" size="7.5">\1</font>', escaped
    )
    return escaped


def _image(path: Path, max_width: float, max_height: float) -> Image:
    image = Image(str(path))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image


def _table(lines: list[str], width: float) -> Table:
    rows = []
    for index, line in enumerate(lines):
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if index == 1 and all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append([Paragraph(_inline(cell), STYLES["TableCell"]) for cell in cells])
    column_width = width / len(rows[0])
    table = Table(
        rows,
        colWidths=[column_width] * len(rows[0]),
        repeatRows=1,
        hAlign="CENTER",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
                ("FONTNAME", (0, 0), (-1, 0), "DejaVuSerif-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#6B7280")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("DejaVuSerif", 7.5)
    canvas.setFillColor(colors.HexColor("#4B5563"))
    canvas.drawString(
        18 * mm, 10 * mm, "Predictive PC-FMCW/DPSK - corrected research draft"
    )
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="DejaVuSerif-Bold",
            fontSize=17,
            leading=21,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#111827"),
            spaceAfter=10,
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="DejaVuSerif-Bold",
            fontSize=11.5,
            leading=14,
            textColor=colors.HexColor("#1F4E79"),
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="DejaVuSerif",
            fontSize=8.6,
            leading=11.4,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#111827"),
            spaceAfter=5,
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="DejaVuSerif",
            fontSize=7.4,
            leading=9.2,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#374151"),
            spaceBefore=2,
            spaceAfter=7,
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName="DejaVuSerif",
            fontSize=6.6,
            leading=8.1,
        ),
    }


def build() -> Path:
    _register_fonts()
    global STYLES
    STYLES = _styles()
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="When Does Trajectory Prediction Help PC-FMCW/DPSK Scheduling?",
        author="Author to be inserted",
    )
    width = A4[0] - document.leftMargin - document.rightMargin
    story = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(_inline(line[2:]), STYLES["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(_inline(line[3:]), STYLES["H2"]))
        elif line.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if match:
                story.append(
                    _image(ROOT / match.group(2), width, 92 * mm)
                )
        elif line.startswith("*") and line.endswith("*"):
            story.append(Paragraph(_inline(line[1:-1]), STYLES["Caption"]))
        elif line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            story.append(_table(table_lines, width))
            story.append(Spacer(1, 5))
            continue
        elif line.startswith("- "):
            items = []
            while index < len(lines) and lines[index].strip().startswith("- "):
                items.append(
                    ListItem(
                        Paragraph(_inline(lines[index].strip()[2:]), STYLES["Body"]),
                        leftIndent=10,
                    )
                )
                index += 1
            story.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    leftIndent=16,
                    bulletFontSize=6,
                )
            )
            continue
        else:
            paragraph = [line]
            while index + 1 < len(lines):
                next_line = lines[index + 1].strip()
                if (
                    not next_line
                    or next_line.startswith(("#", "|", "- ", "!["))
                    or (next_line.startswith("*") and next_line.endswith("*"))
                ):
                    break
                index += 1
                paragraph.append(next_line)
            story.append(Paragraph(_inline(" ".join(paragraph)), STYLES["Body"]))
        index += 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return OUTPUT


if __name__ == "__main__":
    print(build())
