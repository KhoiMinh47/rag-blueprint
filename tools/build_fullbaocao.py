#!/usr/bin/env python3
"""Build the eight-page consolidated NeMo-Retriever report."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Flowable,
    HRFlowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MD = ROOT / "fullbaocao.md"
DEFAULT_OUTPUT = ROOT / "fullbaocao.pdf"

PAGE_W, PAGE_H = A4
LEFT = 17 * mm
RIGHT = 17 * mm
TOP = 18 * mm
BOTTOM = 15 * mm
CONTENT_W = PAGE_W - LEFT - RIGHT

BG = colors.HexColor("#F3F8F5")
NAVY = colors.HexColor("#102A43")
INK = colors.HexColor("#1B2B34")
MUTED = colors.HexColor("#63757D")
LINE = colors.HexColor("#D5E2DE")
TEAL = colors.HexColor("#159A91")
BLUE = colors.HexColor("#1976A8")
PURPLE = colors.HexColor("#7563A8")
ORANGE = colors.HexColor("#D97727")
GREEN = colors.HexColor("#238B62")
PALE_BLUE = colors.HexColor("#EAF4FA")
PALE_TEAL = colors.HexColor("#E8F6F2")
PALE_PURPLE = colors.HexColor("#F0EDFA")
PALE_ORANGE = colors.HexColor("#FFF1E4")
WHITE = colors.white


def font_paths() -> tuple[Path, Path]:
    candidates = (
        (
            Path("/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ),
    )
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            return regular, bold
    raise FileNotFoundError("No Unicode font with Vietnamese glyphs was found")


REGULAR, BOLD = font_paths()
pdfmetrics.registerFont(TTFont("FullRegular", str(REGULAR)))
pdfmetrics.registerFont(TTFont("FullBold", str(BOLD)))


def markup(value: str) -> str:
    escaped = html.escape(str(value), quote=False)
    return re.sub(r"`([^`]+)`", r'<font name="FullBold">\1</font>', escaped)


def flow_markup(value: str) -> str:
    return markup(value).replace("\n", "<br/>")


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="BodyFull", parent=styles["BodyText"], fontName="FullRegular",
    fontSize=9.1, leading=12.7, textColor=INK, spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="SmallFull", parent=styles["BodyText"], fontName="FullRegular",
    fontSize=7.3, leading=9.5, textColor=MUTED, spaceAfter=3,
))
styles.add(ParagraphStyle(
    name="KickerFull", parent=styles["BodyText"], fontName="FullBold",
    fontSize=7.7, leading=10, textColor=TEAL, spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="TitleFull", parent=styles["Title"], fontName="FullBold",
    fontSize=25, leading=29, textColor=NAVY, alignment=TA_LEFT, spaceAfter=10,
))
styles.add(ParagraphStyle(
    name="SubtitleFull", parent=styles["BodyText"], fontName="FullRegular",
    fontSize=11.5, leading=15, textColor=TEAL, spaceAfter=9,
))
styles.add(ParagraphStyle(
    name="H1Full", parent=styles["Heading1"], fontName="FullBold",
    fontSize=16.5, leading=20, textColor=NAVY, spaceBefore=1, spaceAfter=6,
    keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="H2Full", parent=styles["Heading2"], fontName="FullBold",
    fontSize=11.4, leading=14, textColor=BLUE, spaceBefore=6, spaceAfter=4,
    keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="H3Full", parent=styles["Heading3"], fontName="FullBold",
    fontSize=9.7, leading=12, textColor=PURPLE, spaceBefore=4, spaceAfter=3,
    keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="TableFull", parent=styles["BodyText"], fontName="FullRegular",
    fontSize=7.55, leading=9.65, textColor=INK, spaceAfter=0,
))
styles.add(ParagraphStyle(
    name="TableHeaderFull", parent=styles["BodyText"], fontName="FullBold",
    fontSize=7.7, leading=9.5, textColor=WHITE, spaceAfter=0,
))
styles.add(ParagraphStyle(
    name="CalloutFull", parent=styles["BodyText"], fontName="FullRegular",
    fontSize=8.1, leading=10.7, textColor=INK, spaceAfter=0,
))
styles.add(ParagraphStyle(
    name="CodeFull", parent=styles["Code"], fontName="FullRegular",
    fontSize=7.7, leading=10.5, textColor=NAVY, leftIndent=7, rightIndent=7,
    spaceBefore=2, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="CardFull", parent=styles["BodyText"], fontName="FullRegular",
    fontSize=7.55, leading=10, textColor=INK, spaceAfter=0,
))
styles.add(ParagraphStyle(
    name="FlowBoxFull", parent=styles["BodyText"], fontName="FullBold",
    fontSize=6.8, leading=8.1, alignment=TA_CENTER, textColor=NAVY,
    spaceAfter=0,
))
styles.add(ParagraphStyle(
    name="TOCFull", parent=styles["BodyText"], fontName="FullRegular",
    fontSize=8.6, leading=11, textColor=INK, spaceAfter=0,
))
styles.add(ParagraphStyle(
    name="CoverLabelFull", parent=styles["BodyText"], fontName="FullBold",
    fontSize=8.4, leading=11, textColor=TEAL, alignment=TA_CENTER, spaceAfter=9,
))
styles.add(ParagraphStyle(
    name="CoverTitleFull", parent=styles["Title"], fontName="FullBold",
    fontSize=30, leading=35, textColor=NAVY, alignment=TA_CENTER, spaceAfter=10,
))
styles.add(ParagraphStyle(
    name="CoverSubtitleFull", parent=styles["BodyText"], fontName="FullRegular",
    fontSize=13, leading=17, textColor=TEAL, alignment=TA_CENTER, spaceAfter=12,
))
styles.add(ParagraphStyle(
    name="CoverDateFull", parent=styles["BodyText"], fontName="FullRegular",
    fontSize=8.5, leading=11, textColor=MUTED, alignment=TA_CENTER, spaceAfter=0,
))


def P(text: str, style: str = "BodyFull") -> Paragraph:
    return Paragraph(markup(text), styles[style])


def rich(text: str, style: str = "BodyFull") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet_list(items: list[str], color: colors.Color = TEAL) -> ListFlowable:
    return ListFlowable(
        [ListItem(P(item, "BodyFull"), leftIndent=7) for item in items],
        bulletType="bullet", bulletFontName="FullBold", bulletFontSize=6.5,
        bulletColor=color, leftIndent=14, bulletOffsetY=1, spaceAfter=3,
    )


def make_table(
    data: list[list[str | Paragraph]],
    widths: list[float],
    *,
    header: bool = True,
    row_bgs: list[colors.Color] | None = None,
    row_heights: list[float] | None = None,
) -> Table:
    prepared: list[list[Paragraph]] = []
    for index, row in enumerate(data):
        style = "TableHeaderFull" if header and index == 0 else "TableFull"
        prepared.append([
            value if isinstance(value, Paragraph) else P(value, style)
            for value in row
        ])
    commands: list[tuple] = [
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ])
        start = 1
    else:
        start = 0
    for index in range(start, len(prepared)):
        if row_bgs:
            background = row_bgs[(index - start) % len(row_bgs)]
        else:
            background = WHITE if (index - start) % 2 == 0 else colors.HexColor("#F2F7F5")
        commands.append(("BACKGROUND", (0, index), (-1, index), background))
    result = Table(
        prepared,
        colWidths=widths,
        rowHeights=row_heights,
        repeatRows=1 if header else 0,
    )
    result.setStyle(TableStyle(commands))
    return result


def callout(text: str, background: colors.Color = PALE_BLUE, accent: colors.Color = BLUE) -> Table:
    result = Table([[P(text, "CalloutFull")]], colWidths=[CONTENT_W])
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("BOX", (0, 0), (-1, -1), 0.7, accent),
        ("LINEBEFORE", (0, 0), (0, -1), 4, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return result


class FlowDiagram(Flowable):
    """Draw connected rounded boxes with a real arrow between each stage."""

    def __init__(self, labels: list[str], fills: list[colors.Color], *, height: float = 68):
        super().__init__()
        self.labels = labels
        self.fills = fills
        self.height = height
        self.width = CONTENT_W

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        self.width = min(self.width, available_width)
        return self.width, self.height

    def draw(self) -> None:
        canvas = self.canv
        gap = 13
        box_h = min(54, max(42, self.height - 18))
        y = (self.height - box_h) / 2
        count = len(self.labels)
        box_w = (self.width - gap * (count - 1)) / count
        for index, label in enumerate(self.labels):
            x = index * (box_w + gap)
            canvas.setFillColor(self.fills[index % len(self.fills)])
            canvas.setStrokeColor(LINE)
            canvas.setLineWidth(0.7)
            canvas.roundRect(x, y, box_w, box_h, 6, fill=1, stroke=1)
            paragraph = Paragraph(flow_markup(label), styles["FlowBoxFull"])
            pw, ph = paragraph.wrap(box_w - 8, box_h - 4)
            paragraph.drawOn(canvas, x + (box_w - pw) / 2, y + (box_h - ph) / 2)
            if index < count - 1:
                start_x = x + box_w + 2
                end_x = x + box_w + gap - 3
                mid_y = y + box_h / 2
                canvas.setStrokeColor(TEAL)
                canvas.setFillColor(TEAL)
                canvas.setLineWidth(1.2)
                canvas.line(start_x, mid_y, end_x, mid_y)
                canvas.line(end_x, mid_y, end_x - 4, mid_y + 2.5)
                canvas.line(end_x, mid_y, end_x - 4, mid_y - 2.5)


def page_header(number: str, title: str, subtitle: str | None = None) -> list:
    result = [P(f"{number}  {title}", "H1Full")]
    if subtitle:
        result.append(P(subtitle, "SmallFull"))
    result.append(HRFlowable(width="100%", thickness=0.7, color=LINE, spaceBefore=1, spaceAfter=8))
    return result


def pipeline_card(title: str, body: str, background: colors.Color, accent: colors.Color) -> Table:
    content = rich(f'<font name="FullBold" color="{accent.hexval()}">{html.escape(title)}</font><br/>{markup(body)}', "CardFull")
    result = Table([[content]], colWidths=[CONTENT_W / 2 - 5], rowHeights=[82])
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("LINEBEFORE", (0, 0), (0, -1), 3, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return result


def on_page(canvas: Canvas, document: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setFillColor(BG)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, 5.5 * mm, PAGE_H, fill=1, stroke=0)
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(LEFT, PAGE_H - 11 * mm, PAGE_W - RIGHT, PAGE_H - 11 * mm)
    canvas.setFont("FullBold", 7.1)
    canvas.setFillColor(MUTED)
    canvas.drawString(LEFT, PAGE_H - 8.3 * mm, "NeMo-Retriever  ·  BÁO CÁO CHUNG")
    canvas.drawRightString(PAGE_W - RIGHT, PAGE_H - 8.3 * mm, "INGEST + OCR")
    canvas.line(LEFT, 10 * mm, PAGE_W - RIGHT, 10 * mm)
    canvas.setFont("FullRegular", 7.1)
    canvas.drawString(LEFT, 6.2 * mm, "Kiến trúc ingest và bốn pipeline OCR")
    canvas.drawRightString(PAGE_W - RIGHT, 6.2 * mm, f"{canvas.getPageNumber():02d}")
    canvas.restoreState()


def build_story() -> list:
    story: list = []

    # Page 1 — cover only.
    story.extend([Spacer(1, 66 * mm), P("BÁO CÁO CHUNG", "CoverLabelFull")])
    story.append(P("NEMO-RETRIEVER", "CoverTitleFull"))
    story.append(P("Kiến trúc ingest và bốn pipeline OCR", "CoverSubtitleFull"))
    story.append(Spacer(1, 9))
    story.append(HRFlowable(width="42%", thickness=1.2, color=TEAL, hAlign="CENTER", spaceBefore=1, spaceAfter=13))
    story.append(P("13.08.2026", "CoverDateFull"))
    story.append(PageBreak())

    # Page 2 — table of contents only.
    story.extend(page_header("01 /", "Mục lục", "Khung báo cáo được dựng trước, nội dung đi theo đúng thứ tự pipeline."))
    story.append(make_table([
        ["Trang", "Nội dung", "Mục tiêu"],
        ["1", "Bìa", "Tên báo cáo và ngày tổng hợp."],
        ["2", "Mục lục", "Khung đọc của báo cáo chung."],
        ["3", "Kiến trúc chung", "Gộp luồng ingest và điểm tách của bốn route."],
        ["4", "Pipeline Qwen", "Flow, cấu hình và số đo Qwen."],
        ["5", "Pipeline Mistral", "Flow, cấu hình và số đo Ministral."],
        ["6", "Kiến trúc NVIDIA", "Chuỗi component nền và kết quả development."],
        ["7", "Kiến trúc thử nghiệm", "Routing ngôn ngữ và fallback."],
        ["8", "Kết quả và case đặc thù", "Benchmark gộp, scan, nhiều trang, dấu và hình ảnh."],
    ], [48, 180, CONTENT_W - 228], row_bgs=[WHITE, PALE_BLUE], row_heights=[34, 43, 43, 43, 43, 43, 43, 43, 43]))
    story.append(PageBreak())

    # Page 3 — merged architecture overview.
    story.extend(page_header("02 /", "Kiến trúc chung", "Gộp luồng ingest và điểm tách bốn route."))
    story.append(FlowDiagram(
        ["File đầu vào", "Nhận diện\nchuẩn hóa", "PDFium", "Page Elements", "Chọn 1 / 4\nroute"],
        [PALE_BLUE, PALE_TEAL, PALE_PURPLE, PALE_ORANGE, colors.HexColor("#EAF1FF")],
    ))
    story.append(Spacer(1, 4))
    story.append(P("Hệ thống nhận PDF, DOCX/PPTX, TXT, HTML, spreadsheet, image, audio và video. DOCX/PPTX có thể được chuẩn hóa về PDF; XLSX/XLS/CSV đọc trực tiếp bằng spreadsheet parser, không rasterize hoặc OCR mặc định."))
    story.append(P("PDFium tận dụng native text và bbox khi có thể. Trang scan hoặc vùng thiếu text được render thành ảnh. Page Elements v3 nhận diện text, title, table, chart, image và infographic rồi chuyển geometry cho route đã chọn."))
    story.append(P("Bốn điểm tách", "H2Full"))
    story.append(make_table([
        ["Route", "Chuỗi xử lý", "Vai trò"],
        ["Qwen", "Detect → native/scan → Qwen VLM", "OCR scan full-page và bổ sung vùng thiếu ở native."],
        ["Mistral", "Semantic bbox → crop → Ministral", "OCR theo vùng; layout yếu dùng full-page fallback."],
        ["NVIDIA", "Page Elements → Table Structure → Nemotron → Embedding", "Chuỗi component nền cho nhận diện, OCR và embedding."],
        ["Thử nghiệm", "Line crop → language probe → Tesseract/Nemotron", "Routing theo ngôn ngữ, quality check và fallback."],
    ], [72, 205, CONTENT_W - 277], row_bgs=[WHITE, PALE_TEAL], row_heights=[34, 48, 48, 54, 48]))
    story.append(Spacer(1, 10))
    story.append(P("Nguyên tắc dùng chung", "H2Full"))
    story.append(make_table([
        ["Có native text", "Cần OCR", "Sau OCR"],
        [
            "Giữ text và bbox để giảm số crop, giảm request và bảo toàn reading order.",
            "Render full-page hoặc crop theo geometry khi trang scan hay vùng thiếu text.",
            "Ghép text, table và visual theo page order rồi clean/chunk trước embedding.",
        ],
    ], [130, 190, CONTENT_W - 320], row_bgs=[WHITE, PALE_BLUE], row_heights=[32, 70]))
    story.append(PageBreak())

    # Page 4 — Qwen from source report page 4.
    story.extend(page_header("03 /", "Pipeline Qwen", "Page Detect → Qwen 3.5 VLM"))
    story.append(FlowDiagram(
        ["PDFium\ntext + raster", "Page Elements\ndetect", "Crop / queue", "Qwen\nVLM", "Page order"],
        [PALE_BLUE, PALE_TEAL, PALE_ORANGE, PALE_PURPLE, PALE_TEAL],
    ))
    story.append(Spacer(1, 5))
    story.append(P("Qwen giữ native text làm nguồn chính ở trang native. Vùng thiếu character geometry, table hoặc visual hợp lệ mới đi tiếp qua VLM. Trang scan gửi ảnh nguyên trang; table gửi crop riêng để dựng Markdown."))
    story.append(P("Cấu hình", "H2Full"))
    story.append(make_table([
        ["Trường", "Giá trị"],
        ["PDF / detect", "Extract batch 16 × 4 worker; detect/crop batch 128."],
        ["Streaming", "Block 16 trang; queue tối đa 2 block."],
        ["Qwen client", "Batch 8; tối đa 8 request đồng thời; CCR 8."],
        ["vLLM", "Max model length 32K; max sequence 8; max batched tokens 4096; GPU utilization 0.20."],
        ["Prompt", "`word_repair`; temperature 0; thinking tắt."],
    ], [115, CONTENT_W - 115], row_heights=[32, 42, 42, 42, 54, 38]))
    story.append(P("Số đo", "H2Full"))
    story.append(make_table([
        ["Mốc", "Kết quả"],
        ["Scan 40 trang", "32,69 giây; 40/40 đúng thứ tự; 41 requests; khoảng 1.097 token/s aggregate."],
        ["Request đầu / VRAM", "Sau 2,38 giây / khoảng 15,1 GB."],
        ["Native 4 trang", "0,59 giây; 4/4 native pages; 1 table request; `errors=[]`."],
        ["Native report 6 trang", "1,87 giây; 3 bbox gần toàn trang bị loại; không lỗi."],
        ["Chất lượng", "Similarity 0,9991; exact 2/4 trên corpus 4 trang."],
    ], [135, CONTENT_W - 135], row_bgs=[WHITE, PALE_PURPLE], row_heights=[32, 48, 42, 48, 48, 44]))
    story.append(PageBreak())

    # Page 5 — Mistral from source report page 5.
    story.extend(page_header("04 /", "Pipeline Mistral", "Semantic crop → Ministral VLM"))
    story.append(FlowDiagram(
        ["PDFium", "Semantic\nbbox", "Semantic\ncrop", "Ministral\nFP8", "OCR output"],
        [PALE_BLUE, PALE_TEAL, PALE_ORANGE, PALE_PURPLE, PALE_TEAL],
        height=52,
    ))
    story.append(Spacer(1, 5))
    story.append(P("Mistral giữ native text có geometry, sau đó gửi semantic crop text/title/table/visual khi cần. Table dùng whole-table crop để trả Markdown. Scan hoặc layout yếu có thể thêm full-page OCR unit với bbox `[0, 0, 1, 1]`."))
    story.append(P("Cấu hình", "H2Full"))
    story.append(make_table([
        ["Trường", "Giá trị"],
        ["Model", "`mistralai/Ministral-3-3B-Instruct-2512`"],
        ["vLLM", "Max model length 8192; max sequence 10; max batched tokens 4096."],
        ["GPU / VRAM", "GPU utilization 0.33; VRAM báo cáo 16.5 GB."],
        ["Client", "VLM batch/workers 10; max output 1024 token."],
        ["Stage tắt", "Table Structure, line detector và language probe."],
    ], [125, CONTENT_W - 125], row_heights=[28, 34, 34, 34, 34, 30]))
    story.append(P("Kết quả thử nghiệm", "H2Full"))
    story.append(make_table([
        ["Hạng mục", "Kết quả đo", "Ghi chú"],
        ["Mistral · scan 40 trang", "53,89s · file 40 trang · 41 rows", "Thứ tự/request đầu chưa được log lưu"],
        ["Mistral · VRAM", "4,44GiB model load + 10,17GiB KV cache", "Page Elements startup 470MiB · không có peak per-job"],
        ["Mistral · native 4 trang", "1,05s · 4 trang · 6 rows", "8 requests: 1 text + 5 table + 2 visual"],
        ["Mistral · report scan 6 trang", "11,13s · 7 rows", "7 requests: 6 full-page + 1 table · không lỗi"],
        ["Mistral · chất lượng", "Regression 4 rows trong 4,66s", "Similarity/ground truth chưa được persist"],
    ], [132, 205, CONTENT_W - 337], row_bgs=[WHITE, PALE_ORANGE], row_heights=[30, 55, 58, 52, 55, 52]))
    story.append(PageBreak())

    # Page 6 — NVIDIA architecture from source report page 4.
    story.extend(page_header("05 /", "Kiến trúc NVIDIA", "Chuỗi component nền cho nhận diện, OCR và embedding."))
    story.append(FlowDiagram(
        ["Page Elements\nv3", "Table Structure\nv1", "Nemotron OCR\nv2", "Embedding"],
        [PALE_BLUE, PALE_TEAL, PALE_PURPLE, PALE_ORANGE],
    ))
    story.append(Spacer(1, 7))
    story.append(P("Trong kiến trúc này, Page Elements v3 tìm semantic region; Table Structure v1 lấy cell, row và column; Nemotron OCR v2 đọc crop hoặc full-page; dữ liệu sau đó được chuẩn bị cho vector representation."))
    story.append(P("Chức năng trong pipeline", "H2Full"))
    story.append(make_table([
        ["Component", "Chức năng"],
        ["Nhận diện file", "Phân loại PDF, DOCX/PPTX, image, text, HTML, spreadsheet, audio và video."],
        ["DocToPdf / PDFSplit", "Chuẩn hóa DOCX/PPTX về PDF và tách tài liệu theo trang."],
        ["PDFium", "Đọc native text và bbox; render các trang cần OCR."],
        ["Page Elements v3", "Phát hiện text, title, table, chart, image và infographic."],
        ["Table Structure v1", "Phát hiện cell, row và column trong vùng table."],
        ["Nemotron OCR v2", "Đọc crop hoặc full-page OCR, có thể làm fallback."],
        ["Clean / chunk / embedding", "Loại nội dung trùng, tạo rows/chunk và chuẩn bị vector."],
    ], [145, CONTENT_W - 145], row_heights=[32, 46, 46, 46, 46, 46, 46, 46]))
    story.append(Spacer(1, 9))
    story.append(callout(
        "Kết quả development được ghi nhận ổn định hơn với file tiếng Anh. Tiếng Việt vẫn có thể sai dấu hoặc cần một route OCR phù hợp.",
        PALE_BLUE, BLUE,
    ))
    story.append(PageBreak())

    # Page 7 — experimental architecture from source report page 5.
    story.extend(page_header("06 /", "Kiến trúc thử nghiệm", "Routing OCR theo ngôn ngữ và fallback."))
    story.append(FlowDiagram(
        ["OCR unit", "Language\nprobe", "Language\ngate"],
        [PALE_BLUE, PALE_TEAL, PALE_PURPLE],
        height=56,
    ))
    story.append(Spacer(1, 2))
    story.append(FlowDiagram(
        ["Tiếng Việt", "Tesseract `vie`", "Quality check", "Output"],
        [PALE_TEAL, PALE_BLUE, PALE_ORANGE, PALE_PURPLE],
        height=56,
    ))
    story.append(Spacer(1, 2))
    story.append(FlowDiagram(
        ["English / mixed", "Nemotron OCR v2", "Output"],
        [PALE_ORANGE, PALE_PURPLE, PALE_TEAL],
        height=56,
    ))
    story.append(Spacer(1, 5))
    story.append(P("Kiến trúc thử nghiệm bổ sung PP-OCRv6 medium cho line detector và line crop, Tesseract 5 cho probe/tiếng Việt, cùng Nemotron OCR v2 làm backend tiếng Anh và fallback."))
    story.append(P("Route và điều kiện", "H2Full"))
    story.append(make_table([
        ["Route", "Xử lý"],
        ["Tiếng Việt", "Language probe phát hiện tín hiệu tiếng Việt; crop đi tới Tesseract `vie`; quality check giữ hoặc chuyển fallback."],
        ["English / mixed / uncertain", "Nemotron OCR v2 nhận crop trực tiếp hoặc nhận fallback khi kết quả route trước không đạt."],
        ["Quality check", "Ngưỡng quality check là `0.80`."],
        ["Phạm vi crop", "Text, title và table-cell; giữ bbox/provenance khi ghép output."],
    ], [150, CONTENT_W - 150], row_bgs=[WHITE, PALE_TEAL], row_heights=[32, 56, 56, 46, 52]))
    story.append(Spacer(1, 8))
    story.append(callout(
        "Language probe và fallback có thể cải thiện độ chính xác, nhưng thêm bước xử lý và làm latency tăng.",
        PALE_PURPLE, PURPLE,
    ))
    story.append(PageBreak())

    # Page 8 — merged final results and special cases from both source page 6s.
    story.extend(page_header("07 /", "Kết quả và các case đặc thù", "Gộp benchmark với scan, file nhiều trang, dấu, tiêu ngữ và hình ảnh."))
    story.append(P("Benchmark đối chiếu", "H2Full"))
    story.append(make_table([
        ["Hạng mục", "Qwen", "Mistral"],
        ["Scan 40 trang", "32,69 giây; 40/40 đúng thứ tự; 41 requests; request đầu 2,38 giây.", "53,89s · file 40 trang · 41 rows; thứ tự/request đầu chưa được log lưu."],
        ["Native 4 trang", "0,59 giây; 4/4 native; 1 table request; `errors=[]`.", "1,05s · 4 trang · 6 rows; 8 requests: 1 text + 5 table + 2 visual."],
        ["Report 6 trang", "1,87 giây; 3 bbox gần toàn trang bị loại; không lỗi.", "11,13s · 7 rows; 7 requests: 6 full-page + 1 table · không lỗi."],
        ["Chất lượng", "Similarity 0,9991; exact 2/4.", "Regression 4 rows trong 4,66s; similarity/ground truth chưa được persist."],
    ], [102, 190, CONTENT_W - 292], row_bgs=[WHITE, PALE_BLUE], row_heights=[32, 56, 56, 56, 50]))
    story.append(Spacer(1, 6))
    story.append(P("Các trường hợp dữ liệu", "H2Full"))
    story.append(make_table([
        ["Case", "Cách xử lý"],
        ["File scan", "Full-page OCR, overlapping tiles hoặc line detector + projection fallback tùy route."],
        ["File nhiều trang", "Giữ page number và reading order; latency tăng theo số trang và số crop."],
        ["Dấu", "Stamp detector có thể phát hiện; cấu hình mô tả `extract_stamps = false`, nên không mặc định tạo graph node."],
        ["Tiêu ngữ", "Xử lý như text/title nếu Page Elements nhận diện được."],
        ["Hình ảnh / visual", "Giữ image evidence khi phù hợp; gate loại bbox gần toàn trang chứa nhiều text trước khi OCR."],
    ], [105, CONTENT_W - 105], row_heights=[32, 52, 52, 52, 46, 52]))
    story.append(Spacer(1, 7))
    story.append(P("Kết luận: NeMo-Retriever có một luồng ingest chung và bốn route với mục tiêu khác nhau. Các số benchmark được giữ theo đúng input và nguồn của từng phép đo; trường chưa được log lưu lại không được nội suy.", "BodyFull"))
    return story


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not SOURCE_MD.exists():
        raise SystemExit(f"Missing source markdown: {SOURCE_MD}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(args.output), pagesize=A4, leftMargin=LEFT, rightMargin=RIGHT,
        topMargin=TOP, bottomMargin=BOTTOM,
        title="Báo cáo chung NeMo-Retriever",
        author="NeMo-Retriever development workspace",
        subject="Kiến trúc ingest và bốn pipeline OCR",
        creator="ReportLab",
    )
    document.build(build_story(), onFirstPage=on_page, onLaterPages=on_page)
    print(args.output)


if __name__ == "__main__":
    main()
