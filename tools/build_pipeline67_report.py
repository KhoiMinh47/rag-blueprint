#!/usr/bin/env python3
"""Build the A4 visual report for the Qwen and Mistral pipelines.

The layout intentionally follows ``thongtinbaocao.pdf``: an A4 portrait
canvas, a soft green background, black-green typography, rounded cards, and
compact architecture diagrams.  The source facts are kept in the companion
``bao_cao_pipeline_6_7.md`` file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "bao_cao_pipeline_6_7.pdf"
W, H = A4
MARGIN = 42
CONTENT_W = W - 2 * MARGIN

# Black-green palette: near-black structure, soft green surfaces, and
# several green accents so the two pipelines remain visually distinct.
BG = colors.HexColor("#F2F7F3")
NAVY = colors.HexColor("#07130D")
INK = colors.HexColor("#142219")
MUTED = colors.HexColor("#617267")
LINE = colors.HexColor("#D6E3D9")
TABLE_LINE = colors.HexColor("#B7CFBD")
CYAN = colors.HexColor("#159A63")
BLUE = colors.HexColor("#0B6B46")
PURPLE = colors.HexColor("#2F8F56")
ORANGE = colors.HexColor("#6B954B")
GREEN = colors.HexColor("#28B978")
PINK = colors.HexColor("#52765B")
WHITE = colors.HexColor("#FCFEFC")


def _font_paths() -> tuple[Path, Path]:
    candidates = (
        (
            Path("/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf"),
        ),
        (
            Path("/usr/share/fonts/dejavu/DejaVuSansMono.ttf"),
            Path("/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf"),
        ),
    )
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            return regular, bold
    raise FileNotFoundError("No Unicode TTF font with Vietnamese glyphs was found")


REGULAR_PATH, BOLD_PATH = _font_paths()
pdfmetrics.registerFont(TTFont("ReportRegular", str(REGULAR_PATH)))
pdfmetrics.registerFont(TTFont("ReportBold", str(BOLD_PATH)))


def wrap_lines(text: str, font: str, size: float, width: float) -> list[str]:
    result: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        words = paragraph.split()
        if not words:
            result.append("")
            continue
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if line and pdfmetrics.stringWidth(candidate, font, size) > width:
                result.append(line)
                line = word
            else:
                line = candidate
        if line:
            result.append(line)
    return result


def draw_text(
    c: Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    *,
    size: float = 8.5,
    leading: float | None = None,
    font: str = "ReportRegular",
    color: colors.Color = INK,
    max_lines: int | None = None,
) -> float:
    leading = leading or size * 1.38
    c.setFont(font, size)
    c.setFillColor(color)
    lines = wrap_lines(text, font, size, width)
    if max_lines is not None:
        lines = lines[:max_lines]
    for line in lines:
        if line:
            c.drawString(x, y, line)
        y -= leading
    return y


def draw_centered_text(
    c: Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    size: float = 8.5,
    leading: float | None = None,
    font: str = "ReportRegular",
    color: colors.Color = INK,
    max_lines: int | None = None,
) -> None:
    """Draw a wrapped text block centered horizontally and vertically."""
    leading = leading or size * 1.38
    c.setFont(font, size)
    c.setFillColor(color)
    lines = wrap_lines(text, font, size, width)
    if max_lines is not None:
        lines = lines[:max_lines]
    if not lines:
        return
    baseline = y + height / 2 + (len(lines) - 1) * leading / 2 - size * 0.35
    for line in lines:
        if line:
            c.drawCentredString(x + width / 2, baseline, line)
        baseline -= leading


def draw_vertical_centered_text(
    c: Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    size: float = 8.5,
    leading: float | None = None,
    font: str = "ReportRegular",
    color: colors.Color = INK,
    max_lines: int | None = None,
) -> None:
    """Draw wrapped text left-aligned and centered vertically in a box."""
    leading = leading or size * 1.38
    c.setFont(font, size)
    c.setFillColor(color)
    lines = wrap_lines(text, font, size, width)
    if max_lines is not None:
        lines = lines[:max_lines]
    if not lines:
        return
    baseline = y + height / 2 + (len(lines) - 1) * leading / 2 - size * 0.35
    for line in lines:
        if line:
            c.drawString(x, baseline, line)
        baseline -= leading


def draw_label(c: Canvas, text: str, x: float, y: float, color: colors.Color = CYAN) -> None:
    c.setFont("ReportBold", 7.2)
    c.setFillColor(color)
    c.drawString(x, y, text.upper())


def page_base(c: Canvas, page_no: int, section: str = "INGEST / OCR") -> None:
    c.setFillColor(BG)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.rect(0, 0, 7, H, fill=1, stroke=0)
    c.setStrokeColor(LINE)
    c.setLineWidth(0.6)
    c.line(MARGIN, 31, W - MARGIN, 31)
    c.setFont("ReportRegular", 6.5)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, 18, f"NeMo-Retriever  ·  {section}")
    c.drawRightString(W - MARGIN, 18, f"Pipeline Qwen & Mistral  ·  {page_no:02d}")


def pale_circle(c: Canvas, x: float, y: float, radius: float, color: colors.Color) -> None:
    c.saveState()
    try:
        c.setFillAlpha(0.28)
    except AttributeError:
        pass
    c.setFillColor(color)
    c.circle(x, y, radius, fill=1, stroke=0)
    c.restoreState()


def card(
    c: Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    accent: colors.Color = CYAN,
    *,
    title_size: float = 9.2,
    body_size: float = 7.5,
    body_leading: float | None = None,
) -> None:
    c.setFillColor(WHITE)
    c.setStrokeColor(LINE)
    c.setLineWidth(0.7)
    c.roundRect(x, y, width, height, 7, fill=1, stroke=1)
    c.setFillColor(accent)
    c.roundRect(x, y + height - 4, width, 4, 2, fill=1, stroke=0)
    c.setFont("ReportBold", title_size)
    c.setFillColor(INK)
    c.drawString(x + 12, y + height - 23, title)
    draw_text(
        c,
        body,
        x + 12,
        y + height - 39,
        width - 24,
        size=body_size,
        leading=body_leading,
    )


def pill(c: Canvas, x: float, y: float, width: float, text: str, accent: colors.Color) -> None:
    c.setFillColor(colors.Color(accent.red, accent.green, accent.blue, alpha=0.12))
    c.setStrokeColor(accent)
    c.setLineWidth(0.7)
    c.roundRect(x, y, width, 20, 10, fill=1, stroke=1)
    c.setFont("ReportBold", 7.2)
    c.setFillColor(accent)
    c.drawCentredString(x + width / 2, y + 6.5, text)


def flow(
    c: Canvas,
    items: Iterable[tuple[str, colors.Color]],
    x: float,
    y: float,
    width: float = CONTENT_W,
    height: float = 46,
    font_size: float = 6.8,
) -> None:
    items = list(items)
    gap = 9
    box_w = (width - gap * (len(items) - 1)) / len(items)
    for index, (label, accent) in enumerate(items):
        bx = x + index * (box_w + gap)
        c.setFillColor(WHITE)
        c.setStrokeColor(accent)
        c.setLineWidth(1.0)
        c.roundRect(bx, y, box_w, height, 6, fill=1, stroke=1)
        c.setFillColor(accent)
        accent_w = min(22.0, box_w * 0.24)
        c.roundRect(
            bx + (box_w - accent_w) / 2,
            y + height - 7,
            accent_w,
            2.4,
            1.2,
            fill=1,
            stroke=0,
        )
        draw_centered_text(
            c,
            label,
            bx + 8,
            y,
            box_w - 16,
            height - 11,
            size=font_size,
            leading=font_size * 1.25,
            font="ReportBold",
        )
        if index < len(items) - 1:
            c.setStrokeColor(MUTED)
            c.setLineWidth(0.8)
            start = bx + box_w + 2
            end = start + gap - 4
            ay = y + height / 2
            c.line(start, ay, end, ay)
            c.line(end - 3, ay + 2.5, end, ay)
            c.line(end - 3, ay - 2.5, end, ay)


def section_header(c: Canvas, kicker: str, title: str, subtitle: str = "") -> None:
    draw_label(c, kicker, MARGIN, H - 54, CYAN)
    c.setFont("ReportBold", 21)
    c.setFillColor(NAVY)
    c.drawString(MARGIN, H - 86, title)
    if subtitle:
        draw_text(c, subtitle, MARGIN, H - 105, CONTENT_W, size=8.2, leading=11, color=MUTED)


def draw_cover(c: Canvas) -> None:
    page_base(c, 1, "PIPELINE STUDY")
    pale_circle(c, W - 82, H - 70, 74, colors.HexColor("#D8EEDF"))
    pale_circle(c, W - 84, 180, 61, colors.HexColor("#E3F0E5"))
    pale_circle(c, 180, 95, 46, colors.HexColor("#D3EBDD"))
    draw_label(c, "NeMo-Retriever · PDF ingest", MARGIN, H - 67, CYAN)
    c.setFont("ReportBold", 25)
    c.setFillColor(NAVY)
    c.drawString(MARGIN, H - 145, "BÁO CÁO")
    c.drawString(MARGIN, H - 177, "TRIỂN KHAI")
    c.drawString(MARGIN, H - 209, "NEMO-RETRIEVER")
    c.drawString(MARGIN, H - 241, "CHO XỬ LÝ DỮ LIỆU")
    c.setStrokeColor(CYAN)
    c.setLineWidth(2.2)
    c.line(MARGIN, H - 273, MARGIN + 82, H - 273)
    draw_text(
        c,
        "Ghi lại kiến trúc, cấu hình, số đo hiện có và tình hình khi thử nghiệm hai model Qwen và Mistral.",
        MARGIN,
        H - 310,
        255,
        size=9.8,
        leading=15,
        color=INK,
    )
    card(c, MARGIN, 390, 234, 100, "Pipeline Qwen", "Page Elements detect → native PDFium hoặc scan full-page → Qwen3.5-2B NVFP4\n\nGiữ text native, bbox, Markdown table và visual crop hợp lệ.", PURPLE, body_size=8.2)
    card(c, MARGIN, 272, 234, 100, "Pipeline Mistral", "Page Elements semantic crop → Ministral 3 3B FP8\n\nText/title/table/visual gửi theo crop; scan/layout yếu có full-page fallback.", ORANGE, body_size=8.2)
    draw_label(c, "Phạm vi thử nghiệm", MARGIN, 213, BLUE)
    draw_text(c, "Thử nghiệm hai model Qwen và Mistral trên hai pipeline OCR.", MARGIN, 194, 330, size=8.3, leading=12)
    pill(c, W - 194, 100, 152, "QWEN3.5 · NVFP4", PURPLE)
    pill(c, W - 194, 72, 152, "MINISTRAL · FP8", ORANGE)
    c.setFont("ReportRegular", 7)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, 57, "Bản báo cáo: 13.08.2026  ·  Embedding không nằm trong phép đo parse/OCR")


def draw_toc(c: Canvas) -> None:
    page_base(c, 2, "CONTENTS")
    section_header(c, "Mục lục", "Hai pipeline thử nghiệm", "Các phần dưới đây ghi lại luồng xử lý, cấu hình và kết quả thử nghiệm hiện có.")
    entries = [
        ("01", "Kiến trúc tổng quan", "Luồng xử lý chung của hai pipeline OCR.", "03", BLUE),
        ("02", "Pipeline Qwen · NVFP4", "Kiến trúc dùng Qwen NVFP4 làm core OCR.", "04", PURPLE),
        ("03", "Pipeline Mistral · FP8", "Kiến trúc dùng Ministral FP8 làm core OCR.", "05", ORANGE),
        ("04", "Kết quả thử nghiệm", "Benchmark, latency, chất lượng nội dung và vấn đề còn sót.", "06", GREEN),
    ]
    y = 595
    for number, title, body, page, accent in entries:
        c.setFillColor(WHITE)
        c.setStrokeColor(LINE)
        c.roundRect(MARGIN, y, CONTENT_W, 64, 8, fill=1, stroke=1)
        c.setFillColor(accent)
        c.circle(MARGIN + 24, y + 32, 11, fill=1, stroke=0)
        c.setFont("ReportBold", 7)
        c.setFillColor(WHITE)
        c.drawCentredString(MARGIN + 24, y + 29.5, number)
        c.setFont("ReportBold", 10.2)
        c.setFillColor(INK)
        c.drawString(MARGIN + 48, y + 39, title)
        draw_text(c, body, MARGIN + 48, y + 22, CONTENT_W - 105, size=7.5, leading=10, color=MUTED)
        c.setFont("ReportBold", 13)
        c.setFillColor(accent)
        c.drawRightString(W - MARGIN - 18, y + 29, page)
        y -= 82


def draw_overview(c: Canvas) -> None:
    page_base(c, 3, "ARCHITECTURE")
    section_header(c, "01 / Kiến trúc tổng quan", "Hai nhánh sau cùng một PDFium", "Các stage chung được giữ lại, khác biệt chính nằm ở ảnh đưa vào VLM.")
    flow(c, [("PDF / image", BLUE), ("DocToPdf + split", CYAN), ("PDFium", GREEN), ("Page Elements", PURPLE), ("OCR runner", ORANGE)], MARGIN, 612, height=49, font_size=6.8)
    c.setFont("ReportBold", 9)
    c.setFillColor(INK)
    c.drawString(MARGIN, 575, "Sau bước detection")
    card(c, MARGIN, 367, 246, 184, "Pipeline Qwen", "Scan nguyên trang, native theo phần thiếu text\n\n- scan: Qwen đọc nguyên trang\n- native: PDFium giữ text chính\n- native: box thiếu geometry mới gọi VLM\n- table: crop riêng, Markdown\n- visual: crop hợp lệ, nhãn ngắn\n\nBbox gần toàn trang chứa nhiều text bị loại, native 4 trang: 0,59s.", PURPLE, body_size=8.0, body_leading=11.5)
    card(c, MARGIN + 265, 367, 246, 184, "Pipeline Mistral", "Đọc theo trang\n\n- Page Elements tạo vùng cần đọc\n- gửi semantic crop OCR\n- một raster / trang\n- một text block / trang\n- bbox theo vùng hoặc toàn trang\n\nMinistral đọc crop OCR; scan/layout yếu dùng full-page fallback.", ORANGE, body_size=8.1, body_leading=12)


def draw_pipeline6(c: Canvas) -> None:
    page_base(c, 4, "PIPELINE QWEN")
    section_header(c, "02 / Pipeline QWEN", "Page Detect → Qwen 3.5 VLM", "Chuẩn bị trang song song và dồn liên tục các phần đã sẵn sàng vào Qwen NVFP4.")
    flow(c, [("PDFium\ntext + raster", BLUE), ("Detect\nbatch 128", CYAN), ("Crop\nbatch 128", GREEN), ("Queue\n2 blocks", PURPLE), ("Qwen\nCCR 8", ORANGE)], MARGIN, 610, height=55, font_size=6.5)
    card(c, MARGIN, 422, 246, 155, "Nhánh scan", "PDF extract batch 16 × 4 worker. Trang nào render/detect/crop xong được đưa vào queue ngay. Consumer gửi text/table/visual tasks theo block 16, sau VLM ghép lại theo thứ tự trang.", PURPLE, body_size=8.1, body_leading=12)
    card(c, MARGIN + 265, 422, 246, 155, "Nhánh native", "PDFium native text là nguồn chính. Page Elements vẫn tìm table/visual. Text thiếu character geometry mới gọi VLM, native table hiện dùng image crop vì đường native text → Markdown đang tắt trong quality gate.", CYAN, body_size=8.1, body_leading=12)
    card(c, MARGIN, 245, 246, 142, "Prompt và visual gate", "Qwen có nhiều profile: legacy, strict, char_repair và word_repair. Đang dùng word_repair, temperature 0, thinking tắt. Visual prompt chỉ trả nhãn ngắn hoặc BỎ QUA; crop gần nguyên trang có nhiều text bị loại.", GREEN, body_size=7.8, body_leading=10.8)
    card(c, MARGIN + 265, 245, 246, 142, "Số đo đã ghi nhận", "Scan 40 trang: 32,69s, 40/40 đúng thứ tự, 41 VLM requests, khoảng 1.097 token/s aggregate, Qwen + detect khoảng 15,1GB. Native report 6 trang: 1,87s, 3 bbox visual bị loại.", ORANGE, body_size=8.0, body_leading=11.2)
    pill(c, MARGIN, 191, 142, "32,69s / 40 TRANG", PURPLE)
    pill(c, MARGIN + 154, 191, 142, "CCR 8 · BATCH 8", BLUE)
    pill(c, MARGIN + 308, 191, 142, "DETECT 128", GREEN)


def draw_pipeline7(c: Canvas) -> None:
    page_base(c, 5, "PIPELINE MISTRAL")
    section_header(c, "03 / Pipeline MISTRAL", "Semantic crop → Ministral VLM", "Page Elements tạo bbox semantic; Ministral đọc crop và fallback full-page khi scan/layout yếu.")
    flow(c, [("PDFium\nrender/extract", BLUE), ("Page Elements\nsemantic bbox", CYAN), ("Semantic\ncrops", ORANGE), ("Ministral\nFP8", PURPLE), ("OCR output", GREEN)], MARGIN, 610, height=55, font_size=6.4)
    card(c, MARGIN, 414, 246, 166, "Các stage đang bỏ qua", "Không bật Table Structure, line detector hoặc language probe. Semantic OCR crop được bật; Page Elements tạo text/title/table/visual bbox để quyết định crop gửi VLM.", ORANGE, body_size=8.1, body_leading=12)
    card(c, MARGIN + 265, 414, 246, 166, "Cấu hình hiện tại", "Ministral-3-3B-Instruct-2512 · max model len 8192 · max seq 10 · max batched tokens 4096 · GPU utilization 0,33 · VLM batch/workers 10 · max output 1024.", PURPLE, body_size=8.1, body_leading=12)
    card(c, MARGIN, 232, CONTENT_W, 143, "Output hiện tại", "Native text có character geometry được PDFium giữ làm nguồn chính. Semantic text/title/table/visual crop được gửi cho Ministral; table trả whole-table Markdown. Scan/layout yếu có full-page fallback bbox [0,0,1,1]. Reading order semantic block được giữ trước khi ghép theo page order.", CYAN, body_size=8.6, body_leading=13)
    pill(c, MARGIN, 181, 159, "PAGE-LEVEL BBOX", ORANGE)
    pill(c, MARGIN + 171, 181, 159, "NO TABLE STRUCTURE", PURPLE)
    pill(c, MARGIN + 342, 181, 169, "SPEED-FIRST", GREEN)
    draw_text(c, "Pipeline Mistral có log scan 40 trang, native 4 trang và report scan 6 trang. Trường nào log không lưu được được ghi rõ ở trang kết quả.", MARGIN, 147, CONTENT_W, size=7.2, leading=10, color=MUTED)


def draw_comparison(c: Canvas) -> None:
    page_base(c, 6, "EXPERIMENT")
    section_header(c, "04 / Kết quả thử nghiệm", "Kết quả thử nghiệm", "Giữ cùng khuôn đo Qwen. Các chỉ số Pipeline 7 không được log lưu lại được đánh dấu rõ, không nội suy.")
    x = MARGIN
    y_top = 625
    col_w = [145, 210, 156]
    row_h = 43
    rows = [
        ("Qwen · scan 40 trang", "32,69s · 40/40 đúng thứ tự · request đầu sau 2,38s", "41 requests · khoảng 1.097 token/s aggregate"),
        ("Mistral · scan 40 trang", "53,89s · file 40 trang · 41 rows", "Thứ tự/request đầu chưa được log lưu"),
        ("Qwen · VRAM", "Khoảng 15,1GB tổng: Qwen + Page Elements", "Qwen khoảng 10GB · detect khoảng 5GB"),
        ("Mistral · VRAM", "4,44GiB model load + 10,17GiB KV cache", "Page Elements startup 470MiB · không có peak per-job"),
        ("Qwen · native 4 trang", "0,59s · 4/4 native pages", "1 table request · errors=[]"),
        ("Mistral · native 4 trang", "1,05s · 4 trang · 6 rows", "8 requests: 1 text + 5 table + 2 visual"),
        ("Qwen · native report 6 trang", "1,87s · 3 bbox gần toàn trang bị loại", "Không lỗi · sidecar không còn visual giả"),
        ("Mistral · report scan 6 trang", "11,13s · 7 rows", "7 requests: 6 full-page + 1 table · không lỗi"),
        ("Qwen · chất lượng", "word_repair similarity 0,9991 · exact 2/4", "Còn lặp một từ ở trường hợp khó"),
        ("Mistral · chất lượng", "Regression 4 rows trong 4,66s", "Similarity/ground truth chưa được persist"),
    ]
    header_h = 30
    c.setFillColor(NAVY)
    c.roundRect(x, y_top - header_h, sum(col_w), header_h, 5, fill=1, stroke=0)
    headers = ["Hạng mục", "Kết quả đo", "Ghi chú"]
    xx = x
    for i, header in enumerate(headers):
        # Center only the header labels horizontally.  Body cells stay
        # left-aligned so long benchmark notes remain easy to scan.
        draw_centered_text(
            c,
            header,
            xx,
            y_top - header_h,
            col_w[i],
            header_h,
            size=8,
            leading=9.5,
            font="ReportBold",
            color=WHITE,
        )
        xx += col_w[i]
    y = y_top - header_h
    for index, row in enumerate(rows):
        fill = WHITE if index % 2 == 0 else colors.HexColor("#E7F1E9")
        c.setFillColor(fill)
        c.rect(x, y - row_h, sum(col_w), row_h, fill=1, stroke=0)
        xx = x
        for col_index, value in enumerate(row):
            draw_vertical_centered_text(
                c,
                value,
                xx + 8,
                y - row_h,
                col_w[col_index] - 16,
                row_h,
                size=7.2,
                leading=9.2,
                font="ReportBold" if col_index == 0 else "ReportRegular",
                color=INK,
            )
            xx += col_w[col_index]
        y -= row_h

    table_width = sum(col_w)
    table_bottom = y_top - header_h - row_h * len(rows)
    table_body_top = y_top - header_h
    c.saveState()
    c.setStrokeColor(TABLE_LINE)
    c.setLineWidth(0.65)
    # Outer frame and horizontal rules keep the table legible at normal zoom.
    c.roundRect(x, table_bottom, table_width, y_top - table_bottom, 5, fill=0, stroke=1)
    c.line(x, table_body_top, x + table_width, table_body_top)
    for row_index in range(1, len(rows) + 1):
        rule_y = table_body_top - row_index * row_h
        c.line(x, rule_y, x + table_width, rule_y)
    # Vertical rules are drawn through both the header and body.
    xx = x
    for col_width in col_w[:-1]:
        xx += col_width
        c.line(xx, table_bottom, xx, y_top)
    c.restoreState()
    card(c, MARGIN, 52, 246, 88, "Các mốc benchmark đã có", "Cả hai đều có mốc scan 40 trang và native 4 trang. Mistral có thêm report scan 6 trang; không chạy lại Qwen.", PURPLE, body_size=7.2, body_leading=9.2)
    card(c, MARGIN + 265, 52, 246, 88, "Các vấn đề còn sót", "Pipeline 7 chưa lưu per-job: request đầu, tổng requests và similarity. Không thể suy ra lại các số này từ log cũ.", ORANGE, body_size=7.2, body_leading=9.2)
    draw_text(c, "Latency và chất lượng chỉ phản ánh các file đã thử; embedding nằm ngoài phép đo parse/OCR.", MARGIN, 38, CONTENT_W, size=6.4, leading=8, color=MUTED)


def build(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    c = Canvas(str(output), pagesize=A4, pageCompression=1)
    c.setTitle("Báo cáo Pipeline Qwen và Pipeline Mistral - NeMo-Retriever")
    c.setAuthor("NeMo-Retriever development workspace")
    for draw_page in (draw_cover, draw_toc, draw_overview, draw_pipeline6, draw_pipeline7, draw_comparison):
        draw_page(c)
        c.showPage()
    c.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
