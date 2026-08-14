"""Create a small, reproducible Vietnamese OCR benchmark corpus.

The native PDF has a real PDF text layer.  The OCR PDF is rendered from the
same pages and embedded as raster images, so both corpora share exact ground
truth while exercising different document inputs.  The benchmark itself uses
the same line boxes for every recognizer; detector quality is therefore not
mixed into the recognizer comparison.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageEnhance, ImageOps
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


PAGE_WIDTH = 595.2756  # A4, points
PAGE_HEIGHT = 841.8898
MARGIN_X = 48.0
TOP_Y = PAGE_HEIGHT - 58.0
LINE_GAP = 38.0
FONT_PATH = "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf"
FONT_NAME = "BenchmarkDejaVuSans"


PAGE_LINES: list[list[str]] = [
    [
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
        "Độc lập - Tự do - Hạnh phúc",
        "HỢP ĐỒNG THUÊ PHÒNG",
        "Số: 06/2026/HĐTP",
        "Ngày ký: 06 tháng 08 năm 2026",
        "Bên cho thuê: Nguyễn Thị Minh Anh, số điện thoại 0903 123 456.",
        "Bên thuê: Trần Quốc Huy, căn cước công dân số 079203001234.",
        "Địa chỉ phòng thuê: số 302, tầng 3, đường Lê Lợi, Quận 1, Thành phố Hồ Chí Minh.",
        "Hai bên thống nhất ký hợp đồng với các điều khoản dưới đây.",
        "Điều 1. Đối tượng và thời hạn thuê",
        "Bên A cho Bên B thuê căn phòng số 302 tại tầng 3 của tòa nhà.",
        "Thời hạn thuê là 12 tháng, kể từ ngày 01/06/2026 đến hết ngày 31/05/2027.",
        "Mục đích sử dụng là để ở; Bên B không được sử dụng phòng cho hoạt động trái pháp luật.",
        "Mọi sửa đổi hợp đồng phải được lập thành văn bản và có chữ ký của hai bên.",
    ],
    [
        "Điều 2. Tiền thuê, tiền đặt cọc và chi phí",
        "Tiền thuê hàng tháng là 6.500.000 đồng, chưa bao gồm điện, nước và Internet.",
        "Bên B thanh toán tiền thuê trước ngày mùng 05 của mỗi tháng.",
        "Tiền đặt cọc tương đương hai tháng tiền thuê, tức 13.000.000 đồng.",
        "Phí điện được tính theo chỉ số công tơ và đơn giá 3.500 đồng/kWh.",
        "Phí nước là 25.000 đồng/m³; phí Internet cố định là 180.000 đồng/tháng.",
        "Khoản thanh toán chuyển khoản vào tài khoản 0123 4567 890 tại ngân hàng ABC.",
        "Nội dung chuyển khoản: HỌ TÊN - PHÒNG 302 - THÁNG THANH TOÁN.",
        "Nếu thanh toán trễ quá 05 ngày, Bên B phải chịu phí chậm trả 0,05% mỗi ngày.",
        "Bảng kê chi phí phải được gửi cho Bên B trước ngày 03 hàng tháng.",
        "Mọi khoản tiền nêu trong hợp đồng được tính bằng đồng Việt Nam (VND).",
        "Bên A hoàn trả tiền đặt cọc trong vòng 07 ngày kể từ khi kết thúc hợp đồng.",
        "Khoản khấu trừ, nếu có, phải kèm theo hình ảnh hoặc biên bản xác nhận.",
        "Hai bên có trách nhiệm lưu giữ chứng từ thanh toán trong suốt thời hạn hợp đồng.",
    ],
    [
        "Điều 3. Quyền và nghĩa vụ của Bên A",
        "Bên A bảo đảm quyền sử dụng ổn định, riêng tư và hợp pháp cho Bên B.",
        "Bên A sửa chữa các hư hỏng thuộc kết cấu và hệ thống kỹ thuật của tòa nhà.",
        "Bên A thông báo trước ít nhất 24 giờ trước khi kiểm tra phòng, trừ trường hợp khẩn cấp.",
        "Bên A cung cấp biên nhận hoặc xác nhận điện tử cho mỗi khoản thanh toán.",
        "Bên A không được tự ý tăng tiền thuê trong thời hạn đã thỏa thuận.",
        "Điều 4. Quyền và nghĩa vụ của Bên B",
        "Bên B giữ gìn tài sản, thiết bị và bàn giao phòng đúng hiện trạng khi trả phòng.",
        "Bên B không được tự ý chuyển nhượng, cho thuê lại hoặc thay đổi kết cấu phòng.",
        "Bên B tuân thủ nội quy tòa nhà, quy định về phòng cháy chữa cháy và an ninh.",
        "Không được lưu trữ chất dễ cháy, chất độc hại hoặc hàng hóa bị pháp luật cấm.",
        "Bên B chịu trách nhiệm đối với thiệt hại do lỗi của mình hoặc của khách đến thăm.",
        "Mọi khiếu nại về dịch vụ phải được gửi bằng văn bản trong vòng 03 ngày.",
        "Bên B được đăng ký tạm trú theo quy định của cơ quan quản lý địa phương.",
    ],
    [
        "Điều 5. Chấm dứt hợp đồng và giải quyết tranh chấp",
        "Thông báo chấm dứt hợp đồng phải được gửi trước ít nhất 30 ngày.",
        "Một bên có quyền chấm dứt ngay nếu bên kia vi phạm nghiêm trọng nghĩa vụ.",
        "Khi trả phòng, hai bên lập biên bản bàn giao và đối chiếu công tơ điện, nước.",
        "Tranh chấp trước hết được giải quyết bằng thương lượng và thiện chí hợp tác.",
        "Nếu thương lượng không thành, tranh chấp được giải quyết tại Tòa án có thẩm quyền.",
        "Hợp đồng được lập thành 02 bản có giá trị pháp lý như nhau, mỗi bên giữ 01 bản.",
        "Thông tin liên hệ khẩn cấp: 028 7300 0099 hoặc email quanly@example.vn.",
        "Bản scan, bản điện tử và bản giấy có nội dung giống nhau được dùng để đối chiếu.",
        "Các phụ lục kèm theo là bộ phận không tách rời của hợp đồng này.",
        "ĐẠI DIỆN BÊN A                         ĐẠI DIỆN BÊN B",
        "Nguyễn Thị Minh Anh                    Trần Quốc Huy",
        "Ngày ký: 06/08/2026                    Ngày ký: 06/08/2026",
        "Xác nhận: đã đọc, hiểu và đồng ý toàn bộ nội dung hợp đồng.",
    ],
]


def _register_font() -> None:
    if FONT_NAME not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))


def _line_layout(text: str, line_index: int) -> tuple[float, float, float, float, float]:
    font_size = 15.0 if line_index < 3 else 11.5
    if line_index in {9, 15}:
        font_size = 12.5
    max_width = PAGE_WIDTH - 2 * MARGIN_X
    while font_size > 8.5 and pdfmetrics.stringWidth(text, FONT_NAME, font_size) > max_width:
        font_size -= 0.25
    width = pdfmetrics.stringWidth(text, FONT_NAME, font_size)
    baseline = TOP_Y - line_index * LINE_GAP
    # A conservative glyph box with a little vertical margin.  Crops are
    # padded again when materialized, so descenders are never clipped.
    x0 = MARGIN_X
    y0 = baseline - font_size * 0.28
    x1 = min(PAGE_WIDTH - MARGIN_X, x0 + width)
    y1 = baseline + font_size * 0.90
    return x0, y0, x1, y1, font_size


def _write_native_pdf(path: Path) -> list[dict[str, Any]]:
    _register_font()
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=(PAGE_WIDTH, PAGE_HEIGHT), pageCompression=1)
    annotations: list[dict[str, Any]] = []
    for page_number, lines in enumerate(PAGE_LINES, start=1):
        for line_index, text in enumerate(lines):
            x0, y0, x1, y1, font_size = _line_layout(text, line_index)
            pdf.setFont(FONT_NAME, font_size)
            pdf.drawString(x0, y0, text)
            annotations.append(
                {
                    "id": f"native-p{page_number:02d}-l{line_index:02d}",
                    "corpus": "native",
                    "page": page_number,
                    "text": text,
                    "bbox_points": [x0, y0, x1, y1],
                    "bbox_norm": [
                        x0 / PAGE_WIDTH,
                        (PAGE_HEIGHT - y1) / PAGE_HEIGHT,
                        x1 / PAGE_WIDTH,
                        (PAGE_HEIGHT - y0) / PAGE_HEIGHT,
                    ],
                    "font_size": font_size,
                }
            )
        pdf.showPage()
    pdf.save()
    return annotations


def _render_pages(path: Path, dpi: int = 300) -> list[Image.Image]:
    document = pdfium.PdfDocument(str(path))
    return [document[index].render(scale=dpi / 72.0).to_pil().convert("RGB") for index in range(len(document))]


def _make_scan_image(image: Image.Image, page_number: int) -> bytes:
    # Keep geometry unchanged so the native and OCR corpora share the same
    # manifest boxes.  The degradation is intentionally moderate: grayscale,
    # contrast variation, very light sensor noise, and JPEG artifacts.
    gray = ImageOps.grayscale(image).convert("RGB")
    gray = ImageEnhance.Contrast(gray).enhance(0.92 if page_number % 2 else 1.04)
    array = np.asarray(gray).astype(np.float32)
    rng = np.random.default_rng(20260806 + page_number)
    array += rng.normal(0.0, 1.3, size=array.shape[:2])[..., None]
    array = np.clip(array, 0, 255).astype(np.uint8)
    result = Image.fromarray(array, mode="RGB")
    output = io.BytesIO()
    result.save(output, format="JPEG", quality=78, optimize=True)
    return output.getvalue()


def _write_ocr_pdf(native_pdf: Path, path: Path) -> None:
    pages = _render_pages(native_pdf)
    pdf = canvas.Canvas(str(path), pagesize=(PAGE_WIDTH, PAGE_HEIGHT), pageCompression=1)
    for page_number, image in enumerate(pages, start=1):
        encoded = _make_scan_image(image, page_number)
        pdf.drawImage(ImageReader(io.BytesIO(encoded)), 0, 0, width=PAGE_WIDTH, height=PAGE_HEIGHT, preserveAspectRatio=True, mask="auto")
        pdf.showPage()
    pdf.save()


def _materialize_crops(root: Path, annotations: list[dict[str, Any]], pdf_path: Path, corpus: str) -> None:
    pages = _render_pages(pdf_path)
    crop_root = root / "crops" / corpus
    crop_root.mkdir(parents=True, exist_ok=True)
    for item in annotations:
        page = pages[int(item["page"]) - 1]
        width, height = page.size
        x0, y0, x1, y1 = item["bbox_norm"]
        pad_x, pad_y = 0.010, 0.014
        left = max(0, int(round((x0 - pad_x) * width)))
        top = max(0, int(round((y0 - pad_y) * height)))
        right = min(width, int(round((x1 + pad_x) * width)))
        bottom = min(height, int(round((y1 + pad_y) * height)))
        crop_path = crop_root / f"{item['id']}.png"
        page.crop((left, top, max(left + 2, right), max(top + 2, bottom))).save(crop_path)
        item["image"] = str(crop_path.relative_to(root))
        item["image_size"] = [right - left, bottom - top]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("cache/ocr-benchmark/corpus"))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)

    native_pdf = root / "vietnamese_native.pdf"
    ocr_pdf = root / "vietnamese_ocr.pdf"
    annotations = _write_native_pdf(native_pdf)
    _write_ocr_pdf(native_pdf, ocr_pdf)
    native_items = [dict(item) for item in annotations]
    ocr_items = []
    for item in annotations:
        copy = dict(item)
        copy["id"] = item["id"].replace("native", "ocr")
        copy["corpus"] = "ocr"
        ocr_items.append(copy)
    _materialize_crops(root, native_items, native_pdf, "native")
    _materialize_crops(root, ocr_items, ocr_pdf, "ocr")
    manifest = {
        "version": 1,
        "dpi": 300,
        "native_pdf": str(native_pdf.relative_to(root)),
        "ocr_pdf": str(ocr_pdf.relative_to(root)),
        "samples": native_items + ocr_items,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"root": str(root), "native_samples": len(native_items), "ocr_samples": len(ocr_items), "native_pdf": str(native_pdf), "ocr_pdf": str(ocr_pdf)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
