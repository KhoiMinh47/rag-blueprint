# SPDX-License-Identifier: Apache-2.0

"""Pipeline 6: Page Elements crops -> Qwen 3.5 VLM.

This branch is intentionally small and isolated from the older OCR options:

* Page Elements is the only layout service.  The production/default route keeps
  the previously validated behavior: every detected table is sent as one
  image crop, including tables on native PDFs.  An opt-in PDFium-text route is
  retained behind ``OPTION6_NATIVE_TABLE_TEXT_ENABLED`` for a future quality
  gate; scan tables always remain image crops.  The Table Structure NIM is not
  required.
* Text crops and table crops are submitted in large logical batches, while the
  OpenAI-compatible VLM adapter keeps at most 8 requests in flight.  This
  lets vLLM perform continuous batching without creating an unbounded client
  fan-out.
* Native PDF text remains authoritative.  A native text box is sent to Qwen
  only when PDFium has no character geometry inside that box; tables and
  visual regions still come from Page Elements.
"""

from __future__ import annotations

import time
import os
import re
import json
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

from nemo_retriever.common.modality.ocr.isolated.contracts import (
    OCRPage,
    OCRPageOutput,
    OCRUnit,
    page_value,
)
from nemo_retriever.common.modality.ocr.isolated.geometry import (
    PageImageCropper,
    adaptive_local_text_height,
    bbox_area,
    bbox_center,
    bbox_iou,
    clamp_bbox,
    containment,
    crop_image_b64,
)
from nemo_retriever.common.modality.ocr.isolated.units import (
    TEXT_LABELS,
    VISUAL_LABELS,
    page_element_detections,
)


OPTION6_SELECTOR = "pipeline-option6"
OPTION6_PIPELINE_NAME = "option6_page_detect_qwen35_vlm"
OPTION6_MODEL = str(
    os.getenv("OPTION6_MODEL", "AxionML/Qwen3.5-2B-NVFP4")
).strip() or "AxionML/Qwen3.5-2B-NVFP4"

def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = str(os.getenv(name, "true" if default else "false")).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


# Pipeline 6 tuning is deliberately read from environment variables at
# process startup.  The defaults preserve the current safe single-GPU policy;
# the compose preset exposes every knob so a benchmark only needs an env-file
# change and a service restart.  These constants are imported by the graph
# runtime as well as the isolated coordinator, keeping all stages consistent.
OPTION6_DETECTOR_BATCH_SIZE = _env_int("OPTION6_DETECTOR_BATCH_SIZE", 128)
OPTION6_DETECTOR_MAX_POOL_WORKERS = _env_int("OPTION6_DETECTOR_MAX_POOL_WORKERS", 1)
OPTION6_PAGE_ELEMENTS_WORKERS = _env_int("OPTION6_PAGE_ELEMENTS_WORKERS", 1)
OPTION6_CROP_BATCH_SIZE = _env_int("OPTION6_CROP_BATCH_SIZE", 128)
OPTION6_CROP_MAX_CONCURRENCY = _env_int("OPTION6_CROP_MAX_CONCURRENCY", 4)
OPTION6_CROP_IMAGE_FORMAT = str(os.getenv("OPTION6_CROP_IMAGE_FORMAT", "JPEG")).strip().upper()
if OPTION6_CROP_IMAGE_FORMAT not in {"PNG", "JPEG"}:
    OPTION6_CROP_IMAGE_FORMAT = "JPEG"
OPTION6_CROP_JPEG_QUALITY = _env_int("OPTION6_CROP_JPEG_QUALITY", 95, minimum=1)
OPTION6_CROP_JPEG_QUALITY = min(100, OPTION6_CROP_JPEG_QUALITY)
OPTION6_VLM_BATCH_SIZE = _env_int("OPTION6_VLM_BATCH_SIZE", 8)
OPTION6_VLM_MAX_CONCURRENCY = _env_int("OPTION6_VLM_MAX_CONCURRENCY", 8)
OPTION6_MAX_REQUEST_WORKERS = OPTION6_DETECTOR_MAX_POOL_WORKERS
# Keep a bounded budget for long page text and Markdown tables.  The values
# remain environment-overridable for A/B and benchmark runs.
OPTION6_MAX_OUTPUT_TOKENS_TEXT = _env_int("OPTION6_VLM_MAX_OUTPUT_TOKENS_TEXT", 4096)
OPTION6_MAX_OUTPUT_TOKENS_TABLE = _env_int("OPTION6_VLM_MAX_OUTPUT_TOKENS_TABLE", 4096)
OPTION6_MAX_OUTPUT_TOKENS_VISUAL = _env_int("OPTION6_VLM_MAX_OUTPUT_TOKENS_VISUAL", 32)
OPTION6_MAX_OUTPUT_TOKENS_VISUAL_OCR = _env_int(
    "OPTION6_VLM_MAX_OUTPUT_TOKENS_VISUAL_OCR", 2048
)
OPTION6_NATIVE_TABLE_MIN_CHARS = _env_int("OPTION6_NATIVE_TABLE_MIN_CHARS", 8, minimum=2)
OPTION6_PDF_SPLIT_BATCH_SIZE = _env_int("OPTION6_PDF_SPLIT_BATCH_SIZE", 1)
OPTION6_PDF_EXTRACT_BATCH_SIZE = _env_int("OPTION6_PDF_EXTRACT_BATCH_SIZE", 16)
OPTION6_PDF_EXTRACT_WORKERS = _env_int("OPTION6_PDF_EXTRACT_WORKERS", 4)
OPTION6_PDF_EXTRACT_CPUS = _env_float("OPTION6_PDF_EXTRACT_CPUS", 2.0, minimum=0.1)
# Pipeline 6 alone may consume PDF render/Page Elements output as soon as a
# block is ready.  Sixteen pages keeps two full waves queued behind the
# eight-request vLLM admission ceiling while 4 PDF workers can still prepare
# up to 64 pages concurrently.  Disabling this flag restores the old
# document-global barrier without changing any other OCR selector.
OPTION6_STREAMING_ENABLED = _env_bool("OPTION6_STREAMING_ENABLED", True)
OPTION6_STREAM_BATCH_SIZE = _env_int("OPTION6_STREAM_BATCH_SIZE", 16)
# Bound ready render/detect blocks in host RAM while the eight-request VLM
# consumer drains them. Four active render blocks + two queued blocks + one
# consumer block is at most 112 pages with the default 16-page block size.
OPTION6_STREAM_QUEUE_BLOCKS = _env_int("OPTION6_STREAM_QUEUE_BLOCKS", 2)
OPTION6_SCAN_FULL_PAGE = _env_bool("OPTION6_SCAN_FULL_PAGE", True)
OPTION6_SCAN_MASK_LAYOUT = _env_bool("OPTION6_SCAN_MASK_LAYOUT", True)
OPTION6_VISUAL_VLM = _env_bool("OPTION6_VISUAL_VLM", True)
OPTION6_FULL_PAGE_LAYOUT_FALLBACK = _env_bool(
    "OPTION6_FULL_PAGE_LAYOUT_FALLBACK", True
)
OPTION6_FULL_PAGE_VISUAL_AREA = _env_float(
    "OPTION6_FULL_PAGE_VISUAL_AREA", 0.80, minimum=0.0
)
OPTION6_FULL_PAGE_MAX_DETECTIONS = _env_int(
    "OPTION6_FULL_PAGE_MAX_DETECTIONS", 16
)
OPTION6_FULL_PAGE_MIN_NATIVE_CHARS = _env_int(
    "OPTION6_FULL_PAGE_MIN_NATIVE_CHARS", 32, minimum=0
)
# Real native-PDF A/B tests showed that the previously validated image crop
# preserves table columns more reliably than the PDFium-text serializer with
# this VLM. Keep the serializer available behind an explicit quality-gate
# switch, while the production/default route remains image based.
OPTION6_NATIVE_TABLE_TEXT_ENABLED = _env_bool(
    "OPTION6_NATIVE_TABLE_TEXT_ENABLED", False
)
_OPTION6_PROMPT_PROFILES = frozenset(
    {"legacy", "strict", "char_repair", "word_repair", "guarded_repair"}
)
OPTION6_PROMPT_PROFILE = str(
    os.getenv("OPTION6_PROMPT_PROFILE", "word_repair")
).strip().lower()
if OPTION6_PROMPT_PROFILE not in _OPTION6_PROMPT_PROFILES:
    OPTION6_PROMPT_PROFILE = "word_repair"

# PDFium can expose noncharacters around embedded-font ligatures. They are
# extraction artifacts (for example ``Depen\\ufffedency``), not document text;
# remove only these known sentinels before sending native table text to Qwen.
_PDFIUM_TEXT_ARTIFACTS = frozenset({"\ufffd", "\ufffe", "\uffff"})

# Page Elements can classify a photographed/scanned page (especially a phone
# screenshot) as one near-full-page ``infographic``. Masking that box would
# replace the entire OCR input with white pixels and makes Qwen hallucinate.
# Keep small embedded visuals masked, but leave page-sized visual detections in
# the full-page OCR crop.
OPTION6_SCAN_FULL_PAGE_VISUAL_MIN_AREA = 0.80


# Keep the content faithful to the image while allowing the VLM to repair
# Vietnamese that lost diacritics or picked up isolated compression/pixel
# artifacts.  This is deliberately shared by page OCR, semantic text crops,
# and table cells so the three paths do not normalize text differently.
VIETNAMESE_REPAIR_RULES = (
    "Với từ hoặc cụm từ tiếng Việt bị mất dấu, thiếu nét hoặc bị pixel làm "
    "biến dạng, hãy khôi phục về chính tả và dấu tiếng Việt chuẩn nếu từ đó "
    "xác định được nhờ từ vựng, ngữ pháp hoặc ngữ cảnh, kể cả khi dấu không "
    "còn nhìn rõ (ví dụ 'UU ĐÃI' phải sửa thành 'ƯU ĐÃI'). Không chép nguyên "
    "chuỗi ASCII hay ký tự rác nếu rõ ràng đó chỉ là lỗi ảnh; bỏ các nét/ký "
    "tự nhiễu đơn lẻ không thuộc văn bản. Tuyệt đối không tự đoán tên riêng, "
    "thương hiệu, mã sản phẩm, URL, email, số, ngày tháng, đơn vị hoặc ký hiệu; "
    "những phần này phải giữ nguyên như nhìn thấy. Nếu không đủ căn cứ để sửa, "
    "giữ nguyên phần đọc được thay vì bịa nội dung."
)


LEGACY_TEXT_PROMPT = (
    "Bạn là OCR tiếng Việt tốc độ cao. Chép nguyên văn 100% nội dung chữ "
    "nhìn thấy trong crop; "
    "giữ mọi từ, số, ký hiệu, dấu tiếng Việt, hoa thường, xuống dòng, bullet "
    "và URL. Không dịch, tóm tắt, giải thích, thêm hoặc bớt nội dung. "
    + VIETNAMESE_REPAIR_RULES
    + " "
    "Chỉ trả về text, không code fence và không Markdown. Nếu crop rỗng hoặc "
    "không có chữ rõ ràng vì là sơ đồ, biểu đồ, ảnh hoặc vùng không phải văn "
    "bản, trả về đúng một nhãn ngắn (ví dụ: \"sơ đồ\", \"biểu đồ\", "
    "\"hình ảnh\") hoặc chuỗi rỗng."
)

LEGACY_TABLE_PROMPT = (
    "Đọc toàn bộ bảng trong crop. Chỉ trả về một bảng GitHub Markdown, không "
    "code fence và không giải thích. Giữ nguyên 100% ô, hàng, cột, tiêu đề, "
    "số, ký hiệu và thứ tự; không dịch, tóm tắt hoặc tự bịa giá trị. "
    + VIETNAMESE_REPAIR_RULES
    + " Trong từng ô, không được làm mất giá trị hoặc đổi nghĩa. Nếu crop "
    "không phải bảng rõ ràng, trả về chuỗi rỗng."
)

LEGACY_TABLE_TEXT_PROMPT = (
    "Bạn nhận dữ liệu chữ native do PDFium trích từ một vùng bảng PDF; không có "
    "ảnh đi kèm. Hãy chuyển dữ liệu đó thành đúng một bảng GitHub Markdown. "
    "Mỗi dòng đầu vào là một JSON record; các khóa row, y, x0, x1 chỉ là "
    "metadata bố cục và tuyệt đối không được xuất thành nội dung bảng. Chỉ "
    "chuỗi trong khóa value mới là nội dung thật. Dùng x0/x1 để suy ra cột; "
    "những đoạn có x gần nhau thuộc cùng ô, các đoạn thẳng hàng giữa nhiều "
    "row thuộc cùng cột. Một ô có thể bị PDFium xuống thành nhiều row liên "
    "tiếp; hãy ghép lại vào cùng ô khi hình học và ngữ cảnh cho thấy đó là "
    "phần tiếp theo của cùng một ô. Giữ nguyên 100% giá trị, hàng, cột, tiêu đề, thứ tự, số, ký hiệu và "
    "nội dung ô; không thêm, bớt, dịch, tóm tắt hoặc suy diễn ô còn thiếu. "
    + VIETNAMESE_REPAIR_RULES
    + " Không đưa các từ row, y, x0, x1, value, tọa độ hoặc JSON metadata vào "
    "kết quả. Chỉ trả về Markdown table, không code fence và không giải thích. Nếu "
    "dữ liệu không đủ để xác định một bảng, trả về chuỗi rỗng để hệ thống dùng "
    "ảnh dự phòng.\n\nDỮ LIỆU PDFIUM:\n"
)

LEGACY_SCAN_PAGE_PROMPT = (
    "Bạn là OCR tiếng Việt. Đọc toàn bộ chữ trên trang scan này theo đúng thứ "
    "tự từ trên xuống dưới, "
    "trái sang phải, quét liên tục từ mép trên đến tận mép dưới. Bắt buộc đọc "
    "cả phần sau chữ ký, phần cuối trang, chân trang, dòng Nơi nhận và mọi dòng "
    "chữ nhỏ còn lại; không được dừng sau tiêu đề, nội dung chính hoặc chữ ký "
    "khi bên dưới vẫn còn chữ. Chép nguyên văn 100% chữ, số, dấu tiếng Việt, "
    "ký hiệu, hoa thường và xuống dòng; không dịch, tóm tắt, giải thích hay "
    "lược bỏ nội dung. "
    + VIETNAMESE_REPAIR_RULES
    + " Các vùng bảng, biểu đồ, sơ đồ, ảnh hoặc con dấu nhỏ "
    "được che khỏi crop này và sẽ được xử lý riêng; nếu detector đánh dấu gần "
    "toàn trang là một ảnh/sơ đồ thì vẫn phải đọc nguyên trang đó. Chỉ trả về "
    "text OCR, không code fence và không Markdown."
)


# ``legacy`` is retained for A/B comparison.  The other profiles deliberately
# keep OCR, repair, and output-format rules short and non-conflicting.  The
# repair profiles only permit a local correction when the image evidence and
# context leave one unambiguous reading; they do not authorize paraphrasing.
_STRICT_TEXT_PROMPT = (
    "Chép nguyên văn toàn bộ chữ nhìn thấy trong ảnh theo thứ tự từ trên "
    "xuống dưới, trái sang phải. Giữ nguyên từng từ, số, ký hiệu, dấu tiếng "
    "Việt, hoa thường và xuống dòng. Không sửa, không đoán, không dịch, "
    "không tóm tắt, không thêm hoặc bớt. Nếu không chắc, giữ nguyên phần "
    "nhìn thấy. Không có chữ thì trả về rỗng. Chỉ trả về text."
)

_CHAR_REPAIR_RULE = (
    " Chỉ sửa một ký tự hoặc dấu bị mờ khi các ký tự còn lại xác định chắc "
    "chắn đúng cách đọc; không thay cả từ, không đổi câu. Nếu có hơn một "
    "khả năng, giữ nguyên phần nhìn thấy."
)

_WORD_REPAIR_RULE = (
    " Chỉ khôi phục một ký tự, dấu hoặc từ bị mờ khi phần còn lại và ngữ "
    "cảnh xác định chắc chắn duy nhất một cách đọc; không sửa từ đang nhìn "
    "rõ, không thay câu, không diễn giải, không thêm hoặc bớt. Nếu còn nghi "
    "ngờ, giữ nguyên phần nhìn thấy."
)

_GUARDED_REPAIR_RULE = (
    " Ưu tiên chép đúng phần nhìn thấy. Không viết lại câu, không lặp từ, "
    "không bỏ dòng và không thêm ghi chú. Chỉ khôi phục một ký tự, dấu hoặc "
    "từ bị nhòe khi phần còn lại và ngữ cảnh xác định chắc chắn duy nhất một "
    "cách đọc; không sửa từ đang nhìn rõ. Nếu còn nghi ngờ, giữ nguyên phần "
    "nhìn thấy. Kết thúc ngay sau nội dung cuối cùng của ảnh."
)

_CHAR_REPAIR_TEXT_PROMPT = _STRICT_TEXT_PROMPT + _CHAR_REPAIR_RULE
_WORD_REPAIR_TEXT_PROMPT = _STRICT_TEXT_PROMPT + _WORD_REPAIR_RULE
_GUARDED_REPAIR_TEXT_PROMPT = _STRICT_TEXT_PROMPT + _GUARDED_REPAIR_RULE

_STRICT_SCAN_PAGE_PROMPT = (
    "Đọc toàn bộ trang scan từ trên xuống dưới, trái sang phải, đến tận "
    "cuối trang. Chép nguyên văn chữ, số, ký hiệu, dấu tiếng Việt, hoa "
    "thường và xuống dòng. Không sửa, không đoán, không dịch, không tóm tắt, "
    "không thêm hoặc bớt. Các vùng bảng, ảnh, biểu đồ hoặc sơ đồ được xử lý "
    "riêng; không mô tả chúng. Nếu không có chữ thì trả về rỗng. Chỉ trả về "
    "text."
)
_CHAR_REPAIR_SCAN_PAGE_PROMPT = _STRICT_SCAN_PAGE_PROMPT + _CHAR_REPAIR_RULE
_WORD_REPAIR_SCAN_PAGE_PROMPT = _STRICT_SCAN_PAGE_PROMPT + _WORD_REPAIR_RULE
_GUARDED_REPAIR_SCAN_PAGE_PROMPT = _STRICT_SCAN_PAGE_PROMPT + _GUARDED_REPAIR_RULE

_STRICT_TABLE_PROMPT = (
    "Đọc toàn bộ bảng trong ảnh và chỉ trả về một bảng GitHub Markdown. Giữ "
    "nguyên ô, hàng, cột, tiêu đề, số, ký hiệu và thứ tự. Không sửa, không "
    "đoán, không dịch, không thêm hoặc bớt. Nếu không phải bảng rõ ràng, trả "
    "về rỗng. Không code fence và không giải thích."
)
_CHAR_REPAIR_TABLE_PROMPT = (
    _STRICT_TABLE_PROMPT[:-1]
    + _CHAR_REPAIR_RULE
    + " Không code fence và không giải thích."
)
_WORD_REPAIR_TABLE_PROMPT = (
    _STRICT_TABLE_PROMPT[:-1]
    + _WORD_REPAIR_RULE
    + " Không code fence và không giải thích."
)
_GUARDED_REPAIR_TABLE_PROMPT = (
    _STRICT_TABLE_PROMPT[:-1]
    + _GUARDED_REPAIR_RULE
    + " Không code fence và không giải thích."
)

_TABLE_TEXT_LAYOUT_PROMPT = (
    "Bạn nhận dữ liệu chữ native do PDFium trích từ một vùng bảng PDF. "
    "Chuyển dữ liệu thành đúng một bảng GitHub Markdown. Mỗi dòng là JSON; "
    "row, y, x0, x1 chỉ là metadata bố cục, không được xuất ra. Chỉ value "
    "là nội dung thật. Dùng x0/x1 để suy ra cột và ghép các dòng liên tiếp "
    "cùng ô khi hình học cho thấy chúng thuộc cùng nội dung. Giữ nguyên "
    "giá trị, hàng, cột, tiêu đề, thứ tự, số và ký hiệu; không thêm, bớt, "
    "dịch hoặc suy diễn. "
)
_STRICT_TABLE_TEXT_PROMPT = (
    _TABLE_TEXT_LAYOUT_PROMPT
    + "Nếu không đủ dữ liệu để xác định bảng, trả về rỗng. Chỉ trả về "
    "Markdown table, không code fence và không giải thích.\n\nDỮ LIỆU PDFIUM:\n"
)
_CHAR_REPAIR_TABLE_TEXT_PROMPT = (
    _TABLE_TEXT_LAYOUT_PROMPT
    + _CHAR_REPAIR_RULE
    + " Nếu không đủ dữ liệu để xác định bảng, trả về rỗng. Chỉ trả về "
    "Markdown table, không code fence và không giải thích.\n\nDỮ LIỆU PDFIUM:\n"
)
_WORD_REPAIR_TABLE_TEXT_PROMPT = (
    _TABLE_TEXT_LAYOUT_PROMPT
    + _WORD_REPAIR_RULE
    + " Nếu không đủ dữ liệu để xác định bảng, trả về rỗng. Chỉ trả về "
    "Markdown table, không code fence và không giải thích.\n\nDỮ LIỆU PDFIUM:\n"
)
_GUARDED_REPAIR_TABLE_TEXT_PROMPT = (
    _TABLE_TEXT_LAYOUT_PROMPT
    + _GUARDED_REPAIR_RULE
    + " Nếu không đủ dữ liệu để xác định bảng, trả về rỗng. Chỉ trả về "
    "Markdown table, không code fence và không giải thích.\n\nDỮ LIỆU PDFIUM:\n"
)


def _select_option6_prompts(profile: str) -> tuple[str, str, str, str]:
    if profile == "legacy":
        return (
            LEGACY_TEXT_PROMPT,
            LEGACY_TABLE_PROMPT,
            LEGACY_TABLE_TEXT_PROMPT,
            LEGACY_SCAN_PAGE_PROMPT,
        )
    if profile == "strict":
        return (
            _STRICT_TEXT_PROMPT,
            _STRICT_TABLE_PROMPT,
            _STRICT_TABLE_TEXT_PROMPT,
            _STRICT_SCAN_PAGE_PROMPT,
        )
    if profile == "char_repair":
        return (
            _CHAR_REPAIR_TEXT_PROMPT,
            _CHAR_REPAIR_TABLE_PROMPT,
            _CHAR_REPAIR_TABLE_TEXT_PROMPT,
            _CHAR_REPAIR_SCAN_PAGE_PROMPT,
        )
    if profile == "guarded_repair":
        return (
            _GUARDED_REPAIR_TEXT_PROMPT,
            _GUARDED_REPAIR_TABLE_PROMPT,
            _GUARDED_REPAIR_TABLE_TEXT_PROMPT,
            _GUARDED_REPAIR_SCAN_PAGE_PROMPT,
        )
    return (
        _WORD_REPAIR_TEXT_PROMPT,
        _WORD_REPAIR_TABLE_PROMPT,
        _WORD_REPAIR_TABLE_TEXT_PROMPT,
        _WORD_REPAIR_SCAN_PAGE_PROMPT,
    )


TEXT_PROMPT, TABLE_PROMPT, TABLE_TEXT_PROMPT, SCAN_PAGE_PROMPT = (
    _select_option6_prompts(OPTION6_PROMPT_PROFILE)
)

# A suspicious page is still sent as one page image, not as a collection of
# detector crops.  Visual crops are retained separately, so the page prompt
# may read text inside a chart/diagram without losing the original visual
# evidence. Tables are still routed through the table task and are masked from
# this prompt to avoid producing two competing representations of the same
# table.
FULL_PAGE_PROMPT = (
    "Đọc toàn bộ chữ nhìn thấy trên trang theo thứ tự từ trên xuống dưới, "
    "trái sang phải. Bao gồm chữ nằm trong ảnh, biểu đồ hoặc sơ đồ nếu đọc "
    "được. Giữ nguyên từ, số, ký hiệu, dấu tiếng Việt, hoa thường và xuống "
    "dòng; không dịch, tóm tắt, giải thích, thêm hoặc bớt. Vùng chỉ có hình "
    "mà không có chữ thì bỏ qua, không mô tả dài. Vùng bảng đã được xử lý "
    "riêng thành Markdown, không chép lặp lại bảng trong phần text này. Chỉ "
    "trả về text, không code fence và không ghi chú."
    + _WORD_REPAIR_RULE
)

VISUAL_PROMPT = (
    "Đây là crop của đúng một bbox do detector đề xuất. Hãy đánh giá chính "
    "crop này, không đánh giá một visual chỉ nằm bên trong một trang lớn. "
    "Chỉ trả về đúng một nhãn: hình ảnh, biểu đồ, sơ đồ, con dấu hoặc BỎ QUA. "
    "Chỉ dùng bốn nhãn đầu khi đối tượng visual độc lập chiếm phần chính của "
    "crop. Nếu crop gần toàn trang, là trang tài liệu nhiều chữ, nền, khung "
    "rỗng, nhiễu, hoặc visual chỉ là một phần nhỏ bên trong crop, trả BỎ QUA. "
    "Không chép chữ, không mô tả, không giải thích."
)

VISUAL_OCR_PROMPT = (
    "Đây là crop của một vùng hình ảnh, biểu đồ hoặc infographic do detector "
    "tìm thấy. Hãy OCR toàn bộ chữ nhìn thấy bên trong chính crop này, bao gồm "
    "tiêu đề, bullet, nhãn, số, đơn vị, hotline và chữ nằm trên ảnh. Giữ nguyên "
    "nội dung, dấu tiếng Việt, hoa thường, số, ký hiệu và thứ tự dòng; giữ xuống "
    "dòng khi có thể. Không mô tả hình ảnh, không phân loại, không tóm tắt, không "
    "dịch, không thêm Markdown hay lời giải thích. Nếu crop không có chữ đọc được, "
    "chỉ trả về chuỗi rỗng. Chỉ trả về văn bản OCR."
    + _WORD_REPAIR_RULE
)

_VISUAL_SHORT_LABELS = {
    "image": "hình ảnh",
    "chart": "biểu đồ",
    "infographic": "sơ đồ",
    "stamp": "con dấu",
}


@dataclass(frozen=True)
class Option6Config:
    """Latency and fidelity policy for Pipeline 6."""

    language: str | None = "auto"
    detector_batch_size: int = OPTION6_DETECTOR_BATCH_SIZE
    crop_batch_size: int = OPTION6_CROP_BATCH_SIZE
    crop_max_concurrency: int = OPTION6_CROP_MAX_CONCURRENCY
    crop_image_format: str = OPTION6_CROP_IMAGE_FORMAT
    crop_jpeg_quality: int = OPTION6_CROP_JPEG_QUALITY
    vlm_batch_size: int = OPTION6_VLM_BATCH_SIZE
    text_max_output_tokens: int = OPTION6_MAX_OUTPUT_TOKENS_TEXT
    table_max_output_tokens: int = OPTION6_MAX_OUTPUT_TOKENS_TABLE
    request_timeout_s: float = 120.0
    scan_page_fallback: bool = True
    include_visual_regions: bool = True
    native_table_text: bool = OPTION6_NATIVE_TABLE_TEXT_ENABLED
    scan_full_page: bool = OPTION6_SCAN_FULL_PAGE
    scan_mask_layout: bool = OPTION6_SCAN_MASK_LAYOUT
    classify_visual_regions: bool = OPTION6_VISUAL_VLM
    ocr_visual_regions: bool = False
    visual_ocr_max_tokens: int = OPTION6_MAX_OUTPUT_TOKENS_VISUAL_OCR
    full_page_layout_fallback: bool = OPTION6_FULL_PAGE_LAYOUT_FALLBACK
    full_page_visual_area: float = OPTION6_FULL_PAGE_VISUAL_AREA
    full_page_max_detections: int = OPTION6_FULL_PAGE_MAX_DETECTIONS
    full_page_min_native_chars: int = OPTION6_FULL_PAGE_MIN_NATIVE_CHARS


@dataclass
class _PagePlan:
    page: OCRPage
    text_units: list[OCRUnit]
    table_units: list[OCRUnit]
    visuals: list[dict[str, Any]]
    native_page: bool
    native_text: str
    full_page_mode: str = ""
    full_page_reason: str = ""


@dataclass(frozen=True)
class _VLMTask:
    page_index: int
    kind: str
    prompt: str
    max_tokens: int
    unit: OCRUnit | None = None
    visual: dict[str, Any] | None = None
    text_input: str | None = None


class Option6Pipeline:
    """Run Page Elements semantic regions through the Qwen VLM."""

    pipeline_name = OPTION6_PIPELINE_NAME
    model_name = OPTION6_MODEL

    def __init__(self, text_vlm: Any, table_vlm: Any, *, config: Option6Config | None = None) -> None:
        if not hasattr(text_vlm, "recognize"):
            raise TypeError("text_vlm backend must expose recognize(images)")
        if not hasattr(table_vlm, "recognize"):
            raise TypeError("table_vlm backend must expose recognize(images)")
        self.text_vlm = text_vlm
        self.table_vlm = table_vlm
        self.config = config or Option6Config()
        self.last_document_diagnostics: dict[str, Any] = {}

    def process_page(self, page: OCRPage | Mapping[str, Any] | Any) -> OCRPageOutput:
        return self.process_document([page])[0]

    def process_document(
        self,
        pages: Sequence[OCRPage | Mapping[str, Any] | Any],
        *,
        document_key: str | None = None,
    ) -> list[OCRPageOutput]:
        started = time.perf_counter()
        page_list = [page_value(page) for page in pages]
        if not page_list:
            self.last_document_diagnostics = {
                "scope": "document",
                "pipeline": OPTION6_SELECTOR,
                "pipeline_name": self.pipeline_name,
                "document_key": document_key,
                "page_count": 0,
            }
            return []

        plans: list[_PagePlan] = []
        errors: list[dict[str, Any]] = []
        crop_started = time.perf_counter()

        def plan_one(page: OCRPage) -> tuple[_PagePlan, dict[str, Any] | None]:
            try:
                return self._plan_page(page), None
            except Exception as exc:  # noqa: BLE001 - page-local failure
                return (
                    _PagePlan(
                        page=page,
                        text_units=[],
                        table_units=[],
                        visuals=[],
                        native_page=_is_native_page(page),
                        native_text=page.native_text,
                    ),
                    _error("crop", exc, page_number=page.page_number),
                )

        crop_workers = max(1, int(self.config.crop_max_concurrency))
        # Chunking bounds decoded page/crop memory. Within each chunk, the
        # independent page rasters are planned in parallel on CPU.
        for offset in range(0, len(page_list), max(1, int(self.config.crop_batch_size))):
            chunk = page_list[offset : offset + max(1, int(self.config.crop_batch_size))]
            if crop_workers > 1 and len(chunk) > 1:
                with ThreadPoolExecutor(max_workers=min(crop_workers, len(chunk))) as executor:
                    planned = list(executor.map(plan_one, chunk))
            else:
                planned = [plan_one(page) for page in chunk]
            for plan, error in planned:
                plans.append(plan)
                if error is not None:
                    errors.append(error)
        crop_seconds = time.perf_counter() - crop_started

        text_items: list[tuple[int, OCRUnit]] = []
        table_items: list[tuple[int, OCRUnit]] = []
        tasks: list[_VLMTask] = []
        for page_index, plan in enumerate(plans):
            text_items.extend((page_index, unit) for unit in plan.text_units)
            table_items.extend((page_index, unit) for unit in plan.table_units)
            for unit in plan.text_units:
                if unit.metadata.get("full_page_mode") == "scan" or unit.metadata.get(
                    "scan_page_full"
                ):
                    prompt = SCAN_PAGE_PROMPT
                elif unit.metadata.get("full_page"):
                    prompt = FULL_PAGE_PROMPT
                else:
                    prompt = TEXT_PROMPT
                tasks.append(
                    _VLMTask(
                        page_index=page_index,
                        kind="text",
                        prompt=prompt,
                        max_tokens=max(1, int(self.config.text_max_output_tokens)),
                        unit=unit,
                    )
                )
            for unit in plan.table_units:
                native_table = (
                    str(unit.metadata.get("table_input") or "")
                    == "pdfium_native_text"
                )
                tasks.append(
                    _VLMTask(
                        page_index=page_index,
                        kind="table",
                        prompt=TABLE_TEXT_PROMPT if native_table else TABLE_PROMPT,
                        max_tokens=max(1, int(self.config.table_max_output_tokens)),
                        unit=unit,
                        text_input=(
                            str(unit.metadata.get("native_table_text") or "")
                            if native_table
                            else None
                        ),
                    )
                )
            if self.config.classify_visual_regions:
                tasks.extend(
                    _VLMTask(
                        page_index=page_index,
                        kind="visual",
                        prompt=VISUAL_PROMPT,
                        max_tokens=OPTION6_MAX_OUTPUT_TOKENS_VISUAL,
                        visual=visual,
                    )
                    for visual in plan.visuals
                    if visual.get("image_b64")
                    and not _is_page_sized_text_layout(visual)
                )
            if self.config.ocr_visual_regions:
                tasks.extend(
                    _VLMTask(
                        page_index=page_index,
                        kind="visual_ocr",
                        prompt=VISUAL_OCR_PROMPT,
                        max_tokens=max(1, int(self.config.visual_ocr_max_tokens)),
                        visual=visual,
                    )
                    for visual in plan.visuals
                    if visual.get("image_b64")
                    and not _is_page_sized_text_layout(visual)
                )

        vlm_started = time.perf_counter()
        responses, vlm_error, vlm_diagnostics = self._recognize_tasks(tasks)
        vlm_seconds = time.perf_counter() - vlm_started
        if vlm_error is not None:
            errors.append(_error("vlm", vlm_error))

        # Native table text is the cheap first pass. If Qwen cannot turn the
        # geometry-preserving text into Markdown, retry that table with the
        # retained image crop. This keeps the scan/image path unchanged and
        # prevents a weak native stream from losing the table.
        native_table_fallback_indices: set[int] = set()
        native_table_fallback_tasks: list[tuple[int, _VLMTask]] = []
        for task_index, task in enumerate(tasks):
            if (
                task.kind != "table"
                or task.text_input is None
                or task.unit is None
                or not task.unit.crop_b64
            ):
                continue
            value = _clean_markdown(_response_text(responses, task_index))
            if _looks_like_markdown_table(value):
                continue
            native_table_fallback_indices.add(task_index)
            native_table_fallback_tasks.append(
                (
                    task_index,
                    _VLMTask(
                        page_index=task.page_index,
                        kind="table",
                        prompt=TABLE_PROMPT,
                        max_tokens=task.max_tokens,
                        unit=task.unit,
                    ),
                )
            )
        if native_table_fallback_tasks:
            fallback_started = time.perf_counter()
            fallback_responses, fallback_error, fallback_diagnostics = self._recognize_tasks(
                [task for _, task in native_table_fallback_tasks]
            )
            vlm_seconds += time.perf_counter() - fallback_started
            for offset, (task_index, _) in enumerate(native_table_fallback_tasks):
                if offset >= len(fallback_responses):
                    continue
                if task_index >= len(responses):
                    responses.extend([None] * (task_index + 1 - len(responses)))
                responses[task_index] = fallback_responses[offset]
            if fallback_error is not None:
                errors.append(_error("vlm_table_fallback", fallback_error))
            vlm_diagnostics = _merge_vlm_diagnostics(
                vlm_diagnostics, fallback_diagnostics
            )

        text_by_page: list[list[dict[str, Any]]] = [[] for _ in plans]
        tables_by_page: list[list[dict[str, Any]]] = [[] for _ in plans]
        for task_index, (task, response) in enumerate(zip(tasks, responses)):
            value = _response_text([response], 0)
            if task.kind == "text":
                if not value or task.unit is None:
                    continue
                text_by_page[task.page_index].append(
                    _text_block(task.unit, value, model=_backend_model(self.text_vlm))
                )
                continue
            if task.kind == "table":
                if task.unit is None:
                    continue
                markdown = _clean_markdown(value)
                # A detector false-positive should not create an empty table
                # row in the final document.
                if not _looks_like_markdown_table(markdown):
                    continue
                tables_by_page[task.page_index].append(
                    {
                        "table_id": task.unit.table_id or task.unit.unit_id,
                        "bbox_xyxy_norm": list(task.unit.bbox_xyxy_norm),
                        "text": markdown,
                        "markdown": markdown,
                        "cells": [],
                        "table_text_format": "markdown",
                        "content_type": "table",
                        "source": self.pipeline_name,
                        "model": _backend_model(self.table_vlm),
                        "provenance": {
                            "backend": getattr(self.table_vlm, "backend", "qwen35_vlm"),
                            "unit_id": task.unit.unit_id,
                            "prompt": (
                                "table_markdown_image_fallback"
                                if task_index in native_table_fallback_indices
                                else (
                                    "table_markdown_native_text"
                                    if task.text_input is not None
                                    else "table_markdown"
                                )
                            ),
                            "input": (
                                "image_crop_fallback"
                                if task_index in native_table_fallback_indices
                                else task.unit.metadata.get("table_input", "page_image")
                            ),
                            "native_text_chars": task.unit.metadata.get(
                                "native_table_chars"
                            ),
                        },
                    }
                )
                continue
            if task.kind == "visual_ocr" and task.visual is not None:
                task.visual["visual_ocr_attempted"] = True
                if value:
                    text_by_page[task.page_index].append(
                        _visual_ocr_block(
                            task.visual,
                            value,
                            model=_backend_model(self.text_vlm),
                        )
                    )
                    task.visual["visual_ocr_text_available"] = True
                continue
            if task.kind == "visual" and task.visual is not None and value:
                # Keep visuals out of OCR text blocks. Their short VLM label
                # is emitted through the existing visual/caption consumer. A
                # page-sized detector false positive is rejected even if the
                # classifier still guesses ``sơ đồ``/``biểu đồ``.
                visual_label = _clean_visual_label(value, task.visual.get("label"))
                if not visual_label or _is_page_sized_text_layout(task.visual):
                    task.visual["visual_rejected"] = True
                    task.visual["visual_reject_reason"] = (
                        "page_sized_text_layout"
                        if visual_label
                        else "vlm_rejected_candidate"
                    )
                else:
                    task.visual["caption"] = visual_label
                task.visual["vlm_classified"] = True
            elif task.kind == "visual" and task.visual is not None:
                # A missing response must not erase a valid detector crop, but
                # a broad text-heavy page candidate is deterministic noise and
                # should never reach the frontend as an image region.
                if _is_page_sized_text_layout(task.visual):
                    task.visual["visual_rejected"] = True
                    task.visual["visual_reject_reason"] = "page_sized_text_layout"

        # Apply the geometry/text safety gate even when the VLM request failed
        # or a legacy backend did not return visual responses.
        for plan in plans:
            for visual in plan.visuals:
                if _is_page_sized_text_layout(visual):
                    visual["visual_rejected"] = True
                    visual.setdefault("visual_reject_reason", "page_sized_text_layout")

        visual_rejected_count = sum(
            1
            for plan in plans
            for visual in plan.visuals
            if visual.get("visual_rejected")
        )
        for plan in plans:
            plan.visuals = [
                visual for visual in plan.visuals if not visual.get("visual_rejected")
            ]

        outputs: list[OCRPageOutput] = []
        native_pages = 0
        for page_index, plan in enumerate(plans):
            blocks = sorted(
                text_by_page[page_index],
                key=lambda item: (
                    int(item.get("reading_order") or 0),
                    float((item.get("bbox_xyxy_norm") or [0, 0])[1]),
                    float((item.get("bbox_xyxy_norm") or [0, 0])[0]),
                ),
            )
            block_text = "\n".join(
                str(block.get("text") or "").strip()
                for block in blocks
                if str(block.get("text") or "").strip()
            ).strip()
            full_page_primary = plan.full_page_mode == "layout"
            if full_page_primary:
                # A layout fallback is used only when native extraction is
                # missing/weak.  The single page VLM response is therefore
                # the page text source; do not concatenate it with the short
                # native fragment and create duplicate content downstream.
                text = block_text or plan.native_text
            elif plan.native_page:
                text = _join_native_and_missing(plan.native_text, block_text)
            else:
                text = block_text
            if plan.native_page:
                native_pages += 1
            page_errors = [
                item
                for item in errors
                if item.get("page_number") in {None, plan.page.page_number}
            ]
            status = "partial" if page_errors and (text or blocks or tables_by_page[page_index]) else (
                "failed" if page_errors else "completed"
            )
            outputs.append(
                OCRPageOutput(
                    pipeline=self.pipeline_name,
                    text=text,
                    ocr_text_blocks=blocks,
                    source=self.pipeline_name,
                    model=OPTION6_MODEL,
                    language=self.config.language,
                    tables=tables_by_page[page_index],
                    visuals=plan.visuals,
                    errors=page_errors,
                    timing={
                        "total_seconds": time.perf_counter() - started,
                        "crop_seconds": crop_seconds,
                        "text_vlm_seconds": vlm_seconds if text_items else 0.0,
                        "table_vlm_seconds": vlm_seconds if table_items and not text_items else 0.0,
                        "visual_vlm_seconds": vlm_seconds if any(
                            task.kind in {"visual", "visual_ocr"}
                            and task.page_index == page_index
                            for task in tasks
                        ) else 0.0,
                        "vlm_seconds": vlm_seconds,
                        "vlm_mixed_batch": bool(vlm_diagnostics.get("mixed_batch")),
                        "vlm_request_count": int(vlm_diagnostics.get("request_count", 0)),
                        "vlm_request_seconds": float(
                            vlm_diagnostics.get("request_seconds", 0.0)
                        ),
                        "vlm_prompt_tokens": int(
                            vlm_diagnostics.get("prompt_tokens", 0)
                        ),
                        "vlm_generation_tokens": int(vlm_diagnostics.get("generation_tokens", 0)),
                        "vlm_generation_tps": float(vlm_diagnostics.get("generation_tps", 0.0)),
                        "text_units": len(plan.text_units),
                        "table_regions": len(plan.table_units),
                        "visual_regions": len(plan.visuals),
                        "native_page": plan.native_page,
                        "crop_max_concurrency": int(self.config.crop_max_concurrency),
                        "native_text_chars": len(plan.native_text),
                        "native_missing_units": len(plan.text_units) if plan.native_page else 0,
                        "full_page_mode": plan.full_page_mode,
                        "full_page_reason": plan.full_page_reason,
                        "full_page_primary": full_page_primary,
                        "detector_batch_size": int(self.config.detector_batch_size),
                        "crop_batch_size": int(self.config.crop_batch_size),
                        "vlm_batch_size": int(self.config.vlm_batch_size),
                    },
                    status=status,
                )
            )

        total_seconds = time.perf_counter() - started
        self.last_document_diagnostics = {
            "scope": "document",
            "pipeline": OPTION6_SELECTOR,
            "pipeline_name": self.pipeline_name,
            "prompt_profile": OPTION6_PROMPT_PROFILE,
            "document_key": document_key,
            "page_count": len(plans),
            "text_units": len(text_items),
            "table_regions": len(table_items),
            "visual_regions": sum(len(plan.visuals) for plan in plans),
            "visual_rejected_regions": visual_rejected_count,
            "native_pages": native_pages,
            "detector_batch_size": int(self.config.detector_batch_size),
            "crop_batch_size": int(self.config.crop_batch_size),
            "crop_max_concurrency": int(self.config.crop_max_concurrency),
            "vlm_batch_size": int(self.config.vlm_batch_size),
            "vlm_max_concurrency": int(
                getattr(self.text_vlm, "max_pool_workers", None)
                or getattr(self.text_vlm, "batch_size", self.config.vlm_batch_size)
            ),
            "vlm_mixed_batch": bool(vlm_diagnostics.get("mixed_batch")),
            "vlm_request_count": int(vlm_diagnostics.get("request_count", 0)),
            "vlm_request_seconds": float(vlm_diagnostics.get("request_seconds", 0.0)),
            "vlm_prompt_tokens": int(vlm_diagnostics.get("prompt_tokens", 0)),
            "text_vlm_requests": sum(1 for task in tasks if task.kind == "text"),
            "table_vlm_requests": sum(1 for task in tasks if task.kind == "table"),
            "native_table_text_requests": sum(
                1
                for task in tasks
                if task.kind == "table" and task.text_input is not None
            ),
            "native_table_image_fallbacks": len(native_table_fallback_tasks),
            "visual_vlm_requests": sum(
                1 for task in tasks if task.kind in {"visual", "visual_ocr"}
            ),
            "visual_classification_requests": sum(
                1 for task in tasks if task.kind == "visual"
            ),
            "visual_ocr_requests": sum(
                1 for task in tasks if task.kind == "visual_ocr"
            ),
            "full_page_vlm_requests": sum(
                1
                for task in tasks
                if task.kind == "text"
                and task.unit is not None
                and task.unit.metadata.get("full_page")
            ),
            "layout_full_page_pages": sum(
                1 for plan in plans if plan.full_page_mode == "layout"
            ),
            "scan_full_page_pages": sum(
                1 for plan in plans if plan.full_page_mode.startswith("scan")
            ),
            "timing": {
                "total_seconds": total_seconds,
                "crop_seconds": crop_seconds,
                "vlm_seconds": vlm_seconds,
                "vlm_request_seconds": float(vlm_diagnostics.get("request_seconds", 0.0)),
                "vlm_prompt_tokens": int(vlm_diagnostics.get("prompt_tokens", 0)),
                "vlm_generation_tokens": int(vlm_diagnostics.get("generation_tokens", 0)),
                "vlm_generation_tps": float(vlm_diagnostics.get("generation_tps", 0.0)),
            },
            "errors": list(errors),
        }
        for output in outputs:
            output.timing["document"] = dict(self.last_document_diagnostics)
        return outputs

    def _plan_page(self, page: OCRPage) -> _PagePlan:
        native_page = _is_native_page(page)
        scan_page = _is_scan_page(page)
        text_units: list[OCRUnit] = []
        table_units: list[OCRUnit] = []
        visuals: list[dict[str, Any]] = []
        detections = page_element_detections(page.page_elements_v3)
        table_regions = _table_regions(
            page,
            detections,
            include_table_structure=bool(
                getattr(self.config, "table_structure", True)
            ),
        )
        full_page_reason = _full_page_layout_reason(
            detections,
            visual_area_threshold=self.config.full_page_visual_area,
            max_detections=self.config.full_page_max_detections,
        )
        native_char_count = len(re.sub(r"\s+", "", str(page.native_text or "")))
        layout_full_page = bool(
            self.config.full_page_layout_fallback
            and not scan_page
            and full_page_reason
            and native_char_count < max(0, int(self.config.full_page_min_native_chars))
        )
        full_page_mode = "layout" if layout_full_page else ""
        cropper = _page_cropper(
            page,
            output_format=self.config.crop_image_format,
            jpeg_quality=self.config.crop_jpeg_quality,
        )
        text_boxes = [
            bbox
            for detection in detections
            if str(detection.get("label_name") or "").strip().lower() in TEXT_LABELS
            and (bbox := clamp_bbox(detection.get("bbox_xyxy_norm"))) is not None
        ]
        for index, detection in enumerate(detections):
            label = str(detection.get("label_name") or "").strip().lower()
            bbox = clamp_bbox(detection.get("bbox_xyxy_norm"))
            if bbox is None:
                continue
            if label == "table":
                continue
            if label in VISUAL_LABELS:
                if self.config.include_visual_regions:
                    crop = _crop(
                        cropper,
                        page.image_b64,
                        bbox,
                        add_padding=False,
                        output_format=self.config.crop_image_format,
                        jpeg_quality=self.config.crop_jpeg_quality,
                    )
                    if crop is not None:
                        visuals.append(
                            {
                                "bbox_xyxy_norm": list(bbox),
                                "image_b64": crop.image_b64,
                                "image_type": "page_detected_region",
                                "content_type": "image",
                                "label": label,
                                "label_name": label,
                                "unit_id": f"page-{page.page_number or 0}-visual-{index}",
                                "reading_order": _reading_order(detection, index),
                                "caption": _VISUAL_SHORT_LABELS.get(label, "hình ảnh"),
                                "text": "",
                                "score": _score(detection.get("score")),
                                "bbox_area": bbox_area(bbox),
                                "page_sized_candidate": bbox_area(bbox)
                                >= self.config.full_page_visual_area,
                                "text_detections_inside": sum(
                                    1
                                    for text_bbox in text_boxes
                                    if containment(text_bbox, bbox) >= 0.60
                                ),
                                "native_text_chars": native_char_count,
                                "native_text_char_threshold": int(
                                    self.config.full_page_min_native_chars
                                ),
                                "source": self.pipeline_name,
                                "model": "Page Elements v3",
                                "page_number": page.page_number,
                            }
                        )
                continue
            # A scanned page, or a native page whose layout signal is too
            # dense/large for reliable bbox OCR, is read by one page-level VLM
            # request below. The detector still supplies table/visual geometry,
            # but individual text boxes would fragment the page and duplicate
            # reading order. Visual crops were retained above on purpose.
            if scan_page or layout_full_page:
                continue
            if label not in TEXT_LABELS or _overlaps_any(bbox, [region[1] for region in table_regions]):
                continue
            # Native PDFium is authoritative for a region that already has
            # character geometry.  No duplicate VLM request is made there.
            if native_page and _native_covers_bbox(page, bbox):
                continue
            local_height = adaptive_local_text_height(bbox, text_boxes or [bbox])
            crop = _crop(
                cropper,
                page.image_b64,
                bbox,
                local_text_height=local_height,
                add_padding=True,
                output_format=self.config.crop_image_format,
                jpeg_quality=self.config.crop_jpeg_quality,
            )
            if crop is None:
                continue
            text_units.append(
                OCRUnit(
                    unit_id=f"page-{page.page_number or 0}-text-{index}",
                    kind="title" if label == "title" else "text_block",
                    source="page_elements_v3",
                    bbox_xyxy_norm=bbox,
                    crop_bbox_xyxy_norm=crop.bbox_xyxy_norm,
                    crop_b64=crop.image_b64,
                    crop_shape_hw=crop.shape_hw,
                    reading_order=_reading_order(detection, index),
                    detector_score=_score(detection.get("score")),
                    label=label,
                    metadata={"page_number": page.page_number, "explicit_reading_order": _has_order(detection)},
                )
            )

        for table_index, (table_id, bbox) in enumerate(table_regions):
            native_table = (
                _native_table_input(page, bbox)
                if native_page and self.config.native_table_text
                else None
            )
            crop = _crop(
                cropper,
                page.image_b64,
                bbox,
                add_padding=False,
                output_format=self.config.crop_image_format,
                jpeg_quality=self.config.crop_jpeg_quality,
            )
            if crop is None and native_table is None:
                continue
            table_metadata: dict[str, Any] = {
                "page_number": page.page_number,
                "prompt": "table_markdown",
            }
            if native_table is not None:
                serialized, native_char_count, native_row_count = native_table
                table_metadata.update(
                    {
                        "table_input": "pdfium_native_text",
                        "native_table_text": serialized,
                        "native_table_chars": native_char_count,
                        "native_table_rows": native_row_count,
                    }
                )
            elif crop is not None:
                table_metadata["table_input"] = "image_crop"
            table_units.append(
                OCRUnit(
                    unit_id=f"page-{page.page_number or 0}-{table_id}",
                    kind="table",
                    source="page_elements_v3",
                    bbox_xyxy_norm=bbox,
                    crop_bbox_xyxy_norm=crop.bbox_xyxy_norm if crop is not None else bbox,
                    crop_b64=crop.image_b64 if crop is not None else "",
                    crop_shape_hw=crop.shape_hw if crop is not None else (0, 0),
                    reading_order=table_index,
                    label="table",
                    table_id=table_id,
                    metadata=table_metadata,
                )
            )

        if scan_page and self.config.scan_full_page:
            mask_regions: list[tuple[float, float, float, float]] = []
            if self.config.scan_mask_layout:
                mask_regions.extend(region[1] for region in table_regions)
                mask_regions.extend(_scan_maskable_visual_regions(visuals))
            crop = _crop(
                cropper,
                page.image_b64,
                (0.0, 0.0, 1.0, 1.0),
                add_padding=False,
                mask_regions=mask_regions,
                output_format=self.config.crop_image_format,
                jpeg_quality=self.config.crop_jpeg_quality,
            )
            if crop is not None:
                text_units.append(
                    OCRUnit(
                        unit_id=f"page-{page.page_number or 0}-scan-page",
                        kind="text_block",
                        source="option6_full_page_fallback",
                        bbox_xyxy_norm=(0.0, 0.0, 1.0, 1.0),
                        crop_bbox_xyxy_norm=crop.bbox_xyxy_norm,
                        crop_b64=crop.image_b64,
                        crop_shape_hw=crop.shape_hw,
                        reading_order=0,
                        label="text",
                        metadata={
                            "scan_page_full": True,
                            "scan_page_fallback": False,
                            "full_page": True,
                            "full_page_mode": "scan",
                            "full_page_reason": "scan_page",
                            "masked_regions": len(mask_regions),
                        },
                    )
                )
            full_page_mode = "scan"
            full_page_reason = "scan_page"
        elif layout_full_page:
            # On a suspicious native layout, preserve the visual crops but let
            # the page request see the complete raster. Do not mask visual
            # regions here: text inside a chart/diagram belongs in the page
            # text, while the same crop remains available as image evidence.
            mask_regions = [region[1] for region in table_regions] if self.config.scan_mask_layout else []
            crop = _crop(
                cropper,
                page.image_b64,
                (0.0, 0.0, 1.0, 1.0),
                add_padding=False,
                mask_regions=mask_regions,
                output_format=self.config.crop_image_format,
                jpeg_quality=self.config.crop_jpeg_quality,
            )
            if crop is not None:
                text_units.append(
                    OCRUnit(
                        unit_id=f"page-{page.page_number or 0}-layout-page",
                        kind="text_block",
                        source="option6_full_page_layout_fallback",
                        bbox_xyxy_norm=(0.0, 0.0, 1.0, 1.0),
                        crop_bbox_xyxy_norm=crop.bbox_xyxy_norm,
                        crop_b64=crop.image_b64,
                        crop_shape_hw=crop.shape_hw,
                        reading_order=0,
                        label="text",
                        metadata={
                            "full_page": True,
                            "full_page_mode": "layout",
                            "full_page_reason": full_page_reason,
                            "page_layout_fallback": True,
                            "masked_regions": len(mask_regions),
                        },
                    )
                )
        elif (
            self.config.scan_page_fallback
            and scan_page
            and not text_units
            and not table_units
            and not visuals
        ):
            crop = _crop(
                cropper,
                page.image_b64,
                (0.0, 0.0, 1.0, 1.0),
                add_padding=False,
                output_format=self.config.crop_image_format,
                jpeg_quality=self.config.crop_jpeg_quality,
            )
            if crop is not None:
                text_units.append(
                    OCRUnit(
                        unit_id=f"page-{page.page_number or 0}-scan-page",
                        kind="text_block",
                        source="option6_full_page_fallback",
                        bbox_xyxy_norm=(0.0, 0.0, 1.0, 1.0),
                        crop_bbox_xyxy_norm=crop.bbox_xyxy_norm,
                        crop_b64=crop.image_b64,
                        crop_shape_hw=crop.shape_hw,
                        reading_order=0,
                        label="text",
                        metadata={"scan_page_full": True, "scan_page_fallback": True},
                    )
                )

        text_units.sort(key=lambda unit: (unit.reading_order, unit.bbox_xyxy_norm[1], unit.bbox_xyxy_norm[0]))
        return _PagePlan(
            page=page,
            text_units=text_units,
            table_units=table_units,
            visuals=visuals,
            native_page=native_page,
            native_text=page.native_text,
            full_page_mode=full_page_mode,
            full_page_reason=full_page_reason if full_page_mode == "layout" else (
                "scan_page" if full_page_mode == "scan" else ""
            ),
        )

    def _recognize_tasks(
        self, tasks: Sequence[_VLMTask]
    ) -> tuple[list[Any], Exception | None, dict[str, Any]]:
        """Submit text, table, and visual work through one continuous pool.

        The production Qwen backend implements ``recognize_with_inputs`` so
        native-table text and image crops become requests in the same
        persistent bounded client pool. Small injected/legacy test doubles
        still get the old text/table fallback; visual classification is
        intentionally skipped there because those doubles do not expose a
        mixed-prompt contract. Visual OCR is included in the fallback because
        it is a real text-recognition task, not a visual label task.
        """
        if not tasks:
            return [], None, {"mixed_batch": False, "request_count": 0}

        mixed_inputs = [
            (
                {"text": task.text_input}
                if task.text_input is not None
                else {"image_b64": task.visual["image_b64"]}
                if task.visual is not None
                else {"image_b64": task.unit.crop_b64 if task.unit is not None else ""}
            )
            for task in tasks
        ]
        mixed_recognize_inputs = getattr(self.text_vlm, "recognize_with_inputs", None)
        if callable(mixed_recognize_inputs):
            try:
                responses = list(
                    mixed_recognize_inputs(
                        mixed_inputs,
                        [task.prompt for task in tasks],
                        max_tokens=max(task.max_tokens for task in tasks),
                        max_tokens_per_task=[task.max_tokens for task in tasks],
                    )
                )
                return responses, None, {
                    "mixed_batch": True,
                    "text_only_batch": any(task.text_input is not None for task in tasks),
                    "request_count": len(tasks),
                    "request_seconds": float(
                        getattr(self.text_vlm, "last_elapsed_s", 0.0)
                    ),
                    "prompt_tokens": int(
                        getattr(self.text_vlm, "last_prompt_tokens", 0)
                    ),
                    "generation_tokens": int(
                        getattr(self.text_vlm, "last_generation_tokens", 0)
                    ),
                    "generation_tps": float(
                        getattr(self.text_vlm, "last_generation_tps", 0.0)
                    ),
                }
            except Exception as exc:  # noqa: BLE001 - preserve page batch shape
                return [], exc, {
                    "mixed_batch": True,
                    "text_only_batch": any(task.text_input is not None for task in tasks),
                    "request_count": len(tasks),
                    "request_seconds": float(
                        getattr(self.text_vlm, "last_elapsed_s", 0.0)
                    ),
                    "prompt_tokens": int(
                        getattr(self.text_vlm, "last_prompt_tokens", 0)
                    ),
                }

        # Older injected/legacy backends only expose image recognition.  The
        # planner retains a crop for those backends, so this compatibility path
        # continues to work without pretending that the text-only contract is
        # available.
        mixed_recognize = getattr(self.text_vlm, "recognize_with_prompts", None)
        if callable(mixed_recognize):
            try:
                responses = list(
                    mixed_recognize(
                        [
                            task.unit.crop_b64
                            if task.unit is not None
                            else task.visual["image_b64"]
                            for task in tasks
                        ],
                        [task.prompt for task in tasks],
                        max_tokens=max(task.max_tokens for task in tasks),
                        max_tokens_per_task=[task.max_tokens for task in tasks],
                    )
                )
                return responses, None, {
                    "mixed_batch": True,
                    "text_only_batch": False,
                    "request_count": len(tasks),
                    "request_seconds": float(
                        getattr(self.text_vlm, "last_elapsed_s", 0.0)
                    ),
                    "prompt_tokens": int(
                        getattr(self.text_vlm, "last_prompt_tokens", 0)
                    ),
                    "generation_tokens": int(
                        getattr(self.text_vlm, "last_generation_tokens", 0)
                    ),
                    "generation_tps": float(
                        getattr(self.text_vlm, "last_generation_tps", 0.0)
                    ),
                }
            except Exception as exc:  # noqa: BLE001 - preserve page batch shape
                return [], exc, {
                    "mixed_batch": True,
                    "text_only_batch": False,
                    "request_count": len(tasks),
                    "request_seconds": float(
                        getattr(self.text_vlm, "last_elapsed_s", 0.0)
                    ),
                    "prompt_tokens": int(
                        getattr(self.text_vlm, "last_prompt_tokens", 0)
                    ),
                }

        # Compatibility path for the small backends used by existing tests or
        # downstream callers.  It remains bounded, but cannot mix prompts in
        # one vLLM stream because the backend has no mixed-prompt method.
        responses: list[Any] = [None] * len(tasks)
        first_error: Exception | None = None
        for kind, backend in (
            ("text", self.text_vlm),
            ("visual_ocr", self.text_vlm),
            ("table", self.table_vlm),
        ):
            indices = [index for index, task in enumerate(tasks) if task.kind == kind]
            if not indices:
                continue
            values, error = self._recognize(
                backend,
                [
                    tasks[index].unit.crop_b64
                    for index in indices
                    if tasks[index].unit is not None
                ],
            )
            if error is not None and first_error is None:
                first_error = error
            for offset, index in enumerate(indices):
                if offset < len(values):
                    responses[index] = values[offset]
        return responses, first_error, {
            "mixed_batch": False,
            "text_only_batch": False,
            "request_count": sum(1 for task in tasks if task.kind != "visual"),
            "request_seconds": 0.0,
            "prompt_tokens": 0,
        }

    @staticmethod
    def _recognize(backend: Any, images: Sequence[str]) -> tuple[list[Any], Exception | None]:
        if not images:
            return [], None
        try:
            return list(backend.recognize(images)), None
        except Exception as exc:  # noqa: BLE001 - output rows remain batch-shaped
            return [], exc


def make_option6_pipeline(
    endpoint: str,
    *,
    api_key: str | None = None,
    language: str | None = "auto",
    timeout_s: float = 120.0,
    batch_size: int = OPTION6_VLM_BATCH_SIZE,
    crop_batch_size: int = OPTION6_CROP_BATCH_SIZE,
    scan_page_fallback: bool = True,
) -> Option6Pipeline:
    """Construct one shared prompt-specialized Qwen VLM adapter."""

    from nemo_retriever.common.modality.ocr.isolated.adapters import make_qwen35_vlm_backend

    effective_batch = max(1, int(batch_size or OPTION6_VLM_BATCH_SIZE))
    shared_vlm = make_qwen35_vlm_backend(
        endpoint,
        model=OPTION6_MODEL,
        api_key=api_key,
        timeout_s=timeout_s,
        batch_size=effective_batch,
        max_pool_workers=OPTION6_VLM_MAX_CONCURRENCY,
        max_tokens=max(OPTION6_MAX_OUTPUT_TOKENS_TABLE, OPTION6_MAX_OUTPUT_TOKENS_TEXT),
        task_prompt=TEXT_PROMPT,
    )
    return Option6Pipeline(
        shared_vlm,
        shared_vlm,
        config=Option6Config(
            language=language,
            crop_batch_size=max(1, int(crop_batch_size)),
            crop_max_concurrency=OPTION6_CROP_MAX_CONCURRENCY,
            vlm_batch_size=effective_batch,
            request_timeout_s=max(1.0, float(timeout_s or 120.0)),
            scan_page_fallback=bool(scan_page_fallback),
            scan_full_page=OPTION6_SCAN_FULL_PAGE,
            scan_mask_layout=OPTION6_SCAN_MASK_LAYOUT,
            classify_visual_regions=OPTION6_VISUAL_VLM,
        ),
    )


def _table_regions(
    page: OCRPage,
    detections: Sequence[Mapping[str, Any]],
    *,
    include_table_structure: bool = True,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    result: list[tuple[str, tuple[float, float, float, float]]] = []
    if include_table_structure:
        payload = page.table_structure_v1
        if isinstance(payload, Mapping):
            for index, region in enumerate(payload.get("regions") or []):
                if not isinstance(region, Mapping):
                    continue
                bbox = clamp_bbox(region.get("bbox_xyxy_norm"))
                if bbox is None:
                    continue
                result.append((str(region.get("table_id") or region.get("id") or f"table-{index}"), bbox))
        if result:
            return result
    for index, detection in enumerate(detections):
        if str(detection.get("label_name") or "").strip().lower() != "table":
            continue
        bbox = clamp_bbox(detection.get("bbox_xyxy_norm"))
        if bbox is not None:
            result.append((str(detection.get("table_id") or detection.get("id") or f"table-{index}"), bbox))
    return result


def _clean_pdfium_table_text(value: str) -> str:
    """Remove PDFium noncharacters without normalizing document content."""
    text = "".join(char for char in str(value or "") if char not in _PDFIUM_TEXT_ARTIFACTS)
    return re.sub(r"[ \t]+", " ", text).strip()


def _native_table_input(
    page: OCRPage,
    bbox: Sequence[float],
) -> tuple[str, int, int] | None:
    """Serialize PDFium characters in a native table bbox for text-only VLM.

    PDFium exposes character geometry but not table cells.  The serializer
    keeps each visual row and its horizontal positions, allowing Qwen to infer
    columns without paying the multimodal image-token cost.  A short or
    geometry-less native stream returns ``None`` and the caller keeps the
    image crop as the fallback.
    """
    spans = page.native_spans or []
    table_bbox = clamp_bbox(bbox)
    if table_bbox is None:
        return None

    characters: list[dict[str, Any]] = []
    artifact_count = 0
    for index, span in enumerate(spans):
        char = str(span.get("char") or "")
        if not char or char in {"\r", "\n"}:
            continue
        char_bbox = clamp_bbox(span.get("bbox_xyxy_norm"))
        if char_bbox is None:
            continue
        x, y = bbox_center(char_bbox)
        if not (
            table_bbox[0] <= x <= table_bbox[2]
            and table_bbox[1] <= y <= table_bbox[3]
        ):
            continue
        if char in _PDFIUM_TEXT_ARTIFACTS:
            artifact_count += 1
            continue
        characters.append(
            {
                "char": char,
                "bbox": char_bbox,
                "x": x,
                "y": y,
                "height": max(char_bbox[3] - char_bbox[1], 0.001),
                "width": max(char_bbox[2] - char_bbox[0], 0.001),
                "index": index,
            }
        )

    # A replacement/noncharacter in the table bbox usually means PDFium could
    # not decode an embedded-font glyph. Do not let Qwen silently repair a
    # broken native stream when the original crop is available for fallback.
    if artifact_count:
        return None

    non_whitespace_chars = sum(1 for item in characters if item["char"].strip())
    if non_whitespace_chars < OPTION6_NATIVE_TABLE_MIN_CHARS:
        return None

    heights = [float(item["height"]) for item in characters]
    widths = [float(item["width"]) for item in characters]
    line_tolerance = max(float(median(heights)) * 0.75, 0.003)
    line_groups: list[dict[str, Any]] = []
    for item in sorted(characters, key=lambda value: (value["y"], value["x"], value["index"])):
        matching = [
            group
            for group in line_groups
            if abs(float(item["y"]) - float(group["y"])) <= line_tolerance
        ]
        if not matching:
            line_groups.append({"y": item["y"], "items": [item]})
            continue
        group = min(matching, key=lambda value: abs(float(item["y"]) - float(value["y"])))
        group["items"].append(item)
        group["y"] = sum(float(member["y"]) for member in group["items"]) / len(group["items"])

    median_width = float(median(widths))
    gap_tolerance = max(median_width * 2.25, float(median(heights)) * 1.4, 0.008)
    word_gap_tolerance = max(median_width * 0.65, 0.002)
    serialized_rows: list[str] = []
    for row_index, group in enumerate(sorted(line_groups, key=lambda value: value["y"]), start=1):
        row_items = sorted(group["items"], key=lambda value: (value["bbox"][0], value["index"]))
        segments: list[dict[str, Any]] = []
        for item in row_items:
            if not segments or float(item["bbox"][0]) - float(segments[-1]["x1"]) > gap_tolerance:
                segments.append(
                    {
                        "x0": float(item["bbox"][0]),
                        "x1": float(item["bbox"][2]),
                        "text": item["char"],
                    }
                )
            else:
                gap = float(item["bbox"][0]) - float(segments[-1]["x1"])
                if gap > word_gap_tolerance and not str(segments[-1]["text"]).endswith(" "):
                    segments[-1]["text"] += " "
                segments[-1]["text"] += item["char"]
                segments[-1]["x1"] = max(float(segments[-1]["x1"]), float(item["bbox"][2]))
        cells = [
            {
                "x0": round(float(segment["x0"]), 4),
                "x1": round(float(segment["x1"]), 4),
                "value": _clean_pdfium_table_text(str(segment["text"])),
            }
            for segment in segments
            if _clean_pdfium_table_text(str(segment["text"]))
        ]
        if cells:
            serialized_rows.append(
                json.dumps(
                    {
                        "row": row_index,
                        "y": round(float(group["y"]), 4),
                        "cells": cells,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

    if not serialized_rows:
        return None
    return "\n".join(serialized_rows), non_whitespace_chars, len(serialized_rows)


def _crop(
    cropper: PageImageCropper | None,
    image_b64: str,
    bbox: Sequence[float],
    *,
    local_text_height: float | None = None,
    add_padding: bool,
    mask_regions: Sequence[Sequence[float]] | None = None,
    output_format: str = OPTION6_CROP_IMAGE_FORMAT,
    jpeg_quality: int = OPTION6_CROP_JPEG_QUALITY,
) -> Any | None:
    if cropper is not None:
        return cropper.crop(
            bbox,
            local_text_height=local_text_height,
            add_padding=add_padding,
            mask_regions=mask_regions,
        )
    return crop_image_b64(
        image_b64,
        bbox,
        local_text_height=local_text_height,
        add_padding=add_padding,
        mask_regions=mask_regions,
        output_format=output_format,
        jpeg_quality=jpeg_quality,
    )


def _page_cropper(
    page: OCRPage,
    *,
    output_format: str = OPTION6_CROP_IMAGE_FORMAT,
    jpeg_quality: int = OPTION6_CROP_JPEG_QUALITY,
) -> PageImageCropper | None:
    if not page.image_b64:
        return None
    try:
        return PageImageCropper(
            page.image_b64,
            output_format=output_format,
            jpeg_quality=jpeg_quality,
        )
    except Exception:  # noqa: BLE001 - page-local malformed image
        return None


def _overlaps_any(bbox: Sequence[float], regions: Sequence[Sequence[float]]) -> bool:
    return any(bbox_iou(bbox, region) >= 0.15 or containment(bbox, region) >= 0.35 for region in regions)


def _full_page_layout_reason(
    detections: Sequence[Mapping[str, Any]],
    *,
    visual_area_threshold: float,
    max_detections: int,
) -> str:
    """Return a compact reason for escalating a native page to full-page VLM.

    A large visual bbox is the common failure mode: Page Elements can label an
    entire raster page as one ``image``/``infographic`` and make the individual
    text boxes unusable. A very dense set of boxes is the second failure mode,
    because sending every tiny box creates request and ordering overhead. The
    caller still applies a native-text quality gate before using this signal.
    """
    threshold = max(0.0, float(visual_area_threshold))
    for detection in detections:
        label = str(detection.get("label_name") or "").strip().lower()
        if label not in VISUAL_LABELS:
            continue
        bbox = clamp_bbox(detection.get("bbox_xyxy_norm"))
        if bbox is None:
            continue
        area = bbox_area(bbox)
        if area >= threshold:
            return f"large_visual:{label}:{area:.3f}"
    if len(detections) >= max(1, int(max_detections)):
        return f"dense_layout:{len(detections)}_detections"
    return ""


def _scan_maskable_visual_regions(
    visuals: Sequence[Mapping[str, Any]],
) -> list[tuple[float, float, float, float]]:
    """Return only embedded visual boxes that are safe to white out.

    A page-sized ``image``/``infographic`` detection is often the detector's
    way of saying that the *whole input is a raster page or screenshot*. It is
    still the source image that the page-level VLM must read, so masking it
    would destroy the OCR input. Smaller visuals remain isolated for the
    separate one/two-word visual classifier request.
    """
    regions: list[tuple[float, float, float, float]] = []
    for visual in visuals:
        bbox = clamp_bbox(visual.get("bbox_xyxy_norm"))
        if bbox is None or bbox_area(bbox) >= OPTION6_SCAN_FULL_PAGE_VISUAL_MIN_AREA:
            continue
        regions.append(tuple(float(value) for value in bbox))
    return regions


def _native_covers_bbox(page: OCRPage, bbox: Sequence[float]) -> bool:
    spans = page.native_spans or []
    for span in spans:
        char = str(span.get("char") or "")
        span_bbox = clamp_bbox(span.get("bbox_xyxy_norm"))
        if not char.strip() or span_bbox is None:
            continue
        x, y = bbox_center(span_bbox)
        if float(bbox[0]) <= x <= float(bbox[2]) and float(bbox[1]) <= y <= float(bbox[3]):
            return True
    return False


def _is_scan_page(page: OCRPage) -> bool:
    metadata = page.metadata
    if bool(metadata.get("needs_ocr_for_text") or metadata.get("needs_ocr")):
        return True
    if metadata.get("has_text") is False:
        return True
    return str(metadata.get("reader_backend") or "").lower() in {"scan", "image", "raster", "ocr"}


def _is_native_page(page: OCRPage) -> bool:
    metadata = page.metadata
    if bool(metadata.get("needs_ocr_for_text") or metadata.get("needs_ocr")):
        return False
    if str(metadata.get("reader_backend") or "").lower() in {
        "native_pdf",
        "native_spreadsheet",
        "openpyxl",
        "python_csv",
    }:
        return True
    return bool(metadata.get("has_text") is True and page.native_text)


def _response_text(responses: Sequence[Any], index: int) -> str:
    if index >= len(responses):
        return ""
    value = responses[index]
    if isinstance(value, Mapping):
        for key in ("text", "content", "output", "response"):
            if key in value:
                return _strip_thinking_markers(value.get(key))
    return _strip_thinking_markers(value)


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _strip_thinking_markers(value: Any) -> str:
    """Remove Qwen reasoning markers before text/table/visual post-processing.

    Thinking is disabled in the request payload, but some vLLM/template
    combinations can still return a literal closing marker. Keep this
    sanitizer local to Pipeline 6 so existing pipeline response formats stay
    unchanged.
    """
    text = str(value or "").strip()
    text = _THINK_BLOCK_RE.sub("", text)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _clean_markdown(value: str) -> str:
    """Keep only the Markdown table from a chatty VLM response."""
    lines = [line.strip() for line in str(value or "").splitlines()]
    lines = [line for line in lines if line and not line.startswith("```")]
    if not lines:
        return ""

    pipe_indexes = [index for index, line in enumerate(lines) if "|" in line]
    if not pipe_indexes:
        return "\n".join(lines).strip()

    separator_index = next(
        (
            index
            for index in pipe_indexes
            if re.search(r"\|?\s*:?-{3,}:?\s*(\||$)", lines[index])
        ),
        None,
    )
    if separator_index is None:
        return "\n".join(lines).strip()

    start = separator_index
    while start > 0 and "|" in lines[start - 1]:
        start -= 1
    end = separator_index
    while end + 1 < len(lines) and "|" in lines[end + 1]:
        end += 1
    return "\n".join(lines[start : end + 1]).strip()


def _looks_like_markdown_table(value: str) -> bool:
    """Return whether a response is structured enough to keep as a table."""
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    pipe_lines = [line for line in lines if "|" in line]
    separator_lines = [
        line
        for line in pipe_lines
        if re.search(r"\|?\s*:?-{3,}:?\s*(\||$)", line)
    ]
    if len(pipe_lines) < 2 or not separator_lines:
        return False
    # If Qwen echoed the layout schema, the result is not a faithful table.
    # Reject that response so a native page with a confusing layout uses the
    # image crop fallback instead of leaking coordinates into the document.
    # A normal document table may legitimately have a ``Value`` column.  Scan
    # every emitted row instead of only the first one: Qwen may prepend blank
    # cells and echo the transport schema on a later line.
    for line in pipe_lines:
        cells = {part.strip().casefold() for part in line.strip("|").split("|")}
        coordinate_headers = cells & {"x", "x0", "x1", "y"}
        has_layout_schema = (
            "cells" in cells
            or bool({"row", "y"} <= cells)
            or bool("row" in cells and coordinate_headers)
            or bool({"x0", "x1"} <= cells)
        )
        if has_layout_schema:
            return False
    return True


def _merge_vlm_diagnostics(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
) -> dict[str, Any]:
    """Add diagnostics from an image fallback without losing the first pass."""
    merged = dict(primary)
    for key in ("request_count", "prompt_tokens", "generation_tokens"):
        merged[key] = int(primary.get(key, 0) or 0) + int(secondary.get(key, 0) or 0)
    merged["request_seconds"] = float(primary.get("request_seconds", 0.0) or 0.0) + float(
        secondary.get("request_seconds", 0.0) or 0.0
    )
    merged["generation_tps"] = (
        merged["generation_tokens"] / merged["request_seconds"]
        if merged["request_seconds"] > 0.0
        else 0.0
    )
    merged["text_only_batch"] = bool(
        primary.get("text_only_batch") or secondary.get("text_only_batch")
    )
    return merged


def _clean_visual_label(value: str, detector_label: Any = None) -> str:
    """Keep visual classification compact and deterministic for embedding."""
    text = " ".join(str(value or "").strip().splitlines()).strip().strip("`\"'")
    text = " ".join(text.split())
    if not text:
        return _VISUAL_SHORT_LABELS.get(str(detector_label or "").lower(), "hình ảnh")
    folded = "".join(
        character
        for character in unicodedata.normalize("NFD", text.casefold())
        if unicodedata.category(character) != "Mn"
    )
    if folded.startswith(("bo qua", "skip", "none", "not visual", "khong phai")):
        return ""
    aliases = {
        "image": "hình ảnh",
        "photo": "hình ảnh",
        "hình": "hình ảnh",
        "chart": "biểu đồ",
        "graph": "biểu đồ",
        "diagram": "sơ đồ",
        "infographic": "sơ đồ",
        "stamp": "con dấu",
        "seal": "con dấu",
    }
    normalized = text.casefold()
    for key, label in aliases.items():
        if key in normalized:
            return label
    return " ".join(text.split()[:2])


def _is_page_sized_text_layout(visual: Mapping[str, Any]) -> bool:
    """Reject a detector bbox that is really a text-heavy page/background.

    The VLM sees the crop only, so a full-page crop containing a small diagram
    can otherwise be mistaken for that diagram. Page Elements already gives us
    enough geometry to make this gate deterministic: a page-sized candidate
    with multiple text detections, or substantial native text, is layout noise,
    not an independent visual object.
    """
    if not bool(visual.get("page_sized_candidate")):
        return False
    try:
        text_detections = int(visual.get("text_detections_inside") or 0)
    except (TypeError, ValueError):
        text_detections = 0
    try:
        native_chars = int(visual.get("native_text_chars") or 0)
    except (TypeError, ValueError):
        native_chars = 0
    try:
        native_threshold = int(
            visual.get("native_text_char_threshold")
            or OPTION6_FULL_PAGE_MIN_NATIVE_CHARS
        )
    except (TypeError, ValueError):
        native_threshold = OPTION6_FULL_PAGE_MIN_NATIVE_CHARS
    return text_detections >= 2 or native_chars >= native_threshold


def _join_native_and_missing(native_text: str, missing_text: str) -> str:
    native = str(native_text or "").strip()
    missing = str(missing_text or "").strip()
    if native and missing:
        return f"{native}\n\n{missing}"
    return native or missing


def _text_block(unit: OCRUnit, text: str, *, model: str) -> dict[str, Any]:
    return {
        "text": text,
        "bbox_xyxy_norm": list(unit.bbox_xyxy_norm),
        "score": unit.detector_score,
        "confidence": unit.detector_score,
        "source": OPTION6_PIPELINE_NAME,
        "model": model,
        "language": "auto",
        "content_type": "text" if unit.kind != "title" else "title",
        "reading_order": int(unit.reading_order),
        "unit_id": unit.unit_id,
        "provenance": {
            "backend": "qwen35_vlm",
            "label": unit.label,
            "prompt": "text_verbatim",
        },
    }


def _visual_ocr_block(
    visual: Mapping[str, Any],
    text: str,
    *,
    model: str,
) -> dict[str, Any]:
    """Represent OCR text read from a Page Elements visual crop.

    The visual evidence row remains image-only with a short detector caption;
    this separate block carries the actual text read by the VLM so downstream
    retrieval and the debug overlay can distinguish visual OCR from native PDF
    text.
    """
    bbox = list(visual.get("bbox_xyxy_norm") or [0.0, 0.0, 1.0, 1.0])
    try:
        reading_order = int(visual.get("reading_order") or 0)
    except (TypeError, ValueError):
        reading_order = 0
    page_number = visual.get("page_number") or 0
    unit_id = str(
        visual.get("unit_id")
        or f"page-{page_number}-visual-{reading_order}"
    )
    label = str(visual.get("label_name") or visual.get("label") or "image")
    return {
        "text": str(text or "").strip(),
        "bbox_xyxy_norm": bbox,
        "score": visual.get("score"),
        "confidence": visual.get("score"),
        "source": OPTION6_PIPELINE_NAME,
        "model": model,
        "language": "auto",
        "content_type": "text",
        "reading_order": reading_order,
        "unit_id": unit_id,
        "ocr_source": OPTION6_PIPELINE_NAME,
        "ocr_mode": "visual_crop",
        "reader_backend": "ocr",
        "region_label": label,
        "provenance": {
            "backend": "qwen35_vlm",
            "selected_backend": "qwen35_vlm",
            "bbox_source": "page_elements_v3",
            "region_label": label,
            "prompt": "visual_ocr_verbatim",
            "input": "page_elements_visual_crop",
        },
    }


def _reading_order(detection: Mapping[str, Any], fallback: int) -> int:
    for key in ("reading_order", "readingOrder", "order", "index"):
        try:
            if detection.get(key) is not None:
                return int(detection[key])
        except (TypeError, ValueError):
            pass
    return fallback


def _has_order(detection: Mapping[str, Any]) -> bool:
    return any(detection.get(key) is not None for key in ("reading_order", "readingOrder", "order", "index"))


def _score(value: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _backend_model(backend: Any) -> str:
    return str(getattr(backend, "model", None) or OPTION6_MODEL)


def _backend_request_count(backend: Any, input_count: int) -> int:
    if not input_count:
        return 0
    try:
        value = int(getattr(backend, "last_request_count", 0))
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    return 1


def _error(stage: str, exc: Exception, *, page_number: int | None = None) -> dict[str, Any]:
    value = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
    if page_number is not None:
        value["page_number"] = page_number
    return value


__all__ = [
    "OPTION6_CROP_BATCH_SIZE",
    "OPTION6_CROP_MAX_CONCURRENCY",
    "OPTION6_DETECTOR_BATCH_SIZE",
    "OPTION6_DETECTOR_MAX_POOL_WORKERS",
    "OPTION6_MAX_REQUEST_WORKERS",
    "OPTION6_NATIVE_TABLE_TEXT_ENABLED",
    "OPTION6_NATIVE_TABLE_MIN_CHARS",
    "OPTION6_PROMPT_PROFILE",
    "OPTION6_MODEL",
    "OPTION6_PAGE_ELEMENTS_WORKERS",
    "OPTION6_PDF_EXTRACT_BATCH_SIZE",
    "OPTION6_PDF_EXTRACT_CPUS",
    "OPTION6_PDF_EXTRACT_WORKERS",
    "OPTION6_PDF_SPLIT_BATCH_SIZE",
    "OPTION6_STREAMING_ENABLED",
    "OPTION6_STREAM_BATCH_SIZE",
    "OPTION6_STREAM_QUEUE_BLOCKS",
    "OPTION6_PIPELINE_NAME",
    "OPTION6_SELECTOR",
    "OPTION6_VLM_BATCH_SIZE",
    "OPTION6_VLM_MAX_CONCURRENCY",
    "Option6Config",
    "Option6Pipeline",
    "make_option6_pipeline",
]
