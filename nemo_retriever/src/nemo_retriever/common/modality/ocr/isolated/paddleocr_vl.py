# SPDX-License-Identifier: Apache-2.0

"""Option 2 adapter for the full PaddleOCR-VL 1.6 service.

The official PaddleOCR-VL deployment has two layers: a full pipeline API
(layout analysis + result assembly) and a separate VLM inference server.  The
Retriever only talks to the former.  The API service calls the vLLM service
internally, so this adapter never calls Page Elements, Table Structure, or
Nemotron OCR.

The upload selector remains ``pipeline-tesseract`` for backwards
compatibility with the existing dashboard/API contract.  Its implementation
is now PaddleOCR-VL 1.6, not Tesseract or the old PP-OCRv6 route.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from PIL import Image

from nemo_retriever.operators.extract.ocr.ocr import _crop_b64_image_by_norm_bbox


OPTION2_SELECTOR = "pipeline-tesseract"
PADDLEOCR_VL_MODEL = "PaddleOCR-VL-1.6-0.9B"
PADDLEOCR_VL_PIPELINE = "PaddleOCR-VL-1.6"
PADDLEOCR_VL_LAYOUT_MODEL = "PP-DocLayoutV3"

_TEXT_LABELS = frozenset(
    {
        "text",
        "doc_title",
        "paragraph_title",
        "title",
        "header",
        "footer",
        "abstract",
        "content",
        "reference",
        "reference_content",
        "footnote",
        "vision_footnote",
        "number",
        "formula_number",
        "figure_title",
        "table_caption",
    }
)
_TABLE_LABELS = frozenset({"table"})
_VISUAL_LABELS = frozenset(
    {
        "image",
        "chart",
        "infographic",
        "seal",
        "header_image",
        "footer_image",
    }
)


class PaddleOCRVLServiceError(RuntimeError):
    """Raised when the full PaddleOCR-VL service cannot parse a page."""


def _http_post_json(url: str, payload: Mapping[str, Any], timeout_s: float) -> dict[str, Any]:
    """POST one image to the official PaddleOCR-VL ``layout-parsing`` API."""

    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=float(timeout_s)) as response:
            body = response.read()
            status = int(getattr(response, "status", 200))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise PaddleOCRVLServiceError(f"HTTP {exc.code} from PaddleOCR-VL: {detail}") from exc
    except URLError as exc:
        raise PaddleOCRVLServiceError(f"PaddleOCR-VL request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise PaddleOCRVLServiceError("PaddleOCR-VL request timed out") from exc

    if status >= 400:
        raise PaddleOCRVLServiceError(f"HTTP {status} from PaddleOCR-VL")
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PaddleOCRVLServiceError("PaddleOCR-VL returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise PaddleOCRVLServiceError("PaddleOCR-VL returned a non-object JSON response")
    error_code = decoded.get("errorCode")
    if error_code not in (None, 0, "0"):
        raise PaddleOCRVLServiceError(
            f"PaddleOCR-VL error {error_code}: {decoded.get('errorMsg') or 'unknown error'}"
        )
    return decoded


@dataclass(frozen=True)
class PaddleOCRVLClient:
    """Small transport wrapper kept injectable for contract tests."""

    endpoint: str
    timeout_s: float = 180.0
    transport: Callable[[str, Mapping[str, Any], float], dict[str, Any]] = _http_post_json

    def parse_page(self, image_b64: str) -> dict[str, Any]:
        if not image_b64:
            raise PaddleOCRVLServiceError("page image is empty")
        return self.transport(
            self.endpoint,
            {"file": image_b64, "fileType": 1, "visualize": False},
            float(self.timeout_s),
        )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_mapping(value: Any, *keys: str) -> Mapping[str, Any]:
    current = _mapping(value)
    for key in keys:
        candidate = current.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _first_list(value: Any, *keys: str) -> list[Any]:
    current = _mapping(value)
    for key in keys:
        candidate = current.get(key)
        if isinstance(candidate, list):
            return candidate
    return []


def _page_result(response: Mapping[str, Any]) -> Mapping[str, Any]:
    """Get one page's ``prunedResult`` from the official API response."""

    result = _first_mapping(response, "result") or response
    pages = _first_list(result, "layoutParsingResults", "layout_parsing_results")
    if pages:
        page = _mapping(pages[0])
        return _first_mapping(page, "prunedResult", "pruned_result", "res") or page
    return _first_mapping(result, "prunedResult", "pruned_result", "res") or result


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bbox_norm(value: Any, *, width: int, height: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    numbers = [_number(value[index]) for index in range(4)]
    if any(number is None for number in numbers):
        return None
    x0, y0, x1, y1 = (float(number) for number in numbers if number is not None)
    # Paddle returns pixel coordinates for ``block_bbox``.  Accept normalized
    # coordinates as well because hand-built service mocks and future API
    # versions may expose them directly.
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
        normalized = [x0, y0, x1, y1]
    else:
        normalized = [x0 / max(width, 1), y0 / max(height, 1), x1 / max(width, 1), y1 / max(height, 1)]
    x0, y0, x1, y1 = normalized
    clipped = [max(0.0, min(1.0, item)) for item in (x0, y0, x1, y1)]
    left, top, right, bottom = clipped
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _image_shape(image_b64: str) -> tuple[int, int]:
    try:
        with Image.open(BytesIO(base64.b64decode(image_b64))) as image:
            width, height = image.size
            return max(1, int(width)), max(1, int(height))
    except Exception as exc:
        raise PaddleOCRVLServiceError("invalid page image returned by PDF/image extraction") from exc


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("text", "content", "markdown", "block_content"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    if value in (None, "", [], {}):
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value).strip()


def _label(value: Any) -> str:
    return str(value or "text").strip().lower().replace(" ", "_") or "text"


def _raw_blocks(page_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return Paddle's reading-order blocks, with layout fallback."""

    parsing = _first_list(page_result, "parsing_res_list", "parsingResList")
    if parsing:
        return [dict(item) for item in parsing if isinstance(item, Mapping)]

    layout = _first_mapping(page_result, "layout_det_res", "layoutDetRes")
    boxes = _first_list(layout, "boxes", "detections")
    blocks: list[dict[str, Any]] = []
    for index, item in enumerate(boxes):
        if not isinstance(item, Mapping):
            continue
        block = dict(item)
        block.setdefault("block_label", item.get("label") or item.get("label_name") or "text")
        block.setdefault("block_bbox", item.get("coordinate") or item.get("bbox"))
        block.setdefault("block_id", index)
        block.setdefault("block_order", index + 1)
        blocks.append(block)
    return blocks


def _bbox_key(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(round(float(item), 4) for item in value)
    except (TypeError, ValueError):
        return None


def _append_unique_image(images: list[dict[str, Any]], item: dict[str, Any]) -> None:
    key = (_label(item.get("label_name")), _bbox_key(item.get("bbox_xyxy_norm")))
    if any((_label(existing.get("label_name")), _bbox_key(existing.get("bbox_xyxy_norm"))) == key for existing in images):
        return
    images.append(item)


def _normalise_page(
    row: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    extract_text: bool,
    extract_tables: bool,
    extract_charts: bool,
    extract_images: bool,
    extract_infographics: bool,
) -> dict[str, Any]:
    page_image = row.get("page_image") if isinstance(row.get("page_image"), Mapping) else {}
    image_b64 = str(page_image.get("image_b64") or "")
    width, height = _image_shape(image_b64)
    page_result = _page_result(response)
    blocks = _raw_blocks(page_result)

    text_blocks: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    charts: list[dict[str, Any]] = []
    infographics: list[dict[str, Any]] = []
    visual_images: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []

    for index, block in enumerate(blocks):
        label = _label(block.get("block_label") or block.get("label") or block.get("label_name"))
        bbox = _bbox_norm(
            block.get("block_bbox") or block.get("bbox") or block.get("coordinate"),
            width=width,
            height=height,
        )
        if bbox is None:
            continue
        score = _number(block.get("score") or block.get("confidence"))
        content = _text_content(block.get("block_content") or block.get("content") or block.get("text"))
        order = block.get("block_order")
        try:
            reading_order = int(order) if order is not None else index + 1
        except (TypeError, ValueError):
            reading_order = index + 1

        detections.append(
            {
                "label_name": label,
                "bbox_xyxy_norm": bbox,
                "score": score,
                "reading_order": reading_order,
                "model": PADDLEOCR_VL_LAYOUT_MODEL,
                "source": "paddleocr_vl_layout",
            }
        )

        common = {
            "bbox_xyxy_norm": bbox,
            "text": content,
            "score": score,
            "confidence": score,
            "source": "paddleocr_vl_vllm",
            "model": PADDLEOCR_VL_MODEL,
            "ocr_mode": "paddleocr_vl_full_pipeline",
            "region_label": label,
            "reading_order": reading_order,
            "paddle_block_id": block.get("block_id", index),
        }

        if label in _TABLE_LABELS:
            if extract_tables:
                tables.append(
                    {
                        **common,
                        "table_id": f"paddle-table-{index}",
                        "table_structure_status": "paddleocr_vl_structured",
                    }
                )
            continue

        if label in _VISUAL_LABELS:
            crop = None
            try:
                crop, _ = _crop_b64_image_by_norm_bbox(image_b64, bbox_xyxy_norm=bbox)
            except Exception:
                crop = None
            if not crop:
                continue
            visual = {
                **common,
                "image_b64": crop,
                "label_name": label,
                "image_type": "paddleocr_vl_detected_region",
            }
            # A chart/infographic can legitimately have no textual caption.
            # Keep one canonical visual row after explode_content_to_rows;
            # otherwise a text-less structured item would be dropped because
            # that generic row expander emits rows from non-empty text.
            if not visual["text"] and label in {"chart", "infographic"}:
                visual["text"] = f"[{label}]"
            if label == "chart" and extract_charts:
                charts.append(visual)
            elif label == "infographic" and extract_infographics:
                infographics.append(visual)
            elif extract_images:
                _append_unique_image(visual_images, visual)
            continue

        if extract_text and content:
            text_blocks.append(
                {
                    **common,
                    "content_type": "text",
                    "text": content,
                }
            )

    text_blocks.sort(key=lambda item: (int(item.get("reading_order") or 0), item["bbox_xyxy_norm"][1], item["bbox_xyxy_norm"][0]))
    text = "\n".join(str(item.get("text") or "").strip() for item in text_blocks if str(item.get("text") or "").strip())
    base_images = [dict(item) for item in (row.get("images") or []) if isinstance(item, Mapping)]
    all_images = base_images + visual_images
    raw_for_trace = dict(page_result)
    metadata = dict(row.get("metadata") or {}) if isinstance(row.get("metadata"), Mapping) else {}
    metadata.update(
        {
            "ocr_pipeline": OPTION2_SELECTOR,
            "ocr_source": "paddleocr_vl_full_pipeline",
            "ocr_model": PADDLEOCR_VL_MODEL,
            "ocr_layout_model": PADDLEOCR_VL_LAYOUT_MODEL,
            "ocr_status": "success",
            "paddleocr_vl": {
                "pipeline": PADDLEOCR_VL_PIPELINE,
                "backend": "paddlepaddle_layout_plus_vllm",
                "block_count": len(detections),
                "text_block_count": len(text_blocks),
                "table_count": len(tables),
                "visual_count": len(charts) + len(infographics) + len(visual_images),
                "result": raw_for_trace,
            },
        }
    )

    out = dict(row)
    # Native PDF spans would otherwise make the shared cleaner overwrite the
    # Paddle result. Option 2 is page-image-first, so native spans are not a
    # second text source here.
    out.pop("_native_text_spans", None)
    out["text"] = text
    out["table"] = tables
    out["tables"] = tables
    out["chart"] = charts
    out["charts"] = charts
    out["infographic"] = infographics
    out["infographics"] = infographics
    out["images"] = all_images
    out["_ocr_text_blocks"] = text_blocks
    out["ocr_text_blocks"] = text_blocks
    out["page_elements_v3"] = {
        "detections": detections,
        "model": PADDLEOCR_VL_LAYOUT_MODEL,
        "source": "paddleocr_vl_layout",
        "pipeline": PADDLEOCR_VL_PIPELINE,
        "timing": {},
        "error": None,
    }
    out["ocr"] = {
        "pipeline": OPTION2_SELECTOR,
        "source": "paddleocr_vl_full_pipeline",
        "model": PADDLEOCR_VL_MODEL,
        "layout_model": PADDLEOCR_VL_LAYOUT_MODEL,
        "backend": "vllm",
        "status": "success",
        "num_detections": len(detections),
        "counts_by_label": {
            label: sum(1 for item in detections if item.get("label_name") == label)
            for label in sorted({str(item.get("label_name")) for item in detections})
        },
        "output": raw_for_trace,
    }
    out["metadata"] = metadata
    return out


def _failed_page(row: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    out = dict(row)
    metadata = dict(row.get("metadata") or {}) if isinstance(row.get("metadata"), Mapping) else {}
    error = {"type": type(exc).__name__, "message": str(exc), "stage": "paddleocr_vl_full_pipeline"}
    metadata.update(
        {
            "ocr_pipeline": OPTION2_SELECTOR,
            "ocr_source": "paddleocr_vl_full_pipeline",
            "ocr_model": PADDLEOCR_VL_MODEL,
            "ocr_status": "failed",
            "ocr_errors": [error],
        }
    )
    out["text"] = ""
    out["table"] = []
    out["tables"] = []
    out["chart"] = []
    out["charts"] = []
    out["infographic"] = []
    out["infographics"] = []
    out["_ocr_text_blocks"] = []
    out["ocr_text_blocks"] = []
    out["page_elements_v3"] = {"detections": [], "model": PADDLEOCR_VL_LAYOUT_MODEL, "error": error}
    out["ocr"] = {
        "pipeline": OPTION2_SELECTOR,
        "source": "paddleocr_vl_full_pipeline",
        "model": PADDLEOCR_VL_MODEL,
        "backend": "vllm",
        "status": "failed",
        "num_detections": 0,
        "errors": [error],
    }
    out["metadata"] = metadata
    return out


def run_paddleocr_vl_batch(
    batch_df: Any,
    *,
    invoke_url: str | None,
    request_timeout_s: float = 180.0,
    extract_text: bool = True,
    extract_tables: bool = True,
    extract_charts: bool = True,
    extract_images: bool = True,
    extract_infographics: bool = True,
    transport: Callable[[str, Mapping[str, Any], float], dict[str, Any]] = _http_post_json,
) -> Any:
    """Run only the full PaddleOCR-VL path over extracted page rasters.

    Errors are kept as page-local failed results.  There is intentionally no
    fallback to any NVIDIA or Tesseract endpoint.
    """

    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df
    endpoint = str(invoke_url or "").strip()
    if not endpoint:
        raise RuntimeError("pipeline-tesseract (Option 2) requires paddleocr_vl_invoke_url")

    client = PaddleOCRVLClient(endpoint=endpoint, timeout_s=request_timeout_s, transport=transport)
    output_rows: list[dict[str, Any]] = []
    for _, series in batch_df.iterrows():
        row = series.to_dict()
        started = time.perf_counter()
        try:
            response = client.parse_page(
                str(
                    (row.get("page_image") or {}).get("image_b64")
                    if isinstance(row.get("page_image"), Mapping)
                    else ""
                )
            )
            parsed = _normalise_page(
                row,
                response,
                extract_text=extract_text,
                extract_tables=extract_tables,
                extract_charts=extract_charts,
                extract_images=extract_images,
                extract_infographics=extract_infographics,
            )
            parsed["ocr"]["timing"] = {"seconds": time.perf_counter() - started}
            parsed["metadata"]["ocr_timing"] = dict(parsed["ocr"]["timing"])
            output_rows.append(parsed)
        except BaseException as exc:
            failed = _failed_page(row, exc)
            failed["ocr"]["timing"] = {"seconds": time.perf_counter() - started}
            failed["metadata"]["ocr_timing"] = dict(failed["ocr"]["timing"])
            output_rows.append(failed)
    return pd.DataFrame(output_rows).reset_index(drop=True)


__all__ = [
    "OPTION2_SELECTOR",
    "PADDLEOCR_VL_LAYOUT_MODEL",
    "PADDLEOCR_VL_MODEL",
    "PADDLEOCR_VL_PIPELINE",
    "PaddleOCRVLClient",
    "PaddleOCRVLServiceError",
    "run_paddleocr_vl_batch",
]
