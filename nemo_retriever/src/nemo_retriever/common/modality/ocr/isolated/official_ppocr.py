# SPDX-License-Identifier: Apache-2.0

"""Retriever adapter for the official PP-OCRv6 general OCR pipeline."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable, Mapping
from io import BytesIO
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from PIL import Image


OPTION2_SELECTOR = "pipeline-ppocrv6"
PIPELINE_NAME = "ppocrv6_official_general_ocr"


class PPOCROfficialServiceError(RuntimeError):
    """Raised when the PP-OCRv6 official pipeline cannot parse a page."""


def _http_post_json(url: str, payload: Mapping[str, Any], timeout_s: float) -> Any:
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
        raise PPOCROfficialServiceError(f"HTTP {exc.code} from PP-OCRv6: {detail}") from exc
    except URLError as exc:
        raise PPOCROfficialServiceError(f"PP-OCRv6 request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise PPOCROfficialServiceError("PP-OCRv6 request timed out") from exc
    if status >= 400:
        raise PPOCROfficialServiceError(f"HTTP {status} from PP-OCRv6")
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PPOCROfficialServiceError("PP-OCRv6 returned invalid JSON") from exc


def _image_shape(image_b64: str) -> tuple[int, int]:
    try:
        with Image.open(BytesIO(base64.b64decode(image_b64))) as image:
            return max(1, int(image.size[0])), max(1, int(image.size[1]))
    except Exception as exc:
        raise PPOCROfficialServiceError("invalid page image returned by PDF extraction") from exc


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    x0, y0, x1, y1 = result
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _response_page(response: Any) -> Mapping[str, Any]:
    if isinstance(response, list):
        response = response[0] if response else {}
    if isinstance(response, Mapping):
        return response
    return {}


def _safe_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalise_page(row: Mapping[str, Any], response: Any) -> dict[str, Any]:
    page_response = _response_page(response)
    page_image = row.get("page_image") if isinstance(row.get("page_image"), Mapping) else {}
    image_b64 = str(page_image.get("image_b64") or "")
    width, height = _image_shape(image_b64)
    lines = page_response.get("lines") or []
    if not isinstance(lines, list):
        lines = []

    text_blocks: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    for index, item in enumerate(lines):
        if not isinstance(item, Mapping):
            continue
        bbox = _bbox(item.get("bbox"))
        if bbox is None:
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        score = item.get("score")
        detector_score = item.get("detector_score")
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        try:
            detector_score = float(detector_score) if detector_score is not None else None
        except (TypeError, ValueError):
            detector_score = None
        block = {
            "bbox_xyxy_norm": bbox,
            "text": text,
            "score": score,
            "confidence": score,
            "detector_score": detector_score,
            "source": "ppocrv6_official",
            "model": page_response.get("models", {}).get("text_recognition", "PP-OCRv6_medium_rec"),
            "detector_model": page_response.get("models", {}).get("text_detection", "PP-OCRv6_medium_det"),
            "content_type": "text",
            "reading_order": index,
            "line_angle": item.get("line_angle", 0.0),
        }
        text_blocks.append(block)
        detections.append(
            {
                "label_name": "text",
                "bbox_xyxy_norm": bbox,
                "score": detector_score,
                "reading_order": index,
                "model": block["detector_model"],
                "source": "ppocrv6_official_text_detection",
            }
        )

    text = "\n".join(item["text"] for item in text_blocks)
    metadata = _safe_mapping(row.get("metadata"))
    metadata.update(
        {
            "ocr_pipeline": OPTION2_SELECTOR,
            "ocr_source": PIPELINE_NAME,
            "ocr_model": "PP-OCRv6_medium_det + PP-OCRv6_medium_rec",
            "ocr_status": "success",
            "reader_backend": PIPELINE_NAME,
            "ppocrv6_official": {
                "pipeline": page_response.get("pipeline", "PP-OCRv6 official general OCR"),
                "models": _safe_mapping(page_response.get("models")),
                "preprocess": _safe_mapping(page_response.get("preprocess")),
                "raw_counts": _safe_mapping(page_response.get("raw_counts")),
                "image_size": {"width": width, "height": height},
            },
        }
    )
    output = dict(row)
    # Option 2 is page-image-first. Native PDF spans must not become a second
    # text source beside the official PP-OCRv6 result.
    output.pop("_native_text_spans", None)
    output["text"] = text
    output["table"] = []
    output["tables"] = []
    output["chart"] = []
    output["charts"] = []
    output["infographic"] = []
    output["infographics"] = []
    output["_ocr_text_blocks"] = text_blocks
    output["ocr_text_blocks"] = text_blocks
    output["page_elements_v3"] = {
        "detections": detections,
        "model": "PP-OCRv6_medium_det",
        "source": "ppocrv6_official_text_detection",
        "pipeline": "PP-OCRv6 official general OCR",
        "timing": {},
        "error": None,
    }
    output["ocr"] = {
        "pipeline": OPTION2_SELECTOR,
        "source": PIPELINE_NAME,
        "backend": "paddleocr",
        "status": "success",
        "models": _safe_mapping(page_response.get("models")),
        "preprocess": _safe_mapping(page_response.get("preprocess")),
        "num_detections": len(detections),
        "counts_by_label": {"text": len(detections)},
        "output": page_response,
    }
    output["metadata"] = metadata
    return output


def _failed_page(row: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    output = dict(row)
    error = {"type": type(exc).__name__, "message": str(exc), "stage": "ppocrv6_official"}
    metadata = _safe_mapping(row.get("metadata"))
    metadata.update(
        {
            "ocr_pipeline": OPTION2_SELECTOR,
            "ocr_source": PIPELINE_NAME,
            "ocr_status": "failed",
            "ocr_errors": [error],
        }
    )
    output["text"] = ""
    output["table"] = []
    output["tables"] = []
    output["chart"] = []
    output["charts"] = []
    output["infographic"] = []
    output["infographics"] = []
    output["_ocr_text_blocks"] = []
    output["ocr_text_blocks"] = []
    output["page_elements_v3"] = {
        "detections": [],
        "model": "PP-OCRv6_medium_det",
        "source": "ppocrv6_official_text_detection",
        "error": error,
    }
    output["ocr"] = {
        "pipeline": OPTION2_SELECTOR,
        "source": PIPELINE_NAME,
        "backend": "paddleocr",
        "status": "failed",
        "num_detections": 0,
        "errors": [error],
    }
    output["metadata"] = metadata
    return output


def run_official_ppocr_batch(
    batch_df: Any,
    *,
    invoke_url: str | None,
    request_timeout_s: float = 180.0,
    extract_text: bool = True,
    extract_tables: bool = True,
    extract_charts: bool = True,
    extract_images: bool = True,
    extract_infographics: bool = True,
    transport: Callable[[str, Mapping[str, Any], float], Any] = _http_post_json,
) -> Any:
    """Run the official whole-page PP-OCRv6 pipeline with no fallback."""

    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df
    endpoint = str(invoke_url or "").strip()
    if not endpoint:
        raise RuntimeError("pipeline-ppocrv6 (Option 2) requires official_ppocr_invoke_url")

    output_rows: list[dict[str, Any]] = []
    for _, series in batch_df.iterrows():
        row = series.to_dict()
        started = time.perf_counter()
        image = row.get("page_image") if isinstance(row.get("page_image"), Mapping) else {}
        image_b64 = str(image.get("image_b64") or "")
        try:
            if not image_b64:
                raise PPOCROfficialServiceError("page image is unavailable")
            response = transport(endpoint, {"images": [image_b64]}, float(request_timeout_s))
            parsed = _normalise_page(row, response)
            parsed["ocr"]["timing"] = {"seconds": time.perf_counter() - started}
            parsed["metadata"]["ocr_timing"] = dict(parsed["ocr"]["timing"])
            output_rows.append(parsed)
        except BaseException as exc:
            failed = _failed_page(row, exc)
            failed["ocr"]["timing"] = {"seconds": time.perf_counter() - started}
            failed["metadata"]["ocr_timing"] = dict(failed["ocr"]["timing"])
            output_rows.append(failed)
    return pd.DataFrame(output_rows).reset_index(drop=True)


__all__ = ["PIPELINE_NAME", "OPTION2_SELECTOR", "run_official_ppocr_batch"]
