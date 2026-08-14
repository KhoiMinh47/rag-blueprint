# SPDX-License-Identifier: Apache-2.0

"""HTTP wrapper for the official PaddleOCR PP-OCRv6 general OCR pipeline.

This service intentionally owns one ``PaddleOCR`` pipeline.  It is separate
from the split detector/recognizer sidecars used by the other experimental
options so that option 2 follows PaddleOCR's published order exactly:
document orientation, unwarping, text-line orientation, detection, and
recognition.
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image


def _decode_image(value: str) -> np.ndarray:
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    try:
        with Image.open(io.BytesIO(base64.b64decode(value))) as source:
            return np.asarray(source.convert("RGB"))
    except Exception as exc:  # pragma: no cover - exercised through HTTP
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc


def _images(payload: dict[str, Any]) -> list[np.ndarray]:
    values = payload.get("images") or payload.get("input") or payload.get("data") or []
    if isinstance(values, (str, dict)):
        values = [values]
    result: list[np.ndarray] = []
    for item in values:
        if isinstance(item, str):
            result.append(_decode_image(item))
            continue
        if not isinstance(item, dict):
            continue
        value = (
            item.get("image_b64")
            or item.get("image_base64")
            or item.get("image")
            or item.get("url")
        )
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            value = image_url.get("url") or value
        if isinstance(value, str):
            result.append(_decode_image(value))
    return result


def _result_json(value: Any) -> dict[str, Any]:
    payload = value.json if hasattr(value, "json") else value
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict):
        nested = payload.get("res")
        return nested if isinstance(nested, dict) else payload
    return {}


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bbox(value: Any, *, width: int, height: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        points = [(float(value[0]), float(value[1])), (float(value[2]), float(value[3]))]
    else:
        points: list[tuple[float, float]] = []
        for point in value:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    points.append((float(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    continue
    if not points:
        return None
    x0 = max(0.0, min(1.0, min(point[0] for point in points) / max(width, 1)))
    y0 = max(0.0, min(1.0, min(point[1] for point in points) / max(height, 1)))
    x1 = max(0.0, min(1.0, max(point[0] for point in points) / max(width, 1)))
    y1 = max(0.0, min(1.0, max(point[1] for point in points) / max(height, 1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _preprocess_trace(raw: dict[str, Any]) -> dict[str, Any]:
    value = raw.get("doc_preprocessor_res") or raw.get("docPreprocessorRes") or {}
    if not isinstance(value, dict):
        return {}
    trace: dict[str, Any] = {}
    for key in ("angle", "doc_preprocessor_res", "model_settings"):
        item = value.get(key)
        if isinstance(item, (str, int, float, bool, type(None), dict, list)):
            trace[key] = item
    return trace


class ModelState:
    def __init__(self) -> None:
        from paddleocr import PaddleOCR

        self.device = os.getenv("PPOCR_OFFICIAL_DEVICE", "gpu:0")
        self.lang = os.getenv("PPOCR_OFFICIAL_LANG", "vi")
        self.model_names = {
            "doc_orientation": os.getenv(
                "PPOCR_DOC_ORI_MODEL", "PP-LCNet_x1_0_doc_ori"
            ),
            "doc_unwarping": os.getenv("PPOCR_DOC_UNWARP_MODEL", "UVDoc"),
            "textline_orientation": os.getenv(
                "PPOCR_TEXTLINE_ORI_MODEL", "PP-LCNet_x1_0_textline_ori"
            ),
            "text_detection": os.getenv(
                "PPOCR_DET_MODEL", "PP-OCRv6_medium_det"
            ),
            "text_recognition": os.getenv(
                "PPOCR_REC_MODEL", "PP-OCRv6_medium_rec"
            ),
        }
        self.pipeline = PaddleOCR(
            ocr_version="PP-OCRv6",
            lang=self.lang,
            device=self.device,
            doc_orientation_classify_model_name=self.model_names["doc_orientation"],
            doc_unwarping_model_name=self.model_names["doc_unwarping"],
            textline_orientation_model_name=self.model_names["textline_orientation"],
            text_detection_model_name=self.model_names["text_detection"],
            text_recognition_model_name=self.model_names["text_recognition"],
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
        )


state = ModelState()
app = FastAPI(title="PP-OCRv6 official pipeline", version="1.0")


@app.get("/v1/health/ready")
def ready() -> dict[str, Any]:
    return {
        "ready": True,
        "pipeline": "PP-OCRv6 official general OCR",
        "device": state.device,
        "lang": state.lang,
        "models": state.model_names,
    }


@app.post("/v1/ocr")
def ocr(payload: dict[str, Any]) -> list[dict[str, Any]]:
    images = _images(payload)
    if not images:
        raise HTTPException(
            status_code=400,
            detail="images must contain at least one base64 image",
        )
    outputs: list[dict[str, Any]] = []
    for image in images:
        try:
            result = next(iter(state.pipeline.predict(image)))
            raw = _result_json(result)
            height, width = image.shape[:2]
            boxes = _list(raw.get("rec_boxes") or raw.get("dt_polys") or raw.get("rec_polys"))
            texts = _list(raw.get("rec_texts"))
            rec_scores = _list(raw.get("rec_scores"))
            det_scores = _list(raw.get("dt_scores"))
            angles = _list(raw.get("textline_orientation_angles"))
            lines: list[dict[str, Any]] = []
            for index, box in enumerate(boxes):
                bbox = _bbox(box, width=width, height=height)
                text = str(texts[index]).strip() if index < len(texts) else ""
                if bbox is None or not text:
                    continue
                lines.append(
                    {
                        "text": text,
                        "bbox": bbox,
                        "score": _number(rec_scores[index]) if index < len(rec_scores) else None,
                        "detector_score": _number(det_scores[index]) if index < len(det_scores) else None,
                        "line_angle": _number(angles[index]) if index < len(angles) else 0.0,
                    }
                )
            outputs.append(
                {
                    "pipeline": "PP-OCRv6 official general OCR",
                    "models": state.model_names,
                    "preprocess": _preprocess_trace(raw),
                    "lines": lines,
                    "raw_counts": {
                        "detected_boxes": len(boxes),
                        "recognized_lines": len(lines),
                    },
                }
            )
        except Exception as exc:  # pragma: no cover - exercised through HTTP
            raise HTTPException(status_code=502, detail=f"PP-OCRv6 inference failed: {exc}") from exc
    return outputs
