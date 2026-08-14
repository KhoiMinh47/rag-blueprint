"""Small HTTP wrapper exposing PP-OCRv6 detector and recognizer separately."""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image


def _image(value: str) -> np.ndarray:
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    try:
        with Image.open(io.BytesIO(base64.b64decode(value))) as source:
            return np.asarray(source.convert("RGB"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc


def _images(payload: dict[str, Any]) -> list[np.ndarray]:
    values = payload.get("images") or payload.get("input") or payload.get("data") or []
    if isinstance(values, dict):
        values = [values]
    result: list[np.ndarray] = []
    for item in values:
        if isinstance(item, str):
            result.append(_image(item)); continue
        if not isinstance(item, dict):
            continue
        value = item.get("image_b64") or item.get("image") or item.get("url")
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            value = image_url.get("url") or value
        if isinstance(value, str):
            result.append(_image(value))
    return result


def _result_json(value: Any) -> dict[str, Any]:
    if hasattr(value, "json"):
        value = value.json
    if isinstance(value, str):
        import json
        value = json.loads(value)
    if isinstance(value, dict):
        return value.get("res", value)
    return {}


def _as_values(value: Any) -> list[Any]:
    """Normalize Paddle's scalar and batched result fields to a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _xyxy(box: Any) -> list[float] | None:
    if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
        return [float(v) for v in box]
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        points = [(float(p[0]), float(p[1])) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
        if points:
            return [min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points)]
    return None


def _normalized_xyxy(box: Any, *, width: int, height: int) -> list[float] | None:
    """Return detector coordinates in the normalized schema used downstream.

    PaddleOCR returns polygon coordinates in source-image pixels.  The
    Retriever OCR adapter consumes ``bbox_xyxy_norm`` values in the ``0..1``
    range, so normalize here at the HTTP boundary.
    """
    bbox = _xyxy(box)
    if bbox is None:
        return None
    # Keep already-normalized responses compatible with alternate Paddle
    # backends while converting the normal pixel-coordinate response.
    if max(abs(value) for value in bbox) <= 1.5:
        return [max(0.0, min(1.0, value)) for value in bbox]
    image_width = max(1, int(width))
    image_height = max(1, int(height))
    return [
        max(0.0, min(1.0, bbox[0] / image_width)),
        max(0.0, min(1.0, bbox[1] / image_height)),
        max(0.0, min(1.0, bbox[2] / image_width)),
        max(0.0, min(1.0, bbox[3] / image_height)),
    ]


def _detector_output(
    image: np.ndarray,
    result: Any,
    *,
    content_shape_hw: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Normalize one Paddle detector result for the Retriever contract."""
    raw = _result_json(result)
    boxes = raw.get("dt_polys") or raw.get("rec_boxes") or raw.get("boxes") or []
    scores = raw.get("dt_scores") or raw.get("scores") or []
    height, width = image.shape[:2]
    content_height, content_width = content_shape_hw or (height, width)
    normalized_boxes = []
    for index, item in enumerate(boxes):
        raw_box = _xyxy(item)
        if raw_box is None:
            continue
        # Batch padding is added only on the right/bottom. Remove that
        # artificial area before normalizing back to the original crop.
        if max(abs(value) for value in raw_box) > 1.5 and content_shape_hw:
            if (
                raw_box[2] <= 0.0
                or raw_box[3] <= 0.0
                or raw_box[0] >= content_width
                or raw_box[1] >= content_height
            ):
                continue
            raw_box = [
                max(0.0, min(float(content_width), raw_box[0])),
                max(0.0, min(float(content_height), raw_box[1])),
                max(0.0, min(float(content_width), raw_box[2])),
                max(0.0, min(float(content_height), raw_box[3])),
            ]
        box = _normalized_xyxy(
            raw_box,
            width=content_width,
            height=content_height,
        )
        if box is not None:
            normalized_boxes.append(
                {
                    "bbox": box,
                    "score": float(scores[index]) if index < len(scores) else 1.0,
                }
            )
    return {
        "boxes": normalized_boxes,
        "model": os.getenv("PPOCR_DET_MODEL", "PP-OCRv6_medium_det"),
    }


def _configured_detector_batch_size(image_count: int) -> int:
    try:
        configured = int(os.getenv("PPOCR_DET_BATCH_SIZE", "16") or 16)
    except (TypeError, ValueError):
        configured = 16
    return max(1, min(max(1, image_count), configured))


def _pad_detector_batch(
    images: list[np.ndarray],
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Pad one shape-compatible Paddle batch without resizing its content."""
    if not images:
        return [], []
    target_height = max(int(image.shape[0]) for image in images)
    target_width = max(int(image.shape[1]) for image in images)
    padded: list[np.ndarray] = []
    original_shapes: list[tuple[int, int]] = []
    for image in images:
        height, width = int(image.shape[0]), int(image.shape[1])
        original_shapes.append((height, width))
        if height == target_height and width == target_width:
            padded.append(image)
            continue
        # Match the crop's border color instead of inserting a black frame;
        # this avoids creating a synthetic high-contrast edge for the detector.
        border = np.concatenate(
            (image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :]),
            axis=0,
        )
        fill = np.median(border, axis=0).astype(image.dtype)
        canvas = np.empty(
            (target_height, target_width, image.shape[2]),
            dtype=image.dtype,
        )
        canvas[...] = fill
        canvas[:height, :width] = image
        padded.append(canvas)
    return padded, original_shapes


class ModelState:
    def __init__(self) -> None:
        from paddleocr import TextDetection, TextRecognition
        device = os.getenv("PPOCR_DEVICE", "gpu:0")
        self.role = os.getenv("PPOCR_ROLE", "").strip().lower()
        if self.role not in {"detector", "recognizer"}:
            raise RuntimeError("PPOCR_ROLE must be detector or recognizer")
        self.detector = None
        self.recognizer = None
        if self.role == "detector":
            self.detector = TextDetection(
                model_name=os.getenv("PPOCR_DET_MODEL", "PP-OCRv6_medium_det"),
                device=device,
            )
        else:
            self.recognizer = TextRecognition(
                model_name=os.getenv("PPOCR_REC_MODEL", "PP-OCRv6_medium_rec"),
                device=device,
            )


state = ModelState()
app = FastAPI(title="PP-OCRv6 sidecar", version="1.0")


@app.get("/v1/health/ready")
def ready() -> dict[str, Any]:
    return {"ready": True, "role": state.role, "model": os.getenv("PPOCR_DET_MODEL" if state.role == "detector" else "PPOCR_REC_MODEL")}


@app.post("/v1/detect")
def detect(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if state.role != "detector" or state.detector is None:
        raise HTTPException(status_code=404, detail="detector endpoint is disabled on this container")
    outputs: list[dict[str, Any]] = []
    for image in _images(payload):
        outputs.append(
            _detector_output(image, next(iter(state.detector.predict(image))))
        )
    return outputs


@app.post("/v1/detect-batch")
def detect_batch(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Run true Paddle tensor batching for Option 2 line crops.

    ``/v1/detect`` intentionally keeps its historical one-image behavior for
    other callers.  This route passes the complete image list to PaddleX in
    one predictor call, so preprocessing, GPU inference, and postprocessing
    can use the configured batch size instead of looping over HTTP items.
    """
    if state.role != "detector" or state.detector is None:
        raise HTTPException(status_code=404, detail="detector endpoint is disabled on this container")
    images = _images(payload)
    if not images:
        return []
    batch_size = _configured_detector_batch_size(len(images))
    # Similar aspect/scale crops are adjacent, which keeps the right/bottom
    # padding small and protects detector resolution while retaining tensor
    # batching. The result slots are restored to request order below.
    order = sorted(
        range(len(images)),
        key=lambda index: (
            images[index].shape[1] / max(1, images[index].shape[0]),
            images[index].shape[0] * images[index].shape[1],
        ),
    )
    output_slots: list[dict[str, Any] | None] = [None] * len(images)
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        source_batch = [images[index] for index in indices]
        padded_batch, original_shapes = _pad_detector_batch(source_batch)
        results = list(
            state.detector.predict(
                padded_batch,
                batch_size=len(padded_batch),
            )
        )
        if len(results) != len(indices):
            raise HTTPException(
                status_code=502,
                detail=(
                    f"detector returned {len(results)} results for "
                    f"{len(indices)} images"
                ),
            )
        for index, image, original_shape, result in zip(
            indices,
            source_batch,
            original_shapes,
            results,
        ):
            output_slots[index] = _detector_output(
                image,
                result,
                content_shape_hw=original_shape,
            )
    return [output for output in output_slots if output is not None]


@app.post("/v1/recognize")
def recognize(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if state.role != "recognizer" or state.recognizer is None:
        raise HTTPException(status_code=404, detail="recognizer endpoint is disabled on this container")
    outputs: list[dict[str, Any]] = []
    for image in _images(payload):
        raw = _result_json(next(iter(state.recognizer.predict(image))))
        texts = _as_values(raw.get("rec_texts"))
        if not texts:
            texts = _as_values(raw.get("texts"))
        if not texts:
            texts = _as_values(raw.get("rec_text"))
        if not texts:
            texts = _as_values(raw.get("text"))

        scores = _as_values(raw.get("rec_scores"))
        if not scores:
            scores = _as_values(raw.get("scores"))
        if not scores:
            scores = _as_values(raw.get("rec_score"))
        if not scores:
            scores = _as_values(raw.get("score"))

        score_value = scores[0] if scores else None
        try:
            score_value = float(score_value) if score_value is not None else None
        except (TypeError, ValueError):
            score_value = None
        outputs.append({"text": str(texts[0]) if texts else "", "score": score_value, "model": os.getenv("PPOCR_REC_MODEL", "PP-OCRv6_medium_rec")})
    return outputs
