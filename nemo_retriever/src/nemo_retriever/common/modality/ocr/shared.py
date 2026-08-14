# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Document OCR helpers and result-schema normalization.

The primary production path is the original integrated Nemotron OCR v2
pipeline. A split PP-OCRv6 detector/recognizer pair remains available as an
explicit fallback when no Nemotron OCR endpoint is configured.
"""

from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

import base64
import io
import logging
import re
import time
import traceback
import unicodedata

_logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from nemo_retriever.common.params import RemoteRetryParams
from nemo_retriever.models.nim.nim import NIMClient, invoke_image_inference_batches
from nemo_retriever.common.modality.table_and_chart import join_table_structure_and_ocr_output

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageEnhance = None  # type: ignore[assignment]
    ImageFilter = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Page-element labels that carry running text (as opposed to structured
# content like tables/charts/infographics).  Used by the OCR stage to
# decide which detections contribute to the page's ``text`` column.
_TEXT_LABELS: frozenset[str] = frozenset({"text", "title", "header_footer"})

# OCR is run on the page, detector crops, and (for scans) overlapping tiles.
# Those passes often produce slightly different text and coordinates for the
# same line.  These thresholds are intentionally conservative: adjacent lines
# must still have distinct geometry, while small coordinate drift and OCR typos
# are treated as one block.
_OCR_DUPLICATE_TEXT_SIMILARITY = 0.78
_OCR_DUPLICATE_CONTAINMENT = 0.80
_OCR_GEOMETRIC_DUPLICATE_CONTAINMENT = 0.88
_OCR_GEOMETRIC_DUPLICATE_IOU = 0.72
_OCR_DUPLICATE_EDGE_TOLERANCE = 0.008

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_payload(*, stage: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "timing": None,
        "error": {
            "stage": str(stage),
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
    }


def _crop_b64_image_by_norm_bbox(
    page_image_b64: str,
    *,
    bbox_xyxy_norm: Sequence[float],
    image_format: str = "png",
) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    """
    Crop a base64-encoded RGB image by a normalized xyxy bbox.

    Returns
    -------
    cropped_image_b64 : str | None
        Base64-encoded cropped image (PNG), or *None* on failure.
    cropped_shape_hw : tuple[int, int] | None
        (H, W) of the crop, or *None* on failure.
    """
    if Image is None:  # pragma: no cover
        raise ImportError("Cropping requires pillow.")

    if not isinstance(page_image_b64, str) or not page_image_b64:
        return None, None
    try:
        x1n, y1n, x2n, y2n = [float(x) for x in bbox_xyxy_norm]
    except Exception:
        return None, None

    try:
        raw = base64.b64decode(page_image_b64)
        with Image.open(io.BytesIO(raw)) as im0:
            im = im0.convert("RGB")
            w, h = im.size
            if w <= 1 or h <= 1:
                return None, None

            def _clamp_int(v: float, lo: int, hi: int) -> int:
                if v != v:  # NaN
                    return lo
                # Truncating a normalized detector edge can remove a stroke
                # from a small title/line crop. Round to the nearest source
                # pixel so the crop follows the detector geometry more
                # faithfully before OCR sees it.
                return int(round(min(max(v, float(lo)), float(hi))))

            x1 = _clamp_int(x1n * w, 0, w)
            x2 = _clamp_int(x2n * w, 0, w)
            y1 = _clamp_int(y1n * h, 0, h)
            y2 = _clamp_int(y2n * h, 0, h)

            if x2 <= x1 or y2 <= y1:
                return None, None

            crop = im.crop((x1, y1, x2, y2))
            cw, ch = crop.size
            if cw <= 1 or ch <= 1:
                return None, None

            buf = io.BytesIO()
            fmt = str(image_format or "png").lower()
            if fmt not in {"png"}:
                fmt = "png"
            crop.save(buf, format=fmt.upper())
            return base64.b64encode(buf.getvalue()).decode("ascii"), (int(ch), int(cw))
    except Exception:
        return None, None


def _crop_all_from_page(
    page_image_b64: str,
    detections: List[Dict[str, Any]],
    wanted_labels: set,
    *,
    as_b64: bool = False,
) -> List[Tuple[str, List[float], Any]]:
    """
    Decode the page image **once** and crop all matching detections.

    Returns a list of ``(label_name, bbox_xyxy_norm, value)`` tuples for
    detections whose ``label_name`` is in *wanted_labels* and whose crop is
    valid.  Skips detections that fail to crop (bad bbox, tiny region, etc.).

    When *as_b64* is ``False`` (default), *value* is an HWC uint8 numpy array
    suitable for local model inference.  When ``True``, *value* is a base64-
    encoded PNG string — this avoids a wasteful numpy→PIL→PNG round-trip on
    the remote inference path.
    """
    if Image is None:  # pragma: no cover
        raise ImportError("Cropping requires pillow.")

    if not isinstance(page_image_b64, str) or not page_image_b64:
        return []

    try:
        raw = base64.b64decode(page_image_b64)
        im0 = Image.open(io.BytesIO(raw))
        im = im0.convert("RGB")
        im0.close()
    except Exception:
        return []

    w, h = im.size
    if w <= 1 or h <= 1:
        im.close()
        return []

    def _clamp_int(v: float, lo: int, hi: int) -> int:
        if v != v:  # NaN
            return lo
        # Keep the nearest pixel at both edges; ``int`` truncation was
        # dropping thin glyph strokes from small OCR crops.
        return int(round(min(max(v, float(lo)), float(hi))))

    results: List[Tuple[str, List[float], Any]] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        label_name = str(det.get("label_name") or "").strip()
        if label_name not in wanted_labels:
            continue

        bbox = det.get("bbox_xyxy_norm")
        crop_bbox = det.get("crop_bbox_xyxy_norm") or bbox
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        if not isinstance(crop_bbox, (list, tuple)) or len(crop_bbox) != 4:
            continue

        try:
            x1n, y1n, x2n, y2n = [float(x) for x in crop_bbox]
        except Exception:
            continue

        x1 = _clamp_int(x1n * w, 0, w)
        x2 = _clamp_int(x2n * w, 0, w)
        y1 = _clamp_int(y1n * h, 0, h)
        y2 = _clamp_int(y2n * h, 0, h)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = im.crop((x1, y1, x2, y2))
        cw, ch = crop.size
        if cw <= 1 or ch <= 1:
            crop.close()
            continue

        if as_b64:
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            crop.close()
            value = base64.b64encode(buf.getvalue()).decode("ascii")
        else:
            value = np.asarray(crop, dtype=np.uint8).copy()
            crop.close()
        # Return the semantic/model bbox for coordinate mapping and output;
        # only the small padded crop bbox controls pixels sent to OCR.
        results.append((label_name, [float(x) for x in bbox], value))

    im.close()
    return results


def _expanded_bbox(bbox: Sequence[float], padding: float = 0.01) -> List[float]:
    """Add a small normalized margin so OCR does not clip glyph edges."""
    values = [float(value) for value in bbox[:4]]
    margin = max(0.0, float(padding))
    return [
        max(0.0, values[0] - margin),
        max(0.0, values[1] - margin),
        min(1.0, values[2] + margin),
        min(1.0, values[3] + margin),
    ]


def _make_ocr_refinement_crop(
    image_b64: str,
    bbox: Sequence[float],
    *,
    padding: float = 0.01,
    target_height: int = 96,
) -> Tuple[List[float], Optional[str]]:
    """Make a larger line crop for a low-quality full-page OCR block."""
    crop_bbox = _expanded_bbox(bbox, padding=padding)
    crop_b64, _shape = _crop_b64_image_by_norm_bbox(image_b64, bbox_xyxy_norm=crop_bbox)
    if not crop_b64 or Image is None:
        return crop_bbox, crop_b64
    try:
        raw = base64.b64decode(crop_b64)
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGB")
            width, height = image.size
            scale = min(4.0, max(1.0, float(target_height) / max(1, height)))
            if scale <= 1.0:
                return crop_bbox, crop_b64
            new_width = max(1, int(round(width * scale)))
            new_height = max(1, int(round(height * scale)))
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return crop_bbox, base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return crop_bbox, crop_b64


def _np_rgb_to_b64_png(crop_array: np.ndarray) -> str:
    if Image is None:  # pragma: no cover
        raise ImportError("Pillow is required for image encoding.")
    img = Image.fromarray(crop_array.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_remote_ocr_item(response_item: Any) -> Any:
    if isinstance(response_item, dict):
        # NIM text_detections format: return full list (not v[0])
        td = response_item.get("text_detections")
        if isinstance(td, list) and td:
            return td
        for k in ("prediction", "predictions", "output", "outputs", "data"):
            v = response_item.get(k)
            if isinstance(v, list) and v:
                return v[0]
            if v is not None:
                return v
    return response_item


def _bbox_from_points(points: Any) -> Optional[List[float]]:
    """Convert OCR quadrilateral points to normalized xyxy coordinates."""
    if not isinstance(points, list) or not points:
        return None
    coords: List[Tuple[float, float]] = []
    for point in points:
        if isinstance(point, dict):
            x, y = point.get("x"), point.get("y")
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[0], point[1]
        else:
            continue
        try:
            coords.append((float(x), float(y)))
        except (TypeError, ValueError):
            continue
    if not coords:
        return None
    x0 = min(x for x, _ in coords)
    y0 = min(y for _, y in coords)
    x1 = max(x for x, _ in coords)
    y1 = max(y for _, y in coords)
    # The recognizer returns normalized coordinates. Keep malformed pixel-space
    # values out of the public bbox contract rather than inventing a scale.
    if any(value < 0.0 or value > 1.0 for value in (x0, y0, x1, y1)):
        return None
    return [x0, y0, x1, y1]


def _bbox_from_value(value: Any) -> Optional[List[float]]:
    """Convert common OCR bbox forms to normalized xyxy coordinates."""
    if not isinstance(value, list):
        return None
    if len(value) == 4 and all(isinstance(item, (list, tuple)) and len(item) >= 2 for item in value):
        return _bbox_from_points(value)
    if len(value) != 4:
        return None
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if any(item < 0.0 or item > 1.0 for item in box):
        return None
    return [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]


def _confidence_value(value: Any) -> Optional[float]:
    """Read a model confidence from the common OCR response shapes."""
    candidate = value
    if isinstance(value, dict):
        for key in ("confidence", "score", "conf", "probability"):
            if value.get(key) is not None:
                candidate = value.get(key)
                break
    try:
        confidence = float(candidate)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= confidence <= 1.0:
        return None
    return confidence


def _parse_ocr_result(preds: Any) -> List[Dict[str, Any]]:
    """
    Parse a recognizer response into a flat list of
    ``{"text": str, "sort_y": float, "sort_x": float}`` blocks.

    The model may return:
    * ``dict`` with ``boxes`` / ``texts`` keys (packed form), or
    * ``list[dict]`` with ``left``/``right``/``upper``/``lower``/``text`` keys
      (normalized-coordinate form), or
    * ``list[dict]`` with generic ``text`` + ``box``/``bbox`` keys.
    """
    blocks: List[Dict[str, Any]] = []

    # ---- dict form: {"boxes": [...], "texts": [...]} ----
    if isinstance(preds, dict):
        pb = next((preds.get(key) for key in ("boxes", "bboxes", "bounding_boxes") if preds.get(key) is not None), None)
        pt = next((preds.get(key) for key in ("texts", "text_predictions", "text") if preds.get(key) is not None), None)
        pc = next((preds.get(key) for key in ("confidences", "scores") if preds.get(key) is not None), None)
        if isinstance(pb, np.ndarray):
            pb = pb.tolist()
        if isinstance(pt, np.ndarray):
            pt = pt.tolist()
        if isinstance(pc, np.ndarray):
            pc = pc.tolist()
        if isinstance(pb, list) and isinstance(pt, list):
            for index, (b, txt) in enumerate(zip(pb, pt)):
                txt_value = txt.get("text") if isinstance(txt, dict) else txt
                if not isinstance(txt_value, str) or not txt_value.strip():
                    continue
                sort_y, sort_x = 0.0, 0.0
                if isinstance(b, list):
                    if len(b) == 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in b):
                        # quadrilateral [[x1,y1], ...]
                        sort_y = float(b[0][1])
                        sort_x = float(b[0][0])
                    elif len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
                        # xyxy [x1, y1, x2, y2]
                        sort_y = float(b[1])
                        sort_x = float(b[0])
                block = {"text": txt_value.strip(), "sort_y": sort_y, "sort_x": sort_x}
                confidence = _confidence_value(txt)
                if confidence is None and isinstance(pc, list) and index < len(pc):
                    confidence = _confidence_value(pc[index])
                if confidence is not None:
                    block["confidence"] = confidence
                bbox = _bbox_from_value(b)
                if bbox is not None:
                    block["bbox_xyxy_norm"] = bbox
                blocks.append(block)
        return blocks

    # ---- list form: list[dict] with various key conventions ----
    if isinstance(preds, list):
        for item in preds:
            if isinstance(item, str):
                if item.strip():
                    blocks.append({"text": item.strip(), "sort_y": 0.0, "sort_x": 0.0})
                continue
            if not isinstance(item, dict):
                continue

            # NIM text_detections format:
            # {"text_prediction": {"text": "...", "confidence": ...},
            #  "bounding_box": {"points": [{"x": ..., "y": ...}, ...]}}
            tp = item.get("text_prediction")
            if isinstance(tp, dict):
                txt0 = str(tp.get("text") or "").strip()
                if txt0 and txt0 != "nan":
                    sort_y, sort_x = 0.0, 0.0
                    bb = item.get("bounding_box")
                    pts = None
                    if isinstance(bb, dict):
                        pts = bb.get("points")
                        if isinstance(pts, list) and pts:
                            try:
                                sort_x = float(pts[0].get("x", 0.0))
                                sort_y = float(pts[0].get("y", 0.0))
                            except Exception:
                                pass
                    block = {"text": txt0, "sort_y": sort_y, "sort_x": sort_x}
                    confidence = _confidence_value(tp)
                    if confidence is not None:
                        block["confidence"] = confidence
                    bbox = _bbox_from_points(pts)
                    if bbox is not None:
                        block["bbox_xyxy_norm"] = bbox
                    blocks.append(block)
                continue

            # Normalized-coordinate form
            if all(k in item for k in ("left", "right", "upper", "lower")) and isinstance(item.get("text"), str):
                txt0 = str(item.get("text") or "").strip()
                if not txt0 or txt0 == "nan":
                    continue
                try:
                    sort_x = float(item["left"])
                    sort_y = float(item["lower"])
                except Exception:
                    sort_x, sort_y = 0.0, 0.0
                block = {"text": txt0, "sort_y": sort_y, "sort_x": sort_x}
                confidence = _confidence_value(item)
                if confidence is not None:
                    block["confidence"] = confidence
                bbox = _bbox_from_value([item.get("left"), item.get("upper"), item.get("right"), item.get("lower")])
                if bbox is not None:
                    block["bbox_xyxy_norm"] = bbox
                blocks.append(block)
                continue

            # Generic text + box fallback
            txt = item.get("text") or item.get("ocr_text") or item.get("generated_text") or item.get("output_text")
            if not isinstance(txt, str) or not txt.strip():
                continue
            sort_y, sort_x = 0.0, 0.0
            b = item.get("box") or item.get("bbox") or item.get("bounding_box") or item.get("bbox_points")
            if isinstance(b, list):
                if len(b) == 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in b):
                    sort_y = float(b[0][1])
                    sort_x = float(b[0][0])
                elif len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
                    sort_y = float(b[1])
                    sort_x = float(b[0])
            block = {"text": txt.strip(), "sort_y": sort_y, "sort_x": sort_x}
            confidence = _confidence_value(item)
            if confidence is not None:
                block["confidence"] = confidence
            bbox = _bbox_from_value(b)
            if bbox is not None:
                block["bbox_xyxy_norm"] = bbox
            blocks.append(block)

    # ---- last-resort stringify ----
    if not blocks and preds is not None:
        s = ""
        try:
            s = str(preds).strip()
        except Exception:
            s = ""
        if s and s.lower() not in {"none", "null", "[]", "{}"}:
            # ``_fallback`` lets ``ocr_response_to_text`` suppress this row
            # without false-positives on legitimate OCR text that happens to
            # start with ``[`` or ``{``.
            blocks.append({"text": s, "sort_y": 0.0, "sort_x": 0.0, "_fallback": True})

    return blocks


def _blocks_to_text(blocks: List[Dict[str, Any]]) -> str:
    """Sort text blocks by reading order (y then x) and join with whitespace."""
    blocks.sort(key=lambda b: (b.get("sort_y", 0.0), b.get("sort_x", 0.0)))
    return " ".join(b["text"] for b in blocks if b.get("text"))


def _map_bbox_from_crop(
    crop_bbox: Sequence[float],
    local_bbox: Sequence[float] | None,
) -> List[float]:
    """Map a crop-local normalized bbox back to full-page coordinates."""
    crop = [float(value) for value in crop_bbox[:4]]
    if not isinstance(local_bbox, (list, tuple)) or len(local_bbox) != 4:
        return crop
    local = [float(value) for value in local_bbox]
    x0, y0, x1, y1 = crop
    return [
        x0 + local[0] * (x1 - x0),
        y0 + local[1] * (y1 - y0),
        x0 + local[2] * (x1 - x0),
        y0 + local[3] * (y1 - y0),
    ]


def _map_ocr_blocks_to_page(
    blocks: Sequence[Dict[str, Any]],
    crop_bbox: Sequence[float],
) -> List[Dict[str, Any]]:
    """Attach full-page bboxes to OCR blocks returned for one crop."""
    mapped: List[Dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        out = dict(block)
        out["bbox_xyxy_norm"] = _map_bbox_from_crop(crop_bbox, block.get("bbox_xyxy_norm"))
        out["sort_y"] = float(out["bbox_xyxy_norm"][1])
        out["sort_x"] = float(out["bbox_xyxy_norm"][0])
        out.setdefault("ocr_source", "page_elements_crop")
        mapped.append(out)
    return mapped


def _normalise_ocr_text(value: Any) -> str:
    """Return an accent/punctuation-insensitive key for duplicate matching."""
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    # Vietnamese đ/Đ does not decompose under NFKD like the other accented
    # characters, but it should compare as the same base character here.
    text = text.replace("đ", "d")
    return "".join(character for character in text if not unicodedata.combining(character) and character.isalnum())


def _ocr_texts_match(left: Any, right: Any) -> bool:
    """Match exact OCR text plus small recognition variants."""
    left_key = _normalise_ocr_text(left)
    right_key = _normalise_ocr_text(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    shorter, longer = sorted((left_key, right_key), key=len)
    if len(shorter) >= 4 and shorter in longer:
        return True
    if len(shorter) < 8:
        return False
    return SequenceMatcher(None, left_key, right_key).ratio() >= _OCR_DUPLICATE_TEXT_SIMILARITY


def _ocr_text_identity(value: Any) -> str:
    """Keep a small identity key for punctuation-only OCR blocks."""
    return " ".join(str(value or "").casefold().split())


def _ocr_bbox_edges_close(left: Any, right: Any) -> bool:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    if len(left) != 4 or len(right) != 4:
        return False
    try:
        left_values = [float(value) for value in left]
        right_values = [float(value) for value in right]
    except (TypeError, ValueError):
        return False
    return max(abs(a - b) for a, b in zip(left_values, right_values)) <= _OCR_DUPLICATE_EDGE_TOLERANCE


def _ocr_bbox_shapes_similar(left: Any, right: Any) -> bool:
    """Reject paragraph-vs-line containment while accepting crop drift."""
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    if len(left) != 4 or len(right) != 4:
        return False
    try:
        left_width = max(0.0, float(left[2]) - float(left[0]))
        right_width = max(0.0, float(right[2]) - float(right[0]))
        left_height = max(0.0, float(left[3]) - float(left[1]))
        right_height = max(0.0, float(right[3]) - float(right[1]))
    except (TypeError, ValueError):
        return False
    if min(left_width, right_width) <= 0.0 or min(left_height, right_height) <= 0.0:
        return False
    return (
        min(left_width, right_width) / max(left_width, right_width) >= 0.65
        and min(left_height, right_height) / max(left_height, right_height) >= 0.55
    )


def _ocr_bbox_geometry_is_duplicate(left: Any, right: Any) -> bool:
    """Detect nested detector boxes even when OCR strings disagree."""
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    if len(left) != 4 or len(right) != 4:
        return False
    try:
        left_values = [float(value) for value in left]
        right_values = [float(value) for value in right]
    except (TypeError, ValueError):
        return False
    left_area = max(0.0, left_values[2] - left_values[0]) * max(0.0, left_values[3] - left_values[1])
    right_area = max(0.0, right_values[2] - right_values[0]) * max(0.0, right_values[3] - right_values[1])
    smaller = min(left_area, right_area)
    if smaller <= 0.0:
        return False
    intersection = max(0.0, min(left_values[2], right_values[2]) - max(left_values[0], right_values[0])) * max(
        0.0, min(left_values[3], right_values[3]) - max(left_values[1], right_values[1])
    )
    union = left_area + right_area - intersection
    return (
        intersection / smaller >= _OCR_GEOMETRIC_DUPLICATE_CONTAINMENT
        or (union > 0.0 and intersection / union >= _OCR_GEOMETRIC_DUPLICATE_IOU)
    )


def _ocr_text_quality(value: Any) -> float:
    """Score OCR readability without pretending to be a spell checker."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not text:
        return 0.0
    visible = [character for character in text if not character.isspace()]
    letters = [character for character in visible if character.isalpha()]
    digits = [character for character in visible if character.isdigit()]
    words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    repeated = sum(max(0, len(match.group(0)) - 2) for match in re.finditer(r"(.)\1{2,}", text.casefold()))
    replacement_glyphs = sum(character in "�∅π" for character in visible)
    single_letter_words = sum(len(word) == 1 for word in words)
    if not visible:
        return 0.0
    score = (
        0.42 * len(letters) / len(visible)
        + 0.18 * min(1.0, len(letters) / 24.0)
        + 0.16 * min(1.0, len(words) / 6.0)
        - 0.08 * len(digits) / len(visible)
        - 0.06 * min(1.0, single_letter_words / max(1, len(words)))
        - 0.12 * min(1.0, repeated / 5.0)
        - 0.22 * replacement_glyphs / len(visible)
    )
    return max(0.0, min(1.0, score))


def _bbox_iou_norm(left: Any, right: Any) -> float:
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return 0.0
    if len(left) != 4 or len(right) != 4:
        return 0.0
    try:
        ax0, ay0, ax1, ay1 = [float(value) for value in left]
        bx0, by0, bx1, by1 = [float(value) for value in right]
    except (TypeError, ValueError):
        return 0.0
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _ocr_bboxes_match(left: Any, right: Any) -> bool:
    """Return whether two OCR boxes are the same region despite small drift."""
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return left is None or right is None
    if len(left) != 4 or len(right) != 4:
        return False
    try:
        left_values = [float(value) for value in left]
        right_values = [float(value) for value in right]
    except (TypeError, ValueError):
        return False

    if max(abs(a - b) for a, b in zip(left_values, right_values)) <= _OCR_DUPLICATE_EDGE_TOLERANCE:
        return True

    lx0, ly0, lx1, ly1 = left_values
    rx0, ry0, rx1, ry1 = right_values
    intersection = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(0.0, min(ly1, ry1) - max(ly0, ry0))
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    smaller_area = min(left_area, right_area)
    return smaller_area > 0.0 and intersection / smaller_area >= _OCR_DUPLICATE_CONTAINMENT


def _preprocess_scan_image_b64(
    image_b64: str,
    *,
    enabled: bool = True,
    strength: int = 1,
) -> Tuple[str, Dict[str, Any]]:
    """Apply safe, OCR-oriented preprocessing without losing page coordinates.

    The first layer deliberately keeps the image dimensions unchanged.  It
    handles EXIF orientation, low contrast and mild sharpening.  A conservative
    deskew is attempted when OpenCV can identify a clear text angle; its inverse
    affine transform is retained so OCR bboxes can be mapped back to the source
    page.
    """
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "strength": int(max(1, strength)),
        "applied": [],
        "geometry": "identity",
    }
    if not enabled or Image is None or not isinstance(image_b64, str) or not image_b64:
        return image_b64, info

    try:
        raw = base64.b64decode(image_b64)
        with Image.open(io.BytesIO(raw)) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB") if ImageOps is not None else source.convert("RGB")
            original_w, original_h = oriented.size
            info["original_shape_hw"] = [int(original_h), int(original_w)]
            processed = oriented
            info["applied"].append("exif_transpose")

            gray = np.asarray(processed.convert("L"), dtype=np.uint8)
            contrast = float(gray.std()) if gray.size else 0.0
            info["contrast_std"] = contrast
            if ImageOps is not None and contrast < 70.0:
                processed = ImageOps.autocontrast(processed)
                info["applied"].append("autocontrast")
            if ImageEnhance is not None:
                processed = ImageEnhance.Contrast(processed).enhance(1.08 + 0.05 * max(0, strength - 1))
                info["applied"].append("contrast")
            if ImageFilter is not None:
                processed = processed.filter(
                    ImageFilter.UnsharpMask(radius=1, percent=110 + 20 * max(0, strength - 1), threshold=3)
                )
                info["applied"].append("unsharp_mask")

            # Deskew only when the detected angle is small and unambiguous.
            # Large rotations are left to the existing orientation/layout path.
            try:
                import cv2  # type: ignore[import-untyped]

                processed_array = np.asarray(processed, dtype=np.uint8)
                gray_for_angle = cv2.cvtColor(processed_array, cv2.COLOR_RGB2GRAY)
                _, foreground = cv2.threshold(
                    gray_for_angle, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                coords = np.column_stack(np.where(foreground > 0))
                if len(coords) >= 100:
                    angle = float(cv2.minAreaRect(coords.astype(np.float32))[-1])
                    angle = -(90.0 + angle) if angle < -45.0 else -angle
                    if 0.35 < abs(angle) <= 8.0:
                        center = (processed_array.shape[1] / 2.0, processed_array.shape[0] / 2.0)
                        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated = cv2.warpAffine(
                            processed_array,
                            matrix,
                            (processed_array.shape[1], processed_array.shape[0]),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255),
                        )
                        processed = Image.fromarray(rotated, mode="RGB")
                        inverse = cv2.invertAffineTransform(matrix)
                        info["geometry"] = "deskew"
                        info["inverse_affine"] = inverse.tolist()
                        info["deskew_angle_degrees"] = angle
                        info["applied"].append("deskew")
            except Exception:
                # OpenCV is optional and preprocessing must never make OCR fail.
                pass

            info["processed_shape_hw"] = [int(processed.height), int(processed.width)]
            buf = io.BytesIO()
            processed.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii"), info
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return image_b64, info


def _map_bbox_from_preprocess(bbox: Sequence[float], preprocess_info: Dict[str, Any]) -> List[float]:
    """Map a normalized bbox from the preprocessed page to source coordinates."""
    inverse = preprocess_info.get("inverse_affine")
    if not isinstance(inverse, list) or len(inverse) != 2 or any(
        not isinstance(row, list) or len(row) != 3 for row in inverse
    ):
        return [float(value) for value in bbox[:4]]
    processed_hw = preprocess_info.get("processed_shape_hw") or preprocess_info.get("original_shape_hw")
    original_hw = preprocess_info.get("original_shape_hw") or processed_hw
    if not isinstance(processed_hw, (list, tuple)) or len(processed_hw) != 2:
        return [float(value) for value in bbox[:4]]
    if not isinstance(original_hw, (list, tuple)) or len(original_hw) != 2:
        return [float(value) for value in bbox[:4]]
    try:
        processed_h, processed_w = float(processed_hw[0]), float(processed_hw[1])
        original_h, original_w = float(original_hw[0]), float(original_hw[1])
        matrix = np.asarray(inverse, dtype=np.float64)
        x0, y0, x1, y1 = [float(value) for value in bbox[:4]]
        points = np.asarray(
            [
                [x0 * processed_w, y0 * processed_h, 1.0],
                [x1 * processed_w, y0 * processed_h, 1.0],
                [x1 * processed_w, y1 * processed_h, 1.0],
                [x0 * processed_w, y1 * processed_h, 1.0],
            ],
            dtype=np.float64,
        )
        mapped = points @ matrix.T
        xs = mapped[:, 0] / max(original_w, 1.0)
        ys = mapped[:, 1] / max(original_h, 1.0)
        return [
            max(0.0, min(1.0, float(xs.min()))),
            max(0.0, min(1.0, float(ys.min()))),
            max(0.0, min(1.0, float(xs.max()))),
            max(0.0, min(1.0, float(ys.max()))),
        ]
    except (TypeError, ValueError, IndexError, FloatingPointError):
        return [float(value) for value in bbox[:4]]


def _scan_ocr_tiles_from_b64(
    image_b64: str,
    *,
    tile_size: int = 1024,
    overlap: float = 0.15,
) -> List[Tuple[List[float], str]]:
    """Create overlapping OCR tiles in normalized page coordinates."""
    if Image is None or not isinstance(image_b64, str) or not image_b64:
        return []
    try:
        raw = base64.b64decode(image_b64)
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGB")
            width, height = image.size
            size = max(256, int(tile_size))
            if width <= size and height <= size:
                return []
            overlap = min(0.45, max(0.0, float(overlap)))
            step = max(1, int(round(size * (1.0 - overlap))))

            def starts(length: int) -> List[int]:
                if length <= size:
                    return [0]
                values = list(range(0, length - size + 1, step))
                last = length - size
                if not values or values[-1] != last:
                    values.append(last)
                return values

            tiles: List[Tuple[List[float], str]] = []
            for top in starts(height):
                for left in starts(width):
                    crop = image.crop((left, top, min(left + size, width), min(top + size, height)))
                    buf = io.BytesIO()
                    crop.save(buf, format="PNG")
                    crop.close()
                    tiles.append(
                        (
                            [
                                float(left / width),
                                float(top / height),
                                float(min(left + size, width) / width),
                                float(min(top + size, height) / height),
                            ],
                            base64.b64encode(buf.getvalue()).decode("ascii"),
                        )
                    )
            return tiles
    except Exception:
        return []


def _merge_ocr_blocks_once(blocks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run one conservative merge pass over crop/full-page/tile OCR outputs."""
    merged: List[Dict[str, Any]] = []
    for raw_block in blocks:
        if not isinstance(raw_block, dict) or not raw_block.get("text") or raw_block.get("_fallback"):
            continue
        # A detector crop often returns the underline/separator as several
        # standalone "-" blocks. They are not useful text and can steal the
        # hover target from the actual line when retained downstream.
        if not _normalise_ocr_text(raw_block.get("text")):
            continue
        block = dict(raw_block)
        block.setdefault("ocr_source", "page_elements_crop")
        duplicate_index: Optional[int] = None
        geometry_only_duplicate = False
        for index, previous in enumerate(merged):
            left_bbox = block.get("bbox_xyxy_norm")
            right_bbox = previous.get("bbox_xyxy_norm")
            bbox_match = _ocr_bboxes_match(left_bbox, right_bbox)
            if not bbox_match:
                continue
            text_match = _ocr_texts_match(block.get("text"), previous.get("text"))
            same_text_identity = _ocr_text_identity(block.get("text")) == _ocr_text_identity(previous.get("text"))
            geometry_is_strong = (
                _ocr_bbox_edges_close(left_bbox, right_bbox)
                or _ocr_bbox_shapes_similar(left_bbox, right_bbox)
                or _ocr_bbox_geometry_is_duplicate(left_bbox, right_bbox)
            )
            # Page Elements can return several overlapping title/text boxes.
            # When their geometry is strong, merge them even if OCR made very
            # different character mistakes. Missing geometry still requires
            # a text match so repeated words are not collapsed blindly.
            both_have_bbox = isinstance(left_bbox, (list, tuple)) and isinstance(right_bbox, (list, tuple))
            if text_match or same_text_identity or (both_have_bbox and geometry_is_strong):
                duplicate_index = index
                geometry_only_duplicate = bool(
                    both_have_bbox
                    and geometry_is_strong
                    and (
                        not text_match
                        or len(str(block.get("text") or "").strip())
                        > max(12, int(len(str(previous.get("text") or "").strip()) * 1.15))
                        or len(str(previous.get("text") or "").strip())
                        > max(12, int(len(str(block.get("text") or "").strip()) * 1.15))
                    )
                    and not same_text_identity
                )
                break
        if duplicate_index is None:
            existing_sources = block.get("ocr_sources")
            if isinstance(existing_sources, (list, tuple, set)):
                sources = {str(source) for source in existing_sources if source}
            else:
                sources = set()
            sources.add(str(block.get("ocr_source")))
            block["ocr_sources"] = sorted(sources)
            merged.append(block)
            continue

        previous = merged[duplicate_index]
        sources = set(previous.get("ocr_sources") or [])
        sources.update(str(source) for source in (block.get("ocr_sources") or []) if source)
        sources.add(str(block.get("ocr_source")))
        previous["ocr_sources"] = sorted(sources)
        previous_conf = _confidence_value(previous.get("confidence"))
        block_conf = _confidence_value(block.get("confidence"))
        previous_text = _normalise_ocr_text(previous.get("text"))
        block_text = _normalise_ocr_text(block.get("text"))
        prefer_block = False
        raw_previous_text = str(previous.get("text") or "").strip()
        raw_block_text = str(block.get("text") or "").strip()
        preference_decided = False
        if geometry_only_duplicate:
            # A contained child can have a higher confidence while the
            # enclosing region contains additional lines. Preserve the more
            # complete raw text instead of letting confidence discard it.
            if len(raw_block_text) > max(12, int(len(raw_previous_text) * 1.15)):
                prefer_block = True
                preference_decided = True
            elif len(raw_previous_text) > max(12, int(len(raw_block_text) * 1.15)):
                prefer_block = False
                preference_decided = True
        if not preference_decided and block_conf is not None and previous_conf is not None:
            confidence_delta = block_conf - previous_conf
            prefer_block = confidence_delta > 0.025
            if abs(confidence_delta) <= 0.025:
                prefer_block = _ocr_text_quality(block.get("text")) > _ocr_text_quality(previous.get("text")) + 0.03
        elif not preference_decided and block_conf is not None:
            prefer_block = previous_conf is None and block_conf >= 0.82
        elif not preference_decided and previous_conf is None:
            quality_delta = _ocr_text_quality(block.get("text")) - _ocr_text_quality(previous.get("text"))
            prefer_block = quality_delta > 0.03 or (abs(quality_delta) <= 0.03 and len(block_text) > len(previous_text))
        if prefer_block:
            replacement = dict(block)
            replacement["ocr_sources"] = sorted(sources)
            merged[duplicate_index] = replacement
    merged.sort(key=lambda block: (block.get("sort_y", 0.0), block.get("sort_x", 0.0)))
    return merged


def _merge_ocr_blocks(blocks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge OCR passes to convergence while retaining model provenance.

    Crop, full-page, and overlapping-tile results can form a bridge cluster:
    A and C do not initially overlap enough, then B replaces A and makes the
    surviving geometry equivalent to C. A single forward pass cannot revisit
    C. Up to three conservative passes close that gap without changing the
    detector, recognition model, or reading-order policy.
    """
    current = [dict(block) for block in blocks if isinstance(block, dict)]
    for _ in range(3):
        merged = _merge_ocr_blocks_once(current)
        if len(merged) >= len(current):
            return merged
        current = merged
    return current


def _scan_ocr_quality(
    blocks: Sequence[Dict[str, Any]],
    *,
    min_quality: float,
    attempts: int,
    errors: Sequence[str] = (),
) -> Dict[str, Any]:
    valid = [block for block in blocks if isinstance(block, dict) and block.get("text") and not block.get("_fallback")]
    confidences = [
        confidence
        for confidence in (_confidence_value(block.get("confidence")) for block in valid)
        if confidence is not None
    ]
    bbox_valid = [
        block
        for block in valid
        if isinstance(block.get("bbox_xyxy_norm"), (list, tuple)) and len(block["bbox_xyxy_norm"]) == 4
    ]
    unique_texts = {_normalise_ocr_text(block.get("text")) for block in valid}
    duplicate_ratio = 1.0 - (len(unique_texts) / max(1, len(valid)))
    mean_confidence = float(sum(confidences) / len(confidences)) if confidences else None
    confidence_component = mean_confidence if mean_confidence is not None else (0.5 if valid else 0.0)
    text_chars = sum(len(str(block.get("text") or "")) for block in valid)
    text_signal = min(1.0, text_chars / 160.0)
    bbox_signal = len(bbox_valid) / max(1, len(valid))
    unique_signal = max(0.0, 1.0 - duplicate_ratio)
    score = 0.40 * confidence_component + 0.25 * text_signal + 0.20 * bbox_signal + 0.15 * unique_signal
    return {
        "score": float(max(0.0, min(1.0, score))),
        "passed": bool(valid and score >= float(min_quality)),
        "attempts": int(attempts),
        "num_blocks": int(len(valid)),
        "text_chars": int(text_chars),
        "mean_confidence": mean_confidence,
        "bbox_valid_ratio": float(bbox_signal),
        "duplicate_ratio": float(duplicate_ratio),
        "errors": list(errors),
    }


def _run_scan_ocr(
    page_image_b64: str,
    *,
    model: Any,
    invoke_url: str,
    api_key: Optional[str],
    nim_client: Optional[NIMClient],
    request_timeout_s: float,
    inference_batch_size: int,
    retry: RemoteRetryParams,
    preprocess_info: Dict[str, Any],
    tile_size: int,
    tile_overlap: float,
    attempt: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Run OCR on a full scan page and overlapping tiles."""
    jobs: List[Tuple[str, List[float], str]] = [("scan_full_page", [0.0, 0.0, 1.0, 1.0], page_image_b64)]
    jobs.extend(("scan_tile", bbox, tile_b64) for bbox, tile_b64 in _scan_ocr_tiles_from_b64(
        page_image_b64, tile_size=tile_size, overlap=tile_overlap
    ))
    responses: List[Any] = []
    errors: List[str] = []
    if invoke_url:
        try:
            invoke_kw = dict(
                invoke_url=invoke_url,
                image_b64_list=[image for _source, _bbox, image in jobs],
                api_key=api_key,
                timeout_s=float(request_timeout_s),
                max_batch_size=max(1, int(inference_batch_size)),
                max_retries=int(retry.remote_max_retries),
                max_429_retries=int(retry.remote_max_429_retries),
            )
            if nim_client is not None:
                responses = list(nim_client.invoke_image_inference_batches(**invoke_kw))
            else:
                responses = list(
                    invoke_image_inference_batches(
                        **invoke_kw,
                        max_pool_workers=int(retry.remote_max_pool_workers),
                    )
                )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    else:
        for _source, _bbox, image in jobs:
            try:
                responses.append(model.invoke(image.encode("utf-8"), merge_level="paragraph"))
            except Exception as exc:
                responses.append(None)
                errors.append(f"{type(exc).__name__}: {exc}")

    blocks: List[Dict[str, Any]] = []
    if len(responses) != len(jobs):
        errors.append(f"expected {len(jobs)} OCR responses, got {len(responses)}")
    for (source, crop_bbox, _image), response in zip(jobs, responses):
        try:
            parsed = _parse_ocr_result(_extract_remote_ocr_item(response) if invoke_url else response)
            mapped = _map_ocr_blocks_to_page(parsed, crop_bbox)
            for block in mapped:
                block["bbox_xyxy_norm"] = _map_bbox_from_preprocess(block["bbox_xyxy_norm"], preprocess_info)
                block["sort_y"] = float(block["bbox_xyxy_norm"][1])
                block["sort_x"] = float(block["bbox_xyxy_norm"][0])
                block["ocr_source"] = source
                block["ocr_attempt"] = int(attempt)
            blocks.extend(mapped)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    # A full-page pass is useful for recall, but small text is often more
    # accurate when the detected line is enlarged before OCR. Refine only
    # suspicious full-page blocks; this avoids multiplying requests for
    # already reliable text while fixing cases such as the title/date lines.
    if invoke_url and blocks and preprocess_info.get("geometry") == "identity":
        refine_candidates = []
        for block in blocks:
            bbox = block.get("bbox_xyxy_norm")
            if block.get("ocr_source") != "scan_full_page" or not isinstance(bbox, (list, tuple)):
                continue
            confidence = _confidence_value(block.get("confidence"))
            quality = _ocr_text_quality(block.get("text"))
            if (confidence is None or confidence < 0.82) or quality < 0.62:
                refine_candidates.append(block)
        refine_candidates = refine_candidates[:8]
        refine_jobs: List[Tuple[Dict[str, Any], List[float], str]] = []
        for candidate in refine_candidates:
            try:
                candidate_bbox, crop_b64 = _make_ocr_refinement_crop(
                    page_image_b64,
                    candidate["bbox_xyxy_norm"],
                )
            except (TypeError, ValueError):
                continue
            if crop_b64:
                refine_jobs.append((candidate, candidate_bbox, crop_b64))

        if refine_jobs:
            try:
                refine_kwargs = dict(
                    invoke_url=invoke_url,
                    image_b64_list=[crop for _candidate, _bbox, crop in refine_jobs],
                    api_key=api_key,
                    timeout_s=float(request_timeout_s),
                    # Keep focused crops isolated: batching different line
                    # aspect ratios was the source of unstable small-text OCR.
                    max_batch_size=1,
                    max_retries=int(retry.remote_max_retries),
                    max_429_retries=int(retry.remote_max_429_retries),
                )
                if nim_client is not None:
                    refine_responses = list(nim_client.invoke_image_inference_batches(**refine_kwargs))
                else:
                    refine_responses = list(
                        invoke_image_inference_batches(
                            **refine_kwargs,
                            max_pool_workers=max(1, min(int(retry.remote_max_pool_workers), len(refine_jobs))),
                        )
                    )
                for (candidate, crop_bbox, _crop), response in zip(refine_jobs, refine_responses):
                    parsed = _parse_ocr_result(_extract_remote_ocr_item(response))
                    mapped = _map_ocr_blocks_to_page(parsed, crop_bbox)
                    for refined in mapped:
                        refined["bbox_xyxy_norm"] = _map_bbox_from_preprocess(
                            refined["bbox_xyxy_norm"], preprocess_info
                        )
                        refined["sort_y"] = float(refined["bbox_xyxy_norm"][1])
                        refined["sort_x"] = float(refined["bbox_xyxy_norm"][0])
                        refined["ocr_source"] = "scan_refine"
                        refined["ocr_attempt"] = int(attempt)
                        # Keep only the line that belongs to this refinement
                        # target; padding can expose the neighbouring line.
                        if _ocr_bboxes_match(refined.get("bbox_xyxy_norm"), candidate.get("bbox_xyxy_norm")):
                            blocks.append(refined)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: refinement: {exc}")
    return blocks, errors


def ocr_response_to_text(preds: Any) -> str:
    """Extract joined OCR text from a recognizer response, returning ``""``
    when no text is detected.

    Wraps :func:`_parse_ocr_result` + :func:`_blocks_to_text` but suppresses
    ``_parse_ocr_result``'s last-resort stringify fallback (which dumps the
    raw response repr when no shape matches) — that fallback produces noise
    rows for callers like the video frame OCR actor where many frames have
    no on-screen text and we need an empty-string sentinel to drop them.
    """
    blocks = _parse_ocr_result(preds)
    if not blocks:
        return ""
    if len(blocks) == 1 and blocks[0].get("_fallback"):
        return ""
    return _blocks_to_text(blocks)


def ocr_b64_to_text(
    image_b64_list: Sequence[str],
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    nim_client: Optional[NIMClient] = None,
    merge_level: str = "paragraph",
    batch_size: int = 8,
    timeout_s: float = 120.0,
    retry: Optional[RemoteRetryParams] = None,
) -> List[str]:
    """Run a remote OCR endpoint on base64 PNG images; return one text per input.

    Routes to the configured remote endpoint when ``invoke_url`` is set (uses
    ``nim_client`` if provided, otherwise spins up a fresh batched call), or to
    a supplied local model. Empty/non-string
    inputs map to ``""``; per-image parse failures are logged and also map
    to ``""`` so the output preserves input order and length.
    """
    n = len(image_b64_list)
    if n == 0:
        return []

    use_remote = bool((invoke_url or "").strip())
    if not use_remote and model is None:
        raise ValueError("ocr_b64_to_text requires either invoke_url or a local model.")

    valid_idx = [i for i, b in enumerate(image_b64_list) if isinstance(b, str) and b]
    valid_b64 = [image_b64_list[i] for i in valid_idx]
    out = [""] * n
    if not valid_b64:
        return out

    retry_params = retry or RemoteRetryParams()

    if use_remote:
        try:
            invoke_kw = dict(
                invoke_url=invoke_url,
                image_b64_list=valid_b64,
                api_key=api_key,
                timeout_s=float(timeout_s),
                max_batch_size=int(batch_size),
                max_retries=int(retry_params.remote_max_retries),
                max_429_retries=int(retry_params.remote_max_429_retries),
            )
            if nim_client is not None:
                response_items = nim_client.invoke_image_inference_batches(**invoke_kw)
            else:
                response_items = invoke_image_inference_batches(
                    **invoke_kw,
                    max_pool_workers=int(retry_params.remote_max_pool_workers),
                )
        except Exception:
            _logger.exception("Remote OCR call failed")
            return out
        for resp, dst in zip(response_items, valid_idx):
            try:
                preds = _extract_remote_ocr_item(resp)
                out[dst] = ocr_response_to_text(preds)
            except Exception:
                _logger.warning("Failed to parse OCR response for index %d", dst)
                out[dst] = ""
        return out

    # Local model path.
    for b64, dst in zip(valid_b64, valid_idx):
        try:
            preds = model.invoke(b64.encode("utf-8"), merge_level=merge_level)
            out[dst] = ocr_response_to_text(preds)
        except Exception:
            _logger.exception("Local OCR failed on image at index %d", dst)
            out[dst] = ""
    return out


def split_ocrable_rows(
    batch_df: pd.DataFrame,
    ocrable_content_types: Sequence[str] = ("",),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Partition rows into OCR-able rows and passthrough rows.

    Rows whose ``_content_type`` is in ``ocrable_content_types`` are treated
    as OCR-able; all others are passed through unchanged. Batches with no
    ``_content_type`` column at all (audio-free pipelines like PDF / image)
    are fully OCR-able. The default accepts only the empty discriminator;
    pipelines that mix OCR-able and non-OCR-able rows (e.g. video, which
    interleaves ``video_frame`` and ``audio``) pass their own sentinel set.
    """
    if "_content_type" not in batch_df.columns:
        return batch_df.copy(), pd.DataFrame()
    ct = batch_df["_content_type"].astype(str).fillna("")
    ocr_mask = ct.isin(list(ocrable_content_types))
    return (
        batch_df[ocr_mask].reset_index(drop=True),
        batch_df[~ocr_mask].reset_index(drop=True),
    )


def concat_with_passthrough(processed: pd.DataFrame, passthrough: pd.DataFrame) -> pd.DataFrame:
    """Concat the OCR-stage output with passthrough rows, harmonising columns."""
    if passthrough is None or passthrough.empty:
        return processed
    if processed is None or processed.empty:
        return passthrough
    for col in processed.columns:
        if col not in passthrough.columns:
            passthrough = passthrough.assign(**{col: None})
    for col in passthrough.columns:
        if col not in processed.columns:
            processed = processed.assign(**{col: None})
    return pd.concat([processed[passthrough.columns.tolist()], passthrough], ignore_index=True, sort=False)


def is_full_image_batch(batch_df: pd.DataFrame) -> bool:
    """True when the batch carries top-level ``image_b64`` and no usable
    ``page_elements_v3`` — i.e. the input came from frame extraction
    (or any other producer that hands raw images straight to OCR)."""
    if "image_b64" not in batch_df.columns:
        return False
    pe = batch_df.get("page_elements_v3")
    if pe is None:
        return True
    return not pe.notna().any()


def full_image_ocr_df(
    batch_df: pd.DataFrame,
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    nim_client: Optional[NIMClient] = None,
    merge_level: str = "paragraph",
    batch_size: int = 8,
    timeout_s: float = 120.0,
    retry: Optional[RemoteRetryParams] = None,
) -> pd.DataFrame:
    """Run full-image OCR on a DataFrame whose rows carry top-level ``image_b64``.

    Writes ``text`` and drops rows whose OCR result is empty (so frames with
    no on-screen text don't pollute the embedder).
    """
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return pd.DataFrame()
    out = batch_df.copy()
    b64s = [b if isinstance(b, str) else "" for b in out.get("image_b64", [])]
    out["text"] = ocr_b64_to_text(
        b64s,
        model=model,
        invoke_url=invoke_url,
        api_key=api_key,
        nim_client=nim_client,
        merge_level=merge_level,
        batch_size=batch_size,
        timeout_s=timeout_s,
        retry=retry,
    )
    return out[out["text"].astype(bool)].reset_index(drop=True)


def _blocks_to_pseudo_markdown(
    blocks: List[Dict[str, Any]],
    crop_hw: Tuple[int, int] = (0, 0),
) -> str:
    """Convert OCR text blocks into pseudo-markdown table format.

    Uses DBSCAN clustering on pixel y-coordinates to identify rows, then
    sorts within each row by x-coordinate and joins with pipe separators.

    Parameters
    ----------
    blocks : list of dict
        OCR text blocks with ``sort_y`` (normalised [0,1]) and ``sort_x``.
    crop_hw : (height, width)
        Pixel dimensions of the crop image.  When provided the normalised
        ``sort_y`` values are scaled to pixels and clustered with
        ``eps=10`` (matching `nemo_retriever.api` behaviour).  Falls back to the old
        normalised-space heuristic when the height is unavailable.
    """
    if not blocks:
        return ""

    valid = [b for b in blocks if b.get("text")]
    if not valid:
        return ""

    df = pd.DataFrame(valid)
    df = df.sort_values("sort_y")

    y_vals = df["sort_y"].values
    crop_h = crop_hw[0] if crop_hw else 0

    if crop_h > 0:
        y_pixels = (y_vals * crop_h).astype(int)
        eps = 10
    else:
        y_range = y_vals.max() - y_vals.min()
        if y_range > 0:
            y_pixels = (y_vals - y_vals.min()) / y_range
            eps = 0.03
        else:
            y_pixels = y_vals
            eps = 0.1

    try:
        from sklearn.cluster import DBSCAN

        dbscan = DBSCAN(eps=eps, min_samples=1)
        dbscan.fit(y_pixels.reshape(-1, 1))
        df["cluster"] = dbscan.labels_
    except ImportError:
        # Naive fallback: round y to a coarse grid to simulate row grouping.
        df["cluster"] = (y_pixels / (eps if eps > 0 else 1)).round().astype(int)

    df = df.sort_values(["cluster", "sort_x"])

    rows = []
    for _, grp in df.groupby("cluster", sort=True):
        rows.append("| " + " | ".join(grp["text"].tolist()) + " |")
    return "\n".join(rows)


def _bboxes_close(a: Sequence[float], b: Sequence[float], tol: float = 1e-4) -> bool:
    """Check if two normalized bboxes are approximately equal."""
    if len(a) != 4 or len(b) != 4:
        return False
    return all(abs(float(a[i]) - float(b[i])) < tol for i in range(4))


def _find_ts_detections_for_bbox(
    row: Any,
    table_bbox: Sequence[float],
) -> Optional[Tuple[List[Dict[str, Any]], Optional[Tuple[int, int]]]]:
    """Find table-structure detections + crop size for a table bbox.

    Reads the ``table_structure_v1`` column from *row* and returns the
    ``(detections, (H, W))`` tuple for the region whose ``bbox_xyxy_norm``
    matches *table_bbox*. Returns ``None`` if the column is missing, no
    region matches, or the matching region has no detections.
    """
    ts_col = getattr(row, "table_structure_v1", None)
    if not isinstance(ts_col, dict):
        return None
    regions = ts_col.get("regions")
    if not isinstance(regions, list):
        return None

    for region in regions:
        if not isinstance(region, dict):
            continue
        region_bbox = region.get("bbox_xyxy_norm")
        if not isinstance(region_bbox, (list, tuple)) or len(region_bbox) != 4:
            continue
        if not _bboxes_close(table_bbox, region_bbox):
            continue
        dets = region.get("detections")
        if not isinstance(dets, list) or not dets:
            return None
        hw = region.get("orig_shape_hw")
        hw_t: Optional[Tuple[int, int]] = None
        if isinstance(hw, (list, tuple)) and len(hw) == 2:
            try:
                hw_t = (int(hw[0]), int(hw[1]))
            except (TypeError, ValueError):
                hw_t = None
        return (dets, hw_t)
    return None


def _table_structure_status_for_bbox(row: Any, table_bbox: Sequence[float]) -> str:
    """Describe why a table could not be reconstructed from structure output."""
    payload = getattr(row, "table_structure_v1", None)
    if not isinstance(payload, dict):
        return "missing_payload"
    regions = payload.get("regions")
    if not isinstance(regions, list):
        return "missing_regions"
    for region in regions:
        if not isinstance(region, dict):
            continue
        region_bbox = region.get("bbox_xyxy_norm")
        if not isinstance(region_bbox, (list, tuple)) or len(region_bbox) != 4:
            continue
        if not _bboxes_close(table_bbox, region_bbox):
            continue
        detections = region.get("detections")
        if not isinstance(detections, list) or not detections:
            return "empty_detections"
        if not any(str(item.get("label_name") or "") == "cell" for item in detections if isinstance(item, dict)):
            return "no_cells"
        return "available"
    return "unmatched_region"


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def ocr_page_elements(
    batch_df: Any,
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout_s: float = 120.0,
    extract_text: bool = False,
    extract_tables: bool = False,
    extract_charts: bool = False,
    extract_infographics: bool = False,
    extract_images: bool = False,
    extract_stamps: bool = False,
    use_table_structure: bool = False,
    inference_batch_size: int = 8,
    remote_retry: RemoteRetryParams | None = None,
    nim_client: NIMClient | None = None,
    scan_ocr_fallback: bool = True,
    scan_ocr_preprocess: bool = True,
    scan_ocr_tile_size: int = 1024,
    scan_ocr_tile_overlap: float = 0.15,
    scan_ocr_min_quality: float = 0.45,
    scan_ocr_max_retries: int = 1,
    **kwargs: Any,
) -> Any:
    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )
    """
    Run the configured document OCR pipeline on cropped Page Elements regions.

    For each row (page) in ``batch_df``:
    1. Read ``page_elements_v3`` detections and ``page_image["image_b64"]``.
    2. For each detection whose ``label_name`` is a requested type, crop the
       page image, invoke OCR, parse the result, and collect text.
    3. For scan pages, optionally OCR the full page and overlapping tiles,
       then merge/deduplicate blocks and record a quality gate result.
    4. Write per-type content lists and timing metadata to output columns.

    Parameters
    ----------
    batch_df : pandas.DataFrame
        Ray Data batch with ``page_elements_v3`` and ``page_image`` columns.
    model
        Initialised OCR model.
    extract_tables, extract_charts, extract_infographics : bool
        Which element types to OCR.

    Returns
    -------
    pandas.DataFrame
        Original columns plus ``table``, ``chart``,
        ``infographic``, and ``ocr``.
    """
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("ocr_page_elements currently only supports pandas.DataFrame input.")

    # The original integrated endpoint is authoritative. The split PP path is
    # additive and can only be selected when the Nemotron endpoint is absent.
    invoke_url = str(invoke_url or kwargs.get("ocr_invoke_url") or "").strip()
    line_detector_url = str(kwargs.get("line_detector_invoke_url") or "").strip()
    recognizer_url = str(kwargs.get("ocr_recognizer_invoke_url") or "").strip()
    box_ocr_mode = kwargs.get("ocr_pipeline") in {"pipeline-ppocrv6", "pipeline-tesseract"}
    selected_pipeline = str(kwargs.get("ocr_pipeline") or "pipeline-nemotron-ocr")
    if not invoke_url and recognizer_url and (line_detector_url or box_ocr_mode):
        from nemo_retriever.common.modality.ocr.ppocr import ppocrv6_page_elements

        # These values are normalized above and passed explicitly below.  The
        # graph also forwards the original OCR kwargs, so forwarding ``kwargs``
        # unchanged would bind the same keyword twice and fail every job before
        # the worker can publish visual evidence.
        ppocr_kwargs = dict(kwargs)
        for key in (
            "line_detector_invoke_url",
            "ocr_recognizer_invoke_url",
            "ocr_pipeline",
            "api_key",
            "request_timeout_s",
            "inference_batch_size",
            "remote_retry",
            "nim_client",
            "extract_text",
            "extract_tables",
            "extract_charts",
            "extract_infographics",
            "extract_images",
            "use_table_structure",
        ):
            ppocr_kwargs.pop(key, None)

        return ppocrv6_page_elements(
            batch_df,
            line_detector_invoke_url=line_detector_url or None,
            ocr_recognizer_invoke_url=recognizer_url,
            box_ocr_mode=box_ocr_mode,
            api_key=api_key,
            request_timeout_s=request_timeout_s,
            inference_batch_size=inference_batch_size,
            remote_retry=retry,
            nim_client=nim_client,
            extract_text=extract_text,
            extract_tables=extract_tables,
            extract_charts=extract_charts,
            extract_infographics=extract_infographics,
            extract_images=extract_images,
            use_table_structure=use_table_structure,
            scan_ocr_fallback=scan_ocr_fallback,
            scan_ocr_tile_size=scan_ocr_tile_size,
            scan_ocr_tile_overlap=scan_ocr_tile_overlap,
            **ppocr_kwargs,
        )

    use_remote = bool(invoke_url)
    if not use_remote and model is None:
        raise ValueError("A local `model` is required when `invoke_url` is not provided.")

    # Determine which labels we need to process.
    # Text/title labels are added per-row based on needs_ocr_for_text metadata.
    wanted_labels: set[str] = set()
    if extract_tables:
        wanted_labels.add("table")
    if extract_charts:
        wanted_labels.add("chart")
    if extract_infographics:
        wanted_labels.add("infographic")
    if extract_images:
        wanted_labels.add("image")
    if extract_stamps:
        wanted_labels.add("stamp")

    # Per-row accumulators.
    all_table: List[List[Dict[str, Any]]] = []
    all_chart: List[List[Dict[str, Any]]] = []
    all_infographic: List[List[Dict[str, Any]]] = []
    all_stamp: List[List[Dict[str, Any]]] = []
    all_images: List[List[Dict[str, Any]]] = []
    all_ocr_text_blocks: List[List[Dict[str, Any]]] = []
    all_ocr_visual_text_blocks: List[List[Dict[str, Any]]] = []
    all_text: List[str] = []
    all_ocr_meta: List[Dict[str, Any]] = []

    t0_total = time.perf_counter()

    for row in batch_df.itertuples(index=False):
        table_items: List[Dict[str, Any]] = []
        chart_items: List[Dict[str, Any]] = []
        infographic_items: List[Dict[str, Any]] = []
        stamp_items: List[Dict[str, Any]] = []
        image_items: List[Dict[str, Any]] = []
        row_ocr_text_blocks: List[Dict[str, Any]] = []
        row_ocr_visual_text_blocks: List[Dict[str, Any]] = []
        row_error: Any = None
        scan_quality: Optional[Dict[str, Any]] = None

        try:
            # --- get page elements detections ---
            pe = getattr(row, "page_elements_v3", None)
            dets: List[Dict[str, Any]] = []
            if isinstance(pe, dict):
                # Copy the list: stamp OCR appends its own detections below.
                # Mutating the page-elements payload here duplicates stamp
                # blocks in the retained upstream result.
                dets = list(pe.get("detections") or [])
            if not isinstance(dets, list):
                dets = []

            def _detection_for_bbox(bbox: Sequence[float]) -> Dict[str, Any]:
                return next(
                    (
                        det
                        for det in dets
                        if isinstance(det, dict)
                        and _bboxes_close(det.get("bbox_xyxy_norm", []), bbox, tol=1e-3)
                    ),
                    {},
                )
            if extract_stamps:
                stamp_regions = getattr(row, "stamp_regions", None) or []
                if isinstance(stamp_regions, list):
                    for region in stamp_regions:
                        if not isinstance(region, dict):
                            continue
                        bbox = region.get("bbox_xyxy_norm")
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            dets.append(
                                {
                                    "label_name": "stamp",
                                    "bbox_xyxy_norm": list(bbox),
                                    "score": region.get("score"),
                                    "source": "stamp_detector",
                                }
                            )

            # --- get page image ---
            page_image = getattr(row, "page_image", None) or {}
            page_image_b64 = page_image.get("image_b64") if isinstance(page_image, dict) else None

            if not isinstance(page_image_b64, str) or not page_image_b64:
                meta = getattr(row, "metadata", None) or {}
                upstream_err = meta.get("error") if isinstance(meta, dict) else None
                page_num = getattr(row, "page_number", "?")
                path = getattr(row, "path", "?")
                if upstream_err:
                    _logger.warning(
                        "OCR skipping page %s of %s — no page image (upstream error: %s)",
                        page_num,
                        path,
                        upstream_err,
                    )
                else:
                    _logger.debug(
                        "OCR skipping page %s of %s — no page image (text-only or raster not requested)",
                        page_num,
                        path,
                    )
                all_table.append(table_items)
                all_chart.append(chart_items)
                all_infographic.append(infographic_items)
                all_stamp.append(stamp_items)
                all_images.append(list(getattr(row, "images", None) or []))
                all_ocr_text_blocks.append([])
                all_ocr_visual_text_blocks.append([])
                all_text.append(None)
                all_ocr_meta.append(
                    {
                        "timing": None,
                        "error": upstream_err,
                        "num_detections": 0,
                        "counts_by_label": {},
                        "backend": "nemotron_ocr_v2_nim" if use_remote else "nemotron_ocr_v2_local",
                        "pipeline": selected_pipeline,
                    }
                )
                continue

            # --- determine per-row labels (text/title only for pages needing OCR) ---
            row_wanted = wanted_labels
            if extract_text:
                meta = getattr(row, "metadata", None) or {}
                needs_ocr = meta.get("needs_ocr_for_text", False) if isinstance(meta, dict) else False
                if needs_ocr:
                    row_wanted = wanted_labels | _TEXT_LABELS

            # --- decode page image once, crop all matching detections ---
            if use_remote:
                crops = _crop_all_from_page(page_image_b64, dets, row_wanted, as_b64=True)
                crop_b64s: List[str] = [b64 for _label, _bbox, b64 in crops]
                crop_meta: List[Tuple[str, List[float]]] = [(label, bbox) for label, bbox, _b64 in crops]

                if crop_b64s:
                    _invoke_kw = dict(
                        invoke_url=invoke_url,
                        image_b64_list=crop_b64s,
                        api_key=api_key,
                        timeout_s=float(request_timeout_s),
                        # Use the explicit OCR batch setting. The previous
                        # kwargs lookup ignored the function argument, so a
                        # dashboard value of 1/2/4 still sent all crops in an
                        # 8-image request. Mixed-aspect-ratio OCR batches can
                        # degrade small title crops and their bboxes.
                        max_batch_size=max(1, int(inference_batch_size)),
                        max_retries=int(retry.remote_max_retries),
                        max_429_retries=int(retry.remote_max_429_retries),
                    )
                    if nim_client is not None:
                        response_items = nim_client.invoke_image_inference_batches(**_invoke_kw)
                    else:
                        response_items = invoke_image_inference_batches(
                            **_invoke_kw,
                            max_pool_workers=int(retry.remote_max_pool_workers),
                        )
                    if len(response_items) != len(crop_meta):
                        raise RuntimeError(f"Expected {len(crop_meta)} OCR responses, got {len(response_items)}")

                    for i, (label_name, bbox) in enumerate(crop_meta):
                        preds = _extract_remote_ocr_item(response_items[i])

                        blocks = _parse_ocr_result(preds)
                        mapped_blocks = _map_ocr_blocks_to_page(blocks, bbox)
                        if label_name == "table":
                            crop_hw_table: Tuple[int, int] = (0, 0)
                            try:
                                _raw = base64.b64decode(crop_b64s[i])
                                with Image.open(io.BytesIO(_raw)) as _cim:
                                    _cw, _ch = _cim.size
                                    crop_hw_table = (_ch, _cw)
                            except Exception:
                                pass
                            text = ""
                            table_structure_status = "not_requested"
                            if use_table_structure:
                                ts_match = _find_ts_detections_for_bbox(row, bbox)
                                if ts_match is not None:
                                    ts_dets, ts_hw = ts_match
                                    text = join_table_structure_and_ocr_output(ts_dets, preds, ts_hw or crop_hw_table)
                                    table_structure_status = "joined" if text else "empty_join"
                                else:
                                    # Do not turn an unstructured table crop
                                    # into fake Markdown. That output looks
                                    # plausible in the dashboard but loses the
                                    # cell geometry and causes duplicated text.
                                    table_structure_status = _table_structure_status_for_bbox(row, bbox)
                            if not use_table_structure:
                                text = _blocks_to_pseudo_markdown(blocks, crop_hw=crop_hw_table) or _blocks_to_text(
                                    blocks
                                )
                        else:
                            text = _blocks_to_text(mapped_blocks)
                        if label_name in {"image", "chart", "infographic", "stamp"}:
                            row_ocr_visual_text_blocks.extend(
                                [{**block, "source_label": label_name} for block in mapped_blocks]
                            )
                        geometry = _detection_for_bbox(bbox)
                        entry = {
                            "bbox_xyxy_norm": bbox,
                            "model_bbox_xyxy_norm": geometry.get("model_bbox_xyxy_norm") or bbox,
                            "processed_bbox_xyxy_norm": geometry.get("processed_bbox_xyxy_norm"),
                            "crop_bbox_xyxy_norm": geometry.get("crop_bbox_xyxy_norm"),
                            "text": text,
                        }
                        if label_name == "table":
                            entry["table_structure_status"] = table_structure_status
                        if label_name == "table":
                            table_items.append(entry)
                        elif label_name == "chart":
                            chart_items.append(entry)
                        elif label_name == "infographic":
                            infographic_items.append(entry)
                        elif label_name == "stamp":
                            stamp_items.append(entry)
                        elif label_name in _TEXT_LABELS:
                            row_ocr_text_blocks.extend(mapped_blocks)
                        if label_name in {"image", "chart", "infographic", "stamp"}:
                            score = geometry.get("score")
                            image_items.append(
                                {
                                    "bbox_xyxy_norm": bbox,
                                    "model_bbox_xyxy_norm": geometry.get("model_bbox_xyxy_norm") or bbox,
                                    "processed_bbox_xyxy_norm": geometry.get("processed_bbox_xyxy_norm"),
                                    "crop_bbox_xyxy_norm": geometry.get("crop_bbox_xyxy_norm"),
                                    "image_b64": crop_b64s[i],
                                    "text": text,
                                    "label_name": label_name,
                                    "score": score,
                                    "source": "page_elements_v3_crop",
                                    "image_type": "detected_region",
                                }
                            )
            else:
                crops = _crop_all_from_page(page_image_b64, dets, row_wanted)

                if inference_batch_size is None or inference_batch_size < 1:
                    raise ValueError(
                        f"inference_batch_size must be set and greater than 0. Value: {inference_batch_size}"
                    )

                local_batch_size = max(1, int(inference_batch_size))

                # Tables require word-level merging; charts/infographics use paragraph-level.
                # Group by merge level so each batched invoke uses one consistent setting.
                local_jobs: Dict[str, List[Tuple[str, List[float], np.ndarray]]] = {"word": [], "paragraph": []}
                for label_name, bbox, crop_array in crops:
                    ml = "word" if label_name == "table" else "paragraph"
                    local_jobs[ml].append((label_name, bbox, crop_array))

                def _append_local_result(
                    label_name: str, bbox: List[float], preds: Any, crop_hw: Tuple[int, int] = (0, 0)
                ) -> None:
                    blocks = _parse_ocr_result(preds)
                    mapped_blocks = _map_ocr_blocks_to_page(blocks, bbox)
                    if label_name == "table":
                        text = ""
                        table_structure_status = "not_requested"
                        if use_table_structure:
                            ts_match = _find_ts_detections_for_bbox(row, bbox)
                            if ts_match is not None:
                                ts_dets, ts_hw = ts_match
                                text = join_table_structure_and_ocr_output(ts_dets, preds, ts_hw or crop_hw)
                                table_structure_status = "joined" if text else "empty_join"
                            else:
                                table_structure_status = _table_structure_status_for_bbox(row, bbox)
                        if not use_table_structure:
                            text = _blocks_to_pseudo_markdown(blocks, crop_hw=crop_hw)
                        if not text and not use_table_structure:
                            text = _blocks_to_text(blocks)
                    else:
                        text = _blocks_to_text(mapped_blocks)
                    if label_name in {"image", "chart", "infographic", "stamp"}:
                        row_ocr_visual_text_blocks.extend(
                            [{**block, "source_label": label_name} for block in mapped_blocks]
                        )
                    geometry = _detection_for_bbox(bbox)
                    entry = {
                        "bbox_xyxy_norm": bbox,
                        "model_bbox_xyxy_norm": geometry.get("model_bbox_xyxy_norm") or bbox,
                        "processed_bbox_xyxy_norm": geometry.get("processed_bbox_xyxy_norm"),
                        "crop_bbox_xyxy_norm": geometry.get("crop_bbox_xyxy_norm"),
                        "text": text,
                    }
                    if label_name == "table":
                        entry["table_structure_status"] = table_structure_status
                    if label_name == "table":
                        table_items.append(entry)
                    elif label_name == "chart":
                        chart_items.append(entry)
                    elif label_name == "infographic":
                        infographic_items.append(entry)
                    elif label_name == "stamp":
                        stamp_items.append(entry)
                    elif label_name in _TEXT_LABELS:
                        row_ocr_text_blocks.extend(mapped_blocks)

                for ml, jobs in local_jobs.items():
                    if not jobs:
                        continue
                    for start in range(0, len(jobs), local_batch_size):
                        batch_jobs = jobs[start : start + local_batch_size]
                        batch_crops = [crop_array for _, _, crop_array in batch_jobs]

                        # Try batched invoke first; if backend does not return one response
                        # per input, fall back to per-item to preserve correctness.
                        try:
                            batch_preds = model.invoke(batch_crops, merge_level=ml)
                        except Exception:
                            batch_preds = None

                        if isinstance(batch_preds, list) and len(batch_preds) == len(batch_jobs):
                            for (label_name, bbox, crop_array), preds in zip(batch_jobs, batch_preds):
                                _append_local_result(
                                    label_name, bbox, preds, crop_hw=(crop_array.shape[0], crop_array.shape[1])
                                )
                        else:
                            for label_name, bbox, crop_array in batch_jobs:
                                preds = model.invoke(crop_array, merge_level=ml)
                                _append_local_result(
                                    label_name, bbox, preds, crop_hw=(crop_array.shape[0], crop_array.shape[1])
                                )

                        # Keep the exact crop so the retained result can prove
                        # which visual region was detected, independently of
                        # OCR quality and of the model's batch return shape.
                        for label_name, bbox, crop_array in batch_jobs:
                            if label_name not in {"image", "chart", "infographic", "stamp"}:
                                continue
                            geometry = _detection_for_bbox(bbox)
                            image_items.append(
                                {
                                    "bbox_xyxy_norm": bbox,
                                    "model_bbox_xyxy_norm": geometry.get("model_bbox_xyxy_norm") or bbox,
                                    "processed_bbox_xyxy_norm": geometry.get("processed_bbox_xyxy_norm"),
                                    "crop_bbox_xyxy_norm": geometry.get("crop_bbox_xyxy_norm"),
                                    "image_b64": _np_rgb_to_b64_png(crop_array),
                                    "text": "",
                                    "label_name": label_name,
                                    "score": _detection_for_bbox(bbox).get("score"),
                                    "source": "page_elements_v3_crop",
                                    "image_type": "detected_region",
                                }
                            )

            # Additive scan fallback: Page Elements remains the primary
            # detector/crop path above.  For scanned pages, OCR the whole page
            # plus overlapping tiles as a recall layer so text outside a
            # detector bbox is not silently lost.
            row_metadata = getattr(row, "metadata", None) or {}
            scan_page = bool(row_metadata.get("needs_ocr_for_text")) if isinstance(row_metadata, dict) else False
            if extract_text and scan_ocr_fallback and scan_page:
                original_text_blocks = list(row_ocr_text_blocks)
                best_blocks = _merge_ocr_blocks(original_text_blocks)
                best_quality = _scan_ocr_quality(
                    best_blocks,
                    min_quality=float(scan_ocr_min_quality),
                    attempts=0,
                )
                retry_errors: List[str] = []
                total_attempts = max(0, int(scan_ocr_max_retries)) + 1
                for attempt in range(1, total_attempts + 1):
                    prepared_b64, preprocess_info = _preprocess_scan_image_b64(
                        page_image_b64,
                        enabled=bool(scan_ocr_preprocess),
                        strength=attempt,
                    )
                    fallback_blocks, errors = _run_scan_ocr(
                        prepared_b64,
                        model=model,
                        invoke_url=invoke_url,
                        api_key=api_key,
                        nim_client=nim_client,
                        request_timeout_s=float(request_timeout_s),
                        inference_batch_size=max(1, int(inference_batch_size or 1)),
                        retry=retry,
                        preprocess_info=preprocess_info,
                        tile_size=max(256, int(scan_ocr_tile_size)),
                        tile_overlap=float(scan_ocr_tile_overlap),
                        attempt=attempt,
                    )
                    retry_errors.extend(errors)
                    candidate_blocks = _merge_ocr_blocks(original_text_blocks + fallback_blocks)
                    candidate_quality = _scan_ocr_quality(
                        candidate_blocks,
                        min_quality=float(scan_ocr_min_quality),
                        attempts=attempt,
                        errors=retry_errors,
                    )
                    candidate_quality["fallback"] = {
                        "enabled": True,
                        "full_page": True,
                        "tiles": True,
                        "tile_size": int(scan_ocr_tile_size),
                        "tile_overlap": float(scan_ocr_tile_overlap),
                        "preprocess": preprocess_info,
                    }
                    # Always keep the first fallback result: it contains the
                    # recall layer even when the existing crop-only result had
                    # an artificially high score.  Later retries compete on
                    # quality against the selected fallback result.
                    if attempt == 1 or candidate_quality["score"] >= best_quality["score"]:
                        best_blocks = candidate_blocks
                        best_quality = candidate_quality
                    if candidate_quality["passed"]:
                        break
                row_ocr_text_blocks = best_blocks
                scan_quality = best_quality

        except BaseException as e:
            print(f"Warning: OCR failed: {type(e).__name__}: {e}")
            row_error = {
                "stage": "ocr_page_elements",
                "type": e.__class__.__name__,
                "message": str(e),
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

        if extract_tables and use_table_structure and not table_items:
            # Preserve the detector's table geometry when OCR/structure
            # reconstruction fails.  Dropping it makes a real table look like
            # missing input and leaves the frontend unable to explain the
            # failed stage.
            for detection in dets:
                if not isinstance(detection, dict) or str(detection.get("label_name") or "") != "table":
                    continue
                bbox = detection.get("bbox_xyxy_norm")
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                table_items.append(
                    {
                        "bbox_xyxy_norm": [float(value) for value in bbox],
                        "model_bbox_xyxy_norm": detection.get("model_bbox_xyxy_norm") or list(bbox),
                        "processed_bbox_xyxy_norm": detection.get("processed_bbox_xyxy_norm"),
                        "crop_bbox_xyxy_norm": detection.get("crop_bbox_xyxy_norm"),
                        "text": "",
                        "table_structure_status": "ocr_error" if row_error else "missing",
                    }
                )

        # Assemble OCR'd text from text/title detections for this row.
        # Use None as sentinel for "keep existing native text".
        if extract_text and row_ocr_text_blocks:
            all_text.append(_blocks_to_text(row_ocr_text_blocks))
        else:
            all_text.append(None)

        row_det_count = (
            len(table_items)
            + len(chart_items)
            + len(infographic_items)
            + len(stamp_items)
            + len(row_ocr_text_blocks)
        )
        row_counts: Dict[str, int] = {}
        if table_items:
            row_counts["table"] = len(table_items)
        if chart_items:
            row_counts["chart"] = len(chart_items)
        if infographic_items:
            row_counts["infographic"] = len(infographic_items)
        if stamp_items:
            row_counts["stamp"] = len(stamp_items)
        if row_ocr_text_blocks:
            row_counts["text"] = len(row_ocr_text_blocks)

        all_table.append(table_items)
        all_chart.append(chart_items)
        all_infographic.append(infographic_items)
        all_stamp.append(stamp_items)
        base_images = list(getattr(row, "images", None) or [])
        all_images.append(base_images + image_items)
        all_ocr_text_blocks.append(row_ocr_text_blocks)
        all_ocr_visual_text_blocks.append(row_ocr_visual_text_blocks)
        all_ocr_meta.append(
            {
                "timing": None,
                "error": row_error,
                "num_detections": row_det_count,
                "counts_by_label": row_counts,
                "scan_ocr_quality": scan_quality,
                "backend": "nemotron_ocr_v2_nim" if use_remote else "nemotron_ocr_v2_local",
                "pipeline": selected_pipeline,
            }
        )

    elapsed = time.perf_counter() - t0_total

    for meta in all_ocr_meta:
        meta["timing"] = {"seconds": float(elapsed)}

    out = batch_df.copy()
    if extract_tables or "table" not in out.columns:
        out["table"] = all_table
    if extract_charts or "chart" not in out.columns:
        out["chart"] = all_chart
    if extract_infographics or "infographic" not in out.columns:
        out["infographic"] = all_infographic
    if extract_stamps or "stamp" not in out.columns:
        out["stamp"] = all_stamp
    if "images" in out.columns:
        out["images"] = all_images
    else:
        out["images"] = all_images
    out["_ocr_text_blocks"] = all_ocr_text_blocks
    out["_ocr_visual_text_blocks"] = all_ocr_visual_text_blocks
    if extract_text and "text" in out.columns:
        for i, ocr_text in enumerate(all_text):
            if ocr_text is not None:
                out.iat[i, out.columns.get_loc("text")] = ocr_text
    elif extract_text:
        out["text"] = [t if t is not None else "" for t in all_text]
    out["ocr"] = all_ocr_meta
    if "metadata" in out.columns:
        metadata_with_quality: List[Any] = []
        for index, metadata in enumerate(out["metadata"].tolist()):
            quality = all_ocr_meta[index].get("scan_ocr_quality") if index < len(all_ocr_meta) else None
            if quality is None or not isinstance(metadata, dict):
                metadata_with_quality.append(metadata)
            else:
                updated_metadata = dict(metadata)
                updated_metadata["scan_ocr_quality"] = quality
                metadata_with_quality.append(updated_metadata)
        out["metadata"] = metadata_with_quality
    out["ocr_v1_num_detections"] = [m["num_detections"] for m in all_ocr_meta]
    out["ocr_v1_counts_by_label"] = [m["counts_by_label"] for m in all_ocr_meta]
    return out


# ---------------------------------------------------------------------------
# Nemotron Parse v1.2
# ---------------------------------------------------------------------------


def _extract_parse_text(response_item: Any) -> str:
    if response_item is None:
        return ""
    if isinstance(response_item, str):
        return response_item.strip()
    if isinstance(response_item, dict):
        for key in ("generated_text", "text", "output_text", "prediction", "output", "data"):
            value = response_item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, str) and first.strip():
                    return first.strip()
                if isinstance(first, dict):
                    inner = _extract_parse_text(first)
                    if inner:
                        return inner
    if isinstance(response_item, list):
        for item in response_item:
            text = _extract_parse_text(item)
            if text:
                return text
    try:
        return str(response_item).strip()
    except Exception:
        return ""


def nemotron_parse_page_elements(
    batch_df: Any,
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout_s: float = 120.0,
    extract_text: bool = False,
    extract_tables: bool = False,
    extract_charts: bool = False,
    extract_infographics: bool = False,
    task_prompt: str = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>",
    remote_retry: RemoteRetryParams | None = None,
    nim_client: NIMClient | None = None,
    **kwargs: Any,
) -> Any:
    """
    Run Nemotron Parse v1.2 on cropped page elements.

    Emits OCR-compatible content columns (``table``, ``chart``, ``infographic``)
    so this stage can replace the page-elements + OCR pair in pipeline wiring.
    """
    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("nemotron_parse_page_elements currently only supports pandas.DataFrame input.")

    invoke_url = (invoke_url or kwargs.get("nemotron_parse_invoke_url") or "").strip()
    use_remote = bool(invoke_url)
    if not use_remote and model is None:
        raise ValueError("A local `model` is required when `invoke_url` is not provided.")

    wanted_labels: set[str] = set()
    if extract_tables:
        wanted_labels.add("table")
    if extract_charts:
        wanted_labels.add("chart")
    if extract_infographics:
        wanted_labels.add("infographic")

    all_table: List[List[Dict[str, Any]]] = []
    all_chart: List[List[Dict[str, Any]]] = []
    all_infographic: List[List[Dict[str, Any]]] = []
    all_text: List[str] = []
    all_meta: List[Dict[str, Any]] = []

    t0_total = time.perf_counter()

    for row in batch_df.itertuples(index=False):
        table_items: List[Dict[str, Any]] = []
        chart_items: List[Dict[str, Any]] = []
        infographic_items: List[Dict[str, Any]] = []
        row_text: Optional[str] = None
        row_error: Any = None

        try:
            pe = getattr(row, "page_elements_v3", None)
            dets: List[Dict[str, Any]] = []
            if isinstance(pe, dict):
                dets = list(pe.get("detections") or [])
            if not isinstance(dets, list):
                dets = []

            page_image = getattr(row, "page_image", None) or {}
            page_image_b64 = page_image.get("image_b64") if isinstance(page_image, dict) else None
            if not isinstance(page_image_b64, str) or not page_image_b64:
                all_table.append(table_items)
                all_chart.append(chart_items)
                all_infographic.append(infographic_items)
                all_text.append(None)
                all_meta.append({"timing": None, "error": None})
                continue

            if use_remote:
                crops = _crop_all_from_page(page_image_b64, dets, wanted_labels, as_b64=True)
                # Parse-only mode may skip page-elements detection entirely. In that
                # case, parse the full page once and fan out the text to enabled
                # content channels.  The image is already base64 — pass it through.
                if not crops and wanted_labels:
                    crops = [("full_page", [0.0, 0.0, 1.0, 1.0], page_image_b64)]

                crop_b64s: List[str] = [b64 for _label, _bbox, b64 in crops]
                crop_meta: List[Tuple[str, List[float]]] = [(label, bbox) for label, bbox, _b64 in crops]

                if crop_b64s:
                    _invoke_kw = dict(
                        invoke_url=invoke_url,
                        image_b64_list=crop_b64s,
                        api_key=api_key,
                        timeout_s=float(request_timeout_s),
                        max_batch_size=int(kwargs.get("inference_batch_size", 8)),
                        max_retries=int(retry.remote_max_retries),
                        max_429_retries=int(retry.remote_max_429_retries),
                    )
                    if nim_client is not None:
                        response_items = nim_client.invoke_image_inference_batches(**_invoke_kw)
                    else:
                        response_items = invoke_image_inference_batches(
                            **_invoke_kw,
                            max_pool_workers=int(retry.remote_max_pool_workers),
                        )
                    if len(response_items) != len(crop_meta):
                        raise RuntimeError(f"Expected {len(crop_meta)} Parse responses, got {len(response_items)}")

                    for i, (label_name, bbox) in enumerate(crop_meta):
                        text = _extract_parse_text(response_items[i])
                        entry = {"bbox_xyxy_norm": bbox, "text": text}
                        if label_name == "table":
                            table_items.append(entry)
                        elif label_name == "chart":
                            chart_items.append(entry)
                        elif label_name == "infographic":
                            infographic_items.append(entry)
                        elif label_name == "full_page":
                            if extract_tables:
                                table_items.append(dict(entry))
                            if extract_charts:
                                chart_items.append(dict(entry))
                            if extract_infographics:
                                infographic_items.append(dict(entry))
            else:
                crops = _crop_all_from_page(page_image_b64, dets, wanted_labels)
                if not crops and wanted_labels:
                    try:
                        raw = base64.b64decode(page_image_b64)
                        with Image.open(io.BytesIO(raw)) as im0:
                            full_crop = np.asarray(im0.convert("RGB"), dtype=np.uint8).copy()
                        crops = [("full_page", [0.0, 0.0, 1.0, 1.0], full_crop)]
                    except Exception:
                        crops = []
                for label_name, bbox, crop_array in crops:
                    text = str(model.invoke(crop_array, task_prompt=task_prompt) or "").strip()
                    entry = {"bbox_xyxy_norm": bbox, "text": text}
                    if label_name == "table":
                        table_items.append(entry)
                    elif label_name == "chart":
                        chart_items.append(entry)
                    elif label_name == "infographic":
                        infographic_items.append(entry)
                    elif label_name == "full_page":
                        if extract_tables:
                            table_items.append(dict(entry))
                        if extract_charts:
                            chart_items.append(dict(entry))
                        if extract_infographics:
                            infographic_items.append(dict(entry))

            # When extract_text is requested, parse the full page for text
            # (only for pages that need OCR-based text extraction).
            meta = getattr(row, "metadata", None) or {}
            needs_ocr = meta.get("needs_ocr_for_text", False) if isinstance(meta, dict) else False
            if extract_text and needs_ocr:
                try:
                    if use_remote:
                        _text_kw = dict(
                            invoke_url=invoke_url,
                            image_b64_list=[page_image_b64],
                            api_key=api_key,
                            timeout_s=float(request_timeout_s),
                            max_batch_size=1,
                            max_retries=int(retry.remote_max_retries),
                            max_429_retries=int(retry.remote_max_429_retries),
                        )
                        if nim_client is not None:
                            resp = nim_client.invoke_image_inference_batches(**_text_kw)
                        else:
                            resp = invoke_image_inference_batches(
                                **_text_kw,
                                max_pool_workers=int(retry.remote_max_pool_workers),
                            )
                        row_text = _extract_parse_text(resp[0]) if resp else ""
                    else:
                        raw = base64.b64decode(page_image_b64)
                        with Image.open(io.BytesIO(raw)) as im0:
                            full_crop = np.asarray(im0.convert("RGB"), dtype=np.uint8).copy()
                        row_text = str(model.invoke(full_crop, task_prompt=task_prompt) or "").strip()
                except Exception:
                    row_text = ""

        except BaseException as e:
            print(f"Warning: Nemotron Parse failed: {type(e).__name__}: {e}")
            row_error = {
                "stage": "nemotron_parse_page_elements",
                "type": e.__class__.__name__,
                "message": str(e),
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

        all_text.append(row_text)
        all_table.append(table_items)
        all_chart.append(chart_items)
        all_infographic.append(infographic_items)
        all_meta.append({"timing": None, "error": row_error})

    elapsed = time.perf_counter() - t0_total
    for meta in all_meta:
        meta["timing"] = {"seconds": float(elapsed)}

    out = batch_df.copy()
    if extract_text and "text" in out.columns:
        # Only overwrite rows where parse produced text; preserve native text otherwise.
        for i, parse_text in enumerate(all_text):
            if parse_text is not None:
                out.iat[i, out.columns.get_loc("text")] = parse_text
    elif extract_text:
        out["text"] = [t if t is not None else "" for t in all_text]
    out["table"] = all_table
    out["chart"] = all_chart
    out["infographic"] = all_infographic
    # Aliases retained for experiments that read parse-specific columns.
    out["table_parse"] = all_table
    out["chart_parse"] = all_chart
    out["infographic_parse"] = all_infographic
    out["nemotron_parse_v1_2"] = all_meta
    return out
