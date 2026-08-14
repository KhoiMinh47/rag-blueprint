# SPDX-License-Identifier: Apache-2.0

"""Geometry, image-crop, reading-order, and duplicate helpers for OCR options."""

from __future__ import annotations

import base64
import io
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import median
from typing import Any

from nemo_retriever.common.modality.ocr.isolated.contracts import BBox, OCRCandidate

try:
    from PIL import Image, ImageDraw  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - Pillow is a core project dependency.
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CroppedImage:
    """Encoded crop plus the exact page bbox used to create it."""

    bbox_xyxy_norm: BBox
    image_b64: str
    shape_hw: tuple[int, int]


def _mask_crop(
    crop: Any,
    regions: Sequence[Sequence[float]] | None,
    crop_bbox: Sequence[float],
) -> None:
    """White-out layout regions before a page-level OCR request."""
    if not regions or ImageDraw is None or crop is None:
        return
    x0, y0, x1, y1 = [float(value) for value in crop_bbox[:4]]
    span_x = max(1e-9, x1 - x0)
    span_y = max(1e-9, y1 - y0)
    draw = ImageDraw.Draw(crop)
    crop_width, crop_height = crop.size
    for region in regions:
        normalized = clamp_bbox(region)
        if normalized is None:
            continue
        left = max(0.0, (normalized[0] - x0) / span_x)
        top = max(0.0, (normalized[1] - y0) / span_y)
        right = min(1.0, (normalized[2] - x0) / span_x)
        bottom = min(1.0, (normalized[3] - y0) / span_y)
        if right <= left or bottom <= top:
            continue
        draw.rectangle(
            (
                round(left * crop_width),
                round(top * crop_height),
                round(right * crop_width),
                round(bottom * crop_height),
            ),
            fill=(255, 255, 255),
        )


class PageImageCropper:
    """Decode one page once and produce pixel-identical PNG crops from it."""

    def __init__(
        self,
        image_b64: str,
        *,
        output_format: str = "PNG",
        jpeg_quality: int = 95,
    ) -> None:
        if Image is None or not isinstance(image_b64, str) or not image_b64:
            raise ValueError("page image is unavailable")
        value = image_b64.split(",", 1)[1] if image_b64.startswith("data:") else image_b64
        with Image.open(io.BytesIO(base64.b64decode(value))) as source:
            self.image = source.convert("RGB")
            self.image.load()
        self.width, self.height = self.image.size
        normalized_format = str(output_format or "PNG").strip().upper()
        self.output_format = "JPEG" if normalized_format in {"JPG", "JPEG"} else "PNG"
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))

    def crop(
        self,
        bbox: Sequence[float],
        *,
        local_text_height: float | None = None,
        add_padding: bool = False,
        mask_regions: Sequence[Sequence[float]] | None = None,
    ) -> CroppedImage | None:
        normalized = clamp_bbox(bbox)
        if normalized is None:
            return None
        crop_bbox = normalized
        if add_padding:
            crop_bbox = expand_bbox_adaptive(
                normalized,
                local_text_height=float(local_text_height or (normalized[3] - normalized[1])),
                image_shape_hw=(self.height, self.width),
            )
        left = max(0, min(self.width - 1, round(crop_bbox[0] * self.width)))
        top = max(0, min(self.height - 1, round(crop_bbox[1] * self.height)))
        right = max(left + 1, min(self.width, round(crop_bbox[2] * self.width)))
        bottom = max(top + 1, min(self.height, round(crop_bbox[3] * self.height)))
        if right <= left or bottom <= top:
            return None
        crop = self.image.crop((left, top, right, bottom))
        try:
            _mask_crop(crop, mask_regions, crop_bbox)
            buffer = io.BytesIO()
            if self.output_format == "JPEG":
                # OCR crops do not need lossless alpha/metadata.  A high
                # quality JPEG is materially cheaper to encode than PNG on a
                # large scanned page while retaining glyph edges.
                crop.save(
                    buffer,
                    format="JPEG",
                    quality=self.jpeg_quality,
                    optimize=False,
                )
            else:
                crop.save(buffer, format="PNG")
        finally:
            crop.close()
        return CroppedImage(
            bbox_xyxy_norm=(
                left / self.width,
                top / self.height,
                right / self.width,
                bottom / self.height,
            ),
            image_b64=base64.b64encode(buffer.getvalue()).decode("ascii"),
            shape_hw=(bottom - top, right - left),
        )


def clamp_bbox(value: Sequence[float] | None) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in values):
        return None
    x0, y0, x1, y1 = values
    result = (
        max(0.0, min(1.0, min(x0, x1))),
        max(0.0, min(1.0, min(y0, y1))),
        max(0.0, min(1.0, max(x0, x1))),
        max(0.0, min(1.0, max(y0, y1))),
    )
    if result[2] <= result[0] or result[3] <= result[1]:
        return None
    return result


def bbox_area(bbox: Sequence[float]) -> float:
    if len(bbox) != 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
        0.0, float(bbox[3]) - float(bbox[1])
    )


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = bbox_area(left) + bbox_area(right) - intersection
    return intersection / union if union > 0.0 else 0.0


def containment(left: Sequence[float], right: Sequence[float]) -> float:
    """Return intersection over the smaller bbox area."""
    smaller = min(bbox_area(left), bbox_area(right))
    if smaller <= 0.0:
        return 0.0
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    return max(0.0, x1 - x0) * max(0.0, y1 - y0) / smaller


def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    return (
        (float(bbox[0]) + float(bbox[2])) / 2.0,
        (float(bbox[1]) + float(bbox[3])) / 2.0,
    )


def map_local_bbox(
    local: Sequence[float] | None, parent: Sequence[float], shape_hw: Sequence[int]
) -> BBox:
    """Map normalized or pixel detector coordinates into page coordinates."""
    parent_box = clamp_bbox(parent) or (0.0, 0.0, 1.0, 1.0)
    if not isinstance(local, (list, tuple)) or len(local) != 4:
        return parent_box
    try:
        values = [float(item) for item in local]
    except (TypeError, ValueError):
        return parent_box
    height = max(1, int(shape_hw[0])) if len(shape_hw) >= 1 else 1
    width = max(1, int(shape_hw[1])) if len(shape_hw) >= 2 else 1
    if max(abs(item) for item in values) <= 1.5:
        normalized = values
    else:
        normalized = [
            values[0] / width,
            values[1] / height,
            values[2] / width,
            values[3] / height,
        ]
    normalized_box = clamp_bbox(normalized)
    if normalized_box is None:
        return parent_box
    px0, py0, px1, py1 = parent_box
    return (
        clamp_bbox(
            (
                px0 + normalized_box[0] * (px1 - px0),
                py0 + normalized_box[1] * (py1 - py0),
                px0 + normalized_box[2] * (px1 - px0),
                py0 + normalized_box[3] * (py1 - py0),
            )
        )
        or parent_box
    )


def image_size(image_b64: str) -> tuple[int, int] | None:
    if Image is None or not isinstance(image_b64, str) or not image_b64:
        return None
    try:
        value = (
            image_b64.split(",", 1)[1] if image_b64.startswith("data:") else image_b64
        )
        with Image.open(io.BytesIO(base64.b64decode(value))) as image:
            width, height = image.size
            return (int(height), int(width))
    except Exception:  # noqa: BLE001 - malformed image input is page-local.
        return None


def adaptive_local_text_height(
    bbox: Sequence[float], neighboring_boxes: Iterable[Sequence[float]]
) -> float:
    """Estimate local text height from nearby detector geometry.

    The estimate is expressed in normalized page units and scales with the
    page's own detections.  There is no fixed pixel line-height threshold, so
    large headings, small body text, ascenders, descenders, and Vietnamese
    combining marks all get padding relative to their local context.
    """
    current = clamp_bbox(bbox)
    if current is None:
        return 0.0
    current_height = current[3] - current[1]
    current_center = bbox_center(current)
    heights: list[float] = []
    for candidate in neighboring_boxes:
        normalized = clamp_bbox(candidate)
        if normalized is None:
            continue
        height = normalized[3] - normalized[1]
        if height <= 0.0:
            continue
        center = bbox_center(normalized)
        vertical_distance = abs(center[1] - current_center[1])
        horizontal_overlap = max(
            0.0, min(current[2], normalized[2]) - max(current[0], normalized[0])
        )
        if (
            normalized == current
            or horizontal_overlap > 0.0
            or vertical_distance <= max(current_height, height)
        ):
            heights.append(height)
    if not heights:
        heights = [current_height]
    return max(1e-9, float(median(heights)))


def adaptive_padding(
    bbox: Sequence[float],
    *,
    local_text_height: float,
    image_shape_hw: Sequence[int],
) -> tuple[float, float]:
    """Return horizontal/vertical normalized padding from local glyph scale."""
    normalized = clamp_bbox(bbox)
    if normalized is None:
        return (0.0, 0.0)
    height, width = max(1, int(image_shape_hw[0])), max(1, int(image_shape_hw[1]))
    bbox_height = normalized[3] - normalized[1]
    bbox_width = normalized[2] - normalized[0]
    local_height = max(float(local_text_height), bbox_height * 0.25, 1e-9)
    # The pixel terms only protect a one-pixel border on very small rasters;
    # the actual crop margin is driven by local detector geometry.
    vertical = max(local_height * 0.35, 1.0 / height)
    horizontal = max(local_height * 0.18, 1.0 / width)
    return (min(horizontal, bbox_width * 0.45), min(vertical, bbox_height * 0.45))


def expand_bbox_adaptive(
    bbox: Sequence[float],
    *,
    local_text_height: float,
    image_shape_hw: Sequence[int],
) -> BBox:
    normalized = clamp_bbox(bbox) or (0.0, 0.0, 1.0, 1.0)
    horizontal, vertical = adaptive_padding(
        normalized,
        local_text_height=local_text_height,
        image_shape_hw=image_shape_hw,
    )
    return (
        clamp_bbox(
            (
                normalized[0] - horizontal,
                normalized[1] - vertical,
                normalized[2] + horizontal,
                normalized[3] + vertical,
            )
        )
        or normalized
    )


def crop_image_b64(
    image_b64: str,
    bbox: Sequence[float],
    *,
    local_text_height: float | None = None,
    add_padding: bool = False,
    mask_regions: Sequence[Sequence[float]] | None = None,
    output_format: str = "PNG",
    jpeg_quality: int = 95,
) -> CroppedImage | None:
    """Crop a page and preserve the exact normalized crop geometry."""
    if Image is None or not isinstance(image_b64, str) or not image_b64:
        return None
    shape = image_size(image_b64)
    normalized = clamp_bbox(bbox)
    if shape is None or normalized is None:
        return None
    crop_bbox = normalized
    if add_padding:
        crop_bbox = expand_bbox_adaptive(
            normalized,
            local_text_height=float(
                local_text_height or (normalized[3] - normalized[1])
            ),
            image_shape_hw=shape,
        )
    height, width = shape
    try:
        value = (
            image_b64.split(",", 1)[1] if image_b64.startswith("data:") else image_b64
        )
        with Image.open(io.BytesIO(base64.b64decode(value))) as source:
            image = source.convert("RGB")
            left = max(0, min(width - 1, round(crop_bbox[0] * width)))
            top = max(0, min(height - 1, round(crop_bbox[1] * height)))
            right = max(left + 1, min(width, round(crop_bbox[2] * width)))
            bottom = max(top + 1, min(height, round(crop_bbox[3] * height)))
            if right <= left or bottom <= top:
                return None
            crop = image.crop((left, top, right, bottom))
            _mask_crop(crop, mask_regions, crop_bbox)
            buffer = io.BytesIO()
            normalized_format = str(output_format or "PNG").strip().upper()
            if normalized_format in {"JPG", "JPEG"}:
                crop.save(
                    buffer,
                    format="JPEG",
                    quality=max(1, min(100, int(jpeg_quality))),
                    optimize=False,
                )
            else:
                crop.save(buffer, format="PNG")
            crop.close()
            return CroppedImage(
                bbox_xyxy_norm=(
                    left / width,
                    top / height,
                    right / width,
                    bottom / height,
                ),
                image_b64=base64.b64encode(buffer.getvalue()).decode("ascii"),
                shape_hw=(bottom - top, right - left),
            )
    except Exception:  # noqa: BLE001 - malformed image input is page-local.
        return None


def tile_bboxes(
    shape_hw: Sequence[int], *, tile_size: int, overlap: float
) -> list[BBox]:
    """Return gap-free overlapping tile boxes in page-normalized coordinates."""
    height, width = max(1, int(shape_hw[0])), max(1, int(shape_hw[1]))
    size = max(1, int(tile_size))
    if width <= size and height <= size:
        return []
    overlap = max(0.0, min(0.8, float(overlap)))
    stride = max(1, round(size * (1.0 - overlap)))

    def starts(length: int) -> list[int]:
        if length <= size:
            return [0]
        values = list(range(0, length - size + 1, stride))
        last = length - size
        if not values or values[-1] != last:
            values.append(last)
        return sorted(set(values))

    result: list[BBox] = []
    for top in starts(height):
        for left in starts(width):
            result.append(
                (
                    left / width,
                    top / height,
                    min(width, left + size) / width,
                    min(height, top + size) / height,
                )
            )
    return result


def normalized_text(value: Any) -> str:
    """Unicode-normalized key for comparison, retaining Vietnamese letters."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return " ".join(text.split())


def text_identity(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).casefold().replace("đ", "d")
    return "".join(
        char for char in text if not unicodedata.combining(char) and char.isalnum()
    )


def text_similarity(left: Any, right: Any) -> float:
    left_key, right_key = text_identity(left), text_identity(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    shorter, longer = sorted((left_key, right_key), key=len)
    if len(shorter) >= 4 and shorter in longer:
        return len(shorter) / max(1, len(longer))
    return SequenceMatcher(None, left_key, right_key).ratio()


def reading_order_key(candidate: OCRCandidate) -> tuple[float, float, int]:
    box = candidate.bbox_xyxy_norm
    return (float(box[1]), float(box[0]), int(candidate.reading_order))


def _quality(text: str, score: float | None) -> float:
    value = 0.5 if score is None else max(0.0, min(1.0, float(score)))
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    replacement_penalty = sum(char in {"�", "□"} for char in visible) / len(visible)
    control_penalty = sum(unicodedata.category(char) == "Cc" for char in visible) / len(
        visible
    )
    return max(
        0.0,
        min(1.0, 0.65 * value + 0.35 * (1.0 - replacement_penalty - control_penalty)),
    )


def _same_reading_line(left: OCRCandidate, right: OCRCandidate) -> bool:
    left_center = bbox_center(left.bbox_xyxy_norm)
    right_center = bbox_center(right.bbox_xyxy_norm)
    left_height = max(1e-9, left.bbox_xyxy_norm[3] - left.bbox_xyxy_norm[1])
    right_height = max(1e-9, right.bbox_xyxy_norm[3] - right.bbox_xyxy_norm[1])
    return abs(left_center[1] - right_center[1]) <= 0.75 * max(
        left_height, right_height
    )


def candidates_duplicate(left: OCRCandidate, right: OCRCandidate) -> bool:
    if left.content_type != right.content_type and {
        left.content_type,
        right.content_type,
    } != {"text", "title"}:
        return False
    if left.content_type == "table_cell" and (
        (left.table_id or left.provenance.get("table_id"))
        != (right.table_id or right.provenance.get("table_id"))
        or (left.cell_id or left.provenance.get("cell_id"))
        != (right.cell_id or right.provenance.get("cell_id"))
    ):
        return False
    similarity = text_similarity(left.text, right.text)
    geometry = max(
        bbox_iou(left.bbox_xyxy_norm, right.bbox_xyxy_norm),
        containment(left.bbox_xyxy_norm, right.bbox_xyxy_norm),
    )
    if geometry < 0.35 or not _same_reading_line(left, right):
        return False
    return similarity >= 0.78 or geometry >= 0.82


def merge_candidates(candidates: Iterable[OCRCandidate]) -> list[OCRCandidate]:
    """Merge recall/crop duplicates while retaining all provenance."""
    merged: list[OCRCandidate] = []
    for candidate in candidates:
        if not candidate.text.strip():
            continue
        duplicate_index = next(
            (
                index
                for index, previous in enumerate(merged)
                if candidates_duplicate(candidate, previous)
            ),
            None,
        )
        if duplicate_index is None:
            merged.append(candidate)
            continue
        previous = merged[duplicate_index]
        sources = list(previous.provenance.get("sources", [previous.source]))
        if candidate.source not in sources:
            sources.append(candidate.source)
        previous_quality = _quality(previous.text, previous.score)
        candidate_quality = _quality(candidate.text, candidate.score)
        previous_is_title = (
            bool(previous.provenance.get("title_priority"))
            or previous.content_type == "title"
        )
        candidate_is_title = (
            bool(candidate.provenance.get("title_priority"))
            or candidate.content_type == "title"
        )
        if candidate_is_title and not previous_is_title:
            winner, loser = candidate, previous
        elif previous_is_title and not candidate_is_title:
            winner, loser = previous, candidate
        elif candidate_quality > previous_quality + 0.02:
            winner, loser = candidate, previous
        else:
            winner, loser = previous, candidate
        winner.provenance = {
            **loser.provenance,
            **winner.provenance,
            "sources": sources,
            "merged_duplicate": True,
            "duplicate_text": loser.text,
            "duplicate_bbox_xyxy_norm": list(loser.bbox_xyxy_norm),
        }
        winner.candidates = list(previous.candidates) + list(candidate.candidates)
        merged[duplicate_index] = winner
    return sorted(merged, key=reading_order_key)


def union_bbox(candidates: Iterable[OCRCandidate]) -> BBox | None:
    values = [candidate.bbox_xyxy_norm for candidate in candidates]
    if not values:
        return None
    return (
        min(value[0] for value in values),
        min(value[1] for value in values),
        max(value[2] for value in values),
        max(value[3] for value in values),
    )


def overlap_fraction(candidate: Sequence[float], region: Sequence[float]) -> float:
    area = bbox_area(candidate)
    return bbox_iou(candidate, region) * (bbox_area(region) / area if area else 0.0)


def inside_or_overlaps(
    candidate: Sequence[float],
    regions: Iterable[Sequence[float]],
    *,
    threshold: float = 0.45,
) -> bool:
    center = bbox_center(candidate)
    for region in regions:
        normalized = clamp_bbox(region)
        if normalized is None:
            continue
        center_inside = (
            normalized[0] <= center[0] <= normalized[2]
            and normalized[1] <= center[1] <= normalized[3]
        )
        if (
            center_inside
            or overlap_fraction(candidate, normalized) >= threshold
            or containment(candidate, normalized) >= threshold
        ):
            return True
    return False


def language_quality(text: str, language: str | None) -> float:
    """Small deterministic language signal used by Option 4 fusion."""
    if not text:
        return 0.0
    vietnamese = set("ăâđêôơưĂÂĐÊÔƠƯ")
    has_vietnamese = any(char in vietnamese for char in text)
    requested = str(language or "").lower()
    if "vie" in requested and has_vietnamese:
        return 1.0
    if "vie" in requested and not has_vietnamese:
        return 0.55
    return 0.7 if all(ord(char) < 128 or char.isspace() for char in text) else 0.6


def numeric_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*", text or "")
