# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Build the small visual payload used by the dashboard inspector.

The ingest pipeline keeps the full page raster and geometry while it is
running.  The public result serializer intentionally removes that bulky
payload.  This module extracts only what the visual inspector needs:

* one page image per source page;
* normalized ``xyxy`` bounding boxes;
* the text associated with each box; and
* lightweight reader/content provenance.

It is deliberately an additive projection.  It does not change chunking,
OCR, layout detection, or the canonical result rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from difflib import SequenceMatcher
from typing import Any
import math
import numbers
import re
import unicodedata


_MAX_BLOCK_TEXT = 12_000
_NESTED_CONTENT_KEYS = (
    "table",
    "tables",
    "chart",
    "charts",
    "infographic",
    "infographics",
    "stamp",
    "stamps",
    "images",
    "page_elements",
    "page_elements_v3",
)
_VISUAL_TYPES = {"table", "image", "chart", "infographic", "stamp"}
_DEDUP_BBOX_DECIMALS = 3
_DEDUP_CONTAINMENT = 0.80
_DEDUP_GEOMETRY_CONTAINMENT = 0.88
_DEDUP_GEOMETRY_IOU = 0.72
_DEDUP_TEXT_SIMILARITY = 0.78
_DEDUP_EDGE_TOLERANCE = 0.008


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _json_safe(value: Any) -> Any:
    """Replace non-finite/model scalar values before a strict JSON response."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            return None
        if isinstance(value, numbers.Integral):
            return int(value)
        return number
    # pandas/numpy scalars that are not registered as numbers.Real still
    # expose item(); convert those without importing either dependency here.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
        except (TypeError, ValueError):
            converted = value
        if converted is not value:
            return _json_safe(converted)
    return value


def _as_page_number(row: Mapping[str, Any], fallback: int = 1) -> int:
    metadata = _as_mapping(row.get("metadata"))
    nested = _as_mapping(metadata.get("content_metadata"))
    for value in (row.get("page_number"), metadata.get("page_number"), nested.get("page_number")):
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return fallback


def _normalise_bbox(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            import json

            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        box = [float(value[index]) for index in range(4)]
    except (TypeError, ValueError):
        return None
    if any(number != number or number in (float("inf"), float("-inf")) for number in box):
        return None
    if any(number < 0.0 or number > 1.0 for number in box):
        return None
    x0, y0, x1, y1 = box
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _bbox_from(value: Mapping[str, Any]) -> list[float] | None:
    metadata = _as_mapping(value.get("metadata"))
    nested = _as_mapping(value.get("content_metadata"))
    for container in (value, metadata, nested):
        for key in ("_bbox_xyxy_norm", "bbox_xyxy_norm"):
            bbox = _normalise_bbox(container.get(key))
            if bbox is not None:
                return bbox
    return None


def _text_from(value: Mapping[str, Any]) -> str:
    metadata = _as_mapping(value.get("metadata"))
    for candidate in (value.get("text"), value.get("content"), value.get("markdown"), metadata.get("content")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()[:_MAX_BLOCK_TEXT]
    return ""


def _reader_from(value: Mapping[str, Any]) -> str | None:
    metadata = _as_mapping(value.get("metadata"))
    nested = _as_mapping(metadata.get("content_metadata"))
    for candidate in (value.get("_reader_backend"), metadata.get("reader_backend"), nested.get("reader_backend")):
        if candidate:
            return str(candidate)
    return None


def _content_type_from(value: Mapping[str, Any], fallback: str = "text") -> str:
    metadata = _as_mapping(value.get("metadata"))
    return str(
        value.get("_content_type")
        or value.get("content_type")
        or value.get("element_type")
        or value.get("label_name")
        or metadata.get("_content_type")
        or metadata.get("content_type")
        or fallback
    )


def _page_image_from(row: Mapping[str, Any]) -> dict[str, Any] | None:
    page_image = row.get("page_image")
    if not isinstance(page_image, Mapping):
        return None
    image_b64 = page_image.get("image_b64")
    if not isinstance(image_b64, str) or not image_b64.strip():
        return None
    encoding = str(page_image.get("encoding") or "png").lower().replace(".", "")
    if encoding in {"jpg", "jpeg"}:
        mime = "image/jpeg"
    elif encoding == "webp":
        mime = "image/webp"
    else:
        encoding = "png"
        mime = "image/png"
    shape = page_image.get("orig_shape_hw") or page_image.get("shape_hw")
    height = width = None
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        try:
            height, width = int(shape[0]), int(shape[1])
        except (TypeError, ValueError):
            height = width = None
    return {
        "image_b64": image_b64,
        "mime": mime,
        "encoding": encoding,
        "width": width,
        "height": height,
    }


def _iter_rows(data: Any) -> list[Mapping[str, Any]]:
    if hasattr(data, "to_dict"):
        try:
            data = data.to_dict(orient="records")
        except (AttributeError, TypeError, ValueError):
            pass
    if not isinstance(data, Iterable) or isinstance(data, (str, bytes, bytearray, Mapping)):
        return []
    return [row for row in data if isinstance(row, Mapping)]


def _dedupe_text(value: Any) -> str:
    """Normalize OCR text for duplicate detection only.

    This is intentionally not used as answer text.  Removing accents and
    punctuation lets us recognize OCR variants such as ``công``/``cong`` or
    ``.vn``/``vn`` when they point to the same visual box.
    """
    text = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", str(value or "")).casefold())
    text = text.replace("đ", "d")
    return "".join(character for character in text if not unicodedata.combining(character) and character.isalnum())


def _text_is_related(left: Any, right: Any) -> bool:
    """Recognize OCR variants without treating every nearby line as a duplicate."""
    left_key = _dedupe_text(left)
    right_key = _dedupe_text(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    shorter, longer = sorted((left_key, right_key), key=len)
    if len(shorter) >= 4 and shorter in longer:
        return True
    if len(shorter) < 8:
        return False
    return SequenceMatcher(None, left_key, right_key).ratio() >= _DEDUP_TEXT_SIMILARITY


def _text_identity(value: Any) -> str:
    """Keep a small identity key for punctuation-only OCR blocks."""
    return " ".join(str(value or "").casefold().split())


def _bbox_edges_close(left: list[float], right: list[float]) -> bool:
    return max(abs(float(a) - float(b)) for a, b in zip(left, right)) <= _DEDUP_EDGE_TOLERANCE


def _bbox_shapes_similar(left: list[float], right: list[float]) -> bool:
    """Reject paragraph-vs-line containment while accepting crop drift."""
    left_width = max(0.0, left[2] - left[0])
    right_width = max(0.0, right[2] - right[0])
    left_height = max(0.0, left[3] - left[1])
    right_height = max(0.0, right[3] - right[1])
    if min(left_width, right_width) <= 0.0 or min(left_height, right_height) <= 0.0:
        return False
    return (
        min(left_width, right_width) / max(left_width, right_width) >= 0.65
        and min(left_height, right_height) / max(left_height, right_height) >= 0.55
    )


def _text_quality(value: Any) -> float:
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


def _bbox_area(bbox: list[float] | tuple[float, ...] | None) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_intersection(left: list[float], right: list[float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _bbox_intersection_over_area(left: list[float], right: list[float], *, area: str) -> float:
    intersection = _bbox_intersection(left, right)
    if area == "left":
        denominator = _bbox_area(left)
    else:
        denominator = _bbox_area(right)
    return intersection / denominator if denominator > 0.0 else 0.0


def _bbox_iou(left: list[float], right: list[float]) -> float:
    intersection = _bbox_intersection(left, right)
    union = _bbox_area(left) + _bbox_area(right) - intersection
    return intersection / union if union > 0.0 else 0.0


def _bbox_geometry_is_duplicate(left: list[float], right: list[float]) -> bool:
    """Detect nested/near-identical boxes without looking at OCR spelling.

    Page Elements can emit a broad text region and a second text region for
    the same pixels.  OCR text is not a safe prerequisite here because two
    recognizer passes may disagree on Vietnamese accents or small glyphs.
    Adjacent lines do not satisfy either condition: they have low containment
    and low IoU even when their edges touch.
    """
    smaller = min(_bbox_area(left), _bbox_area(right))
    if smaller <= 0.0:
        return False
    contained = _bbox_intersection(left, right) / smaller
    return contained >= _DEDUP_GEOMETRY_CONTAINMENT or _bbox_iou(left, right) >= _DEDUP_GEOMETRY_IOU


def _is_option3_parent_bbox_block(value: Mapping[str, Any]) -> bool:
    provenance = value.get("provenance")
    source = str(value.get("ocr_source") or value.get("source") or "")
    return (
        source.startswith("option3_")
        and isinstance(provenance, Mapping)
        and provenance.get("bbox_fallback") is True
    )


def _is_duplicate_block(existing: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    """Return whether two projected blocks represent one hover target.

    Final rows can contain two OCR/native variants with tiny coordinate
    differences.  Text equality alone is not enough because the same word
    can legitimately occur many times on a page; geometry is always required
    when both blocks have a bbox.
    """
    existing_type = str(existing.get("content_type") or "text")
    candidate_type = str(candidate.get("content_type") or "text")
    # A chart/image bbox may contain a caption or label. Different semantic
    # types are separate inspector targets even when their rectangles and
    # text happen to match; otherwise fuzzy dedup makes the visual block
    # swallow the text block in the dashboard.
    if existing_type != candidate_type:
        return False
    # A Nemotron crop can return several recognition items without local
    # boxes.  They intentionally share the semantic-unit bbox; do not let
    # the visual projection drop distinct lines based on that fallback
    # geometry.  This condition is scoped to Option 3 provenance only.
    if _is_option3_parent_bbox_block(existing) or _is_option3_parent_bbox_block(candidate):
        return False

    existing_bbox = existing.get("bbox")
    candidate_bbox = candidate.get("bbox")
    existing_text = _dedupe_text(existing.get("text"))
    candidate_text = _dedupe_text(candidate.get("text"))

    if existing_bbox is None or candidate_bbox is None:
        return existing_bbox is None and candidate_bbox is None and _text_is_related(existing_text, candidate_text)

    # Same semantic type plus near-total geometric overlap is a detector
    # duplicate even when OCR strings differ. Keep the full raw OCR text from
    # the preferred representative; never normalize or strip accents here.
    if _bbox_geometry_is_duplicate(existing_bbox, candidate_bbox):
        return True

    rounded_existing = tuple(round(float(value), _DEDUP_BBOX_DECIMALS) for value in existing_bbox)
    rounded_candidate = tuple(round(float(value), _DEDUP_BBOX_DECIMALS) for value in candidate_bbox)
    if rounded_existing == rounded_candidate:
        return True
    if _bbox_edges_close(existing_bbox, candidate_bbox):
        return True

    # If one OCR box contains almost all of another box and their normalized
    # text is equal or one is a longer OCR variant, keep the more complete
    # block instead of drawing two nearly coincident rectangles.
    geometry_is_strong = _bbox_shapes_similar(existing_bbox, candidate_bbox)
    same_text_identity = _text_identity(existing.get("text")) == _text_identity(candidate.get("text"))
    if not (_text_is_related(existing_text, candidate_text) or geometry_is_strong or same_text_identity):
        return False
    smaller_area = min(_bbox_area(existing_bbox), _bbox_area(candidate_bbox))
    return smaller_area > 0.0 and _bbox_intersection(existing_bbox, candidate_bbox) / smaller_area >= _DEDUP_CONTAINMENT


def _prefer_block(candidate: Mapping[str, Any], existing: Mapping[str, Any]) -> bool:
    """Choose the more useful representative when duplicate boxes collide."""
    candidate_text = str(candidate.get("text") or "").strip()
    existing_text = str(existing.get("text") or "").strip()
    if bool(candidate_text) != bool(existing_text):
        return bool(candidate_text)

    # For a nested detector duplicate with different OCR text, prefer the
    # more complete region. A child line can be more confident than its
    # parent, but choosing it would discard the parent's remaining lines.
    candidate_bbox = candidate.get("bbox")
    existing_bbox = existing.get("bbox")
    if isinstance(candidate_bbox, list) and isinstance(existing_bbox, list):
        if _bbox_geometry_is_duplicate(candidate_bbox, existing_bbox):
            candidate_length = len(candidate_text)
            existing_length = len(existing_text)
            if candidate_length > max(12, int(existing_length * 1.15)):
                return True
            if existing_length > max(12, int(candidate_length * 1.15)):
                return False
    candidate_confidence = candidate.get("confidence")
    existing_confidence = existing.get("confidence")
    if candidate_confidence is not None and existing_confidence is None:
        try:
            if float(candidate_confidence) >= 0.82:
                return True
        except (TypeError, ValueError):
            pass
    elif candidate_confidence is not None and existing_confidence is not None:
        try:
            confidence_delta = float(candidate_confidence) - float(existing_confidence)
            if abs(confidence_delta) > 0.025:
                return confidence_delta > 0.0
            if abs(confidence_delta) <= 0.025:
                quality_delta = _text_quality(candidate_text) - _text_quality(existing_text)
                if abs(quality_delta) > 0.03:
                    return quality_delta > 0.0
        except (TypeError, ValueError):
            pass
    elif candidate_confidence is None and existing_confidence is None:
        quality_delta = _text_quality(candidate_text) - _text_quality(existing_text)
        if abs(quality_delta) > 0.03:
            return quality_delta > 0.0
    if len(candidate_text) != len(existing_text):
        return len(candidate_text) > len(existing_text)
    return candidate.get("origin") == "canonical_row" and existing.get("origin") != "canonical_row"


def _append_deduplicated_block(blocks: list[dict[str, Any]], candidate: dict[str, Any]) -> None:
    for index, existing in enumerate(blocks):
        if not _is_duplicate_block(existing, candidate):
            continue
        if _prefer_block(candidate, existing):
            block_id = existing.get("id")
            merged = dict(existing)
            for key, value in candidate.items():
                # Do not erase useful geometry/provenance with a missing
                # field from an alternate OCR representation.
                if value is not None or merged.get(key) in (None, ""):
                    merged[key] = value
            if block_id:
                merged["id"] = block_id
            blocks[index] = merged
        return
    blocks.append(candidate)


def _deduplicate_blocks(blocks: Any, page_number: int | None = None) -> list[dict[str, Any]]:
    """Normalize/deduplicate blocks from both new and older sidecars."""
    result: list[dict[str, Any]] = []
    for block in blocks if isinstance(blocks, (list, tuple)) else []:
        if not isinstance(block, Mapping):
            continue
        candidate = dict(block)
        # PDFium exposes a scan as a page-sized raster. That raster is the
        # background served by the page-image endpoint, not a semantic image
        # detected by Page Elements. The tolerance also cleans old sidecars
        # whose box ended at 0.998 instead of exactly 1.0.
        if (
            candidate.get("content_type") == "image"
            and not str(candidate.get("text") or "").strip()
            and _bbox_area(candidate.get("bbox")) >= 0.90
            and candidate.get("image_type") != "detected_region"
        ):
            continue
        _append_deduplicated_block(result, candidate)
    for index, block in enumerate(result):
        if not block.get("id"):
            block["id"] = f"p{page_number or 1}-b{index}"
    return result


def _annotate_region_overlaps(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep text blocks and visual regions separate while recording overlap.

    Overlap is provenance, not a deduplication signal.  A detector's chart
    bbox can be wider than the plotted chart and touch a native text line; the
    text must remain a text block even when it is visually inside that bbox.
    """
    # Rebuilding an already annotated sidecar must not append the same
    # relationship repeatedly on every dashboard request.
    for block in blocks:
        if str(block.get("content_type") or "") in _VISUAL_TYPES:
            block.pop("contains_text_blocks", None)
        if str(block.get("content_type") or "text") == "text":
            block.pop("overlaps_regions", None)

    regions = [
        block for block in blocks
        if str(block.get("content_type") or "") in _VISUAL_TYPES and block.get("bbox")
    ]
    text_blocks = [
        block for block in blocks
        if str(block.get("content_type") or "text") == "text" and block.get("bbox")
    ]
    for text_block in text_blocks:
        text_bbox = text_block.get("bbox")
        matches: list[dict[str, Any]] = []
        for region in regions:
            region_bbox = region.get("bbox")
            if not isinstance(text_bbox, list) or not isinstance(region_bbox, list):
                continue
            text_area_overlap = _bbox_intersection_over_area(text_bbox, region_bbox, area="left")
            center_x = (text_bbox[0] + text_bbox[2]) / 2.0
            center_y = (text_bbox[1] + text_bbox[3]) / 2.0
            center_inside = (
                region_bbox[0] <= center_x <= region_bbox[2]
                and region_bbox[1] <= center_y <= region_bbox[3]
            )
            if text_area_overlap < 0.05 and not center_inside:
                continue
            matches.append({
                "block_id": region.get("id"),
                "content_type": region.get("content_type"),
                "bbox": region_bbox,
                "overlap_of_text": round(text_area_overlap, 6),
            })
        if matches:
            text_block["overlaps_regions"] = matches
            for match in matches:
                region = next((item for item in regions if item.get("id") == match["block_id"]), None)
                if region is not None:
                    region.setdefault("contains_text_blocks", []).append(text_block.get("id"))
    return blocks


def deduplicate_visual_blocks(
    blocks: Any,
    page_number: int | None = None,
    *,
    suppress_page_sized_visual_noise: bool = False,
) -> list[dict[str, Any]]:
    """Return canonical page blocks for both visual sidecars and trace API.

    The dashboard trace and the visual sidecar must use the same geometry
    policy. Otherwise the PDF overlay can still draw duplicate text boxes even
    after the visual endpoint has cleaned them.
    """
    result = _deduplicate_blocks(blocks, page_number)
    result = [
        block for _index, block in sorted(
            enumerate(result), key=lambda item: _block_geometry_order(item[1], item[0])
        )
    ]
    if suppress_page_sized_visual_noise:
        # Pipeline 6 sends the complete page to OCR when Page Elements emits a
        # near-page-sized image/infographic candidate. If the sidecar kept
        # that raw detector record, the dashboard would show the whole page as
        # a "sơ đồ" crop even though the candidate was rejected by the visual
        # review gate. Keep standalone visual crops, remove only page-sized
        # image/chart/infographic/stamp records on text-bearing pages.
        has_text = any(
            str(block.get("text") or "").strip()
            and str(block.get("content_type") or "text") not in _VISUAL_TYPES
            for block in result
        )
        if has_text:
            result = [
                block
                for block in result
                if not (
                    str(block.get("content_type") or "")
                    in {"image", "chart", "infographic", "stamp"}
                    and _bbox_area(block.get("bbox")) >= 0.80
                )
            ]
        # Empty title/header/footer detections are geometry hints, not parsed
        # output. Keep image/table evidence even when its caption is empty,
        # but do not publish empty non-visual boxes to the dashboard.
        result = [
            block
            for block in result
            if str(block.get("content_type") or "text") in _VISUAL_TYPES
            or str(block.get("text") or "").strip()
        ]
    return _annotate_region_overlaps(result)


def _suppress_option2_title_overlays(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove Page Elements title overlays already represented by OCR text.

    Page Elements does not guarantee ordering between its ``title`` records
    and the OCR line records projected from the canonical Option 2 rows. A
    streaming check in ``build_visual_evidence`` therefore cannot reliably
    see the text block before the title. Apply this narrow, Option 2-only
    cleanup after geometry deduplication so a title is removed only when a
    text block occupies the same visual region. Independent titles remain
    available in the inspector.
    """
    text_blocks = [
        block
        for block in blocks
        if str(block.get("content_type") or "text") == "text"
        and isinstance(block.get("bbox"), list)
    ]
    result: list[dict[str, Any]] = []
    for block in blocks:
        if (
            str(block.get("content_type") or "") == "title"
            and isinstance(block.get("bbox"), list)
            and any(_bbox_geometry_is_duplicate(block["bbox"], text["bbox"]) for text in text_blocks)
        ):
            continue
        result.append(block)
    return result


def _block_geometry_order(block: Mapping[str, Any], index: int) -> tuple[float, float, int, int]:
    """Order inspector cards by their physical position on the page."""
    bbox = block.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return (1.0, 1.0, 9, index)
    try:
        y, x = float(bbox[1]), float(bbox[0])
    except (TypeError, ValueError):
        return (1.0, 1.0, 9, index)
    type_order = {"text": 0, "table": 1, "chart": 2, "image": 3, "infographic": 4, "stamp": 5}
    return (y, x, type_order.get(str(block.get("content_type") or ""), 9), index)


def build_visual_evidence(data: Any) -> dict[str, Any]:
    """Project final pipeline rows into dashboard visual evidence.

    The returned object contains page image base64 values because it is
    stored server-side and served through a page-image endpoint.  Callers
    returning the manifest should remove those values with
    :func:`manifest_without_images`.
    """

    pages: dict[int, dict[str, Any]] = {}
    # Option 2 explodes one page into many text/element rows. The raw
    # page-elements detection list is a page-level payload and is copied onto
    # each row; projecting it repeatedly creates thousands of duplicate
    # candidates for the quadratic geometry deduplicator.
    seen_option2_page_nested: set[tuple[str, int, str]] = set()
    # The Option 2 result adapter also keeps the complete OCR block list on
    # every exploded row for API compatibility.  It is a page-level payload,
    # not a row-local list.  Project it once per source page; the canonical
    # row blocks are still projected below, so this only removes transport
    # copies and never removes a real visual target.
    seen_option2_page_ocr: set[tuple[str, int]] = set()
    seen_native_page_elements: set[tuple[str, int]] = set()
    option2_pages: set[int] = set()
    option6_pages: set[int] = set()
    option7_pages: set[int] = set()
    rows = _iter_rows(data)
    for row_index, row in enumerate(rows):
        page_number = _as_page_number(row)
        row_metadata = _as_mapping(row.get("metadata"))
        option2_visual = row_metadata.get("ocr_pipeline") == "pipeline-ppocrv6" or str(
            row.get("ocr_source") or row.get("source") or ""
        ).startswith("option2_")
        option6_visual = row_metadata.get("ocr_pipeline") == "pipeline-option6" or str(
            row.get("ocr_source") or row.get("source") or ""
        ).startswith("option6_")
        if option2_visual:
            option2_pages.add(page_number)
        if option6_visual:
            option6_pages.add(page_number)
        option7_visual = row_metadata.get("ocr_pipeline") == "pipeline-option7" or str(
            row.get("ocr_source") or row.get("source") or ""
        ).startswith("option7_")
        if option7_visual:
            option7_pages.add(page_number)
        page = pages.setdefault(
            page_number,
            {
                "page_number": page_number,
                "source_id": row.get("source_id") or row.get("path"),
                "image_b64": None,
                "mime": None,
                "encoding": None,
                "width": None,
                "height": None,
                "blocks": [],
            },
        )
        if page["image_b64"] is None:
            page_image = _page_image_from(row)
            if page_image is not None:
                page.update(page_image)

        def add_block(value: Mapping[str, Any], *, fallback_type: str = "text", origin: str = "row") -> None:
            bbox = _bbox_from(value)
            text = _text_from(value)
            content_type = _content_type_from(value, fallback=fallback_type)
            # Pipeline 6 deliberately keeps visual crops out of OCR text. Its
            # short classifier caption is still useful in the visual trace,
            # so expose it as the block label without treating it as native
            # text that should be deduplicated.
            if not text and content_type in _VISUAL_TYPES:
                caption = value.get("caption")
                if isinstance(caption, str) and caption.strip():
                    text = caption.strip()[:_MAX_BLOCK_TEXT]
            # A full-page image entry is the page background, not a useful
            # hover target. The page image is already stored separately.
            if (
                content_type == "image"
                and _bbox_area(bbox) >= 0.90
                and not text
                and value.get("image_type") != "detected_region"
            ):
                return
            if bbox is None and not text:
                return
            confidence = value.get("confidence")
            if confidence is None:
                confidence = value.get("score")
            try:
                confidence = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence = None
            if confidence is not None and not math.isfinite(confidence):
                confidence = None
            reading_order = value.get("_reading_order", value.get("reading_order"))
            try:
                reading_order = int(reading_order) if math.isfinite(float(reading_order)) else None
            except (TypeError, ValueError, OverflowError):
                reading_order = None
            image_type = value.get("image_type")
            image_type = str(image_type) if isinstance(image_type, str) and image_type else None
            label_name = value.get("label_name")
            label_name = str(label_name) if isinstance(label_name, str) and label_name else None
            # Page Elements may repeat an OCR title with the same geometry.
            # In Option 2 the OCR text row is the canonical hover target; a
            # second title rectangle only creates a visual stack in the UI.
            # Keep a title when it has no matching OCR text, but suppress the
            # exact/near-identical overlay in this branch only.
            if option2_visual and content_type == "title" and bbox is not None:
                if any(
                    str(existing.get("content_type") or "") == "text"
                    and isinstance(existing.get("bbox"), list)
                    and _bbox_geometry_is_duplicate(existing["bbox"], bbox)
                    for existing in page["blocks"]
                ):
                    return
            provenance = value.get("provenance")
            provenance = dict(provenance) if isinstance(provenance, Mapping) else None
            _append_deduplicated_block(
                page["blocks"],
                {
                    "id": f"p{page_number}-b{len(page['blocks'])}",
                    "row_index": row_index,
                    "bbox": bbox,
                    "model_bbox": _normalise_bbox(value.get("model_bbox_xyxy_norm")) or bbox,
                    "processed_bbox": _normalise_bbox(value.get("processed_bbox_xyxy_norm")),
                    "crop_bbox": _normalise_bbox(value.get("crop_bbox_xyxy_norm")) or bbox,
                    "text": text,
                    "content_type": content_type,
                    "reader_backend": _reader_from(value),
                    "confidence": confidence,
                    "origin": origin,
                    "image_type": image_type,
                    "label_name": label_name,
                    "reading_order": reading_order,
                    # Keep line-level OCR provenance in the debug sidecar.
                    # These fields already come from the detector/recognizer
                    # adapter; this projection only makes them visible to the
                    # dashboard and does not alter ingest output.
                    "ocr_source": value.get("ocr_source") or value.get("source"),
                    "ocr_mode": value.get("ocr_mode"),
                    "line_detector_score": value.get("line_detector_score"),
                    "page_elements_score": value.get("page_elements_score"),
                    "region_label": value.get("region_label"),
                    "model": value.get("model") or value.get("ocr_model"),
                    "provenance": _json_safe(provenance) if provenance else None,
                    "selected_backend": (
                        provenance.get("selected_backend") if provenance else value.get("selected_backend")
                    ),
                    "route": provenance.get("route") if provenance else value.get("route"),
                    "fallback_reason": (
                        provenance.get("fallback_reason") if provenance else value.get("fallback_reason")
                    ),
                    "bbox_source": (
                        provenance.get("bbox_source") if provenance else value.get("bbox_source")
                    ),
                    "nemotron_original_text": (
                        provenance.get("nemotron_original_text") if provenance else None
                    ),
                    "vietnamese_candidate_text": (
                        provenance.get("vietnamese_candidate_text") if provenance else None
                    ),
                },
            )

        # Native PDFium text is already the authoritative OCR result for a
        # text-bearing PDF page.  Pipeline 7 also keeps a private projection
        # of those native characters into Page Elements text/title boxes so
        # the inspector can show the detected region with its actual text.
        # The projection is page-scoped and may be copied onto every exploded
        # row, so consume it once per source page.
        native_projection = row.get("_native_page_element_blocks")
        projection_key = (str(page.get("source_id") or row.get("path") or ""), page_number)
        if (
            option7_visual
            and projection_key not in seen_native_page_elements
            and isinstance(native_projection, (list, tuple))
        ):
            seen_native_page_elements.add(projection_key)
            for item in native_projection:
                if isinstance(item, Mapping):
                    add_block(item, fallback_type="text", origin="native_page_element")

        # A page-level scan row can contain aggregate text plus the original
        # line OCR records. Prefer those records for the visual sidecar so
        # every OCR line gets its own bbox. If an old/native row has no line
        # geometry, retain the canonical-row fallback.
        raw_ocr_lines = row.get("_ocr_text_blocks") or row.get("ocr_text_blocks")
        ocr_lines = []
        if isinstance(raw_ocr_lines, (list, tuple)):
            ocr_lines = [
                item
                for item in raw_ocr_lines
                if isinstance(item, Mapping) and _bbox_from(item) is not None and _text_from(item)
            ]
        if ocr_lines and option2_visual:
            page_source = str(page.get("source_id") or "")
            ocr_page_key = (page_source, page_number)
            if ocr_page_key in seen_option2_page_ocr:
                ocr_lines = []
            else:
                seen_option2_page_ocr.add(ocr_page_key)
        if ocr_lines:
            for line in ocr_lines:
                line_value = dict(line)
                line_value.setdefault("content_type", "text")
                add_block(line_value, fallback_type="text", origin="ocr_line")
        else:
            add_block(row, fallback_type="text", origin="canonical_row")

        # Keep structured output boxes as well. They may be image-only and
        # therefore have no canonical text row of their own.
        for key in _NESTED_CONTENT_KEYS:
            nested = row.get(key)
            if isinstance(nested, Mapping):
                # Page Elements stores its result as one page-level payload
                # (``{"detections": [...]}``), not as one detector item.
                # Flatten that envelope so the dashboard receives the actual
                # title/text/table/etc. boxes.
                detections = nested.get("detections")
                nested = (
                    detections
                    if key in {"page_elements", "page_elements_v3"}
                    and isinstance(detections, (list, tuple))
                    else [nested]
                )
            elif isinstance(nested, (list, tuple)) and key in {"page_elements", "page_elements_v3"}:
                flattened: list[Any] = []
                for item in nested:
                    if isinstance(item, Mapping) and isinstance(item.get("detections"), (list, tuple)):
                        flattened.extend(item["detections"])
                    else:
                        flattened.append(item)
                nested = flattened
            if not isinstance(nested, (list, tuple)):
                continue
            if option2_visual and key in {"page_elements", "page_elements_v3"}:
                page_source = str(page.get("source_id") or "")
                nested_key = (page_source, page_number, key)
                if nested_key in seen_option2_page_nested:
                    continue
                seen_option2_page_nested.add(nested_key)
            for item in nested:
                if not isinstance(item, Mapping):
                    continue
                fallback = key.rstrip("s") if key.rstrip("s") in _VISUAL_TYPES else "text"
                add_block(item, fallback_type=fallback, origin=key)

    output_pages: list[dict[str, Any]] = []
    for page_number in sorted(pages):
        page = pages[page_number]
        page["blocks"] = deduplicate_visual_blocks(
            page.get("blocks"),
            page_number,
            suppress_page_sized_visual_noise=page_number in option6_pages or page_number in option7_pages,
        )
        if page_number in option2_pages:
            page["blocks"] = _suppress_option2_title_overlays(page["blocks"])
        page["image_available"] = bool(page.get("image_b64"))
        page["block_count"] = len(page["blocks"])
        output_pages.append(page)
    return {
        "schema_version": "visual-evidence-v2",
        "pages": output_pages,
        "page_count": len(output_pages),
        "block_count": sum(page["block_count"] for page in output_pages),
    }


def deduplicate_visual_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return a lightweight copy with one visual hover target per box.

    This also sanitizes sidecars created before geometry deduplication was
    added, so users do not need to re-ingest an already completed document.
    Page rasters and all non-block metadata are preserved.
    """
    result = dict(evidence) if isinstance(evidence, Mapping) else {}
    pages: list[dict[str, Any]] = []
    for source_page in evidence.get("pages") or [] if isinstance(evidence, Mapping) else []:
        if not isinstance(source_page, Mapping):
            continue
        page = dict(source_page)
        page_number = page.get("page_number")
        try:
            page_number = int(page_number)
        except (TypeError, ValueError):
            page_number = None
        page["blocks"] = deduplicate_visual_blocks(page.get("blocks"), page_number)
        page["block_count"] = len(page["blocks"])
        pages.append(page)
    result["pages"] = pages
    result["page_count"] = len(pages)
    result["block_count"] = sum(page["block_count"] for page in pages)
    return result


def manifest_without_images(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe manifest without embedding page rasters."""
    evidence = deduplicate_visual_evidence(evidence)
    pages: list[dict[str, Any]] = []
    for source_page in evidence.get("pages") or []:
        if not isinstance(source_page, Mapping):
            continue
        page = _json_safe({key: value for key, value in source_page.items() if key != "image_b64"})
        page["image_available"] = bool(source_page.get("image_b64"))
        pages.append(page)
    return {
        "schema_version": evidence.get("schema_version", "visual-evidence-v2"),
        "pages": pages,
        "page_count": len(pages),
        "block_count": sum(int(page.get("block_count") or len(page.get("blocks") or [])) for page in pages),
    }


def page_image_payload(evidence: Mapping[str, Any], page_number: int) -> tuple[str, str] | None:
    """Return ``(mime, base64)`` for one page, if retained."""
    for page in evidence.get("pages") or []:
        if not isinstance(page, Mapping):
            continue
        try:
            matches = int(page.get("page_number")) == int(page_number)
        except (TypeError, ValueError):
            matches = False
        if matches and isinstance(page.get("image_b64"), str) and page["image_b64"]:
            return str(page.get("mime") or "image/png"), page["image_b64"]
    return None
