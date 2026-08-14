# SPDX-License-Identifier: Apache-2.0

"""Line splitting for semantic OCR boxes.

Page Elements produces semantic text boxes, while VietOCR is a line-oriented
recognizer.  This module bridges that shape mismatch with an optional batched
PP-OCRv6 detector and a CPU projection fallback.  It deliberately returns the
original unit unchanged when both methods are ambiguous; callers can then use
their normal fallback path instead of risking a bad crop.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from nemo_retriever.common.modality.ocr.isolated.adapters import detector_boxes
from nemo_retriever.common.modality.ocr.isolated.contracts import OCRPage, OCRUnit
from nemo_retriever.common.modality.ocr.isolated.geometry import (
    PageImageCropper,
    bbox_iou,
    clamp_bbox,
    map_local_bbox,
)


def split_multiline_units(
    page: OCRPage,
    units: Sequence[OCRUnit],
    *,
    cropper: PageImageCropper | None,
    min_height_ratio: float = 1.65,
    max_lines: int = 64,
    line_detector: Any | None = None,
    detector_responses: Mapping[int, Any] | None = None,
    stats: dict[str, Any] | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> list[OCRUnit]:
    """Split clearly multi-line text blocks into line-sized OCR units.

    A simple grayscale row projection finds ink bands vertically, and the line
    crops inherit the parent table/cell identity and reading order.  When the
    optional detector is supplied, all candidate crops in this call are sent
    as one detector batch.  ``detector_responses`` lets the document
    coordinator pre-batch candidates from multiple pages, avoiding one HTTP
    round trip per page.  Table cells are intentionally left intact for now: a
    cell is a single canonical output value and splitting it would require a
    cell-level merge step.
    """

    if cropper is None or not units:
        return list(units)

    metrics = stats if stats is not None else {}
    metrics.setdefault("line_detector_seconds", 0.0)
    metrics.setdefault("line_detector_input_count", 0)
    metrics.setdefault("line_detector_line_count", 0)
    detector_candidates = multiline_detector_candidates(
        units,
        min_height_ratio=min_height_ratio,
    )
    detector_lines_by_unit: dict[
        int, list[tuple[tuple[float, float, float, float], float | None]]
    ] = {}
    if detector_candidates and detector_responses is not None:
        # The caller already paid for one document-level detector batch.  Do
        # not increment transport timing/count metrics a second time here.
        responses = [
            detector_responses.get(id(unit)) for unit in detector_candidates
        ]
        for unit, response in zip(detector_candidates, responses):
            mapped = _detector_line_boxes(unit, response)
            if len(mapped) > max(2, int(max_lines)):
                mapped = []
            if mapped:
                detector_lines_by_unit[id(unit)] = mapped
    elif detector_candidates and line_detector is not None:
        detector_started = time.perf_counter()
        metrics["line_detector_input_count"] += len(detector_candidates)
        try:
            responses = list(
                line_detector.detect([unit.crop_b64 for unit in detector_candidates])
            )
        except Exception as exc:  # noqa: BLE001 - projection remains a local fallback
            responses = []
            if errors is not None:
                errors.append(
                    {
                        "stage": "option5.line_detector",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        metrics["line_detector_seconds"] += time.perf_counter() - detector_started
        for index, unit in enumerate(detector_candidates):
            response = responses[index] if index < len(responses) else None
            mapped = _detector_line_boxes(unit, response)
            if len(mapped) > max(2, int(max_lines)):
                # A detector hallucinating many tiny boxes should not turn a
                # single semantic region into an OCR request storm.  The
                # projection fallback below can still recover a bounded set
                # of real text bands.
                mapped = []
            if mapped:
                detector_lines_by_unit[id(unit)] = mapped

    result: list[OCRUnit] = []
    for unit in units:
        if unit.kind == "table_cell":
            result.append(unit)
            continue
        detector_lines = detector_lines_by_unit.get(id(unit))
        projected: list[tuple[float, float, float, float]] | None = None
        if detector_lines and len(detector_lines) >= 2:
            line_specs = [
                (bbox, score, "ppocrv6_line_detector", True)
                for bbox, score in detector_lines
            ]
        else:
            projected = _line_boxes(
                unit,
                cropper=cropper,
                min_height_ratio=min_height_ratio,
                max_lines=max_lines,
            )
            if len(projected) >= 2:
                line_specs = [
                    (bbox, None, "horizontal_projection", False)
                    for bbox in projected
                ]
            else:
                # A one-line detector response is useful evidence that this
                # is probably a text region, but it is not enough to replace
                # a tall semantic crop.  Keep it only when the cheap
                # projection cannot establish multiple lines.
                line_specs = [
                    (bbox, score, "ppocrv6_line_detector", True)
                    for bbox, score in (detector_lines or [])
                ]
        if len(line_specs) < 2:
            result.append(unit)
            continue
        # A broad semantic parent often overlaps precise Page Elements boxes
        # that already cover a few of its rows.  Do not emit those rows twice:
        # the sibling unit remains the canonical crop, while the parent line
        # units retain the uncovered body text.  Horizontal overlap is checked
        # as well so a right-column sibling cannot suppress an uncovered
        # left-column line in a two-column form.
        line_specs = _remove_sibling_overlaps(unit, line_specs, units)
        if len(line_specs) < 2:
            result.append(unit)
            continue
        line_count = len(line_specs)
        split_units: list[OCRUnit] = []
        for line_index, (
            line_bbox,
            detector_score,
            split_source,
            detector_padding,
        ) in enumerate(line_specs):
            # ``local_text_height_norm`` belongs to the parent semantic box
            # and may be the full paragraph height when the box has no
            # neighbors.  Reusing it here makes ``expand_bbox_adaptive`` add
            # paragraph-sized margins around every line, which changes the
            # aspect ratio and makes a line recognizer repeat or hallucinate
            # text.  Projection boxes already include a small margin; detector
            # boxes receive only a glyph-height-sized border.
            crop = cropper.crop(
                line_bbox,
                local_text_height=(line_bbox[3] - line_bbox[1]),
                add_padding=detector_padding,
            )
            if crop is None:
                continue
            metadata = dict(unit.metadata)
            metadata.update(
                {
                    "parent_unit_id": unit.unit_id,
                    "line_index": line_index,
                    "line_count": line_count,
                    "multiline_split": True,
                    "line_split_method": split_source,
                    "line_detector": (
                        "PP-OCRv6_medium_det" if detector_score is not None else None
                    ),
                    "line_detector_score": detector_score,
                }
            )
            split_units.append(
                OCRUnit(
                    unit_id=f"{unit.unit_id}-line-{line_index}",
                    kind=unit.kind,
                    source=split_source,
                    bbox_xyxy_norm=line_bbox,
                    crop_bbox_xyxy_norm=crop.bbox_xyxy_norm,
                    crop_b64=crop.image_b64,
                    crop_shape_hw=crop.shape_hw,
                    reading_order=unit.reading_order * 100 + line_index,
                    detector_score=detector_score or unit.detector_score,
                    label=unit.label,
                    table_id=unit.table_id,
                    cell_id=unit.cell_id,
                    metadata=metadata,
                )
            )
        if len(split_units) == line_count:
            result.extend(split_units)
        else:
            # Preserve recall if a crop failed unexpectedly.
            result.append(unit)
        if line_specs and line_specs[0][2] == "ppocrv6_line_detector":
            metrics["line_detector_line_count"] += len(split_units)
    return result


def multiline_detector_candidates(
    units: Sequence[OCRUnit],
    *,
    min_height_ratio: float = 1.65,
) -> list[OCRUnit]:
    """Return the non-table units worth sending to PP-OCRv6.

    Keeping this predicate in the low-level splitter lets Option 5 collect
    candidates across an entire document and issue one bounded detector batch.
    One-line boxes never pay for the remote model.
    """

    return [
        unit
        for unit in units
        if unit.kind != "table_cell"
        and _likely_multiline(unit, min_height_ratio=min_height_ratio)
    ]


def _likely_multiline(unit: OCRUnit, *, min_height_ratio: float) -> bool:
    """Identify boxes worth sending to the optional line detector."""

    if unit.kind not in {"text_block", "title"}:
        return False
    bbox = clamp_bbox(unit.bbox_xyxy_norm)
    if bbox is None or not unit.crop_b64:
        return False
    height = bbox[3] - bbox[1]
    if height <= 0.0:
        return False
    try:
        local_height = float(unit.metadata.get("local_text_height_norm") or 0.0)
    except (TypeError, ValueError):
        local_height = 0.0
    if local_height > 0.0:
        # 0.75 * 1.65 = 1.2375: catch two-line boxes even when Page
        # Elements' local-height estimate is taken from a nearby title.
        threshold = max(1.20, float(min_height_ratio) * 0.75)
        if height / local_height >= threshold:
            return True
        # ``adaptive_local_text_height`` legitimately returns the whole box
        # height when a paragraph is isolated on a page.  Do not let that
        # suppress PP-OCRv6 for an obviously tall, compact region; normal
        # one-line body boxes are much flatter and remain on the CPU path.
        if local_height >= height * 0.90:
            width = bbox[2] - bbox[0]
            return height >= 0.08 and width / max(height, 1e-9) <= 8.0
        return False
    return height >= 0.05


def _detector_line_boxes(
    unit: OCRUnit,
    response: Any,
) -> list[tuple[tuple[float, float, float, float], float | None]]:
    """Map PP-OCRv6 crop-local boxes back to the semantic parent span."""

    parent = clamp_bbox(unit.bbox_xyxy_norm)
    if parent is None:
        return []
    accepted: list[tuple[tuple[float, float, float, float], float | None]] = []
    for detection in detector_boxes(response):
        local = clamp_bbox(detection.bbox)
        if local is None:
            continue
        mapped = map_local_bbox(
            local,
            unit.crop_bbox_xyxy_norm,
            unit.crop_shape_hw,
        )
        if mapped is None:
            continue
        # Keep PP-OCRv6's horizontal span when it is useful (especially for a
        # normal single-column paragraph), while still preserving separate
        # left/right boxes in forms.  Forcing every detector box to the full
        # parent width makes VietOCR see two columns at once and is a common
        # source of low-confidence/repeated text.
        horizontal_padding = max(
            0.001,
            (mapped[3] - mapped[1]) * 0.18,
        )
        mapped = (
            max(parent[0], mapped[0] - horizontal_padding),
            max(parent[1], min(parent[3], mapped[1])),
            min(parent[2], mapped[2] + horizontal_padding),
            max(parent[1], min(parent[3], mapped[3])),
        )
        if mapped[3] <= mapped[1]:
            continue
        if any(bbox_iou(mapped, previous) >= 0.88 for previous, _ in accepted):
            continue
        accepted.append((mapped, detection.score))
    accepted.sort(key=lambda item: (item[0][1], item[0][0]))
    return _coalesce_same_row_boxes(accepted)


def _coalesce_same_row_boxes(
    boxes: list[tuple[tuple[float, float, float, float], float | None]],
) -> list[tuple[tuple[float, float, float, float], float | None]]:
    """Collapse left/right detector boxes that represent one visual row.

    PP-OCR can return one box per column for a form.  After mapping those
    boxes back to a semantic parent span they have the same horizontal extent
    and become duplicate OCR requests.  Adjacent visual rows have a gap, so a
    vertical IoU threshold is sufficient and avoids merging real rows.
    """

    if len(boxes) < 2:
        return boxes
    merged: list[tuple[tuple[float, float, float, float], float | None]] = []
    for bbox, score in boxes:
        if not merged:
            merged.append((bbox, score))
            continue
        previous_bbox, previous_score = merged[-1]
        if (
            _vertical_iou(previous_bbox, bbox) < 0.35
            or _horizontal_iou(previous_bbox, bbox) < 0.35
        ):
            merged.append((bbox, score))
            continue
        merged[-1] = (
            (
                min(previous_bbox[0], bbox[0]),
                min(previous_bbox[1], bbox[1]),
                max(previous_bbox[2], bbox[2]),
                max(previous_bbox[3], bbox[3]),
            ),
            _higher_score(previous_score, score),
        )
    return merged


def _vertical_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    top = max(left[1], right[1])
    bottom = min(left[3], right[3])
    overlap = max(0.0, bottom - top)
    if overlap <= 0.0:
        return 0.0
    left_height = max(1e-9, left[3] - left[1])
    right_height = max(1e-9, right[3] - right[1])
    return overlap / max(left_height + right_height - overlap, 1e-9)


def _horizontal_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_edge = max(left[0], right[0])
    right_edge = min(left[2], right[2])
    overlap = max(0.0, right_edge - left_edge)
    if overlap <= 0.0:
        return 0.0
    left_width = max(1e-9, left[2] - left[0])
    right_width = max(1e-9, right[2] - right[0])
    return overlap / max(left_width + right_width - overlap, 1e-9)


def _higher_score(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(float(left), float(right))


def _remove_sibling_overlaps(
    parent: OCRUnit,
    line_specs: list[tuple[tuple[float, float, float, float], float | None, str, bool]],
    units: Sequence[OCRUnit],
) -> list[tuple[tuple[float, float, float, float], float | None, str, bool]]:
    parent_bbox = clamp_bbox(parent.bbox_xyxy_norm)
    if parent_bbox is None:
        return line_specs
    parent_area = max(
        1e-9,
        (parent_bbox[2] - parent_bbox[0]) * (parent_bbox[3] - parent_bbox[1]),
    )
    siblings: list[tuple[float, float, float, float]] = []
    for sibling in units:
        if sibling is parent or sibling.unit_id == parent.unit_id:
            continue
        if sibling.kind == "table_cell":
            continue
        sibling_bbox = clamp_bbox(sibling.bbox_xyxy_norm)
        if sibling_bbox is None:
            continue
        sibling_area = (sibling_bbox[2] - sibling_bbox[0]) * (
            sibling_bbox[3] - sibling_bbox[1]
        )
        # Only precise children/siblings can safely claim a parent row.  A
        # second broad recall box is deliberately left to the parent-level
        # suppression pass in Option 5.
        if sibling_area / parent_area > 0.50:
            continue
        siblings.append(sibling_bbox)
    if not siblings:
        return line_specs

    kept: list[tuple[tuple[float, float, float, float], float | None, str, bool]] = []
    for spec in line_specs:
        if any(
            _vertical_coverage(spec[0], sibling_bbox) >= 0.65
            and _horizontal_iou(spec[0], sibling_bbox) >= 0.25
            for sibling_bbox in siblings
        ):
            continue
        kept.append(spec)
    return kept


def _vertical_coverage(
    line: tuple[float, float, float, float],
    sibling: tuple[float, float, float, float],
) -> float:
    """How much of a candidate line is covered by a sibling region."""

    overlap = max(0.0, min(line[3], sibling[3]) - max(line[1], sibling[1]))
    return overlap / max(line[3] - line[1], 1e-9)


def _line_boxes(
    unit: OCRUnit,
    *,
    cropper: PageImageCropper,
    min_height_ratio: float,
    max_lines: int,
) -> list[tuple[float, float, float, float]]:
    bbox = clamp_bbox(unit.bbox_xyxy_norm)
    if bbox is None:
        return []
    local_height = _float_value(unit.metadata.get("local_text_height_norm"))
    box_height = bbox[3] - bbox[1]
    # A lone paragraph may have no neighboring detector box, so its local
    # height estimate can equal the whole paragraph.  In that case let the
    # image projection decide; a genuine one-line crop still yields one band.
    if (
        local_height > 0.0
        and local_height < box_height * 0.90
        and box_height / local_height < float(min_height_ratio)
    ):
        return []

    left = max(0, min(cropper.width - 1, round(bbox[0] * cropper.width)))
    top = max(0, min(cropper.height - 1, round(bbox[1] * cropper.height)))
    right = max(left + 1, min(cropper.width, round(bbox[2] * cropper.width)))
    bottom = max(top + 1, min(cropper.height, round(bbox[3] * cropper.height)))
    if right <= left or bottom <= top:
        return []

    try:
        gray = cropper.image.crop((left, top, right, bottom)).convert("L")
        rows = _ink_rows(gray)
    except Exception:  # noqa: BLE001 - line splitting is best effort
        return []
    expected_px = (
        local_height * cropper.height
        if local_height > 0.0 and local_height < box_height * 0.90
        else (bottom - top) * 0.25
    )
    expected_px = max(2.0, expected_px)
    bands = _runs(rows, expected_px=expected_px)
    if len(bands) < 2 or len(bands) > max(2, int(max_lines)):
        return []

    padding = max(1, round(expected_px * 0.16))
    result: list[tuple[float, float, float, float]] = []
    for y0, y1 in bands:
        line_top = max(top, top + y0 - padding)
        line_bottom = min(bottom, top + y1 + padding)
        if line_bottom <= line_top:
            continue
        result.append(
            (
                left / cropper.width,
                line_top / cropper.height,
                right / cropper.width,
                line_bottom / cropper.height,
            )
        )
    return result


def _ink_rows(image: Any) -> list[bool]:
    """Return rows containing enough dark pixels, with a numpy fast path."""

    try:
        import numpy as np

        pixels = np.asarray(image, dtype=np.uint8)
        if pixels.ndim != 2:
            return []
        # 235 handles white scans and light anti-aliased glyph edges without
        # treating a one-pixel compression artifact as a text line.
        counts = (pixels < 235).sum(axis=1)
        threshold = max(1, int(pixels.shape[1] * 0.004))
        return [bool(value >= threshold) for value in counts.tolist()]
    except Exception:  # noqa: BLE001 - numpy is optional at this boundary
        width, height = image.size
        pixels = image.load()
        threshold = max(1, int(width * 0.004))
        return [
            sum(1 for x in range(width) if pixels[x, y] < 235) >= threshold
            for y in range(height)
        ]


def _runs(rows: Sequence[bool], *, expected_px: float) -> list[tuple[int, int]]:
    if not rows:
        return []
    gap_limit = max(1, round(expected_px * 0.22))
    filled = list(rows)
    gap_start: int | None = None
    for index, active in enumerate(filled + [True]):
        if active:
            if gap_start is not None and index - gap_start <= gap_limit:
                for gap_index in range(gap_start, index):
                    filled[gap_index] = True
            gap_start = None
        elif gap_start is None:
            gap_start = index

    runs: list[tuple[int, int]] = []
    start: int | None = None
    minimum_height = max(2, round(expected_px * 0.18))
    for index, active in enumerate(filled + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= minimum_height:
                runs.append((start, index))
            start = None
    return runs


def _float_value(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if result > 0.0 else 0.0
