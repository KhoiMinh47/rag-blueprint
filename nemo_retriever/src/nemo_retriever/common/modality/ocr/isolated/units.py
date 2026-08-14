# SPDX-License-Identifier: Apache-2.0

"""Build one deterministic OCR-unit set from Page Elements and table geometry."""

from __future__ import annotations

from collections.abc import Mapping
from statistics import median
from typing import Any

from nemo_retriever.common.modality.ocr.isolated.contracts import OCRPage, OCRUnit
from nemo_retriever.common.modality.ocr.isolated.geometry import (
    adaptive_local_text_height,
    bbox_iou,
    clamp_bbox,
    crop_image_b64,
    containment,
    map_local_bbox,
)

TEXT_LABELS = frozenset({"text", "title", "header_footer"})
VISUAL_LABELS = frozenset({"image", "chart", "infographic", "stamp"})


def page_element_detections(
    page_elements: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(page_elements, Mapping):
        return []
    detections = page_elements.get("detections") or []
    return [dict(item) for item in detections if isinstance(item, Mapping)]


def _unit_order(detection: Mapping[str, Any], fallback: int) -> int:
    for key in ("reading_order", "readingOrder", "order", "index"):
        value = detection.get(key)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return fallback


def _has_explicit_order(detection: Mapping[str, Any]) -> bool:
    for key in ("reading_order", "readingOrder", "order", "index"):
        value = detection.get(key)
        if value is None:
            continue
        try:
            int(value)
        except (TypeError, ValueError):
            continue
        return True
    return False


def _infer_text_reading_order(units: list[OCRUnit]) -> None:
    """Infer column-major order only when Page Elements gave no order.

    Page Elements v3 is authoritative when it supplies ``reading_order``.  A
    small geometry-only fallback handles detectors that return a flat list in
    scan order: x-aligned blocks are clustered into columns, then each column
    is read top-to-bottom.  The tolerance scales from the page's own block
    geometry rather than a fixed pixel line height.
    """
    text_units = [
        unit
        for unit in units
        if unit.kind in {"text_block", "title"}
        and not unit.metadata.get("explicit_reading_order")
    ]
    if len(text_units) <= 1 or any(
        unit.metadata.get("explicit_reading_order")
        for unit in units
        if unit.kind in {"text_block", "title"}
    ):
        return

    heights = [unit.bbox_xyxy_norm[3] - unit.bbox_xyxy_norm[1] for unit in text_units]
    widths = [unit.bbox_xyxy_norm[2] - unit.bbox_xyxy_norm[0] for unit in text_units]
    x_tolerance = max(float(median(heights)) * 2.0, float(median(widths)) * 0.15)
    columns: list[list[OCRUnit]] = []
    anchors: list[float] = []
    for unit in sorted(text_units, key=lambda item: item.bbox_xyxy_norm[0]):
        x0 = unit.bbox_xyxy_norm[0]
        matching = [
            index
            for index, anchor in enumerate(anchors)
            if abs(x0 - anchor) <= x_tolerance
        ]
        if not matching:
            anchors.append(x0)
            columns.append([unit])
            continue
        column_index = min(matching, key=lambda index: abs(x0 - anchors[index]))
        columns[column_index].append(unit)
        anchors[column_index] = sum(
            member.bbox_xyxy_norm[0] for member in columns[column_index]
        ) / len(columns[column_index])

    reading_order = 0
    for column in sorted(
        columns, key=lambda value: min(item.bbox_xyxy_norm[0] for item in value)
    ):
        for unit in sorted(
            column,
            key=lambda item: (item.bbox_xyxy_norm[1], item.bbox_xyxy_norm[0]),
        ):
            unit.reading_order = reading_order
            unit.metadata["reading_order_inferred"] = True
            reading_order += 1


def _table_regions(
    page: OCRPage,
) -> list[
    tuple[str, tuple[float, float, float, float], list[dict[str, Any]], tuple[int, int]]
]:
    payload = page.table_structure_v1
    if not isinstance(payload, Mapping):
        return []
    regions = payload.get("regions") or []
    result: list[
        tuple[
            str,
            tuple[float, float, float, float],
            list[dict[str, Any]],
            tuple[int, int],
        ]
    ] = []
    for region_index, region in enumerate(regions):
        if not isinstance(region, Mapping):
            continue
        bbox = clamp_bbox(region.get("bbox_xyxy_norm"))
        detections = region.get("detections") or []
        if bbox is None or not isinstance(detections, list):
            continue
        shape = region.get("orig_shape_hw") or (1, 1)
        try:
            shape_hw = (max(1, int(shape[0])), max(1, int(shape[1])))
        except (TypeError, ValueError, IndexError):
            shape_hw = (1, 1)
        table_id = str(
            region.get("table_id") or region.get("id") or f"table-{region_index}"
        )
        result.append(
            (
                table_id,
                bbox,
                [dict(item) for item in detections if isinstance(item, Mapping)],
                shape_hw,
            )
        )
    return result


def build_ocr_units(
    page: OCRPage,
    *,
    include_table_cells: bool = True,
    include_page_element_table_regions: bool = False,
    include_visual_regions: bool = False,
    pad_table_cells: bool = True,
    cropper: Any | None = None,
) -> list[OCRUnit]:
    """Create one deterministic crop set from the available layout signals.

    ``include_page_element_table_regions`` is intentionally opt-in.  It is a
    compatibility fallback for semantic OCR callers that have only a
    Page-Elements table bbox; callers with Table Structure data use the
    structure-provided regions/cells instead.
    """
    shape = None
    from nemo_retriever.common.modality.ocr.isolated.geometry import image_size

    shape = image_size(page.image_b64)
    if shape is None:
        return []
    detections = page_element_detections(page.page_elements_v3)
    page_table_boxes = [
        bbox
        for item in detections
        if str(item.get("label_name") or "").strip().lower() == "table"
        and (bbox := clamp_bbox(item.get("bbox_xyxy_norm"))) is not None
    ]
    candidate_boxes = [
        clamp_bbox(item.get("bbox_xyxy_norm"))
        for item in detections
        if str(item.get("label_name") or "").strip().lower() in TEXT_LABELS
    ]
    usable_boxes = [box for box in candidate_boxes if box is not None]
    units: list[OCRUnit] = []
    for index, detection in enumerate(detections):
        label = str(detection.get("label_name") or "").strip().lower()
        if label not in TEXT_LABELS and not (
            include_visual_regions and label in VISUAL_LABELS
        ):
            continue
        bbox = clamp_bbox(detection.get("bbox_xyxy_norm"))
        if bbox is None:
            continue
        if include_page_element_table_regions and any(
            containment(bbox, table_bbox) >= 0.55
            or bbox_iou(bbox, table_bbox) >= 0.35
            for table_bbox in page_table_boxes
        ):
            # The Page Elements table region is authoritative for this
            # detector-free branch.  Do not OCR the same table once as a
            # generic text block and once as a table region.
            continue
        local_height = adaptive_local_text_height(bbox, usable_boxes or [bbox])
        crop = (
            cropper.crop(bbox, local_text_height=local_height, add_padding=True)
            if cropper is not None
            else crop_image_b64(
                page.image_b64, bbox, local_text_height=local_height, add_padding=True
            )
        )
        if crop is None:
            continue
        kind = (
            "title"
            if label == "title"
            else ("visual" if label in VISUAL_LABELS else "text_block")
        )
        unit_id = f"page-{page.page_number or 0}-block-{index}"
        units.append(
            OCRUnit(
                unit_id=unit_id,
                kind=kind,
                source="page_elements_v3",
                bbox_xyxy_norm=bbox,
                crop_bbox_xyxy_norm=crop.bbox_xyxy_norm,
                crop_b64=crop.image_b64,
                crop_shape_hw=crop.shape_hw,
                reading_order=_unit_order(detection, index),
                detector_score=_score(detection.get("score")),
                label=label,
                metadata={
                    "page_number": page.page_number,
                    "padding_applied": crop.bbox_xyxy_norm != bbox,
                    "local_text_height_norm": local_height,
                    "priority": "title" if label == "title" else "body",
                    "explicit_reading_order": _has_explicit_order(detection),
                },
            )
        )

    _infer_text_reading_order(units)

    if include_page_element_table_regions:
        table_offset = len(units)
        for table_index, detection in enumerate(detections):
            if str(detection.get("label_name") or "").strip().lower() != "table":
                continue
            bbox = clamp_bbox(detection.get("bbox_xyxy_norm"))
            if bbox is None:
                continue
            local_height = adaptive_local_text_height(
                bbox, usable_boxes or [bbox]
            )
            crop = (
                cropper.crop(
                    bbox,
                    local_text_height=local_height,
                    add_padding=True,
                )
                if cropper is not None
                else crop_image_b64(
                    page.image_b64,
                    bbox,
                    local_text_height=local_height,
                    add_padding=True,
                )
            )
            if crop is None:
                continue
            table_id = f"page-elements-table-{table_index}"
            units.append(
                OCRUnit(
                    unit_id=f"page-{page.page_number or 0}-{table_id}",
                    kind="table_region",
                    source="page_elements_v3",
                    bbox_xyxy_norm=bbox,
                    crop_bbox_xyxy_norm=crop.bbox_xyxy_norm,
                    crop_b64=crop.image_b64,
                    crop_shape_hw=crop.shape_hw,
                    reading_order=table_offset + _unit_order(detection, table_index),
                    detector_score=_score(detection.get("score")),
                    label="table",
                    table_id=table_id,
                    metadata={
                        "page_number": page.page_number,
                        "table_id": table_id,
                        "table_structure_disabled": True,
                        "local_text_height_norm": local_height,
                        "padding_applied": crop.bbox_xyxy_norm != bbox,
                    },
                )
            )

    if include_table_cells:
        cell_offset = len(units)
        for table_index, (
            table_id,
            table_bbox,
            cell_detections,
            table_shape,
        ) in enumerate(_table_regions(page)):
            cell_boxes = [
                map_local_bbox(
                    item.get("bbox_xyxy_norm") or item.get("bbox"),
                    table_bbox,
                    table_shape,
                )
                for item in cell_detections
                if str(item.get("label_name") or "").lower() == "cell"
            ]
            usable_cell_boxes = [box for box in cell_boxes if box is not None]
            for cell_index, (detection, cell_bbox) in enumerate(
                (item, box)
                for item, box in zip(
                    [
                        item
                        for item in cell_detections
                        if str(item.get("label_name") or "").lower() == "cell"
                    ],
                    usable_cell_boxes,
                )
            ):
                local_height = adaptive_local_text_height(
                    cell_bbox, usable_cell_boxes or [cell_bbox]
                )
                crop = (
                    cropper.crop(
                        cell_bbox,
                        local_text_height=local_height,
                        add_padding=bool(pad_table_cells),
                    )
                    if cropper is not None
                    else crop_image_b64(
                        page.image_b64,
                        cell_bbox,
                        local_text_height=local_height,
                        add_padding=bool(pad_table_cells),
                    )
                )
                if crop is None:
                    continue
                cell_id = str(
                    detection.get("cell_id")
                    or detection.get("id")
                    or f"cell-{cell_index}"
                )
                units.append(
                    OCRUnit(
                        unit_id=f"page-{page.page_number or 0}-{table_id}-{cell_id}",
                        kind="table_cell",
                        source="table_structure_v1",
                        bbox_xyxy_norm=cell_bbox,
                        crop_bbox_xyxy_norm=crop.bbox_xyxy_norm,
                        crop_b64=crop.image_b64,
                        crop_shape_hw=crop.shape_hw,
                        reading_order=cell_offset
                        + table_index * 10000
                        + _unit_order(detection, cell_index),
                        detector_score=_score(detection.get("score")),
                        label="cell",
                        table_id=table_id,
                        cell_id=cell_id,
                        metadata={
                            "page_number": page.page_number,
                            "table_id": table_id,
                            "cell_id": cell_id,
                            "padding_applied": crop.bbox_xyxy_norm != cell_bbox,
                            "local_text_height_norm": local_height,
                        },
                    )
                )
    return sorted(
        units,
        key=lambda unit: (
            unit.reading_order,
            unit.bbox_xyxy_norm[1],
            unit.bbox_xyxy_norm[0],
        ),
    )


def visual_exclusion_boxes(page: OCRPage) -> list[tuple[float, float, float, float]]:
    """Return Page Elements regions that recall OCR must not read as body text."""
    result: list[tuple[float, float, float, float]] = []
    for item in page_element_detections(page.page_elements_v3):
        if str(item.get("label_name") or "").lower() not in VISUAL_LABELS:
            continue
        bbox = clamp_bbox(item.get("bbox_xyxy_norm"))
        if bbox is not None:
            result.append(bbox)
    result.extend(region[1] for region in _table_regions(page))
    return result


def table_payload(page: OCRPage) -> list[dict[str, Any]]:
    """Expose table geometry for canonical output without changing it."""
    result = []
    for table_id, bbox, _detections, _shape in _table_regions(page):
        result.append({"table_id": table_id, "bbox_xyxy_norm": list(bbox), "cells": []})
    return result


def _score(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, number))
