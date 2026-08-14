# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-friendly content row transforms used by example pipelines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import math
import re
import copy
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

from nemo_retriever.common.io.image_store import inline_image_b64
from nemo_retriever.common.params import TextChunkParams
from nemo_retriever.operators.extract.ocr.ocr import _crop_b64_image_by_norm_bbox
from nemo_retriever.common.params.models import IMAGE_MODALITIES

_CONTENT_COLUMNS = ("table", "chart", "infographic", "stamp")
_STRUCTURED_LABELS = frozenset(_CONTENT_COLUMNS)


def _list_items(value: Any) -> List[Any]:
    """Return list-like row content without iterating pandas ``NaN`` values.

    DataFrame construction fills a missing list-valued column with ``NaN``
    (a float).  Structured-content transforms treat that as an empty list;
    attempting to iterate it is what caused Pipeline 7's ``float object is
    not iterable`` failure on pages with different visual-content columns.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _normalised_bbox(value: Any) -> Optional[List[float]]:
    """Return a valid top-left-origin normalized xyxy bbox."""
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        box = [float(value[i]) for i in range(4)]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in box) or any(v < 0.0 or v > 1.0 for v in box):
        return None
    x0, y0, x1, y1 = box
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _union_bbox(boxes: Sequence[Sequence[float]]) -> Optional[List[float]]:
    valid = [_normalised_bbox(box) for box in boxes]
    valid = [box for box in valid if box is not None]
    if not valid:
        return None
    return [
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    ]


def _bbox_contains_point(box: Sequence[float], x: float, y: float, margin: float = 0.002) -> bool:
    return box[0] - margin <= x <= box[2] + margin and box[1] - margin <= y <= box[3] + margin


def _project_native_text_to_page_elements(
    row: Dict[str, Any],
    spans: Sequence[Dict[str, Any]],
    *,
    authoritative_regions: Sequence[Dict[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """Attach PDFium text to Page Elements text/title boxes.

    PDFium remains the authoritative text source for native PDFs, but its
    geometry-bearing blocks are built independently from Page Elements.  That
    used to leave a detector-only ``title``/``text`` box beside a valid native
    text block in the visual sidecar.  Project the characters into the
    detector box so downstream consumers get one useful block with both text
    and semantic bbox provenance.  This is intentionally limited to Pipeline
    7; it does not invoke an OCR model or change the page text.
    """
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if metadata.get("ocr_pipeline") != "pipeline-option7":
        return []
    page_elements = row.get("page_elements_v3")
    detections = page_elements.get("detections", []) if isinstance(page_elements, dict) else []
    if not isinstance(detections, list):
        return []

    lines: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    def flush() -> None:
        nonlocal current
        if current:
            lines.append(current)
            current = []

    for span in spans:
        if not isinstance(span, dict):
            continue
        char = str(span.get("char") or "")
        if char in {"\r", "\n"}:
            flush()
            continue
        bbox = _normalised_bbox(span.get("bbox_xyxy_norm"))
        if bbox is not None:
            current.append({"char": char, "bbox": bbox})
    flush()

    projected: List[Dict[str, Any]] = []
    seen: set[tuple[str, tuple[float, ...]]] = set()
    for fallback_order, detection in enumerate(detections):
        if not isinstance(detection, dict):
            continue
        label = str(detection.get("label_name") or "").strip().lower()
        if label not in {"text", "title", "header_footer"}:
            continue
        detector_bbox = _normalised_bbox(detection.get("bbox_xyxy_norm"))
        if detector_bbox is None:
            continue

        # If a VLM/table output is authoritative for an overlapping region,
        # leave that region to its own block instead of duplicating its native
        # characters into a Page Elements text box.
        blocked = False
        for region in authoritative_regions:
            region_bbox = region.get("bbox") if isinstance(region, dict) else None
            region_type = str(region.get("type") or "") if isinstance(region, dict) else ""
            if region_type not in {"table", "image", "chart", "infographic", "stamp"}:
                continue
            if _bbox_intersection_over_line(detector_bbox, region_bbox or []) >= 0.55:
                blocked = True
                break
        if blocked:
            continue

        selected_lines: List[str] = []
        selected_boxes: List[Sequence[float]] = []
        for line in lines:
            selected: List[str] = []
            line_boxes: List[Sequence[float]] = []
            for item in line:
                bbox = item["bbox"]
                center_x = (bbox[0] + bbox[2]) / 2.0
                center_y = (bbox[1] + bbox[3]) / 2.0
                if _bbox_contains_point(detector_bbox, center_x, center_y, margin=0.002):
                    selected.append(str(item.get("char") or ""))
                    line_boxes.append(bbox)
            text = re.sub(r"[ \t]+", " ", "".join(selected)).strip()
            if text:
                selected_lines.append(text)
                selected_boxes.extend(line_boxes)
        if not selected_lines:
            continue

        text = "\n".join(selected_lines).strip()[:12_000]
        native_bbox = _union_bbox(selected_boxes) or detector_bbox
        order = detection.get("reading_order", detection.get("readingOrder", fallback_order))
        try:
            reading_order = int(order)
        except (TypeError, ValueError):
            reading_order = fallback_order
        key = (label, tuple(round(float(value), 4) for value in detector_bbox))
        if key in seen:
            continue
        seen.add(key)
        projected.append(
            {
                "text": text,
                # Use the detector bbox as the canonical hover geometry. Keep
                # the tighter PDFium geometry separately for provenance.
                "bbox_xyxy_norm": detector_bbox,
                "native_bbox_xyxy_norm": native_bbox,
                "model_bbox_xyxy_norm": detector_bbox,
                "source": "pdfium_native",
                "ocr_source": "pdfium_native",
                "ocr_mode": "native_page_element",
                "reader_backend": "native_pdf",
                "model": "PDFium native text",
                "content_type": "title" if label == "title" else "text",
                "label_name": label,
                "reading_order": reading_order,
                "score": detection.get("score", detection.get("confidence")),
                "page_elements_score": detection.get("score", detection.get("confidence")),
                "provenance": {
                    "selected_backend": "pdfium_native",
                    "bbox_source": "page_elements_v3",
                    "native_bbox_source": "pdfium_character_spans",
                    "native_bbox_xyxy_norm": native_bbox,
                },
            }
        )
    return projected


def _structured_text_candidates(item: Dict[str, Any]) -> List[str]:
    """Return text emitted by a structured region for safe deduplication.

    A table region may expose either one Markdown ``text`` value or individual
    ``cells``.  We deliberately do not use chart/image text here: a detector
    bbox is not proof that native text inside it is duplicated.
    """
    candidates: List[str] = []
    for key in ("text", "caption"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    cells = item.get("cells")
    if isinstance(cells, list):
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            value = cell.get("text")
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    return candidates


def _structured_regions(
    row: Dict[str, Any],
    *,
    allow_page_elements_fallback: bool = True,
) -> List[Dict[str, Any]]:
    """Collect authoritative structured OCR regions.

    Semantic pipelines use Page Elements as layout evidence, but only a crop
    or structured region that was actually processed by the VLM may replace
    native characters.  Detector-only image/chart regions therefore arrive
    through ``images`` rather than the raw detector fallback.
    """
    regions: List[Dict[str, Any]] = []
    for column in _CONTENT_COLUMNS:
        values = row.get(column)
        if not isinstance(values, list):
            continue
        for index, item in enumerate(values):
            if not isinstance(item, dict):
                continue
            bbox = _normalised_bbox(item.get("bbox_xyxy_norm"))
            if bbox is not None:
                regions.append({
                    "type": column,
                    "index": index,
                    "bbox": bbox,
                    "texts": _structured_text_candidates(item),
                })

    images = row.get("images")
    if isinstance(images, list):
        for index, item in enumerate(images):
            if not isinstance(item, dict):
                continue
            label = str(item.get("label_name") or "").strip().lower()
            text = str(item.get("text") or "").strip()
            authoritative = item.get("ocr_authoritative") is True or (
                item.get("ocr_mode") == "visual_crop" and bool(text)
            )
            bbox = _normalised_bbox(item.get("bbox_xyxy_norm"))
            if label not in {"image", "chart", "infographic", "stamp"}:
                continue
            if not authoritative or bbox is None:
                continue
            regions.append({
                "type": label,
                "index": index,
                "bbox": bbox,
                "texts": _structured_text_candidates(item),
            })

    if regions or not allow_page_elements_fallback:
        return regions

    page_elements = row.get("page_elements_v3")
    detections = page_elements.get("detections", []) if isinstance(page_elements, dict) else []
    if not isinstance(detections, list):
        return regions
    for index, detection in enumerate(detections):
        if not isinstance(detection, dict) or detection.get("label_name") not in _STRUCTURED_LABELS:
            continue
        bbox = _normalised_bbox(detection.get("bbox_xyxy_norm"))
        if bbox is not None:
            regions.append({
                "type": detection["label_name"],
                "index": index,
                "bbox": bbox,
                "texts": _structured_text_candidates(detection),
            })
    return regions


def _has_authoritative_table(row: Dict[str, Any]) -> bool:
    """Return whether the selected pipeline actually emitted usable table content.

    Page Elements detecting a table is not sufficient evidence to delete the
    native PDFium characters. A VLM can reject a false positive or return no
    text; in that case the native text must remain intact.
    """
    values = row.get("table")
    if not isinstance(values, list):
        return False
    for item in values:
        if not isinstance(item, dict) or _normalised_bbox(item.get("bbox_xyxy_norm")) is None:
            continue
        if any(isinstance(item.get(key), str) and item.get(key).strip() for key in ("text", "markdown")):
            return True
        cells = item.get("cells")
        if isinstance(cells, list) and any(
            isinstance(cell, dict) and str(cell.get("text") or "").strip()
            for cell in cells
        ):
            return True
    return False


def _normalise_match_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(character for character in text if not unicodedata.combining(character))
    return " ".join(re.findall(r"[\w]+", text, flags=re.UNICODE))


def _line_matches_structured_text(line_text: str, region: Dict[str, Any]) -> bool:
    """Require strong textual evidence before suppressing a native line."""
    line_key = _normalise_match_text(line_text)
    if len(line_key) < 4:
        return False
    line_compact = line_key.replace(" ", "")
    for candidate in region.get("texts") or []:
        candidate_key = _normalise_match_text(candidate)
        if not candidate_key:
            continue
        candidate_compact = candidate_key.replace(" ", "")
        if len(line_key) < 8:
            if line_key in set(candidate_key.split()):
                return True
            continue
        if line_compact in candidate_compact:
            return True
        if len(line_key) >= 12 and SequenceMatcher(None, line_key, candidate_key).ratio() >= 0.90:
            return True
    return False


def _bbox_intersection_over_line(line_box: Sequence[float], region_box: Sequence[float]) -> float:
    width = max(0.0, min(line_box[2], region_box[2]) - max(line_box[0], region_box[0]))
    height = max(0.0, min(line_box[3], region_box[3]) - max(line_box[1], region_box[1]))
    line_area = max(0.0, line_box[2] - line_box[0]) * max(0.0, line_box[3] - line_box[1])
    return (width * height) / line_area if line_area > 0.0 else 0.0


def _structured_region_between_lines(
    previous_box: Sequence[float],
    line_box: Sequence[float],
    region: Dict[str, Any],
) -> bool:
    """Return whether a structured region separates two native text lines.

    A native PDF can expose one character stream for text above, inside, and
    below a table/chart.  Merging those lines by vertical gap creates one
    giant text bbox that hides the structured block in the UI and corrupts
    reading order.  Structured regions are therefore hard boundaries for the
    native paragraph merger, even when their own text is not duplicated.
    """
    region_box = region.get("bbox") if isinstance(region, dict) else None
    if not isinstance(region_box, (list, tuple)) or len(region_box) != 4:
        return False
    try:
        px0, py0, px1, py1 = [float(value) for value in previous_box[:4]]
        lx0, ly0, lx1, ly1 = [float(value) for value in line_box[:4]]
        rx0, ry0, rx1, ry1 = [float(value) for value in region_box[:4]]
    except (TypeError, ValueError):
        return False

    # The text column must actually overlap the structured region.  This
    # avoids splitting two independent columns merely because a chart exists
    # elsewhere on the page.
    text_x0, text_x1 = max(px0, lx0), min(px1, lx1)
    region_x0, region_x1 = max(text_x0, rx0), min(text_x1, rx1)
    if region_x1 <= region_x0:
        return False

    # A line touching the region is also a boundary.  This matters for native
    # labels that sit on a chart/table edge and otherwise get merged into the
    # paragraph immediately beside it.
    if _bbox_intersection_over_line(previous_box, region_box) > 0.0:
        return True
    if _bbox_intersection_over_line(line_box, region_box) > 0.0:
        return True

    # Normal reading order is top-to-bottom.  Keep the reverse form for PDFs
    # whose character stream is emitted bottom-to-top.
    previous_above = py1 <= ry0 and ly0 >= ry1
    line_above = ly1 <= ry0 and py0 >= ry1
    return previous_above or line_above


def _clean_native_spans(
    spans: Sequence[Dict[str, Any]],
    regions: Sequence[Dict[str, Any]],
    *,
    preserve_visual_captions: bool = False,
    suppress_table_native_text: bool = False,
) -> tuple[str, List[Dict[str, Any]], int, Dict[str, int]]:
    """Route native characters by geometry before rebuilding text blocks.

    Visual regions use the raw model bbox. Native characters whose centers are
    inside a chart/image/infographic/stamp are removed from the paragraph
    block so vector labels in a PDF chart are not emitted twice. Characters
    outside the region stay in the native block, including a paragraph line
    immediately above or beside the chart. Tables keep the stricter
    text-match rule because their cell text is separately structured, unless
    a pipeline explicitly marks its table Markdown as authoritative for the
    detected table bbox.
    """
    lines: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    suppressed = 0
    visual_suppressed_by_region: Dict[str, int] = {}

    def flush() -> None:
        nonlocal current
        if current:
            lines.append(current)
            current = []

    for span in spans:
        char = str(span.get("char") or "")
        if char in {"\r", "\n"}:
            flush()
            continue
        bbox = _normalised_bbox(span.get("bbox_xyxy_norm"))
        current.append({"char": char, "bbox": bbox})
    flush()

    clean_lines: List[Dict[str, Any]] = []
    for line in lines:
        outside_line: List[Dict[str, Any]] = []
        line_boxes = [
            item.get("bbox")
            for item in line
            if isinstance(item.get("bbox"), (list, tuple))
        ]
        line_bbox = _union_bbox(line_boxes)
        if suppress_table_native_text and line_bbox is not None:
            authoritative_table = next(
                (
                    region
                    for region in regions
                    if region.get("type") == "table"
                    and _bbox_intersection_over_line(line_bbox, region["bbox"]) >= 0.85
                ),
                None,
            )
            if authoritative_table is not None:
                suppressed += sum(
                    1 for item in line if str(item.get("char") or "").strip()
                )
                continue
        for item in line:
            char = str(item.get("char") or "")
            bbox = item.get("bbox")
            if not char or bbox is None:
                outside_line.append(item)
                continue
            center_x = (bbox[0] + bbox[2]) / 2.0
            center_y = (bbox[1] + bbox[3]) / 2.0
            visual_region = next(
                (
                    region
                    for region in regions
                    if region.get("type") in {"chart", "image", "infographic", "stamp"}
                    and _bbox_contains_point(region["bbox"], center_x, center_y, margin=0.0)
                ),
                None,
            )
            preserve_caption = False
            if preserve_visual_captions and visual_region is not None and line_bbox is not None:
                region_bbox = visual_region.get("bbox")
                if isinstance(region_bbox, (list, tuple)) and len(region_bbox) == 4:
                    # Page Elements can include a caption immediately above a
                    # chart/image in its visual bbox. Preserve only that
                    # shallow top band; text inside the visual is still
                    # suppressed to avoid duplicate native text.
                    top = float(region_bbox[1])
                    preserve_caption = (
                        line_bbox[1] >= top - 0.005
                        and line_bbox[3] <= top + 0.035
                    )
            if visual_region is not None and not preserve_caption:
                if char.strip():
                    key = f'{visual_region.get("type", "visual")}:{visual_region.get("index", 0)}'
                    visual_suppressed_by_region[key] = visual_suppressed_by_region.get(key, 0) + 1
                    suppressed += 1
                continue
            outside_line.append(item)

        raw = "".join(item["char"] for item in outside_line)
        text = re.sub(r"\s+", " ", raw).strip()
        boxes = [item["bbox"] for item in outside_line if item.get("bbox") is not None and item["char"].strip()]
        bbox = _union_bbox(boxes)
        if not text or bbox is None:
            continue

        duplicate_table = any(
            region.get("type") == "table"
            and _bbox_intersection_over_line(bbox, region["bbox"]) >= 0.85
            and _line_matches_structured_text(text, region)
            for region in regions
        )
        if duplicate_table:
            suppressed += sum(1 for item in outside_line if str(item.get("char") or "").strip())
            continue
        clean_lines.append({"text": text, "bbox_xyxy_norm": bbox})

    # Merge adjacent lines into readable blocks, but stop across a real vertical gap.
    blocks: List[Dict[str, Any]] = []
    for line in clean_lines:
        if not blocks:
            blocks.append({"text": line["text"], "bbox_xyxy_norm": line["bbox_xyxy_norm"]})
            continue
        previous = blocks[-1]
        prev_box = previous["bbox_xyxy_norm"]
        line_box = line["bbox_xyxy_norm"]
        prev_height = max(prev_box[3] - prev_box[1], 0.005)
        vertical_gap = line_box[1] - prev_box[3]
        same_column = abs(line_box[0] - prev_box[0]) < 0.08 or min(line_box[2], prev_box[2]) > max(line_box[0], prev_box[0])
        crosses_structured_region = any(
            _structured_region_between_lines(prev_box, line_box, region)
            for region in regions
        )
        if vertical_gap <= max(prev_height * 1.35, 0.012) and same_column and not crosses_structured_region:
            previous["text"] = f'{previous["text"]} {line["text"]}'.strip()
            previous["bbox_xyxy_norm"] = _union_bbox([prev_box, line_box])
        else:
            blocks.append({"text": line["text"], "bbox_xyxy_norm": line_box})

    clean_text = "\n".join(block["text"] for block in blocks)
    return clean_text, blocks, suppressed, visual_suppressed_by_region


def clean_content_rows(batch_df: Any) -> Any:
    """Clean native PDF text against structured regions before embedding.

    PDFium native text is retained as ``raw_text``. Characters whose centers
    fall inside a detected table/chart/infographic bbox are removed from the
    clean text, and the remaining native text is split into geometry-bearing
    blocks for reading-order sorting in ``explode_content_to_rows``.
    """
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df

    cleaned_rows: List[Dict[str, Any]] = []
    for _, series in batch_df.iterrows():
        row = series.to_dict()
        # High-resolution scan pages keep a separate model-sized detector
        # raster. It is an internal transport column and must not be copied
        # into every final text/visual row.
        row.pop("page_elements_image", None)
        spans = row.get("_native_text_spans")
        if not isinstance(spans, list):
            cleaned_rows.append(row)
            continue

        metadata = row.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        pipeline = metadata.get("ocr_pipeline")
        raw_text = str(row.get("raw_text") or row.get("text") or "")
        regions = _structured_regions(
            row,
            # Pipeline 6 keeps Page Elements visual detections as evidence,
            # not as proof that native PDFium text must be deleted. Only a
            # visual crop that actually carried authoritative OCR text may
            # suppress native characters. Pipeline 7 follows the same rule.
            allow_page_elements_fallback=pipeline not in {
                "pipeline-option6",
                "pipeline-option7",
            },
        )
        preserve_visual_captions = metadata.get("ocr_pipeline") == "pipeline-ppocrv6"
        authoritative_table = bool(
            pipeline in {"pipeline-option6", "pipeline-option7"}
            and _has_authoritative_table(row)
        )
        clean_text, native_blocks, suppressed, visual_suppressed = _clean_native_spans(
            spans,
            regions,
            preserve_visual_captions=preserve_visual_captions,
            suppress_table_native_text=authoritative_table,
        )
        native_page_element_blocks = _project_native_text_to_page_elements(
            row,
            spans,
            authoritative_regions=regions,
        )
        # Pipeline 6 can merge PDFium text with VLM blocks for native pages.
        # Keep those supplemental blocks when the native spans are rebuilt;
        # otherwise retaining spans for table de-duplication would silently
        # discard the VLM text that filled a missing native region.
        if metadata.get("ocr_pipeline") == "pipeline-option6":
            supplemental_blocks = [
                item for item in _list_items(row.get("_ocr_text_blocks")) if isinstance(item, dict)
            ]
            if supplemental_blocks:
                native_blocks = list(native_blocks) + supplemental_blocks
        metadata["cleaning"] = {
            "algorithm": "bbox-dedup-v2",
            "native_spans": len(spans),
            "suppressed_native_characters": suppressed,
            "structured_regions": regions,
            "native_blocks": len(native_blocks),
            "raw_text_retained": True,
            "native_visual_characters_suppressed": sum(visual_suppressed.values()),
            "native_visual_suppression_by_region": visual_suppressed,
            "visual_regions_route_native_text_by_span": True,
            "table_suppression_requires_text_match": not authoritative_table,
        }

        row["raw_text"] = raw_text
        row["text"] = clean_text
        row["metadata"] = metadata
        # This is a work column. It is consumed here and not copied into every
        # final result row, which keeps retained result_data manageable.
        row["_native_text_blocks"] = native_blocks
        # Keep the semantic Page Elements projection as a private sidecar for
        # the visual inspector. The normal text/chunk path continues to use
        # the geometry-preserving native blocks above, so this cannot alter
        # embeddings or native text deduplication.
        row["_native_page_element_blocks"] = native_page_element_blocks
        row.pop("_native_text_spans", None)
        cleaned_rows.append(row)
    return pd.DataFrame(cleaned_rows).reset_index(drop=True)


def _is_pdf_row(row: Dict[str, Any]) -> bool:
    """Return whether a row belongs to a PDF source."""
    metadata = row.get("metadata")
    source = row.get("path") or row.get("source_id") or row.get("source")
    if not source and isinstance(metadata, dict):
        source = metadata.get("source_path") or metadata.get("source_id")
    return str(source or "").split("?", 1)[0].lower().endswith(".pdf")


def _reader_backend(row: Dict[str, Any], content_type: str) -> str | None:
    """Classify the reader that produced one PDF result row.

    ``pdfium`` supplies native page text without an inference call. Structured
    elements and pages marked ``needs_ocr_for_text`` come from the OCR/model
    branch. Non-PDF sources are deliberately left unlabelled here because
    their readers are not native PDF or OCR.
    """
    if not _is_pdf_row(row):
        return None
    if content_type != "text":
        return "ocr"
    metadata = row.get("metadata")
    needs_ocr = isinstance(metadata, dict) and bool(
        metadata.get("needs_ocr_for_text") or metadata.get("needs_ocr")
    )
    return "ocr" if needs_ocr else "native_pdf"


def _set_reader_backend(row: Dict[str, Any], content_type: str) -> None:
    """Persist reader provenance in both the row and its metadata."""
    backend = _reader_backend(row, content_type)
    if backend is None:
        return
    row["_reader_backend"] = backend
    metadata = row.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    metadata["reader_backend"] = backend
    row["metadata"] = metadata


def _preserved_native_content_type(row: Dict[str, Any]) -> str | None:
    """Return a non-PDF native content type when a mixed graph is exploded."""
    metadata = row.get("metadata")
    nested = metadata.get("content_metadata") if isinstance(metadata, dict) else None
    if not isinstance(nested, dict):
        return None
    source_type = str(nested.get("source_type") or "")
    return {
        "native_cell": "spreadsheet_table",
        "native_csv": "spreadsheet_table",
        "chart_data": "chart",
        "embedded_image": "image",
    }.get(source_type)


def _combine_text_with_content(row: Any, text_column: str, content_columns: Sequence[str]) -> str:
    """Combine page text with OCR content text for embedding."""
    parts = []
    base = row.get(text_column)
    if isinstance(base, str) and base.strip():
        parts.append(base.strip())
    for col in content_columns:
        content_list = row.get(col)
        if isinstance(content_list, list):
            for item in content_list:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                    caption = item.get("caption", "")
                    if isinstance(caption, str) and caption.strip():
                        parts.append(caption.strip())
    return "\n\n".join(parts) if parts else ""


def _deep_copy_row(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Copy row dicts without sharing nested mutable values across exploded rows."""
    import copy

    out: Dict[str, Any] = {}
    for key, value in row_dict.items():
        if isinstance(value, (dict, list)):
            out[key] = copy.deepcopy(value)
        else:
            out[key] = value
    return out


def _reading_order_key(row: Dict[str, Any], original_index: int) -> tuple[Any, ...]:
    """Sort page blocks in physical top-to-bottom/left-to-right order."""
    source = str(row.get("source_id") or row.get("path") or "")
    try:
        page = int(row.get("page_number") or 0)
    except (TypeError, ValueError):
        page = 0
    bbox = _normalised_bbox(row.get("_bbox_xyxy_norm"))
    content_type = str(row.get("_content_type") or "text")
    type_order = {"text": 0, "table": 1, "chart": 2, "infographic": 3, "stamp": 4}.get(content_type, 9)
    if bbox is None:
        # A legacy/page-level text row has no geometry but is still the
        # document's primary text stream. Keep it ahead of structured rows
        # whose physical bbox is known; OCR blocks with an explicit full-page
        # bbox already sort naturally at y=0.
        return (source, page, 0.0, 0.0, type_order, original_index)
    return (source, page, bbox[1], bbox[0], type_order, original_index)


def explode_content_to_rows(
    batch_df: Any,
    *,
    text_column: str = "text",
    content_columns: Sequence[str] = _CONTENT_COLUMNS,
    modality: str = "text",
    text_elements_modality: Optional[str] = None,
    structured_elements_modality: Optional[str] = None,
) -> Any:
    """Expand each page row into multiple rows for per-element embedding."""
    text_mod = text_elements_modality or modality
    struct_mod = structured_elements_modality or modality

    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df

    any_images = text_mod in IMAGE_MODALITIES or struct_mod in IMAGE_MODALITIES

    if not any(column in batch_df.columns for column in content_columns):
        batch_df = batch_df.copy()
        if text_mod in IMAGE_MODALITIES and "page_image" in batch_df.columns:
            batch_df["_image_b64"] = batch_df["page_image"].apply(
                lambda page_image: inline_image_b64(page_image) if isinstance(page_image, dict) else None
            )
        if "page_image" in batch_df.columns:
            batch_df["_stored_image_uri"] = batch_df["page_image"].apply(
                lambda page_image: page_image.get("stored_image_uri") if isinstance(page_image, dict) else None
            )
        batch_df["_embed_modality"] = text_mod
        for index in batch_df.index:
            row_dict = batch_df.loc[index].to_dict()
            _set_reader_backend(row_dict, "text")
            if "_reader_backend" in row_dict:
                batch_df.at[index, "_reader_backend"] = row_dict["_reader_backend"]
                batch_df.at[index, "metadata"] = row_dict["metadata"]
        return batch_df

    new_rows: List[Dict[str, Any]] = []
    for _, row in batch_df.iterrows():
        row_dict = row.to_dict()
        row_metadata = row_dict.get("metadata")
        option2_fast_copy = (
            isinstance(row_metadata, dict)
            and row_metadata.get("ocr_pipeline") == "pipeline-ppocrv6"
        )

        def copy_row_for_explode() -> Dict[str, Any]:
            """Copy one output row without cloning the immutable page raster."""

            # Option 2 keeps the page raster and model payloads as read-only
            # sidecars. Deep-copying those base64-heavy values once per text
            # block dominated the post-OCR stage on multi-page documents.
            # Every field mutated below is replaced before the row leaves
            # this function, so a shallow row copy is safe for this branch.
            return dict(row_dict) if option2_fast_copy else _deep_copy_row(row_dict)

        exploded_any = False

        page_images = [
            item for item in _list_items(row_dict.get("images")) if isinstance(item, dict)
        ]
        has_visual_rows = any(
            str(item.get("label_name") or "")
            in {"image", "chart", "infographic", "stamp"}
            and _normalised_bbox(item.get("bbox_xyxy_norm")) is not None
            for item in page_images
        )
        has_structured_content = has_visual_rows or any(
            any(
                isinstance(item, dict)
                and (item.get("text") or item.get("bbox_xyxy_norm"))
                for item in _list_items(row_dict.get(column))
            )
            for column in content_columns
        )

        def images_for_bbox(bbox: Any) -> List[Dict[str, Any]]:
            normalized = _normalised_bbox(bbox)
            if normalized is None:
                return []
            matched: List[Dict[str, Any]] = []
            for image in page_images:
                image_bbox = _normalised_bbox(image.get("bbox_xyxy_norm"))
                if image_bbox is not None and all(abs(image_bbox[i] - normalized[i]) <= 0.01 for i in range(4)):
                    matched.append(_deep_copy_row(image))
            return matched

        page_image = row_dict.get("page_image")
        page_image_b64: Optional[str] = None
        page_stored_uri: Optional[str] = None
        if isinstance(page_image, dict):
            page_stored_uri = page_image.get("stored_image_uri")
            if any_images:
                page_image_b64 = inline_image_b64(page_image)

        native_blocks = row_dict.get("_native_text_blocks")
        ocr_blocks = row_dict.get("_ocr_text_blocks")
        if isinstance(native_blocks, list) and native_blocks:
            text_items = [item for item in native_blocks if isinstance(item, dict)]
        elif isinstance(ocr_blocks, list) and ocr_blocks:
            # OCR blocks already carry page-level bboxes. Keeping them as
            # separate rows is what makes scan text hoverable in the UI and
            # prevents one full-page OCR string from hiding missed regions.
            text_items = [item for item in ocr_blocks if isinstance(item, dict)]
        else:
            page_text = row_dict.get(text_column)
            text_items = [{"text": page_text, "bbox_xyxy_norm": None}] if isinstance(page_text, str) else []

        for text_item in text_items:
            value = str(text_item.get("text") or "").strip()
            if not value:
                continue
            page_row = copy_row_for_explode()
            page_row[text_column] = value
            page_row["_embed_modality"] = text_mod
            page_row["_content_type"] = _preserved_native_content_type(row_dict) or "text"
            # Structured page content is emitted as its own physical row
            # below. Do not copy the whole page's chart/table lists onto
            # every native text chunk, or the dashboard/result serializer
            # will expose the same visual block repeatedly.
            for column in content_columns:
                page_row[column] = []
            if text_mod in IMAGE_MODALITIES:
                page_row["_image_b64"] = page_image_b64
            # A page-level images list is otherwise copied onto every OCR
            # text block. Keep it on the text row only when this page has no
            # structured/visual rows to which the images can be mapped.
            page_row["images"] = [] if has_structured_content else copy.deepcopy(page_images)
            page_row["_stored_image_uri"] = page_stored_uri
            page_row["_bbox_xyxy_norm"] = _normalised_bbox(text_item.get("bbox_xyxy_norm"))
            if text_item.get("confidence") is not None:
                page_row["confidence"] = text_item.get("confidence")
            for provenance_key in ("source", "ocr_source", "ocr_mode", "line_detector_score", "page_elements_score", "region_label", "model", "ocr_model"):
                if text_item.get(provenance_key) is not None:
                    page_row[provenance_key] = text_item.get(provenance_key)
            _set_reader_backend(page_row, "text")
            new_rows.append(page_row)
            exploded_any = True

        for column in content_columns:
            content_list = _list_items(row_dict.get(column))
            for item in content_list:
                if not isinstance(item, dict):
                    continue
                item_b64 = inline_image_b64(item) if struct_mod in IMAGE_MODALITIES else None
                # Emit rows for text and (optionally) caption fields.
                for field, content_type in [("text", column), ("caption", f"{column}_caption")]:
                    value = item.get(field, "")
                    if not isinstance(value, str) or not value.strip():
                        continue
                    content_row = copy_row_for_explode()
                    content_row[text_column] = value.strip()
                    content_row["_embed_modality"] = struct_mod
                    content_row["_content_type"] = content_type
                    for other_column in content_columns:
                        content_row[other_column] = []
                    if struct_mod in IMAGE_MODALITIES:
                        if item_b64:
                            content_row["_image_b64"] = item_b64
                        elif page_image_b64:
                            bbox = item.get("bbox_xyxy_norm")
                            if bbox and len(bbox) == 4:
                                cropped_b64, _ = _crop_b64_image_by_norm_bbox(page_image_b64, bbox_xyxy_norm=bbox)
                                content_row["_image_b64"] = cropped_b64
                            else:
                                content_row["_image_b64"] = page_image_b64
                        else:
                            content_row["_image_b64"] = None
                    content_row["_stored_image_uri"] = item.get("stored_image_uri") or page_stored_uri
                    content_row["_bbox_xyxy_norm"] = item.get("bbox_xyxy_norm")
                    content_row["images"] = images_for_bbox(item.get("bbox_xyxy_norm"))
                    _set_reader_backend(content_row, content_type)
                    new_rows.append(content_row)
                    exploded_any = True

        # ``image`` is a detector/OCR visual class, while chart/stamp can be
        # image-only when OCR returns no text. Emit a visual row so a
        # seal/photo/chart crop is not silently attached to the first text row.
        structured_bboxes = [
            _normalised_bbox(item.get("bbox_xyxy_norm"))
            for column in content_columns
            for item in _list_items(row_dict.get(column))
            if isinstance(item, dict)
        ]
        for image in page_images:
            label_name = str(image.get("label_name") or "")
            image_bbox = _normalised_bbox(image.get("bbox_xyxy_norm"))
            if label_name not in {"image", "chart", "infographic", "stamp"} or image_bbox is None:
                continue
            if any(
                candidate is not None
                and all(abs(candidate[i] - image_bbox[i]) <= 0.01 for i in range(4))
                for candidate in structured_bboxes
            ):
                continue
            image_row = copy_row_for_explode()
            image_row[text_column] = str(
                image.get("text") or image.get("caption") or ""
            ).strip()
            image_row["_embed_modality"] = struct_mod
            image_row["_content_type"] = label_name
            for column in content_columns:
                image_row[column] = []
            image_row["_bbox_xyxy_norm"] = image_bbox
            image_row["_image_b64"] = image.get("image_b64")
            image_row["images"] = [_deep_copy_row(image)]
            image_row["_stored_image_uri"] = image.get("stored_image_uri") or page_stored_uri
            _set_reader_backend(image_row, label_name)
            new_rows.append(image_row)
            exploded_any = True

        if not exploded_any:
            preserved = copy_row_for_explode()
            preserved["_embed_modality"] = text_mod
            preserved["_content_type"] = _preserved_native_content_type(row_dict) or "text"
            if text_mod in IMAGE_MODALITIES:
                preserved["_image_b64"] = page_image_b64
            preserved["_stored_image_uri"] = page_stored_uri
            preserved["_bbox_xyxy_norm"] = None
            _set_reader_backend(preserved, "text")
            new_rows.append(preserved)

    # The previous implementation emitted page text first, then table/chart
    # columns.  That is data-column order, not reading order.  Sort only after
    # all blocks for all pages have been created, then retain an explicit order
    # field for downstream debugging and UI display.
    ordered_rows = sorted(enumerate(new_rows), key=lambda item: _reading_order_key(item[1], item[0]))
    order_by_page: Dict[tuple[str, int], int] = {}
    raw_owner_by_page: set[tuple[str, int]] = set()
    seen_image_keys_by_page: Dict[tuple[str, int], set[tuple[Any, ...]]] = {}
    final_rows: List[Dict[str, Any]] = []
    for _, row in ordered_rows:
        source = str(row.get("source_id") or row.get("path") or "")
        try:
            page = int(row.get("page_number") or 0)
        except (TypeError, ValueError):
            page = 0
        page_key = (source, page)
        order_by_page[page_key] = order_by_page.get(page_key, 0) + 1
        row["_reading_order"] = order_by_page[page_key]
        # Keep the full native source only once per page. Structured rows use
        # their own clean text and should not make raw page text look like a
        # second copy of their payload in retained result_data.
        if "raw_text" in row:
            if str(row.get("_content_type") or "text") == "text" and page_key not in raw_owner_by_page:
                raw_owner_by_page.add(page_key)
            else:
                row.pop("raw_text", None)
        row.pop("_native_text_blocks", None)
        row.pop("_native_text_spans", None)
        row.pop("_ocr_text_blocks", None)
        if "images" in row:
            unique_images: List[Dict[str, Any]] = []
            seen_image_keys = seen_image_keys_by_page.setdefault(page_key, set())
            for image in _list_items(row.get("images")):
                if not isinstance(image, dict):
                    continue
                bbox = _normalised_bbox(image.get("bbox_xyxy_norm"))
                bbox_key = tuple(round(value, 4) for value in bbox) if bbox is not None else (None,)
                image_key = (str(image.get("label_name") or ""), bbox_key, str(image.get("source") or ""))
                if image_key in seen_image_keys:
                    continue
                seen_image_keys.add(image_key)
                unique_images.append(image)
            row["images"] = unique_images
        final_rows.append(row)

    return pd.DataFrame(final_rows).reset_index(drop=True)


def chunk_pdf_content_rows(batch_df: Any, params: TextChunkParams | None = None) -> Any:
    """Chunk parsed PDF text blocks without losing page/block provenance.

    PDF extraction produces geometry-bearing ``_native_text_blocks`` or
    ``_ocr_text_blocks``.  The generic token splitter used to run before the
    content exploder and therefore its output could be ignored: the exploder
    preferred the original geometry blocks.  This transform chunks those
    canonical blocks first, while leaving tables, charts, images and stamps
    atomic.  It is intentionally PDF-specific; text/HTML/audio keep their
    existing chunkers.

    Every emitted text row keeps the source PDF page number.  Chunk identity
    and parent-block identity are persisted in metadata so adjacent chunks
    can be reconstructed without pretending that a token chunk is a new PDF
    page.
    """
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df

    from nemo_retriever.common.modality.txt.split import _get_tokenizer, split_text_by_tokens

    chunk_params = params or TextChunkParams()
    tokenizer = _get_tokenizer(
        chunk_params.tokenizer_model_id or "nvidia/llama-nemotron-embed-1b-v2",
        cache_dir=chunk_params.tokenizer_cache_dir,
    )

    def _copy(value: Any) -> Any:
        return copy.deepcopy(value) if isinstance(value, (dict, list)) else value

    def _source_key(row: Dict[str, Any]) -> str:
        return str(row.get("source_id") or row.get("path") or row.get("source") or "document")

    def _page_number(row: Dict[str, Any]) -> int:
        try:
            return int(row.get("page_number") or 0)
        except (TypeError, ValueError):
            return 0

    emitted: List[Dict[str, Any]] = []
    for _, series in batch_df.iterrows():
        row = series.to_dict()
        source = _source_key(row)
        page = _page_number(row)

        blocks = row.get("_native_text_blocks")
        reader_blocks = "native"
        if not isinstance(blocks, list) or not blocks:
            blocks = row.get("_ocr_text_blocks")
            reader_blocks = "ocr"
        if not isinstance(blocks, list) or not blocks:
            text = row.get("text")
            blocks = [{"text": text, "bbox_xyxy_norm": None}] if isinstance(text, str) and text.strip() else []
            reader_blocks = "page"

        # A page with no text can still contain an image/chart/stamp. Preserve
        # it as one row; visual content is intentionally never token-split.
        if not blocks:
            emitted.append(row)
            continue

        page_rows: List[Dict[str, Any]] = []
        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            text = str(block.get("text") or "").strip()
            if not text:
                continue

            parent_id = f"{source}::page-{page}::block-{block_index}"
            chunks = split_text_by_tokens(
                text,
                tokenizer=tokenizer,
                max_tokens=int(chunk_params.max_tokens),
                overlap_tokens=int(chunk_params.overlap_tokens),
            ) or [text]

            for chunk_index, chunk_text in enumerate(chunks):
                child = {key: _copy(value) for key, value in row.items()}
                child["text"] = chunk_text.strip()
                child["_native_text_blocks"] = []
                child["_ocr_text_blocks"] = []
                block_payload = {
                    "text": child["text"],
                    "bbox_xyxy_norm": _normalised_bbox(block.get("bbox_xyxy_norm")),
                    "parent_block_id": parent_id,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                }
                if block.get("confidence") is not None:
                    block_payload["confidence"] = block.get("confidence")
                    child["confidence"] = block.get("confidence")
                for provenance_key in ("source", "ocr_source", "ocr_mode", "line_detector_score", "page_elements_score", "region_label", "model", "ocr_model"):
                    if block.get(provenance_key) is not None:
                        block_payload[provenance_key] = block.get(provenance_key)
                        child[provenance_key] = block.get(provenance_key)
                if reader_blocks == "native":
                    child["_native_text_blocks"] = [block_payload]
                else:
                    child["_ocr_text_blocks"] = [block_payload]
                child["_content_block_id"] = f"{parent_id}::chunk-{chunk_index}"
                child["_parent_block_id"] = parent_id
                child["_chunk_index"] = chunk_index
                child["_chunk_count"] = len(chunks)
                metadata = child.get("metadata")
                metadata = dict(metadata) if isinstance(metadata, dict) else {}
                metadata["parse_chunk"] = {
                    "algorithm": "pdf-block-token-v1",
                    "reader_blocks": reader_blocks,
                    "parent_block_id": parent_id,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "page_number": page,
                    "bbox_xyxy_norm": block_payload["bbox_xyxy_norm"],
                    "max_tokens": int(chunk_params.max_tokens),
                    "overlap_tokens": int(chunk_params.overlap_tokens),
                }
                child["metadata"] = metadata
                page_rows.append(child)

        if not page_rows:
            emitted.append(row)
            continue

        # Structured content belongs to the page, not to every text chunk.
        # Keep it on the first emitted row only; explode_content_to_rows then
        # emits one table/chart/stamp/image block rather than duplicates.
        for index, child in enumerate(page_rows):
            if index > 0:
                for column in (*_CONTENT_COLUMNS, "images"):
                    if column in child:
                        child[column] = []
            emitted.append(child)

    if not emitted:
        return batch_df.iloc[:0].copy()
    return pd.DataFrame(emitted).reset_index(drop=True)


def collapse_content_to_page_rows(
    batch_df: Any,
    *,
    text_column: str = "text",
    content_columns: Sequence[str] = _CONTENT_COLUMNS,
    modality: str = "text",
) -> Any:
    """Collapse each page into a single row for page-level embedding."""
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df

    batch_df = batch_df.copy()
    batch_df[text_column] = batch_df.apply(
        lambda row: _combine_text_with_content(row, text_column, content_columns),
        axis=1,
    )

    if modality in IMAGE_MODALITIES:
        if "page_image" in batch_df.columns:
            batch_df["_image_b64"] = batch_df["page_image"].apply(
                lambda page_image: inline_image_b64(page_image) if isinstance(page_image, dict) else None
            )
        else:
            batch_df["_image_b64"] = None

    if "page_image" in batch_df.columns:
        batch_df["_stored_image_uri"] = batch_df["page_image"].apply(
            lambda page_image: page_image.get("stored_image_uri") if isinstance(page_image, dict) else None
        )

    batch_df["_embed_modality"] = modality
    return batch_df
