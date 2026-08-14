"""Page Elements crop OCR adapters for document ingest.

    The document pipeline deliberately keeps detection and recognition as two
    separate HTTP services. Page Elements owns semantic regions. Option 2 keeps
    that original route and adds PP-OCRv6 line detection inside each semantic
    text region before GPU recognition.
"""

from __future__ import annotations

import base64
import io
import math
import time
from difflib import SequenceMatcher
from typing import Any, Iterable, Sequence

import pandas as pd

from nemo_retriever.common.params import RemoteRetryParams
from nemo_retriever.models.nim.nim import (
    NIMClient,
    invoke_image_inference_batches,
)


_TEXT_LABELS = frozenset({"text", "title", "header_footer"})
_VISUAL_LABELS = frozenset({"image", "chart", "infographic", "stamp"})
_BOX_GEOMETRY_DUPLICATE_CONTAINMENT = 0.88
_BOX_GEOMETRY_DUPLICATE_IOU = 0.72

def _b64_crop(image_b64: str, bbox: Sequence[float]) -> tuple[str, tuple[int, int]] | None:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(base64.b64decode(image_b64))) as source:
            image = source.convert("RGB")
            width, height = image.size
            x0, y0, x1, y1 = [float(v) for v in bbox]
            # Preserve the nearest detector edge. Truncating both sides can
            # clip thin glyph strokes before the recognizer sees the line.
            left = max(0, min(width - 1, int(round(x0 * width))))
            top = max(0, min(height - 1, int(round(y0 * height))))
            right = max(left + 1, min(width, int(round(x1 * width))))
            bottom = max(top + 1, min(height, int(round(y1 * height))))
            crop = image.crop((left, top, right, bottom))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii"), (crop.height, crop.width)
    except Exception:
        return None


def _map_box(parent: Sequence[float], local: Sequence[float], shape_hw: Sequence[int]) -> list[float]:
    px0, py0, px1, py1 = [float(v) for v in parent]
    lx0, ly0, lx1, ly1 = [float(v) for v in local]
    height, width = max(1, int(shape_hw[0])), max(1, int(shape_hw[1]))
    # PP-OCRv6 returns normalized line boxes, while older detector adapters
    # may return pixel coordinates.  Do not divide normalized coordinates by
    # the crop dimensions a second time.
    if max(abs(value) for value in (lx0, ly0, lx1, ly1)) <= 1.5:
        local_x0, local_y0, local_x1, local_y1 = lx0, ly0, lx1, ly1
    else:
        local_x0, local_y0 = lx0 / width, ly0 / height
        local_x1, local_y1 = lx1 / width, ly1 / height
    return [
        px0 + local_x0 * (px1 - px0),
        py0 + local_y0 * (py1 - py0),
        px0 + local_x1 * (px1 - px0),
        py0 + local_y1 * (py1 - py0),
    ]


def _response_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        value = response.get("result")
        if isinstance(value, dict):
            return value
        return response
    return {}


def _detector_boxes(response: Any) -> list[tuple[list[float], float]]:
    value = _response_dict(response)
    raw = value.get("boxes") or value.get("dt_boxes") or value.get("rec_boxes") or []
    scores = value.get("scores") or value.get("dt_scores") or []
    result: list[tuple[list[float], float]] = []
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            box = item.get("bbox") or item.get("bbox_xyxy") or item.get("box")
            score = item.get("score", 1.0)
        else:
            box, score = item, scores[index] if index < len(scores) else 1.0
        if not isinstance(box, (list, tuple)):
            continue
        if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            bbox = [float(v) for v in box]
        elif len(box) >= 4 and isinstance(box[0], (list, tuple)):
            points = [(float(p[0]), float(p[1])) for p in box if len(p) >= 2]
            if not points:
                continue
            bbox = [min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points)]
        else:
            continue
        result.append((bbox, float(score or 0.0)))
    return result


def _recognized_text(response: Any) -> tuple[str, float | None]:
    value = _response_dict(response)
    texts = value.get("texts") or value.get("rec_texts")
    scores = value.get("scores") or value.get("rec_scores")
    text = value.get("text") or value.get("rec_text")
    score = value.get("score")
    if score is None:
        score = value.get("rec_score")
    if isinstance(texts, list):
        text = texts[0] if texts else ""
        score = scores[0] if isinstance(scores, list) and scores else score
    return str(text or "").strip(), float(score) if isinstance(score, (int, float)) else None


def _recognizer_source(response: Any) -> str:
    """Expose the actual recognizer backend in visual-evidence blocks."""
    value = _response_dict(response)
    backend = str(value.get("backend") or value.get("model") or "").lower()
    if "tesseract" in backend:
        return "tesseract-5"
    return "ppocrv6_recognizer"


def _bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_intersection(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    width = max(0.0, min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0])))
    height = max(0.0, min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1])))
    return width * height


def _bbox_containment(left: Sequence[float], right: Sequence[float]) -> float:
    smaller = min(_bbox_area(left), _bbox_area(right))
    return _bbox_intersection(left, right) / smaller if smaller > 0.0 else 0.0


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = _bbox_intersection(left, right)
    union = _bbox_area(left) + _bbox_area(right) - intersection
    return intersection / union if union > 0.0 else 0.0


def _bbox_geometry_is_duplicate(left: Sequence[float], right: Sequence[float]) -> bool:
    return (
        _bbox_containment(left, right) >= _BOX_GEOMETRY_DUPLICATE_CONTAINMENT
        or _bbox_iou(left, right) >= _BOX_GEOMETRY_DUPLICATE_IOU
    )


def _ocr_text_key(value: Any) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKD", str(value or "").casefold()).replace("đ", "d")
    return "".join(char for char in text if not unicodedata.combining(char) and char.isalnum())


def _ocr_text_related(left: Any, right: Any) -> bool:
    left_key = _ocr_text_key(left)
    right_key = _ocr_text_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    shorter, longer = sorted((left_key, right_key), key=len)
    if len(shorter) >= 4 and shorter in longer:
        return True
    return len(shorter) >= 8 and SequenceMatcher(None, left_key, right_key).ratio() >= 0.82


def _expand_bbox(bbox: Sequence[float], padding: float = 0.01) -> list[float]:
    values = [float(value) for value in bbox[:4]]
    return [
        max(0.0, values[0] - padding),
        max(0.0, values[1] - padding),
        min(1.0, values[2] + padding),
        min(1.0, values[3] + padding),
    ]


def _region_geometry(region: dict[str, Any]) -> tuple[list[float], list[float]] | None:
    """Return raw model geometry for ownership and a tight OCR crop.

    Page Elements postprocessing may expand a text region for layout matching.
    That expanded rectangle is not suitable as the canonical OCR bbox: it can
    swallow a neighbouring chart or several unrelated lines.  The raw model
    bbox is therefore used for option 2's output and receives only a small crop
    margin for glyph edges.
    """
    value = region.get("model_bbox_xyxy_norm") or region.get("bbox_xyxy_norm")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in bbox):
        return None
    bbox = [max(0.0, min(1.0, item)) for item in bbox]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox, _expand_bbox(bbox)


def _deduplicate_box_blocks(blocks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge only repeated region/page/tile outputs, never adjacent lines."""
    result: list[dict[str, Any]] = []
    for candidate in blocks:
        if not isinstance(candidate, dict) or not str(candidate.get("text") or "").strip():
            continue
        candidate = dict(candidate)
        duplicate = None
        matched_left = None
        matched_right = None
        for index, existing in enumerate(result):
            left = candidate.get("bbox_xyxy_norm")
            right = existing.get("bbox_xyxy_norm")
            if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
                continue
            if _bbox_containment(left, right) < 0.80 and _bbox_iou(left, right) < _BOX_GEOMETRY_DUPLICATE_IOU:
                continue
            text_related = _ocr_text_related(candidate.get("text"), existing.get("text"))
            if not text_related and not _bbox_geometry_is_duplicate(left, right):
                continue
            duplicate = index
            matched_left = left
            matched_right = right
            break
        if duplicate is None:
            source = candidate.get("source")
            candidate.setdefault("ocr_sources", [str(source)] if source else [])
            result.append(candidate)
            continue
        existing = result[duplicate]
        sources = set(existing.get("ocr_sources") or [])
        sources.update(candidate.get("ocr_sources") or [])
        if candidate.get("source"):
            sources.add(str(candidate["source"]))
        existing["ocr_sources"] = sorted(source for source in sources if source)
        existing_score = existing.get("score")
        candidate_score = candidate.get("score")
        prefer = False
        if (
            matched_left is not None
            and matched_right is not None
            and _bbox_geometry_is_duplicate(matched_left, matched_right)
        ):
            candidate_length = len(str(candidate.get("text") or "").strip())
            existing_length = len(str(existing.get("text") or "").strip())
            if candidate_length > max(12, int(existing_length * 1.15)):
                prefer = True
            elif existing_length > max(12, int(candidate_length * 1.15)):
                prefer = False
        try:
            prefer = prefer or candidate_score is not None and (
                existing_score is None or float(candidate_score) > float(existing_score) + 0.025
            )
        except (TypeError, ValueError):
            pass
        if not prefer and len(str(candidate.get("text") or "")) > len(str(existing.get("text") or "")):
            prefer = True
        if prefer:
            replacement = dict(candidate)
            replacement["ocr_sources"] = sorted(source for source in sources if source)
            result[duplicate] = replacement
    return sorted(
        result,
        key=lambda item: (
            float((item.get("bbox_xyxy_norm") or [0.0, 0.0])[1]),
            float((item.get("bbox_xyxy_norm") or [0.0, 0.0])[0]),
        ),
    )


def _visual_text_coverage(bbox: Sequence[float], detections: Sequence[dict[str, Any]]) -> float:
    """Estimate how much a visual candidate is already covered by text boxes."""
    area = _bbox_area(bbox)
    if area <= 0.0:
        return 0.0
    text_boxes = [
        detection.get("bbox_xyxy_norm")
        for detection in detections
        if isinstance(detection, dict)
        and str(detection.get("label_name") or "") in _TEXT_LABELS
        and isinstance(detection.get("bbox_xyxy_norm"), (list, tuple))
        and len(detection["bbox_xyxy_norm"]) == 4
    ]
    if not text_boxes:
        return 0.0
    # A fixed grid gives a deterministic union estimate without adding a
    # geometry dependency to the service image.
    samples = 40
    covered = 0
    for x_index in range(samples):
        x = float(bbox[0]) + (x_index + 0.5) / samples * (float(bbox[2]) - float(bbox[0]))
        for y_index in range(samples):
            y = float(bbox[1]) + (y_index + 0.5) / samples * (float(bbox[3]) - float(bbox[1]))
            if any(float(box[0]) <= x <= float(box[2]) and float(box[1]) <= y <= float(box[3]) for box in text_boxes):
                covered += 1
    return covered / float(samples * samples)


def _keep_visual_detection(detection: dict[str, Any], detections: Sequence[dict[str, Any]]) -> bool:
    """Keep bounded visual regions, rejecting page/header false positives."""
    label = str(detection.get("label_name") or "")
    bbox = detection.get("bbox_xyxy_norm")
    if label not in _VISUAL_LABELS or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    area = _bbox_area(bbox)
    # The page raster is already retained separately; never emit a page-sized
    # crop as a second image block.
    if area >= 0.80:
        return False
    if label != "infographic":
        return True
    try:
        score = float(detection.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    # Large, low-confidence infographic boxes dominated by text are usually
    # logos/headers or a scan-page false positive, not an image object.
    return not (area >= 0.12 and score < 0.75 and _visual_text_coverage(bbox, detections) >= 0.30)


def _invoke(
    url: str,
    images: list[str],
    *,
    api_key: str | None,
    request_timeout_s: float,
    retry: RemoteRetryParams,
    nim_client: NIMClient | None,
) -> list[Any]:
    if not images:
        return []
    kwargs = {
        "invoke_url": url,
        "image_b64_list": images,
        "api_key": api_key or None,
        "timeout_s": float(request_timeout_s),
        "max_batch_size": max(1, min(8, len(images))),
        "max_retries": int(retry.remote_max_retries),
        "max_429_retries": int(retry.remote_max_429_retries),
    }
    if nim_client is not None:
        return list(nim_client.invoke_image_inference_batches(**kwargs))
    return list(invoke_image_inference_batches(**kwargs, max_pool_workers=int(retry.remote_max_pool_workers)))


def _run_option2_scan_recall(
    image_b64: str,
    *,
    api_key: str | None,
    request_timeout_s: float,
    retry: RemoteRetryParams,
    nim_client: NIMClient | None,
    ocr_url: str,
    tile_size: int,
    tile_overlap: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the existing full-page/tile recall using the Option 2 recognizer.

    The recognizer sidecar returns text for a crop rather than detector boxes.
    Consequently recall blocks deliberately retain the full-page/tile bbox;
    they are used only when Page Elements produced no usable text regions, so
    they cannot duplicate or replace a good region result.
    """
    from nemo_retriever.common.modality.ocr.shared import _scan_ocr_tiles_from_b64

    jobs: list[tuple[str, list[float], str]] = [("scan_full_page", [0.0, 0.0, 1.0, 1.0], image_b64)]
    jobs.extend(
        ("scan_tile", bbox, tile)
        for bbox, tile in _scan_ocr_tiles_from_b64(
            image_b64,
            tile_size=max(256, int(tile_size)),
            overlap=min(0.45, max(0.0, float(tile_overlap))),
        )
    )
    errors: list[str] = []
    try:
        responses = _invoke(
            ocr_url,
            [image for _source, _bbox, image in jobs],
            api_key=api_key,
            request_timeout_s=request_timeout_s,
            retry=retry,
            nim_client=nim_client,
        )
    except Exception as exc:  # page-local recall; region OCR remains usable.
        errors.append(f"{type(exc).__name__}: {exc}")
        responses = []

    blocks: list[dict[str, Any]] = []
    for (source, bbox, _crop), response in zip(jobs, responses):
        text, score = _recognized_text(response)
        if not text:
            continue
        blocks.append(
            {
                "bbox_xyxy_norm": bbox,
                "model_bbox_xyxy_norm": bbox,
                "processed_bbox_xyxy_norm": bbox,
                "crop_bbox_xyxy_norm": bbox,
                "text": text,
                "score": score,
                "confidence": score,
                "source": "ppocrv6_recognizer",
                "ocr_source": "ppocrv6_recognizer",
                "ocr_mode": source,
                "region_label": "scan_recall",
                "scan_recall": True,
                "sort_y": float(bbox[1]),
                "sort_x": float(bbox[0]),
            }
        )

    scores = [float(block["score"]) for block in blocks if isinstance(block.get("score"), (int, float))]
    quality = {
        "enabled": True,
        "full_page": True,
        "tiles": max(0, len(jobs) - 1),
        "tile_size": max(256, int(tile_size)),
        "tile_overlap": min(0.45, max(0.0, float(tile_overlap))),
        "requests": len(jobs),
        "responses": len(responses),
        "blocks": len(blocks),
        "text_chars": sum(len(str(block.get("text") or "")) for block in blocks),
        "mean_confidence": sum(scores) / len(scores) if scores else None,
        "errors": errors,
    }
    return blocks, quality


def _table_regions(row: Any) -> Iterable[tuple[list[float], list[dict[str, Any]], tuple[int, int]]]:
    payload = getattr(row, "table_structure_v1", None)
    if not isinstance(payload, dict):
        return
    for region in payload.get("regions") or []:
        if not isinstance(region, dict):
            continue
        bbox = region.get("bbox_xyxy_norm")
        detections = region.get("detections") or []
        shape = region.get("orig_shape_hw") or [1, 1]
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and isinstance(detections, list):
            yield [float(v) for v in bbox], detections, (int(shape[0]), int(shape[1]))


def _markdown_cells(cells: list[dict[str, Any]]) -> str:
    if not cells:
        return ""
    ordered = sorted(cells, key=lambda item: (float(item["bbox_xyxy_norm"][1]), float(item["bbox_xyxy_norm"][0])))
    rows: list[list[dict[str, Any]]] = []
    for cell in ordered:
        cy = (cell["bbox_xyxy_norm"][1] + cell["bbox_xyxy_norm"][3]) / 2
        row = next((r for r in rows if abs(cy - r[0]["_cy"]) <= max(0.012, (cell["bbox_xyxy_norm"][3] - cell["bbox_xyxy_norm"][1]) * 0.7)), None)
        if row is None:
            row = [{"_cy": cy}]
            rows.append(row)
        row.append(cell)
    lines: list[str] = []
    for row in rows:
        values = [cell.get("text", "").replace("|", "\\|").replace("\n", " ").strip() for cell in sorted(row[1:], key=lambda item: item["bbox_xyxy_norm"][0])]
        if values:
            lines.append("| " + " | ".join(values) + " |")
    if not lines:
        return ""
    return "\n".join([lines[0], "| " + " | ".join("---" for _ in lines[0].split("|")[1:-1]) + " |", *lines[1:]])


def ppocrv6_page_elements(
    batch_df: pd.DataFrame,
    *,
    line_detector_invoke_url: str | None = None,
    ocr_recognizer_invoke_url: str,
    page_elements_invoke_url: str | None = None,
    box_ocr_mode: bool = False,
    api_key: str | None = None,
    request_timeout_s: float = 120.0,
    inference_batch_size: int = 8,
    remote_retry: RemoteRetryParams | None = None,
    nim_client: NIMClient | None = None,
    extract_text: bool = True,
    extract_tables: bool = True,
    extract_charts: bool = True,
    extract_infographics: bool = True,
    extract_images: bool = True,
    use_table_structure: bool = True,
    # The graph passes ExtractParams.scan_ocr_fallback explicitly. Keep the
    # direct adapter default disabled so callers that only want one crop per
    # Page Elements box do not unexpectedly issue a page/tile request.
    scan_ocr_fallback: bool = False,
    scan_ocr_tile_size: int = 1024,
    scan_ocr_tile_overlap: float = 0.15,
    **_: Any,
) -> pd.DataFrame:
    """Run Page Elements crop OCR, optionally with PP-OCRv6 line splitting.

    ``box_ocr_mode=True`` is Option 2: Page Elements bboxes remain the
    authoritative semantic regions, PP-OCRv6 detects lines only inside those
    regions, and the GPU PP-OCRv6 recognizer reads those line crops. Scan pages
    additionally run the existing full-page/tile recall only when the region
    pass returns no usable text.
    """
    retry = remote_retry or RemoteRetryParams()
    all_text: list[str | None] = []
    all_tables: list[list[dict[str, Any]]] = []
    all_charts: list[list[dict[str, Any]]] = []
    all_infographics: list[list[dict[str, Any]]] = []
    all_images: list[list[dict[str, Any]]] = []
    all_blocks: list[list[dict[str, Any]]] = []
    all_meta: list[dict[str, Any]] = []

    for row in batch_df.itertuples(index=False):
        page_image = getattr(row, "page_image", None) or {}
        image_b64 = page_image.get("image_b64") if isinstance(page_image, dict) else None
        pe = getattr(row, "page_elements_v3", None)
        detections = list(pe.get("detections") or []) if isinstance(pe, dict) else []
        text_blocks: list[dict[str, Any]] = []
        tables: list[dict[str, Any]] = []
        charts: list[dict[str, Any]] = []
        infographics: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []
        errors: list[str] = []
        line_detection_meta: dict[str, Any] = {
            "enabled": False,
            "model": None,
            "region_count": 0,
            "line_count": 0,
            "fallback_regions": 0,
        }
        line_count = 0
        rec_count = 0
        started = time.perf_counter()

        existing_text = getattr(row, "text", None)
        if not isinstance(image_b64, str) or not image_b64:
            all_text.append(existing_text); all_tables.append([]); all_charts.append([]); all_infographics.append([]); all_images.append([]); all_blocks.append([])
            all_meta.append({
                "stage": "page_elements_box_ocr" if box_ocr_mode else "ppocrv6_line_det_rec",
                "pipeline": "pipeline-tesseract" if box_ocr_mode else "split_ppocrv6",
                "error": "page image unavailable",
                "timing": {"seconds": 0.0},
            })
            continue

        row_metadata = getattr(row, "metadata", {}) or {}
        wants_text = bool(extract_text and row_metadata.get("needs_ocr_for_text", False))
        text_regions = [d for d in detections if str(d.get("label_name") or "") in _TEXT_LABELS]
        line_jobs: list[tuple[list[float], str, tuple[int, int], dict[str, Any]]] = []
        region_crops: list[str] = []
        for region in text_regions if wants_text else []:
            geometry = _region_geometry(region)
            if geometry is None:
                continue
            bbox, crop_bbox = geometry
            crop = _b64_crop(image_b64, crop_bbox)
            if crop:
                region_crops.append(crop[0])
                line_jobs.append(([float(v) for v in bbox], crop[0], crop[1], region))

        if box_ocr_mode and region_crops:
            # Option 2 keeps Page Elements as the semantic detector, then
            # uses PP-OCRv6 detection only to split each Page Elements text
            # crop into line boxes before GPU recognition.
            try:
                if line_detector_invoke_url:
                    detected_lines = _invoke(
                        line_detector_invoke_url,
                        region_crops,
                        api_key=api_key,
                        request_timeout_s=request_timeout_s,
                        retry=retry,
                        nim_client=nim_client,
                    )
                    line_detection_meta.update(
                        {
                            "enabled": True,
                            "model": "PP-OCRv6_medium_det",
                            "region_count": len(region_crops),
                        }
                    )
                else:
                    # Direct adapter callers may omit the optional endpoint;
                    # the service wiring supplies it for production Option 2.
                    detected_lines = [None] * len(region_crops)
                    line_detection_meta["fallback_regions"] = len(region_crops)
                rec_crops: list[str] = []
                rec_boxes: list[list[float]] = []
                rec_regions: list[dict[str, Any]] = []
                for (region_bbox, _region_b64, region_shape, region), detector_result in zip(line_jobs, detected_lines):
                    boxes = _detector_boxes(detector_result)
                    # Preserve the old one-box behavior only when PP-OCRv6
                    # cannot find a line inside this Page Elements crop.
                    if not boxes:
                        boxes = [([0.0, 0.0, 1.0, 1.0], float(region.get("score") or 0.0))]
                    for local_box, line_score in boxes:
                        mapped = _map_box(region_bbox, local_box, region_shape)
                        crop = _b64_crop(image_b64, mapped)
                        if crop:
                            rec_crops.append(crop[0])
                            rec_boxes.append(mapped)
                            rec_regions.append(
                                {
                                    "region": region,
                                    "page_bbox": list(region_bbox),
                                    "score": line_score,
                                }
                            )
                recognized = _invoke(
                    ocr_recognizer_invoke_url,
                    rec_crops,
                    api_key=api_key,
                    request_timeout_s=request_timeout_s,
                    retry=retry,
                    nim_client=nim_client,
                )
                line_count = len(rec_crops)
                line_detection_meta["line_count"] = line_count
                for bbox, info, result in zip(rec_boxes, rec_regions, recognized):
                    text, score = _recognized_text(result)
                    if not text:
                        continue
                    text_blocks.append({
                        "bbox_xyxy_norm": bbox,
                        # Keep both layers for the dashboard: this is the
                        # original Page Elements semantic bbox, while
                        # ``bbox_xyxy_norm`` is the PP-OCRv6 line bbox.
                        "model_bbox_xyxy_norm": info.get("page_bbox") or bbox,
                        "processed_bbox_xyxy_norm": info["region"].get("processed_bbox_xyxy_norm"),
                        "crop_bbox_xyxy_norm": _expand_bbox(bbox),
                        "text": text,
                        "score": score,
                        "confidence": score,
                        "source": _recognizer_source(result),
                        "ocr_source": _recognizer_source(result),
                        "ocr_mode": (
                            "page_elements_ppocr_line"
                            if line_detection_meta["enabled"]
                            else "page_elements_box"
                        ),
                        "region_label": info["region"].get("label_name"),
                        "page_elements_score": info["region"].get("score"),
                        "line_detector_score": info["score"],
                    })
                rec_count = len(recognized)
            except Exception as exc:
                errors.append(f"box_text: {type(exc).__name__}: {exc}")
        elif region_crops:
            if not line_detector_invoke_url:
                errors.append("line_text: line_detector_invoke_url is required outside box_ocr_mode")
            else:
                # Line-based option: Page Elements regions are subdivided by
                # the dedicated PP-OCRv6 detector before recognition.
                # This branch is intentionally kept for the non-option-2
                # pipeline and is not used by pipeline-tesseract.
                try:
                    detected_lines = _invoke(line_detector_invoke_url, region_crops, api_key=api_key, request_timeout_s=request_timeout_s, retry=retry, nim_client=nim_client)
                    rec_crops: list[str] = []
                    rec_boxes: list[list[float]] = []
                    rec_regions: list[dict[str, Any]] = []
                    for (region_bbox, region_b64, region_shape, region), detector_result in zip(line_jobs, detected_lines):
                        boxes = _detector_boxes(detector_result)
                        for local_box, score in boxes:
                            mapped = _map_box(region_bbox, local_box, region_shape)
                            crop = _b64_crop(image_b64, mapped)
                            if crop:
                                rec_crops.append(crop[0]); rec_boxes.append(mapped); rec_regions.append({"region": region, "score": score})
                    line_count = len(rec_crops)
                    recognized = _invoke(ocr_recognizer_invoke_url, rec_crops, api_key=api_key, request_timeout_s=request_timeout_s, retry=retry, nim_client=nim_client)
                    for bbox, info, result in zip(rec_boxes, rec_regions, recognized):
                        text, score = _recognized_text(result)
                        if not text:
                            continue
                        text_blocks.append({"bbox_xyxy_norm": bbox, "text": text, "score": score, "source": _recognizer_source(result), "line_detector_score": info["score"], "region_label": info["region"].get("label_name"), "ocr_mode": "line_detected"})
                    rec_count = len(recognized)
                except Exception as exc:
                    errors.append(f"text: {type(exc).__name__}: {exc}")

        # Table Structure remains the source of table/cell geometry. Only the
        # cell crops go to PP-OCRv6 recognition; the table image itself is not
        # sent through a second page-level OCR pass.
        structured_table_bboxes: list[list[float]] = []
        for table_bbox, structure, shape in _table_regions(row) if extract_tables and use_table_structure else []:
            structured_table_bboxes.append(table_bbox)
            cells: list[dict[str, Any]] = []
            cell_crops: list[str] = []
            cell_boxes: list[list[float]] = []
            for detection in structure:
                if str(detection.get("label_name") or "") != "cell":
                    continue
                local = detection.get("bbox_xyxy_norm")
                if not isinstance(local, (list, tuple)) or len(local) != 4:
                    continue
                global_box = _map_box(table_bbox, local, shape)
                crop = _b64_crop(image_b64, global_box)
                if crop:
                    cell_crops.append(crop[0]); cell_boxes.append(global_box)
            try:
                for box, result in zip(cell_boxes, _invoke(ocr_recognizer_invoke_url, cell_crops, api_key=api_key, request_timeout_s=request_timeout_s, retry=retry, nim_client=nim_client)):
                    text, score = _recognized_text(result)
                    cells.append({"bbox_xyxy_norm": box, "text": text, "score": score, "source": _recognizer_source(result), "ocr_mode": "table_cell"})
                tables.append({"bbox_xyxy_norm": table_bbox, "text": _markdown_cells(cells), "cells": cells, "source": "nim_table_structure+ppocrv6_recognizer" if box_ocr_mode else "nim_table_structure+ppocrv6_recognizer", "table_structure_status": "joined" if cells else "empty_cells"})
            except Exception as exc:
                errors.append(f"table: {type(exc).__name__}: {exc}")
                tables.append(
                    {
                        "bbox_xyxy_norm": table_bbox,
                        "text": _markdown_cells(cells),
                        "cells": cells,
                        "source": "nim_table_structure+ppocrv6_recognizer",
                        "table_structure_status": "recognizer_error",
                    }
                )

        if extract_tables and use_table_structure:
            # Keep the Page Elements table visible even when the structure NIM
            # failed, returned no regions, or returned no cells.  Previously
            # this silently dropped the table and made the dashboard look as
            # if the OCR stage had never seen it.
            for detection in detections:
                if str(detection.get("label_name") or "") != "table":
                    continue
                bbox = detection.get("bbox_xyxy_norm")
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                bbox = [float(value) for value in bbox]
                if any(
                    all(abs(bbox[index] - existing[index]) <= 1e-3 for index in range(4))
                    for existing in structured_table_bboxes
                ):
                    continue
                tables.append(
                    {
                        "bbox_xyxy_norm": bbox,
                        "text": "",
                        "cells": [],
                        "source": "page_elements_v3_table_without_structure",
                        "table_structure_status": "missing",
                    }
                )

        scan_recall_blocks: list[dict[str, Any]] = []
        scan_quality: dict[str, Any] | None = None
        scan_recall_used_as_output = False
        if box_ocr_mode and wants_text and scan_ocr_fallback:
            scan_recall_blocks, scan_quality = _run_option2_scan_recall(
                image_b64,
                api_key=api_key,
                request_timeout_s=request_timeout_s,
                retry=retry,
                nim_client=nim_client,
                ocr_url=ocr_recognizer_invoke_url,
                tile_size=scan_ocr_tile_size,
                tile_overlap=scan_ocr_tile_overlap,
            )
            # A full-page/tile recognizer response has no reliable internal
            # line geometry in this contract. It is therefore a recall path,
            # not a second canonical text source. Use it only when the Page
            # Elements region pass yielded no text, avoiding giant duplicate
            # page boxes and preserving the original region-first contract.
            if not text_blocks:
                text_blocks = _deduplicate_box_blocks(scan_recall_blocks)
                scan_recall_used_as_output = bool(text_blocks)

        filtered_visual_count = 0
        for detection in detections:
            label = str(detection.get("label_name") or "")
            # Everything outside text/title/table is retained as a visual
            # crop only when Page Elements produced a bounded visual label.
            # Text-dominated infographic/header boxes are discarded by the
            # guard below; the page raster remains available separately.
            if label in _TEXT_LABELS or label == "table":
                continue
            if not _keep_visual_detection(detection, detections):
                if label in _VISUAL_LABELS:
                    filtered_visual_count += 1
                continue
            bbox = detection.get("bbox_xyxy_norm")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            crop = _b64_crop(image_b64, detection.get("crop_bbox_xyxy_norm") or bbox)
            if not crop:
                continue
            item = {
                "bbox_xyxy_norm": [float(v) for v in bbox],
                "model_bbox_xyxy_norm": detection.get("model_bbox_xyxy_norm") or [float(v) for v in bbox],
                "processed_bbox_xyxy_norm": detection.get("processed_bbox_xyxy_norm"),
                "crop_bbox_xyxy_norm": detection.get("crop_bbox_xyxy_norm") or [float(v) for v in bbox],
                "image_b64": crop[0],
                "label_name": label,
                "score": detection.get("score"),
                "source": "page_elements_v3_crop",
                "image_type": "detected_region",
            }
            if label == "chart" and extract_charts: charts.append(item)
            elif label == "infographic" and extract_infographics: infographics.append(item)
            elif label == "image" and extract_images: images.append(item)

        ordered_blocks = _deduplicate_box_blocks(text_blocks)
        all_text.append("\n".join(block["text"] for block in ordered_blocks) if ordered_blocks else existing_text)
        base_images = list(getattr(row, "images", None) or [])
        all_tables.append(tables); all_charts.append(charts); all_infographics.append(infographics); all_images.append(base_images + images); all_blocks.append(ordered_blocks)
        all_meta.append({
            "stage": "page_elements_box_ocr" if box_ocr_mode else "ppocrv6_line_det_rec",
            "models": {
                "page_detector": "NIM Page Elements v3",
                "recognizer": "PP-OCRv6_medium_rec (GPU)" if box_ocr_mode else "PP-OCRv6_medium_rec",
                "line_detector": "PP-OCRv6_medium_det" if line_detection_meta["enabled"] else None,
            },
            "box_count": len(text_regions) if box_ocr_mode else None,
            "line_count": line_count,
            "recognized_count": rec_count,
            "nonempty_recognized_count": sum(1 for block in text_blocks if str(block.get("text") or "").strip()),
            "scan_recall": scan_quality,
            "line_detection": line_detection_meta,
            "scan_recall_used_as_output": scan_recall_used_as_output,
            "visual_crops": len(charts) + len(infographics) + len(images),
            "filtered_visual_crops": filtered_visual_count,
            "errors": errors,
            "timing": {"seconds": time.perf_counter() - started},
            "pipeline": "pipeline-tesseract" if box_ocr_mode else "split_ppocrv6",
            "ocr_strategy": "page_elements_regions_ppocr_line_then_rec" if box_ocr_mode else "page_elements_regions_line_det",
        })

    out = batch_df.copy()
    out["text"] = all_text
    out["table"] = all_tables
    out["chart"] = all_charts
    out["infographic"] = all_infographics
    out["images"] = all_images
    out["ocr_text_blocks"] = all_blocks
    out["_ocr_text_blocks"] = all_blocks
    out["ocr"] = all_meta
    out["ocr_v1_num_detections"] = [len(v) for v in all_blocks]
    out["ocr_v1_counts_by_label"] = [{"text": len(v)} if v else {} for v in all_blocks]
    return out
