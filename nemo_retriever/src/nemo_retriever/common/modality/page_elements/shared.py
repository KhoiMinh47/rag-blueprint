# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import base64
import io
import time
import traceback

import pandas as pd
from nemo_retriever.models.nim.nim import NIMClient, invoke_page_elements_batches
from nemo_retriever.common.params import RemoteRetryParams
from nemo_retriever.common.modality.page_elements.local import (
    YOLOX_PAGE_V3_CLASS_LABELS,
    YOLOX_PAGE_V3_FINAL_SCORE,
    postprocess_page_elements_v3,
    postprocess_preds_page_element,
)

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

TensorOrArray = Union["torch.Tensor", "np.ndarray"]

_SCAN_TILE_SIZE = 1024
_SCAN_TILE_OVERLAP = 0.15
_SCAN_TILE_MIN_VISUAL_SCORE = 0.50
_ENABLE_SCAN_PAGE_ELEMENT_TILES = False
_VISUAL_LABELS = frozenset({"table", "chart", "image", "infographic", "stamp"})


def _ensure_chw_float_tensor(x: TensorOrArray) -> "torch.Tensor":
    """
    Normalize a single image into a CHW float32 torch.Tensor suitable for batching.

    Accepts either:
    - torch.Tensor in CHW or 1xCHW (or CHW-like) formats
    - np.ndarray in CHW or HWC (RGB) formats (optionally with leading batch dim=1)
    """
    if torch is None or np is None:  # pragma: no cover
        raise ImportError("page element detection requires torch and numpy.")

    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, np.ndarray):
        arr = x
        # Squeeze trivial batch dimension if present.
        if arr.ndim == 4 and int(arr.shape[0]) == 1:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D image array, got shape {getattr(arr, 'shape', None)}")

        # Heuristic: HWC (RGB) -> CHW; otherwise assume already CHW-like.
        if int(arr.shape[-1]) == 3 and int(arr.shape[0]) != 3:
            t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
        else:
            t = torch.from_numpy(np.ascontiguousarray(arr))
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)!r}")

    # Squeeze trivial batch dimension if present.
    if t.ndim == 4 and int(t.shape[0]) == 1:
        t = t[0]
    if t.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(t.shape)}")

    # Keep 0-255 range: resize_pad pads with 114.0 (designed for 0-255),
    # and YoloXWrapper.forward() handles the 0-255 → model-input conversion.
    t = t.to(dtype=torch.float32)

    return t.contiguous()


def _error_payload(*, stage: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "detections": [],
        "error": {
            "stage": str(stage),
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
    }


def _scan_tile_starts(length: int, tile_size: int, overlap: float) -> List[int]:
    """Return monotonic tile origins that cover an axis without gaps."""
    if length <= tile_size:
        return [0]
    stride = max(1, int(tile_size * (1.0 - overlap)))
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def _scan_tiles_from_b64(
    image_b64: str,
    *,
    tile_size: int = _SCAN_TILE_SIZE,
    overlap: float = _SCAN_TILE_OVERLAP,
) -> List[Tuple[List[float], str]]:
    """Create overlapping model-sized crops and page-normalized tile boxes.

    The full page remains the first inference request. Tiles are only an
    additional high-resolution pass for scan pages, where small seals,
    illustrations, and dense text can disappear during the model resize.
    """
    if Image is None:
        return []
    try:
        raw = base64.b64decode(image_b64)
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGB")
            width, height = image.size
            if width <= tile_size and height <= tile_size:
                return []
            x_starts = _scan_tile_starts(width, tile_size, overlap)
            y_starts = _scan_tile_starts(height, tile_size, overlap)
            result: List[Tuple[List[float], str]] = []
            for top in y_starts:
                for left in x_starts:
                    right = min(width, left + tile_size)
                    bottom = min(height, top + tile_size)
                    crop = image.crop((left, top, right, bottom))
                    buf = io.BytesIO()
                    crop.save(buf, format="PNG", compress_level=3)
                    result.append(
                        (
                            [
                                left / width,
                                top / height,
                                right / width,
                                bottom / height,
                            ],
                            base64.b64encode(buf.getvalue()).decode("ascii"),
                        )
                    )
            return result
    except Exception:
        return []


def _map_detection_bbox_to_page(detection: Dict[str, Any], tile_bbox: Sequence[float]) -> Dict[str, Any]:
    """Map one tile-local detector bbox to normalized full-page coordinates."""
    local = detection.get("bbox_xyxy_norm")
    if not isinstance(local, (list, tuple)) or len(local) != 4:
        return detection
    x0, y0, x1, y1 = [float(value) for value in tile_bbox[:4]]
    lx0, ly0, lx1, ly1 = [float(value) for value in local]
    mapped = [
        x0 + lx0 * (x1 - x0),
        y0 + ly0 * (y1 - y0),
        x0 + lx1 * (x1 - x0),
        y0 + ly1 * (y1 - y0),
    ]
    return {**detection, "bbox_xyxy_norm": mapped}


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    lx0, ly0, lx1, ly1 = [float(value) for value in left]
    rx0, ry0, rx1, ry1 = [float(value) for value in right]
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _merge_scan_visual_detections(
    full_page_detections: List[Dict[str, Any]],
    tile_detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Add only conservative visual candidates from tiles.

    Page Elements v3 has page-level expansion rules. Applying those rules to
    overlapping tile outputs can turn a small false positive into a nearly
    full-page box, so full-page output owns normal postprocessing. Tiles only
    contribute small visual candidates that do not overlap a same-class
    full-page detection.
    """
    merged = list(full_page_detections)
    for candidate in tile_detections:
        label = str(candidate.get("label_name") or "")
        bbox = candidate.get("bbox_xyxy_norm")
        if label not in {"table", "chart", "infographic"} or not isinstance(bbox, list):
            continue
        area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
        score = float(candidate.get("score") or 0.0)
        if score < _SCAN_TILE_MIN_VISUAL_SCORE:
            continue
        # Large low-confidence tile boxes are usually page-context artifacts;
        # they are not allowed to override the full-page result.
        if area > 0.35 and score < 0.70:
            continue
        if any(
            str(existing.get("label_name") or "") == label
            and _bbox_iou(existing.get("bbox_xyxy_norm") or [], bbox) >= 0.50
            for existing in merged
        ):
            continue
        merged.append(candidate)
    return merged


def _decode_b64_image_to_chw_tensor(image_b64: str) -> Tuple["torch.Tensor", Tuple[int, int]]:
    if torch is None or Image is None or np is None:  # pragma: no cover
        raise ImportError("page element detection requires torch, pillow, and numpy.")

    raw = base64.b64decode(image_b64)
    with Image.open(io.BytesIO(raw)) as im0:
        im = im0.convert("RGB")
        w, h = im.size
        arr = np.array(im)  # (H,W,3)

    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W) uint8
    t = t.to(dtype=torch.float32) / 255.0
    return t, (int(h), int(w))


def _decode_b64_image_to_np_array(image_b64: str) -> Tuple["np.array", Tuple[int, int]]:
    if torch is None or Image is None or np is None:  # pragma: no cover
        raise ImportError("page element detection requires torch, pillow, and numpy.")

    raw = base64.b64decode(image_b64)
    with Image.open(io.BytesIO(raw)) as im0:
        im = im0.convert("RGB")
        w, h = im.size
        arr = np.array(im)
        # The NIM container receives BGR images (PNG encoded from BGR numpy
        # arrays) and decodes the raw channels as-is, so the model effectively
        # runs on BGR input.  Match that here by reversing the channel order.
        arr = arr[:, :, ::-1].copy()

    return arr, (int(h), int(w))


def _labels_from_model(_model: Any) -> List[str]:
    return [
        "table",
        "chart",
        "title",
        "infographic",
        "text",
        "header_footer",
    ]


def _counts_by_label(detections: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in detections:
        if not isinstance(d, dict):
            continue
        name = d.get("label_name")
        if not isinstance(name, str) or not name.strip():
            name = f"label_{d.get('label')}"
        k = str(name)
        out[k] = int(out.get(k, 0) + 1)
    return out


def _postprocess_to_per_image_detections(
    *,
    boxes: Any,
    labels: Any,
    scores: Any,
    batch_size: int,
    label_names: List[str],
) -> List[List[Dict[str, Any]]]:
    """
    Convert model postprocess outputs into a list of per-image detection dicts.

    Expected detection format matches the "stage2 page_elements_v3 json" used by `nemo_retriever.utils.image.render`.
    """
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for page element detection postprocess.")

    # Normalize to per-image tensors.
    def _as_list(x: Any) -> List[Any]:
        if isinstance(x, list):
            return x
        return [x]

    # If tensors include a batch dimension, split them.
    if isinstance(boxes, torch.Tensor) and boxes.ndim == 3:
        boxes_list = [boxes[i] for i in range(int(boxes.shape[0]))]
    else:
        boxes_list = _as_list(boxes)

    if isinstance(labels, torch.Tensor) and labels.ndim == 2:
        labels_list = [labels[i] for i in range(int(labels.shape[0]))]
    else:
        labels_list = _as_list(labels)

    if isinstance(scores, torch.Tensor) and scores.ndim == 2:
        scores_list = [scores[i] for i in range(int(scores.shape[0]))]
    else:
        scores_list = _as_list(scores)

    n = min(len(boxes_list), len(labels_list), len(scores_list), int(batch_size))
    out: List[List[Dict[str, Any]]] = []
    for i in range(n):
        bi = boxes_list[i]
        li = labels_list[i]
        si = scores_list[i]

        if not isinstance(bi, torch.Tensor) or not isinstance(li, torch.Tensor) or not isinstance(si, torch.Tensor):
            out.append([])
            continue

        # Move to CPU for safe conversion.
        bi = bi.detach().cpu()
        li = li.detach().cpu()
        si = si.detach().cpu()

        # Common shapes:
        # - boxes: (N,4)
        # - labels: (N,)
        # - scores: (N,)
        if bi.ndim != 2 or bi.shape[-1] != 4:
            out.append([])
            continue

        n_det = int(bi.shape[0])
        dets: List[Dict[str, Any]] = []
        for j in range(n_det):
            try:
                x1, y1, x2, y2 = [float(x) for x in bi[j].tolist()]
            except Exception:
                continue

            label_i: Optional[int]
            try:
                label_i = int(li[j].item())
            except Exception:
                label_i = None

            score_f: Optional[float]
            try:
                score_f = float(si[j].item())
            except Exception:
                score_f = None

            label_name = None
            if label_i is not None and 0 <= label_i < len(label_names):
                label_name = label_names[label_i]
            if not label_name:
                label_name = f"label_{label_i}" if label_i is not None else "unknown"

            dets.append(
                {
                    "bbox_xyxy_norm": [x1, y1, x2, y2],
                    "label": label_i,
                    "label_name": str(label_name),
                    "score": score_f,
                }
            )
        out.append(dets)

    # If model returned fewer splits than requested, pad.
    while len(out) < int(batch_size):
        out.append([])
    return out[: int(batch_size)]


# -- Label mapping between retriever ("text") and API ("paragraph") --
_RETRIEVER_LABEL_NAMES = ["table", "chart", "title", "infographic", "text", "header_footer"]
_RETRIEVER_TO_API = {"text": "paragraph"}
_API_TO_RETRIEVER = {"paragraph": "text"}


def _detections_to_annotation_dict(
    dets: List[Dict[str, Any]],
) -> Dict[str, List[List[float]]]:
    """Convert a list of detection dicts into the annotation_dict format expected by
    ``postprocess_page_elements_v3``.

    Each detection dict has keys ``bbox_xyxy_norm``, ``label_name``, ``score``.
    The annotation_dict maps label names (using API naming, i.e. "paragraph") to
    ``[[x0, y0, x1, y1, confidence], ...]``.
    """
    ann: Dict[str, List[List[float]]] = {}
    for d in dets:
        name = _RETRIEVER_TO_API.get(d["label_name"], d["label_name"])
        bbox = list(d["bbox_xyxy_norm"])  # [x0, y0, x1, y1]
        bbox.append(float(d["score"]) if d["score"] is not None else 0.0)
        ann.setdefault(name, []).append(bbox)
    return ann


def _annotation_dict_to_detections(
    ann_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert an annotation_dict back into a list of detection dicts.

    Maps API label names back to retriever names (e.g. "paragraph" -> "text")
    and assigns integer label IDs from the retriever label order.
    """
    dets: List[Dict[str, Any]] = []
    for api_name, entries in ann_dict.items():
        retriever_name = _API_TO_RETRIEVER.get(api_name, api_name)
        try:
            label_id = _RETRIEVER_LABEL_NAMES.index(retriever_name)
        except ValueError:
            label_id = None
        for entry in entries:
            # entry is [x0, y0, x1, y1, confidence]
            dets.append(
                {
                    "bbox_xyxy_norm": list(entry[:4]),
                    "label": label_id,
                    "label_name": retriever_name,
                    "score": float(entry[4]) if len(entry) > 4 else 0.0,
                }
            )
    return dets


def _bounding_boxes_to_detections(
    bb_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert a bounding_boxes dict (NIM API format) to detection dicts.

    Input format: {"label": [{"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ..., "confidence": ...}, ...]}
    """
    dets: List[Dict[str, Any]] = []
    for api_name, entries in bb_dict.items():
        retriever_name = _API_TO_RETRIEVER.get(api_name, api_name)
        try:
            label_id = _RETRIEVER_LABEL_NAMES.index(retriever_name)
        except ValueError:
            label_id = None
        for entry in entries:
            dets.append(
                {
                    "bbox_xyxy_norm": [
                        float(entry["x_min"]),
                        float(entry["y_min"]),
                        float(entry["x_max"]),
                        float(entry["y_max"]),
                    ],
                    "label": label_id,
                    "label_name": retriever_name,
                    "score": float(entry.get("confidence", 0.0)),
                }
            )
    return dets


def _apply_final_score_filter(
    dets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filter detections by per-class final score thresholds (YOLOX_PAGE_V3_FINAL_SCORE).

    This should be applied **after** WBF post-processing to match the NIM pipeline ordering.
    Maps retriever label "text" to API label "paragraph" for threshold lookup.
    """
    if not YOLOX_PAGE_V3_FINAL_SCORE or not dets:
        return dets
    filtered: List[Dict[str, Any]] = []
    for d in dets:
        api_name = _RETRIEVER_TO_API.get(d["label_name"], d["label_name"])
        threshold = YOLOX_PAGE_V3_FINAL_SCORE.get(api_name, 0.0)
        if d.get("score") is not None and d["score"] >= threshold:
            filtered.append(d)
    return filtered


def _apply_page_elements_v3_postprocess(
    dets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply ``postprocess_page_elements_v3`` (box fusion, title matching,
    expansion, overlap removal) to a single image's detection list.

    Returns the original detections unchanged if the API function is unavailable.
    """
    if not dets:
        return dets

    original_dets = [dict(detection) for detection in dets if isinstance(detection, dict)]

    def _small_crop_padding(bbox: Sequence[float], padding: float = 0.01) -> List[float]:
        values = [float(value) for value in bbox[:4]]
        return [
            max(0.0, values[0] - padding),
            max(0.0, values[1] - padding),
            min(1.0, values[2] + padding),
            min(1.0, values[3] + padding),
        ]

    def _match_original(processed: Dict[str, Any], used: set[int]) -> Optional[tuple[int, Dict[str, Any]]]:
        label = str(processed.get("label_name") or "")
        candidates = [
            (index, original)
            for index, original in enumerate(original_dets)
            if index not in used and str(original.get("label_name") or "") == label
        ]
        if not candidates:
            return None
        processed_score = processed.get("score")

        def rank(item: tuple[int, Dict[str, Any]]) -> tuple[float, float]:
            _index, original = item
            try:
                score_delta = abs(float(original.get("score")) - float(processed_score))
            except (TypeError, ValueError):
                score_delta = 1.0
            return (score_delta, -_bbox_iou(processed.get("bbox_xyxy_norm") or [], original.get("bbox_xyxy_norm") or []))

        return min(candidates, key=rank)

    def _preserve_model_geometry(processed_dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep model geometry separate from layout postprocessing geometry.

        Page Elements v3 intentionally expands chart/table boxes for layout
        matching. That geometry is useful for layout reasoning, but it is too
        broad for a visual crop or native-text ownership test. Visual blocks
        therefore expose the raw model box as ``bbox_xyxy_norm`` and keep the
        expanded box in ``processed_bbox_xyxy_norm``.
        """
        used: set[int] = set()
        for processed in processed_dets:
            if not isinstance(processed, dict):
                continue
            matched = _match_original(processed, used)
            if matched is None:
                continue
            original_index, original = matched
            used.add(original_index)
            model_bbox = original.get("bbox_xyxy_norm")
            processed_bbox = processed.get("bbox_xyxy_norm")
            if not isinstance(model_bbox, (list, tuple)) or len(model_bbox) != 4:
                continue
            model_bbox = [float(value) for value in model_bbox]
            processed["model_bbox_xyxy_norm"] = model_bbox
            if isinstance(processed_bbox, (list, tuple)) and len(processed_bbox) == 4:
                processed["processed_bbox_xyxy_norm"] = [float(value) for value in processed_bbox]
            if str(processed.get("label_name") or "") in _VISUAL_LABELS:
                processed["bbox_xyxy_norm"] = model_bbox
                processed["crop_bbox_xyxy_norm"] = _small_crop_padding(model_bbox)
        return processed_dets

    if postprocess_page_elements_v3 is None:
        return _preserve_model_geometry(dets)
    try:
        ann_dict = _detections_to_annotation_dict(dets)
        labels = YOLOX_PAGE_V3_CLASS_LABELS if YOLOX_PAGE_V3_CLASS_LABELS is not None else list(ann_dict.keys())
        result = postprocess_page_elements_v3(ann_dict, labels=labels)
        return _preserve_model_geometry(_annotation_dict_to_detections(result))
    except Exception:
        return _preserve_model_geometry(dets)


def _remote_response_to_detections(
    *,
    response_json: Dict[str, Any],
    label_names: List[str],
    thresholds_per_class: Sequence[float],
    apply_v3_postprocess: bool = True,
) -> List[Dict[str, Any]]:
    # Try direct model-pred style payload first (or common wrappers around it).
    candidates: List[Any] = [response_json]
    data_list = response_json.get("data")
    if isinstance(data_list, list) and data_list:
        candidates.append(data_list[0])
    output_list = response_json.get("output")
    if isinstance(output_list, list) and output_list:
        candidates.append(output_list[0])
    pred_list = response_json.get("predictions")
    if isinstance(pred_list, list) and pred_list:
        candidates.append(pred_list[0])

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        try:
            boxes, labels, scores = postprocess_preds_page_element(cand, list(thresholds_per_class), label_names)
            dets = _postprocess_to_per_image_detections(
                boxes=[boxes],
                labels=[labels],
                scores=[scores],
                batch_size=1,
                label_names=label_names,
            )[0]
            return _apply_page_elements_v3_postprocess(dets) if apply_v3_postprocess else dets
        except Exception:
            pass

    # NIM bounding_boxes format:
    # {"index": 0, "bounding_boxes": {"title": [{"x_min": ..., "y_min": ..., ...}]}}
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        bb = cand.get("bounding_boxes")
        if isinstance(bb, dict):
            try:
                dets = _bounding_boxes_to_detections(bb)
                return _apply_page_elements_v3_postprocess(dets) if apply_v3_postprocess else dets
            except Exception:
                pass

    # Fall back to API-style annotation dict:
    # {"table": [[x0,y0,x1,y1,conf], ...], "paragraph": [...]}
    for cand in candidates:
        if not isinstance(cand, dict) or not cand:
            continue
        if all(isinstance(v, list) for v in cand.values()):
            try:
                dets = _annotation_dict_to_detections(cand)  # type: ignore[arg-type]
                return _apply_page_elements_v3_postprocess(dets) if apply_v3_postprocess else dets
            except Exception:
                pass

    raise RuntimeError(f"Unsupported remote response format (keys={list(response_json.keys())!r})")


def detect_page_elements_v3(
    pages_df: Any,
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout_s: float = 120.0,
    inference_batch_size: int = 8,
    output_column: str = "page_elements_v3",
    num_detections_column: str = "page_elements_v3_num_detections",
    counts_by_label_column: str = "page_elements_v3_counts_by_label",
    remote_retry: RemoteRetryParams | None = None,
    nim_client: NIMClient | None = None,
    **kwargs: Any,
) -> Any:
    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )
    """
    Run Nemotron Page Elements v3 on a pandas batch.

    Input:
      - `pages_df`: pandas.DataFrame (typical Ray Data `batch_format="pandas"`)
        Must contain an image base64 source either in `image_b64` or one of
        `images`/`tables`/`charts`/`infographics` (each as list[{"image_b64": ...}]).

    Output:
      - returns a pandas.DataFrame with original columns preserved, plus:
        - `output_column`: dict payload {"detections": [...], "timing": {...}, "error": {...?}}
        - `num_detections_column`: int
        - `counts_by_label_column`: dict[str,int]

    Notes:
      - This function internally batches model invocations in chunks of `inference_batch_size`
        to enforce batch=8 even if Ray provides larger `map_batches` frames.
    """
    if not isinstance(pages_df, pd.DataFrame):
        raise NotImplementedError("detect_page_elements_v3 currently only supports pandas.DataFrame input.")

    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0")

    # Working snippet for single image inference and debugging
    # breakpoint()
    # first_page = pages_df.iloc[0]
    # b64 = first_page.get("page_image")["image_b64"]

    # t, orig_shape = _decode_b64_image_to_np_array(b64)

    # # Inference
    # with torch.inference_mode():
    #     x = model.preprocess(t)
    #     preds = model(x, orig_shape)[0]

    # print(preds)
    # breakpoint()

    invoke_url = (invoke_url or kwargs.get("page_elements_invoke_url") or "").strip()
    use_remote = bool(invoke_url)

    if not use_remote and model is None:
        raise ValueError("A local `model` is required when `invoke_url` is not provided.")

    # Prepare per-row decode artifacts (local mode), raw base64 (remote mode),
    # and placeholders for missing/errored rows.
    row_tensors: List[Optional[TensorOrArray]] = []
    row_shapes: List[Optional[Tuple[int, int]]] = []
    row_b64: List[Optional[str]] = []
    row_scan_b64: List[Optional[str]] = []
    row_payloads: List[Dict[str, Any]] = []

    label_names = _labels_from_model(model) if model is not None else list(_RETRIEVER_LABEL_NAMES)
    if model is not None and hasattr(model, "thresholds_per_class"):
        thresholds_per_class = getattr(model, "thresholds_per_class")
    else:
        # Use the same per-class thresholds as the yolox pipeline.
        # label_names uses "text" where yolox uses "paragraph"; _RETRIEVER_TO_API maps between them.
        thresholds_per_class = [
            YOLOX_PAGE_V3_FINAL_SCORE.get(_RETRIEVER_TO_API.get(name, name), 0.0) for name in label_names
        ]

    for _, row in pages_df.iterrows():
        try:
            # Keep Page Elements on the canonical page raster.  The separate
            # fit-to-model raster is useful as a fallback/transport artifact,
            # but using it as the primary detector input makes scanned pages
            # collapse into giant infographic boxes and removes paragraph
            # regions before OCR gets a chance to run.
            detector_image = row.get("page_image") or row.get("page_elements_image")
            b64 = detector_image["image_b64"]
            if not b64:
                raise ValueError("No usable image_b64 found in row.")
            row_b64.append(b64)
            page_image = row.get("page_image") or detector_image
            row_scan_b64.append(page_image.get("image_b64") if isinstance(page_image, dict) else b64)
            if use_remote:
                row_tensors.append(None)
                row_shapes.append(None)
            else:
                t, orig_shape = _decode_b64_image_to_np_array(b64)
                row_tensors.append(t)
                row_shapes.append(orig_shape)
            row_payloads.append({"detections": []})
        except BaseException as e:
            row_tensors.append(None)
            row_shapes.append(None)
            row_b64.append(None)
            row_scan_b64.append(None)
            row_payloads.append(_error_payload(stage="decode_image", exc=e))

    # Run inference over only valid rows, but write results back in original order.
    if use_remote:
        valid_indices = [i for i, b64 in enumerate(row_b64) if b64]
    else:
        valid_indices = [i for i, t in enumerate(row_tensors) if t is not None and row_shapes[i] is not None]

    if (not use_remote) and valid_indices and torch is None:  # pragma: no cover
        raise ImportError("torch is required for page element detection.")

    if use_remote and valid_indices:
        remote_requests: List[Tuple[int, List[float], str]] = []
        for row_i in valid_indices:
            b64 = row_b64[row_i]
            if b64:
                # Keep a full-page request for large regions. On scan pages,
                # add overlapping high-resolution tiles for small regions.
                remote_requests.append((row_i, [0.0, 0.0, 1.0, 1.0], b64))
                metadata = pages_df.iloc[row_i].get("metadata") or {}
                is_scan = isinstance(metadata, dict) and bool(
                    metadata.get("needs_ocr_for_text") or not metadata.get("has_text", True)
                )
                if is_scan and _ENABLE_SCAN_PAGE_ELEMENT_TILES:
                    scan_b64 = row_scan_b64[row_i] or b64
                    remote_requests.extend(
                        (row_i, tile_bbox, tile_b64)
                        for tile_bbox, tile_b64 in _scan_tiles_from_b64(scan_b64)
                    )

        t0 = time.perf_counter()
        try:
            valid_b64 = [b64 for _row_i, _tile_bbox, b64 in remote_requests]
            _invoke_kw = dict(
                invoke_url=invoke_url,
                image_b64_list=valid_b64,
                api_key=api_key,
                timeout_s=float(request_timeout_s),
                max_batch_size=int(inference_batch_size),
                max_retries=int(retry.remote_max_retries),
                max_429_retries=int(retry.remote_max_429_retries),
            )
            if nim_client is not None:
                response_items = nim_client.invoke_page_elements_batches(**_invoke_kw)
            else:
                response_items = invoke_page_elements_batches(
                    **_invoke_kw,
                    max_pool_workers=int(retry.remote_max_pool_workers),
                )
            elapsed = time.perf_counter() - t0

            if len(response_items) != len(remote_requests):
                raise RuntimeError(
                    "Remote response count mismatch: "
                    f"expected {len(remote_requests)}, got {len(response_items)}"
                )

            full_page_detections: Dict[int, List[Dict[str, Any]]] = {row_i: [] for row_i in valid_indices}
            tile_detections: Dict[int, List[Dict[str, Any]]] = {row_i: [] for row_i in valid_indices}
            request_counts: Dict[int, int] = {row_i: 0 for row_i in valid_indices}
            for local_i, (row_i, tile_bbox, _tile_b64) in enumerate(remote_requests):
                metadata = pages_df.iloc[row_i].get("metadata") or {}
                is_scan = isinstance(metadata, dict) and bool(
                    metadata.get("needs_ocr_for_text") or not metadata.get("has_text", True)
                )
                tile_dets = _remote_response_to_detections(
                    response_json=response_items[local_i],
                    label_names=label_names,
                    thresholds_per_class=thresholds_per_class,
                    # NIM already returns normalized page boxes. The generic
                    # v3 expansion/matching pass is useful for native pages,
                    # but on scans it can merge a distant title into a table
                    # and materially enlarge the crop used by OCR.
                    # Apply the v3 expansion/matching pass once, after all
                    # full-page responses have been collected. Applying it
                    # here and again below expands visual boxes twice.
                    apply_v3_postprocess=False,
                )
                mapped = [_map_detection_bbox_to_page(detection, tile_bbox) for detection in tile_dets]
                if tile_bbox == [0.0, 0.0, 1.0, 1.0]:
                    full_page_detections[row_i].extend(mapped)
                else:
                    tile_detections[row_i].extend(mapped)
                request_counts[row_i] += 1

            for row_i in valid_indices:
                metadata = pages_df.iloc[row_i].get("metadata") or {}
                is_scan = isinstance(metadata, dict) and bool(
                    metadata.get("needs_ocr_for_text") or not metadata.get("has_text", True)
                )
                full_dets = full_page_detections[row_i]
                if not is_scan:
                    full_dets = _apply_page_elements_v3_postprocess(full_dets)
                full_dets = _apply_final_score_filter(full_dets)
                tile_dets = _apply_final_score_filter(tile_detections[row_i])
                dets = _merge_scan_visual_detections(full_dets, tile_dets)
                row_payloads[row_i] = {
                    "detections": dets,
                    "timing": {
                        "seconds": float(elapsed),
                        "inference_requests": int(request_counts.get(row_i, 0)),
                        "tiled": bool(request_counts.get(row_i, 0) > 1),
                    },
                    "error": None,
                }
        except BaseException as e:
            elapsed = time.perf_counter() - t0
            print(f"Warning: page_elements remote inference failed: {type(e).__name__}: {e}")
            for row_i in valid_indices:
                row_payloads[row_i] = _error_payload(stage="remote_inference", exc=e) | {
                    "timing": {"seconds": float(elapsed)}
                }

    for chunk_start in range(0, len(valid_indices), int(inference_batch_size)):
        chunk_idx = valid_indices[chunk_start : chunk_start + int(inference_batch_size)]
        if not chunk_idx:
            continue

        if use_remote:
            continue

        # Preprocess each image to a fixed shape so we can stack.
        pre_list: List[TensorOrArray] = []
        orig_shapes: List[Tuple[int, int]] = []
        for i in chunk_idx:
            t = row_tensors[i]
            sh = row_shapes[i]
            if t is None or sh is None:
                continue
            orig_shapes.append(sh)
            try:
                # `preprocess` may accept/return torch.Tensor or np.ndarray.
                pre = model.preprocess(t)  # type: ignore[arg-type]

                # Normalize to a single-image CHW-like item (torch or numpy); we'll convert to torch at stack time.
                if isinstance(pre, torch.Tensor):
                    if pre.ndim == 4 and int(pre.shape[0]) == 1:
                        pre_list.append(pre[0])
                    elif pre.ndim == 3:
                        pre_list.append(pre)
                    else:
                        pre_list.append(pre)
                elif isinstance(pre, np.ndarray):
                    if pre.ndim == 4 and int(pre.shape[0]) == 1:
                        pre_list.append(pre[0])
                    else:
                        pre_list.append(pre)
                else:
                    pre_list.append(t)
            except Exception:
                pre_list.append(t)

        if not pre_list:
            continue

        batch = torch.stack([_ensure_chw_float_tensor(x) for x in pre_list], dim=0)

        t0 = time.perf_counter()
        try:
            # Best-effort: pass list of shapes for batching; fall back to per-image if unsupported.
            with torch.inference_mode():
                with torch.autocast(device_type="cuda"):
                    preds = model(batch, orig_shapes) if len(pre_list) > 1 else model(batch, orig_shapes[0])
            # Some local wrappers return only the first prediction dict even for batched inputs.
            # Detect that and force per-image invocation so every row gets its own detections.
            if len(pre_list) > 1:
                if isinstance(preds, dict):
                    raise RuntimeError("Model returned a single pred dict for batched input.")
                if isinstance(preds, list) and len(preds) != len(pre_list):
                    raise RuntimeError(
                        f"Model returned {len(preds)} preds for batch size {len(pre_list)}; falling back to per-image."
                    )
        except Exception as ex:
            print(f"Error invoking model: {ex}")
            preds_list: List[Any] = []
            for j in range(int(batch.shape[0])):
                preds_list.append(model(batch[j : j + 1], orig_shapes[j]))
            preds = preds_list
        elapsed = time.perf_counter() - t0

        # Normalize preds into a list of per-image prediction dicts.
        if isinstance(preds, dict):
            preds_list2 = [preds]
        elif isinstance(preds, list):
            preds_list2 = preds
        else:
            preds_list2 = [preds]  # type: ignore[list-item]

        try:
            # Preferred: allow model wrapper to handle batched postprocess.
            if hasattr(model, "postprocess"):
                boxes, labels, scores = model.postprocess(preds_list2)  # type: ignore[attr-defined]
            else:
                # Fallback: run upstream util per-image.
                # `postprocess_preds_page_element` expects a single pred dict and returns numpy arrays.
                boxes_list: List["torch.Tensor"] = []
                labels_list: List["torch.Tensor"] = []
                scores_list: List["torch.Tensor"] = []
                for p in preds_list2:
                    if not isinstance(p, dict):
                        boxes_list.append(torch.empty((0, 4), dtype=torch.float32))
                        labels_list.append(torch.empty((0,), dtype=torch.int64))
                        scores_list.append(torch.empty((0,), dtype=torch.float32))
                        continue
                    b_np, l_np, s_np = postprocess_preds_page_element(
                        p,
                        thresholds_per_class,
                        label_names,
                    )
                    boxes_list.append(torch.as_tensor(b_np, dtype=torch.float32))
                    labels_list.append(torch.as_tensor(l_np, dtype=torch.int64))
                    scores_list.append(torch.as_tensor(s_np, dtype=torch.float32))
                boxes, labels, scores = boxes_list, labels_list, scores_list

            per_image_dets = _postprocess_to_per_image_detections(
                boxes=boxes,
                labels=labels,
                scores=scores,
                batch_size=len(pre_list),
                label_names=label_names,
            )
            # Apply v3 postprocessing (box fusion via WBF at iou=0.01, title matching, expansion, overlap removal)
            per_image_dets = [_apply_page_elements_v3_postprocess(dets) for dets in per_image_dets]
            # Apply per-class final score filtering AFTER WBF (matches NIM pipeline ordering)
            per_image_dets = [_apply_final_score_filter(dets) for dets in per_image_dets]
            for local_i, row_i in enumerate(chunk_idx):
                dets = per_image_dets[local_i] if local_i < len(per_image_dets) else []
                row_payloads[row_i] = {
                    "detections": dets,
                    "timing": {"seconds": float(elapsed)},
                    "error": None,
                }
        except BaseException as e:
            # If postprocess fails, attach an error but keep job alive.
            for row_i in chunk_idx:
                row_payloads[row_i] = _error_payload(stage="postprocess", exc=e) | {
                    "timing": {"seconds": float(elapsed)}
                }

    out = pages_df.copy()
    out[output_column] = row_payloads
    out[num_detections_column] = [
        int(len(p.get("detections") or [])) if isinstance(p, dict) else 0 for p in row_payloads
    ]
    out[counts_by_label_column] = [
        _counts_by_label(p.get("detections") or []) if isinstance(p, dict) else {} for p in row_payloads
    ]
    return out
