"""Remote stamp/seal detection and page-coordinate crop retention."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests

from nemo_retriever.models.nim.nim import NIMClient, invoke_image_inference_batches
from nemo_retriever.common.modality.ocr.shared import _crop_b64_image_by_norm_bbox, _error_payload

logger = logging.getLogger(__name__)

_LOCAL_STAMP_HOSTS = frozenset({"stamp-detector"})
_STAMP_PROBE_TTL_S = 30.0
_stamp_probe_lock = threading.Lock()
_stamp_probe_cache: dict[str, tuple[float, bool]] = {}


class StampDetectorUnavailable(RuntimeError):
    """Raised internally when the optional stamp sidecar is not ready."""


def _local_stamp_health_url(invoke_url: str) -> str | None:
    """Return a health URL only for the Compose-local optional sidecar.

    Hosted or user-provided stamp endpoints are intentionally not probed: they
    may not expose the same health route and should retain their existing
    request/retry behaviour.
    """
    parts = urlsplit(str(invoke_url or ""))
    if parts.hostname not in _LOCAL_STAMP_HOSTS:
        return None
    netloc = parts.hostname
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme or "http", netloc, "/v1/health/ready", "", ""))


def _stamp_endpoint_ready(invoke_url: str, *, timeout_s: float = 2.0) -> bool:
    """Fail fast when the optional local stamp detector is absent/unready."""
    health_url = _local_stamp_health_url(invoke_url)
    if health_url is None:
        return True

    now = time.monotonic()
    with _stamp_probe_lock:
        cached = _stamp_probe_cache.get(health_url)
    if cached is not None and now - cached[0] < _STAMP_PROBE_TTL_S:
        return cached[1]

    ready = False
    try:
        response = requests.get(health_url, timeout=float(timeout_s))
        ready = 200 <= response.status_code < 300
        if not ready:
            logger.warning(
                "Optional stamp detector is not ready at %s (HTTP %s); skipping stamp detection for this batch.",
                health_url,
                response.status_code,
            )
    except requests.RequestException as exc:
        logger.warning(
            "Optional stamp detector is unavailable at %s; skipping stamp detection for this batch: %s",
            health_url,
            exc,
        )

    with _stamp_probe_lock:
        _stamp_probe_cache[health_url] = (now, ready)
    return ready


def _is_scan_row(row: Any) -> bool:
    metadata = row.get("metadata") if hasattr(row, "get") else None
    if not isinstance(metadata, dict):
        return False
    return bool(metadata.get("needs_ocr_for_text") or not metadata.get("has_text", True))


def _parse_box(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if any(item < 0.0 or item > 1.0 for item in box):
        return None
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _iou(left: List[float], right: List[float]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - inter
    return inter / union if union else 0.0


def _response_detections(response: Any, *, min_score: float) -> List[Dict[str, Any]]:
    if not isinstance(response, dict):
        return []
    detections = response.get("detections") or []
    output: List[Dict[str, Any]] = []
    for item in detections:
        if not isinstance(item, dict):
            continue
        bbox = _parse_box(item.get("bbox_xyxy_norm"))
        score = float(item.get("score") or 0.0)
        if bbox is None or score < min_score:
            continue
        candidate = {
            "label_name": "stamp",
            "score": score,
            "prompt": item.get("prompt"),
            "bbox_xyxy_norm": bbox,
        }
        if any(_iou(bbox, previous["bbox_xyxy_norm"]) >= 0.45 for previous in output):
            continue
        output.append(candidate)
    return sorted(output, key=lambda item: item["score"], reverse=True)


def _visual_boxes(row: Any) -> List[List[float]]:
    """Return page-element regions that should remain one visual block.

    A stamp detector can mistake chart labels, logos, or dense infographic
    details for a seal. Those regions already have a dedicated visual crop,
    so a stamp candidate whose centre falls inside one is not emitted as a
    second overlapping block.
    """
    payload = row.get("page_elements_v3") if hasattr(row, "get") else None
    detections = payload.get("detections") if isinstance(payload, dict) else None
    result: List[List[float]] = []
    for item in detections or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label_name") or "").lower()
        if label not in {"chart", "infographic", "image"}:
            continue
        bbox = _parse_box(item.get("bbox_xyxy_norm"))
        if bbox is not None:
            result.append(bbox)
    return result


def _centre_in_box(candidate: List[float], container: List[float]) -> bool:
    cx = (candidate[0] + candidate[2]) / 2.0
    cy = (candidate[1] + candidate[3]) / 2.0
    return container[0] <= cx <= container[2] and container[1] <= cy <= container[3]


def detect_stamps(
    pages_df: Any,
    *,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout_s: float = 120.0,
    inference_batch_size: int = 4,
    min_score: float = 0.50,
    remote_max_retries: int = 5,
    remote_max_429_retries: int = 3,
    remote_max_pool_workers: int = 8,
    nim_client: NIMClient | None = None,
    **_: Any,
) -> Any:
    """Detect stamps only on scanned pages and retain full-page crop images."""
    if not isinstance(pages_df, pd.DataFrame):
        raise NotImplementedError("detect_stamps currently only supports pandas.DataFrame input")
    invoke_url = (invoke_url or "").strip()
    out = pages_df.copy()
    payloads: List[Dict[str, Any]] = [{"detections": [], "regions": [], "timing": None, "error": None} for _ in range(len(out))]
    valid: List[tuple[int, str]] = []
    for index, (_, row) in enumerate(out.iterrows()):
        if not invoke_url or not _is_scan_row(row):
            continue
        page_image = row.get("page_image") or {}
        image_b64 = page_image.get("image_b64") if isinstance(page_image, dict) else None
        if isinstance(image_b64, str) and image_b64:
            valid.append((index, image_b64))

    if valid:
        if not _stamp_endpoint_ready(invoke_url):
            unavailable = StampDetectorUnavailable(
                f"Optional stamp detector is unavailable: {invoke_url}"
            )
            payload = _error_payload(stage="stamp_detection", exc=unavailable)
            for row_index, _ in valid:
                payloads[row_index] = payload | {"detections": [], "regions": []}
            valid = []

    if valid:
        started = time.perf_counter()
        try:
            kwargs = {
                "invoke_url": invoke_url,
                "image_b64_list": [item[1] for item in valid],
                "api_key": api_key,
                "timeout_s": float(request_timeout_s),
                "max_batch_size": int(inference_batch_size),
                "max_retries": int(remote_max_retries),
                "max_429_retries": int(remote_max_429_retries),
            }
            if nim_client is not None:
                responses = nim_client.invoke_image_inference_batches(**kwargs)
            else:
                responses = invoke_image_inference_batches(**kwargs, max_pool_workers=int(remote_max_pool_workers))
            elapsed = time.perf_counter() - started
            for (row_index, image_b64), response in zip(valid, responses):
                detections = _response_detections(response, min_score=float(min_score))
                visual_boxes = _visual_boxes(out.iloc[row_index])
                if visual_boxes:
                    detections = [
                        detection
                        for detection in detections
                        if not any(_centre_in_box(detection["bbox_xyxy_norm"], visual) for visual in visual_boxes)
                    ]
                regions: List[Dict[str, Any]] = []
                for detection in detections:
                    crop_b64, crop_hw = _crop_b64_image_by_norm_bbox(
                        image_b64,
                        bbox_xyxy_norm=detection["bbox_xyxy_norm"],
                    )
                    if not crop_b64:
                        continue
                    regions.append(
                        {
                            **detection,
                            "image_b64": crop_b64,
                            "orig_shape_hw": crop_hw,
                            "image_type": "detected_region",
                            "source": "stamp_detector",
                            "text": "",
                        }
                    )
                payloads[row_index] = {
                    "detections": detections,
                    "regions": regions,
                    "timing": {"seconds": float(elapsed), "inference_requests": 1},
                    "error": None,
                }
        except BaseException as exc:
            payload = _error_payload(stage="stamp_detection", exc=exc)
            for row_index, _ in valid:
                payloads[row_index] = payload | {"detections": [], "regions": []}

    out["stamp_detection"] = payloads
    out["stamp_regions"] = [payload.get("regions") or [] for payload in payloads]
    out["stamp_detection_num_detections"] = [len(payload.get("detections") or []) for payload in payloads]
    return out
