# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dashboard UI router for gateway and standalone service roles.

Serves the SPA shell and provides REST/SSE API endpoints consumed by the
React frontend for the Overview, Job Tracker, and VDB Explorer views.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from pathlib import Path
from typing import Any
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard"])

_STATIC_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "static"


# ── Request models ───────────────────────────────────────────────────


class VdbQueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=1000)


# ── SPA shell ────────────────────────────────────────────────────────


@router.get("", include_in_schema=False)
@router.get("/", include_in_schema=False)
async def index():
    index_path = _STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(500, f"Dashboard UI not found at {index_path}")
    # The dashboard shell contains the asset versioning loader. Do not let a
    # browser keep an older shell that still points the View action at the
    # raw document endpoint.
    return FileResponse(
        str(index_path),
        media_type="text/html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


# ── Overview API ─────────────────────────────────────────────────────


async def _fetch_pool_stats(client: httpx.AsyncClient, base_url: str) -> dict:
    """Best-effort fetch of ``GET /v1/admin/pool_stats`` from a backend.

    Returns ``{}`` on any error so the overview never fails to render
    just because one worker pod is briefly unhealthy.
    """
    try:
        resp = await client.get(f"{base_url}/v1/admin/pool_stats", timeout=2.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        logger.debug("pool_stats fetch failed for %s: %s", base_url, exc)
    return {}


@router.get("/api/overview")
async def overview(request: Request) -> JSONResponse:
    """Aggregate cluster status for the overview panel.

    On gateway pods, this fans out to each worker Service to collect
    live pool stats (queue depth, queue ratio, processed counts) so
    the dashboard can surface scaling pressure without forcing the
    operator to open Grafana. On standalone pods the local pool is
    read in-process via :func:`get_pipeline_pool`.
    """
    config = request.app.state.config

    backends = {}
    pool_stats: dict[str, dict] = {}
    try:
        from nemo_retriever.service.services.proxy import get_proxy
        from nemo_retriever.service.services.pipeline_pool import (
            PoolType,
            get_pipeline_pool,
        )

        proxy = get_proxy()
        if proxy is not None:
            backends["realtime"] = await proxy.check_backend(PoolType.REALTIME)
            backends["batch"] = await proxy.check_backend(PoolType.BATCH)
            # H6: fan out to each backend for live queue depth. The
            # gateway has no local pool, so this is the only way the
            # overview page can show "realtime queue 50% full" without
            # going through Prometheus.
            gateway_cfg = getattr(config, "gateway", None)
            if gateway_cfg is not None:
                async with httpx.AsyncClient() as client:
                    rt_task = _fetch_pool_stats(client, gateway_cfg.realtime_url)
                    bt_task = _fetch_pool_stats(client, gateway_cfg.batch_url)
                    rt_stats, bt_stats = await asyncio.gather(rt_task, bt_task)
                # Each worker's response carries its own pools dict; the
                # realtime pod returns {"realtime": {...}} and batch
                # returns {"batch": {...}}. Merge for the consumer.
                for stats in (rt_stats, bt_stats):
                    for pool_name, pool_data in (stats.get("pools") or {}).items():
                        pool_stats[pool_name] = pool_data
        else:
            # Standalone (or worker) pod — pull stats from the local
            # singleton directly to avoid an HTTP round trip to ourselves.
            local_pool = get_pipeline_pool()
            if local_pool is not None:
                for pt in (PoolType.REALTIME, PoolType.BATCH):
                    p = local_pool.pool_for(pt)
                    if p is None:
                        continue
                    depth = p.queue_depth
                    max_qs = max(1, p.max_queue_size)
                    pool_stats[pt.value] = {
                        "queue_depth": depth,
                        "queue_depth_ratio": round(depth / max_qs, 4),
                        "max_queue_size": p.max_queue_size,
                        "num_workers": p.num_workers,
                        "processed": p.processed,
                        "is_running": p.is_running,
                    }
    except Exception as exc:
        logger.debug("Could not check backends / pool stats: %s", exc)

    vdb_status = None
    vdb_url = getattr(config, "vectordb", None)
    if vdb_url and getattr(vdb_url, "enabled", False):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{vdb_url.vectordb_url}/v1/health")
                if resp.status_code == 200:
                    vdb_status = resp.json()
        except Exception as exc:
            logger.debug("VDB health check failed: %s", exc)

    from nemo_retriever.service.services.job_tracker import get_job_tracker

    tracker = get_job_tracker()
    job_summary = tracker.summary() if tracker else {}

    pool_cfg = getattr(config, "pipeline", None)
    worker_config = {}
    if pool_cfg:
        worker_config = {
            "realtime_workers": pool_cfg.realtime_workers,
            "realtime_queue_size": pool_cfg.realtime_queue_size,
            "batch_workers": pool_cfg.batch_workers,
            "batch_queue_size": pool_cfg.batch_queue_size,
        }

    gateway_cfg = getattr(config, "gateway", None)
    gateway_info = {}
    if gateway_cfg:
        gateway_info = {
            "realtime_url": gateway_cfg.realtime_url,
            "batch_url": gateway_cfg.batch_url,
        }

    return JSONResponse(
        {
            "mode": config.mode,
            "backends": backends,
            "pool_stats": pool_stats,
            "vectordb": vdb_status,
            "job_summary": job_summary,
            "worker_config": worker_config,
            "gateway": gateway_info,
        }
    )


# ── Jobs SSE stream ─────────────────────────────────────────────────


@router.get("/api/jobs")
async def jobs_sse(request: Request) -> StreamingResponse:
    """SSE stream of job-tracker events with periodic summary heartbeats.

    Subscribes to the global event bus (no ``job_id`` filter) so the
    dashboard sees both per-document events and the J5 job lifecycle
    events for every job. The initial ``snapshot`` payload bundles both
    layers (``documents`` for back-compat with the legacy doc-grid view
    and ``jobs`` for the new job-aggregate view) so the SPA can render
    immediately without an extra REST hop.
    """
    from nemo_retriever.service.services.event_bus import get_event_bus
    from nemo_retriever.service.services.job_tracker import get_job_tracker

    bus = get_event_bus()
    tracker = get_job_tracker()

    if bus is None:
        raise HTTPException(503, "Event bus not available")

    sub_id, queue = bus.subscribe()

    async def event_generator():
        try:
            if tracker:
                snapshot = {
                    "type": "snapshot",
                    "summary": tracker.summary(),
                    "jobs": [_serialize_job(j) for j in tracker.all_jobs()],
                    "documents": [rec.model_dump() for rec in tracker.all_documents()],
                }
                yield f"event: snapshot\ndata: {json.dumps(snapshot)}\n\n"

            last_heartbeat = asyncio.get_event_loop().time()

            while True:
                if await request.is_disconnected():
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                    # Job lifecycle events vs per-doc events are
                    # distinguished by the ``type`` field set by the
                    # tracker (``job_created`` etc. vs status strings).
                    evt_type = event.get("type", "")
                    sse_event = "job_lifecycle" if evt_type.startswith("job_") else "job_update"
                    yield f"event: {sse_event}\ndata: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    pass

                now = asyncio.get_event_loop().time()
                if now - last_heartbeat >= 5.0:
                    heartbeat = {"type": "heartbeat"}
                    if tracker:
                        heartbeat["summary"] = tracker.summary()
                    yield f"event: heartbeat\ndata: {json.dumps(heartbeat)}\n\n"
                    last_heartbeat = now

        finally:
            bus.unsubscribe(sub_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Jobs snapshot (REST fallback) ────────────────────────────────────


def _serialize_job(agg) -> dict:
    """Project a :class:`JobAggregate` to the wire shape the UI expects.

    Kept compact (no document records) so list views can paginate
    hundreds of jobs without bloating the payload.
    """
    return {
        "job_id": agg.job_id,
        "status": agg.status.value,
        "expected_documents": agg.expected_documents,
        "counts": dict(agg.counts),
        "created_at": agg.created_at,
        "started_at": agg.started_at,
        "finalized_at": agg.finalized_at,
        "elapsed_s": agg.elapsed_s,
        "label": agg.label,
        "trace_id": agg.trace_id,
        "document_ids": list(agg.document_ids),
    }


@router.get("/api/jobs/snapshot")
async def jobs_snapshot(request: Request) -> JSONResponse:
    """REST fallback for the SSE stream.

    Returns both ``jobs`` (the J2+ aggregate view) and ``documents``
    (per-doc rows for the legacy table). The SPA prefers ``jobs`` and
    falls back to ``documents`` for older builds.
    """
    from nemo_retriever.service.services.job_tracker import get_job_tracker

    tracker = get_job_tracker()
    if tracker is None:
        return JSONResponse({"summary": {}, "jobs": [], "documents": []})

    return JSONResponse(
        {
            "summary": tracker.summary(),
            "jobs": [_serialize_job(j) for j in tracker.all_jobs()],
            "documents": [rec.model_dump() for rec in tracker.all_documents()],
        }
    )


# ── Jobs list / detail (J8 — paginated REST API for the UI) ──────────


@router.get("/api/jobs/list")
async def jobs_list(
    request: Request,
    status: str | None = None,
    offset: int = 0,
    limit: int = 50,
    sort: str = "created_desc",
) -> JSONResponse:
    """Paginated list of job aggregates, newest first by default.

    Parameters
    ----------
    status:
        Optional aggregate-status filter (``pending``, ``running``,
        ``completed``, ``failed``, ``partial_success``).
    offset:
        Zero-based page start (>= 0).
    limit:
        Page size, 1..500.
    sort:
        ``created_desc`` (default), ``created_asc``,
        ``finalized_desc``, ``finalized_asc``.

    Returns ``{jobs, total, total_filtered, offset, limit, sort}`` with
    a compact projection per job (see :func:`_serialize_job`).
    """
    from nemo_retriever.service.services.job_tracker import (
        JobAggregateStatus,
        get_job_tracker,
    )

    if offset < 0:
        raise HTTPException(400, "offset must be >= 0")
    if limit < 1 or limit > 500:
        raise HTTPException(400, "limit must be in [1, 500]")

    valid_sorts = {"created_desc", "created_asc", "finalized_desc", "finalized_asc"}
    if sort not in valid_sorts:
        raise HTTPException(400, f"sort must be one of {sorted(valid_sorts)}, got {sort!r}")

    tracker = get_job_tracker()
    if tracker is None:
        return JSONResponse(
            {
                "jobs": [],
                "total": 0,
                "total_filtered": 0,
                "offset": offset,
                "limit": limit,
                "sort": sort,
            }
        )

    jobs = list(tracker.all_jobs())

    if status is not None:
        valid_status = {s.value for s in JobAggregateStatus}
        if status not in valid_status:
            raise HTTPException(
                400,
                f"status must be one of {sorted(valid_status)}, got {status!r}",
            )
        filtered = [j for j in jobs if j.status.value == status]
    else:
        filtered = jobs

    # Sort newest/oldest by either creation or finalization timestamp.
    # ISO-8601 sorts lexicographically; ``None`` slots are pushed last
    # in descending order, first in ascending order so the UI always
    # surfaces *known* timestamps first.
    if sort.startswith("created"):
        key = lambda j: j.created_at  # noqa: E731
        reverse = sort == "created_desc"
    else:
        key = lambda j: (j.finalized_at or "")  # noqa: E731
        reverse = sort == "finalized_desc"
    filtered.sort(key=key, reverse=reverse)

    page = filtered[offset : offset + limit]
    return JSONResponse(
        {
            "jobs": [_serialize_job(j) for j in page],
            "total": len(jobs),
            "total_filtered": len(filtered),
            "offset": offset,
            "limit": limit,
            "sort": sort,
        }
    )


@router.get("/api/jobs/{job_id}")
async def jobs_detail(request: Request, job_id: str) -> JSONResponse:
    """Single-job aggregate view for the detail page.

    Returns the compact aggregate projection together with up to 500
    document records — enough to render the per-doc table immediately
    on page load. The full list is paginated separately at
    :func:`jobs_documents`.
    """
    from nemo_retriever.service.services.job_tracker import get_job_tracker

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(503, "Job tracker not available")

    agg = tracker.get_job(job_id)
    if agg is None:
        raise HTTPException(404, f"Job {job_id!r} not found")

    docs = tracker.job_documents(job_id)
    sample_cap = 500
    return JSONResponse(
        {
            **_serialize_job(agg),
            "documents": [d.model_dump() for d in docs[:sample_cap]],
            "documents_truncated": len(docs) > sample_cap,
        }
    )


@router.delete("/api/jobs/{job_id}")
async def delete_job(request: Request, job_id: str) -> JSONResponse:
    """Delete a dashboard debug job and its retained result artifacts.

    VectorDB rows are intentionally not touched: the current ingest write
    schema has no guaranteed job_id ownership key and deleting by filename or
    source would risk removing rows belonging to another ingest.
    """
    from nemo_retriever.service.services.job_tracker import get_job_tracker
    from nemo_retriever.service.services.worker_result_store import (
        ResultStoreTemporarilyUnavailable,
        delete_result_data,
    )

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(503, "Job tracker not available")

    document_ids = tracker.delete_job(job_id)
    if document_ids is None:
        raise HTTPException(404, f"Job {job_id!r} not found")

    try:
        for document_id in document_ids:
            delete_result_data(document_id)
    except ResultStoreTemporarilyUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc

    return JSONResponse(
        {
            "job_id": job_id,
            "deleted_documents": len(document_ids),
            "retained_results_deleted": True,
            "vectordb_rows_deleted": False,
            "vectordb_note": "VectorDB rows were kept because the current schema has no safe job ownership key.",
        }
    )


@router.get("/api/jobs/{job_id}/documents")
async def jobs_documents(
    request: Request,
    job_id: str,
    status: str | None = None,
    offset: int = 0,
    limit: int = 100,
) -> JSONResponse:
    """Paginated documents for one job — backs the detail-page table."""
    from nemo_retriever.service.services.job_tracker import (
        DocumentStatus,
        get_job_tracker,
    )

    if offset < 0:
        raise HTTPException(400, "offset must be >= 0")
    if limit < 1 or limit > 1000:
        raise HTTPException(400, "limit must be in [1, 1000]")

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(503, "Job tracker not available")

    if tracker.get_job(job_id) is None:
        raise HTTPException(404, f"Job {job_id!r} not found")

    docs = tracker.job_documents(job_id)

    if status is not None:
        valid = {s.value for s in DocumentStatus}
        if status not in valid:
            raise HTTPException(
                400,
                f"status must be one of {sorted(valid)}, got {status!r}",
            )
        filtered = [d for d in docs if d.status.value == status]
    else:
        filtered = docs

    page = filtered[offset : offset + limit]
    return JSONResponse(
        {
            "job_id": job_id,
            "total": len(docs),
            "total_filtered": len(filtered),
            "offset": offset,
            "limit": limit,
            "items": [d.model_dump() for d in page],
        }
    )


# ── Per-document pipeline trace (debug UI) ──────────────────────────


def _trace_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _trace_page(row: dict[str, Any], fallback: int = 1) -> int:
    metadata = _trace_metadata(row)
    nested = metadata.get("content_metadata") if isinstance(metadata.get("content_metadata"), dict) else {}
    for value in (row.get("page_number"), metadata.get("page_number"), nested.get("page_number")):
        try:
            page = int(value)
            if page > 0:
                return page
        except (TypeError, ValueError):
            continue
    return fallback


def _trace_content_type(row: dict[str, Any]) -> str:
    metadata = _trace_metadata(row)
    content_type = row.get("_content_type") or row.get("content_type") or metadata.get("_content_type")
    nested = metadata.get("content_metadata") if isinstance(metadata.get("content_metadata"), dict) else {}
    source_type = str(nested.get("source_type") or "")
    if not content_type and source_type in {"native_cell", "native_csv"}:
        return "spreadsheet_table"
    if not content_type and source_type == "chart_data":
        return "chart"
    if not content_type and source_type == "embedded_image":
        return "image"
    return str(content_type or "text")


def _trace_reader(row: dict[str, Any]) -> str:
    metadata = _trace_metadata(row)
    nested = metadata.get("content_metadata") if isinstance(metadata.get("content_metadata"), dict) else {}
    reader = row.get("_reader_backend") or metadata.get("reader_backend") or nested.get("reader_backend")
    if reader:
        return str(reader)
    source_type = str(nested.get("source_type") or "")
    if source_type in {"native_cell", "native_csv"}:
        return "native_spreadsheet"
    return "unknown"


def _trace_ocr_pipeline(row: dict[str, Any]) -> str | None:
    """Return the request-scoped OCR selector, including legacy row evidence."""
    metadata = _trace_metadata(row)
    selected = metadata.get("ocr_pipeline")
    ocr = row.get("ocr") if isinstance(row.get("ocr"), dict) else {}
    if selected is None:
        selected = ocr.get("pipeline")
    # Older jobs did not persist the selector. Their stage/models still give
    # an unambiguous answer and prevent the dashboard from showing the active
    # worker config (often Nemotron) for a historical Tesseract job.
    if selected is None and ocr.get("stage") == "page_elements_box_ocr":
        selected = "pipeline-tesseract"
    if selected is None:
        models = ocr.get("models") if isinstance(ocr.get("models"), dict) else {}
        recognizer = str(models.get("recognizer") or "").lower()
        if "tesseract" in recognizer:
            selected = "pipeline-tesseract"
        elif "pp-ocr" in recognizer or "ppocr" in recognizer:
            selected = "pipeline-option3"
    return str(selected) if selected else None


def _trace_document_ocr_pipeline(document: Any) -> str | None:
    """Read a request-scoped selector from compact document diagnostics.

    Raw result rows are optional in the FE flow.  The worker therefore stores
    the selector in ``pipeline_diagnostics`` so the dashboard remains
    accurate after the tracker intentionally drops those rows.
    """
    diagnostics = getattr(document, "pipeline_diagnostics", None)
    if not isinstance(diagnostics, dict):
        return None
    selected = diagnostics.get("ocr_pipeline") or diagnostics.get("pipeline_selector")
    return str(selected) if selected else None


def _trace_has_ocr_output(row: dict[str, Any]) -> bool:
    """Treat native passthrough bookkeeping as non-OCR output."""
    if _trace_reader(row) == "ocr":
        return True
    value = row.get("ocr")
    return not (isinstance(value, dict) and value.get("status") == "skipped") and value not in (None, "", [], {})


def _trace_text(row: dict[str, Any]) -> str:
    metadata = _trace_metadata(row)
    for value in (row.get("text"), row.get("content"), row.get("markdown"), metadata.get("content")):
        if value is not None and str(value):
            return str(value)
    return ""


def _trace_bbox(row: dict[str, Any]) -> list[float] | None:
    metadata = _trace_metadata(row)
    for value in (
        row.get("_bbox_xyxy_norm"),
        row.get("bbox_xyxy_norm"),
        metadata.get("_bbox_xyxy_norm"),
        metadata.get("bbox_xyxy_norm"),
    ):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                continue
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            try:
                box = [float(v) for v in value[:4]]
                if all(0 <= v <= 1 for v in box):
                    return [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]
            except (TypeError, ValueError):
                pass
    return None


def _trace_safe_output(value: Any, *, key: str = "") -> Any:
    """Keep model output inspectable without sending huge image/vector blobs."""
    lowered = key.lower()
    if "embedding" in lowered or "vector" in lowered:
        if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
            return {"kind": "vector", "dim": len(value), "preview": value[:16]}
    if lowered.endswith("image_b64") or lowered == "image_b64":
        if value:
            return {"kind": "binary_image", "retained": True, "chars": len(str(value))}
        return None
    if isinstance(value, dict):
        return {str(k): _trace_safe_output(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trace_safe_output(v, key=key) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _trace_output_fields(row: dict[str, Any]) -> dict[str, Any]:
    output_keys = (
        "page_elements_v3",
        "table_structure_v1",
        "table_structure_ocr_v1",
        "ocr",
        "chart",
        "table",
        "infographic",
        "stamp",
        "stamps",
        "stamp_detection",
        "stamp_regions",
        "images",
    )
    outputs: dict[str, Any] = {}
    for key in output_keys:
        value = row.get(key)
        if value not in (None, "", [], {}):
            outputs[key] = _trace_safe_output(value, key=key)
    return outputs


def _trace_block_outputs(row: dict[str, Any], content_type: str) -> dict[str, Any]:
    """Select outputs that belong to this canonical block, not page baggage."""
    outputs: dict[str, Any] = {"text": _trace_text(row)}
    if content_type in {"table", "spreadsheet_table"}:
        keys = ("table", "table_structure_v1", "table_structure_ocr_v1", "ocr")
    elif content_type in {"chart", "infographic", "stamp"}:
        keys = ("chart", "infographic", "stamp", "stamps", "stamp_detection", "stamp_regions", "ocr", "_ocr_visual_text_blocks", "images")
    elif content_type == "image":
        keys = ("images", "ocr", "_ocr_visual_text_blocks")
    else:
        keys = ()
    for key in keys:
        value = row.get(key)
        if value not in (None, "", [], {}):
            outputs[key] = _trace_safe_output(value, key=key)
    metadata = _trace_metadata(row)
    nested = metadata.get("content_metadata") if isinstance(metadata.get("content_metadata"), dict) else {}
    if content_type.startswith("spreadsheet") or nested.get("source_type") in {"chart_data", "embedded_image"}:
        outputs["content_metadata"] = _trace_safe_output(nested, key="content_metadata")
    return outputs


def _trace_embedding(row: dict[str, Any]) -> dict[str, Any] | None:
    for key, value in row.items():
        if "embedding" not in str(key).lower():
            continue
        if isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value):
            return {"column": key, "dim": len(value), "preview": value[:16], "retained": True}
        if isinstance(value, dict) or value is None:
            dim = row.get(f"{key}_dim") or row.get("text_embeddings_1b_v2_dim")
            has_embedding = row.get(f"{key}_has_embedding")
            if has_embedding is True or row.get("_contains_embeddings") is True or dim:
                return {"column": key, "dim": int(dim or 0), "preview": None, "retained": False, "note": "Vector đã được ghi vào sink nhưng retained result chỉ giữ metadata vector."}
    return None


def _trace_endpoint(config: dict[str, Any], key: str) -> str | None:
    value = config.get(key)
    if not value:
        return None
    # Internal service hostnames are useful in server logs but not in the UI.
    try:
        from urllib.parse import urlsplit

        parsed = urlsplit(str(value))
        return parsed.path or str(value)
    except Exception:
        return str(value)


def _trace_model_config(
    request: Request,
    *,
    selector_override: str | None = None,
) -> dict[str, Any]:
    """Return the redacted config, honoring a request-scoped OCR selector.

    The dashboard worker has one server-owned default config, while the
    upload debug screen can select an OCR pipeline per document.  Endpoint
    wiring remains server-owned, but the displayed backend/model family must
    follow the selected request; otherwise a Pipeline 5 result is rendered as
    the worker's default Option 2 backend.
    """
    try:
        from nemo_retriever.service.services.pipeline_executor import get_pipeline_configs

        configs = get_pipeline_configs()
        selected = configs.get("batch") or configs.get("realtime") or {}
        extract = selected.get("extract_params") or {}
        embed = selected.get("embed_params") or {}
        nim = selected.get("nim_endpoints") or {}
        integrated_ocr_endpoint = _trace_endpoint(nim, "ocr_invoke_url")
        line_detector_endpoint = _trace_endpoint(nim, "line_detector_invoke_url")
        ocr_recognizer_endpoint = _trace_endpoint(nim, "ocr_recognizer_invoke_url")
        tesseract_endpoint = _trace_endpoint(nim, "tesseract_ocr_invoke_url")
        vintern_endpoint = _trace_endpoint(nim, "vintern_ocr_invoke_url")
        ministral_vlm_endpoint = _trace_endpoint(
            nim, "ministral_vlm_invoke_url"
        )
        vietnamese_ocr_endpoint = _trace_endpoint(
            nim, "vietnamese_ocr_invoke_url"
        )
        official_ppocr_endpoint = _trace_endpoint(nim, "official_ppocr_invoke_url")
        paddleocr_vl_endpoint = _trace_endpoint(nim, "paddleocr_vl_invoke_url")
        configured_selector = selector_override or extract.get("ocr_pipeline")
        # The Compose stack exposes the VietOCR endpoint by default so an
        # Option 3 request can start without extra wiring.  Endpoint presence
        # therefore cannot identify the active selector; use the explicit
        # request/config selector so the default Nemotron pipeline is not
        # mislabeled as Option 3 in the dashboard.
        if configured_selector in {"pipeline-ppocrv6", "pipeline-tesseract"}:
            ocr_backend = "option2_nemotron_language_routed_vietnamese_ocr"
            ocr_models = {
                "layout": "Page Elements v3 + Table Structure v1",
                "baseline": "Nemotron OCR v2",
                "router": "Vietnamese Unicode signal + langdetect vi/en",
                "vietnamese_recognizer": "VietOCR vgg_seq2seq",
            }
        elif configured_selector == "pipeline-option3":
            ocr_backend = "option3_nemotron_language_routed_vietnamese_ocr"
            ocr_models = {
                "layout": "Page Elements v3 + Table Structure v1",
                "baseline": "Nemotron OCR v2",
                "router": "Vietnamese Unicode signal + langdetect vi/en",
                "vietnamese_recognizer": "VietOCR vgg_seq2seq",
            }
        elif configured_selector == "pipeline-option5":
            ocr_backend = "option5_nemotron_language_routed_vietnamese_ocr"
            ocr_models = {
                "layout": "Page Elements v3 + Table Structure v1",
                "baseline": "Nemotron OCR v2",
                "router": "Vietnamese Unicode signal + langdetect vi/en",
                "vietnamese_recognizer": "VietOCR vgg_seq2seq",
            }
        elif configured_selector == "pipeline-option6":
            ocr_backend = "option6_page_detect_qwen35_vlm"
            ocr_models = {
                "layout": "NIM Page Elements v3 · semantic bbox · logical batch 128",
                "ocr": "Qwen3.5-2B VLM · model selected by OPTION6_MODEL · vLLM · max 25 concurrent",
                "table": "Native PDFium text → Qwen Markdown; scan/weak-native fallback crop · Table Structure NIM tắt",
                "visual": "Page Elements visual crop · short label only",
            }
        elif configured_selector == "pipeline-option7":
            ocr_backend = "option7_ministral_vlm"
            ocr_models = {
                "layout": "NIM Page Elements v3 · semantic bbox + visual evidence",
                "ocr": "Ministral-3-3B-Instruct-2512 · FP8 OCR semantic crop/full-page",
                "table": "Page Elements table bbox → Ministral whole-table Markdown · Table Structure tắt",
                "semantic_ocr": "enabled · Page Elements text/title/table bbox",
                "visual_ocr": "disabled · không gửi visual crop riêng",
                "visual_classification": "disabled · VLM chỉ làm OCR text",
                "line_detector": "disabled",
            }
        elif (
            integrated_ocr_endpoint and vintern_endpoint and tesseract_endpoint
        ):
            ocr_backend = "option2_language_routed"
            ocr_models = {
                "language_probe": "Tesseract 5 vie+eng",
                "vietnamese_recognizer": "Vintern-1B-v3.5 · vLLM",
                "english_recognizer": "Nemotron OCR v2",
            }
        elif official_ppocr_endpoint:
            ocr_backend = "ppocrv6_official"
            ocr_models = {
                "doc_orientation": "PP-LCNet_x1_0_doc_ori",
                "doc_unwarping": "UVDoc",
                "textline_orientation": "PP-LCNet_x1_0_textline_ori",
                "detector": "PP-OCRv6_medium_det",
                "recognizer": "PP-OCRv6_medium_rec",
            }
        elif integrated_ocr_endpoint:
            ocr_backend = "nemotron_ocr_v2"
            ocr_models = {"integrated": "Nemotron OCR v2"}
        elif line_detector_endpoint and ocr_recognizer_endpoint:
            ocr_backend = "ppocrv6"
            ocr_models = {
                "line_detector": "PP-OCRv6_medium_det",
                "recognizer": "PP-OCRv6_medium_rec",
            }
        else:
            ocr_backend = None
            ocr_models = {}
        trace_extract_params = {
            key: extract.get(key)
            for key in (
                "extract_text",
                "extract_images",
                "extract_tables",
                "extract_charts",
                "extract_stamps",
                "extract_page_as_image",
                "use_page_elements",
                "use_table_structure",
                "ocr_pipeline",
            )
            if key in extract
        }
        if selector_override is not None:
            trace_extract_params["ocr_pipeline"] = selector_override
            if selector_override == "pipeline-option7":
                trace_extract_params.update(
                    {
                        "extract_page_as_image": True,
                        "use_page_elements": True,
                        "use_table_structure": False,
                    }
                )
        table_structure_endpoint = (
            None
            if configured_selector == "pipeline-option7"
            else _trace_endpoint(nim, "table_structure_invoke_url")
        )
        return {
            "extract_method": extract.get("method"),
            "dpi": extract.get("dpi"),
            "ocr_backend": ocr_backend,
            "ocr_models": ocr_models,
            "page_elements_endpoint": _trace_endpoint(nim, "page_elements_invoke_url"),
            "table_structure_endpoint": table_structure_endpoint,
            "ocr_endpoint": integrated_ocr_endpoint,
            "vietnamese_ocr_endpoint": vietnamese_ocr_endpoint,
            "line_detector_endpoint": line_detector_endpoint,
            "ocr_recognizer_endpoint": ocr_recognizer_endpoint,
            "tesseract_endpoint": tesseract_endpoint,
            "vintern_endpoint": vintern_endpoint,
            "ministral_vlm_endpoint": ministral_vlm_endpoint,
            "official_ppocr_endpoint": official_ppocr_endpoint,
            "paddleocr_vl_endpoint": paddleocr_vl_endpoint,
            "embedding_endpoint": _trace_endpoint(nim, "embed_invoke_url"),
            "embedding_model": embed.get("embed_model_name") or embed.get("model_name"),
            "extract_params": trace_extract_params,
        }
    except Exception as exc:
        logger.debug("Could not build dashboard pipeline trace config: %s", exc)
        return {}


def _build_pipeline_trace(request: Request, *, job: Any, document: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    # A dashboard upload may override the OCR selector per document.  Resolve
    # it from retained row metadata before building the redacted config so the
    # FE does not show the worker's default backend for a request-scoped
    # Pipeline 5 job.
    request_selector = next(
        (
            selected
            for row in rows
            if (selected := _trace_ocr_pipeline(row)) is not None
        ),
        None,
    ) or _trace_document_ocr_pipeline(document)
    config = _trace_model_config(request, selector_override=request_selector)
    extension = Path(str(document.filename or "")).suffix.lower().lstrip(".")
    is_spreadsheet = extension in {"xlsx", "xls", "csv"}
    by_page: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        by_page.setdefault(_trace_page(row), []).append((index, row))

    pages: list[dict[str, Any]] = []
    for page_number in sorted(by_page):
        page_rows = by_page[page_number]
        base_row_index, base_row = next(
            ((row_index, row) for row_index, row in page_rows if _trace_content_type(row) == "text"),
            page_rows[0],
        )
        text = "\n".join(
            _trace_text(row)
            for _, row in page_rows
            if is_spreadsheet or _trace_content_type(row) == "text"
        ).strip()
        metadata = _trace_metadata(base_row)
        selected_ocr_pipeline = _trace_ocr_pipeline(base_row)
        selected_backend = {
            "pipeline-nemotron-ocr": "nemotron_ocr_v2",
            "pipeline-ppocrv6": "option2_nemotron_language_routed_vietnamese_ocr",
            "pipeline-option3": "option3_nemotron_language_routed_vietnamese_ocr",
            "pipeline-option4": "option4",
            "pipeline-option5": "option5_nemotron_language_routed_vietnamese_ocr",
            "pipeline-option6": "option6_page_detect_qwen35_vlm",
            "pipeline-option7": "option7_ministral_vlm",
            "pipeline-tesseract": "option2_nemotron_language_routed_vietnamese_ocr",
        }.get(selected_ocr_pipeline)
        page_ocr_backend = selected_backend or config.get("ocr_backend")
        native = any(
            _trace_reader(row) in {"native_pdf", "openpyxl", "python_csv", "native_spreadsheet"}
            for _, row in page_rows
        )
        page_elements = base_row.get("page_elements_v3")
        has_page_elements = page_elements not in (None, "", [], {})
        has_table = any(row.get("table") not in (None, "", [], {}) for _, row in page_rows)
        has_table_structure_stage = any(
            isinstance(row.get("table_structure_v1"), dict)
            and (
                "regions" in row.get("table_structure_v1", {})
                or "error" in row.get("table_structure_v1", {})
            )
            for _, row in page_rows
        )
        has_chart = any(row.get("chart") not in (None, "", [], {}) for _, row in page_rows)
        stamp_result = next((row.get("stamp_detection") for _, row in page_rows if row.get("stamp_detection") not in (None, "", [], {})), None)
        has_stamp = any(row.get("stamp") not in (None, "", [], {}) or row.get("stamp_regions") not in (None, "", [], {}) for _, row in page_rows)
        has_ocr = any(_trace_has_ocr_output(row) for _, row in page_rows)
        vectors = [_trace_embedding(row) for _, row in page_rows]
        vectors = [v for v in vectors if v]

        blocks: list[dict[str, Any]] = []
        for row_index, row in page_rows:
            content_type = _trace_content_type(row)
            reader = _trace_reader(row)
            nested = _trace_metadata(row).get("content_metadata")
            nested = nested if isinstance(nested, dict) else {}
            models: list[dict[str, Any]] = []
            if reader == "native_pdf":
                models.append({"kind": "library", "name": "pypdfium2", "function": "pdf_extraction"})
            elif reader in {"openpyxl", "native_spreadsheet"}:
                models.append({"kind": "library", "name": "openpyxl", "function": "_xlsx_to_rows"})
            elif reader == "python_csv":
                models.append({"kind": "library", "name": "Python csv", "function": "_csv_to_rows"})
            if content_type in {"text", "table", "chart", "infographic", "image", "stamp"} and not is_spreadsheet:
                row_has_ocr = _trace_has_ocr_output(row)
                if page_ocr_backend == "ppocrv6_official":
                    models.extend([
                        {
                            "kind": "preprocess",
                            "name": "PP-LCNet_x1_0_doc_ori + UVDoc + PP-LCNet_x1_0_textline_ori",
                            "function": "official_ppocr_preprocess",
                            "endpoint": config.get("official_ppocr_endpoint"),
                        },
                        {
                            "kind": "detector",
                            "name": "PP-OCRv6_medium_det · whole page",
                            "function": "official_ppocr_text_detection",
                            "endpoint": config.get("official_ppocr_endpoint"),
                        },
                    ])
                    if row_has_ocr:
                        models.append({
                            "kind": "ocr",
                            "name": "PP-OCRv6_medium_rec · detected text boxes",
                            "function": "official_ppocr_text_recognition",
                            "endpoint": config.get("official_ppocr_endpoint"),
                        })
                elif page_ocr_backend == "paddleocr_vl":
                    models.append({
                        "kind": "detector",
                        "name": "PP-DocLayoutV3 · PaddleOCR-VL 1.6",
                        "function": "paddleocr_vl_layout_parse",
                        "endpoint": config.get("paddleocr_vl_endpoint"),
                    })
                    if content_type == "table":
                        models.append({
                            "kind": "structure",
                            "name": "PaddleOCR-VL 1.6 · table structured output",
                            "function": "paddleocr_vl_table_parse",
                            "endpoint": config.get("paddleocr_vl_endpoint"),
                        })
                    if content_type in {"text", "table"} and row_has_ocr:
                        models.append({
                            "kind": "ocr",
                            "name": "PaddleOCR-VL-1.6-0.9B · vLLM",
                            "function": "paddleocr_vl_vllm_recognize",
                            "endpoint": config.get("paddleocr_vl_endpoint"),
                        })
                else:
                    if page_ocr_backend == "option7_ministral_vlm":
                        models.append({
                            "kind": "layout",
                            "name": "NIM Page Elements v3 · semantic bbox",
                            "function": "detect_page_elements_v3",
                            "endpoint": config.get("page_elements_endpoint"),
                        })
                    elif page_ocr_backend in {
                        "option2_nemotron_language_routed_vietnamese_ocr",
                        "option3_nemotron_language_routed_vietnamese_ocr",
                        "option5_nemotron_language_routed_vietnamese_ocr",
                        "option6_page_detect_qwen35_vlm",
                    }:
                        models.append({
                            "kind": "layout",
                            "name": "NIM Page Elements v3",
                            "function": "detect_page_elements_v3",
                            "endpoint": config.get("page_elements_endpoint"),
                        })
                    else:
                        models.append({"kind": "detector", "name": "NIM Page Elements v3", "function": "detect_page_elements_v3", "endpoint": config.get("page_elements_endpoint")})
                    if content_type == "table" and page_ocr_backend != "option7_ministral_vlm":
                        models.append({"kind": "structure", "name": "NIM Table Structure v1", "function": "TableStructureActor", "endpoint": config.get("table_structure_endpoint")})
                if page_ocr_backend == "option2_language_routed" and content_type in {"text", "table"} and row_has_ocr:
                    models.append({
                        "kind": "router",
                        "name": "Tesseract 5 · probe vie+eng",
                        "function": "detect_probe_language",
                        "endpoint": config.get("tesseract_endpoint"),
                    })
                    routes = set()
                    route_values = _trace_metadata(row).get("ocr_language_routes")
                    if isinstance(route_values, list):
                        routes = {
                            str(value.get("selected_backend"))
                            for value in route_values
                            if isinstance(value, dict) and value.get("selected_backend")
                        }
                    if not routes:
                        routes = {"nemotron", "vintern"}
                    if "nemotron" in routes:
                        models.append({
                            "kind": "ocr",
                            "name": "Nemotron OCR v2 · English/uncertain crop",
                            "function": "invoke_image_inference_batches",
                            "endpoint": config.get("ocr_endpoint"),
                        })
                    if "vintern" in routes:
                        models.append({
                            "kind": "ocr",
                            "name": "Vintern-1B-v3.5 · vLLM · Vietnamese crop",
                            "function": "vllm_chat_completions_images",
                            "endpoint": config.get("vintern_endpoint"),
                        })
                if page_ocr_backend not in {"paddleocr_vl", "ppocrv6_official"} and content_type in {"text", "table"} and row_has_ocr and page_ocr_backend == "nemotron_ocr_v2":
                    models.append({
                        "kind": "ocr",
                        "name": "Nemotron OCR v2 · detect + recognize",
                        "function": "invoke_image_inference_batches",
                        "endpoint": config.get("ocr_endpoint"),
                    })
                elif page_ocr_backend in {
                    "option2_nemotron_language_routed_vietnamese_ocr",
                    "option3_nemotron_language_routed_vietnamese_ocr",
                    "option5_nemotron_language_routed_vietnamese_ocr",
                    "option6_page_detect_qwen35_vlm",
                    "option7_ministral_vlm",
                } and content_type in {"text", "table"} and row_has_ocr:
                    if page_ocr_backend == "option6_page_detect_qwen35_vlm":
                        models.append({
                            "kind": "ocr",
                            "name": "Qwen3.5-2B VLM · text semantic crop",
                            "function": "vllm_chat_completions_images",
                            "endpoint": config.get("vintern_endpoint"),
                        })
                        if content_type == "table":
                            models.append({
                                "kind": "format",
                                "name": "Qwen VLM · whole table → GitHub Markdown",
                                "function": "qwen_table_markdown_prompt",
                                "endpoint": config.get("vintern_endpoint"),
                            })
                    elif (
                        page_ocr_backend == "option7_ministral_vlm"
                    ):
                        models.append({
                            "kind": "ocr",
                            "name": "Ministral 3 3B · OCR semantic crop / full-page fallback",
                            "function": "vllm_chat_completions_images",
                            "endpoint": config.get("ministral_vlm_endpoint"),
                        })
                        if content_type == "table":
                            models.append({
                                "kind": "format",
                                "name": "Ministral VLM · whole table → GitHub Markdown",
                                "function": "ministral_table_markdown_prompt",
                                "endpoint": config.get("ministral_vlm_endpoint"),
                            })
                    else:
                        models.append({
                            "kind": "ocr",
                            "name": "Nemotron OCR v2 · baseline/fallback semantic crops",
                            "function": "invoke_image_inference_batches",
                            "endpoint": config.get("ocr_endpoint"),
                        })
                        models.append({
                            "kind": "router",
                            "name": "Unicode Vietnamese signal + langdetect vi/en",
                            "function": "route_nemotron_text",
                            "endpoint": None,
                        })
                        models.append({
                            "kind": "ocr",
                            "name": "VietOCR vgg_seq2seq · Vietnamese batch candidate",
                            "function": "vietnamese_recognizer_batch",
                            "endpoint": config.get("vietnamese_ocr_endpoint"),
                        })
                elif page_ocr_backend not in {"paddleocr_vl", "ppocrv6_official"} and content_type in {"text", "table"} and row_has_ocr and page_ocr_backend == "ppocrv6":
                    models.append({
                        "kind": "detector",
                        "name": "PP-OCRv6 medium det · tách dòng",
                        "function": "ppocrv6_line_detect",
                        "endpoint": config.get("line_detector_endpoint"),
                    })
                    models.append({"kind": "ocr", "name": "PP-OCRv6 medium rec · đọc crop", "function": "ppocrv6_recognize", "endpoint": config.get("ocr_recognizer_endpoint")})
                elif page_ocr_backend not in {"paddleocr_vl", "ppocrv6_official"} and content_type in {"text", "table"} and row_has_ocr and page_ocr_backend == "option4":
                    models.append({
                        "kind": "detector",
                        "name": "PP-OCRv6 medium det · tách line cho Option 4",
                        "function": "ppocrv6_line_detect",
                        "endpoint": config.get("line_detector_endpoint"),
                    })
                    models.append({
                        "kind": "ocr",
                        "name": "Option 4 · Tesseract-first → Nemotron fallback",
                        "function": "tesseract_first_nemotron_fallback",
                        "endpoint": config.get("ocr_endpoint"),
                    })
                elif page_ocr_backend not in {"paddleocr_vl", "ppocrv6_official"} and content_type in {"text", "table"} and row_has_ocr and page_ocr_backend == "tesseract":
                    models.append({
                        "kind": "ocr",
                        "name": "Tesseract 5 · đọc bbox Page Elements / tile scan",
                        "function": "page_elements_region_ocr",
                        "endpoint": config.get("tesseract_endpoint"),
                    })
            embedding = _trace_embedding(row)
            if embedding:
                models.append({"kind": "embedding", "name": config.get("embedding_model") or "Embedding NIM", "function": "embed_text_1b_v2", "endpoint": config.get("embedding_endpoint")})
            blocks.append(
                {
                    "block_id": str(row_index),
                    "row_index": row_index,
                    "reading_order": row.get("_reading_order"),
                    "content_type": content_type,
                    "reader_backend": reader,
                    "page_number": page_number,
                    "bbox": _trace_bbox(row),
                    "range": nested.get("range"),
                    "sheet_name": nested.get("sheet_name"),
                    "text": _trace_text(row),
                    "models": models,
                    "output_keys": sorted(_trace_block_outputs(row, content_type)),
                    "outputs": _trace_block_outputs(row, content_type),
                    "image_index": next((index for index, image in enumerate(row.get("images") or []) if isinstance(image, dict) and str(image.get("label_name") or "") in ({content_type} if content_type in {"image", "chart", "infographic", "stamp"} else set())), None),
                    "embedding": embedding,
                }
            )

        # PDFium exposes a scanned page as one full-page image. The useful
        # visual blocks come from Page Elements crops retained by the OCR
        # stage. Add them as trace-only blocks so they can be inspected and
        # overlaid without duplicating the canonical text rows.
        visual_items = base_row.get("images")
        if isinstance(visual_items, list):
            for image_index, image in enumerate(visual_items):
                if not isinstance(image, dict):
                    continue
                bbox = _trace_bbox(image)
                if bbox is None:
                    continue
                is_detected_region = image.get("image_type") == "detected_region"
                label_name = str(image.get("label_name") or "image")
                # A scanned PDF is represented by PDFium as one page-sized
                # raster. It is the page background used by OCR, not a semantic
                # image block. Keep the page image endpoint, but do not show a
                # misleading full-page "image" block in the trace.
                bbox_area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
                if has_ocr and not native and not is_detected_region and bbox_area >= 0.90:
                    continue
                visual_type = label_name if is_detected_region and label_name in {"chart", "infographic", "stamp"} else "image"
                visual_text = str(image.get("text") or "")
                visual_models = []
                if is_detected_region:
                    visual_models.append({
                        "kind": "detector",
                        "name": "PP-DocLayoutV3 · PaddleOCR-VL 1.6" if page_ocr_backend == "paddleocr_vl" else "NIM Page Elements v3",
                        "function": "paddleocr_vl_layout_parse" if page_ocr_backend == "paddleocr_vl" else "detect_page_elements_v3",
                        "endpoint": config.get("paddleocr_vl_endpoint") if page_ocr_backend == "paddleocr_vl" else config.get("page_elements_endpoint"),
                    })
                else:
                    visual_models.append(
                        {"kind": "library", "name": "pypdfium2", "function": "extract_image_like_objects_from_pdfium_page"}
                    )
                blocks.append(
                    {
                        "block_id": f"{base_row_index}:visual:{image_index}",
                        "row_index": base_row_index,
                        "image_index": image_index,
                        "reading_order": None,
                        "content_type": visual_type,
                            "reader_backend": "page_elements" if is_detected_region else "native_pdf_image",
                        "page_number": page_number,
                        "bbox": bbox,
                        "range": None,
                        "sheet_name": None,
                        "text": visual_text,
                        "models": visual_models,
                        "output_keys": ["images"] + (["text"] if visual_text else []),
                        "outputs": {
                            "text": visual_text,
                            "images": _trace_safe_output(image, key="images"),
                        },
                        "embedding": None,
                    }
                )

        if page_ocr_backend == "option2_language_routed":
            ocr_stages = [
                {
                    "id": "language_probe",
                    "label": "Probe ngôn ngữ từng crop",
                    "status": "observed" if has_ocr else "not_applicable",
                    "model": "Tesseract 5 · vie+eng",
                    "function": "detect_probe_language",
                    "endpoint": config.get("tesseract_endpoint"),
                    "description": "Đọc probe song ngữ để chọn backend: có tín hiệu tiếng Việt → Vintern; tiếng Anh/không chắc → Nemotron OCR.",
                    "output": _trace_safe_output(base_row.get("ocr"), key="ocr") if has_ocr else None,
                },
                {
                    "id": "language_routed_recognition",
                    "label": "OCR theo ngôn ngữ",
                    "status": "observed" if has_ocr else "not_applicable",
                    "model": "Vintern-1B-v3.5 · vLLM (vi) + Nemotron OCR v2 (en/uncertain)",
                    "function": "option2_language_routed_batch_ocr",
                    "endpoint": config.get("vintern_endpoint"),
                    "secondary_endpoint": config.get("ocr_endpoint"),
                    "description": "Các crop text/cell được gom batch; hai nhóm backend chạy song song, giữ bbox Page Elements/Table Structure.",
                    "output": _trace_safe_output(base_row.get("ocr"), key="ocr") if has_ocr else None,
                },
            ]
        elif page_ocr_backend == "ppocrv6_official":
            ocr_stages = [
                {
                    "id": "ppocr_preprocess",
                    "label": "PP-OCRv6 tiền xử lý ảnh",
                    "status": "observed" if has_ocr else "not_applicable",
                    "model": "PP-LCNet_x1_0_doc_ori + UVDoc + PP-LCNet_x1_0_textline_ori",
                    "function": "official_ppocr_preprocess",
                    "endpoint": config.get("official_ppocr_endpoint"),
                    "description": "Xoay trang, sửa méo ảnh và xác định hướng từng dòng theo pipeline chính thức PaddleOCR.",
                    "output": _trace_safe_output(base_row.get("ocr"), key="ocr") if has_ocr else None,
                },
                {
                    "id": "ppocr_detection",
                    "label": "PP-OCRv6 detect toàn trang",
                    "status": "observed" if has_page_elements else "not_applicable",
                    "model": "PP-OCRv6_medium_det",
                    "function": "official_ppocr_text_detection",
                    "endpoint": config.get("official_ppocr_endpoint"),
                    "description": "Detect các text box trực tiếp trên ảnh trang; không qua NIM Page Elements hoặc layout model khác.",
                    "output": _trace_safe_output(base_row.get("page_elements_v3"), key="page_elements_v3") if has_page_elements else None,
                },
                {
                    "id": "ppocr_recognition",
                    "label": "PP-OCRv6 recognize từng text box",
                    "status": "observed" if has_ocr else "not_applicable",
                    "model": "PP-OCRv6_medium_rec",
                    "function": "official_ppocr_text_recognition",
                    "endpoint": config.get("official_ppocr_endpoint"),
                    "description": "Crop từng bbox do PP-OCRv6 detect trả về và nhận diện bằng model recognition chính thức.",
                    "output": _trace_safe_output(base_row.get("ocr"), key="ocr") if has_ocr else None,
                },
            ]
        elif page_ocr_backend == "paddleocr_vl":
            ocr_stages = [
                {
                    "id": "paddle_layout",
                    "label": "PaddleOCR-VL nhận diện layout và bbox",
                    "status": "observed" if has_page_elements else "not_applicable",
                    "model": "PP-DocLayoutV3",
                    "function": "paddleocr_vl_layout_parse",
                    "endpoint": config.get("paddleocr_vl_endpoint"),
                    "description": "PaddleOCR-VL API nhận ảnh trang, phát hiện block và sắp xếp reading order.",
                    "output": _trace_safe_output(base_row.get("page_elements_v3"), key="page_elements_v3") if has_page_elements else None,
                },
                {
                    "id": "paddle_vllm",
                    "label": "PaddleOCR-VL đọc nội dung block",
                    "status": "observed" if has_ocr else "not_applicable",
                    "model": "PaddleOCR-VL-1.6-0.9B · vLLM",
                    "function": "paddleocr_vl_vllm_recognize",
                    "endpoint": config.get("paddleocr_vl_endpoint"),
                    "description": "Service Paddle gọi vLLM nội bộ để đọc text, table, chart và các block phức tạp.",
                    "output": _trace_safe_output(base_row.get("ocr"), key="ocr") if has_ocr else None,
                },
            ]
        elif page_ocr_backend == "nemotron_ocr_v2":
            ocr_stages = [
                {
                    "id": "ocr",
                    "label": "Detect dòng và đọc chữ trên crop",
                    "status": "observed" if has_ocr else "not_applicable",
                    "model": "Nemotron OCR v2",
                    "function": "invoke_image_inference_batches",
                    "endpoint": config.get("ocr_endpoint"),
                    "description": "OCR tích hợp của pipeline cũ: detect bbox và recognize; trang scan còn chạy toàn trang + tile chồng lấn rồi hợp nhất.",
                    "output": _trace_safe_output(base_row.get("ocr"), key="ocr") if has_ocr else None,
                }
            ]
        elif page_ocr_backend in {
            "option2_nemotron_language_routed_vietnamese_ocr",
            "option3_nemotron_language_routed_vietnamese_ocr",
            "option5_nemotron_language_routed_vietnamese_ocr",
        }:
            ocr_stages = [
                {
                    "id": "nemotron_baseline",
                    "label": "Nemotron OCR v2 đọc tất cả semantic crops",
                    "status": "observed" if has_ocr else "not_applicable",
                    "model": "Nemotron OCR v2",
                    "function": "invoke_image_inference_batches",
                    "endpoint": config.get("ocr_endpoint"),
                    "description": "Nemotron là baseline authoritative; response từng recognition item giữ text, score và bbox local rồi map về bbox trang.",
                    "output": _trace_safe_output(base_row.get("ocr"), key="ocr") if has_ocr else None,
                },
                {
                    "id": "language_router",
                    "label": "Route raw text: Unicode + langdetect",
                    "status": "observed" if has_ocr else "not_applicable",
                    "model": "Deterministic Unicode/language router",
                    "function": "route_nemotron_text",
                    "description": "Chỉ raw text Nemotron đi vào router; tiếng Anh và uncertain giữ Nemotron.",
                    "output": _trace_safe_output(base_row.get("ocr"), key="ocr") if has_ocr else None,
                },
                {
                    "id": "vietnamese_recognizer",
                    "label": "VietOCR batch cho candidate tiếng Việt",
                    "status": "observed" if has_ocr else "not_applicable",
                    "model": "VietOCR vgg_seq2seq",
                    "function": "vietnamese_recognizer_batch",
                    "endpoint": config.get("vietnamese_ocr_endpoint"),
                    "description": "Chỉ candidate route Vietnamese được gửi một logical batch/page; Quality Gate quyết định có thay Nemotron hay fallback.",
                    "output": _trace_safe_output(base_row.get("ocr"), key="ocr") if has_ocr else None,
                },
            ]
        elif page_ocr_backend == "option6_page_detect_qwen35_vlm":
            ocr_stages = [
                {
                    "id": "qwen35_vlm",
                    "label": "Qwen 3.5 đọc text + bảng Markdown",
                    "status": "observed" if has_ocr else "not_applicable",
                    "model": "Qwen3.5-2B · model selected by OPTION6_MODEL",
                    "function": "vllm_chat_completions_images",
                    "endpoint": config.get("vintern_endpoint"),
                    "description": (
                        "Page Elements gửi semantic text crop và whole-table crop; "
                        "vLLM giữ tối đa 25 request liên tục, native PDFium text "
                        "được giữ nguyên và chỉ bù block thiếu."
                    ),
                    "output": _trace_safe_output(base_row.get("ocr"), key="ocr") if has_ocr else None,
                },
            ]
        elif page_ocr_backend == "option7_ministral_vlm":
            ocr_stages = [
                {
                    "id": "ministral_vlm",
                    "label": "Ministral 3 3B đọc semantic crop / scan fallback",
                    "status": "observed" if has_ocr else "not_applicable",
                    "model": "Ministral-3-3B-Instruct-2512 · FP8",
                    "function": "vllm_chat_completions_images",
                    "endpoint": config.get("ministral_vlm_endpoint"),
                    "description": (
                        "Page Elements tạo semantic text/title/table bbox và giữ "
                        "visual bbox làm evidence. Ministral chỉ OCR semantic crop, whole-table crop "
                        "hoặc full-page fallback, không nhận visual crop riêng. "
                        "Language probe và line detector không dùng."
                    ),
                    "output": _trace_safe_output(base_row.get("ocr"), key="ocr") if has_ocr else None,
                },
            ]
        elif page_ocr_backend == "ppocrv6":
            ocr_stages = [
                {
                    "id": "line_detector",
                    "label": "Tách bbox block / dòng",
                    "status": "observed" if has_ocr else "not_applicable",
                    "model": "PP-OCRv6_medium_det",
                    "function": "ppocrv6_line_detect",
                    "endpoint": config.get("line_detector_endpoint"),
                    "output": _trace_safe_output(base_row.get("ocr"), key="ocr") if has_ocr else None,
                },
                {
                    "id": "recognizer",
                    "label": "Đọc từng crop dòng / ô",
                    "status": "observed" if has_ocr else "not_applicable",
                    "model": "PP-OCRv6_medium_rec",
                    "function": "ppocrv6_recognize",
                    "endpoint": config.get("ocr_recognizer_endpoint"),
                    "output": _trace_safe_output(base_row.get("ocr"), key="ocr") if has_ocr else None,
                },
            ]
        elif page_ocr_backend == "option4":
            ocr_stages = [
                {
                    "id": "line_detector",
                    "label": "PP-OCRv6 tách line trong từng block / ô",
                    "status": "observed" if has_ocr else "not_applicable",
                    "model": "PP-OCRv6_medium_det",
                    "function": "ppocrv6_line_detect",
                    "endpoint": config.get("line_detector_endpoint"),
                    "output": _trace_safe_output(base_row.get("ocr"), key="ocr") if has_ocr else None,
                },
                {
                    "id": "parallel_fusion",
                    "label": "Tesseract đọc trước → Nemotron fallback → fusion khi cần",
                    "status": "observed" if has_ocr else "not_applicable",
                    "model": "Nemotron OCR v2 + Tesseract 5",
                    "function": "tesseract_first_nemotron_fallback",
                    "endpoint": config.get("ocr_endpoint"),
                    "secondary_endpoint": config.get("tesseract_endpoint"),
                    "output": _trace_safe_output(base_row.get("ocr"), key="ocr") if has_ocr else None,
                }
            ]
        elif page_ocr_backend == "tesseract":
            ocr_stages = [
                {
                    "id": "region_ocr",
                    "label": "Tesseract 5 đọc từng vùng Page Elements / cell",
                    "status": "observed" if has_ocr else "not_applicable",
                    "model": "Tesseract 5",
                    "function": "page_elements_region_ocr",
                    "endpoint": config.get("tesseract_endpoint"),
                    "description": "Page Elements giữ bbox vùng; Tesseract đọc mỗi vùng một lần. Table dùng cell bbox từ NIM Table Structure.",
                    "output": _trace_safe_output(base_row.get("ocr"), key="ocr") if has_ocr else None,
                },
                {
                    "id": "scan_recall",
                    "label": "OCR toàn trang / tile khi vùng không có text",
                    "status": "observed" if isinstance(base_row.get("ocr"), dict) and base_row.get("ocr", {}).get("scan_recall") else "not_applicable",
                    "model": "Tesseract 5",
                    "function": "scan_full_page_tile_recall",
                    "endpoint": config.get("tesseract_endpoint"),
                    "description": "Chỉ dùng cho scan khi OCR theo vùng không trả text; kết quả giữ bbox toàn trang hoặc tile để tránh nhân đôi block.",
                    "output": _trace_safe_output(base_row.get("ocr"), key="ocr") if has_ocr else None,
                }
            ]
        else:
            ocr_stages = [
                {
                    "id": "ocr",
                    "label": "OCR",
                    "status": "not_configured",
                    "output": None,
                }
            ]

        page_stages = [
            {
                "id": "split",
                "label": "Tách PDF thành trang",
                "status": "observed",
                "executor": "PDFSplitActor",
                "function": "split_pdf_batch",
                "library": "pypdfium2",
                "output": {"page_number": page_number, "source_id": base_row.get("source_id"), "single_page": True},
            },
            {
                "id": "extract",
                "label": "Đọc nội dung trang",
                "status": "observed" if text or native else "not_observed",
                "executor": "PDFExtractionActor",
                "function": "pdf_extraction",
                "library": "pypdfium2",
                "output": {"reader_backend": "native_pdf" if native else "ocr", "has_text": bool(text), "text_chars": len(text), "page_image_retained": bool(base_row.get("page_image"))},
            },
            {
                "id": "page_elements",
                "label": "Nhận diện bố cục / block",
                "status": "observed" if has_page_elements else "not_observed",
                "model": (
                    "PP-OCRv6_medium_det"
                    if page_ocr_backend == "ppocrv6_official"
                    else "PP-DocLayoutV3" if page_ocr_backend == "paddleocr_vl" else (
                        "NIM Page Elements v3 · semantic bbox"
                        if page_ocr_backend == "option7_ministral_vlm"
                        else "NIM Page Elements v3"
                    )
                ),
                "function": (
                    "official_ppocr_text_detection"
                    if page_ocr_backend == "ppocrv6_official"
                    else "paddleocr_vl_layout_parse" if page_ocr_backend == "paddleocr_vl" else "detect_page_elements_v3"
                ),
                "endpoint": (
                    config.get("official_ppocr_endpoint")
                    if page_ocr_backend == "ppocrv6_official"
                    else config.get("paddleocr_vl_endpoint") if page_ocr_backend == "paddleocr_vl" else config.get("page_elements_endpoint")
                ),
                "output": _trace_safe_output(page_elements, key="page_elements_v3") if has_page_elements else None,
            },
            {
                "id": "table_structure",
                "label": (
                    "Table Structure tắt · dùng bbox table của Page Elements"
                    if page_ocr_backend == "option7_ministral_vlm"
                    else "Nhận diện cấu trúc bảng"
                ),
                "status": (
                    "not_applicable"
                    if page_ocr_backend == "option7_ministral_vlm"
                    else "observed" if (has_table or has_table_structure_stage) else "not_applicable"
                ),
                "model": (
                    "Tắt · Page Elements table bbox"
                    if page_ocr_backend == "option7_ministral_vlm"
                    else "PaddleOCR-VL 1.6" if page_ocr_backend == "paddleocr_vl" else "NIM Table Structure v1"
                ),
                "function": (
                    "disabled"
                    if page_ocr_backend == "option7_ministral_vlm"
                    else "paddleocr_vl_table_parse" if page_ocr_backend == "paddleocr_vl" else "TableStructureActor"
                ),
                "endpoint": (
                    None
                    if page_ocr_backend == "option7_ministral_vlm"
                    else config.get("paddleocr_vl_endpoint") if page_ocr_backend == "paddleocr_vl" else config.get("table_structure_endpoint")
                ),
                "output": (
                    None
                    if page_ocr_backend == "option7_ministral_vlm"
                    else _trace_safe_output(base_row.get("table"), key="table")
                    if page_ocr_backend == "paddleocr_vl" and has_table
                    else _trace_safe_output(base_row.get("table_structure_v1"), key="table_structure_v1")
                    if has_table or has_table_structure_stage
                    else None
                ),
            },
            *ocr_stages,
            {
                "id": "clean",
                "label": "Làm sạch text trùng",
                "status": "observed" if isinstance(metadata.get("cleaning"), dict) else "not_observed",
                "executor": "CleanContentRows",
                "function": "clean_content_rows",
                "output": metadata.get("cleaning"),
            },
            {
                "id": "explode",
                "label": "Tạo block đầu ra",
                "status": "observed" if blocks else "not_observed",
                "executor": "GraphIngestor",
                "function": "explode_content_to_rows",
                "output": {"block_count": len(blocks), "content_types": sorted({block["content_type"] for block in blocks})},
            },
            {
                "id": "embedding",
                "label": "Tạo embedding cho block",
                "status": "observed" if vectors else "not_observed",
                "model": config.get("embedding_model") or "Embedding NIM",
                "function": "embed_text_1b_v2",
                "endpoint": config.get("embedding_endpoint"),
                "output": {"embedded_blocks": len(vectors), "dimensions": sorted({item["dim"] for item in vectors})},
            },
        ]
        if page_ocr_backend == "ppocrv6_official":
            # The official adapter already exposes detection and recognition
            # as ``ppocr_*`` stages above.  The generic page-elements and
            # table-structure entries would otherwise make the trace claim
            # that those NIM operators ran, although Option 2 disables them.
            page_stages = [
                stage
                for stage in page_stages
                if stage.get("id") not in {"page_elements", "table_structure"}
            ]
        if is_spreadsheet:
            nested_metadata = metadata.get("content_metadata")
            nested_metadata = nested_metadata if isinstance(nested_metadata, dict) else {}
            source_types = sorted(
                {
                    str(
                        ((_trace_metadata(row).get("content_metadata") or {}).get("source_type"))
                        or "native"
                    )
                    for _, row in page_rows
                }
            )
            page_stages = [
                {
                    "id": "split",
                    "label": "Tách workbook thành sheet / vùng dữ liệu",
                    "status": "observed",
                    "executor": "SpreadsheetExtractActor",
                    "function": "spreadsheet_bytes_to_chunks_df",
                    "library": "openpyxl" if extension in {"xlsx", "xls"} else "Python csv",
                    "output": {"sheet_name": nested_metadata.get("sheet_name"), "page_number": page_number},
                },
                {
                    "id": "extract",
                    "label": "Đọc cell native / bản ghi CSV",
                    "status": "observed" if text else "not_observed",
                    "executor": "SpreadsheetExtractActor",
                    "function": "_xlsx_to_rows" if extension in {"xlsx", "xls"} else "_csv_to_rows",
                    "library": "openpyxl" if extension in {"xlsx", "xls"} else "csv.reader",
                    "output": {
                        "reader_backend": sorted({_trace_reader(row) for _, row in page_rows}),
                        "text_chars": len(text),
                        "source_types": source_types,
                    },
                },
                {"id": "page_elements", "label": "Nhận diện bố cục / block", "status": "not_applicable", "output": None},
                {"id": "table_structure", "label": "Nhận diện cấu trúc bảng bằng NIM", "status": "not_applicable", "output": None},
                {"id": "ocr", "label": "OCR", "status": "not_applicable", "output": None},
                {
                    "id": "clean",
                    "label": "Chuẩn hóa grid / giữ provenance",
                    "status": "observed",
                    "executor": "SpreadsheetExtractActor",
                    "function": "_regions_from_values",
                    "output": {"native_first": True},
                },
                {
                    "id": "explode",
                    "label": "Sinh block Markdown",
                    "status": "observed" if blocks else "not_observed",
                    "executor": "SpreadsheetExtractActor",
                    "function": "_table_markdown",
                    "output": {"block_count": len(blocks), "content_types": sorted({block["content_type"] for block in blocks})},
                },
                {
                    "id": "embedding",
                    "label": "Tạo embedding cho block",
                    "status": "observed" if vectors else "not_observed",
                    "model": config.get("embedding_model") or "Embedding NIM",
                    "function": "embed_text_1b_v2",
                    "endpoint": config.get("embedding_endpoint"),
                    "output": {"embedded_blocks": len(vectors), "dimensions": sorted({item["dim"] for item in vectors})},
                },
            ]
        # Use the exact same canonical bbox policy as the visual sidecar.
        # Without this step the trace endpoint could still expose nested text
        # rows even though /visual had already removed them.
        from nemo_retriever.service.services.visual_evidence import deduplicate_visual_blocks

        blocks = deduplicate_visual_blocks(blocks, page_number)

        pages.append(
            {
                "page_number": page_number,
                "source_id": base_row.get("source_id"),
                "reader_backend": "native_spreadsheet" if is_spreadsheet and native else ("native_pdf" if native else "ocr"),
                "ocr_pipeline": selected_ocr_pipeline,
                "ocr_backend": page_ocr_backend,
                "ocr_timing": (
                    (base_row.get("ocr") or {}).get("timing")
                    if isinstance(base_row.get("ocr"), dict)
                    else metadata.get("ocr_timing")
                ),
                "ocr_route_counts": (
                    ((base_row.get("ocr") or {}).get("timing") or {}).get("route_counts")
                    if isinstance(base_row.get("ocr"), dict)
                    else None
                ),
                "ocr_selected_backend_counts": (
                    ((base_row.get("ocr") or {}).get("timing") or {}).get("selected_backend_counts")
                    if isinstance(base_row.get("ocr"), dict)
                    else None
                ),
                "text_chars": len(text),
                "block_count": len(blocks),
                "content_types": sorted({block["content_type"] for block in blocks}),
                "cleaning": metadata.get("cleaning"),
                "stages": page_stages,
                "blocks": blocks,
            }
        )

    document_diagnostics = getattr(document, "pipeline_diagnostics", None)
    diagnostic_page_count = (
        document_diagnostics.get("page_count")
        if isinstance(document_diagnostics, dict)
        else None
    )
    try:
        diagnostic_page_count = int(diagnostic_page_count) if diagnostic_page_count is not None else None
    except (TypeError, ValueError):
        diagnostic_page_count = None
    retained_rows = len(rows)
    result_rows = document.result_rows if getattr(document, "result_rows", None) is not None else retained_rows

    return {
        "job_id": job.job_id,
        "document_id": document.id,
        "status": document.status.value,
        "pipeline_diagnostics": document_diagnostics,
        "file": {
            "filename": document.filename,
            "extension": extension,
            "classification": "spreadsheet" if is_spreadsheet else ("pdf" if extension == "pdf" else extension or "unknown"),
            "pages": len(pages) or diagnostic_page_count or 0,
            "result_rows": result_rows,
            "pipeline_source": "retained result_data when enabled + persistent document diagnostics",
            "note": "Detailed stage blocks require retained result_data; compact routing/timing diagnostics are persisted even when raw rows are discarded.",
            "stages": [
                {"id": "receive", "label": "Tiếp nhận file", "status": "observed", "executor": "JobTracker.register_document", "output": {"filename": document.filename, "submitted_at": document.submitted_at}},
                {"id": "classify", "label": "Phân loại định dạng", "status": "observed", "executor": "FileClassifier.classify", "output": {"extension": extension, "classification": "spreadsheet" if is_spreadsheet else ("pdf" if extension == "pdf" else extension or "unknown")}},
                {"id": "route", "label": "Định tuyến pipeline", "status": "observed", "executor": "ingest job worker", "output": {"pool": "not_persisted", "route_evidence": "whole-document job result; exact request path is not persisted by the tracker"}},
                {"id": "split", "label": "Tách file thành từng sheet / trang", "status": "observed" if pages or diagnostic_page_count else "not_observed", "executor": "SpreadsheetExtractActor" if is_spreadsheet else "PDFSplitActor", "function": "spreadsheet_bytes_to_chunks_df" if is_spreadsheet else "split_pdf_batch", "library": "openpyxl" if is_spreadsheet and extension in {"xlsx", "xls"} else ("Python csv" if is_spreadsheet else "pypdfium2"), "output": {"page_count": len(pages) or diagnostic_page_count or 0}},
                {"id": "pages", "label": "Xử lý từng trang", "status": "observed" if pages or diagnostic_page_count else "not_observed", "executor": "page pipeline", "output": {"page_count": len(pages) or diagnostic_page_count or 0, "block_count": sum(page["block_count"] for page in pages), "raw_rows_retained": bool(retained_rows)}},
                {"id": "vdb", "label": "Ghi VectorDB", "status": "configured", "executor": "VectorDB sink", "output": {"configured": True, "note": "Job-level ownership/count is not persisted in the current VDB schema"}},
                {"id": "retain", "label": "Giữ result_data", "status": "observed" if retained_rows else ("not_retained" if result_rows else "not_observed"), "executor": "JobTracker / worker result store", "output": {"rows": retained_rows, "result_rows": result_rows, "raw_rows_retained": bool(retained_rows)}},
            ],
        },
        "config": config,
        "pages": pages,
    }


@router.get("/api/jobs/{job_id}/documents/{document_id}/pipeline")
async def document_pipeline_trace(request: Request, job_id: str, document_id: str) -> JSONResponse:
    """Return a page/block pipeline trace backed by retained model outputs."""
    from nemo_retriever.service.services.job_tracker import get_job_tracker

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(503, "Job tracker not available")
    job = tracker.get_job(job_id)
    document = tracker.get_document(document_id)
    if job is None or document is None or document.job_id != job_id:
        raise HTTPException(404, f"Document {document_id!r} not found in job {job_id!r}")
    rows = tracker.get_result_data(document_id) or []
    return JSONResponse(_build_pipeline_trace(request, job=job, document=document, rows=rows))


def _visual_evidence_for_document(tracker: Any, document_id: str) -> dict[str, Any] | None:
    """Read the compact sidecar, with a compatibility fallback for old jobs."""
    from nemo_retriever.service.services.worker_result_store import (
        ResultStoreTemporarilyUnavailable,
        get_result_data,
        get_visual_evidence,
        store_visual_evidence,
    )
    from nemo_retriever.service.services.visual_evidence import build_visual_evidence, deduplicate_visual_evidence

    try:
        evidence = get_visual_evidence(document_id)
    except ResultStoreTemporarilyUnavailable:
        raise
    if evidence is not None:
        # v2 projects the private per-line OCR records into separate visual
        # blocks. Rebuild an older sidecar when its source rows are still
        # available; otherwise retain the old evidence rather than returning
        # an empty inspector.
        if evidence.get("schema_version") != "visual-evidence-v2":
            rows = tracker.get_result_data(document_id) or []
            if not rows:
                rows = get_result_data(document_id) or []
            if rows:
                rebuilt = build_visual_evidence(rows)
                if rebuilt.get("pages"):
                    try:
                        store_visual_evidence(document_id, rebuilt)
                    except (OSError, TypeError, ValueError):
                        logger.debug("Unable to migrate visual evidence for %s", document_id, exc_info=True)
                    return rebuilt
        cleaned = deduplicate_visual_evidence(evidence)
        # Existing jobs may contain the pre-fuzzy-dedup sidecar. Return the
        # cleaned projection immediately and persist it so later page/image
        # requests do not repeat the same work.
        try:
            previous_count = int(evidence.get("block_count") or 0)
            cleaned_count = int(cleaned.get("block_count") or 0)
        except (AttributeError, TypeError, ValueError):
            previous_count = cleaned_count = 0
        if cleaned_count < previous_count:
            try:
                store_visual_evidence(document_id, cleaned)
            except (OSError, TypeError, ValueError):
                logger.debug("Unable to persist cleaned visual evidence for %s", document_id, exc_info=True)
        return cleaned

    # Jobs created before the sidecar existed can still be inspected when
    # they retained image-bearing result rows. Prefer the tracker, then the
    # shared worker-result store used by gateway mode.
    rows = tracker.get_result_data(document_id) or []
    if not rows:
        rows = get_result_data(document_id) or []
    if not rows:
        return None
    evidence = build_visual_evidence(rows)
    if evidence.get("pages"):
        try:
            store_visual_evidence(document_id, evidence)
        except (OSError, TypeError, ValueError):
            logger.debug("Unable to cache compatibility visual evidence for %s", document_id, exc_info=True)
    return evidence


@router.get("/api/jobs/{job_id}/documents/{document_id}/visual")
async def document_visual_evidence(request: Request, job_id: str, document_id: str) -> JSONResponse:
    """Return page geometry/text metadata without embedding page rasters."""
    from nemo_retriever.service.services.job_tracker import get_job_tracker
    from nemo_retriever.service.services.worker_result_store import ResultStoreTemporarilyUnavailable
    from nemo_retriever.service.services.visual_evidence import manifest_without_images

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(503, "Job tracker not available")
    job = tracker.get_job(job_id)
    document = tracker.get_document(document_id)
    if job is None or document is None or document.job_id != job_id:
        raise HTTPException(404, f"Document {document_id!r} not found in job {job_id!r}")
    try:
        evidence = _visual_evidence_for_document(tracker, document_id)
    except ResultStoreTemporarilyUnavailable as exc:
        raise HTTPException(503, str(exc), headers={"Retry-After": "1"}) from exc
    if evidence is None:
        return JSONResponse(
            {
                "job_id": job_id,
                "document_id": document_id,
                "status": document.status.value,
                "available": False,
                "pages": [],
                "page_count": 0,
                "block_count": 0,
            }
        )
    return JSONResponse(
        {
            "job_id": job_id,
            "document_id": document_id,
            "status": document.status.value,
            "available": True,
            "image_endpoint": f"/v1/dashboard/api/jobs/{job_id}/documents/{document_id}/visual/pages/{{page_number}}/image",
            "block_image_endpoint": f"/v1/dashboard/api/jobs/{job_id}/documents/{document_id}/visual/pages/{{page_number}}/blocks/{{block_id}}/image",
            **manifest_without_images(evidence),
        }
    )


@router.get("/api/jobs/{job_id}/documents/{document_id}/visual/pages/{page_number}/image")
async def document_visual_page_image(request: Request, job_id: str, document_id: str, page_number: int) -> Response:
    """Serve one retained page raster for the visual inspector."""
    from nemo_retriever.service.services.job_tracker import get_job_tracker
    from nemo_retriever.service.services.worker_result_store import ResultStoreTemporarilyUnavailable
    from nemo_retriever.service.services.visual_evidence import page_image_payload

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(503, "Job tracker not available")
    job = tracker.get_job(job_id)
    document = tracker.get_document(document_id)
    if job is None or document is None or document.job_id != job_id:
        raise HTTPException(404, f"Document {document_id!r} not found in job {job_id!r}")
    try:
        evidence = _visual_evidence_for_document(tracker, document_id)
    except ResultStoreTemporarilyUnavailable as exc:
        raise HTTPException(503, str(exc), headers={"Retry-After": "1"}) from exc
    if evidence is None:
        raise HTTPException(404, f"No visual evidence for document {document_id!r}")
    image = page_image_payload(evidence, page_number)
    if image is None:
        raise HTTPException(404, f"No retained page image for page {page_number}")
    mime, image_b64 = image
    try:
        content = base64.b64decode(image_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(503, f"Retained page image for page {page_number} is invalid") from exc
    return Response(
        content=content,
        media_type=mime,
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/api/jobs/{job_id}/documents/{document_id}/visual/pages/{page_number}/blocks/{block_id}/image")
async def document_visual_block_image(
    request: Request,
    job_id: str,
    document_id: str,
    page_number: int,
    block_id: str,
) -> Response:
    """Serve a crop for a visual block from the retained page raster.

    Visual evidence intentionally keeps the page image once and stores block
    geometry in the manifest. Cropping here avoids duplicating large base64
    payloads in every block while still letting the dashboard inspect images,
    charts, stamps, and other detector regions.
    """
    from PIL import Image

    from nemo_retriever.service.services.job_tracker import get_job_tracker
    from nemo_retriever.service.services.worker_result_store import ResultStoreTemporarilyUnavailable
    from nemo_retriever.service.services.visual_evidence import page_image_payload

    tracker = get_job_tracker()
    if tracker is None:
        raise HTTPException(503, "Job tracker not available")
    job = tracker.get_job(job_id)
    document = tracker.get_document(document_id)
    if job is None or document is None or document.job_id != job_id:
        raise HTTPException(404, f"Document {document_id!r} not found in job {job_id!r}")
    try:
        evidence = _visual_evidence_for_document(tracker, document_id)
    except ResultStoreTemporarilyUnavailable as exc:
        raise HTTPException(503, str(exc), headers={"Retry-After": "1"}) from exc
    if evidence is None:
        raise HTTPException(404, f"No visual evidence for document {document_id!r}")

    page = None
    for item in evidence.get("pages") or []:
        if not isinstance(item, dict):
            continue
        try:
            if int(item.get("page_number", -1)) == int(page_number):
                page = item
                break
        except (TypeError, ValueError):
            continue
    block = next(
        (
            item for item in (page or {}).get("blocks") or []
            if isinstance(item, dict) and str(item.get("id")) == str(block_id)
        ),
        None,
    )
    bbox = block.get("bbox") if block else None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise HTTPException(404, f"Block {block_id!r} has no crop bbox")

    image = page_image_payload(evidence, page_number)
    if image is None:
        raise HTTPException(404, f"No retained page image for page {page_number}")
    _mime, image_b64 = image
    try:
        raw = base64.b64decode(image_b64, validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            source = source.convert("RGB")
            width, height = source.size
            x1, y1, x2, y2 = [max(0.0, min(1.0, float(value))) for value in bbox]
            left = max(0, min(width - 1, int(round(x1 * width))))
            top = max(0, min(height - 1, int(round(y1 * height))))
            right = max(left + 1, min(width, int(round(x2 * width))))
            bottom = max(top + 1, min(height, int(round(y2 * height))))
            # A tiny margin makes thin rules and detector edges inspectable.
            margin = max(2, round(min(width, height) * 0.002))
            left = max(0, left - margin)
            top = max(0, top - margin)
            right = min(width, right + margin)
            bottom = min(height, bottom + margin)
            crop = source.crop((left, top, right, bottom))
            output = io.BytesIO()
            crop.save(output, format="JPEG", quality=95)
            content = output.getvalue()
    except (ValueError, TypeError, OSError) as exc:
        raise HTTPException(503, f"Unable to crop visual block {block_id!r}") from exc

    return Response(
        content=content,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


# ── VDB tables ───────────────────────────────────────────────────────


@router.get("/api/vdb/tables")
async def vdb_tables(request: Request) -> JSONResponse:
    config = request.app.state.config
    vdb_cfg = getattr(config, "vectordb", None)

    if not vdb_cfg or not getattr(vdb_cfg, "enabled", False):
        return JSONResponse({"error": "VectorDB not enabled", "tables": []})

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{vdb_cfg.vectordb_url}/v1/health")
            resp.raise_for_status()
            health = resp.json()
            return JSONResponse(
                {
                    "tables": [
                        {
                            "name": health.get("table", ""),
                            "total_rows": health.get("total_rows", 0),
                            "exists": health.get("table_exists", False),
                        }
                    ],
                }
            )
    except Exception as exc:
        return JSONResponse({"error": str(exc), "tables": []})


# ── VDB query proxy ──────────────────────────────────────────────────


@router.post("/api/vdb/query")
async def vdb_query(req: VdbQueryRequest, request: Request) -> JSONResponse:
    config = request.app.state.config
    vdb_cfg = getattr(config, "vectordb", None)

    if not vdb_cfg or not getattr(vdb_cfg, "enabled", False):
        raise HTTPException(501, "VectorDB not enabled")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{vdb_cfg.vectordb_url}/v1/query",
                json={"query": req.query, "top_k": req.top_k},
            )
            resp.raise_for_status()
            return JSONResponse(resp.json())
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text[:500])
    except Exception as exc:
        raise HTTPException(502, f"VDB query failed: {exc}")
