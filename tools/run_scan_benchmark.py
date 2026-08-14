"""Run the prepared scanned-PDF benchmark through the live API."""

from __future__ import annotations

import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "test-pdfs"


def iou(a: list[float], b: list[float]) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def counts(gt: list[dict[str, Any]], pred: list[dict[str, Any]], threshold: float) -> dict[str, int]:
    candidates = []
    for gi, expected in enumerate(gt):
        for pi, actual in enumerate(pred):
            if expected["label"] == actual["label"]:
                score = iou(expected["bbox_xyxy_norm"], actual["bbox_xyxy_norm"])
                if score >= threshold:
                    candidates.append((score, gi, pi))
    used_gt, used_pred = set(), set()
    tp = 0
    for _score, gi, pi in sorted(candidates, reverse=True):
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        tp += 1
    return {"tp": tp, "fp": len(pred) - tp, "fn": len(gt) - tp}


def metric(value: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn = value["tp"], value["fp"], value["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**value, "precision": precision, "recall": recall, "f1": f1}


def vram() -> tuple[int, int]:
    try:
        line = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip().splitlines()[0]
        used, total = [int(x.strip()) for x in line.split(",")]
        return used, total
    except Exception:
        return 0, 0


def predictions(trace: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for page in trace.get("pages", []):
        for block in page.get("blocks", []):
            bbox = block.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            if block.get("reader_backend") == "native_pdf_image":
                continue
            content = str(block.get("content_type") or "text")
            label = {"table": "table", "chart": "visual", "infographic": "visual", "image": "visual", "title": "title"}.get(content, "text")
            result.append({"label": label, "content_type": content, "bbox_xyxy_norm": [float(x) for x in bbox], "page": page.get("page_number"), "block_id": block.get("block_id")})
    return result


def detector_predictions(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the Page Elements boxes before OCR/content-row expansion."""
    result: list[dict[str, Any]] = []
    seen_pages: set[Any] = set()
    for row in document.get("result_data") or []:
        if not isinstance(row, dict):
            continue
        page = row.get("page_number")
        # Result rows are exploded content blocks, so the page-level detector
        # payload is repeated on every row. Evaluate it once per page.
        if page in seen_pages:
            continue
        seen_pages.add(page)
        payload = row.get("page_elements_v3") or {}
        if not isinstance(payload, dict):
            continue
        for detection in payload.get("detections") or []:
            if not isinstance(detection, dict):
                continue
            bbox = detection.get("bbox_xyxy_norm")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            name = str(detection.get("label_name") or "text")
            label = {"table": "table", "chart": "visual", "infographic": "visual", "image": "visual"}.get(name, name)
            result.append(
                {
                    "label": label,
                    "content_type": name,
                    "bbox_xyxy_norm": [float(x) for x in bbox],
                    "page": page,
                    "score": detection.get("score"),
                }
            )
    return result


def overlay(record: dict[str, Any], pred: list[dict[str, Any]], path: Path) -> None:
    image = Image.open(ROOT / record["preview_image"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for item in record.get("ground_truth", []):
        box = item["bbox_xyxy_norm"]
        rect = (box[0] * width, box[1] * height, box[2] * width, box[3] * height)
        draw.rectangle(rect, outline=(30, 180, 50), width=3)
        draw.text((rect[0] + 2, rect[1] + 2), "GT:" + item["label"], fill=(30, 180, 50))
    for item in pred:
        box = item["bbox_xyxy_norm"]
        rect = (box[0] * width, box[1] * height, box[2] * width, box[3] * height)
        draw.rectangle(rect, outline=(230, 40, 40), width=2)
        draw.text((rect[0] + 2, max(0, rect[1] - 12)), "P:" + item["label"], fill=(230, 40, 40))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    manifest = json.loads((CACHE / "manifest.json").read_text())
    result_dir = CACHE / "results"
    with httpx.Client(base_url="http://127.0.0.1:7780", timeout=httpx.Timeout(180.0, connect=20.0)) as client:
        response = client.post("/v1/ingest/job", json={"expected_documents": len(manifest), "retain_results": True, "label": "scan benchmark"})
        response.raise_for_status()
        job_id = response.json()["job_id"]
        documents = []
        peak_used = peak_total = 0
        for record in manifest:
            pdf = ROOT / record["pdf"]
            with pdf.open("rb") as handle:
                response = client.post(
                    f"/v1/ingest/job/{job_id}/whole",
                    files={"file": (pdf.name, handle, "application/pdf")},
                    data={
                        "metadata": json.dumps(
                            {
                                "filename": pdf.name,
                                "metadata": {"ocr_pipeline": "pipeline-option6"},
                                "pipeline": {
                                    "ocr_pipeline": "pipeline-option6",
                                    "return_images": True,
                                    "return_embeddings": False,
                                },
                            }
                        )
                    },
                )
            response.raise_for_status()
            documents.append((record, response.json()["document_id"]))
            used, total = vram()
            peak_used, peak_total = max(peak_used, used), max(peak_total, total)

        while True:
            status_response = client.get(f"/v1/ingest/job/{job_id}")
            status_response.raise_for_status()
            status = status_response.json()
            used, total = vram()
            peak_used, peak_total = max(peak_used, used), max(peak_total, total)
            if total and used / total >= 0.97:
                raise RuntimeError(f"VRAM safety limit reached: {used}/{total} MiB")
            if status.get("status") in {"completed", "failed", "partial_success"}:
                break
            time.sleep(5)

        rows = []
        aggregate = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for record, document_id in documents:
            doc = client.get(f"/v1/ingest/job/{job_id}/document/{document_id}")
            doc.raise_for_status()
            document = doc.json()
            trace = client.get(f"/v1/dashboard/api/jobs/{job_id}/documents/{document_id}/pipeline")
            trace.raise_for_status()
            trace_payload = trace.json()
            trace_pages = trace_payload.get("pages") or []
            actual_pipelines = {
                str(page.get("ocr_pipeline") or "")
                for page in trace_pages
                if isinstance(page, dict)
            }
            if actual_pipelines != {"pipeline-option6"}:
                raise RuntimeError(
                    "Pipeline 6 benchmark selector mismatch: expected only "
                    "pipeline-option6, got "
                    f"{sorted(actual_pipelines)!r} for {record['id']}"
                )
            pred = predictions(trace_payload)
            detector_pred = detector_predictions(document)
            gt = record.get("ground_truth", [])
            detector_visual = [item for item in detector_pred if item["label"] == "visual"]
            ground_truth_visual = [item for item in gt if item["label"] == "visual"]
            row = {"id": record["id"], "document_id": document_id, "source": record["source"], "status": document.get("status"), "result_rows": len(document.get("result_data") or []), "trace_pages": len(trace_payload.get("pages") or []), "ground_truth": len(gt), "predictions": len(pred), "detector_predictions": len(detector_pred), "ground_truth_visual": len(ground_truth_visual), "detector_visual_predictions": len(detector_visual), "images_retained": sum(bool(image.get("image_b64")) for item in document.get("result_data") or [] for image in item.get("images") or [] if isinstance(image, dict))}
            for threshold in (0.5, 0.75):
                value = counts(gt, pred, threshold)
                row[f"iou_{threshold}"] = metric(value)
                for key, number in value.items():
                    aggregate[f"{record['source']}:iou_{threshold}"][key] += number
                detector_value = counts(gt, detector_pred, threshold)
                for key, number in detector_value.items():
                    aggregate[f"{record['source']}:detector_iou_{threshold}"][key] += number
                for label in sorted({item["label"] for item in gt} | {item["label"] for item in detector_pred}):
                    label_value = counts(
                        [item for item in gt if item["label"] == label],
                        [item for item in detector_pred if item["label"] == label],
                        threshold,
                    )
                    for key, number in label_value.items():
                        aggregate[f"{record['source']}:detector:{label}:iou_{threshold}"][key] += number
            overlay(record, pred, result_dir / "overlays" / f"{record['id']}.png")
            overlay(record, detector_pred, result_dir / "detector_overlays" / f"{record['id']}.png")
            rows.append(row)

        summary = {"job_id": job_id, "status": status.get("status"), "documents": len(rows), "completed_documents": sum(x["status"] == "completed" for x in rows), "failed_documents": sum(x["status"] == "failed" for x in rows), "peak_vram_mib": peak_used, "vram_total_mib": peak_total, "vram_ratio": peak_used / peak_total if peak_total else None, "aggregate": {key: metric(value) for key, value in aggregate.items()}}
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "benchmark_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        (result_dir / "benchmark_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
