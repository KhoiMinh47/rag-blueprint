"""Compare recognizers on identical detector-produced OCR blocks.

The detector has already produced the boxes and PP-OCRv6 has recognized them.
This script crops those exact boxes and sends the same pixels to Tesseract.
Metrics are reported separately from detector coverage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps


def norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value or "").split()).strip()


def strip_marks(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", value) if unicodedata.category(c) != "Mn").replace("đ", "d").replace("Đ", "D")


def distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def metric(predictions: list[str], references: list[str]) -> dict:
    char_errors = sum(distance(p, r) for p, r in zip(predictions, references))
    char_total = sum(len(r) for r in references)
    word_errors = sum(distance(p.split(), r.split()) for p, r in zip(predictions, references))
    word_total = sum(len(r.split()) for r in references)
    return {"blocks": len(references), "cer": char_errors / max(1, char_total), "wer": word_errors / max(1, word_total), "char_errors": char_errors, "char_total": char_total, "word_errors": word_errors, "word_total": word_total}


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, by2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ub = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1, ua + ub - inter)


def vintext_refs(image_path: Path) -> list[dict]:
    label = image_path.parents[1] / "labels" / f"gt_{int(image_path.stem[2:])}.txt"
    refs = []
    for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines()):
        fields = line.split(",")
        if len(fields) < 9 or fields[8] == "###":
            continue
        points = [int(v) for v in fields[:8]]
        refs.append({"line": line_no, "text": ",".join(fields[8:]), "bbox": [min(points[0::2]), min(points[1::2]), max(points[0::2]), max(points[1::2])]})
    return refs


def funsd_refs(image_path: Path, annotation_dir: Path) -> list[dict]:
    data = json.loads((annotation_dir / f"{image_path.stem}.json").read_text(encoding="utf-8"))
    refs = []
    for line_no, line in enumerate(data["pages"][0].get("lines", [])):
        points = line["polygon"]
        refs.append({"line": line_no, "text": line["content"], "bbox": [min(p["x"] for p in points), min(p["y"] for p in points), max(p["x"] for p in points), max(p["y"] for p in points)]})
    return refs


def references_for_page(page: dict, annotation_dir: Path | None) -> list[dict]:
    image = Path(page["image"])
    if page["lang"] == "vi":
        return vintext_refs(image)
    return funsd_refs(image, annotation_dir)


def ref_for_detection(box, refs) -> str:
    x1, y1, x2, y2 = box
    selected = []
    for ref in refs:
        rx1, ry1, rx2, ry2 = ref["bbox"]
        cx, cy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2 or iou(box, ref["bbox"]) >= 0.15:
            selected.append(ref)
    selected.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return norm(" ".join(ref["text"] for ref in selected))


def center_hit(ref, boxes) -> bool:
    x1, y1, x2, y2 = ref["bbox"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return any(box[0] <= cx <= box[2] and box[1] <= cy <= box[3] for box in boxes)


def tesseract_one(job: tuple[str, list[int], str, str, int, int, int]) -> tuple[str, str]:
    image_path, box, lang, tessdata, pad, scale, border = job
    with Image.open(image_path) as image:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(image.width, x2 + pad), min(image.height, y2 + pad)
        crop = image.crop((x1, y1, x2, y2)).convert("L")
        if scale != 1:
            crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
        if border:
            crop = ImageOps.expand(crop, border=border, fill=255)
        # stdin avoids creating thousands of temporary crop files.
        import io
        payload = io.BytesIO()
        crop.save(payload, format="PNG")
    command = ["tesseract", "--tessdata-dir", tessdata, "stdin", "stdout", "--psm", "7", "--oem", "1", "-l", lang, "-c", "preserve_interword_spaces=1"]
    result = subprocess.run(command, input=payload.getvalue(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30, check=False)
    return norm(result.stdout.decode("utf-8", errors="replace")), lang


def run(args):
    source = json.loads(args.detected.read_text(encoding="utf-8"))
    pages = source["pages"]
    by_lang = defaultdict(lambda: {"refs": [], "pp": [], "tess": [], "all_refs": 0, "hit_refs": 0, "detections": 0, "matched_detections": 0})
    jobs = []
    job_meta = []
    for page in pages.values():
        refs = references_for_page(page, args.funsd_annotations)
        boxes = [d["bbox"] for d in page["detections"]]
        bucket = by_lang[page["lang"]]
        bucket["all_refs"] += len(refs)
        bucket["hit_refs"] += sum(center_hit(ref, boxes) for ref in refs)
        bucket["detections"] += len(boxes)
        for detection in page["detections"]:
            reference = ref_for_detection(detection["bbox"], refs)
            if not reference:
                continue
            bucket["matched_detections"] += 1
            bucket["refs"].append(reference)
            bucket["pp"].append(norm(detection["text"]))
            jobs.append((page["image"], detection["bbox"], "vie" if page["lang"] == "vi" else "eng", str(args.tessdata), args.pad, args.scale, args.border))
            job_meta.append(page["lang"])

    print(f"recognizing {len(jobs)} identical detector crops with tesseract", flush=True)
    tess = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(tesseract_one, job): i for i, job in enumerate(jobs)}
        for n, future in enumerate(futures, 1):
            i = futures[future]
            try:
                tess[i] = future.result()[0]
            except Exception as exc:
                tess[i] = ""
                print(f"tesseract failed at {i}: {exc}", flush=True)
            if n % 250 == 0:
                print(f"recognized {n}/{len(jobs)}", flush=True)

    offset = 0
    output = {"source": str(args.detected), "protocol": "PP-OCRv6 text detector -> identical predicted bbox crop -> PP-OCRv6 vs Tesseract", "languages": {}}
    for lang, bucket in by_lang.items():
        start = offset
        end = start + len(bucket["refs"])
        tess_predictions = tess[start:end]
        raw = {"pp_ocrv6": metric(bucket["pp"], bucket["refs"]), "tesseract": metric(tess_predictions, bucket["refs"])}
        nodiac_refs = [strip_marks(x) for x in bucket["refs"]]
        nodiac_pp = [strip_marks(x) for x in bucket["pp"]]
        nodiac_tess = [strip_marks(x) for x in tess_predictions]
        output["languages"][lang] = {"detector": {"pages": sum(p["lang"] == lang for p in pages.values()), "detections": bucket["detections"], "detections_with_reference": bucket["matched_detections"], "gt_blocks": bucket["all_refs"], "gt_blocks_hit_by_predicted_bbox_center": bucket["hit_refs"], "gt_center_coverage": bucket["hit_refs"] / max(1, bucket["all_refs"])}, "raw_unicode": raw, "without_diacritics": {"pp_ocrv6": metric(nodiac_pp, nodiac_refs), "tesseract": metric(nodiac_tess, nodiac_refs)}}
        offset = end
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["languages"], ensure_ascii=False, indent=2), flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("--detected", type=Path, required=True)
parser.add_argument("--funsd-annotations", type=Path, required=True)
parser.add_argument("--tessdata", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--pad", type=int, default=3)
parser.add_argument("--scale", type=int, default=3)
parser.add_argument("--border", type=int, default=12)
run(parser.parse_args())
