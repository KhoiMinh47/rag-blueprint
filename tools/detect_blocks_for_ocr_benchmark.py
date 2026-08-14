"""Generate detector-produced OCR blocks and attach available ground truth.

This is a benchmark harness only.  PP-OCRv6 text detection produces the
blocks; the recognizer output is retained for comparison with Tesseract on
the exact same crops.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from paddleocr import PaddleOCR


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ub = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1, ua + ub - inter)


def attach_reference(pred_box, refs):
    selected = []
    for ref in refs:
        x1, y1, x2, y2 = ref["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        px1, py1, px2, py2 = pred_box
        if px1 <= cx <= px2 and py1 <= cy <= py2 or iou(pred_box, ref["bbox"]) >= 0.15:
            selected.append(ref)
    selected.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return " ".join(x["text"] for x in selected).strip()


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


def run(args):
    pages = []
    vi_dir = args.vintext_root / "vintext/vintext/vintext/test_image"
    for path in sorted(vi_dir.glob("im*.jpg"), key=lambda p: int(p.stem[2:])):
        pages.append(("vi", path, vintext_refs(path)))
    for path in sorted(args.funsd_images.glob("*.png")):
        if path.stem not in args.funsd_ids:
            continue
        pages.append(("en", path, funsd_refs(path, args.funsd_annotations)))

    outputs = {"benchmark": {"detector": "PP-OCRv6 text detector", "pages": len(pages)}, "pages": {}}
    models = {
        lang: PaddleOCR(ocr_version="PP-OCRv6", lang=lang, device="gpu:0", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
        for lang in {page[0] for page in pages}
    }
    for lang, path, refs in pages:
        result = next(iter(models[lang].predict(str(path)))).json["res"]
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        boxes = result.get("rec_boxes") or []
        texts = result.get("rec_texts") or []
        scores = result.get("rec_scores") or []
        detections = []
        for box, text, score in zip(boxes, texts, scores):
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            pred_box = [x1, y1, x2, y2]
            detections.append({"bbox": pred_box, "text": str(text), "score": float(score), "reference": attach_reference(pred_box, refs)})
        gt_matched = sum(any(iou(d["bbox"], r["bbox"]) >= 0.3 for d in detections) for r in refs)
        outputs["pages"][path.stem] = {"lang": lang, "image": str(path), "width": width, "height": height, "gt_count": len(refs), "gt_matched": gt_matched, "detections": detections}
        print(f"detected {len(outputs['pages'])}/{len(pages)} {path.name} blocks={len(detections)}", flush=True)
    args.out.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")


parser = argparse.ArgumentParser()
parser.add_argument("--vintext-root", type=Path, required=True)
parser.add_argument("--funsd-images", type=Path, required=True)
parser.add_argument("--funsd-annotations", type=Path, required=True)
parser.add_argument("--funsd-ids", nargs="+", required=True)
parser.add_argument("--out", type=Path, required=True)
run(parser.parse_args())
