"""Prepare a small, reproducible scanned-PDF benchmark in cache/test-pdfs.

The source archives are never unpacked wholesale. DocLayNet is read through
HTTP range requests so only the COCO validation metadata and selected PNG
members are downloaded.
"""

from __future__ import annotations

import argparse
import json
import struct
import urllib.request
import zlib
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "test-pdfs"
DOC_ZIP_URL = "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip"
DOC_ZIP_SIZE = 30_012_083_650
FUNSD_URL = "https://www.crc.nd.edu/~pmoreira/funsd.zip"


def fetch_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def load_doclaynet_entries() -> dict[str, tuple[int, int, int, int]]:
    tail_path = CACHE / "source" / "doclaynet_tail.bin"
    tail_path.parent.mkdir(parents=True, exist_ok=True)
    if not tail_path.exists() or tail_path.stat().st_size < 25_000_000:
        tail_path.write_bytes(fetch_range(DOC_ZIP_URL, DOC_ZIP_SIZE - 25_000_000, DOC_ZIP_SIZE - 1))
    tail = tail_path.read_bytes()
    central_offset = 29_999_021_762 - (DOC_ZIP_SIZE - len(tail))
    entries: dict[str, tuple[int, int, int, int]] = {}
    position = central_offset
    while position + 46 <= len(tail) and tail[position : position + 4] == b"PK\x01\x02":
        values = struct.unpack_from("<4s6H3L5H2L", tail, position)
        _, _, _, _, method, _, _, _, compressed_size, size, name_len, extra_len, comment_len, _, _, _, offset = values
        name = tail[position + 46 : position + 46 + name_len].decode("utf-8", "replace")
        extra = tail[position + 46 + name_len : position + 46 + name_len + extra_len]
        if size == 0xFFFFFFFF or compressed_size == 0xFFFFFFFF or offset == 0xFFFFFFFF:
            cursor = 0
            while cursor + 4 <= len(extra):
                field_id, field_size = struct.unpack_from("<HH", extra, cursor)
                field = extra[cursor + 4 : cursor + 4 + field_size]
                cursor += 4 + field_size
                if field_id != 0x0001:
                    continue
                field_cursor = 0
                if size == 0xFFFFFFFF:
                    size = struct.unpack_from("<Q", field, field_cursor)[0]
                    field_cursor += 8
                if compressed_size == 0xFFFFFFFF:
                    compressed_size = struct.unpack_from("<Q", field, field_cursor)[0]
                    field_cursor += 8
                if offset == 0xFFFFFFFF:
                    offset = struct.unpack_from("<Q", field, field_cursor)[0]
                break
        entries[name] = (method, compressed_size, size, offset)
        position += 46 + name_len + extra_len + comment_len
    if "COCO/val.json" not in entries:
        raise RuntimeError("DocLayNet central directory did not contain COCO/val.json")
    return entries


def read_zip_member(entries: dict[str, tuple[int, int, int, int]], name: str) -> bytes:
    method, compressed_size, _size, offset = entries[name]
    local = fetch_range(DOC_ZIP_URL, offset, offset + 30 + 4096)
    name_len, extra_len = struct.unpack_from("<HH", local, 26)
    data_start = offset + 30 + name_len + extra_len
    raw = fetch_range(DOC_ZIP_URL, data_start, data_start + compressed_size - 1)
    return zlib.decompress(raw, -15) if method == 8 else raw


def save_scan_pdf(image_path: Path, pdf_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(pdf_path, format="PDF", resolution=150.0)


def normalize_coco_bbox(bbox: list[float], width: float, height: float) -> list[float]:
    x, y, w, h = bbox
    return [x / width, y / height, (x + w) / width, (y + h) / height]


def prepare_doclaynet(count: int) -> list[dict[str, Any]]:
    entries = load_doclaynet_entries()
    coco_path = CACHE / "ground_truth" / "doclaynet_val.json"
    coco_path.parent.mkdir(parents=True, exist_ok=True)
    if not coco_path.exists():
        coco_path.write_bytes(read_zip_member(entries, "COCO/val.json"))
    coco = json.loads(coco_path.read_text())
    categories = {item["id"]: item["name"] for item in coco["categories"]}
    annotations: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco["annotations"]:
        annotations.setdefault(annotation["image_id"], []).append(annotation)

    selected = []
    for image in coco["images"]:
        labels = {categories[a["category_id"]] for a in annotations.get(image["id"], [])}
        if {"Picture", "Table"}.issubset(labels):
            selected.append(image)
        if len(selected) >= count:
            break

    records: list[dict[str, Any]] = []
    for index, image in enumerate(selected, start=1):
        member = f"PNG/{image['file_name']}"
        image_path = CACHE / "source" / "doclaynet" / image["file_name"]
        if not image_path.exists():
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(read_zip_member(entries, member))
        pdf_path = CACHE / "pdfs" / f"doclaynet_{index:02d}.pdf"
        save_scan_pdf(image_path, pdf_path)
        gt = []
        for annotation in annotations.get(image["id"], []):
            name = categories[annotation["category_id"]]
            if name not in {"Picture", "Table", "Text", "Title"}:
                continue
            gt.append(
                {
                    "label": {"Picture": "visual", "Table": "table", "Text": "text", "Title": "title"}[name],
                    "source_label": name,
                    "bbox_xyxy_norm": normalize_coco_bbox(annotation["bbox"], image["width"], image["height"]),
                }
            )
        records.append(
            {
                "id": f"doclaynet_{index:02d}",
                "source": "DocLayNet",
                "source_url": DOC_ZIP_URL,
                "pdf": str(pdf_path.relative_to(ROOT)),
                "preview_image": str(image_path.relative_to(ROOT)),
                "ground_truth": gt,
            }
        )
    return records


def prepare_funsd(count: int) -> list[dict[str, Any]]:
    root = CACHE / "source" / "funsd" / "FUNSD" / "testing_data"
    images = sorted((root / "images").glob("*.png"))[:count]
    records: list[dict[str, Any]] = []
    for index, image_path in enumerate(images, start=1):
        annotation_path = root / "annotations" / f"{image_path.stem}.json"
        annotation = json.loads(annotation_path.read_text())
        with Image.open(image_path) as image:
            width, height = image.size
        gt = []
        for entity in annotation.get("form", []):
            x0, y0, x1, y1 = entity.get("box", [0, 0, 0, 0])
            if x1 <= x0 or y1 <= y0:
                continue
            gt.append(
                {
                    "label": "text",
                    "source_label": entity.get("label", "other"),
                    "bbox_xyxy_norm": [x0 / width, y0 / height, x1 / width, y1 / height],
                }
            )
        pdf_path = CACHE / "pdfs" / f"funsd_{index:02d}.pdf"
        save_scan_pdf(image_path, pdf_path)
        records.append(
            {
                "id": f"funsd_{index:02d}",
                "source": "FUNSD",
                "source_url": FUNSD_URL,
                "pdf": str(pdf_path.relative_to(ROOT)),
                "preview_image": str(image_path.relative_to(ROOT)),
                "ground_truth": gt,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doclaynet", type=int, default=12)
    parser.add_argument("--funsd", type=int, default=8)
    args = parser.parse_args()
    records = prepare_doclaynet(args.doclaynet) + prepare_funsd(args.funsd)
    manifest = CACHE / "manifest.json"
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"files": len(records), "manifest": str(manifest), "pdfs": str(CACHE / "pdfs")}, indent=2))


if __name__ == "__main__":
    main()
