"""Recognition-only benchmark for Tesseract and VietOCR.

The detector is held constant: line boxes come from the previously captured
PP-OCRv6 result.  This deliberately does not call or modify the ingest code.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(reference: str, prediction: str) -> float:
    reference, prediction = clean(reference).lower(), clean(prediction).lower()
    return round(levenshtein(reference, prediction) / max(1, len(reference)), 5)


def crop_for_box(image: Image.Image, box: list[float]) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = [float(v) for v in box]
    # Paddle benchmark boxes are normalized.  Add a small margin for accents.
    x1, x2 = int(max(0, x1 * width - 4)), int(min(width, x2 * width + 4))
    y1, y2 = int(max(0, y1 * height - 4)), int(min(height, y2 * height + 4))
    if x2 <= x1 or y2 <= y1:
        return Image.new("RGB", (8, 8), "white")
    return image.crop((x1, y1, x2, y2)).convert("RGB")


def tesseract_one(image: Image.Image, lang: str, tessdata_dir: str | None) -> tuple[str, float | None]:
    # PPM avoids temporary image files and is accepted by the tesseract CLI.
    command = ["tesseract", "stdin", "stdout", "--psm", "7", "--oem", "1", "-l", lang]
    if tessdata_dir:
        command[1:1] = ["--tessdata-dir", tessdata_dir]
    command += ["-c", "preserve_interword_spaces=1"]
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        input=_ppm(image),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    text = proc.stdout.decode("utf-8", "replace").strip()
    if proc.returncode:
        text = ""
    return text, round(time.perf_counter() - started, 5)


def _ppm(image: Image.Image) -> bytes:
    import io

    output = io.BytesIO()
    image.save(output, format="PPM")
    return output.getvalue()


def vietocr_predictor():
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor

    cfg = Cfg.load_config_from_name("vgg_transformer")
    cfg["device"] = "cuda:0"
    cfg["predictor"]["beamsearch"] = False
    return Predictor(cfg)


def vietocr_one(predictor: Any, image: Image.Image) -> tuple[str, float]:
    started = time.perf_counter()
    value = predictor.predict(image)
    return clean(str(value)), round(time.perf_counter() - started, 5)


def vietocr_batch(predictor: Any, images: list[Image.Image]) -> list[dict[str, Any]]:
    if not images:
        return []
    started = time.perf_counter()
    values = predictor.predict_batch(images)
    elapsed = round((time.perf_counter() - started) / len(images), 5)
    return [{"text": clean(str(value)), "elapsed_s": elapsed} for value in values]


def funsd_reference(image_path: Path, corpus_root: Path) -> str | None:
    annotation_dir = corpus_root / "source/funsd/FUNSD/testing_data/annotations_azure_model__prebuilt_read"
    matches = list(annotation_dir.glob(f"{image_path.stem}.json"))
    if not matches:
        return None
    return str(json.loads(matches[0].read_text(encoding="utf-8")).get("content") or "")


def run(args: argparse.Namespace) -> None:
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    raw = benchmark["raw"]
    manifest = {item["id"]: item["preview_image"] for item in json.loads((args.corpus_root / "manifest.json").read_text(encoding="utf-8"))}
    pages: list[dict[str, Any]] = []
    for key, image_dir, reference_kind in (
        ("paddle_ocr_en", args.funsd_dir, "en"),
        ("paddle_ocr_vi", args.vi_dir, "vi"),
    ):
        for page_id, item in raw[key].items():
            if key == "paddle_ocr_en" and not page_id.startswith("funsd_"):
                continue
            if page_id.startswith("vietnamese_page_"):
                image_path = image_dir / f"page_{page_id.rsplit('_', 1)[-1]}.png"
            else:
                preview = manifest.get(page_id, "")
                prefix = "cache/test-pdfs/"
                image_path = args.corpus_root / (preview[len(prefix):] if preview.startswith(prefix) else preview)
                if not image_path.exists():
                    image_path = image_dir / f"{page_id}.png"
            if not image_path.exists():
                image_path = image_dir / f"{page_id}.jpg"
            if image_path.exists():
                pages.append({
                    "id": page_id,
                    "lang": reference_kind,
                    "path": image_path,
                    "boxes": item.get("boxes", []),
                    "reference": funsd_reference(image_path, args.corpus_root) if reference_kind == "en" else None,
                })

    args.out.mkdir(parents=True, exist_ok=True)
    tessdata_dir = str(args.tessdata_dir) if args.tessdata_dir else None
    results: dict[str, Any] = {
        "benchmark": {
            "detector": "captured PP-OCRv6 boxes from benchmark.json",
            "device": "cuda:0 for VietOCR",
            "tesseract_version": subprocess.run(["tesseract", "--version"], stdout=subprocess.PIPE, check=False).stdout.decode().splitlines()[0] if Path("/usr/bin/tesseract").exists() else None,
            "vietocr_model": "vgg_transformer",
        },
        "pages": {},
    }
    predictor = vietocr_predictor() if not args.tesseract_only else None
    total = len(pages)
    for page_index, page in enumerate(pages, 1):
        with Image.open(page["path"]) as source:
            image = source.copy()
        outputs: dict[str, list[dict[str, Any]]] = {"tesseract_shared_eng_vie": [], "tesseract_split": [], "vietocr_vgg_transformer": []}
        crops = [crop_for_box(image, box) for box in page["boxes"]]
        split_lang = "eng" if page["lang"] == "en" else "vie"
        # Tesseract is CPU-bound and process-startup-heavy.  Parallelize only
        # this independent part; VietOCR remains serialized on cuda:0.
        with ThreadPoolExecutor(max_workers=8) as pool:
            shared_values = list(pool.map(lambda crop: tesseract_one(crop, "eng+vie", tessdata_dir), crops))
            split_values = list(pool.map(lambda crop: tesseract_one(crop, split_lang, tessdata_dir), crops))
        viet_values = vietocr_batch(predictor, crops) if predictor is not None else []
        for index, crop in enumerate(crops):
            shared, shared_s = shared_values[index]
            split, split_s = split_values[index]
            outputs["tesseract_shared_eng_vie"].append({"text": shared, "elapsed_s": shared_s})
            outputs["tesseract_split"].append({"text": split, "elapsed_s": split_s, "lang": split_lang})
            if predictor is not None:
                outputs["vietocr_vgg_transformer"].append(viet_values[index])
        page_result: dict[str, Any] = {"lang": page["lang"], "path": str(page["path"]), "boxes": page["boxes"], "outputs": outputs}
        if page["reference"] is not None:
            reference = page["reference"]
            for name, values in outputs.items():
                prediction = "\n".join(v["text"] for v in values)
                page_result.setdefault("metrics", {})[name] = {"cer": cer(reference, prediction), "reference_chars": len(clean(reference)), "prediction_chars": len(clean(prediction))}
        else:
            for name, values in outputs.items():
                prediction = "\n".join(v["text"] for v in values)
                page_result.setdefault("metrics", {})[name] = {"nonempty_lines": sum(bool(clean(v["text"])) for v in values), "lines": len(values), "prediction_chars": len(clean(prediction))}
        results["pages"][page["id"]] = page_result
        print(f"recognition {page_index}/{total} {page['id']}", flush=True)

    # Aggregate page-level metrics; keep every page output for visual review.
    for lang in ("en", "vi"):
        selected = [v for v in results["pages"].values() if v["lang"] == lang]
        results.setdefault("aggregate", {})[lang] = {}
        for model in ("tesseract_shared_eng_vie", "tesseract_split", "vietocr_vgg_transformer"):
            metrics = [v["metrics"][model] for v in selected]
            if lang == "en":
                results["aggregate"][lang][model] = {"mean_cer": round(sum(v["cer"] for v in metrics) / max(1, len(metrics)), 5), "pages": len(metrics)}
            else:
                results["aggregate"][lang][model] = {
                    "mean_nonempty_rate": round(sum(v["nonempty_lines"] / max(1, v["lines"]) for v in metrics) / max(1, len(metrics)), 5),
                    "mean_prediction_chars": round(sum(v["prediction_chars"] for v in metrics) / max(1, len(metrics)), 2),
                    "pages": len(metrics),
                }
    (args.out / "recognition.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--funsd-dir", type=Path, required=True)
    parser.add_argument("--vi-dir", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tessdata-dir", type=Path)
    parser.add_argument("--tesseract-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
