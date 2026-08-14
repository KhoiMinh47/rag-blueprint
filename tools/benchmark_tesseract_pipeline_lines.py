"""Run Tesseract on the exact line crops produced by the pipeline benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def normalise(value: str) -> str:
    return " ".join((value or "").split()).strip()


def run(args):
    data = json.loads(args.pipeline.read_text(encoding="utf-8"))
    total = 0
    for page in data["pages"]:
        lang = "vie" if page["language"] == "vi" else "eng"
        for line in page["line_detector"]["lines"]:
            crop = Path(line["crop"])
            result = subprocess.run(
                ["tesseract", "--tessdata-dir", str(args.tessdata), "stdin", "stdout", "--psm", "7", "--oem", "1", "-l", lang, "-c", "preserve_interword_spaces=1"],
                input=crop.read_bytes(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
            line["tesseract_text"] = normalise(result.stdout.decode("utf-8", errors="replace"))
            total += 1
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for page in data["pages"]:
        lines = page["line_detector"]["lines"]
        print(f"{page['language']}: lines={len(lines)}")
        for line in lines[:args.samples]:
            print(json.dumps({"pp": line["pp_ocrv6_text"], "tesseract": line.get("tesseract_text", ""), "bbox": line["bbox_xyxy_norm"]}, ensure_ascii=False))
    print(f"tesseract_lines={total}")


parser = argparse.ArgumentParser()
parser.add_argument("--pipeline", type=Path, required=True)
parser.add_argument("--tessdata", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--samples", type=int, default=12)
run(parser.parse_args())
