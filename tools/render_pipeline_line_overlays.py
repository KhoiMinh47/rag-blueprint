"""Render visual evidence for the two-page pipeline benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {"table": (220, 30, 30), "text": (30, 90, 220), "title": (220, 150, 0), "infographic": (150, 40, 180), "header_footer": (20, 150, 100)}


def box(draw, bbox, size, color, label):
    x0, y0, x1, y1 = [int(v * size[i % 2]) for i, v in enumerate(bbox)]
    draw.rectangle((x0, y0, x1, y1), outline=color, width=max(2, size[0] // 1200))
    draw.text((x0 + 3, max(0, y0 - 16)), label, fill=color)


def map_box(parent, child):
    x0, y0, x1, y1 = parent
    a, b, c, d = child
    return [x0 + a * (x1 - x0), y0 + b * (y1 - y0), x0 + c * (x1 - x0), y0 + d * (y1 - y0)]


def run(args):
    data = json.loads(args.pipeline.read_text(encoding="utf-8"))
    for page in data["pages"]:
        source = Path(page["source"].replace("/workspace/", str(args.workspace) + "/"))
        # The benchmark image is the rendered 300 dpi page next to the PDFs.
        rendered = args.workspace / "cache/pipeline-line-benchmark" / f"{source.stem}_page{page['page_number']}_300dpi.png"
        image = Image.open(rendered).convert("RGB")
        draw = ImageDraw.Draw(image)
        size = image.size
        for det in page["page_elements"]["detections"]:
            label = det.get("label_name", "unknown")
            box(draw, det["bbox_xyxy_norm"], size, COLORS.get(label, (100, 100, 100)), label)
        for line_no, line in enumerate(page["line_detector"]["lines"]):
            box(draw, line["bbox_xyxy_norm"], size, (0, 180, 0), f"L{line_no}")
        for region in page["table_structure"]["payload"].get("regions", []):
            parent = region["bbox_xyxy_norm"]
            for det in region.get("detections", []):
                if det.get("label_name") == "cell":
                    box(draw, map_box(parent, det["bbox_xyxy_norm"]), size, (0, 170, 170), "cell")
        dest = args.out / f"{page['language']}_page{page['page_number']}_overlay.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest, format="PNG", compress_level=3)
        print(dest)


parser = argparse.ArgumentParser()
parser.add_argument("--pipeline", type=Path, required=True)
parser.add_argument("--workspace", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
run(parser.parse_args())
