"""Ground-truth recognition benchmark for PP-OCRv6 on VinText.

This intentionally uses the dataset's annotated quadrilaterals so the score
measures recognition and Vietnamese diacritics, not page text detection.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from paddleocr import TextRecognition


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def word_distance(a: str, b: str) -> int:
    return edit_distance(a.split(), b.split())


def no_marks(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def perspective_crop(image: np.ndarray, points: list[int]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    width_a = np.linalg.norm(pts[1] - pts[0])
    width_b = np.linalg.norm(pts[2] - pts[3])
    height_a = np.linalg.norm(pts[3] - pts[0])
    height_b = np.linalg.norm(pts[2] - pts[1])
    width = max(8, int(round(max(width_a, width_b))))
    height = max(8, int(round(max(height_a, height_b))))
    # The dataset ordering is top-left, top-right, bottom-right, bottom-left.
    target = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts, target)
    return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def aligned_diacritic_errors(reference: str, prediction: str) -> tuple[int, int]:
    """Count aligned base-equal but mark-different characters."""
    ref = list(unicodedata.normalize("NFD", reference))
    pred = list(unicodedata.normalize("NFD", prediction))
    # Compare base characters and mark signatures as grapheme-like tokens.
    def tokens(chars: list[str]) -> list[tuple[str, str]]:
        out = []
        for ch in chars:
            if unicodedata.category(ch) == "Mn" and out:
                out[-1] = (out[-1][0], out[-1][1] + ch)
            else:
                out.append((ch, ""))
        return out
    a, b = tokens(ref), tokens(pred)
    # State is (edit_cost, diacritic_mismatches, base_equal_pairs).
    prev = [(0, 0, 0)] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [(i, 0, 0)]
        for j in range(1, len(b) + 1):
            candidates = [
                (prev[j][0] + 1, prev[j][1], prev[j][2]),
                (cur[j - 1][0] + 1, cur[j - 1][1], cur[j - 1][2]),
            ]
            if a[i - 1][0] == b[j - 1][0]:
                mismatch = a[i - 1][1] != b[j - 1][1]
                candidates.append((prev[j - 1][0] + mismatch, prev[j - 1][1] + mismatch, prev[j - 1][2] + 1))
            else:
                candidates.append((prev[j - 1][0] + 1, prev[j - 1][1], prev[j - 1][2]))
            cur.append(min(candidates))
        prev = cur
    return prev[-1][1], prev[-1][2]


def main(args: argparse.Namespace) -> None:
    root = args.root
    data_root = root / "vintext/vintext/vintext"
    image_root = data_root / "test_image"
    label_root = data_root / "labels"
    rows: list[dict] = []
    for image_path in sorted(image_root.glob("im*.jpg"), key=lambda p: int(p.stem[2:])):
        label_path = label_root / f"gt_{int(image_path.stem[2:])}.txt"
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path))
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            fields = line.split(",")
            if len(fields) < 9:
                continue
            text = normalize(",".join(fields[8:]))
            if not text or text == "###":
                continue
            points = [int(v) for v in fields[:8]]
            rows.append({"image": image_path.name, "line": line_no, "reference": text, "crop": perspective_crop(image, points)})
            if args.limit and len(rows) >= args.limit:
                break
        if args.limit and len(rows) >= args.limit:
            break

    print(f"instances={len(rows)}", flush=True)
    model = TextRecognition(model_name="PP-OCRv6_medium_rec", device="gpu:0")
    predictions: list[str] = []
    scores: list[float] = []
    elapsed = 0.0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        t0 = time.perf_counter()
        outputs = model.predict([row["crop"] for row in batch])
        elapsed += time.perf_counter() - t0
        for output in outputs:
            result = output.json.get("res", {})
            predictions.append(normalize(str(result.get("rec_text", ""))))
            scores.append(float(result.get("rec_score", 0.0)))
        if (start // args.batch_size + 1) % 20 == 0 or start + len(batch) == len(rows):
            print(f"processed={start + len(batch)}/{len(rows)}", flush=True)

    total_ref_chars = total_edit = total_base_edit = total_ref_words = total_word_edit = 0
    diacritic_mismatch = aligned_base_pairs = 0
    samples = []
    for index, (row, prediction) in enumerate(zip(rows, predictions)):
        ref = row["reference"]
        total_ref_chars += len(ref)
        total_edit += edit_distance(ref, prediction)
        total_base_edit += edit_distance(no_marks(ref), no_marks(prediction))
        total_ref_words += len(ref.split())
        total_word_edit += word_distance(ref, prediction)
        d_err, pairs = aligned_diacritic_errors(ref, prediction)
        diacritic_mismatch += d_err
        aligned_base_pairs += pairs
        if len(samples) < args.samples and (ref != prediction):
            samples.append({"image": row["image"], "line": row["line"], "reference": ref, "prediction": prediction, "score": scores[index]})

    result = {
        "model": "PP-OCRv6_medium_rec",
        "device": "gpu:0",
        "dataset": "VinText test",
        "instances": len(rows),
        "reference_chars": total_ref_chars,
        "reference_words": total_ref_words,
        "cer": round(total_edit / max(1, total_ref_chars), 6),
        "base_cer_without_diacritics": round(total_base_edit / max(1, len(no_marks("".join(row["reference"] for row in rows)))), 6),
        "wer": round(total_word_edit / max(1, total_ref_words), 6),
        "diacritic_error_rate_on_aligned_bases": round(diacritic_mismatch / max(1, aligned_base_pairs), 6),
        "mean_confidence": round(sum(scores) / max(1, len(scores)), 6),
        "elapsed_s": round(elapsed, 3),
        "instances_per_second": round(len(rows) / max(0.001, elapsed), 2),
        "samples": samples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--samples", type=int, default=100)
    main(parser.parse_args())
