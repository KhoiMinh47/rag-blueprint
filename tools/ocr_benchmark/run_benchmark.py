"""Run and score the isolated OCR benchmark server."""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import requests


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def no_marks(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFD", text) if unicodedata.category(char) != "Mn")


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for index, char_left in enumerate(left, start=1):
        current = [index]
        for j, char_right in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char_left != char_right)))
        previous = current
    return previous[-1]


def numeric_tokens(text: str) -> list[str]:
    return re.findall(r"(?<!\w)\d[\d.,:/-]*(?!\w)", text)


def score_sample(reference: str, prediction: str) -> dict[str, Any]:
    reference = normalize(reference)
    prediction = normalize(prediction)
    reference_base = no_marks(reference)
    prediction_base = no_marks(prediction)
    reference_numbers = numeric_tokens(reference)
    prediction_numbers = numeric_tokens(prediction)
    return {
        "reference_chars": len(reference),
        "prediction_chars": len(prediction),
        "edit_distance": edit_distance(reference, prediction),
        "base_edit_distance": edit_distance(reference_base, prediction_base),
        "exact": reference == prediction,
        "nonempty": bool(prediction),
        "numbers_expected": reference_numbers,
        "numbers_predicted": prediction_numbers,
        "numbers_exact": reference_numbers == prediction_numbers,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0}
    chars = sum(row["reference_chars"] for row in rows)
    base_chars = sum(len(no_marks(row["reference"])) for row in rows)
    edit = sum(row["edit_distance"] for row in rows)
    base_edit = sum(row["base_edit_distance"] for row in rows)
    return {
        "samples": len(rows),
        "nonempty_rate": sum(row["nonempty"] for row in rows) / len(rows),
        "exact_line_rate": sum(row["exact"] for row in rows) / len(rows),
        "cer": edit / max(chars, 1),
        "base_cer_without_diacritics": base_edit / max(base_chars, 1),
        "diacritic_gap": max(0.0, (edit / max(chars, 1)) - (base_edit / max(base_chars, 1))),
        "numeric_exact_rate": sum(row["numbers_exact"] for row in rows) / len(rows),
        "mean_server_latency_ms": sum(row["server_latency_ms"] for row in rows) / len(rows),
        "p95_server_latency_ms": sorted(row["server_latency_ms"] for row in rows)[max(0, int(len(rows) * 0.95) - 1)],
        "errors": sum(bool(row.get("error")) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("cache/ocr-benchmark/corpus/manifest.json"))
    parser.add_argument("--url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, default=Path("cache/ocr-benchmark/results"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-per-corpus", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = list(manifest["samples"])
    if args.limit_per_corpus:
        selected = []
        counts: dict[str, int] = {}
        for sample in samples:
            corpus = str(sample["corpus"])
            if counts.get(corpus, 0) < args.limit_per_corpus:
                selected.append(sample)
                counts[corpus] = counts.get(corpus, 0) + 1
        samples = selected

    health = requests.get(f"{args.url.rstrip('/')}/v1/health/ready", timeout=30)
    health.raise_for_status()
    health_json = health.json()
    if not health_json.get("ready"):
        raise RuntimeError(f"benchmark server is not ready: {health_json}")

    rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for batch_start in range(0, len(samples), max(1, args.batch_size)):
        batch = samples[batch_start : batch_start + max(1, args.batch_size)]
        payload = {"images": []}
        for sample in batch:
            image_bytes = (root / sample["image"]).read_bytes()
            payload["images"].append({"image_b64": base64.b64encode(image_bytes).decode("ascii")})
        request_started = time.perf_counter()
        response = requests.post(f"{args.url.rstrip('/')}/v1/recognize", json=payload, timeout=args.timeout)
        response.raise_for_status()
        outputs = response.json()
        if len(outputs) != len(batch):
            raise RuntimeError(f"response count mismatch: expected {len(batch)}, got {len(outputs)}")
        request_elapsed = (time.perf_counter() - request_started) * 1000.0
        for sample, output in zip(batch, outputs):
            prediction = normalize(str(output.get("text") or ""))
            measured = score_sample(sample["text"], prediction)
            reported_latency = output.get("latency_ms")
            try:
                reported_latency = float(reported_latency)
            except (TypeError, ValueError):
                reported_latency = 0.0
            if reported_latency <= 0:
                # Some existing sidecars implement the OCR response contract
                # but omit latency. Keep the measured request timing usable.
                reported_latency = request_elapsed / max(len(batch), 1)
            row = {
                "id": sample["id"],
                "corpus": sample["corpus"],
                "page": sample["page"],
                "image": sample["image"],
                "reference": normalize(sample["text"]),
                "prediction": prediction,
                "score": output.get("score"),
                "server_latency_ms": reported_latency,
                "batch_request_latency_ms": request_elapsed / max(len(batch), 1),
                "error": output.get("error"),
                **measured,
            }
            rows.append(row)
        print(f"processed={min(batch_start + len(batch), len(samples))}/{len(samples)}", flush=True)

    by_corpus = {corpus: aggregate([row for row in rows if row["corpus"] == corpus]) for corpus in ("native", "ocr")}
    summary = {
        "label": args.label,
        "health": health_json,
        "samples": len(rows),
        "elapsed_s": time.perf_counter() - started_all,
        "overall": aggregate(rows),
        "by_corpus": by_corpus,
        "worst_examples": sorted(rows, key=lambda row: row["edit_distance"], reverse=True)[:12],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.label}.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    (args.output / f"{args.label}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
