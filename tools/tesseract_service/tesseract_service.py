"""CPU Tesseract 5 recognizer sidecar for the split document OCR pipeline.

The retriever sends cropped Page Elements boxes and table-cell images using
the same small image-list contract as the PP-OCR sidecars. Detection remains
owned by Page Elements; this service only recognizes text. Callers may
override ``language`` and ``psm`` per request; environment variables remain
the defaults for legacy callers.
"""

from __future__ import annotations

import base64
import io
import os
import re
import shutil
from typing import Any

from fastapi import FastAPI, HTTPException
from PIL import Image


def _decode_image(value: str) -> Image.Image:
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    try:
        raw = base64.b64decode(value, validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            return source.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc


def _images(payload: dict[str, Any]) -> list[Image.Image]:
    values = payload.get("images") or payload.get("input") or payload.get("data") or []
    if isinstance(values, dict):
        values = [values]
    result: list[Image.Image] = []
    for item in values:
        if isinstance(item, str):
            result.append(_decode_image(item))
            continue
        if not isinstance(item, dict):
            continue
        value = item.get("image_b64") or item.get("image") or item.get("url")
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            value = image_url.get("url") or value
        if isinstance(value, str):
            result.append(_decode_image(value))
    return result


def _config(
    *, language: Any = None, psm: Any = None
) -> tuple[str, str]:
    configured_language = os.getenv("TESSERACT_LANG", "vie").strip() or "vie"
    configured_psm = os.getenv("TESSERACT_PSM", "6").strip() or "6"

    requested_language = str(language or "").strip()
    if requested_language and re.fullmatch(r"[A-Za-z0-9_.+-]+", requested_language):
        configured_language = requested_language

    requested_psm = str(psm or "").strip()
    if requested_psm.isdigit() and 0 <= int(requested_psm) <= 13:
        configured_psm = requested_psm
    return configured_language, configured_psm


def _recognize(image: Image.Image, *, language: str, psm: str) -> dict[str, Any]:
    import pytesseract

    try:
        data = pytesseract.image_to_data(
            image,
            lang=language,
            config=f"--psm {psm} -c preserve_interword_spaces=1",
            output_type=pytesseract.Output.DICT,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Tesseract recognition failed: {exc}") from exc

    grouped: dict[tuple[int, int, int], list[tuple[int, str, float]]] = {}
    texts = data.get("text") or []
    confidences = data.get("conf") or []
    blocks = data.get("block_num") or []
    paragraphs = data.get("par_num") or []
    lines = data.get("line_num") or []
    lefts = data.get("left") or []
    for index, raw_text in enumerate(texts):
        text = str(raw_text or "").strip()
        if not text:
            continue
        try:
            confidence = float(confidences[index]) if index < len(confidences) else -1.0
        except (TypeError, ValueError):
            confidence = -1.0
        key = (
            int(blocks[index]) if index < len(blocks) else 0,
            int(paragraphs[index]) if index < len(paragraphs) else 0,
            int(lines[index]) if index < len(lines) else 0,
        )
        left = int(lefts[index]) if index < len(lefts) else index
        grouped.setdefault(key, []).append((left, text, confidence))

    output_lines: list[str] = []
    valid_confidences: list[float] = []
    for words in grouped.values():
        words.sort(key=lambda value: value[0])
        output_lines.append(" ".join(value[1] for value in words))
        valid_confidences.extend(value[2] for value in words if value[2] >= 0.0)

    score = sum(valid_confidences) / (100.0 * len(valid_confidences)) if valid_confidences else None
    return {
        "text": "\n".join(output_lines).strip(),
        "score": score,
        "model": "tesseract-5",
        "backend": "tesseract",
        "language": language,
    }


app = FastAPI(title="Tesseract 5 OCR sidecar", version="1.0")


@app.get("/v1/health/ready")
def ready() -> dict[str, Any]:
    binary = shutil.which("tesseract")
    if binary is None:
        raise HTTPException(status_code=503, detail="tesseract binary is not installed")
    import pytesseract

    return {
        "ready": True,
        "model": "tesseract-5",
        "binary": binary,
        "version": str(pytesseract.get_tesseract_version()).splitlines()[0],
        "language": _config()[0],
    }


@app.post("/v1/ocr")
def ocr(payload: dict[str, Any]) -> list[dict[str, Any]]:
    images = _images(payload)
    language, psm = _config(
        language=payload.get("language"),
        psm=payload.get("psm"),
    )
    return [_recognize(image, language=language, psm=psm) for image in images]
