"""Isolated OCR benchmark server.

Start one instance per model with ``OCR_BENCH_BACKEND``.  It intentionally
uses the same small HTTP contract as the project's OCR sidecars, but it is
not imported by or connected to the production ingest pipeline.
"""

from __future__ import annotations

import base64
import io
import os
import re
import time
from typing import Any, Iterable

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image


BACKEND = os.getenv("OCR_BENCH_BACKEND", "").strip().lower()
MODEL_ID = os.getenv("OCR_BENCH_MODEL", "").strip()
DEVICE = os.getenv("OCR_BENCH_DEVICE", "cuda:0" if os.getenv("CUDA_VISIBLE_DEVICES", "") != "" else "cpu")


def _decode(value: Any) -> Image.Image:
    if isinstance(value, dict):
        value = value.get("image_b64") or value.get("image") or value.get("url")
    if not isinstance(value, str):
        raise ValueError("image must be a base64 string or an image object")
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")


def _images(payload: dict[str, Any]) -> list[Image.Image]:
    values = payload.get("images") or payload.get("input") or payload.get("data") or []
    if isinstance(values, (str, dict)):
        values = [values]
    return [_decode(item) for item in values]


def _text_normalize(value: Any) -> str:
    text = str(value or "")
    return re.sub(r"\s+", " ", text).strip()


def _json_value(value: Any) -> Any:
    if hasattr(value, "json"):
        value = value.json
    if isinstance(value, str):
        try:
            import json

            return json.loads(value)
        except Exception:
            return value
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _walk_texts(value: Any) -> Iterable[str]:
    value = _json_value(value)
    if isinstance(value, dict):
        preferred = ("rec_text", "text", "block_content", "content", "markdown", "ocr_text")
        for key in preferred:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                yield candidate
        for key, child in value.items():
            if key not in preferred and key not in {"model", "score", "confidence", "input_path"}:
                yield from _walk_texts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_texts(child)


def _first_text(value: Any) -> str:
    texts = [_text_normalize(item) for item in _walk_texts(value)]
    texts = [item for item in texts if item]
    return "\n".join(dict.fromkeys(texts))


class Backend:
    def __init__(self) -> None:
        self.name = BACKEND
        self.model_id = MODEL_ID
        self.model: Any = None
        self.ready = False
        self._load()
        self.ready = True

    def _load(self) -> None:
        if self.name == "vietocr":
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor

            config = Cfg.load_config_from_name(self.model_id or "vgg_transformer")
            config["device"] = DEVICE
            self.model = Predictor(config)
            self.model_id = self.model_id or "vgg_transformer"
            return

        if self.name == "easyocr":
            import easyocr

            languages = [item.strip() for item in os.getenv("EASYOCR_LANGS", "vi,en").split(",") if item.strip()]
            self.model = easyocr.Reader(
                languages,
                gpu=DEVICE.startswith("cuda"),
                model_storage_directory=os.getenv("EASYOCR_MODEL_DIR", "/tmp/easyocr-models"),
                download_enabled=True,
            )
            self.model_id = self.model_id or "+".join(languages)
            return

        if self.name == "trocr":
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            self.processor = TrOCRProcessor.from_pretrained(self.model_id or "microsoft/trocr-base-printed")
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_id or "microsoft/trocr-base-printed")
            self.model.to(DEVICE).eval()
            self.model_id = self.model_id or "microsoft/trocr-base-printed"
            return

        if self.name == "tesseract":
            import pytesseract

            self.model = pytesseract
            self.model_id = self.model_id or "tesseract-5"
            return

        if self.name == "paddle_vl":
            from paddleocr import PaddleOCRVL

            requested = self.model_id or "PaddleOCR-VL-1.6"
            version = {
                "PaddleOCR-VL": "v1",
                "PaddleOCR-VL-1.5": "v1.5",
                "PaddleOCR-VL-1.6": "v1.6",
            }.get(requested, requested)
            try:
                self.model = PaddleOCRVL(pipeline_version=version, device=DEVICE)
            except TypeError:
                # Older PaddleOCR builds may not expose ``device`` on the
                # pipeline wrapper; let Paddle select its configured device.
                self.model = PaddleOCRVL(pipeline_version=version)
            self.model_id = requested
            return

        raise RuntimeError(f"unsupported OCR_BENCH_BACKEND={self.name!r}")

    def recognize(self, image: Image.Image) -> tuple[str, float | None]:
        if self.name == "vietocr":
            return _text_normalize(self.model.predict(image)), None

        if self.name == "easyocr":
            # A line crop can be split into overlapping word boxes. EasyOCR's
            # paragraph merger resolves those overlaps and restores reading
            # order more reliably than sorting the raw boxes by their top edge.
            output = self.model.readtext(np.asarray(image), detail=1, paragraph=True, mag_ratio=1.0)
            texts = [_text_normalize(item[1]) for item in (output or []) if len(item) >= 2 and _text_normalize(item[1])]
            return " ".join(texts), None

        if self.name == "trocr":
            import torch

            with torch.inference_mode():
                pixels = self.processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
                generated = self.model.generate(pixels, max_new_tokens=128, num_beams=1)
            return _text_normalize(self.processor.batch_decode(generated, skip_special_tokens=True)[0]), None

        if self.name == "tesseract":
            data = self.model.image_to_data(
                image,
                lang=os.getenv("TESSERACT_LANG", "vie+eng"),
                config=os.getenv("TESSERACT_CONFIG", "--oem 1 --psm 7"),
                output_type=self.model.Output.DICT,
            )
            words = []
            scores = []
            for text, confidence in zip(data.get("text", []), data.get("conf", [])):
                text = _text_normalize(text)
                if not text:
                    continue
                words.append(text)
                try:
                    value = float(confidence)
                    if value >= 0:
                        scores.append(value / 100.0)
                except (TypeError, ValueError):
                    pass
            return " ".join(words), (sum(scores) / len(scores) if scores else None)

        if self.name == "paddle_vl":
            output = self.model.predict(input=np.asarray(image))
            return _first_text(list(output)), None

        raise RuntimeError(f"backend {self.name!r} is not loaded")


try:
    backend = Backend()
    startup_error: str | None = None
except Exception as exc:  # Keep /health useful for benchmark diagnostics.
    backend = None
    startup_error = f"{type(exc).__name__}: {exc}"

app = FastAPI(title="Isolated OCR benchmark server", version="1.0")


@app.get("/v1/health/ready")
def ready() -> dict[str, Any]:
    return {"ready": bool(backend and backend.ready), "backend": BACKEND, "model": MODEL_ID, "error": startup_error}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {"backend": BACKEND, "model": MODEL_ID, "device": DEVICE, "error": startup_error}


@app.post("/v1/recognize")
def recognize(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if backend is None:
        raise HTTPException(status_code=503, detail=startup_error or "backend is not ready")
    outputs: list[dict[str, Any]] = []
    for image in _images(payload):
        started = time.perf_counter()
        try:
            text, score = backend.recognize(image)
            outputs.append(
                {
                    "text": text,
                    "score": score,
                    "model": backend.model_id,
                    "backend": backend.name,
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                }
            )
        except Exception as exc:
            outputs.append(
                {
                    "text": "",
                    "score": None,
                    "model": backend.model_id,
                    "backend": backend.name,
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return outputs
