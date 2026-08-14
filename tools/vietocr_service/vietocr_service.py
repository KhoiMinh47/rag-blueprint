"""Production VietOCR sidecar for Option 3.

The benchmark server in ``tools/ocr_benchmark`` is intentionally not used
here.  This service owns one predictor for its whole process lifetime and
accepts an ordered list of base64 image crops per request.
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, *, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


class VRAMAdmissionError(RuntimeError):
    """The request would violate the configured free-VRAM safety floor."""


class OCRRequest(BaseModel):
    images: list[Any] = Field(default_factory=list)
    # Accept the singular name used by a few existing image-inference clients
    # while keeping ``images`` the canonical batch field.
    input: Any | None = None


class VietOCRRuntime:
    def __init__(self) -> None:
        self.model_name = os.getenv("VIETOCR_MODEL", "vgg_seq2seq").strip() or "vgg_seq2seq"
        self.device = os.getenv("VIETOCR_DEVICE", "cuda:0").strip() or "cuda:0"
        self.allow_cpu = _env_bool("VIETOCR_ALLOW_CPU", default=False)
        # FastAPI runs synchronous handlers in a thread pool.  Bound that
        # pool at the model boundary so multiple pages cannot overlap
        # unbounded VietOCR inference on the single development GPU.
        self.max_concurrency = max(1, min(4, _env_int("VIETOCR_MAX_CONCURRENCY", default=2)))
        self.queue_timeout_s = max(1.0, _env_float("VIETOCR_QUEUE_TIMEOUT_S", default=120.0))
        self.min_free_vram_fraction = min(
            0.90,
            max(0.20, _env_float("VIETOCR_MIN_FREE_VRAM_FRACTION", default=0.20)),
        )
        self._request_gate = threading.BoundedSemaphore(self.max_concurrency)
        # The dedicated /v1/ocr/batch route is owned by Option 2. It may
        # fan out more small width-preserving predict_batch calls than the
        # legacy /v1/ocr route, while retaining an independent safety gate.
        self.batch_max_concurrency = max(
            1,
            min(8, _env_int("VIETOCR_BATCH_MAX_CONCURRENCY", default=8)),
        )
        self._batch_request_gate = threading.BoundedSemaphore(self.batch_max_concurrency)
        self.width_bucket_px = max(
            16,
            min(256, _env_int("VIETOCR_WIDTH_BUCKET_PX", default=128)),
        )
        # Padding changes the effective aspect ratio seen by the recognizer.
        # Keep the optimization conservative by default; FP16 can be enabled
        # after a deployment-specific accuracy gate.
        self.max_width_padding_ratio = min(
            1.50,
            max(1.0, _env_float("VIETOCR_MAX_WIDTH_PADDING_RATIO", default=1.35)),
        )
        self.use_fp16 = _env_bool("VIETOCR_USE_FP16", default=False)
        self.predictor: Any | None = None
        self.startup_error: str | None = None

    def load(self) -> None:
        try:
            import torch

            is_cpu = self.device.lower() == "cpu" or not self.device.lower().startswith("cuda")
            if is_cpu and not self.allow_cpu:
                raise RuntimeError(
                    "VietOCR CPU execution is disabled; set VIETOCR_ALLOW_CPU=true explicitly"
                )
            if self.device.lower().startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(
                    f"VietOCR requires GPU device {self.device!r}, but CUDA is unavailable"
                )

            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor

            config = Cfg.load_config_from_name(self.model_name)
            config["device"] = self.device
            # Beam search is unnecessary for this speed-first recognizer and
            # disabling it keeps the service's startup/request behavior
            # deterministic across the supported VietOCR configs.
            if isinstance(config.get("predictor"), dict):
                config["predictor"]["beamsearch"] = False
            self.predictor = Predictor(config)
            self.startup_error = None
        except Exception as exc:  # noqa: BLE001 - readiness reports startup failure
            self.predictor = None
            self.startup_error = f"{type(exc).__name__}: {exc}"

    @property
    def ready(self) -> bool:
        return self.predictor is not None and self.startup_error is None

    def recognize(self, image_payloads: list[Any]) -> list[dict[str, Any]]:
        if not self.ready:
            raise RuntimeError(self.startup_error or "VietOCR predictor is not ready")
        assert self.predictor is not None
        acquired = self._request_gate.acquire(timeout=self.queue_timeout_s)
        if not acquired:
            raise VRAMAdmissionError(
                "VietOCR concurrency limit is busy; retry after an active OCR request finishes"
            )
        try:
            self._check_free_vram()
            result: list[dict[str, Any]] = []
            for payload in image_payloads:
                try:
                    image = _decode_image(payload)
                    prediction = self.predictor.predict(image, return_prob=True)
                    text, score = _prediction_values(prediction)
                    result.append(
                        {
                            "text": text,
                            "score": score,
                            "model": self.model_name,
                            "backend": "vietocr",
                            "language": "vi",
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - preserve one output slot per input
                    result.append(
                        {
                            "text": "",
                            "score": None,
                            "model": self.model_name,
                            "backend": "vietocr",
                            "language": "vi",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            return result
        finally:
            self._request_gate.release()

    def recognize_batch(self, image_payloads: list[Any]) -> list[dict[str, Any]]:
        """Run VietOCR's native GPU batch path while preserving slot order."""

        if not self.ready:
            raise RuntimeError(self.startup_error or "VietOCR predictor is not ready")
        assert self.predictor is not None
        acquired = self._batch_request_gate.acquire(timeout=self.queue_timeout_s)
        if not acquired:
            raise VRAMAdmissionError(
                "VietOCR concurrency limit is busy; retry after an active OCR request finishes"
            )
        try:
            self._check_free_vram()
            results: list[dict[str, Any] | None] = [None] * len(image_payloads)
            valid_images: list[Image.Image] = []
            valid_indices: list[int] = []
            for index, payload in enumerate(image_payloads):
                try:
                    valid_images.append(_decode_image(payload))
                    valid_indices.append(index)
                except Exception as exc:  # noqa: BLE001 - preserve response slots
                    results[index] = _error_result(self.model_name, exc)
            if valid_images:
                try:
                    texts, scores = self._predict_batch_bucketed(valid_images)
                    for offset, index in enumerate(valid_indices):
                        text = texts[offset] if offset < len(texts) else ""
                        score = scores[offset] if offset < len(scores) else None
                        parsed_text, parsed_score = _prediction_values((text, score))
                        results[index] = _success_result(
                            self.model_name,
                            parsed_text,
                            parsed_score,
                        )
                except Exception as exc:  # noqa: BLE001 - preserve response slots
                    for index in valid_indices:
                        results[index] = _error_result(self.model_name, exc)
            return [item or _error_result(self.model_name, RuntimeError("missing result")) for item in results]
        finally:
            self._release_cuda_cache()
            self._batch_request_gate.release()

    def _release_cuda_cache(self) -> None:
        """Return unused predictor workspace after a bounded OCR request.

        Long PDFs are split into several transport batches.  VietOCR's
        decoder can leave temporary tensors in PyTorch's CUDA caching
        allocator between those requests; the next admission check then
        sees an artificially low free-VRAM value and returns 503 even though
        the model itself is healthy.  Emptying only the unused cache keeps
        the 20% safety floor intact and does not discard model weights.
        """
        if not self.device.lower().startswith("cuda"):
            return
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            # Cache cleanup is best-effort; inference results must not be
            # turned into an error merely because the allocator API changed.
            pass

    def _predict_batch_bucketed(
        self,
        images: list[Image.Image],
    ) -> tuple[list[str], list[float | None]]:
        """Run padded width buckets instead of one pass per exact width.

        VietOCR's stock ``predict_batch`` groups by the exact resized width.
        OCR crops naturally have many widths, which silently turns one HTTP
        batch into many GPU forward passes.  Quantized right-padding keeps
        the sequence geometry valid while substantially reducing that fanout.
        """

        assert self.predictor is not None
        import torch
        import torch.nn.functional as functional
        from vietocr.tool.translate import process_input, translate

        config = self.predictor.config
        width_items: list[tuple[int, int, Any]] = []
        for index, image in enumerate(images):
            tensor = process_input(
                image,
                config["dataset"]["image_height"],
                config["dataset"]["image_min_width"],
                config["dataset"]["image_max_width"],
            )
            width = int(tensor.shape[-1])
            quantum = self.width_bucket_px
            width_items.append((index, width, tensor))

        # Build groups in width order and only pad when the smallest crop in
        # that group stays close enough to the largest one.  Wide aspect-ratio
        # changes use separate groups, preserving the original recognizer's
        # accuracy characteristics.
        groups: list[tuple[int, list[tuple[int, Any]]]] = []
        current: list[tuple[int, int, Any]] = []
        current_min_width = 0
        for item in sorted(width_items, key=lambda value: value[1]):
            index, width, tensor = item
            candidate_max = max(width, current[-1][1] if current else width)
            candidate_bucket = min(
                int(config["dataset"]["image_max_width"]),
                max(
                    int(config["dataset"]["image_min_width"]),
                    ((candidate_max + quantum - 1) // quantum) * quantum,
                ),
            )
            fits = (
                not current
                or current_min_width <= 0
                or candidate_bucket / current_min_width
                <= self.max_width_padding_ratio
            )
            if not fits:
                previous_max = max(value[1] for value in current)
                previous_bucket = min(
                    int(config["dataset"]["image_max_width"]),
                    max(
                        int(config["dataset"]["image_min_width"]),
                        ((previous_max + quantum - 1) // quantum) * quantum,
                    ),
                )
                groups.append(
                    (
                        previous_bucket,
                        [
                            (old_index, old_tensor)
                            for old_index, _old_width, old_tensor in current
                        ],
                    )
                )
                current = []
                current_min_width = 0
            if not current:
                current_min_width = width
            current.append(item)
        if current:
            previous_max = max(value[1] for value in current)
            previous_bucket = min(
                int(config["dataset"]["image_max_width"]),
                max(
                    int(config["dataset"]["image_min_width"]),
                    ((previous_max + quantum - 1) // quantum) * quantum,
                ),
            )
            groups.append(
                (
                    previous_bucket,
                    [
                        (old_index, old_tensor)
                        for old_index, _old_width, old_tensor in current
                    ],
                )
            )

        texts: list[str] = [""] * len(images)
        scores: list[float | None] = [None] * len(images)
        device_type = "cuda" if str(self.device).lower().startswith("cuda") else "cpu"
        autocast_enabled = bool(self.use_fp16 and device_type == "cuda")
        for bucket_width, items in groups:
            batch = torch.cat(
                [
                    functional.pad(
                        tensor,
                        (0, max(0, bucket_width - int(tensor.shape[-1]))),
                    )
                    for _index, tensor in items
                ],
                dim=0,
            ).to(self.device)
            with torch.inference_mode():
                if autocast_enabled:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        decoded, probabilities = translate(batch, self.predictor.model)
                else:
                    decoded, probabilities = translate(batch, self.predictor.model)
            decoded_text = self.predictor.vocab.batch_decode(decoded.tolist())
            probability_values = probabilities.tolist()
            for offset, (index, _tensor) in enumerate(items):
                texts[index] = str(
                    decoded_text[offset] if offset < len(decoded_text) else ""
                )
                if offset < len(probability_values):
                    try:
                        scores[index] = float(probability_values[offset])
                    except (TypeError, ValueError):
                        scores[index] = None
        return texts, scores

    def _check_free_vram(self) -> None:
        if not self.device.lower().startswith("cuda"):
            return
        try:
            import torch

            with torch.cuda.device(self.device):
                free_bytes, total_bytes = torch.cuda.mem_get_info()
        except Exception as exc:  # noqa: BLE001 - fail closed for GPU admission
            raise VRAMAdmissionError(f"unable to inspect GPU memory: {exc}") from exc
        if total_bytes <= 0:
            raise VRAMAdmissionError("GPU memory capacity is unavailable")
        free_fraction = free_bytes / total_bytes
        if free_fraction < self.min_free_vram_fraction:
            raise VRAMAdmissionError(
                "VietOCR paused because free VRAM is below the configured "
                f"{self.min_free_vram_fraction:.0%} safety floor"
            )


runtime = VietOCRRuntime()
app = FastAPI(title="Option 3 VietOCR", version="1.0")


@app.on_event("startup")
def _startup() -> None:
    # Model construction is the only place this process may initialize or
    # download configured VietOCR weights.  Request handlers reuse predictor.
    runtime.load()


@app.get("/v1/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if runtime.ready else "starting",
        "ready": runtime.ready,
        "model": runtime.model_name,
        "device": runtime.device,
        "max_concurrency": runtime.max_concurrency,
        "min_free_vram_fraction": runtime.min_free_vram_fraction,
        "error": runtime.startup_error,
    }


@app.get("/v1/health/ready")
def readiness() -> JSONResponse:
    payload = health()
    if not runtime.ready:
        return JSONResponse(status_code=503, content=payload)
    return JSONResponse(status_code=200, content=payload)


@app.post("/v1/ocr")
def ocr(request: OCRRequest) -> list[dict[str, Any]]:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.startup_error or "VietOCR is not ready")
    images = list(request.images)
    if not images and request.input is not None:
        images = request.input if isinstance(request.input, list) else [request.input]
    if not images:
        raise HTTPException(status_code=400, detail="images must contain at least one base64 image")
    try:
        return runtime.recognize(images)
    except VRAMAdmissionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/ocr/batch")
def ocr_batch(request: OCRRequest) -> list[dict[str, Any]]:
    """Native GPU batch endpoint reserved for the speed-optimised pipeline."""

    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.startup_error or "VietOCR is not ready")
    images = list(request.images)
    if not images and request.input is not None:
        images = request.input if isinstance(request.input, list) else [request.input]
    if not images:
        raise HTTPException(status_code=400, detail="images must contain at least one base64 image")
    try:
        return runtime.recognize_batch(images)
    except VRAMAdmissionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _success_result(model_name: str, text: str, score: float | None) -> dict[str, Any]:
    return {
        "text": text,
        "score": score,
        "model": model_name,
        "backend": "vietocr",
        "language": "vi",
    }


def _error_result(model_name: str, exc: Exception) -> dict[str, Any]:
    return {
        "text": "",
        "score": None,
        "model": model_name,
        "backend": "vietocr",
        "language": "vi",
        "error": f"{type(exc).__name__}: {exc}",
    }




def _decode_image(payload: Any) -> Image.Image:
    if isinstance(payload, dict):
        payload = (
            payload.get("image_b64")
            or payload.get("image")
            or payload.get("data")
            or payload.get("base64")
            or payload.get("url")
        )
    if not isinstance(payload, str) or not payload.strip():
        raise ValueError("image payload must be a non-empty base64 string")
    encoded = payload.split(",", 1)[1] if payload.startswith("data:") and "," in payload else payload
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid base64 image payload") from exc
    with Image.open(io.BytesIO(raw)) as image:
        return image.convert("RGB")


def _prediction_values(prediction: Any) -> tuple[str, float | None]:
    if isinstance(prediction, dict):
        text = prediction.get("text") or prediction.get("output") or ""
        score = prediction.get("score") or prediction.get("probability")
    elif isinstance(prediction, (list, tuple)):
        text = prediction[0] if prediction else ""
        score = prediction[1] if len(prediction) > 1 else None
    else:
        text, score = prediction, None
    text = str(text or "").strip()
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    if score is not None:
        score = max(0.0, min(1.0, score))
    return text, score
