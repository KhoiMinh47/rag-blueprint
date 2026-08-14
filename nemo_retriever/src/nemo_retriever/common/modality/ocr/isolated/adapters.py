# SPDX-License-Identifier: Apache-2.0

"""Opt-in adapters for the existing OCR and OpenAI-compatible VLM APIs."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from nemo_retriever.common.modality.ocr.isolated.contracts import BBox
from nemo_retriever.common.modality.ocr.isolated.geometry import clamp_bbox


class BatchInvoker(Protocol):
    """Callable compatible with the current image-list NIM service contract."""

    def __call__(
        self,
        endpoint: str,
        images: Sequence[str],
        *,
        extra_payload: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        batch_size: int = 8,
        max_pool_workers: int | None = None,
        max_retries: int = 3,
        max_429_retries: int = 2,
    ) -> Sequence[Any]: ...


class OCRBackend(Protocol):
    """Recognition backend used by Option 4."""

    model: str
    language: str | None

    def recognize(self, images: Sequence[str]) -> Sequence[Any]: ...


class VietnameseRecognizerBackend(OCRBackend, Protocol):
    """Neutral batch contract for Option 3's Vietnamese recognizer branch.

    The default implementation is VietOCR, but keeping this contract
    backend-neutral lets a future server-owned Vintern adapter replace it
    without changing routing, geometry, or merge code.
    """

    backend: str


class OCRDetectorBackend(Protocol):
    """Text-line detector used by the opt-in Option 4/5 branches."""

    def detect(self, images: Sequence[str]) -> Sequence[Any]: ...


def default_batch_invoker(
    endpoint: str,
    images: Sequence[str],
    *,
    extra_payload: Mapping[str, Any] | None = None,
    api_key: str | None = None,
    timeout_s: float = 120.0,
    batch_size: int = 8,
    max_pool_workers: int | None = None,
    max_retries: int = 3,
    max_429_retries: int = 2,
) -> Sequence[Any]:
    """Call the same batch transport used by the existing remote OCR stage."""
    from nemo_retriever.models.nim.nim import invoke_image_inference_batches

    return list(
        invoke_image_inference_batches(
            invoke_url=endpoint,
            image_b64_list=list(images),
            extra_payload=dict(extra_payload or {}),
            api_key=api_key,
            timeout_s=float(timeout_s),
            max_batch_size=max(1, int(batch_size)),
            max_retries=max(1, int(max_retries)),
            max_429_retries=max(0, int(max_429_retries)),
            max_pool_workers=(
                max(1, int(max_pool_workers))
                if max_pool_workers is not None
                else max(1, min(int(batch_size), len(images) or 1))
            ),
        )
    )


@dataclass
class HTTPImageBackend:
    """Small endpoint adapter; endpoint paths remain explicit and injectable."""

    endpoint: str
    model: str
    language: str | None = None
    api_key: str | None = None
    timeout_s: float = 120.0
    batch_size: int = 8
    invoker: BatchInvoker = default_batch_invoker
    request_payload: Mapping[str, Any] | None = None
    max_retries: int = 3
    max_429_retries: int = 2
    max_pool_workers: int | None = None
    backend: str | None = None
    # Updated after every call so document-level diagnostics can distinguish
    # one logical batch from the number of transport chunks used by the NIM.
    last_request_count: int = field(default=0, init=False, repr=False)

    def recognize(self, images: Sequence[str]) -> Sequence[Any]:
        if not images:
            self.last_request_count = 0
            return []
        max_batch_size = max(1, int(self.batch_size))
        self.last_request_count = (
            len(images) + max_batch_size - 1
        ) // max_batch_size
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout_s": self.timeout_s,
            "batch_size": self.batch_size,
        }
        # Keep the legacy Option 4 call shape unchanged when it uses the
        # adapter defaults; Option 5 opts into the faster fail-fast values.
        if int(self.max_retries) != 3:
            kwargs["max_retries"] = max(1, int(self.max_retries))
        if int(self.max_429_retries) != 2:
            kwargs["max_429_retries"] = max(0, int(self.max_429_retries))
        if self.request_payload:
            kwargs["extra_payload"] = dict(self.request_payload)
        if self.max_pool_workers is not None:
            kwargs["max_pool_workers"] = max(1, int(self.max_pool_workers))
        return self.invoker(
            self.endpoint,
            images,
            **kwargs,
        )


@dataclass
class VLLMImageBackend:
    """OpenAI-compatible multimodal adapter for a vLLM OCR server.

    vLLM does not expose the project's NIM ``input`` contract.  This adapter
    keeps the Option 2 backend contract unchanged while translating each crop
    to ``/v1/chat/completions``.  Requests are deliberately sent concurrently
    so vLLM can perform continuous batching; ``batch_size`` remains the
    safety limit for the single-GPU development host.
    """

    endpoint: str
    model: str
    language: str | None = None
    api_key: str | None = None
    timeout_s: float = 120.0
    batch_size: int = 2
    task_prompt: str = (
        "Chép lại chính xác toàn bộ chữ nhìn thấy trong ảnh. "
        "Chỉ trả về văn bản OCR, giữ nguyên dấu tiếng Việt và thứ tự dòng, "
        "không giải thích. Nếu ảnh không có chữ rõ ràng, trả về chuỗi rỗng."
    )
    max_tokens: int = 256
    repetition_penalty: float = 2.5
    max_retries: int = 2
    max_429_retries: int = 1
    backend: str = "vllm"
    max_pool_workers: int | None = None
    # Qwen 3.5 exposes thinking as a chat-template switch.  It is opt-in so
    # the existing Ministral/Vintern adapters keep their legacy payload shape.
    chat_template_kwargs: Mapping[str, Any] | None = None
    # Pipeline 7 batches semantic text/title/table crops, scan pages, and
    # Page Elements visual crops through one VLM. A bounded document cache
    # avoids re-sending an identical raster; it remains opt-in so existing
    # Vintern behavior is unchanged.
    cache_images: bool = False
    cache_max_entries: int = 512
    last_request_count: int = field(default=0, init=False, repr=False)
    last_elapsed_s: float = field(default=0.0, init=False, repr=False)
    last_prompt_tokens: int = field(default=0, init=False, repr=False)
    last_generation_tokens: int = field(default=0, init=False, repr=False)
    last_generation_tps: float = field(default=0.0, init=False, repr=False)
    _response_cache: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _client: Any = field(default=None, init=False, repr=False)

    def _get_client(self) -> Any:
        if self._client is None:
            from nemo_retriever.models.nim.nim import NIMClient

            # Pipeline 6 supplies an explicit concurrency ceiling so vLLM can
            # keep its continuous batch full. Older pipelines fall back to
            # their existing batch-size-derived pool.
            pool_workers = (
                int(self.max_pool_workers)
                if self.max_pool_workers is not None
                else min(int(self.batch_size), 25)
            )
            self._client = NIMClient(
                max_pool_workers=max(1, pool_workers)
            )
        return self._client

    def recognize(self, images: Sequence[str]) -> Sequence[Any]:
        if not images:
            self.last_request_count = 0
            return []
        image_list = [str(image or "") for image in images]
        results: list[dict[str, Any] | None] = [None] * len(image_list)
        pending_images: list[str] = []
        pending_indices: list[int] = []
        for index, image in enumerate(image_list):
            cached = self._response_cache.get(image) if self.cache_images else None
            if cached is not None:
                results[index] = dict(cached)
            else:
                pending_indices.append(index)
                pending_images.append(image)

        if pending_images:
            extra_body: dict[str, Any] = {"max_tokens": max(1, int(self.max_tokens))}
            if self.chat_template_kwargs:
                extra_body["chat_template_kwargs"] = dict(self.chat_template_kwargs)
            texts = self._get_client().invoke_chat_completions_images(
                invoke_url=self.endpoint,
                image_b64_list=pending_images,
                model=self.model,
                api_key=self.api_key,
                timeout_s=float(self.timeout_s),
                task_prompt=self.task_prompt,
                temperature=0.0,
                repetition_penalty=float(self.repetition_penalty),
                extra_body=extra_body,
                max_retries=max(1, int(self.max_retries)),
                max_429_retries=max(0, int(self.max_429_retries)),
            )
            self.last_request_count = len(pending_images)
            for offset, index in enumerate(pending_indices):
                text = texts[offset] if offset < len(texts) else ""
                value = {
                    "text": str(text or "").strip(),
                    "model": self.model,
                    "backend": self.backend,
                    "language": self.language,
                }
                results[index] = value
                if self.cache_images and image_list[index]:
                    self._response_cache[image_list[index]] = dict(value)
                    while len(self._response_cache) > max(
                        1, int(self.cache_max_entries)
                    ):
                        self._response_cache.pop(next(iter(self._response_cache)))
        else:
            self.last_request_count = 0

        return [result or {"text": "", "model": self.model, "backend": self.backend}
                for result in results]

    def recognize_with_prompts(
        self,
        images: Sequence[str],
        prompts: Sequence[str],
        *,
        max_tokens: int | None = None,
        max_tokens_per_task: Sequence[int] | None = None,
    ) -> Sequence[Any]:
        """Send mixed page/crop tasks through one persistent bounded pool.

        Pipeline 6 uses this method so scan-page OCR, native table crops, and
        native visual crops share one continuous vLLM batch.  The server
        returns usage metadata when available; it is retained only for the
        document diagnostics and does not change the legacy ``recognize``
        return contract.
        """
        image_list = [str(image or "") for image in images]
        prompt_list = [str(prompt or "") for prompt in prompts]
        if len(image_list) != len(prompt_list):
            raise ValueError(
                "images and prompts must have the same length "
                f"({len(image_list)} != {len(prompt_list)})"
            )
        if max_tokens_per_task is not None and len(max_tokens_per_task) != len(image_list):
            raise ValueError(
                "max_tokens_per_task must have one value per image "
                f"({len(max_tokens_per_task)} != {len(image_list)})"
            )
        if not image_list:
            self.last_request_count = 0
            self.last_elapsed_s = 0.0
            self.last_prompt_tokens = 0
            self.last_generation_tokens = 0
            self.last_generation_tps = 0.0
            return []

        extra_body: dict[str, Any] = {
            "max_tokens": max(1, int(max_tokens or self.max_tokens)),
        }
        if self.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = dict(self.chat_template_kwargs)

        started = time.perf_counter()
        raw_results = self._get_client().invoke_chat_completions_images(
            invoke_url=self.endpoint,
            image_b64_list=image_list,
            model=self.model,
            api_key=self.api_key,
            timeout_s=float(self.timeout_s),
            task_prompts=prompt_list,
            temperature=0.0,
            repetition_penalty=float(self.repetition_penalty),
            extra_body=extra_body,
            max_tokens_per_request=max_tokens_per_task,
            max_retries=max(1, int(self.max_retries)),
            max_429_retries=max(0, int(self.max_429_retries)),
            return_metadata=True,
        )
        elapsed = max(0.0, time.perf_counter() - started)
        prompt_tokens = 0
        generation_tokens = 0
        results: list[dict[str, Any]] = []
        for raw in raw_results:
            if isinstance(raw, Mapping):
                text = str(raw.get("text") or raw.get("content") or "").strip()
                usage = raw.get("usage")
                usage_map = usage if isinstance(usage, Mapping) else {}
                prompt_tokens += _usage_int(usage_map, "prompt_tokens", "input_tokens")
                generation_tokens += _usage_int(
                    usage_map, "completion_tokens", "output_tokens", "generated_tokens"
                )
            else:
                text = str(raw or "").strip()
            results.append(
                {
                    "text": text,
                    "model": self.model,
                    "backend": self.backend,
                    "language": self.language,
                }
            )
        self.last_request_count = len(image_list)
        self.last_elapsed_s = elapsed
        self.last_prompt_tokens = prompt_tokens
        self.last_generation_tokens = generation_tokens
        self.last_generation_tps = generation_tokens / elapsed if elapsed > 0.0 else 0.0
        return results

    def recognize_with_inputs(
        self,
        inputs: Sequence[Mapping[str, Any]],
        prompts: Sequence[str],
        *,
        max_tokens: int | None = None,
        max_tokens_per_task: Sequence[int] | None = None,
    ) -> Sequence[Any]:
        """Send mixed text-only and image chat tasks through one pool.

        Pipeline 6 uses a text-only request for native PDF table regions.  The
        same method also accepts ``{"image_b64": ...}``, which keeps scan-page,
        missing-text, and visual tasks in the same bounded continuous batch.
        Older callers continue to use :meth:`recognize_with_prompts` unchanged.
        """
        input_list = [dict(item) if isinstance(item, Mapping) else {} for item in inputs]
        prompt_list = [str(prompt or "") for prompt in prompts]
        if len(input_list) != len(prompt_list):
            raise ValueError(
                "inputs and prompts must have the same length "
                f"({len(input_list)} != {len(prompt_list)})"
            )
        if max_tokens_per_task is not None and len(max_tokens_per_task) != len(input_list):
            raise ValueError(
                "max_tokens_per_task must have one value per input "
                f"({len(max_tokens_per_task)} != {len(input_list)})"
            )
        if not input_list:
            self.last_request_count = 0
            self.last_elapsed_s = 0.0
            self.last_prompt_tokens = 0
            self.last_generation_tokens = 0
            self.last_generation_tps = 0.0
            return []

        messages_list: list[list[dict[str, Any]]] = []
        for item, prompt in zip(input_list, prompt_list):
            text_input = item.get("text")
            image_b64 = str(item.get("image_b64") or "")
            if text_input is not None:
                # A plain string is intentional: vLLM does not invoke the
                # multimodal processor when a native table is sent this way.
                content: str | list[dict[str, Any]] = (
                    f"{prompt}\n\n{str(text_input)}" if prompt else str(text_input)
                )
            else:
                content_parts: list[dict[str, Any]] = []
                if prompt:
                    content_parts.append({"type": "text", "text": prompt})
                if image_b64:
                    from nemo_retriever.models.nim.nim import _mime_from_b64

                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{_mime_from_b64(image_b64)};base64,{image_b64}"
                            },
                        }
                    )
                content = content_parts
            messages_list.append([{"role": "user", "content": content}])

        extra_body: dict[str, Any] = {
            "max_tokens": max(1, int(max_tokens or self.max_tokens)),
            "repetition_penalty": float(self.repetition_penalty),
        }
        if self.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        merged_extra = dict(extra_body)
        per_request_extra: list[dict[str, Any]] | None = None
        if max_tokens_per_task is not None:
            per_request_extra = []
            for task_max_tokens in max_tokens_per_task:
                request_extra = dict(merged_extra)
                request_extra["max_tokens"] = max(1, int(task_max_tokens))
                per_request_extra.append(request_extra)

        started = time.perf_counter()
        raw_results = self._get_client().invoke_chat_completions(
            invoke_url=self.endpoint,
            messages_list=messages_list,
            model=self.model,
            api_key=self.api_key,
            timeout_s=float(self.timeout_s),
            temperature=0.0,
            extra_body=merged_extra,
            extra_bodies=per_request_extra,
            max_retries=max(1, int(self.max_retries)),
            max_429_retries=max(0, int(self.max_429_retries)),
            return_metadata=True,
        )
        elapsed = max(0.0, time.perf_counter() - started)
        prompt_tokens = 0
        generation_tokens = 0
        results: list[dict[str, Any]] = []
        for raw in raw_results:
            if isinstance(raw, Mapping):
                text = str(raw.get("text") or raw.get("content") or "").strip()
                usage = raw.get("usage")
                usage_map = usage if isinstance(usage, Mapping) else {}
                prompt_tokens += _usage_int(usage_map, "prompt_tokens", "input_tokens")
                generation_tokens += _usage_int(
                    usage_map, "completion_tokens", "output_tokens", "generated_tokens"
                )
            else:
                text = str(raw or "").strip()
            results.append(
                {
                    "text": text,
                    "model": self.model,
                    "backend": self.backend,
                    "language": self.language,
                }
            )
        self.last_request_count = len(input_list)
        self.last_elapsed_s = elapsed
        self.last_prompt_tokens = prompt_tokens
        self.last_generation_tokens = generation_tokens
        self.last_generation_tps = generation_tokens / elapsed if elapsed > 0.0 else 0.0
        return results

    def close(self) -> None:
        client = self._client
        self._client = None
        self._response_cache.clear()
        if client is not None:
            client.shutdown()

    def clear_document_cache(self) -> None:
        """Drop cached image responses while keeping the HTTP client warm."""
        self._response_cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass
class HTTPDetectorBackend:
    """Batched adapter for the existing PP-OCRv6 detector sidecar."""

    endpoint: str
    model: str = "PP-OCRv6_medium_det"
    api_key: str | None = None
    timeout_s: float = 120.0
    batch_size: int = 8
    invoker: BatchInvoker = default_batch_invoker
    max_retries: int = 3
    max_429_retries: int = 2
    max_pool_workers: int | None = None
    # Updated after every call so diagnostics can distinguish a logical
    # detector batch from the number of HTTP chunks used by the transport.
    last_request_count: int = field(default=0, init=False, repr=False)

    def detect(self, images: Sequence[str]) -> Sequence[Any]:
        if not images:
            self.last_request_count = 0
            return []
        max_batch_size = max(1, int(self.batch_size))
        self.last_request_count = (
            len(images) + max_batch_size - 1
        ) // max_batch_size
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout_s": self.timeout_s,
            "batch_size": self.batch_size,
        }
        # Preserve the legacy Option 4 invoker call shape when the adapter
        # uses its defaults; Option 5 explicitly opts into fail-fast values.
        if int(self.max_retries) != 3:
            kwargs["max_retries"] = max(1, int(self.max_retries))
        if int(self.max_429_retries) != 2:
            kwargs["max_429_retries"] = max(0, int(self.max_429_retries))
        if self.max_pool_workers is not None:
            kwargs["max_pool_workers"] = max(1, int(self.max_pool_workers))
        return self.invoker(
            self.endpoint,
            images,
            **kwargs,
        )


@dataclass
class PPOCRv6Adapter:
    """Adapter for the existing split PP-OCRv6 sidecars.

    The sidecar currently exposes ``POST /v1/detect`` and ``POST /v1/recognize``
    with an image-list body.  This adapter deliberately does not import or
    mutate the current ingest ``ppocr.py`` implementation.
    """

    detector_endpoint: str
    recognizer_endpoint: str
    api_key: str | None = None
    timeout_s: float = 120.0
    batch_size: int = 8
    invoker: BatchInvoker = default_batch_invoker
    detector_model: str = "PP-OCRv6_medium_det"
    recognizer_model: str = "PP-OCRv6_medium_rec"

    def detect(self, images: Sequence[str]) -> Sequence[Any]:
        if not images:
            return []
        return self.invoker(
            self.detector_endpoint,
            images,
            api_key=self.api_key,
            timeout_s=self.timeout_s,
            batch_size=self.batch_size,
        )

    def recognize(self, images: Sequence[str]) -> Sequence[Any]:
        if not images:
            return []
        return self.invoker(
            self.recognizer_endpoint,
            images,
            api_key=self.api_key,
            timeout_s=self.timeout_s,
            batch_size=self.batch_size,
        )


@dataclass(frozen=True)
class DetectedBox:
    bbox: Sequence[float]
    score: float | None
    model: str | None = None


@dataclass(frozen=True)
class RecognitionItem:
    text: str
    score: float | None
    bbox: BBox | None = None
    # Keep the endpoint's original coordinate values.  Nemotron may return
    # crop-pixel coordinates rather than normalized coordinates; Option 3
    # maps those values with the crop shape instead of discarding them.
    raw_bbox: Sequence[float] | None = None
    model: str | None = None
    language: str | None = None


def _unwrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("result", "prediction", "output"):
            nested = value.get(key)
            if isinstance(nested, Mapping):
                return nested
        return value
    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], Mapping)
    ):
        return _unwrap(value[0])
    return value


def _as_score(value: Any) -> float | None:
    candidate: Any = value
    if isinstance(value, Mapping):
        for key in ("score", "confidence", "conf", "probability", "rec_score"):
            if value.get(key) is not None:
                candidate = value.get(key)
                break
    try:
        return max(0.0, min(1.0, float(candidate)))
    except (TypeError, ValueError):
        return None


def _raw_box(value: Any) -> Sequence[float] | None:
    if isinstance(value, Mapping):
        for key in ("bbox", "bbox_xyxy", "box", "points", "polygon", "bounding_box"):
            if value.get(key) is not None:
                nested = value.get(key)
                if isinstance(nested, Mapping):
                    nested = nested.get("points") or nested.get("bbox")
                value = nested
                break
    if (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
    ):
        return [float(item) for item in value]
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        points = []
        for point in value:
            if isinstance(point, Mapping):
                point = (point.get("x"), point.get("y"))
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    if point[0] is None or point[1] is None:
                        continue
                    points.append((float(str(point[0])), float(str(point[1]))))
                except (TypeError, ValueError):
                    continue
        if points:
            return [
                min(point[0] for point in points),
                min(point[1] for point in points),
                max(point[0] for point in points),
                max(point[1] for point in points),
            ]
    return None


def detector_boxes(response: Any) -> list[DetectedBox]:
    """Normalize current PP-OCRv6 detector responses without assuming one shape."""
    value = _unwrap(response)
    if isinstance(value, Mapping):
        raw = (
            value.get("boxes")
            or value.get("dt_boxes")
            or value.get("rec_boxes")
            or value.get("detections")
            or []
        )
        scores = value.get("scores") or value.get("dt_scores") or []
        model = str(value.get("model")) if value.get("model") else None
    elif isinstance(value, list):
        raw, scores, model = value, [], None
    else:
        return []
    result: list[DetectedBox] = []
    for index, item in enumerate(raw if isinstance(raw, list) else [raw]):
        bbox = _raw_box(item)
        if bbox is None:
            continue
        score = (
            _as_score(item)
            if isinstance(item, Mapping)
            else (_as_score(scores[index]) if index < len(scores) else None)
        )
        item_model = item.get("model") if isinstance(item, Mapping) else None
        result.append(
            DetectedBox(
                bbox=bbox,
                score=score,
                model=str(item_model or model) if (item_model or model) else None,
            )
        )
    return result


def _text_value(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("text", "ocr_text", "output_text", "generated_text", "rec_text"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        if isinstance(value.get("text_prediction"), Mapping):
            return _text_value(value["text_prediction"])
    return str(value or "").strip() if isinstance(value, str) else ""


def recognition_items(response: Any) -> list[RecognitionItem]:
    """Normalize PP, Tesseract, and integrated Nemotron response shapes."""
    value = _unwrap(response)
    if isinstance(value, Mapping) and isinstance(value.get("text_detections"), list):
        raw_items: list[Any] = list(value.get("text_detections") or [])
    elif isinstance(value, Mapping):
        texts = value.get("texts") or value.get("rec_texts")
        if texts is not None and not isinstance(texts, list):
            texts = [texts]
        if isinstance(texts, list):
            scores = value.get("scores") or value.get("rec_scores") or []
            if not isinstance(scores, list):
                scores = [scores]
            raw_items = [
                {
                    "text": text,
                    "score": scores[index]
                    if index < len(scores)
                    else value.get("score"),
                    "model": value.get("model"),
                }
                for index, text in enumerate(texts)
            ]
        else:
            raw_items = [value]
    elif isinstance(value, list):
        raw_items = list(value)
    else:
        raw_items = []

    result: list[RecognitionItem] = []
    for item in raw_items:
        if isinstance(item, str):
            text = item.strip()
            score = None
            bbox = None
            raw_bbox_value = None
            model = None
            language = None
        elif isinstance(item, Mapping):
            text = _text_value(item)
            prediction = (
                item.get("text_prediction")
                if isinstance(item.get("text_prediction"), Mapping)
                else item
            )
            score = _as_score(prediction)
            bbox = None
            raw_bbox_value = (
                item.get("bounding_box")
                or item.get("bbox")
                or item.get("box")
                or item.get("points")
            )
            if raw_bbox_value is not None:
                raw = _raw_box(raw_bbox_value)
                if raw is not None and max(abs(float(part)) for part in raw) <= 1.5:
                    bbox = clamp_bbox(raw)
            model = (
                str(item.get("model") or item.get("backend"))
                if (item.get("model") or item.get("backend"))
                else None
            )
            language = (
                str(item.get("language") or item.get("lang"))
                if (item.get("language") or item.get("lang"))
                else None
            )
        else:
            continue
        if text:
            result.append(
                RecognitionItem(
                    text=text,
                    score=score,
                    bbox=bbox,
                    raw_bbox=_raw_box(raw_bbox_value),
                    model=model,
                    language=language,
                )
            )
    return result


def first_recognition(response: Any) -> RecognitionItem | None:
    items = recognition_items(response)
    return items[0] if items else None


def _usage_int(usage: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        try:
            value = usage.get(key)
            if value is not None:
                return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


def make_nemotron_backend(
    endpoint: str,
    *,
    api_key: str | None = None,
    language: str | None = None,
    timeout_s: float = 120.0,
    batch_size: int = 8,
    max_pool_workers: int | None = None,
    invoker: BatchInvoker = default_batch_invoker,
) -> HTTPImageBackend:
    """Construct the integrated Nemotron OCR adapter used by Option 4."""
    return HTTPImageBackend(
        endpoint=endpoint,
        model="Nemotron OCR v2",
        language=language,
        api_key=api_key,
        timeout_s=timeout_s,
        batch_size=batch_size,
        max_pool_workers=max_pool_workers,
        invoker=invoker,
    )


def make_vietnamese_recognizer(
    endpoint: str,
    *,
    model: str = "VietOCR vgg_seq2seq",
    api_key: str | None = None,
    timeout_s: float = 120.0,
    batch_size: int = 8,
    max_pool_workers: int | None = None,
    invoker: BatchInvoker = default_batch_invoker,
) -> VietnameseRecognizerBackend:
    """Construct the server-owned, batched Vietnamese OCR adapter.

    ``model`` is only the local fallback label.  A response-provided model
    name takes precedence when Option 3 normalizes recognition items.
    """

    return HTTPImageBackend(
        endpoint=endpoint,
        model=model,
        language="vi",
        api_key=api_key,
        timeout_s=timeout_s,
        batch_size=batch_size,
        max_pool_workers=max_pool_workers,
        invoker=invoker,
        backend="vietocr",
    )


def make_ministral_vlm_backend(
    endpoint: str,
    *,
    model: str = "mistralai/Ministral-3-3B-Instruct-2512",
    api_key: str | None = None,
    timeout_s: float = 120.0,
    batch_size: int = 8,
    max_pool_workers: int | None = None,
    max_tokens: int = 1024,
    max_retries: int = 2,
    max_429_retries: int = 1,
    task_prompt: str | None = None,
) -> VLLMImageBackend:
    """Construct the server-owned Ministral 3B FP8 OCR adapter.

    The endpoint must expose an OpenAI-compatible ``/v1/chat/completions``
    route. Pipeline 7 sends semantic text/title/table/visual crops and
    full-page scan/layout fallbacks through this endpoint; the persistent
    client pool provides bounded concurrent inference across document units.
    """

    return VLLMImageBackend(
        endpoint=endpoint,
        model=model,
        language=None,
        backend="ministral_vlm",
        api_key=api_key,
        timeout_s=timeout_s,
        batch_size=max(1, int(batch_size)),
        max_pool_workers=(
            max(1, int(max_pool_workers))
            if max_pool_workers is not None
            else None
        ),
        task_prompt=task_prompt
        or (
            "Bạn là engine OCR. Chép chính xác toàn bộ chữ nhìn thấy trong ảnh, "
            "giữ nguyên dấu tiếng Việt, số, ký hiệu và thứ tự dòng. Có thể có "
            "nhiều dòng trong cùng một vùng; không được tự ý rút gọn, dịch, "
            "diễn giải hoặc thêm Markdown. Chỉ trả về văn bản OCR; ảnh không "
            "có chữ rõ ràng thì trả về chuỗi rỗng."
        ),
        max_retries=max(1, int(max_retries)),
        max_429_retries=max(0, int(max_429_retries)),
        cache_images=True,
        max_tokens=max(256, int(max_tokens)),
        repetition_penalty=1.0,
    )


def make_qwen35_vlm_backend(
    endpoint: str,
    *,
    model: str = "AxionML/Qwen3.5-2B-NVFP4",
    api_key: str | None = None,
    timeout_s: float = 120.0,
    batch_size: int = 25,
    max_pool_workers: int | None = None,
    max_tokens: int = 1536,
    max_retries: int = 2,
    max_429_retries: int = 1,
    task_prompt: str | None = None,
) -> VLLMImageBackend:
    """Construct the Pipeline 6 Qwen 3.5 image backend.

    This factory is separate from ``make_ministral_vlm_backend`` so the
    existing Pipeline 7 prompt, cache, and request behavior remain unchanged.
    """

    return VLLMImageBackend(
        endpoint=endpoint,
        model=model,
        language=None,
        backend="qwen35_vlm",
        api_key=api_key,
        timeout_s=timeout_s,
        batch_size=max(1, int(batch_size)),
        max_pool_workers=(
            max(1, int(max_pool_workers)) if max_pool_workers is not None else None
        ),
        task_prompt=task_prompt
        or (
            "Bạn là OCR. Chỉ trả về nội dung nhìn thấy trong ảnh, không giải thích."
        ),
        max_retries=max(1, int(max_retries)),
        max_429_retries=max(0, int(max_429_retries)),
        cache_images=False,
        max_tokens=max(128, int(max_tokens)),
        repetition_penalty=1.0,
        chat_template_kwargs={"enable_thinking": False},
    )


def make_tesseract_backend(
    endpoint: str,
    *,
    language: str | None = "vie",
    psm: int | str = 7,
    api_key: str | None = None,
    timeout_s: float = 120.0,
    batch_size: int = 8,
    max_retries: int = 1,
    max_429_retries: int = 0,
    invoker: BatchInvoker = default_batch_invoker,
) -> HTTPImageBackend:
    """Construct Tesseract with request-scoped language and page segmentation."""
    request_payload: dict[str, Any] = {"psm": str(psm)}
    if language:
        request_payload["language"] = language
    return HTTPImageBackend(
        endpoint=endpoint,
        model="tesseract-5",
        language=language,
        api_key=api_key,
        timeout_s=timeout_s,
        batch_size=batch_size,
        max_retries=max_retries,
        max_429_retries=max_429_retries,
        request_payload=request_payload,
        invoker=invoker,
    )
