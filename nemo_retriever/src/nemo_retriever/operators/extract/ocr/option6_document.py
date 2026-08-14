# SPDX-License-Identifier: Apache-2.0

"""Ray operator for the document-scoped Pipeline 6 OCR coordinator."""

from __future__ import annotations

import copy
import logging
import queue
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections.abc import Mapping
from typing import Any

import pandas as pd

from nemo_retriever.common.modality.ocr.isolated.option6 import (
    OPTION6_PDF_EXTRACT_BATCH_SIZE,
    OPTION6_PDF_EXTRACT_WORKERS,
    OPTION6_SELECTOR,
    OPTION6_STREAM_BATCH_SIZE,
    OPTION6_STREAM_QUEUE_BLOCKS,
    OPTION6_STREAMING_ENABLED,
)
from nemo_retriever.common.modality.ocr.isolated.runtime import (
    _build_runner,
    run_isolated_ocr_batch,
)
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator
from nemo_retriever.operators.extract.page_elements.cpu_actor import (
    PageElementDetectionCPUActor,
)
from nemo_retriever.operators.extract.pdf.extract import pdf_extraction


logger = logging.getLogger(__name__)

_STREAM_ORDER_COLUMN = "_option6_stream_order"


class Option6DocumentOCRActor(AbstractOperator, CPUOperator):
    """Consume ready Pipeline 6 page blocks with one persistent Qwen pool."""

    # The dedicated PDF producer/consumer owns streaming. Keep this fallback
    # document actor global so disabling or bypassing the composite restores
    # the previous quality-preserving whole-document behavior.
    REQUIRES_GLOBAL_BATCH = True
    GLOBAL_BATCH_GROUP_KEYS: tuple[str, ...] = ()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pipeline_kwargs = dict(kwargs)
        self._runner = None
        self._stream_diagnostics: dict[str, dict[str, Any]] = {}

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        if not OPTION6_STREAMING_ENABLED:
            return run_isolated_ocr_batch(data, **self._pipeline_kwargs)
        if self._runner is None:
            self._runner = self._build_persistent_runner()
        output = run_isolated_ocr_batch(
            data,
            **self._pipeline_kwargs,
            _runner=self._runner,
        )
        return self._attach_stream_diagnostics(output)

    def _build_persistent_runner(self) -> Any:
        return _build_option6_runner(self._pipeline_kwargs)

    def _attach_stream_diagnostics(self, output: Any) -> Any:
        if not isinstance(output, pd.DataFrame) or output.empty:
            return output

        batch_diagnostics: dict[str, dict[str, Any]] = {}
        for metadata in output.get("metadata", []):
            if not isinstance(metadata, Mapping):
                continue
            candidate = metadata.get("ocr_document_diagnostics")
            if not isinstance(candidate, Mapping):
                continue
            document_key = str(candidate.get("document_key") or "option6-document")
            batch_diagnostics.setdefault(document_key, dict(candidate))

        for document_key, current in batch_diagnostics.items():
            previous = self._stream_diagnostics.get(document_key)
            self._stream_diagnostics[document_key] = _merge_stream_diagnostics(
                previous,
                current,
            )

        rows: list[dict[str, Any]] = []
        for _, series in output.iterrows():
            row = series.to_dict()
            metadata = row.get("metadata")
            metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
            candidate = metadata.get("ocr_document_diagnostics")
            document_key = (
                str(candidate.get("document_key") or "option6-document")
                if isinstance(candidate, Mapping)
                else "option6-document"
            )
            cumulative = self._stream_diagnostics.get(document_key)
            if cumulative is not None:
                metadata["ocr_document_diagnostics"] = copy.deepcopy(cumulative)
                timing = metadata.get("ocr_timing")
                timing = dict(timing) if isinstance(timing, Mapping) else {}
                timing["document"] = copy.deepcopy(cumulative)
                metadata["ocr_timing"] = timing
            row["metadata"] = metadata
            rows.append(row)
        return pd.DataFrame(rows).reset_index(drop=True)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class Option6PDFProducerConsumerActor(AbstractOperator, CPUOperator):
    """Pipeline-6-only PDF render -> detect -> VLM producer/consumer.

    The frontend executor runs graph nodes sequentially over the whole
    dataframe. Keeping PDF extraction, Page Elements, and Qwen as separate
    nodes therefore creates two full-document barriers. This actor owns those
    three existing P6 operations and overlaps them without changing their
    output contracts or the graph used by any other OCR selector.
    """

    REQUIRES_GLOBAL_BATCH = True
    GLOBAL_BATCH_GROUP_KEYS: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        extract_kwargs: Mapping[str, Any],
        detect_kwargs: Mapping[str, Any],
        ocr_kwargs: Mapping[str, Any],
        pdf_extract_batch_size: int = OPTION6_PDF_EXTRACT_BATCH_SIZE,
        stream_batch_size: int = OPTION6_STREAM_BATCH_SIZE,
        pdf_extract_workers: int = OPTION6_PDF_EXTRACT_WORKERS,
        queue_blocks: int = OPTION6_STREAM_QUEUE_BLOCKS,
    ) -> None:
        constructor_kwargs = {
            "extract_kwargs": dict(extract_kwargs),
            "detect_kwargs": dict(detect_kwargs),
            "ocr_kwargs": dict(ocr_kwargs),
            "pdf_extract_batch_size": max(1, int(pdf_extract_batch_size)),
            "stream_batch_size": max(1, int(stream_batch_size)),
            "pdf_extract_workers": max(1, int(pdf_extract_workers)),
            "queue_blocks": max(1, int(queue_blocks)),
        }
        super().__init__(**constructor_kwargs)
        self._extract_kwargs = constructor_kwargs["extract_kwargs"]
        self._detect_kwargs = constructor_kwargs["detect_kwargs"]
        self._ocr_kwargs = constructor_kwargs["ocr_kwargs"]
        self._pdf_extract_batch_size = int(
            constructor_kwargs["pdf_extract_batch_size"]
        )
        self._stream_batch_size = int(constructor_kwargs["stream_batch_size"])
        self._pdf_extract_workers = int(constructor_kwargs["pdf_extract_workers"])
        self._queue_blocks = int(constructor_kwargs["queue_blocks"])
        # Build network clients lazily in the actual executor process. Graph
        # construction itself must not allocate thread pools or connections.
        self._detector: PageElementDetectionCPUActor | None = None
        self._runner: Any | None = None

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return data
        if not OPTION6_STREAMING_ENABLED:
            return self._run_sequential(data)
        return self._run_streaming(data)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def _detector_actor(self) -> PageElementDetectionCPUActor:
        if self._detector is None:
            self._detector = PageElementDetectionCPUActor(**self._detect_kwargs)
        return self._detector

    def _persistent_runner(self) -> Any:
        if self._runner is None:
            self._runner = _build_option6_runner(self._ocr_kwargs)
        return self._runner

    def _run_sequential(self, data: pd.DataFrame) -> pd.DataFrame:
        rendered = _as_dataframe(pdf_extraction(data, **self._extract_kwargs))
        detected = _as_dataframe(self._detector_actor()(rendered))
        return run_isolated_ocr_batch(
            detected,
            **self._ocr_kwargs,
            _runner=self._persistent_runner(),
        )

    def _run_streaming(self, data: pd.DataFrame) -> pd.DataFrame:
        started = time.perf_counter()
        source = data.copy().reset_index(drop=True)
        source[_STREAM_ORDER_COLUMN] = list(range(len(source.index)))
        blocks = [
            source.iloc[offset : offset + self._pdf_extract_batch_size].copy()
            for offset in range(0, len(source.index), self._pdf_extract_batch_size)
        ]
        if not blocks:
            return source.drop(columns=[_STREAM_ORDER_COLUMN], errors="ignore")

        ready: queue.Queue[Any] = queue.Queue(maxsize=self._queue_blocks)
        sentinel = object()
        stop_event = threading.Event()
        output_blocks: list[pd.DataFrame] = []
        stream_diagnostics: dict[str, dict[str, Any]] = {}
        consumer_errors: list[BaseException] = []
        first_consumer_at: float | None = None
        consumer_active_seconds = 0.0

        def consume_ready_blocks() -> None:
            nonlocal consumer_active_seconds, first_consumer_at
            while True:
                item = ready.get()
                try:
                    if item is sentinel:
                        return
                    if stop_event.is_set():
                        continue
                    if first_consumer_at is None:
                        first_consumer_at = time.perf_counter()
                    batch_started = time.perf_counter()
                    output = run_isolated_ocr_batch(
                        item,
                        **self._ocr_kwargs,
                        _runner=self._persistent_runner(),
                    )
                    consumer_active_seconds += time.perf_counter() - batch_started
                    output = _as_dataframe(output)
                    output_blocks.append(output)
                    _accumulate_output_diagnostics(stream_diagnostics, output)
                except BaseException as exc:  # noqa: BLE001 - cross-thread handoff
                    consumer_errors.append(exc)
                    stop_event.set()
                finally:
                    ready.task_done()

        consumer = threading.Thread(
            target=consume_ready_blocks,
            name="option6-vlm-consumer",
            daemon=True,
        )
        consumer.start()

        render_worker_seconds = 0.0
        detect_seconds = 0.0
        producer_error: BaseException | None = None
        producer_done_at = started

        def render_block(block: pd.DataFrame) -> tuple[pd.DataFrame, float]:
            block_started = time.perf_counter()
            rendered = _as_dataframe(
                pdf_extraction(block, **self._extract_kwargs)
            )
            return rendered, time.perf_counter() - block_started

        def put_ready(block: pd.DataFrame) -> None:
            while not stop_event.is_set():
                try:
                    ready.put(block, timeout=0.1)
                    return
                except queue.Full:
                    continue
            if consumer_errors:
                raise RuntimeError("Pipeline 6 VLM consumer failed") from consumer_errors[0]
            raise RuntimeError("Pipeline 6 producer/consumer stopped")

        try:
            block_iterator = iter(enumerate(blocks))
            worker_count = min(self._pdf_extract_workers, len(blocks))
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="option6-pdf-render",
            ) as executor:
                pending: dict[Any, int] = {}

                def submit_next() -> bool:
                    try:
                        block_index, block = next(block_iterator)
                    except StopIteration:
                        return False
                    pending[executor.submit(render_block, block)] = block_index
                    return True

                for _ in range(worker_count):
                    submit_next()

                while pending:
                    completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                    for future in sorted(completed, key=lambda item: pending[item]):
                        pending.pop(future)
                        rendered, elapsed = future.result()
                        render_worker_seconds += elapsed
                        detect_started = time.perf_counter()
                        detected = _as_dataframe(self._detector_actor()(rendered))
                        detect_seconds += time.perf_counter() - detect_started
                        for offset in range(
                            0,
                            len(detected.index),
                            self._stream_batch_size,
                        ):
                            put_ready(
                                detected.iloc[
                                    offset : offset + self._stream_batch_size
                                ].copy()
                            )
                        submit_next()
            producer_done_at = time.perf_counter()
        except BaseException as exc:  # noqa: BLE001 - join consumer before re-raise
            producer_error = exc
            stop_event.set()
            producer_done_at = time.perf_counter()
        finally:
            # The consumer keeps draining after an error, so this bounded put
            # cannot deadlock behind a full queue.
            ready.put(sentinel)
            consumer.join()

        finished = time.perf_counter()
        if consumer_errors:
            raise RuntimeError("Pipeline 6 VLM consumer failed") from consumer_errors[0]
        if producer_error is not None:
            raise producer_error
        if not output_blocks:
            return source.iloc[0:0].drop(columns=[_STREAM_ORDER_COLUMN], errors="ignore")

        result = pd.concat(output_blocks, ignore_index=True, sort=False)
        if _STREAM_ORDER_COLUMN in result.columns:
            result = result.sort_values(
                _STREAM_ORDER_COLUMN,
                kind="stable",
            ).drop(columns=[_STREAM_ORDER_COLUMN])
        result = result.reset_index(drop=True)

        wall_seconds = finished - started
        first_submit_seconds = (
            max(0.0, first_consumer_at - started)
            if first_consumer_at is not None
            else 0.0
        )
        overlap_seconds = (
            max(0.0, min(producer_done_at, finished) - first_consumer_at)
            if first_consumer_at is not None
            else 0.0
        )
        stream_metrics = {
            "producer_consumer": "p6_inprocess_pdf_stream",
            "streaming_enabled": True,
            "stream_batch_size": self._stream_batch_size,
            "stream_queue_blocks": self._queue_blocks,
            "pdf_extract_batch_size": self._pdf_extract_batch_size,
            "pdf_extract_workers": self._pdf_extract_workers,
            "pdf_render_worker_seconds": render_worker_seconds,
            "page_elements_seconds": detect_seconds,
            "producer_seconds": max(0.0, producer_done_at - started),
            "consumer_active_seconds": consumer_active_seconds,
            "first_vlm_submit_seconds": first_submit_seconds,
            "producer_consumer_overlap_seconds": overlap_seconds,
            "stream_wall_seconds": wall_seconds,
        }
        return _attach_final_stream_diagnostics(
            result,
            stream_diagnostics,
            stream_metrics,
        )


def _build_option6_runner(values: Mapping[str, Any]) -> Any:
    return _build_runner(
        ocr_pipeline=str(values.get("ocr_pipeline") or OPTION6_SELECTOR),
        line_detector_invoke_url=values.get("line_detector_invoke_url"),
        ocr_recognizer_invoke_url=values.get("ocr_recognizer_invoke_url"),
        ocr_invoke_url=values.get("ocr_invoke_url"),
        vietnamese_ocr_invoke_url=values.get("vietnamese_ocr_invoke_url"),
        vintern_ocr_invoke_url=values.get("vintern_ocr_invoke_url"),
        ministral_vlm_invoke_url=values.get("ministral_vlm_invoke_url"),
        tesseract_ocr_invoke_url=values.get("tesseract_ocr_invoke_url"),
        api_key=values.get("ocr_api_key") or values.get("api_key"),
        ocr_lang=values.get("ocr_lang"),
        inference_batch_size=max(1, int(values.get("inference_batch_size") or 1)),
        request_timeout_s=float(values.get("request_timeout_s") or 120.0),
        scan_ocr_fallback=bool(values.get("scan_ocr_fallback", True)),
        scan_ocr_tile_size=int(values.get("scan_ocr_tile_size") or 1024),
        scan_ocr_tile_overlap=float(values.get("scan_ocr_tile_overlap") or 0.15),
        extract_tables=bool(values.get("extract_tables", True)),
    )


def _as_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if value is None:
        return pd.DataFrame()
    return pd.DataFrame(value)


def _accumulate_output_diagnostics(
    accumulated: dict[str, dict[str, Any]],
    output: pd.DataFrame,
) -> None:
    seen: set[str] = set()
    for metadata in output.get("metadata", []):
        if not isinstance(metadata, Mapping):
            continue
        candidate = metadata.get("ocr_document_diagnostics")
        if not isinstance(candidate, Mapping):
            continue
        document_key = str(candidate.get("document_key") or "option6-document")
        if document_key in seen:
            continue
        seen.add(document_key)
        accumulated[document_key] = _merge_stream_diagnostics(
            accumulated.get(document_key),
            candidate,
        )


def _attach_final_stream_diagnostics(
    output: pd.DataFrame,
    accumulated: Mapping[str, Mapping[str, Any]],
    stream_metrics: Mapping[str, Any],
) -> pd.DataFrame:
    finalized: dict[str, dict[str, Any]] = {}
    for document_key, source in accumulated.items():
        diagnostics = copy.deepcopy(dict(source))
        diagnostics.update(dict(stream_metrics))
        timing = diagnostics.get("timing")
        timing = dict(timing) if isinstance(timing, Mapping) else {}
        timing["chunk_total_seconds"] = float(timing.get("total_seconds", 0.0) or 0.0)
        timing["total_seconds"] = float(stream_metrics.get("stream_wall_seconds", 0.0) or 0.0)
        for key in (
            "pdf_render_worker_seconds",
            "page_elements_seconds",
            "producer_seconds",
            "consumer_active_seconds",
            "first_vlm_submit_seconds",
            "producer_consumer_overlap_seconds",
            "stream_wall_seconds",
        ):
            timing[key] = float(stream_metrics.get(key, 0.0) or 0.0)
        diagnostics["timing"] = timing
        finalized[str(document_key)] = diagnostics

    rows: list[dict[str, Any]] = []
    for _, series in output.iterrows():
        row = series.to_dict()
        metadata = row.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        current = metadata.get("ocr_document_diagnostics")
        document_key = (
            str(current.get("document_key") or "option6-document")
            if isinstance(current, Mapping)
            else "option6-document"
        )
        diagnostics = finalized.get(document_key)
        if diagnostics is not None:
            metadata["ocr_document_diagnostics"] = copy.deepcopy(diagnostics)
            ocr_timing = metadata.get("ocr_timing")
            ocr_timing = dict(ocr_timing) if isinstance(ocr_timing, Mapping) else {}
            ocr_timing["document"] = copy.deepcopy(diagnostics)
            metadata["ocr_timing"] = ocr_timing
        row["metadata"] = metadata
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


__all__ = ["Option6DocumentOCRActor", "Option6PDFProducerConsumerActor"]


_SUM_KEYS = (
    "page_count",
    "text_units",
    "table_regions",
    "visual_regions",
    "native_pages",
    "vlm_request_count",
    "vlm_prompt_tokens",
    "text_vlm_requests",
    "table_vlm_requests",
    "native_table_text_requests",
    "native_table_image_fallbacks",
    "visual_vlm_requests",
)
_TIMING_SUM_KEYS = (
    "total_seconds",
    "crop_seconds",
    "vlm_seconds",
    "vlm_request_seconds",
    "vlm_prompt_tokens",
    "vlm_generation_tokens",
)


def _number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _merge_stream_diagnostics(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Accumulate ready-block metrics without changing page output."""
    merged = copy.deepcopy(dict(previous or current))
    if previous is not None:
        for key in _SUM_KEYS:
            merged[key] = int(_number(previous.get(key)) + _number(current.get(key)))
        previous_timing = previous.get("timing")
        current_timing = current.get("timing")
        previous_timing = previous_timing if isinstance(previous_timing, Mapping) else {}
        current_timing = current_timing if isinstance(current_timing, Mapping) else {}
        timing = dict(current_timing)
        for key in _TIMING_SUM_KEYS:
            timing[key] = _number(previous_timing.get(key)) + _number(current_timing.get(key))
        generation_tokens = _number(timing.get("vlm_generation_tokens"))
        request_seconds = _number(timing.get("vlm_request_seconds"))
        timing["vlm_generation_tps"] = (
            generation_tokens / request_seconds if request_seconds > 0.0 else 0.0
        )
        merged["timing"] = timing
        merged["errors"] = list(previous.get("errors") or []) + list(current.get("errors") or [])

    merged.update(
        {
            "scope": "document",
            "pipeline": OPTION6_SELECTOR,
            "streaming_enabled": True,
            "stream_batch_size": int(OPTION6_STREAM_BATCH_SIZE),
            "stream_batches": int(_number((previous or {}).get("stream_batches"))) + 1,
            "producer_consumer": "ray_block_stream",
        }
    )
    return merged
