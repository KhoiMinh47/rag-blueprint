# SPDX-License-Identifier: Apache-2.0

"""Document-scope contracts for the Option 5 coordinator."""

from __future__ import annotations

import base64
import io
from typing import Any

import pandas as pd
from PIL import Image

from nemo_retriever.common.modality.ocr.isolated import runtime
from nemo_retriever.common.modality.ocr.isolated.contracts import OCRPageOutput
from nemo_retriever.common.modality.ocr.isolated.option5 import (
    Option5Pipeline,
    option5_line_detector_endpoint,
    option5_vietnamese_endpoint,
)


def _image_b64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (320, 240), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _page(page_number: int, source_path: str) -> dict[str, Any]:
    return {
        "page_number": page_number,
        "page_image": {"image_b64": _image_b64()},
        "page_elements_v3": {
            "detections": [
                {
                    "label_name": "text",
                    "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.25],
                    "reading_order": 0,
                }
            ]
        },
        "metadata": {
            "source_path": source_path,
            "has_text": False,
            "needs_ocr_for_text": True,
        },
        "text": "",
    }


class _BatchBackend:
    model = "Nemotron OCR v2"
    language = None

    def __init__(self, text: str, model: str | None = None) -> None:
        self.text = text
        if model is not None:
            self.model = model
        self.calls: list[list[str]] = []

    def recognize(self, images: list[str]) -> list[dict[str, Any]]:
        self.calls.append(list(images))
        return [
            {
                "text": self.text,
                "score": 0.96,
                "model": self.model,
            }
            for _ in images
        ]


def test_document_coordinator_probe_first_skips_full_nemotron_for_vietnamese() -> None:
    nemotron = _BatchBackend(
        "Đây là một đoạn văn bản tiếng Việt đủ dài để xác định ngôn ngữ."
    )
    vietnamese = _BatchBackend(
        "Đây là kết quả VietOCR chính xác hơn.",
        model="vgg_seq2seq",
    )
    pages = [_page(page_number, "/docs/report.pdf") for page_number in range(1, 7)]

    outputs = Option5Pipeline(nemotron, vietnamese).process_document(
        pages,
        document_key="/docs/report.pdf",
    )

    assert len(outputs) == 6
    # Five probe crops are enough to classify this document.  The fast path
    # sends the six document units directly to VietOCR instead of running a
    # second Nemotron pass over the whole document.
    assert len(nemotron.calls) == 1
    assert len(nemotron.calls[0]) == 5
    assert len(vietnamese.calls) == 1
    assert len(vietnamese.calls[0]) == 6
    assert all(output.language == "vietnamese" for output in outputs)
    assert all(output.text == "Đây là kết quả VietOCR chính xác hơn." for output in outputs)

    diagnostics = outputs[0].timing["document"]
    assert diagnostics["scope"] == "document"
    assert diagnostics["page_count"] == 6
    assert len(diagnostics["probe_pages"]) == 5
    assert diagnostics["probe_unit_count"] == 5
    assert diagnostics["cache_hits"] == 5
    assert diagnostics["direct_vietnamese"] is True
    assert diagnostics["nemotron_input_count"] == 5
    assert diagnostics["nemotron_logical_batches"] == 1
    assert diagnostics["vietnamese_logical_batches"] == 1
    assert diagnostics["vietnamese_input_count"] == 6


def test_option5_prefers_native_vietnamese_batch_endpoint() -> None:
    assert (
        option5_vietnamese_endpoint("http://vietocr-ocr:8000/v1/ocr")
        == "http://vietocr-ocr:8000/v1/ocr/batch"
    )
    assert (
        option5_vietnamese_endpoint("http://recognizer/custom")
        == "http://recognizer/custom"
    )
    assert (
        option5_line_detector_endpoint("http://detector/v1/detect")
        == "http://detector/v1/detect-batch"
    )
    assert (
        option5_line_detector_endpoint("http://detector/custom")
        == "http://detector/custom"
    )


def test_runtime_groups_option5_pages_by_document_and_restores_order(
    monkeypatch: Any,
) -> None:
    calls: list[tuple[str, int]] = []

    class _Runner:
        pipeline_name = "option5_nemotron_language_routed_vietnamese_ocr"

        def process_document(
            self,
            pages: list[dict[str, Any]],
            *,
            document_key: str,
        ) -> list[OCRPageOutput]:
            calls.append((document_key, len(pages)))
            return [
                OCRPageOutput(
                    pipeline=self.pipeline_name,
                    source=self.pipeline_name,
                    model="Nemotron OCR v2",
                    text=f"page {page['page_number']}",
                    language="english",
                )
                for page in pages
            ]

    runner = _Runner()
    monkeypatch.setattr(runtime, "_build_runner", lambda **_kwargs: runner)
    frame = pd.DataFrame(
        [
            {
                "page_number": 2,
                "metadata": {"source_path": "/docs/a.pdf"},
                "text": "",
            },
            {
                "page_number": 1,
                "metadata": {"source_path": "/docs/a.pdf"},
                "text": "",
            },
            {
                "page_number": 1,
                "metadata": {"source_path": "/docs/b.pdf"},
                "text": "",
            },
        ]
    )

    result = runtime.run_isolated_ocr_batch(
        frame,
        ocr_pipeline="pipeline-option5",
    )

    assert calls == [("/docs/a.pdf", 2), ("/docs/b.pdf", 1)]
    assert list(result["text"]) == ["page 2", "page 1", "page 1"]
    assert all(
        metadata["ocr_pipeline"] == "pipeline-option5"
        for metadata in result["metadata"]
    )


def test_runtime_keeps_option5_language_automatic_when_not_configured() -> None:
    runner = runtime._build_runner(
        ocr_pipeline="pipeline-option5",
        line_detector_invoke_url=None,
        ocr_recognizer_invoke_url=None,
        ocr_invoke_url="http://nemotron/v1/ocr",
        vietnamese_ocr_invoke_url="http://vietocr/v1/ocr",
        tesseract_ocr_invoke_url=None,
        api_key=None,
        ocr_lang=None,
        inference_batch_size=8,
        request_timeout_s=120.0,
        scan_ocr_fallback=True,
        scan_ocr_tile_size=1024,
        scan_ocr_tile_overlap=0.15,
        extract_tables=True,
    )
    assert runner.config.language == "auto"

    forced = runtime._build_runner(
        ocr_pipeline="pipeline-option5",
        line_detector_invoke_url=None,
        ocr_recognizer_invoke_url=None,
        ocr_invoke_url="http://nemotron/v1/ocr",
        vietnamese_ocr_invoke_url="http://vietocr/v1/ocr",
        tesseract_ocr_invoke_url=None,
        api_key=None,
        ocr_lang="english",
        inference_batch_size=8,
        request_timeout_s=120.0,
        scan_ocr_fallback=True,
        scan_ocr_tile_size=1024,
        scan_ocr_tile_overlap=0.15,
        extract_tables=True,
    )
    assert forced.config.language == "english"


def test_runtime_enables_option5_detector_with_batched_endpoint() -> None:
    runner = runtime._build_runner(
        ocr_pipeline="pipeline-option5",
        line_detector_invoke_url="http://detector:8000/v1/detect",
        ocr_recognizer_invoke_url=None,
        ocr_invoke_url="http://nemotron/v1/ocr",
        vietnamese_ocr_invoke_url="http://vietocr/v1/ocr",
        tesseract_ocr_invoke_url=None,
        api_key=None,
        ocr_lang=None,
        inference_batch_size=8,
        request_timeout_s=120.0,
        scan_ocr_fallback=True,
        scan_ocr_tile_size=1024,
        scan_ocr_tile_overlap=0.15,
        extract_tables=True,
    )

    assert runner.line_detector is not None
    assert runner.line_detector.endpoint == "http://detector:8000/v1/detect-batch"
    assert runner.line_detector.batch_size == 100
    assert runner.line_detector.max_pool_workers == 1
    assert runner.config.line_detection is True
