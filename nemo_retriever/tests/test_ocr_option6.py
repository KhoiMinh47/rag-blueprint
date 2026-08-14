# SPDX-License-Identifier: Apache-2.0

"""Contract tests for Pipeline 6 Page Elements -> Qwen VLM."""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from nemo_retriever.common.modality.ocr.isolated import runtime
from nemo_retriever.common.modality.ocr.isolated.contracts import OCRPage, OCRPageOutput
from nemo_retriever.common.modality.content_transforms import clean_content_rows
from nemo_retriever.common.modality.ocr.isolated.option6 import (
    OPTION6_MODEL,
    OPTION6_PIPELINE_NAME,
    Option6Config,
    TABLE_TEXT_PROMPT,
    FULL_PAGE_PROMPT,
    VISUAL_PROMPT,
    Option6Pipeline,
    _clean_markdown,
    _clean_visual_label,
    _looks_like_markdown_table,
    _native_table_input,
    _full_page_layout_reason,
    _scan_maskable_visual_regions,
)
from nemo_retriever.common.params.models import ExtractParams
from nemo_retriever.common.schemas.pipeline_spec import PipelineSpec


def _image_b64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (320, 240), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _Backend:
    model = OPTION6_MODEL
    backend = "qwen35_vlm"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[list[str]] = []
        self.last_request_count = 0

    def recognize(self, images: list[str]) -> list[dict[str, Any]]:
        self.calls.append(list(images))
        self.last_request_count = 1 if images else 0
        return [{"text": self.responses[index]} for index in range(len(images))]


class _MixedBackend(_Backend):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.input_calls: list[list[dict[str, Any]]] = []
        self.prompt_calls: list[list[str]] = []
        self.response_index = 0

    def recognize_with_inputs(
        self,
        inputs: list[dict[str, Any]],
        prompts: list[str],
        *,
        max_tokens: int,
        max_tokens_per_task: list[int],
    ) -> list[dict[str, Any]]:
        self.input_calls.append(list(inputs))
        self.prompt_calls.append(list(prompts))
        self.last_request_count = len(inputs)
        start = self.response_index
        self.response_index += len(inputs)
        return [
            {"text": self.responses[index]}
            for index in range(start, min(self.response_index, len(self.responses)))
        ]


def _scan_page() -> dict[str, Any]:
    return {
        "page_number": 1,
        "page_image": {"image_b64": _image_b64()},
        "metadata": {
            "source_path": "/docs/report.pdf",
            "has_text": False,
            "needs_ocr_for_text": True,
        },
        "page_elements_v3": {
            "detections": [
                {
                    "label_name": "text",
                    "bbox_xyxy_norm": [0.05, 0.05, 0.45, 0.30],
                    "reading_order": 0,
                },
                {"label_name": "table", "bbox_xyxy_norm": [0.50, 0.10, 0.95, 0.70]},
                {"label_name": "chart", "bbox_xyxy_norm": [0.10, 0.75, 0.35, 0.95]},
            ]
        },
        "text": "",
    }


def test_option6_batches_text_and_whole_table_and_keeps_visual_crop() -> None:
    text_backend = _Backend(["Nội dung nguyên văn"])
    table_backend = _Backend(["| Cột A | Cột B |\n|---|---|\n| 1 | 2 |"])

    output = Option6Pipeline(text_backend, table_backend).process_page(_scan_page())

    assert output.pipeline == OPTION6_PIPELINE_NAME
    assert output.model == OPTION6_MODEL
    assert output.text == "Nội dung nguyên văn"
    assert output.tables[0]["table_text_format"] == "markdown"
    assert output.tables[0]["text"].startswith("| Cột A")
    assert output.tables[0]["cells"] == []
    assert output.visuals[0]["caption"] == "biểu đồ"
    assert len(text_backend.calls) == 1 and len(text_backend.calls[0]) == 1
    assert len(table_backend.calls) == 1 and len(table_backend.calls[0]) == 1
    assert output.timing["document"]["vlm_batch_size"] == 8
    assert output.timing["document"]["detector_batch_size"] == 128


def test_option6_rejects_page_sized_text_layout_visual_and_allows_skip_label() -> None:
    backend = _MixedBackend(["đọc trang", "sơ đồ"])
    page = _scan_page()
    page["page_elements_v3"]["detections"] = [
        {"label_name": "infographic", "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0]},
        {"label_name": "text", "bbox_xyxy_norm": [0.08, 0.08, 0.80, 0.14]},
        {"label_name": "text", "bbox_xyxy_norm": [0.08, 0.20, 0.80, 0.27]},
    ]

    output = Option6Pipeline(backend, backend).process_page(page)

    assert output.text == "đọc trang"
    assert output.visuals == []
    assert output.timing["document"]["visual_rejected_regions"] == 1
    assert "BỎ QUA" in VISUAL_PROMPT
    assert _clean_visual_label("BỎ QUA", "infographic") == ""


def test_option6_keeps_page_sized_visual_in_scan_ocr_input() -> None:
    assert _scan_maskable_visual_regions(
        [
            {
                "bbox_xyxy_norm": [0.009, 0.005, 0.962, 0.996],
                "label_name": "infographic",
            },
            {
                "bbox_xyxy_norm": [0.10, 0.70, 0.35, 0.90],
                "label_name": "chart",
            },
        ]
    ) == [(0.1, 0.7, 0.35, 0.9)]


def test_option6_escalates_weak_native_page_to_full_page_but_keeps_visual_crop() -> None:
    backend = _MixedBackend(["Đọc đủ nội dung trang", "biểu đồ"])
    page = _scan_page()
    page["text"] = "native"
    page["metadata"] = {
        "source_path": "/docs/weak-native-layout.pdf",
        "has_text": True,
        "needs_ocr_for_text": False,
        "reader_backend": "native_pdf",
    }
    page["page_elements_v3"]["detections"] = [
        {"label_name": "infographic", "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0]},
    ]

    output = Option6Pipeline(backend, backend).process_page(page)

    assert output.timing["full_page_mode"] == "layout"
    assert output.timing["full_page_primary"] is True
    assert output.timing["full_page_reason"].startswith("large_visual:infographic")
    assert output.text == "Đọc đủ nội dung trang"
    assert output.visuals and output.visuals[0]["caption"] == "biểu đồ"
    assert backend.prompt_calls[0][0] == FULL_PAGE_PROMPT
    assert "image_b64" in backend.input_calls[0][0]
    assert output.timing["document"]["layout_full_page_pages"] == 1


def test_option6_visual_bbox_does_not_delete_native_text_without_authoritative_visual_ocr() -> None:
    row = {
        "path": "/docs/native-visual-evidence.pdf",
        "page_number": 1,
        "text": "Text native nằm cạnh hình",
        "metadata": {"ocr_pipeline": "pipeline-option6"},
        "page_elements_v3": {
            "detections": [
                {"label_name": "infographic", "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0]},
            ]
        },
        "images": [
            {
                "label_name": "infographic",
                "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0],
                "caption": "sơ đồ",
                "text": "",
            }
        ],
        "_native_text_spans": [
            {"char": char, "bbox_xyxy_norm": [0.1 + index * 0.01, 0.1, 0.11 + index * 0.01, 0.14]}
            for index, char in enumerate("Text native nằm cạnh hình")
        ],
    }

    cleaned = clean_content_rows(pd.DataFrame([row]))

    assert cleaned.iloc[0]["text"] == "Text native nằm cạnh hình"
    assert cleaned.iloc[0]["metadata"]["cleaning"]["suppressed_native_characters"] == 0


def test_option6_native_pdfium_text_is_kept_and_only_missing_box_reaches_vlm() -> None:
    text_backend = _Backend(["block bị thiếu"])
    table_backend = _Backend([])
    page = _scan_page()
    page["text"] = "Văn bản native"
    page["metadata"] = {
        "source_path": "/docs/native.pdf",
        "has_text": True,
        "needs_ocr_for_text": False,
    }
    page["_native_text_spans"] = [
        {"char": "V", "bbox_xyxy_norm": [0.08, 0.08, 0.12, 0.18]},
    ]
    page["page_elements_v3"]["detections"] = [
        {"label_name": "text", "bbox_xyxy_norm": [0.05, 0.05, 0.20, 0.22]},
        {"label_name": "text", "bbox_xyxy_norm": [0.55, 0.75, 0.90, 0.90]},
    ]

    output = Option6Pipeline(text_backend, table_backend).process_page(page)

    assert output.timing["native_page"] is True
    assert output.text == "Văn bản native\n\nblock bị thiếu"
    assert len(text_backend.calls) == 1
    assert len(text_backend.calls[0]) == 1


def test_option6_native_table_sends_pdfium_text_only_and_keeps_image_fallback() -> None:
    backend = _MixedBackend(["| Tên | Giá |\n|---|---|\n| A | 10 |"])
    page = _scan_page()
    page["text"] = "Native page text"
    page["metadata"] = {
        "source_path": "/docs/native-table.pdf",
        "has_text": True,
        "needs_ocr_for_text": False,
        "reader_backend": "native_pdf",
    }
    page["_native_text_spans"] = [
        {"char": char, "bbox_xyxy_norm": bbox}
        for char, bbox in [
            ("T", [0.12, 0.20, 0.13, 0.23]),
            ("ê", [0.13, 0.20, 0.14, 0.23]),
            ("n", [0.14, 0.20, 0.15, 0.23]),
            ("G", [0.52, 0.20, 0.53, 0.23]),
            ("i", [0.53, 0.20, 0.54, 0.23]),
            ("á", [0.54, 0.20, 0.55, 0.23]),
            ("A", [0.12, 0.32, 0.13, 0.35]),
            ("1", [0.13, 0.32, 0.14, 0.35]),
            ("1", [0.52, 0.32, 0.53, 0.35]),
            ("0", [0.53, 0.32, 0.54, 0.35]),
        ]
    ]
    page["page_elements_v3"]["detections"] = [
        {"label_name": "table", "bbox_xyxy_norm": [0.05, 0.10, 0.95, 0.50]}
    ]

    output = Option6Pipeline(
        backend,
        backend,
        config=Option6Config(native_table_text=True),
    ).process_page(page)

    assert len(backend.input_calls) == 1
    assert "text" in backend.input_calls[0][0]
    assert "image_b64" not in backend.input_calls[0][0]
    assert TABLE_TEXT_PROMPT in backend.prompt_calls[0][0]
    serialized = backend.input_calls[0][0]["text"].splitlines()
    assert json.loads(serialized[0])["cells"][0]["value"] == "Tên"
    assert "ROW" not in backend.input_calls[0][0]["text"]
    assert output.tables[0]["provenance"]["input"] == "pdfium_native_text"
    assert output.tables[0]["provenance"]["prompt"] == "table_markdown_native_text"


def test_option6_native_table_defaults_to_legacy_image_input() -> None:
    backend = _MixedBackend(["| Tên | Giá |\n|---|---|\n| A | 10 |"])
    page = _scan_page()
    page["text"] = "Native page text"
    page["metadata"] = {
        "source_path": "/docs/native-table-default.pdf",
        "has_text": True,
        "needs_ocr_for_text": False,
        "reader_backend": "native_pdf",
    }
    page["_native_text_spans"] = [
        {"char": char, "bbox_xyxy_norm": bbox}
        for char, bbox in [
            ("T", [0.12, 0.20, 0.13, 0.23]),
            ("ê", [0.13, 0.20, 0.14, 0.23]),
            ("n", [0.14, 0.20, 0.15, 0.23]),
            ("G", [0.52, 0.20, 0.53, 0.23]),
            ("i", [0.53, 0.20, 0.54, 0.23]),
            ("á", [0.54, 0.20, 0.55, 0.23]),
            ("A", [0.12, 0.32, 0.13, 0.35]),
            ("1", [0.13, 0.32, 0.14, 0.35]),
            ("1", [0.52, 0.32, 0.53, 0.35]),
            ("0", [0.53, 0.32, 0.54, 0.35]),
        ]
    ]
    page["page_elements_v3"]["detections"] = [
        {"label_name": "table", "bbox_xyxy_norm": [0.05, 0.10, 0.95, 0.50]}
    ]

    output = Option6Pipeline(backend, backend).process_page(page)

    assert len(backend.input_calls) == 1
    assert "image_b64" in backend.input_calls[0][0]
    assert "text" not in backend.input_calls[0][0]
    assert output.tables[0]["provenance"]["input"] == "image_crop"
    assert output.tables[0]["provenance"]["prompt"] == "table_markdown"


def test_option6_native_table_with_pdfium_artifact_uses_image_fallback() -> None:
    page = OCRPage.from_row(
        {
            "page_number": 1,
            "metadata": {"has_text": True, "reader_backend": "native_pdf"},
            "text": "Dependency",
            "_native_text_spans": [
                {"char": "D", "bbox_xyxy_norm": [0.10, 0.20, 0.12, 0.24]},
                {"char": "\ufffd", "bbox_xyxy_norm": [0.12, 0.20, 0.13, 0.24]},
                {"char": "e", "bbox_xyxy_norm": [0.13, 0.20, 0.15, 0.24]},
                *[
                    {"char": "x", "bbox_xyxy_norm": [0.15 + i * 0.01, 0.20, 0.16 + i * 0.01, 0.24]}
                    for i in range(8)
                ],
            ],
        }
    )
    assert _native_table_input(page, [0.05, 0.10, 0.95, 0.50]) is None


def test_option6_scan_table_stays_image_input() -> None:
    backend = _MixedBackend(
        [
            "scan text",
            "| A | B |\n|---|---|\n| 1 | 2 |",
            "biểu đồ",
        ]
    )
    output = Option6Pipeline(backend, backend).process_page(_scan_page())

    table_inputs = [
        item
        for call in backend.input_calls
        for item in call
        if "image_b64" in item
    ]
    assert table_inputs
    assert all("text" not in item for item in table_inputs)


def test_option6_native_table_falls_back_to_image_when_text_markdown_is_invalid() -> None:
    backend = _MixedBackend(["not a table", "| A | B |\n|---|---|\n| 1 | 2 |"])
    page = _scan_page()
    page["text"] = "Native page text"
    page["metadata"] = {
        "source_path": "/docs/native-table-fallback.pdf",
        "has_text": True,
        "needs_ocr_for_text": False,
        "reader_backend": "native_pdf",
    }
    page["_native_text_spans"] = [
        {"char": str(index % 10), "bbox_xyxy_norm": [0.12 + index * 0.01, 0.20, 0.13 + index * 0.01, 0.23]}
        for index in range(10)
    ]
    page["page_elements_v3"]["detections"] = [
        {"label_name": "table", "bbox_xyxy_norm": [0.05, 0.10, 0.95, 0.50]}
    ]

    output = Option6Pipeline(
        backend,
        backend,
        config=Option6Config(native_table_text=True),
    ).process_page(page)

    assert len(backend.input_calls) == 2
    assert "text" in backend.input_calls[0][0]
    assert "image_b64" in backend.input_calls[1][0]
    assert output.tables[0]["provenance"]["input"] == "image_crop_fallback"
    assert output.tables[0]["provenance"]["prompt"] == "table_markdown_image_fallback"


def test_option6_cleans_chatty_markdown_and_rejects_layout_metadata() -> None:
    response = """```markdown
| Tên | Giá |
| --- | --- |
| A | 10 |
```

Giải thích: đã chuyển xong.
"""
    cleaned = _clean_markdown(response)
    assert cleaned == "| Tên | Giá |\n| --- | --- |\n| A | 10 |"
    assert _looks_like_markdown_table(cleaned)
    assert not _looks_like_markdown_table(
        "| row | y | x | value |\n|---|---|---|---|\n| 1 | 0.2 | 0.1-0.2 | Tên |"
    )
    assert _looks_like_markdown_table(
        "| Item | Value |\n|---|---|\n| A | 10 |"
    )


def test_option6_keeps_native_geometry_until_cleaner_suppresses_table_duplicate() -> None:
    row = {
        "path": "/docs/native-table-with-missing-text.pdf",
        "page_number": 1,
        "text": "A 10\nblock bị thiếu",
        "metadata": {"ocr_pipeline": "pipeline-option6"},
        "_native_text_spans": [
            {"char": char, "bbox_xyxy_norm": [0.10 + index * 0.03, 0.20, 0.12 + index * 0.03, 0.24]}
            for index, char in enumerate("A 10")
        ],
    }
    output = OCRPageOutput(
        pipeline=OPTION6_PIPELINE_NAME,
        text="A 10\nblock bị thiếu",
        ocr_text_blocks=[
            {
                "text": "block bị thiếu",
                "bbox_xyxy_norm": [0.10, 0.70, 0.80, 0.78],
                "content_type": "text",
            }
        ],
        tables=[
            {
                "bbox_xyxy_norm": [0.05, 0.10, 0.95, 0.50],
                "text": "| Cột | Giá |\n|---|---|\n| A | 10 |",
                "markdown": "| Cột | Giá |\n|---|---|\n| A | 10 |",
                "table_text_format": "markdown",
            }
        ],
        timing={"native_page": True},
    )

    applied = runtime._apply_page_output(
        row,
        output,
        selector="pipeline-option6",
        extract_text=True,
        extract_tables=True,
    )
    assert "_native_text_spans" in applied

    cleaned = clean_content_rows(pd.DataFrame([applied]))
    assert cleaned.iloc[0]["text"] == ""
    assert [item["text"] for item in cleaned.iloc[0]["_native_text_blocks"]] == [
        "block bị thiếu"
    ]
    assert cleaned.iloc[0]["table"][0]["text"].startswith("| Cột")
    assert cleaned.iloc[0]["metadata"]["cleaning"]["suppressed_native_characters"] == 3


def test_option6_keeps_native_table_text_when_vlm_markdown_is_rejected() -> None:
    row = {
        "path": "/docs/native-table-invalid-vlm.pdf",
        "page_number": 1,
        "text": "A 10",
        "table": [],
        "metadata": {"ocr_pipeline": "pipeline-option6"},
        "page_elements_v3": {
            "detections": [
                {
                    "label_name": "table",
                    "bbox_xyxy_norm": [0.05, 0.10, 0.95, 0.50],
                }
            ]
        },
        "_native_text_spans": [
            {
                "char": char,
                "bbox_xyxy_norm": [
                    0.10 + index * 0.03,
                    0.20,
                    0.12 + index * 0.03,
                    0.24,
                ],
            }
            for index, char in enumerate("A 10")
        ],
    }

    cleaned = clean_content_rows(pd.DataFrame([row]))
    assert cleaned.iloc[0]["text"] == "A 10"
    cleaning = cleaned.iloc[0]["metadata"]["cleaning"]
    assert cleaning["suppressed_native_characters"] == 0
    assert cleaning["table_suppression_requires_text_match"] is True


def test_option6_runtime_groups_document_and_propagates_tables_and_images(monkeypatch: Any) -> None:
    calls: list[int] = []

    class _Runner:
        pipeline_name = OPTION6_PIPELINE_NAME
        model_name = OPTION6_MODEL

        def process_document(self, pages: list[dict[str, Any]], *, document_key: str) -> list[OCRPageOutput]:
            calls.append(len(pages))
            return [
                OCRPageOutput(
                    pipeline=self.pipeline_name,
                    source=self.pipeline_name,
                    model=self.model_name,
                    text=f"page {page['page_number']}",
                    tables=[
                        {
                            "text": "|a|",
                            "table_text_format": "markdown",
                            "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.9],
                        }
                    ],
                    visuals=[{"label": "image", "image_b64": "crop"}],
                    timing={"document": {"scope": "document"}, "native_page": False},
                )
                for page in pages
            ]

    monkeypatch.setattr(runtime, "_build_runner", lambda **_kwargs: _Runner())
    result = runtime.run_isolated_ocr_batch(
        pd.DataFrame(
            [
                {"page_number": 2, "metadata": {"source_path": "/docs/a.pdf"}},
                {"page_number": 1, "metadata": {"source_path": "/docs/a.pdf"}},
            ]
        ),
        ocr_pipeline="pipeline-option6",
        vintern_ocr_invoke_url="http://qwen/v1/chat/completions",
    )

    assert calls == [2]
    assert list(result["text"]) == ["page 2", "page 1"]
    assert result.iloc[0]["table"][0]["text"] == "|a|"
    assert result.iloc[0]["images"][0]["label"] == "image"


def test_option6_selector_and_runtime_require_server_owned_qwen_endpoint() -> None:
    assert PipelineSpec(ocr_pipeline="pipeline-option6").ocr_pipeline == "pipeline-option6"
    params = ExtractParams(
        ocr_pipeline="pipeline-option6",
        page_elements_invoke_url="http://page-elements/v1",
        vintern_ocr_invoke_url="http://qwen/v1/chat/completions",
        use_table_structure=False,
    )
    assert params.ocr_pipeline == "pipeline-option6"
    assert isinstance(
        runtime._build_runner(
            ocr_pipeline="pipeline-option6",
            line_detector_invoke_url=None,
            ocr_recognizer_invoke_url=None,
            ocr_invoke_url=None,
            vietnamese_ocr_invoke_url=None,
            vintern_ocr_invoke_url="http://qwen/v1/chat/completions",
            ministral_vlm_invoke_url=None,
            tesseract_ocr_invoke_url=None,
            api_key=None,
            ocr_lang=None,
            inference_batch_size=25,
            request_timeout_s=120.0,
            scan_ocr_fallback=True,
            scan_ocr_tile_size=1024,
            scan_ocr_tile_overlap=0.15,
            extract_tables=True,
        ),
        # The factory returns the isolated Pipeline 6 coordinator.
        Option6Pipeline,
    )


def test_option6_streaming_resource_overrides_are_pipeline_local() -> None:
    from nemo_retriever.graph.ingestor_runtime import batch_tuning_to_node_overrides

    option6 = batch_tuning_to_node_overrides(
        ExtractParams(
            ocr_pipeline="pipeline-option6",
            page_elements_invoke_url="http://page-elements/v1",
            vintern_ocr_invoke_url="http://qwen/v1/chat/completions",
            use_table_structure=False,
        ),
        embed_params=None,
    )
    assert option6["PDFExtractionActor"]["batch_size"] == 16
    assert option6["PDFExtractionActor"]["concurrency"] == 4
    assert option6["PageElementDetectionActor"]["batch_size"] == 16
    assert "target_num_rows_per_block" not in option6["PageElementDetectionActor"]
    assert option6["Option6PDFProducerConsumer"] == {
        "concurrency": 1,
        "num_cpus": 2.0,
    }

    legacy = batch_tuning_to_node_overrides(
        ExtractParams(
            ocr_pipeline="pipeline-nemotron-ocr",
            page_elements_invoke_url="http://page-elements/v1",
        ),
        embed_params=None,
    )
    assert "Option6PDFProducerConsumer" not in legacy


def test_option6_pdf_producer_consumer_streams_and_restores_page_order(
    monkeypatch: Any,
) -> None:
    from nemo_retriever.operators.extract.ocr import option6_document

    activity_lock = threading.Lock()
    active_renders = 0
    max_active_renders = 0

    def fake_pdf_extraction(batch: pd.DataFrame, **_kwargs: Any) -> pd.DataFrame:
        nonlocal active_renders, max_active_renders
        first_page = int(batch.iloc[0]["page_number"])
        with activity_lock:
            active_renders += 1
            max_active_renders = max(max_active_renders, active_renders)
        try:
            # Force block 2 to reach Qwen before block 1. Final output must
            # still return pages in source order.
            time.sleep(0.04 if first_page == 1 else 0.005)
            output = batch.copy()
            output["page_image"] = [
                {"image_b64": _image_b64()} for _ in range(len(output.index))
            ]
            output["metadata"] = [
                {
                    "source_path": "/docs/stream.pdf",
                    "has_text": False,
                    "needs_ocr_for_text": True,
                }
                for _ in range(len(output.index))
            ]
            return output
        finally:
            with activity_lock:
                active_renders -= 1

    class _Detector:
        def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
            output = batch.copy()
            output["page_elements_v3"] = [
                {"detections": []} for _ in range(len(output.index))
            ]
            return output

    vlm_batches: list[list[int]] = []

    def fake_run_isolated(batch: pd.DataFrame, **_kwargs: Any) -> pd.DataFrame:
        pages = [int(value) for value in batch["page_number"]]
        vlm_batches.append(pages)
        time.sleep(0.01)
        output = batch.copy()
        output["text"] = [f"page {page}" for page in pages]
        diagnostics = {
            "scope": "document",
            "pipeline": "pipeline-option6",
            "document_key": "/docs/stream.pdf",
            "page_count": len(pages),
            "vlm_request_count": len(pages),
            "timing": {
                "total_seconds": 0.01,
                "vlm_seconds": 0.01,
                "vlm_request_seconds": 0.01,
                "vlm_generation_tokens": len(pages),
            },
            "errors": [],
        }
        output["metadata"] = [
            {
                **dict(metadata),
                "ocr_document_diagnostics": dict(diagnostics),
            }
            for metadata in output["metadata"]
        ]
        return output

    monkeypatch.setattr(option6_document, "pdf_extraction", fake_pdf_extraction)
    monkeypatch.setattr(
        option6_document,
        "run_isolated_ocr_batch",
        fake_run_isolated,
    )
    actor = option6_document.Option6PDFProducerConsumerActor(
        extract_kwargs={},
        detect_kwargs={},
        ocr_kwargs={"ocr_pipeline": "pipeline-option6"},
        pdf_extract_batch_size=2,
        stream_batch_size=2,
        pdf_extract_workers=2,
        queue_blocks=1,
    )
    actor._detector = _Detector()
    actor._runner = object()
    result = actor.run(
        pd.DataFrame(
            [
                {
                    "bytes": b"pdf",
                    "path": "/docs/stream.pdf",
                    "page_number": page,
                }
                for page in range(1, 6)
            ]
        )
    )

    assert list(result["page_number"]) == [1, 2, 3, 4, 5]
    assert list(result["text"]) == [f"page {page}" for page in range(1, 6)]
    assert max_active_renders == 2
    assert sorted(page for batch in vlm_batches for page in batch) == [1, 2, 3, 4, 5]
    diagnostics = result.iloc[0]["metadata"]["ocr_document_diagnostics"]
    assert diagnostics["page_count"] == 5
    assert diagnostics["stream_batches"] == 3
    assert diagnostics["producer_consumer"] == "p6_inprocess_pdf_stream"
    assert diagnostics["pdf_extract_workers"] == 2
    assert diagnostics["timing"]["producer_consumer_overlap_seconds"] > 0.0


def test_option6_graph_uses_page_elements_and_skips_table_structure() -> None:
    from nemo_retriever.graph.ingestor_runtime import build_graph

    params = ExtractParams(
        ocr_pipeline="pipeline-option6",
        page_elements_invoke_url="http://page-elements/v1",
        vintern_ocr_invoke_url="http://qwen/v1/chat/completions",
        use_page_elements=True,
        use_table_structure=False,
        extract_images=False,
        extract_infographics=True,
        extract_tables=True,
    )
    graph = build_graph(extraction_mode="pdf", extract_params=params)
    names: list[str] = []

    def visit(node: Any) -> None:
        names.append(getattr(node.operator, "name", node.name))
        for child in getattr(node, "children", []):
            visit(child)

    for root in graph.roots:
        visit(root)

    assert "Option6PDFProducerConsumer" in names
    assert "PDFExtractionActor" not in names
    assert "PageElementDetectionActor" not in names
    assert "Option6Qwen35NVFP4VLMOCR" not in names
    assert "TableStructureActor" not in names


def test_option6_frontend_selector_and_trace_labels_are_wired() -> None:
    root = Path(__file__).parents[1]
    ingest_debug = (
        root / "src/nemo_retriever/service/dashboard/static/views/ingest_debug.jsx"
    ).read_text()
    job_detail = (
        root / "src/nemo_retriever/service/dashboard/static/views/job_detail.jsx"
    ).read_text()
    dashboard = (root / "src/nemo_retriever/service/routers/dashboard.py").read_text()
    assert "value: 'pipeline-option6'" in ingest_debug
    assert "option6_page_detect_qwen35_vlm" in job_detail
    assert "OPTION6_MODEL" in dashboard
