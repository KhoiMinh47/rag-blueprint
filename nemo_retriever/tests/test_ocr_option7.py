# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the restored Pipeline 7 semantic VLM path."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from PIL import Image

from nemo_retriever.common.modality.ocr.isolated import runtime
from nemo_retriever.common.modality.ocr.isolated.option7 import (
    OPTION7_MODEL,
    OPTION7_PIPELINE_NAME,
    Option7Config,
    Option7Pipeline,
)
from nemo_retriever.common.params.models import ExtractParams
from nemo_retriever.common.schemas.pipeline_spec import PipelineSpec
from nemo_retriever.graph.ingestor_runtime import build_graph


def _image_b64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (320, 240), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _MixedVLM:
    model = OPTION7_MODEL
    backend = "ministral_vlm"
    language = None

    def __init__(self) -> None:
        self.input_calls: list[tuple[list[dict[str, Any]], list[str]]] = []
        self.image_calls: list[list[str]] = []

    def recognize(self, images: list[str]) -> list[dict[str, Any]]:
        self.image_calls.append(list(images))
        return [{"text": "fallback text", "model": OPTION7_MODEL} for _ in images]

    def recognize_with_inputs(
        self,
        inputs: list[dict[str, Any]],
        prompts: list[str],
        **_: Any,
    ) -> list[dict[str, Any]]:
        self.input_calls.append((list(inputs), list(prompts)))
        values: list[dict[str, Any]] = []
        for prompt in prompts:
            lowered = prompt.casefold()
            if "bảng" in lowered or "table" in lowered:
                text = "| A | B |\n|---|---|\n| 1 | 2 |"
            elif "đúng một nhãn" in lowered or "bỏ qua" in lowered:
                text = "biểu đồ"
            else:
                text = "Semantic crop text"
            values.append({"text": text, "model": OPTION7_MODEL})
        return values


def _page(
    *,
    reader_backend: str = "scan",
    has_text: bool = False,
    native_text: str = "",
    detections: list[dict[str, Any]] | None = None,
    table_structure: dict[str, Any] | None = None,
    native_spans: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "page_number": 1,
        "page_image": {"image_b64": _image_b64()},
        "page_elements_v3": {
            "detections": detections
            if detections is not None
            else [
                {
                    "label_name": "text",
                    "bbox_xyxy_norm": [0.10, 0.10, 0.90, 0.30],
                    "reading_order": 0,
                }
            ]
        },
        "table_structure_v1": table_structure,
        "metadata": {
            "source_path": "/docs/report.pdf",
            "reader_backend": reader_backend,
            "has_text": has_text,
            "needs_ocr_for_text": not has_text,
        },
        "text": native_text,
        "_native_text_spans": native_spans,
    }


def _table_payload() -> dict[str, Any]:
    return {
        "regions": [
            {
                "table_id": "table-1",
                "bbox_xyxy_norm": [0.05, 0.45, 0.95, 0.85],
                "detections": [
                    {
                        "label_name": "cell",
                        "bbox_xyxy_norm": [0.10, 0.40, 0.50, 0.60],
                    }
                ],
            }
        ]
    }


def test_option7_sends_page_elements_text_and_table_bbox_to_ministral() -> None:
    vlm = _MixedVLM()
    page = _page(
        reader_backend="native_pdf",
        has_text=True,
        native_text="Native heading",
        table_structure=_table_payload(),
        detections=[
            {
                "label_name": "text",
                "bbox_xyxy_norm": [0.10, 0.10, 0.90, 0.30],
                "reading_order": 0,
            },
            {
                "label_name": "table",
                "bbox_xyxy_norm": [0.10, 0.40, 0.90, 0.80],
                "reading_order": 1,
            },
        ],
    )

    output = Option7Pipeline(
        vlm,
        config=Option7Config(classify_visual_regions=False),
    ).process_document([page], document_key="/docs/report.pdf")[0]

    assert output.pipeline == OPTION7_PIPELINE_NAME
    assert output.model == OPTION7_MODEL
    assert output.text == "Native heading\n\nSemantic crop text"
    assert len(output.ocr_text_blocks) == 1
    assert output.ocr_text_blocks[0]["ocr_mode"] == "semantic_crop"
    assert output.ocr_text_blocks[0]["bbox_xyxy_norm"] == [0.1, 0.1, 0.9, 0.3]
    assert len(output.tables) == 1
    assert output.tables[0]["table_id"] == "table-1"
    assert output.tables[0]["bbox_xyxy_norm"] == [0.1, 0.4, 0.9, 0.8]
    assert output.tables[0]["provenance"]["table_structure_enabled"] is False
    assert len(vlm.input_calls) == 1
    assert len(vlm.input_calls[0][0]) == 2
    assert any("bảng" in prompt.casefold() for prompt in vlm.input_calls[0][1])

    diagnostics = output.timing["document"]
    assert diagnostics["pipeline"] == "pipeline-option7"
    assert diagnostics["semantic_ocr"] is True
    assert diagnostics["semantic_text_crop_ocr"] is True
    assert diagnostics["semantic_text_crop_count"] == 1
    assert diagnostics["table_structure_enabled"] is False
    assert diagnostics["table_structure_called"] is False
    assert diagnostics["table_structure_cell_count"] == 0
    assert diagnostics["table_region_count"] == 1
    assert diagnostics["full_page_count"] == 0


def test_option7_scan_uses_full_page_fallback_and_keeps_table_crop() -> None:
    vlm = _MixedVLM()
    page = _page(
        table_structure=_table_payload(),
        detections=[
            {
                "label_name": "text",
                "bbox_xyxy_norm": [0.10, 0.10, 0.90, 0.30],
            },
            {
                "label_name": "table",
                "bbox_xyxy_norm": [0.10, 0.40, 0.90, 0.80],
            },
        ],
    )
    output = Option7Pipeline(
        vlm,
        config=Option7Config(classify_visual_regions=False),
    ).process_page(page)

    assert output.ocr_text_blocks[0]["bbox_xyxy_norm"] == [0.0, 0.0, 1.0, 1.0]
    assert output.ocr_text_blocks[0]["ocr_mode"] == "full_page"
    assert len(output.tables) == 1
    diagnostics = output.timing["document"]
    assert diagnostics["full_page_count"] == 1
    assert diagnostics["semantic_text_crop_count"] == 0
    assert diagnostics["table_crop_count"] == 1
    assert len(vlm.input_calls[0][0]) == 2


def test_option7_native_pdfium_coverage_avoids_duplicate_text_ocr() -> None:
    vlm = _MixedVLM()
    page = _page(
        reader_backend="native_pdf",
        has_text=True,
        native_text="Native text",
        native_spans=[
            {"char": "N", "bbox_xyxy_norm": [0.12, 0.12, 0.14, 0.20]},
        ],
    )
    output = Option7Pipeline(vlm).process_page(page)

    assert output.text == "Native text"
    assert output.ocr_text_blocks == []
    assert output.tables == []
    assert vlm.input_calls == []
    assert output.timing["document"]["semantic_text_crop_count"] == 0


def test_option7_keeps_page_element_visual_as_evidence_without_visual_crop() -> None:
    vlm = _MixedVLM()
    page = _page(
        reader_backend="native_pdf",
        has_text=True,
        native_text="Native text",
        detections=[
            {
                "label_name": "chart",
                "bbox_xyxy_norm": [0.20, 0.25, 0.75, 0.65],
            }
        ],
    )
    output = Option7Pipeline(vlm).process_page(page)

    assert output.visuals == []
    assert output.ocr_text_blocks == []
    assert vlm.input_calls == []
    assert output.timing["document"]["visual_vlm_requests"] == 0
    assert output.timing["document"]["visual_ocr_enabled"] is False


def test_option7_runtime_builds_semantic_ministral_runner() -> None:
    runner = runtime._build_runner(
        ocr_pipeline="pipeline-option7",
        line_detector_invoke_url=None,
        ocr_recognizer_invoke_url=None,
        ocr_invoke_url=None,
        vietnamese_ocr_invoke_url=None,
        vintern_ocr_invoke_url=None,
        ministral_vlm_invoke_url="http://ministral/v1/chat/completions",
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

    assert isinstance(runner, Option7Pipeline)
    assert runner.config.semantic_ocr is True
    assert runner.config.table_structure is False
    assert runner.config.include_visual_regions is False
    assert runner.config.ocr_visual_regions is False
    assert hasattr(runner.text_vlm, "recognize_with_inputs")


def test_option7_selector_and_server_owned_extract_contract() -> None:
    assert PipelineSpec(ocr_pipeline="pipeline-option7").ocr_pipeline == "pipeline-option7"
    params = ExtractParams(
        ocr_pipeline="pipeline-option7",
        page_elements_invoke_url="http://nim-page-elements/v1/detect",
        table_structure_invoke_url="http://nim-table-structure/v1/detect",
        ministral_vlm_invoke_url="http://ministral/v1/chat/completions",
        extract_tables=True,
        use_table_structure=True,
    )
    assert params.use_page_elements is True
    assert params.use_table_structure is True
    assert params.extract_tables is True


def test_option7_graph_uses_page_elements_then_semantic_ocr_without_table_structure() -> None:
    graph = build_graph(
        extraction_mode="pdf",
        extract_params=ExtractParams(
            method="pdfium_hybrid",
            ocr_pipeline="pipeline-option7",
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_infographics=True,
            extract_images=False,
            use_page_elements=True,
            # P7 deliberately bypasses the shared Table Structure stage.
            use_table_structure=True,
            page_elements_invoke_url="http://nim-page-elements/v1/detect",
            table_structure_invoke_url="http://nim-table-structure/v1/detect",
            ministral_vlm_invoke_url="http://ministral/v1/chat/completions",
        ),
    )

    names: list[str] = []

    def visit(node: Any) -> None:
        names.append(str(getattr(node.operator, "name", node.name)))
        for child in node.children:
            visit(child)

    for root in graph.roots:
        visit(root)
    assert "TableStructureActor" not in names
    assert names.index("PageElementDetectionActor") < names.index("Option7MinistralVLMOCR")


def test_option7_dashboard_describes_semantic_table_pipeline_without_table_structure() -> None:
    root = Path(__file__).parents[1]
    ingest_debug = (
        root / "src/nemo_retriever/service/dashboard/static/views/ingest_debug.jsx"
    ).read_text()
    job_detail = (
        root / "src/nemo_retriever/service/dashboard/static/views/job_detail.jsx"
    ).read_text()
    dashboard = (
        root / "src/nemo_retriever/service/routers/dashboard.py"
    ).read_text()
    assert "value: 'pipeline-option7'" in ingest_debug
    assert "semantic OCR" in ingest_debug
    assert "semantic text/title/table crop" in job_detail
    assert "Table Structure tắt" in dashboard
    assert "semantic text/title/table bbox" in dashboard
    assert "Nhận diện cấu trúc bảng · Pipeline 7" in dashboard
    assert '"use_table_structure": False' in dashboard
