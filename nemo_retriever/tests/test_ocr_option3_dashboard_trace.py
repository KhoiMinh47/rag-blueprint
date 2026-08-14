# SPDX-License-Identifier: Apache-2.0

"""Small static checks for Option 3's dashboard trace contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nemo_retriever.service.services.visual_evidence import deduplicate_visual_blocks
from nemo_retriever.service.routers.dashboard import (
    _build_pipeline_trace,
    _trace_model_config,
)

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "nemo_retriever" / "service" / "dashboard" / "static" / "views"


def test_option3_dashboard_uses_nemotron_router_vietocr_labels() -> None:
    ingest = (STATIC / "ingest_debug.jsx").read_text(encoding="utf-8")
    detail = (STATIC / "job_detail.jsx").read_text(encoding="utf-8")
    backend = (ROOT / "src" / "nemo_retriever" / "service" / "routers" / "dashboard.py").read_text(
        encoding="utf-8"
    )

    assert "Option 3 · Nemotron → router → VietOCR" in ingest
    assert "option3_nemotron_language_routed_vietnamese_ocr" in detail
    assert "VietOCR vgg_seq2seq" in detail
    assert "option3_nemotron_language_routed_vietnamese_ocr" in backend
    assert "vietnamese_ocr_endpoint" in backend


def test_dashboard_debug_defaults_to_full_page_pipeline7() -> None:
    ingest = (STATIC / "ingest_debug.jsx").read_text(encoding="utf-8")

    assert "React.useState('pipeline-option7')" in ingest


def test_option3_parent_bbox_fallback_lines_are_not_visual_deduplicated() -> None:
    blocks = [
        {
            "bbox": [0.1, 0.1, 0.9, 0.3],
            "content_type": "text",
            "text": "First distinct line",
            "ocr_source": "option3_nemotron",
            "provenance": {"bbox_fallback": True},
        },
        {
            "bbox": [0.1, 0.1, 0.9, 0.3],
            "content_type": "text",
            "text": "Second distinct line",
            "ocr_source": "option3_nemotron",
            "provenance": {"bbox_fallback": True},
        },
    ]

    assert [block["text"] for block in deduplicate_visual_blocks(blocks)] == [
        "First distinct line",
        "Second distinct line",
    ]


def test_dashboard_trace_model_config_honors_request_scoped_option5(
    monkeypatch,
) -> None:
    class _Request:
        app = SimpleNamespace(state=SimpleNamespace(config=SimpleNamespace()))

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.get_pipeline_configs",
        lambda: {
            "batch": {
                "extract_params": {
                    "ocr_pipeline": None,
                    "extract_text": True,
                },
                "embed_params": {},
                "nim_endpoints": {
                    "ocr_invoke_url": "http://nim-ocr/v1/ocr",
                    "vietnamese_ocr_invoke_url": "http://vietocr/v1/ocr",
                    "line_detector_invoke_url": "http://ppocr/v1/detect",
                    "ocr_recognizer_invoke_url": "http://ppocr/v1/recognize",
                },
            }
        },
    )

    config = _trace_model_config(_Request(), selector_override="pipeline-option5")

    assert config["ocr_backend"] == "option5_nemotron_language_routed_vietnamese_ocr"
    assert config["ocr_models"]["vietnamese_recognizer"] == "VietOCR vgg_seq2seq"
    assert config["extract_params"]["ocr_pipeline"] == "pipeline-option5"


def test_dashboard_trace_uses_document_diagnostics_when_rows_are_not_retained(
    monkeypatch,
) -> None:
    class _Request:
        app = SimpleNamespace(state=SimpleNamespace(config=SimpleNamespace()))

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.get_pipeline_configs",
        lambda: {
            "batch": {
                "extract_params": {"ocr_pipeline": None},
                "embed_params": {},
                "nim_endpoints": {
                    "ocr_invoke_url": "http://nim-ocr/v1/ocr",
                    "vietnamese_ocr_invoke_url": "http://vietocr/v1/ocr",
                },
            }
        },
    )

    document = SimpleNamespace(
        id="doc",
        status=SimpleNamespace(value="completed"),
        filename="sample.pdf",
        submitted_at="2026-01-01T00:00:00Z",
        result_rows=24,
        pipeline_diagnostics={
            "scope": "document",
            "ocr_pipeline": "pipeline-option5",
            "page_count": 1,
        },
    )
    job = SimpleNamespace(job_id="job")

    trace = _build_pipeline_trace(_Request(), job=job, document=document, rows=[])

    assert trace["config"]["ocr_backend"] == "option5_nemotron_language_routed_vietnamese_ocr"
    assert trace["config"]["extract_params"]["ocr_pipeline"] == "pipeline-option5"
    assert trace["pipeline_diagnostics"]["ocr_pipeline"] == "pipeline-option5"
    assert trace["file"]["pages"] == 1
    assert trace["file"]["result_rows"] == 24
    assert trace["file"]["stages"][-1]["status"] == "not_retained"
