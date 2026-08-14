# SPDX-License-Identifier: Apache-2.0

"""Service/graph wiring tests for the opt-in OCR selectors."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from nemo_retriever.common.modality.ocr.isolated import runtime
from nemo_retriever.common.modality.ocr.isolated.contracts import OCRPageOutput
from nemo_retriever.common.params import ExtractParams
from nemo_retriever.common.schemas.pipeline_spec import PipelineSpec
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.service.services.pipeline_executor import (
    _build_graph_ingestor_from_spec,
)


def _node_names(graph: Any) -> list[str]:
    names: list[str] = []

    def visit(node: Any) -> None:
        names.append(str(node.name))
        for child in node.children:
            visit(child)

    for root in graph.roots:
        visit(root)
    return names


def _base_extract() -> dict[str, Any]:
    return {
        "official_ppocr_invoke_url": "http://ppocrv6-official/v1/ocr",
        "paddleocr_vl_invoke_url": "http://paddleocr-vl/layout-parsing",
        "page_elements_invoke_url": "http://page-elements/v1",
        "table_structure_invoke_url": "http://table-structure/v1",
        "line_detector_invoke_url": "http://ppocr-det/v1",
        "ocr_recognizer_invoke_url": "http://ppocr-rec/v1",
        "ocr_invoke_url": "http://nemotron/v1",
        "tesseract_ocr_invoke_url": "http://tesseract/v1",
        "vintern_ocr_invoke_url": "http://vintern/v1",
        "vietnamese_ocr_invoke_url": "http://vietocr/v1",
    }


@pytest.mark.parametrize("selector", ["pipeline-ppocrv6", "pipeline-option3", "pipeline-option4"])
def test_pipeline_spec_accepts_opt_in_selectors(selector: str) -> None:
    assert PipelineSpec(ocr_pipeline=selector).ocr_pipeline == selector


def test_extract_params_default_has_no_isolated_selector() -> None:
    assert ExtractParams().ocr_pipeline is None


def test_option3_worker_wiring_requires_nemotron_and_vietnamese_endpoint() -> None:
    ingestor, mode, _ = _build_graph_ingestor_from_spec(
        "document.pdf",
        b"%PDF-1.7",
        _base_extract(),
        None,
        {"ocr_pipeline": "pipeline-option3", "stage_order": ["extract"]},
    )
    assert mode == "pdf"
    assert ingestor._extract_params.ocr_pipeline == "pipeline-option3"
    assert ingestor._extract_params.ocr_invoke_url == "http://nemotron/v1"
    assert ingestor._extract_params.vietnamese_ocr_invoke_url == "http://vietocr/v1"


def test_option3_does_not_require_detector_or_ppocr_recognizer() -> None:
    extract = {
        "page_elements_invoke_url": "http://page-elements/v1",
        "table_structure_invoke_url": "http://table-structure/v1",
        "ocr_invoke_url": "http://nemotron/v1",
        "vietnamese_ocr_invoke_url": "http://vietocr/v1",
    }
    ingestor, _, _ = _build_graph_ingestor_from_spec(
        "document.pdf",
        b"%PDF-1.7",
        extract,
        None,
        {"ocr_pipeline": "pipeline-option3", "stage_order": ["extract"]},
    )
    assert ingestor._extract_params.line_detector_invoke_url is None
    assert ingestor._extract_params.ocr_recognizer_invoke_url is None


def test_option3_runtime_builder_uses_only_nemotron_and_vietnamese_endpoint() -> None:
    runner = runtime._build_runner(
        ocr_pipeline="pipeline-option3",
        line_detector_invoke_url=None,
        ocr_recognizer_invoke_url=None,
        ocr_invoke_url="http://nemotron/v1",
        vietnamese_ocr_invoke_url="http://vietocr/v1",
        tesseract_ocr_invoke_url=None,
        api_key=None,
        ocr_lang=None,
        inference_batch_size=4,
        request_timeout_s=30.0,
        scan_ocr_fallback=True,
        scan_ocr_tile_size=1024,
        scan_ocr_tile_overlap=0.15,
        extract_tables=True,
    )
    assert isinstance(runner, runtime.Option3Pipeline)
    assert not hasattr(runner, "line_detector")
    assert runner.nemotron.endpoint == "http://nemotron/v1"
    assert runner.vietnamese_recognizer.endpoint == "http://vietocr/v1"
    assert runner.config.language == "auto"


def test_option4_worker_wiring_keeps_server_fusion_endpoints() -> None:
    ingestor, mode, _ = _build_graph_ingestor_from_spec(
        "document.pdf",
        b"%PDF-1.7",
        _base_extract(),
        None,
        {"ocr_pipeline": "pipeline-option4", "stage_order": ["extract"]},
    )
    assert mode == "pdf"
    assert ingestor._extract_params.ocr_pipeline == "pipeline-option4"
    assert ingestor._extract_params.ocr_invoke_url == "http://nemotron/v1"
    assert ingestor._extract_params.tesseract_ocr_invoke_url is None


def test_option4_defaults_tesseract_to_vietnamese_line_mode() -> None:
    runner = runtime._build_runner(
        ocr_pipeline="pipeline-option4",
        line_detector_invoke_url="http://ppocr-det/v1",
        ocr_recognizer_invoke_url=None,
        ocr_invoke_url="http://nemotron/v1",
        tesseract_ocr_invoke_url="http://tesseract/v1",
        api_key=None,
        ocr_lang=None,
        inference_batch_size=4,
        request_timeout_s=30.0,
        scan_ocr_fallback=True,
        scan_ocr_tile_size=1024,
        scan_ocr_tile_overlap=0.15,
        extract_tables=True,
    )
    assert isinstance(runner, runtime.Option4Pipeline)
    assert runner.config.language == "auto"
    assert runner.config.tesseract_first is True
    assert runner.config.tesseract_language == "vie"
    assert runner.config.language_routing is True
    assert runner.config.language_probe_language == "vie+eng"
    assert runner.config.tesseract_psm == 7
    assert runner.tesseract.language == "vie"
    assert runner.tesseract.request_payload == {"language": "vie", "psm": "7"}
    assert runner.language_probe.language == "vie+eng"
    assert runner.language_probe.request_payload == {"language": "vie+eng", "psm": "7"}
    assert runtime._ocr_language(None) == "vie"
    assert runner.tesseract.max_retries == 1


@pytest.mark.parametrize("requested_language", ["english", "multi"])
def test_option4_ignores_non_vietnamese_language_override(requested_language: str) -> None:
    runner = runtime._build_runner(
        ocr_pipeline="pipeline-option4",
        line_detector_invoke_url="http://ppocr-det/v1",
        ocr_recognizer_invoke_url=None,
        ocr_invoke_url="http://nemotron/v1",
        tesseract_ocr_invoke_url="http://tesseract/v1",
        api_key=None,
        ocr_lang=requested_language,
        inference_batch_size=1,
        request_timeout_s=30.0,
        scan_ocr_fallback=True,
        scan_ocr_tile_size=1024,
        scan_ocr_tile_overlap=0.15,
        extract_tables=True,
    )
    assert isinstance(runner, runtime.Option4Pipeline)
    assert runner.config.language == "auto"
    assert runner.config.tesseract_language == "vie"
    assert runner.config.language_routing is True
    assert runner.config.language_probe_language == "vie+eng"
    assert runner.tesseract.language == "vie"
    assert runner.tesseract.request_payload == {"language": "vie", "psm": "7"}


def test_option3_fails_fast_when_nemotron_or_vietnamese_endpoint_is_missing() -> None:
    with pytest.raises(RuntimeError, match="pipeline-option3 requires"):
        _build_graph_ingestor_from_spec(
            "document.pdf",
            b"%PDF-1.7",
            {"page_elements_invoke_url": "http://page-elements/v1"},
            None,
            {"ocr_pipeline": "pipeline-option3", "stage_order": ["extract"]},
        )


def test_option4_fails_fast_when_fusion_endpoints_are_not_provisioned() -> None:
    with pytest.raises(RuntimeError, match="pipeline-option4 requires"):
        _build_graph_ingestor_from_spec(
            "document.pdf",
            b"%PDF-1.7",
            {"page_elements_invoke_url": "http://page-elements/v1"},
            None,
            {"ocr_pipeline": "pipeline-option4", "stage_order": ["extract"]},
        )


def test_option2_worker_wiring_keeps_the_independent_option3_baseline() -> None:
    ingestor, _, _ = _build_graph_ingestor_from_spec(
        "document.pdf",
        b"%PDF-1.7",
        _base_extract(),
        None,
        {"ocr_pipeline": "pipeline-ppocrv6", "stage_order": ["extract"]},
    )
    assert ingestor._extract_params.ocr_pipeline == "pipeline-ppocrv6"
    assert ingestor._extract_params.use_page_elements is True
    assert ingestor._extract_params.use_table_structure is True
    assert ingestor._extract_params.extract_page_as_image is True
    assert ingestor._extract_params.ocr_invoke_url == "http://nemotron/v1"
    assert ingestor._extract_params.page_elements_invoke_url == "http://page-elements/v1"
    assert ingestor._extract_params.table_structure_invoke_url == "http://table-structure/v1"
    assert ingestor._extract_params.vietnamese_ocr_invoke_url == "http://vietocr/v1"


def test_option5_worker_wiring_skips_redundant_embedded_image_extraction() -> None:
    ingestor, mode, _ = _build_graph_ingestor_from_spec(
        "document.pdf",
        b"%PDF-1.7",
        _base_extract(),
        None,
        {"ocr_pipeline": "pipeline-option5", "stage_order": ["extract"]},
    )
    assert mode == "pdf"
    assert ingestor._extract_params.ocr_pipeline == "pipeline-option5"
    assert ingestor._extract_params.extract_images is False
    assert ingestor._extract_params.extract_page_as_image is True
    assert ingestor._extract_params.use_page_elements is True
    assert ingestor._extract_params.use_table_structure is True


def test_option2_fails_fast_without_vietnamese_or_page_structure_endpoints() -> None:
    with pytest.raises(RuntimeError, match=r"Option 2\) requires page-elements"):
        _build_graph_ingestor_from_spec(
            "document.pdf",
            b"%PDF-1.7",
            {"page_elements_invoke_url": "http://page-elements/v1"},
            None,
            {"ocr_pipeline": "pipeline-ppocrv6", "stage_order": ["extract"]},
        )


def test_default_graph_has_no_isolated_option_operator() -> None:
    graph = build_graph(
        extraction_mode="pdf",
        extract_params=ExtractParams(
            extract_images=False,
            extract_tables=False,
            extract_charts=False,
            page_elements_invoke_url="http://page-elements/v1",
        ),
    )
    names = _node_names(graph)
    assert not any(name.startswith(("Option2", "Option3", "Option4")) for name in names)


def test_option1_default_selector_still_uses_existing_ocr_archetype() -> None:
    graph = build_graph(
        extraction_mode="pdf",
        extract_params=ExtractParams(
            extract_images=False,
            extract_tables=False,
            extract_charts=False,
            page_elements_invoke_url="http://page-elements/v1",
            ocr_invoke_url="http://nemotron/v1",
        ),
    )
    names = _node_names(graph)
    assert "OCRActor" in names
    assert not any(name.startswith(("Option2", "Option3", "Option4")) for name in names)


@pytest.mark.parametrize(
    ("selector", "expected_name"),
    [
        ("pipeline-ppocrv6", "Option2LanguageRoutedOCR"),
        ("pipeline-option3", "Option3NemotronLanguageRoutedVietnameseOCR"),
        ("pipeline-option4", "Option4ParallelOCRFusion"),
    ],
)
def test_isolated_selector_adds_only_its_opt_in_graph_stage(selector: str, expected_name: str) -> None:
    params = ExtractParams(
        ocr_pipeline=selector,
        page_elements_invoke_url="http://page-elements/v1",
        line_detector_invoke_url="http://ppocr-det/v1",
        ocr_recognizer_invoke_url="http://ppocr-rec/v1",
        ocr_invoke_url="http://nemotron/v1",
        tesseract_ocr_invoke_url="http://tesseract/v1",
        vietnamese_ocr_invoke_url=(
            "http://vietocr/v1"
            if selector in {"pipeline-ppocrv6", "pipeline-option3"}
            else None
        ),
    )
    names = _node_names(build_graph(extraction_mode="pdf", extract_params=params))
    assert expected_name in names
    assert not any(
        name in {"OCRActor", "OCRCPUActor", "OCRGPUActor"}
        for name in names
        if name != expected_name
    )


def test_isolated_dataframe_adapter_keeps_table_cells_out_of_text_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOption3:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def process_page(self, _row: Any) -> OCRPageOutput:
            block = {
                "text": "Body",
                "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.2],
                "score": 0.9,
                "confidence": 0.9,
                "source": "option3",
                "model": "PP-OCRv6_medium_rec",
                "content_type": "text",
            }
            cell = {
                "text": "42",
                "bbox_xyxy_norm": [0.1, 0.3, 0.3, 0.4],
                "score": 0.95,
                "confidence": 0.95,
                "source": "option3",
                "model": "PP-OCRv6_medium_rec",
                "content_type": "table_cell",
                "table_id": "t1",
                "cell_id": "c1",
            }
            return OCRPageOutput(
                pipeline="option3_nemotron_language_routed_vietnamese_ocr",
                text="Body",
                ocr_text_blocks=[block, cell],
                source="option3",
                model="Nemotron OCR v2 + VietOCR",
                tables=[{"table_id": "t1", "bbox_xyxy_norm": [0.1, 0.3, 0.9, 0.8], "cells": [cell]}],
            )

    monkeypatch.setattr(runtime, "Option3Pipeline", FakeOption3)
    rows = pd.DataFrame(
        [
            {
                "page_number": 1,
                "text": "",
                "metadata": {"has_text": False, "needs_ocr_for_text": True},
            }
        ]
    )
    output = runtime.run_isolated_ocr_batch(
        rows,
        ocr_pipeline="pipeline-option3",
        ocr_invoke_url="nemotron",
        vietnamese_ocr_invoke_url="vietocr",
    )
    assert output.iloc[0]["_ocr_text_blocks"][0]["text"] == "Body"
    assert all(item["content_type"] != "table_cell" for item in output.iloc[0]["_ocr_text_blocks"])
    assert output.iloc[0]["table"][0]["cells"][0]["text"] == "42"
    assert output.iloc[0]["table"][0]["text"] == "42"


def test_isolated_native_passthrough_does_not_replace_pdfium_text(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOption4:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def process_page(self, row: Any) -> OCRPageOutput:
            return OCRPageOutput(
                pipeline="option4_parallel_nemotron_tesseract_fusion",
                text=row["text"],
                source="native_passthrough",
                status="skipped",
            )

    monkeypatch.setattr(runtime, "Option4Pipeline", FakeOption4)
    rows = pd.DataFrame(
        [
            {
                "page_number": 1,
                "text": "Native PDFium text",
                "metadata": {
                    "has_text": True,
                    "needs_ocr_for_text": False,
                    "reader_backend": "native_pdf",
                },
            }
        ]
    )
    output = runtime.run_isolated_ocr_batch(
        rows,
        ocr_pipeline="pipeline-option4",
        line_detector_invoke_url="detector",
        ocr_invoke_url="nemotron",
        tesseract_ocr_invoke_url="tesseract",
    )
    assert output.iloc[0]["text"] == "Native PDFium text"
    assert "_ocr_text_blocks" not in output.columns


def test_isolated_ocr_failure_does_not_keep_stale_upstream_text(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOption3:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def process_page(self, _row: Any) -> OCRPageOutput:
            return OCRPageOutput(
                pipeline="option3_nemotron_language_routed_vietnamese_ocr",
                source="option3",
                errors=[{"stage": "input", "message": "page image is unavailable"}],
                status="failed",
            )

    monkeypatch.setattr(runtime, "Option3Pipeline", FakeOption3)
    rows = pd.DataFrame(
        [
            {
                "page_number": 1,
                "text": "stale text from upstream",
                "metadata": {"has_text": False, "needs_ocr_for_text": True},
            }
        ]
    )
    output = runtime.run_isolated_ocr_batch(
        rows,
        ocr_pipeline="pipeline-option3",
        ocr_invoke_url="nemotron",
        vietnamese_ocr_invoke_url="vietocr",
    )
    assert output.iloc[0]["text"] == ""
    assert output.iloc[0]["metadata"]["ocr_status"] == "failed"
