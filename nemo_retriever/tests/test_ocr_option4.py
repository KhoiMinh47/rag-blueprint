# SPDX-License-Identifier: Apache-2.0

"""Unit tests for isolated parallel Nemotron/Tesseract fusion."""

from __future__ import annotations

import base64
import io
from typing import Any

from nemo_retriever.common.modality.ocr.isolated.option4 import Option4Pipeline
from PIL import Image, ImageDraw


def _image_b64(width: int = 420, height: int = 280) -> str:
    image = Image.new("RGB", (width, height), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_b64_with_two_lines(width: int = 420, height: int = 280) -> str:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 42, 340, 52), fill="black")
    draw.rectangle((80, 76, 340, 86), fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _page(
    *,
    detections: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    table: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "page_number": 1,
        "page_image": {"image_b64": _image_b64()},
        "page_elements_v3": {"detections": detections},
        "table_structure_v1": table,
        "metadata": metadata or {"has_text": False, "needs_ocr_for_text": True},
    }


class FakeBackend:
    def __init__(
        self,
        model: str,
        outputs: list[dict[str, Any]],
        *,
        language: str | None = "eng+vie",
        fail: bool = False,
    ) -> None:
        self.model = model
        self.language = language
        self.outputs = outputs
        self.fail = fail
        self.received: list[str] = []
        self._index = 0

    def recognize(self, images: list[str]) -> list[dict[str, Any]]:
        if self.fail:
            raise RuntimeError(f"{self.model} unavailable")
        self.received.extend(images)
        result = []
        for _ in images:
            result.append(self.outputs[min(self._index, len(self.outputs) - 1)])
            self._index += 1
        return result


class FakeLineDetector:
    model = "PP-OCRv6_medium_det"

    def __init__(self, boxes: list[dict[str, Any]]) -> None:
        self.boxes = boxes
        self.received: list[str] = []

    def detect(self, images: list[str]) -> list[dict[str, Any]]:
        self.received.extend(images)
        return [{"boxes": self.boxes} for _ in images]


def test_option4_accepts_confident_tesseract_first() -> None:
    nemotron = FakeBackend(
        "Nemotron OCR v2",
        [{"text": "Hợp đồng 2026", "score": 0.86}],
        language="vie+eng",
    )
    tesseract = FakeBackend(
        "tesseract-5", [{"text": "Hop dong 2026", "score": 0.84}], language="eng+vie"
    )
    page = _page(
        detections=[{"label_name": "text", "bbox_xyxy_norm": [0.15, 0.15, 0.85, 0.25]}]
    )
    result = Option4Pipeline(nemotron, tesseract).process_page(page).to_dict()
    assert nemotron.received == []
    assert tesseract.received
    assert len(result["ocr_text_blocks"]) == 1
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["selected_backend"] == "tesseract"
    assert result["candidates"][0]["decision"] == "tesseract_first_accepted"
    assert len(result["candidates"][0]["candidates"]) == 1
    assert result["ocr_text_blocks"][0]["source"] == "option4_fusion"


def test_option4_calls_nemotron_for_low_confidence_tesseract_lines() -> None:
    nemotron = FakeBackend(
        "Nemotron OCR v2",
        [{"text": "Dòng một", "score": 0.86}, {"text": "Dòng hai", "score": 0.86}],
        language="vie+eng",
    )
    tesseract = FakeBackend(
        "tesseract-5",
        [{"text": "Dong mot", "score": 0.60}, {"text": "Dong hai", "score": 0.60}],
        language="eng+vie",
    )
    detector = FakeLineDetector(
        [
            {"bbox": [0.0, 0.0, 1.0, 0.45], "score": 0.92},
            {"bbox": [0.0, 0.55, 1.0, 1.0], "score": 0.91},
        ]
    )
    page = _page(
        detections=[{"label_name": "text", "bbox_xyxy_norm": [0.15, 0.15, 0.85, 0.45]}]
    )
    result = Option4Pipeline(
        nemotron, tesseract, line_detector=detector
    ).process_page(page).to_dict()

    assert len(detector.received) == 1
    assert nemotron.received == tesseract.received
    assert len(result["ocr_text_blocks"]) == 2
    assert all(":line-" in block["unit_id"] for block in result["ocr_text_blocks"])
    assert result["timing"]["line_detector"] is True
    assert result["timing"]["recognition_strategy"] == "tesseract_first_nemotron_fallback"


def test_option4_preserves_parent_horizontal_bounds_when_detector_trims_line() -> None:
    nemotron = FakeBackend("Nemotron OCR v2", [])
    tesseract = FakeBackend(
        "tesseract-5", [{"text": "Pursuant to Civil Code", "score": 0.91}]
    )
    detector = FakeLineDetector(
        [{"bbox": [0.25, 0.0, 0.75, 1.0], "score": 0.94}]
    )
    parent_bbox = [0.15, 0.15, 0.85, 0.25]
    page = _page(
        detections=[{"label_name": "text", "bbox_xyxy_norm": parent_bbox}]
    )

    result = Option4Pipeline(
        nemotron, tesseract, line_detector=detector
    ).process_page(page).to_dict()

    bbox = result["ocr_text_blocks"][0]["bbox_xyxy_norm"]
    assert bbox[0] <= parent_bbox[0]
    assert bbox[2] >= parent_bbox[2]
    assert result["ocr_text_blocks"][0]["provenance"]["ocr_unit"][
        "bbox_xyxy_norm"
    ][0] == parent_bbox[0]


def test_option4_projection_fallback_splits_detector_empty_multiline_region() -> None:
    nemotron = FakeBackend(
        "Nemotron OCR v2",
        [{"text": "first line", "score": 0.90}, {"text": "second line", "score": 0.90}],
    )
    tesseract = FakeBackend(
        "tesseract-5",
        [{"text": "first line", "score": 0.50}, {"text": "second line", "score": 0.50}],
    )
    detector = FakeLineDetector([])
    page = _page(
        detections=[{"label_name": "text", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.45]}]
    )
    page["page_image"]["image_b64"] = _image_b64_with_two_lines()

    result = Option4Pipeline(
        nemotron, tesseract, line_detector=detector
    ).process_page(page).to_dict()

    assert len(result["ocr_text_blocks"]) == 2
    assert all(
        block["provenance"]["ocr_unit"]["source"]
        == "horizontal_projection_fallback"
        for block in result["ocr_text_blocks"]
    )
    assert all(
        block["provenance"]["line_detector_fallback"]
        for block in result["ocr_text_blocks"]
    )


def test_option4_routes_english_probe_to_nemotron_without_vietnamese_tesseract() -> None:
    nemotron = FakeBackend(
        "Nemotron OCR v2", [{"text": "OFFICE LEASE AGREEMENT", "score": 0.91}]
    )
    tesseract = FakeBackend(
        "tesseract-5", [{"text": "VĂN BẢN SAI", "score": 0.96}], language="vie"
    )
    language_probe = FakeBackend(
        "tesseract-5",
        [{"text": "OFFICE LEASE AGREEMENT", "score": 0.93}],
        language="vie+eng",
    )
    page = _page(
        detections=[{"label_name": "text", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.2]}]
    )
    result = Option4Pipeline(
        nemotron,
        tesseract,
        language_probe=language_probe,
    ).process_page(page).to_dict()

    assert tesseract.received == []
    assert nemotron.received
    assert result["ocr_text_blocks"][0]["text"] == "OFFICE LEASE AGREEMENT"
    assert result["candidates"][0]["selected_backend"] == "nemotron"
    assert result["candidates"][0]["language_router"]["route"] == "non_vietnamese"


def test_option4_routes_vietnamese_probe_to_vietnamese_tesseract() -> None:
    nemotron = FakeBackend(
        "Nemotron OCR v2", [{"text": "Nemotron English", "score": 0.95}]
    )
    tesseract = FakeBackend(
        "tesseract-5", [{"text": "Hợp đồng thuê nhà", "score": 0.91}], language="vie"
    )
    language_probe = FakeBackend(
        "tesseract-5",
        [{"text": "Hợp đồng thuê nhà", "score": 0.93}],
        language="vie+eng",
    )
    page = _page(
        detections=[{"label_name": "text", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.2]}]
    )
    result = Option4Pipeline(
        nemotron,
        tesseract,
        language_probe=language_probe,
    ).process_page(page).to_dict()

    assert language_probe.received
    assert tesseract.received
    assert nemotron.received == []
    assert result["ocr_text_blocks"][0]["text"] == "Hợp đồng thuê nhà"
    assert result["candidates"][0]["selected_backend"] == "tesseract"
    assert result["candidates"][0]["language_router"]["route"] == "vietnamese"


def test_option4_backend_failure_uses_the_other_backend_without_duplicate() -> None:
    nemotron = FakeBackend("Nemotron OCR v2", [], fail=True)
    tesseract = FakeBackend(
        "tesseract-5", [{"text": "Only Tesseract", "score": 0.91}], language="eng+vie"
    )
    page = _page(
        detections=[{"label_name": "text", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.2]}]
    )
    result = Option4Pipeline(nemotron, tesseract).process_page(page).to_dict()
    assert result["text"] == "Only Tesseract"
    assert len(result["ocr_text_blocks"]) == 1
    assert result["candidates"][0]["selected_backend"] == "tesseract"
    assert result["errors"] == []
    assert nemotron.received == []

    nemotron_ok = FakeBackend(
        "Nemotron OCR v2", [{"text": "Only Nemotron", "score": 0.91}]
    )
    tesseract_fail = FakeBackend("tesseract-5", [], fail=True)
    result2 = Option4Pipeline(nemotron_ok, tesseract_fail).process_page(page).to_dict()
    assert result2["text"] == "Only Nemotron"
    assert len(result2["ocr_text_blocks"]) == 1
    assert result2["candidates"][0]["selected_backend"] == "nemotron"
    # Tesseract is an optional first-pass backend. Its outage must not turn a
    # successful Nemotron fallback into a fatal graph row error.
    assert result2["errors"] == []


def test_option4_near_equal_outputs_keep_one_and_preserve_both_provenances() -> None:
    nemotron = FakeBackend(
        "Nemotron OCR v2",
        [{"text": "Ngày ký: 07/08/2026", "score": 0.88}],
        language="vie+eng",
    )
    tesseract = FakeBackend(
        "tesseract-5",
        [{"text": "Ngày ký: 07/08/2026", "score": 0.70}],
        language="eng+vie",
    )
    nemotron.outputs[0]["score"] = 0.70
    page = _page(
        detections=[{"label_name": "title", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.2]}]
    )
    result = Option4Pipeline(nemotron, tesseract).process_page(page).to_dict()
    assert len(result["ocr_text_blocks"]) == 1
    assert {item["backend"] for item in result["candidates"][0]["candidates"]} == {
        "nemotron",
        "tesseract",
    }
    assert result["candidates"][0]["selected_backend"] == "tesseract"
    assert result["ocr_text_blocks"][0]["provenance"]["selected_backend"] in {
        "nemotron",
        "tesseract",
    }


def test_option4_different_candidates_use_normalized_quality_for_numeric_code() -> None:
    nemotron = FakeBackend(
        "Nemotron OCR v2",
        [{"text": "Mã hợp đồng ABC-2026-0042", "score": 0.90}],
        language="vie+eng",
    )
    tesseract = FakeBackend(
        "tesseract-5",
        [{"text": "Ma hop dong A8C-2026-O042", "score": 0.62}],
        language="eng+vie",
    )
    page = _page(
        detections=[{"label_name": "text", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.2]}]
    )
    result = Option4Pipeline(nemotron, tesseract).process_page(page).to_dict()
    block = result["ocr_text_blocks"][0]
    assert "ABC-2026-0042" in block["text"]
    assert result["candidates"][0]["selected_backend"] == "nemotron"
    assert {item["backend"] for item in result["candidates"][0]["candidates"]} == {
        "nemotron",
        "tesseract",
    }


def test_option4_table_fuses_per_cell_and_preserves_geometry() -> None:
    nemotron = FakeBackend(
        "Nemotron OCR v2", [{"text": "A", "score": 0.88}, {"text": "B", "score": 0.88}]
    )
    tesseract = FakeBackend(
        "tesseract-5", [{"text": "A", "score": 0.87}, {"text": "B", "score": 0.87}]
    )
    table = {
        "regions": [
            {
                "table_id": "table-1",
                "bbox_xyxy_norm": [0.1, 0.3, 0.9, 0.8],
                "orig_shape_hw": [100, 200],
                "detections": [
                    {
                        "label_name": "cell",
                        "cell_id": "c1",
                        "bbox_xyxy_norm": [0.0, 0.0, 0.5, 0.5],
                    },
                    {
                        "label_name": "cell",
                        "cell_id": "c2",
                        "bbox_xyxy_norm": [0.5, 0.0, 1.0, 0.5],
                    },
                ],
            }
        ]
    }
    page = _page(
        detections=[{"label_name": "table", "bbox_xyxy_norm": [0.1, 0.3, 0.9, 0.8]}],
        table=table,
    )
    result = Option4Pipeline(nemotron, tesseract).process_page(page).to_dict()
    assert len(result["tables"][0]["cells"]) == 2
    assert {cell["cell_id"] for cell in result["tables"][0]["cells"]} == {"c1", "c2"}
    assert len(result["ocr_text_blocks"]) == 2
    assert len({block["unit_id"] for block in result["ocr_text_blocks"]}) == 2
    assert result["tables"][0]["text"] == "| A | B |"


def test_option4_discards_border_only_table_ocr() -> None:
    nemotron = FakeBackend("Nemotron OCR v2", [{"text": "||||", "score": 0.99}])
    tesseract = FakeBackend("tesseract-5", [{"text": "| |", "score": 0.99}])
    table = {
        "regions": [
            {
                "table_id": "table-1",
                "bbox_xyxy_norm": [0.1, 0.3, 0.9, 0.8],
                "orig_shape_hw": [100, 200],
                "detections": [
                    {"label_name": "cell", "cell_id": "c1", "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0]}
                ],
            }
        ]
    }
    result = Option4Pipeline(nemotron, tesseract).process_page(
        _page(detections=[{"label_name": "table", "bbox_xyxy_norm": [0.1, 0.3, 0.9, 0.8]}], table=table)
    ).to_dict()

    assert result["ocr_text_blocks"] == []
    assert result["tables"][0]["cells"] == []
    assert result["tables"][0]["text"] == ""


def test_option4_native_page_is_not_touched() -> None:
    nemotron = FakeBackend(
        "Nemotron OCR v2", [{"text": "should not run", "score": 1.0}]
    )
    tesseract = FakeBackend("tesseract-5", [{"text": "should not run", "score": 1.0}])
    page = _page(
        detections=[{"label_name": "text", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.2]}],
        metadata={
            "has_text": True,
            "needs_ocr_for_text": False,
            "reader_backend": "native_pdf",
        },
    )
    page["text"] = "Native path remains canonical"
    result = Option4Pipeline(nemotron, tesseract).process_page(page).to_dict()
    assert result["status"] == "skipped"
    assert result["text"] == "Native path remains canonical"
    assert nemotron.received == []
    assert tesseract.received == []


def test_option4_does_not_use_scan_fallback_for_visual_only_page() -> None:
    nemotron = FakeBackend("Nemotron OCR v2", [{"text": "not visual", "score": 1.0}])
    tesseract = FakeBackend("tesseract-5", [{"text": "not visual", "score": 1.0}])
    page = _page(
        detections=[{"label_name": "chart", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.8]}]
    )
    result = Option4Pipeline(nemotron, tesseract).process_page(page).to_dict()
    assert result["ocr_text_blocks"] == []
    assert nemotron.received == []
    assert tesseract.received == []
