# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the detector-free Option 3 OCR pipeline."""

from __future__ import annotations

import base64
import io
from typing import Any

import pytest
from PIL import Image

from nemo_retriever.common.modality.ocr.isolated import language_router
from nemo_retriever.common.modality.ocr.isolated.contracts import OCRPage
from nemo_retriever.common.modality.ocr.isolated.geometry import map_local_bbox
from nemo_retriever.common.modality.ocr.isolated.option3 import (
    OPTION3_PIPELINE_NAME,
    Option3Config,
    Option3Pipeline,
)


def _image_b64(width: int = 480, height: int = 320) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _page(
    detections: list[dict[str, Any]] | None = None,
    *,
    table: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    native_text: str = "",
) -> dict[str, Any]:
    return {
        "page_number": 1,
        "page_image": {"image_b64": _image_b64()},
        "page_elements_v3": {"detections": detections or []},
        "table_structure_v1": table,
        "metadata": metadata or {"has_text": False, "needs_ocr_for_text": True},
        "text": native_text,
    }


class FakeBackend:
    def __init__(self, responses: list[Any], *, model: str) -> None:
        self.responses = list(responses)
        self.model = model
        self.language = None
        self.calls: list[list[str]] = []
        self._offset = 0

    def recognize(self, images: list[str]) -> list[Any]:
        self.calls.append(list(images))
        result: list[Any] = []
        for _ in images:
            response = self.responses[min(self._offset, len(self.responses) - 1)]
            self._offset += 1
            if isinstance(response, BaseException):
                raise response
            result.append(response)
        return result


def _recognition(
    text: str,
    *,
    score: float | None = 0.95,
    model: str = "Nemotron OCR v2",
    bbox: list[float] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {"text": text, "score": score, "model": model}
    if bbox is not None:
        item["bbox"] = bbox
    return item


def _english_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        language_router,
        "_langdetect_probabilities",
        lambda _text: ({"en": 0.96, "vi": 0.01}, None),
    )


def test_native_page_passthrough_does_not_call_either_backend() -> None:
    nemotron = FakeBackend([], model="Nemotron OCR v2")
    vietnamese = FakeBackend([], model="vgg_seq2seq")
    page = _page(
        [{"label_name": "text", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.2]}],
        metadata={
            "has_text": True,
            "needs_ocr_for_text": False,
            "reader_backend": "native_pdf",
        },
        native_text="Native PDFium text",
    )

    output = Option3Pipeline(nemotron, vietnamese).process_page(page)

    assert output.status == "skipped"
    assert output.text == "Native PDFium text"
    assert output.pipeline == OPTION3_PIPELINE_NAME
    assert nemotron.calls == []
    assert vietnamese.calls == []


def test_english_block_keeps_nemotron_and_skips_vietnamese(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _english_detector(monkeypatch)
    nemotron = FakeBackend(
        [_recognition("This is a sufficiently long English block for routing.")],
        model="Nemotron OCR v2",
    )
    vietnamese = FakeBackend(
        [_recognition("không được gọi", model="vgg_seq2seq")],
        model="vgg_seq2seq",
    )
    output = Option3Pipeline(nemotron, vietnamese).process_page(
        _page([{"label_name": "text", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.2]}])
    )

    block = output.ocr_text_blocks[0]
    assert output.text == "This is a sufficiently long English block for routing."
    assert block["model"] == "Nemotron OCR v2"
    assert block["provenance"]["route"] == "english"
    assert block["provenance"]["selected_backend"] == "nemotron"
    assert len(nemotron.calls) == 1
    assert vietnamese.calls == []


def test_vietnamese_block_runs_nemotron_first_then_accepts_vietocr() -> None:
    nemotron = FakeBackend(
        [_recognition("Đây là văn bản tiếng Việt.")],
        model="Nemotron OCR v2",
    )
    vietnamese = FakeBackend(
        [_recognition("Đây là văn bản tiếng Việt chuẩn.", model="vgg_seq2seq")],
        model="vgg_seq2seq",
    )

    output = Option3Pipeline(nemotron, vietnamese).process_page(
        _page([{"label_name": "text", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.2]}])
    )

    block = output.ocr_text_blocks[0]
    assert len(nemotron.calls) == 1
    assert len(vietnamese.calls) == 1
    assert output.text == "Đây là văn bản tiếng Việt chuẩn."
    assert block["source"] == "option3_vietnamese_recognizer"
    assert block["model"] == "vgg_seq2seq"
    assert block["provenance"]["route"] == "vietnamese"
    assert block["provenance"]["selected_backend"] == "vietnamese_recognizer"
    assert block["provenance"]["nemotron_original_text"] == "Đây là văn bản tiếng Việt."


@pytest.mark.parametrize(
    "response",
    [
        _recognition("candidate yếu", score=0.2, model="vgg_seq2seq"),
        _recognition("", score=0.99, model="vgg_seq2seq"),
        RuntimeError("VietOCR unavailable"),
    ],
)
def test_vietnamese_quality_gate_falls_back_without_failing_page(response: Any) -> None:
    nemotron = FakeBackend(
        [_recognition("Nội dung Nemotron gốc có dấu Việt.")],
        model="Nemotron OCR v2",
    )
    vietnamese = FakeBackend([response], model="vgg_seq2seq")

    output = Option3Pipeline(nemotron, vietnamese).process_page(
        _page([{"label_name": "text", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.2]}])
    )

    block = output.ocr_text_blocks[0]
    assert output.status in {"completed", "partial"}
    assert output.text == "Nội dung Nemotron gốc có dấu Việt."
    assert block["source"] == "option3_nemotron"
    assert block["provenance"]["selected_backend"] == "nemotron"
    assert block["provenance"]["fallback_reason"]


def test_short_numeric_and_uncertain_text_stays_nemotron(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _english_detector(monkeypatch)
    nemotron = FakeBackend(
        [_recognition("A-01"), _recognition("hello")],
        model="Nemotron OCR v2",
    )
    vietnamese = FakeBackend([], model="vgg_seq2seq")
    output = Option3Pipeline(nemotron, vietnamese).process_page(
        _page(
            [
                {"label_name": "text", "bbox_xyxy_norm": [0.1, 0.1, 0.3, 0.2]},
                {"label_name": "text", "bbox_xyxy_norm": [0.1, 0.25, 0.3, 0.35]},
            ]
        )
    )

    assert [block["text"] for block in output.ocr_text_blocks] == ["A-01", "hello"]
    assert all(block["source"] == "option3_nemotron" for block in output.ocr_text_blocks)
    assert output.timing["route_counts"]["uncertain"] == 2
    assert vietnamese.calls == []


def test_mixed_page_sends_only_vietnamese_candidates_to_one_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _english_detector(monkeypatch)
    nemotron = FakeBackend(
        [
            _recognition("This is a long English paragraph for the page."),
            _recognition("Đây là một đoạn tiếng Việt có dấu."),
            _recognition("99"),
        ],
        model="Nemotron OCR v2",
    )
    vietnamese = FakeBackend(
        [_recognition("Đây là một đoạn tiếng Việt đã nhận diện.", model="vgg_seq2seq")],
        model="vgg_seq2seq",
    )
    output = Option3Pipeline(nemotron, vietnamese).process_page(
        _page(
            [
                {"label_name": "text", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.2], "reading_order": 0},
                {"label_name": "text", "bbox_xyxy_norm": [0.1, 0.25, 0.9, 0.35], "reading_order": 1},
                {"label_name": "text", "bbox_xyxy_norm": [0.1, 0.4, 0.3, 0.5], "reading_order": 2},
            ]
        )
    )

    assert len(nemotron.calls) == 1
    assert len(nemotron.calls[0]) == 3
    assert len(vietnamese.calls) == 1
    assert len(vietnamese.calls[0]) == 1
    assert output.timing["route_counts"] == {
        "vietnamese": 1,
        "english": 1,
        "uncertain": 1,
    }


def test_local_bbox_is_mapped_from_nemotron_crop_pixels() -> None:
    local_bbox = [10, 20, 80, 60]
    nemotron = FakeBackend(
        [
            {
                "text_detections": [
                    {"text": "mapped", "score": 0.95, "bbox": local_bbox}
                ],
                "model": "Nemotron OCR v2",
            }
        ],
        model="Nemotron OCR v2",
    )
    vietnamese = FakeBackend([], model="vgg_seq2seq")
    page = OCRPage.from_row(
        _page([{"label_name": "text", "bbox_xyxy_norm": [0.2, 0.3, 0.8, 0.7]}])
    )
    from nemo_retriever.common.modality.ocr.isolated.units import build_ocr_units

    unit = build_ocr_units(page)[0]
    expected = map_local_bbox(local_bbox, unit.crop_bbox_xyxy_norm, unit.crop_shape_hw)
    output = Option3Pipeline(nemotron, vietnamese).process_page(page)

    assert output.ocr_text_blocks[0]["bbox_xyxy_norm"] == list(expected)
    assert output.ocr_text_blocks[0]["provenance"]["bbox_source"] == "nemotron_local"
    assert output.ocr_text_blocks[0]["provenance"]["bbox_fallback"] is False


def test_missing_local_bbox_uses_parent_and_records_fallback() -> None:
    nemotron = FakeBackend([_recognition("parent bbox")], model="Nemotron OCR v2")
    vietnamese = FakeBackend([], model="vgg_seq2seq")
    page = OCRPage.from_row(
        _page([{"label_name": "text", "bbox_xyxy_norm": [0.2, 0.3, 0.8, 0.7]}])
    )
    from nemo_retriever.common.modality.ocr.isolated.units import build_ocr_units

    parent = build_ocr_units(page)[0].bbox_xyxy_norm
    output = Option3Pipeline(nemotron, vietnamese).process_page(page)
    block = output.ocr_text_blocks[0]

    assert block["bbox_xyxy_norm"] == list(parent)
    assert block["provenance"]["bbox_source"] == "parent_semantic_unit"
    assert block["provenance"]["bbox_fallback"] is True


def test_multiple_items_without_local_boxes_are_not_collapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _english_detector(monkeypatch)
    nemotron = FakeBackend(
        [
            {
                "text_detections": [
                    {"text": "First distinct line", "score": 0.95},
                    {"text": "Second distinct line", "score": 0.94},
                ]
            }
        ],
        model="Nemotron OCR v2",
    )
    output = Option3Pipeline(nemotron, FakeBackend([], model="vgg_seq2seq")).process_page(
        _page([{"label_name": "text", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.3]}])
    )

    assert [block["text"] for block in output.ocr_text_blocks] == [
        "First distinct line",
        "Second distinct line",
    ]
    assert all(
        block["provenance"]["bbox_fallback"] is True
        for block in output.ocr_text_blocks
    )


def test_table_cells_keep_ids_geometry_and_are_not_generic_text() -> None:
    table = {
        "regions": [
            {
                "table_id": "t1",
                "bbox_xyxy_norm": [0.1, 0.4, 0.9, 0.8],
                "orig_shape_hw": [100, 200],
                "detections": [
                    {"label_name": "cell", "cell_id": "a", "bbox_xyxy_norm": [0, 0, 0.5, 1]},
                    {"label_name": "cell", "cell_id": "b", "bbox_xyxy_norm": [0.5, 0, 1, 1]},
                ],
            }
        ]
    }
    nemotron = FakeBackend(
        [_recognition("A"), _recognition("B")], model="Nemotron OCR v2"
    )
    vietnamese = FakeBackend([], model="vgg_seq2seq")
    output = Option3Pipeline(nemotron, vietnamese).process_page(
        _page([], table=table)
    )

    assert len(output.ocr_text_blocks) == 2
    assert all(block["content_type"] == "table_cell" for block in output.ocr_text_blocks)
    assert {block["cell_id"] for block in output.tables[0]["cells"]} == {"a", "b"}
    assert all(block["table_id"] == "t1" for block in output.tables[0]["cells"])
    assert output.text == ""


def test_reading_order_is_stable_without_detector() -> None:
    nemotron = FakeBackend(
        [_recognition("left top"), _recognition("left bottom"), _recognition("right top")],
        model="Nemotron OCR v2",
    )
    vietnamese = FakeBackend([], model="vgg_seq2seq")
    output = Option3Pipeline(nemotron, vietnamese).process_page(
        _page(
            [
                {"label_name": "text", "bbox_xyxy_norm": [0.6, 0.1, 0.9, 0.14]},
                {"label_name": "text", "bbox_xyxy_norm": [0.1, 0.1, 0.4, 0.14]},
                {"label_name": "text", "bbox_xyxy_norm": [0.1, 0.2, 0.4, 0.24]},
            ]
        )
    )

    assert [block["text"] for block in output.ocr_text_blocks] == [
        "left top",
        "left bottom",
        "right top",
    ]


def test_scoreless_policy_defaults_to_nemotron_and_can_be_server_enabled() -> None:
    page = _page([{"label_name": "text", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.2]}])
    nemotron = FakeBackend([_recognition("Đầu vào Việt Nam")], model="Nemotron OCR v2")
    vietnamese = FakeBackend([_recognition("Kết quả không score", score=None)], model="vgg_seq2seq")
    rejected = Option3Pipeline(nemotron, vietnamese).process_page(page)
    assert rejected.ocr_text_blocks[0]["source"] == "option3_nemotron"
    assert rejected.ocr_text_blocks[0]["provenance"]["fallback_reason"] == "vietnamese_score_missing"

    nemotron = FakeBackend([_recognition("Đầu vào Việt Nam")], model="Nemotron OCR v2")
    vietnamese = FakeBackend([_recognition("Kết quả không score", score=None)], model="vgg_seq2seq")
    accepted = Option3Pipeline(
        nemotron,
        vietnamese,
        config=Option3Config(allow_scoreless_vietnamese=True),
    ).process_page(page)
    assert accepted.ocr_text_blocks[0]["source"] == "option3_vietnamese_recognizer"


def test_option3_metrics_include_timings_routes_and_batch_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _english_detector(monkeypatch)
    nemotron = FakeBackend(
        [_recognition("A long English block for metrics."), _recognition("Đoạn tiếng Việt có dấu")],
        model="Nemotron OCR v2",
    )
    vietnamese = FakeBackend([_recognition("Đoạn tiếng Việt có dấu")], model="vgg_seq2seq")
    output = Option3Pipeline(nemotron, vietnamese).process_page(
        _page(
            [
                {"label_name": "text", "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.2]},
                {"label_name": "text", "bbox_xyxy_norm": [0.1, 0.3, 0.9, 0.4]},
            ]
        )
    )
    timing = output.timing
    assert timing["total_seconds"] >= 0
    assert timing["nemotron_seconds"] >= 0
    assert timing["language_router_seconds"] >= 0
    assert timing["vietnamese_recognizer_seconds"] >= 0
    assert timing["nemotron_input_count"] == 2
    assert timing["vietnamese_input_count"] == 1
    assert timing["vietnamese_batch_count"] == 1
    assert timing["route_counts"]["vietnamese"] == 1
    assert (
        timing["selected_backend_counts"].get("vietocr", 0) == 1
        or timing["selected_backend_counts"].get("vietnamese_recognizer", 0) == 1
    )
