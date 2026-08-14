# SPDX-License-Identifier: Apache-2.0

"""Speed/recall contracts for the Option 5 Vietnamese fast path."""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image, ImageDraw

from nemo_retriever.common.modality.ocr.isolated.contracts import OCRPage
from nemo_retriever.common.modality.ocr.isolated.geometry import PageImageCropper
from nemo_retriever.common.modality.ocr.isolated.multiline import split_multiline_units
from nemo_retriever.common.modality.ocr.isolated.option5 import Option5Pipeline
from nemo_retriever.common.modality.ocr.isolated.units import build_ocr_units


def _multiline_image() -> str:
    image = Image.new("RGB", (400, 240), "white")
    draw = ImageDraw.Draw(image)
    for y in (60, 100, 140):
        draw.rectangle((45, y, 350, y + 12), fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _multiline_page(image_b64: str) -> dict[str, Any]:
    return {
        "page_number": 1,
        "page_image": {"image_b64": image_b64},
        "page_elements_v3": {
            "detections": [
                {
                    "label_name": "text",
                    "bbox_xyxy_norm": [0.1, 0.15, 0.9, 0.7],
                    "reading_order": 0,
                }
            ]
        },
        "metadata": {"source_path": "/docs/multiline.pdf", "has_text": False},
        "text": "",
    }


class _Backend:
    def __init__(self, text: str, *, model: str, score: float = 0.96) -> None:
        self.text = text
        self.model = model
        self.score = score
        self.language = None
        self.calls: list[list[str]] = []

    def recognize(self, images: list[str]) -> list[dict[str, Any]]:
        self.calls.append(list(images))
        return [
            {"text": self.text, "score": self.score, "model": self.model}
            for _ in images
        ]


class _LineDetector:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def detect(self, images: list[str]) -> list[dict[str, Any]]:
        self.calls.append(list(images))
        return [
            {
                "boxes": [
                    {"bbox": [0.05, 0.12, 0.95, 0.28], "score": 0.95},
                    {"bbox": [0.05, 0.43, 0.95, 0.58], "score": 0.94},
                    {"bbox": [0.05, 0.73, 0.95, 0.88], "score": 0.93},
                ],
                "model": "PP-OCRv6_medium_det",
            }
            for _ in images
        ]


class _TwoColumnLineDetector:
    def detect(self, images: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "boxes": [
                    {"bbox": [0.03, 0.12, 0.42, 0.28], "score": 0.95},
                    {"bbox": [0.58, 0.12, 0.97, 0.28], "score": 0.94},
                    {"bbox": [0.03, 0.50, 0.42, 0.66], "score": 0.93},
                    {"bbox": [0.58, 0.50, 0.97, 0.66], "score": 0.92},
                ],
                "model": "PP-OCRv6_medium_det",
            }
            for _ in images
        ]


def test_multiline_projection_splits_a_semantic_box_into_line_units() -> None:
    image_b64 = _multiline_image()
    page = OCRPage.from_row(_multiline_page(image_b64))
    cropper = PageImageCropper(image_b64)
    units = build_ocr_units(page, cropper=cropper)

    split = split_multiline_units(page, units, cropper=cropper)

    assert len(split) == 3
    assert [unit.metadata["line_index"] for unit in split] == [0, 1, 2]
    assert all(unit.metadata["multiline_split"] for unit in split)
    assert all(unit.metadata["parent_unit_id"] == units[0].unit_id for unit in split)


def test_ppocr_line_detector_maps_lines_inside_parent_bbox() -> None:
    image_b64 = _multiline_image()
    page = OCRPage.from_row(_multiline_page(image_b64))
    cropper = PageImageCropper(image_b64)
    units = build_ocr_units(page, cropper=cropper)
    # A lone semantic box has a local-height estimate equal to its own height.
    # The splitter must still recognize this obviously tall compact region as
    # a detector candidate rather than silently using only the old fallback.
    detector = _LineDetector()
    stats: dict[str, Any] = {}

    split = split_multiline_units(
        page,
        units,
        cropper=cropper,
        line_detector=detector,
        stats=stats,
    )

    assert len(detector.calls) == 1
    assert len(detector.calls[0]) == 1
    assert len(split) == 3
    assert all(unit.source == "ppocrv6_line_detector" for unit in split)
    assert all(unit.metadata["line_split_method"] == "ppocrv6_line_detector" for unit in split)
    assert all(
        units[0].bbox_xyxy_norm[0] <= unit.bbox_xyxy_norm[0]
        < unit.bbox_xyxy_norm[2]
        <= units[0].bbox_xyxy_norm[2]
        for unit in split
    )
    assert stats["line_detector_input_count"] == 1
    assert stats["line_detector_line_count"] == 3


def test_ppocr_line_detector_keeps_two_columns_as_separate_crops() -> None:
    image_b64 = _multiline_image()
    page = OCRPage.from_row(_multiline_page(image_b64))
    cropper = PageImageCropper(image_b64)
    units = build_ocr_units(page, cropper=cropper)

    split = split_multiline_units(
        page,
        units,
        cropper=cropper,
        line_detector=_TwoColumnLineDetector(),
    )

    assert len(split) == 4
    assert split[0].bbox_xyxy_norm[2] < split[1].bbox_xyxy_norm[0]
    assert split[2].bbox_xyxy_norm[2] < split[3].bbox_xyxy_norm[0]


def test_direct_vietnamese_path_ocr_lines_without_full_nemotron_pass() -> None:
    image_b64 = _multiline_image()
    page = _multiline_page(image_b64)
    nemotron = _Backend(
        "Đây là văn bản tiếng Việt đủ dài để xác định ngôn ngữ.",
        model="Nemotron OCR v2",
    )
    vietnamese = _Backend(
        "Dòng tiếng Việt đã được nhận diện.",
        model="vgg_seq2seq",
    )

    output = Option5Pipeline(nemotron, vietnamese).process_document(
        [page], document_key="/docs/multiline.pdf"
    )[0]

    diagnostics = output.timing["document"]
    assert diagnostics["direct_vietnamese"] is True
    assert diagnostics["line_split_unit_count"] == 2
    assert len(nemotron.calls) == 1  # probe only
    # The language probe tests the semantic parent once; line splitting is
    # deliberately deferred until the document is known to be Vietnamese.
    assert len(nemotron.calls[0]) == 1
    assert len(vietnamese.calls) == 1
    assert len(vietnamese.calls[0]) == 3
    assert output.text.count("Dòng tiếng Việt") == 3


def test_direct_vietnamese_path_batches_ppocr_detector_across_pages() -> None:
    image_b64 = _multiline_image()
    pages = [_multiline_page(image_b64), _multiline_page(image_b64)]
    nemotron = _Backend(
        "Đây là văn bản tiếng Việt đủ dài để xác định ngôn ngữ.",
        model="Nemotron OCR v2",
    )
    vietnamese = _Backend(
        "Dòng tiếng Việt đã được nhận diện.",
        model="vgg_seq2seq",
    )
    detector = _LineDetector()

    outputs = Option5Pipeline(
        nemotron,
        vietnamese,
        line_detector=detector,
    ).process_document(pages, document_key="/docs/multiline.pdf")

    assert len(outputs) == 2
    assert len(detector.calls) == 1
    assert len(detector.calls[0]) == 2
    diagnostics = outputs[0].timing["document"]
    assert diagnostics["line_detector_input_count"] == 2
    assert diagnostics["line_detector_line_count"] == 6
    assert len(vietnamese.calls) == 1
    assert len(vietnamese.calls[0]) == 6


def test_multiline_quality_gate_keeps_readable_low_score_vietnamese_line() -> None:
    image_b64 = _multiline_image()
    page = _multiline_page(image_b64)
    nemotron = _Backend(
        "Đây là văn bản tiếng Việt đủ dài để xác định ngôn ngữ.",
        model="Nemotron OCR v2",
    )
    vietnamese = _Backend(
        "CHO XỬ LÝ DỮ LIỆU",
        model="vgg_seq2seq",
        score=0.76,
    )

    output = Option5Pipeline(nemotron, vietnamese).process_document(
        [page], document_key="/docs/multiline.pdf"
    )[0]

    assert output.ocr_text_blocks
    assert all(
        block["source"] == "option5_vietnamese_recognizer"
        for block in output.ocr_text_blocks
    )
