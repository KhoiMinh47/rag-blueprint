# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import pandas as pd
from PIL import Image

from nemo_retriever.common.modality.ocr.isolated.official_ppocr import (
    run_official_ppocr_batch,
)


def _page_image() -> str:
    image = Image.new("RGB", (100, 100), "white")
    payload = BytesIO()
    image.save(payload, format="PNG")
    return base64.b64encode(payload.getvalue()).decode("ascii")


def _row() -> dict[str, Any]:
    return {
        "path": "scan.pdf",
        "page_number": 1,
        "text": "native text must not be copied",
        "_native_text_spans": [{"char": "x", "bbox_xyxy_norm": [0, 0, 1, 1]}],
        "page_image": {"image_b64": _page_image()},
        "images": [],
        "metadata": {"needs_ocr": True},
    }


def _response() -> list[dict[str, Any]]:
    return [
        {
            "pipeline": "PP-OCRv6 official general OCR",
            "models": {
                "doc_orientation": "PP-LCNet_x1_0_doc_ori",
                "doc_unwarping": "UVDoc",
                "textline_orientation": "PP-LCNet_x1_0_textline_ori",
                "text_detection": "PP-OCRv6_medium_det",
                "text_recognition": "PP-OCRv6_medium_rec",
            },
            "preprocess": {"angle": 0},
            "lines": [
                {
                    "text": "NỘI QUY",
                    "bbox": [0.1, 0.1, 0.9, 0.2],
                    "score": 0.98,
                    "detector_score": 0.99,
                    "line_angle": 0,
                },
                {
                    "text": "Đơn giá",
                    "bbox": [0.1, 0.3, 0.5, 0.4],
                    "score": 0.96,
                    "detector_score": 0.97,
                    "line_angle": 0,
                },
            ],
            "raw_counts": {"detected_boxes": 2, "recognized_lines": 2},
        }
    ]


def test_option2_uses_one_official_whole_page_request() -> None:
    calls: list[tuple[str, dict[str, Any], float]] = []

    def transport(url: str, payload: dict[str, Any], timeout: float) -> list[dict[str, Any]]:
        calls.append((url, payload, timeout))
        return _response()

    result = run_official_ppocr_batch(
        pd.DataFrame([_row()]),
        invoke_url="http://ppocrv6-official:8000/v1/ocr",
        transport=transport,
    )
    row = result.iloc[0].to_dict()

    assert len(calls) == 1
    assert calls[0][0].endswith("/v1/ocr")
    assert calls[0][1]["images"]
    assert row["text"] == "NỘI QUY\nĐơn giá"
    assert len(row["_ocr_text_blocks"]) == 2
    assert len(row["page_elements_v3"]["detections"]) == 2
    assert row["page_elements_v3"]["detections"][0]["model"] == "PP-OCRv6_medium_det"
    assert row["ocr"]["models"]["text_recognition"] == "PP-OCRv6_medium_rec"
    assert row["metadata"]["ocr_source"] == "ppocrv6_official_general_ocr"
    assert row["table"] == []
    assert "_native_text_spans" not in row


def test_option2_does_not_fallback_when_official_service_fails() -> None:
    def failing_transport(_url: str, _payload: dict[str, Any], _timeout: float) -> Any:
        raise RuntimeError("PP-OCRv6 service unavailable")

    result = run_official_ppocr_batch(
        pd.DataFrame([_row()]),
        invoke_url="http://ppocrv6-official:8000/v1/ocr",
        transport=failing_transport,
    )
    row = result.iloc[0].to_dict()

    assert row["ocr"]["status"] == "failed"
    assert row["metadata"]["ocr_status"] == "failed"
    assert row["text"] == ""
    assert row["ocr"]["source"] == "ppocrv6_official_general_ocr"
    assert "nemotron" not in str(row["ocr"]).lower()
    assert "tesseract" not in str(row["ocr"]["errors"]).lower()
