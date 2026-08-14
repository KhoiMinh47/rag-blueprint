# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import pandas as pd
from PIL import Image

from nemo_retriever.common.modality.ocr.isolated.paddleocr_vl import run_paddleocr_vl_batch


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


def _response() -> dict[str, Any]:
    return {
        "errorCode": 0,
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {
                        "parsing_res_list": [
                            {
                                "block_id": "text-1",
                                "block_order": 1,
                                "block_label": "text",
                                "block_content": "Đơn giá",
                                "block_bbox": [10, 10, 90, 30],
                            },
                            {
                                "block_id": "table-1",
                                "block_order": 2,
                                "block_label": "table",
                                "block_content": "| A | B |",
                                "block_bbox": [10, 40, 90, 70],
                            },
                            {
                                "block_id": "image-1",
                                "block_order": 3,
                                "block_label": "image",
                                "block_content": "",
                                "block_bbox": [10, 75, 40, 95],
                            },
                            {
                                "block_id": "chart-1",
                                "block_order": 4,
                                "block_label": "chart",
                                "block_content": "",
                                "block_bbox": [45, 75, 90, 95],
                            },
                        ]
                    }
                }
            ]
        },
    }


def test_option2_calls_full_paddleocr_vl_and_keeps_visual_trace() -> None:
    calls: list[tuple[str, dict[str, Any], float]] = []

    def transport(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        calls.append((url, payload, timeout))
        return _response()

    result = run_paddleocr_vl_batch(
        pd.DataFrame([_row()]),
        invoke_url="http://paddleocr-vl/layout-parsing",
        transport=transport,
    )
    row = result.iloc[0].to_dict()

    assert len(calls) == 1
    assert calls[0][0] == "http://paddleocr-vl/layout-parsing"
    assert calls[0][1]["fileType"] == 1
    assert calls[0][1]["visualize"] is False
    assert calls[0][1]["file"]
    assert row["text"] == "Đơn giá"
    assert row["table"][0]["text"] == "| A | B |"
    assert row["table"][0]["bbox_xyxy_norm"] == [0.1, 0.4, 0.9, 0.7]
    assert len(row["images"]) == 1
    assert row["images"][0]["label_name"] == "image"
    assert row["chart"][0]["text"] == "[chart]"
    assert len(row["_ocr_text_blocks"]) == 1
    assert len(row["page_elements_v3"]["detections"]) == 4
    assert row["page_elements_v3"]["detections"][0]["model"] == "PP-DocLayoutV3"
    assert row["metadata"]["ocr_model"] == "PaddleOCR-VL-1.6-0.9B"
    assert row["ocr"]["backend"] == "vllm"
    assert "_native_text_spans" not in row


def test_option2_keeps_page_failure_without_nvidia_or_tesseract_fallback() -> None:
    def failing_transport(_url: str, _payload: dict[str, Any], _timeout: float) -> dict[str, Any]:
        raise RuntimeError("Paddle service unavailable")

    result = run_paddleocr_vl_batch(
        pd.DataFrame([_row()]),
        invoke_url="http://paddleocr-vl/layout-parsing",
        transport=failing_transport,
    )
    row = result.iloc[0].to_dict()

    assert row["ocr"]["status"] == "failed"
    assert row["metadata"]["ocr_status"] == "failed"
    assert row["text"] == ""
    assert row["table"] == []
    assert row["ocr"]["source"] == "paddleocr_vl_full_pipeline"
    failure_text = str(row["ocr"]) + str(row["metadata"])
    assert "nemotron" not in failure_text.lower()
    assert row["ocr"]["model"] == "PaddleOCR-VL-1.6-0.9B"
