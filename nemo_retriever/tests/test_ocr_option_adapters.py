# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the isolated OCR service adapters."""

from __future__ import annotations

from typing import Any

from nemo_retriever.common.modality.ocr.isolated.adapters import (
    PPOCRv6Adapter,
    VLLMImageBackend,
    detector_boxes,
    make_tesseract_backend,
    recognition_items,
)


def test_adapters_normalize_current_ppocr_nim_and_tesseract_shapes() -> None:
    detections = detector_boxes(
        {
            "boxes": [{"bbox": [0.1, 0.2, 0.7, 0.3], "score": 0.91}],
            "model": "PP-OCRv6_medium_det",
        }
    )
    assert detections[0].bbox == [0.1, 0.2, 0.7, 0.3]
    assert detections[0].score == 0.91

    nemotron = recognition_items(
        {
            "text_detections": [
                {
                    "text_prediction": {"text": "Đơn giá", "confidence": 0.88},
                    "bounding_box": {
                        "points": [
                            {"x": 0.1, "y": 0.2},
                            {"x": 0.4, "y": 0.2},
                            {"x": 0.4, "y": 0.3},
                            {"x": 0.1, "y": 0.3},
                        ]
                    },
                }
            ]
        }
    )
    assert nemotron[0].text == "Đơn giá"
    assert nemotron[0].score == 0.88
    assert nemotron[0].bbox == (0.1, 0.2, 0.4, 0.3)

    tesseract = recognition_items(
        {
            "text": "Hợp đồng 2026",
            "score": 0.82,
            "model": "tesseract-5",
            "backend": "tesseract",
            "language": "eng+vie",
        }
    )
    assert tesseract[0].text == "Hợp đồng 2026"
    assert tesseract[0].model == "tesseract-5"
    assert tesseract[0].language == "eng+vie"


def test_ppocr_adapter_reuses_image_list_transport_for_both_endpoints() -> None:
    calls: list[tuple[str, list[str], dict[str, Any]]] = []

    def invoker(
        endpoint: str, images: list[str], **kwargs: Any
    ) -> list[dict[str, Any]]:
        calls.append((endpoint, images, kwargs))
        return [{"endpoint": endpoint} for _ in images]

    adapter = PPOCRv6Adapter(
        detector_endpoint="http://detector/v1/detect",
        recognizer_endpoint="http://recognizer/v1/recognize",
        invoker=invoker,
    )
    images = ["crop-a", "crop-b"]
    assert len(adapter.detect(images)) == 2
    assert len(adapter.recognize(images)) == 2
    assert [call[0] for call in calls] == [
        "http://detector/v1/detect",
        "http://recognizer/v1/recognize",
    ]
    assert calls[0][1] == calls[1][1] == images


def test_tesseract_adapter_sends_request_scoped_language_and_psm() -> None:
    calls: list[tuple[str, list[str], dict[str, Any]]] = []

    def invoker(
        endpoint: str, images: list[str], **kwargs: Any
    ) -> list[dict[str, Any]]:
        calls.append((endpoint, images, kwargs))
        return [{"text": "Đơn giá", "score": 0.95} for _ in images]

    adapter = make_tesseract_backend(
        "http://tesseract/v1/ocr",
        language="vie",
        psm=7,
        invoker=invoker,
    )
    assert len(adapter.recognize(["crop-a"])) == 1
    assert calls[0][2]["extra_payload"] == {"language": "vie", "psm": "7"}


def test_vllm_adapter_translates_image_crops_to_chat_completions(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    class _FakeClient:
        def __init__(self, *, max_pool_workers: int) -> None:
            calls.append({"max_pool_workers": max_pool_workers})

        def invoke_chat_completions_images(self, **kwargs: Any) -> list[str]:
            calls.append(kwargs)
            return ["Đơn giá", "Total"]

        def shutdown(self) -> None:
            calls.append({"shutdown": True})

    monkeypatch.setattr(
        "nemo_retriever.models.nim.nim.NIMClient",
        _FakeClient,
    )
    adapter = VLLMImageBackend(
        endpoint="http://vintern-ocr:8000/v1/chat/completions",
        model="Vintern-1B-v3.5",
        batch_size=2,
    )

    result = adapter.recognize(["crop-a", "crop-b"])

    assert [item["text"] for item in result] == ["Đơn giá", "Total"]
    assert calls[0] == {"max_pool_workers": 2}
    assert calls[1]["invoke_url"].endswith("/v1/chat/completions")
    assert calls[1]["model"] == "Vintern-1B-v3.5"
    assert calls[1]["task_prompt"].startswith("Chép lại chính xác")
    assert calls[1]["extra_body"] == {"max_tokens": 256}
    adapter.close()
    assert calls[-1] == {"shutdown": True}
